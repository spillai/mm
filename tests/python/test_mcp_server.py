"""Tests for the mm MCP server.

Exercised through a real :class:`fastmcp.Client` against the in-process
server, so tool schemas and serialization are covered the same way a real
MCP client would hit them.

The suite is synchronous; ``_run`` drives the async client via anyio so no
async pytest plugin is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import pytest

pytest.importorskip("fastmcp", reason="requires the 'mcp' extra")

import asyncio  # noqa: E402

from fastmcp import Client  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

T = TypeVar("T")

EXPECTED_TOOLS = {"cat", "cat_many", "chat", "peek", "find", "grep", "sql"}

_loop: asyncio.AbstractEventLoop | None = None


def _run(coro_fn: Callable[[], Awaitable[T]]) -> T:
    """Run an async client interaction from a sync test.

    One event loop is reused for the whole module. Creating and tearing
    down a loop per call churns FastMCP's worker threads alongside
    pyarrow, which segfaults the interpreter.
    """
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro_fn())


def call_tool(name: str, args: dict[str, Any], *, sampling_handler: Any = None) -> Any:
    """Invoke one MCP tool and return its deserialized payload.

    Args:
        name: Tool to call.
        args: Tool arguments.
        sampling_handler: Optional client-side LLM stub. Without one the
            client advertises no sampling support, which is what the
            ``chat`` failure-path test relies on.
    """
    from mm.mcp import mcp

    async def go():
        async with Client(mcp, sampling_handler=sampling_handler) as client:
            return (await client.call_tool(name, args)).data

    return _run(go)


class RecordingSampler:
    """Client-side sampling stub that records what the server asked for."""

    def __init__(self, reply: str = "STUB ANSWER") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.params: list[Any] = []

    def __call__(self, messages, params, context) -> str:
        self.prompts.append("\n".join(m.content.text for m in messages))
        self.params.append(params)
        return self.reply

    @property
    def prompt(self) -> str:
        """The single prompt sent, for the common one-call case."""
        assert len(self.prompts) == 1, f"expected 1 sampling call, got {len(self.prompts)}"
        return self.prompts[0]


def list_tools() -> list[Any]:
    """Return the server's advertised tool definitions."""
    from mm.mcp import mcp

    async def go():
        async with Client(mcp) as client:
            return await client.list_tools()

    return _run(go)


@pytest.fixture
def media_dir(tmp_path: Path) -> Path:
    """A small directory of text files — no LLM needed to extract them."""
    (tmp_path / "a.txt").write_text("alpha\nbravo\ncharlie")
    (tmp_path / "b.txt").write_text("delta")
    (tmp_path / "notes.md").write_text("# heading\nbody")
    return tmp_path


class TestToolRegistration:
    def test_all_tools_exposed(self):
        assert {t.name for t in list_tools()} == EXPECTED_TOOLS

    def test_tools_are_documented(self):
        """Descriptions drive tool selection — none may be empty."""
        for tool in list_tools():
            assert tool.description and tool.description.strip()

    def test_cat_exposes_full_pipeline_surface(self):
        """The cat tool must forward every meaningful mm cat option."""
        tools = {t.name: t for t in list_tools()}
        params = set(tools["cat"].inputSchema["properties"])
        assert {"path", "mode", "pipeline", "encode", "generate", "n"} <= params

    def test_chat_mirrors_the_cat_surface(self):
        """chat takes an instruction plus everything cat takes."""
        tools = {t.name: t for t in list_tools()}
        cat_params = set(tools["cat"].inputSchema["properties"]) - {"dry_run"}
        chat_schema = tools["chat"].inputSchema
        chat_params = set(chat_schema["properties"])

        assert cat_params <= chat_params
        assert {"instruction", "system_prompt", "max_tokens", "temperature"} <= chat_params
        assert set(chat_schema["required"]) == {"instruction", "path"}

    def test_chat_does_not_leak_the_context_param(self):
        """`ctx` is injected by FastMCP and must stay out of the model's schema."""
        tools = {t.name: t for t in list_tools()}
        assert "ctx" not in tools["chat"].inputSchema["properties"]


class TestCat:
    def test_reads_a_file(self, media_dir: Path):
        assert "alpha" in call_tool("cat", {"path": str(media_dir / "a.txt")})

    def test_head_limit(self, media_dir: Path):
        result = call_tool("cat", {"path": str(media_dir / "a.txt"), "n": 1})
        assert result.strip() == "alpha"

    def test_missing_file_errors(self, media_dir: Path):
        with pytest.raises(ToolError):
            call_tool("cat", {"path": str(media_dir / "ghost.txt")})


class TestCatMany:
    def test_reads_several_files(self, media_dir: Path):
        paths = [str(media_dir / "a.txt"), str(media_dir / "b.txt")]
        result = call_tool("cat_many", {"paths": paths})
        assert [r["path"] for r in result] == paths
        assert "delta" in result[1]["content"]


class TestChat:
    """chat = cat + MCP sampling, so the answering model is the caller's."""

    def test_extracted_content_reaches_the_sampler(self, media_dir: Path):
        sampler = RecordingSampler()
        result = call_tool(
            "chat",
            {"instruction": "Summarize this.", "path": str(media_dir / "a.txt")},
            sampling_handler=sampler,
        )
        assert result == "STUB ANSWER"
        assert "Summarize this." in sampler.prompt
        assert "alpha" in sampler.prompt  # the file's content, via cat

    def test_prompt_labels_the_source_file(self, media_dir: Path):
        sampler = RecordingSampler()
        call_tool(
            "chat",
            {"instruction": "What is this?", "path": str(media_dir / "a.txt")},
            sampling_handler=sampler,
        )
        assert "a.txt" in sampler.prompt
        assert "text" in sampler.prompt  # the detected kind

    def test_forwards_cat_options(self, media_dir: Path):
        """cat kwargs must reach extraction — n=1 truncates what is sent."""
        sampler = RecordingSampler()
        call_tool(
            "chat",
            {"instruction": "Summarize.", "path": str(media_dir / "a.txt"), "n": 1},
            sampling_handler=sampler,
        )
        assert "alpha" in sampler.prompt
        assert "charlie" not in sampler.prompt

    def test_forwards_sampling_options(self, media_dir: Path):
        sampler = RecordingSampler()
        call_tool(
            "chat",
            {
                "instruction": "Summarize.",
                "path": str(media_dir / "a.txt"),
                "system_prompt": "Be terse.",
                "max_tokens": 64,
                "temperature": 0.0,
            },
            sampling_handler=sampler,
        )
        params = sampler.params[0]
        assert params.systemPrompt == "Be terse."
        assert params.maxTokens == 64
        assert params.temperature == 0.0

    def test_without_sampling_support_explains_itself(self, media_dir: Path):
        """A client with no sampler must get an actionable error, not a stack trace."""
        with pytest.raises(ToolError) as exc:
            call_tool("chat", {"instruction": "x", "path": str(media_dir / "a.txt")})
        message = str(exc.value)
        assert "sampling" in message
        assert "cat" in message  # points at the fallback

    def test_empty_answer_is_reported(self, media_dir: Path):
        """An empty completion is a failure, not a valid answer to return."""
        with pytest.raises(ToolError, match="empty answer"):
            call_tool(
                "chat",
                {"instruction": "Summarize.", "path": str(media_dir / "a.txt")},
                sampling_handler=RecordingSampler(reply=""),
            )

    def test_missing_file_errors_before_sampling(self, media_dir: Path):
        sampler = RecordingSampler()
        with pytest.raises(ToolError):
            call_tool(
                "chat",
                {"instruction": "x", "path": str(media_dir / "ghost.txt")},
                sampling_handler=sampler,
            )
        assert sampler.prompts == []


class TestPeek:
    def test_returns_metadata_without_nulls(self, media_dir: Path):
        result = call_tool("peek", {"path": str(media_dir / "a.txt")})
        assert result["name"] == "a.txt"
        assert result["size"] > 0
        assert all(v is not None for v in result.values())


class TestFind:
    def test_lists_directory(self, media_dir: Path):
        result = call_tool("find", {"directory": str(media_dir)})
        assert {r["name"] for r in result} == {"a.txt", "b.txt", "notes.md"}

    def test_limit_is_applied(self, media_dir: Path):
        assert len(call_tool("find", {"directory": str(media_dir), "limit": 1})) == 1


class TestSql:
    def test_aggregates_over_files(self, media_dir: Path):
        result = call_tool(
            "sql", {"directory": str(media_dir), "query": "SELECT COUNT(*) AS n FROM files"}
        )
        assert result[0]["n"] == 3


class TestGrep:
    def test_finds_matches(self, media_dir: Path):
        result = call_tool("grep", {"directory": str(media_dir), "pattern": "bravo"})
        assert any("bravo" in m["line"] for m in result)
