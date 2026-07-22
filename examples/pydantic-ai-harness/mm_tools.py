"""mm as a pydantic-ai toolset — the in-process integration.

Each tool is a thin forward to mm's public API. Nothing here knows about
encoders, pipelines, or VLMs; ``mm.cat`` resolves all of that from the
file's type and the requested mode.

Use it with the ``Toolset`` capability::

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Toolset

    agent = Agent(model, capabilities=[Toolset(mm_toolset())])

For the out-of-process equivalent, run ``mm mcp serve`` and mount it with
the ``MCP`` capability instead — same tools, different transport.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic_ai.toolsets import FunctionToolset

Mode = Literal["fast", "accurate"]

INSTRUCTIONS = """\
You can read files that are not plain text — images, video, audio, PDFs — \
using the `cat` tool. Use `find` to discover files, `peek` for cheap \
metadata, and `cat_many` when you need several files at once. Prefer \
mode="fast" unless the question needs careful detail.\
"""


def mm_toolset() -> FunctionToolset:
    """Build a toolset exposing mm's multimodal surface to an agent."""
    toolset = FunctionToolset(instructions=INSTRUCTIONS)

    @toolset.tool_plain
    def cat(
        path: str,
        mode: Mode = "fast",
        pipeline: str | None = None,
        generate: dict[str, Any] | None = None,
        n: int | None = None,
    ) -> str:
        """Read any file as text — image, video, audio, PDF, or code.

        Args:
            path: File to read.
            mode: 'fast' for a quick read, 'accurate' for a careful LLM read.
            pipeline: Named encoder ('tile', 'mosaic', 'keyframes') or a pipeline YAML.
            generate: LLM overrides, e.g. {'prompt': 'List every total.'}.
            n: Line limit — positive keeps the first N lines, negative the last N.
        """
        import mm

        return mm.cat(path, mode=mode, pipeline=pipeline, generate=generate, n=n).content

    @toolset.tool_plain
    def cat_many(
        paths: list[str],
        mode: Mode = "fast",
        generate: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read many files at once, concurrently. Prefer this over repeated `cat`.

        Args:
            paths: Files to read.
            mode: Pipeline mode applied to every file.
            generate: LLM overrides applied to every file.
        """
        import mm

        return [
            {"path": str(r.path), "kind": r.kind, "content": r.content}
            for r in mm.cat_many(paths, mode=mode, generate=generate)
        ]

    @toolset.tool_plain
    def peek(path: str) -> dict[str, Any]:
        """Get a file's metadata (dimensions, duration, codec) without reading it.

        Cheap and local — no LLM call. Use before `cat` to size up a file.

        Args:
            path: File to inspect.
        """
        from mm.peek import FileMetadata

        meta = FileMetadata.from_path(path).to_dict()
        return {k: v for k, v in meta.items() if v is not None}

    @toolset.tool_plain
    def find(directory: str, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List files in a directory with their type and size.

        Args:
            directory: Directory to scan (respects .gitignore).
            kind: Filter by kind — image, video, audio, document, code, data, text.
            limit: Maximum files to return.
        """
        from mm.context import Context

        ctx = Context(directory)
        if kind:
            ctx = ctx.filter(kind=kind)
        keep = ("path", "name", "ext", "kind", "size")
        return [{k: r[k] for k in keep if k in r} for r in ctx.to_arrow().to_pylist()[:limit]]

    @toolset.tool_plain
    def sql(directory: str, query: str) -> list[dict[str, Any]]:
        """Run SQL over a directory's file metadata.

        Columns: path, name, stem, ext, size, modified, mime, kind, depth,
        parent, width, height. Best for aggregates across many files.

        Args:
            directory: Directory to scan into the 'files' table.
            query: SQL query, e.g. "SELECT kind, COUNT(*) FROM files GROUP BY kind".
        """
        from mm.context import Context

        return Context(directory).sql(query).to_pylist()

    return toolset
