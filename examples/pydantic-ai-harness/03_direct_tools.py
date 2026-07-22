"""Variant 3 — mm as an agentic toolset, in-process.

Variants 1 and 2 pre-digest everything. Here the agent decides for
itself: list the directory, check a file's metadata, and read only what
the question needs. That matters once a directory is too big to extract
wholesale — the agent skips the 2 GB video it doesn't care about.

    uv run python examples/pydantic-ai-harness/03_direct_tools.py

Requires a tool-calling model. Check yours first:

    uv run python examples/pydantic-ai-harness/check_model.py
"""

from __future__ import annotations

from _model import DATA_DIR, describe_profile, mm_model
from mm_tools import mm_toolset
from pydantic_ai import Agent
from pydantic_ai.capabilities import Toolset

agent = Agent(
    mm_model(),
    capabilities=[Toolset(mm_toolset())],
)


def main() -> None:
    print(f"[{describe_profile()}]\n")

    result = agent.run_sync(
        f"Look in {DATA_DIR}. Find the largest file, check its metadata, "
        "then read it and summarise what it contains."
    )
    print(result.output)

    calls = [
        p.tool_name
        for m in result.all_messages()
        for p in m.parts
        if type(p).__name__ == "ToolCallPart"
    ]
    print(f"\n--- tools called: {calls or 'NONE (model cannot call tools)'} ---")
    print(f"--- {result.usage} ---")


if __name__ == "__main__":
    main()
