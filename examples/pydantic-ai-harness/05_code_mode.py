"""Variant 5 — CodeMode over mm's MCP tools.

CodeMode collapses every tool into one sandboxed ``run_code`` tool. The
model writes Python that loops over a directory, calls ``cat_many``,
filters locally, and returns only the answer — N files in one model
round-trip instead of N.

This is the combination the pydantic-ai harness was built for, and it
suits mm particularly well: media extraction is slow and parallel, so
batching it inside one sandboxed program beats a chatty tool loop.

    mm mcp serve                                       # terminal 1
    uv run python examples/pydantic-ai-harness/05_code_mode.py  # terminal 2

Requires:  uv pip install 'pydantic-ai-harness[code-mode]'
plus the mcp extra and a tool-calling model.
"""

from __future__ import annotations

import sys

from _model import DATA_DIR, describe_profile, mm_model
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai_harness import CodeMode

MCP_URL = "http://127.0.0.1:8765/mcp"

agent = Agent(
    mm_model(),
    capabilities=[
        # Wraps the mm tools into a single sandboxed run_code tool, so the
        # model can batch reads with loops and comprehensions.
        CodeMode(),
        # native=False keeps execution local so CodeMode can wrap the tools.
        MCP(MCP_URL, native=False),
    ],
)


def main() -> None:
    print(f"[{describe_profile()}]\n")

    from importlib.util import find_spec

    if find_spec("monty") is None and find_spec("pydantic_monty") is None:
        sys.exit("CodeMode needs the sandbox:  uv pip install 'pydantic-ai-harness[code-mode]'")

    result = agent.run_sync(
        f"In one run_code call: list every file in {DATA_DIR}, read them all "
        "with cat_many, and return a one-line summary per file sorted by size."
    )
    print(result.output)
    print(f"\n--- {result.usage} ---")


if __name__ == "__main__":
    main()
