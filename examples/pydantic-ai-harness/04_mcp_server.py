"""Variant 4 — mm over MCP.

Same tools as variant 3, but out-of-process behind the Model Context
Protocol. The win is reach: one running server is usable by pydantic-ai,
Claude Code, an IDE, or anything else that speaks MCP — and heavy media
work stays off the agent process.

Start the server, then run this:

    mm mcp serve                                        # terminal 1
    uv run python examples/pydantic-ai-harness/04_mcp_server.py   # terminal 2

``native=False`` forces the local MCP toolset so the tools execute here
rather than provider-side — required for CodeMode to wrap them (variant 5).

Requires the mcp extra (`uv pip install 'mm-ctx[mcp]'`) and a
tool-calling model.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

from _model import DATA_DIR, describe_profile, mm_model
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

MCP_URL = "http://127.0.0.1:8765/mcp"

agent = Agent(
    mm_model(),
    capabilities=[MCP(MCP_URL, native=False)],
)


def server_is_up(url: str) -> bool:
    """True if something is listening on the MCP endpoint."""
    try:
        urllib.request.urlopen(url, timeout=2)
    except urllib.error.HTTPError:
        return True  # responded, just not to a bare GET
    except Exception:
        return False
    return True


def main() -> None:
    print(f"[{describe_profile()}]\n")

    if not server_is_up(MCP_URL):
        sys.exit(f"No MCP server at {MCP_URL}\nStart one with:  mm mcp serve")

    result = agent.run_sync(
        f"Using the mm tools, count the files in {DATA_DIR} by kind, "
        "then read the PDF and tell me what it is."
    )
    print(result.output)
    print(f"\n--- {result.usage} ---")


if __name__ == "__main__":
    main()
