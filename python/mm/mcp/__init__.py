"""MCP server for mm — multimodal file reading over the Model Context Protocol.

Requires the ``mcp`` extra::

    uv pip install "mm-ctx[mcp]"

Then either run it from the CLI (``mm mcp serve``) or embed it::

    from mm.mcp import mcp

    mcp.run(transport="http", port=8765)
"""

from __future__ import annotations

__all__ = ["mcp", "serve"]


def __getattr__(name: str):
    """Defer importing the server (and fastmcp) until first use."""
    if name in __all__:
        from mm.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module 'mm.mcp' has no attribute {name!r}")
