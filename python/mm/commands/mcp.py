"""mm mcp -- serve mm's multimodal tools over the Model Context Protocol."""

from __future__ import annotations

from typing import Annotated

import typer

mcp_app = typer.Typer(
    name="mcp",
    help=(
        "Serve mm as an MCP server.\n\n"
        "Exposes cat / cat_many / peek / find / grep / sql as MCP tools so any\n"
        "MCP client — Claude Code, pydantic-ai, an IDE — can read images,\n"
        "video, audio and PDFs through mm.\n\n"
        "Examples:\n\n"
        "  mm mcp serve                        # http://127.0.0.1:8765/mcp\n"
        "  mm mcp serve --port 9000\n"
        "  mm mcp serve --transport stdio      # for subprocess-launched clients\n\n"
        "Requires the mcp extra:  uv pip install 'mm-ctx\\[mcp]'"
    ),
    no_args_is_help=True,
)


@mcp_app.command("serve")
def mcp_serve(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 8765,
    host: Annotated[str, typer.Option("--host", help="Interface to bind")] = "127.0.0.1",
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="Transport: http (default) or stdio"),
    ] = "http",
) -> None:
    """Start the mm MCP server."""
    try:
        from mm.mcp import serve
    except ImportError as e:
        typer.echo(
            "Error: the MCP server needs the 'mcp' extra.\n  uv pip install 'mm-ctx[mcp]'",
            err=True,
        )
        raise typer.Exit(1) from e

    if transport not in ("http", "stdio"):
        typer.echo(f"Error: unknown transport {transport!r}. Use 'http' or 'stdio'.", err=True)
        raise typer.Exit(1)

    if transport == "http":
        typer.echo(f"mm MCP server → http://{host}:{port}/mcp", err=True)

    serve(host=host, port=port, transport=transport)
