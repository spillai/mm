"""FastMCP server exposing mm's multimodal surface as MCP tools.

Every tool is a thin forward to the corresponding public API — the
pipeline logic lives in :mod:`mm.extract` and :mod:`mm.context`, not
here. ``cat`` mirrors the full ``mm cat`` flag set so an agent can
reach for a different encoder, prompt, or model without the server
having to anticipate it.

Run it::

    mm mcp serve --port 8765

Then mount it from any MCP client — with pydantic-ai::

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import MCP

    agent = Agent(model, capabilities=[MCP('http://127.0.0.1:8765/mcp', native=False)])
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

INSTRUCTIONS = """\
mm gives you multimodal context: it turns files an LLM cannot natively \
read — images, video, audio, PDFs, Office docs — into text.

Start with `find` to see what a directory holds, `peek` for cheap local \
metadata (dimensions, duration, codec — no LLM), then `cat` to actually \
read a file's content.

`cat` picks the right pipeline from the file type. Use mode="fast" \
(default) for a quick caption or page text, and mode="accurate" when you \
need a careful description — accurate costs an LLM call and is slower, so \
prefer fast unless the task demands detail. Results are cached by content \
hash, so re-reading a file is free.\
"""

mcp: FastMCP = FastMCP(name="mm", instructions=INSTRUCTIONS)

Mode = Literal["fast", "accurate"]
Transport = Literal["http", "stdio"]

_SCANNER_IN_THREAD = False
"""Run directory-scanning tools on the event loop rather than the thread pool.

``Context`` scans through the Rust extension and hands the result to
pyarrow over Arrow IPC. Doing that in a *freshly spawned* thread, then
querying the table, segfaults the interpreter on the second such thread —
which is exactly the churn a worker pool produces. A single long-lived
thread is stable, so these tools run inline on the event loop.

They are metadata-only (no LLM, no network), so the loop is not held for
long. ``cat``/``cat_many``/``peek`` keep the default thread offload since
they can block on model calls.
"""


@mcp.tool(
    annotations={"readOnlyHint": True},
    tags={"multimodal", "extract"},
)
def cat(
    path: Annotated[str, Field(description="Path to the file to read.")],
    mode: Annotated[
        Mode, Field(description="'fast' for a quick read, 'accurate' for a careful LLM read.")
    ] = "fast",
    pipeline: Annotated[
        str | None,
        Field(description="Named encoder (e.g. 'tile', 'mosaic') or path to a pipeline YAML."),
    ] = None,
    encode: Annotated[
        dict[str, Any] | None,
        Field(description="Encoder overrides, e.g. {'strategy': 'tile'}."),
    ] = None,
    generate: Annotated[
        dict[str, Any] | None,
        Field(
            description="LLM overrides, e.g. {'prompt': 'List every table.', 'max_tokens': 512}."
        ),
    ] = None,
    n: Annotated[
        int | None,
        Field(description="Line limit: positive keeps the first N lines, negative the last N."),
    ] = None,
    no_cache: Annotated[bool, Field(description="Force a fresh run, ignoring the cache.")] = False,
    no_generate: Annotated[bool, Field(description="Run the encoder only; skip the LLM.")] = False,
    dry_run: Annotated[
        bool, Field(description="Describe the pipeline that would run, without running it.")
    ] = False,
) -> str:
    """Read any file as text — image, video, audio, PDF, Office doc, or code.

    This is the main tool: it converts media an LLM cannot read directly
    into text you can reason over.
    """
    from mm.extract import cat as _cat

    return _cat(
        path,
        mode=mode,
        pipeline=pipeline,
        encode=encode,
        generate=generate,
        n=n,
        no_cache=no_cache,
        no_generate=no_generate,
        dry_run=dry_run,
    ).content


@mcp.tool(annotations={"readOnlyHint": True}, tags={"multimodal", "extract"})
def cat_many(
    paths: Annotated[list[str], Field(description="Files to read.")],
    mode: Annotated[Mode, Field(description="Pipeline mode for every file.")] = "fast",
    pipeline: Annotated[str | None, Field(description="Named encoder or pipeline YAML.")] = None,
    generate: Annotated[
        dict[str, Any] | None, Field(description="LLM overrides applied to every file.")
    ] = None,
    max_workers: Annotated[int, Field(description="Maximum concurrent reads.", ge=1)] = 8,
) -> list[dict[str, Any]]:
    """Read many files concurrently. Much faster than repeated `cat` calls."""
    from mm.extract import cat_many as _cat_many

    results = _cat_many(
        paths,
        mode=mode,
        pipeline=pipeline,
        generate=generate,
        max_workers=max_workers,
    )
    return [
        {"path": str(r.path), "kind": r.kind, "content": r.content, "cached": r.cached}
        for r in results
    ]


@mcp.tool(annotations={"readOnlyHint": True}, tags={"metadata"})
def peek(
    path: Annotated[str, Field(description="File to inspect.")],
    full: Annotated[
        bool, Field(description="Include document author/title/subject/page-count fields.")
    ] = False,
) -> dict[str, Any]:
    """Get a file's metadata without reading its content.

    Cheap and local — no LLM call. Returns dimensions, duration, codec,
    mime type, and content hash. Use this before `cat` to decide whether a
    file is worth reading.
    """
    from mm.peek import FileMetadata

    meta = FileMetadata.from_path(path, full=full).to_dict()
    # Most fields are kind-specific and null; dropping them keeps the
    # agent's context tight without hiding anything it can act on.
    return {k: v for k, v in meta.items() if v is not None}


@mcp.tool(annotations={"readOnlyHint": True}, tags={"discovery"}, run_in_thread=_SCANNER_IN_THREAD)
def find(
    directory: Annotated[str, Field(description="Directory to scan.")],
    kind: Annotated[
        str | None,
        Field(description="Filter by kind: image, video, audio, document, code, data, text."),
    ] = None,
    ext: Annotated[
        str | None, Field(description="Filter by extension(s), comma-separated (e.g. '.pdf,.png').")
    ] = None,
    min_size: Annotated[str | None, Field(description="Minimum file size, e.g. '1MB'.")] = None,
    max_size: Annotated[str | None, Field(description="Maximum file size, e.g. '500KB'.")] = None,
    limit: Annotated[int, Field(description="Maximum files to return.", ge=1)] = 100,
) -> list[dict[str, Any]]:
    """List files in a directory with their type, size, and dimensions.

    Respects .gitignore. Start here to discover what a directory holds.
    """
    from mm.context import Context

    ctx = Context(directory)
    if any(v is not None for v in (kind, ext, min_size, max_size)):
        ctx = ctx.filter(kind=kind, ext=ext, min_size=min_size, max_size=max_size)

    rows = ctx.to_arrow().to_pylist()[:limit]
    keep = ("path", "name", "ext", "kind", "size", "mime", "width", "height")
    return [{k: r[k] for k in keep if k in r} for r in rows]


@mcp.tool(annotations={"readOnlyHint": True}, tags={"discovery"}, run_in_thread=_SCANNER_IN_THREAD)
def grep(
    directory: Annotated[str, Field(description="Directory to search.")],
    pattern: Annotated[str, Field(description="Regular expression to search for.")],
    kind: Annotated[str | None, Field(description="Restrict the search to one kind.")] = None,
    limit: Annotated[int, Field(description="Maximum matches to return.", ge=1)] = 100,
) -> list[dict[str, Any]]:
    """Search file contents by regex, across text and extracted media text."""
    from mm.context import Context

    return Context(directory).grep(pattern, kind=kind)[:limit]


@mcp.tool(annotations={"readOnlyHint": True}, tags={"query"}, run_in_thread=_SCANNER_IN_THREAD)
def sql(
    query: Annotated[
        str, Field(description="SQL against the 'files' table, e.g. SELECT kind, COUNT(*) ...")
    ],
    directory: Annotated[str, Field(description="Directory to scan into the 'files' table.")],
) -> list[dict[str, Any]]:
    """Run SQL over a directory's file metadata.

    Columns: path, name, stem, ext, size, modified, created, mime, kind,
    is_binary, depth, parent, width, height. Best for aggregate questions
    ("how much video do I have?") that would otherwise need many `find` calls.
    """
    from mm.context import Context

    return Context(directory).sql(query).to_pylist()


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    transport: Transport = "http",
) -> None:
    """Run the mm MCP server.

    Args:
        host: Interface to bind (HTTP transports only).
        port: Port to listen on (HTTP transports only).
        transport: ``"http"`` for a mountable URL, ``"stdio"`` for a
            subprocess-launched client.
    """
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    serve()
