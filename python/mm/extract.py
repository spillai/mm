"""Public content-extraction API — the programmatic face of ``mm cat``.

This module is the single entry point for turning a media file into text.
It runs the same pipeline the CLI runs (encode → generate, with unified
extraction caching), so ``mm.cat(path)`` and ``mm cat path`` produce
identical content for identical options.

Every ``mm cat`` flag has a keyword here, so a tool that forwards
``**kwargs`` to :func:`cat` exposes the full pipeline surface — that is
what the MCP server and the pydantic-ai toolset do.

Typical use — hand an agent a tool that understands any file type::

    import mm

    mm.cat("slides.pdf")                      # page text, no LLM
    mm.cat("clip.mp4", mode="accurate")       # keyframe mosaic → VLM
    mm.cat("photo.jpg", pipeline="tile")      # named encoder
    mm.cat("photo.jpg", encode={"strategy": "tile"}, generate={"max_tokens": 512})
    mm.cat_many(paths, mode="fast")           # threaded batch

Every call returns an :class:`Extraction`, which stringifies to its
content so it drops straight into a prompt::

    print(f"Slide deck:\\n{mm.cat('slides.pdf')}")
"""

from __future__ import annotations

import dataclasses
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

from mm.cat_utils.base_utils import (
    CatMode,
    CatOpts,
    RunResult,
    effective_model,
    override_extra,
)
from mm.cat_utils.extract_meta import extract_meta
from mm.utils import file_kind

if TYPE_CHECKING:
    from mm.constants import BinaryFileKind
    from mm.pipelines.schema import PipelineSpec

__all__ = ["Extraction", "cat", "cat_many"]


@dataclass(slots=True, frozen=True)
class Extraction:
    """The result of extracting one file.

    Stringifies to :attr:`content`, so an ``Extraction`` can be used
    anywhere a prompt fragment is expected.

    Attributes:
        path: The file that was extracted.
        kind: Detected file kind (``image``, ``video``, ``audio``,
            ``document``, ``text``).
        mode: Pipeline mode used (``fast`` or ``accurate``).
        content: Extracted text.
        cached: True when served from the extraction cache rather than
            recomputed (an LLM call was avoided).
    """

    path: Path
    kind: str
    mode: CatMode
    content: str
    cached: bool = False

    def __str__(self) -> str:
        return self.content


def cat(
    path: str | Path,
    *,
    mode: CatMode = "fast",
    pipeline: str | Path | Iterable[str | Path] | None = None,
    encode: dict[str, Any] | None = None,
    generate: dict[str, Any] | None = None,
    n: int | None = None,
    no_cache: bool = False,
    no_generate: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Extraction:
    """Extract text content from a single file.

    Mirrors ``mm cat``: the pipeline is chosen from the file's kind and
    ``mode``, and results are cached by content hash + profile + model,
    so repeat calls avoid the LLM entirely. Each keyword maps to the
    identically-named CLI flag.

    Args:
        path: File to extract.
        mode: ``"fast"`` (default) runs the kind's fast pipeline —
            images/videos get a short caption, PDFs get page text, audio
            gets a Whisper transcript, code/text pass through.
            ``"accurate"`` runs the LLM-powered pipeline.
        pipeline: Named encoder (``"tile"``, ``"mosaic"``) or path to a
            pipeline YAML — the ``-p`` flag. Accepts a list to supply
            one per kind.
        encode: Encoder overrides, e.g. ``{"strategy": "tile"}`` —
            equivalent to ``--encode.strategy tile``.
        generate: Generation overrides, e.g. ``{"max_tokens": 512}`` —
            equivalent to ``--generate.max-tokens 512``.
        n: Head/tail line limit. Positive keeps the first ``n`` lines,
            negative keeps the last ``n`` — the ``-n`` flag.
        no_cache: Bypass and evict the cached extraction for this file.
        no_generate: Run the encode step only, skipping the LLM call.
        dry_run: Resolve and describe the pipeline without running it.
        verbose: Append the pipeline/token diagnostics tail to the content.

    Returns:
        An :class:`Extraction`. Use ``.content`` for the text, or pass
        the object itself where a string is expected.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Examples:
        >>> mm.cat("paper.pdf").content[:40]
        'Attention Is All You Need\\n\\nAshish Vaswani'
        >>> mm.cat("clip.mp4", mode="accurate").cached
        False
        >>> mm.cat("notes.md", n=5).content.count("\\n")
        4
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")

    opts = _build_opts(
        mode=mode,
        pipeline=pipeline,
        encode=encode,
        generate=generate,
        no_cache=no_cache,
        no_generate=no_generate,
        dry_run=dry_run,
        verbose=verbose,
    )
    result = run(p, opts)
    return _head_tail(result, n)


def cat_many(
    paths: Iterable[str | Path],
    *,
    max_workers: int = 8,
    **kwargs: Any,
) -> list[Extraction]:
    """Extract many files concurrently, preserving input order.

    Extraction is I/O-bound (LLM calls, ffmpeg, disk), so a thread pool
    gives near-linear speedup across files.

    Args:
        paths: Files to extract.
        max_workers: Maximum concurrent extractions.
        **kwargs: Forwarded to :func:`cat` (``mode``, ``pipeline``, …).

    Returns:
        One :class:`Extraction` per input path, in the original order.

    Examples:
        >>> [e.kind for e in mm.cat_many(["a.png", "b.mp4"])]
        ['image', 'video']
    """
    targets = [Path(p).expanduser() for p in paths]
    if not targets:
        return []
    if len(targets) == 1:
        return [cat(targets[0], **kwargs)]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        return list(pool.map(lambda p: cat(p, **kwargs), targets))


def _head_tail(result: Extraction, n: int | None) -> Extraction:
    """Apply the ``-n`` head/tail line limit to an extraction."""
    if n is None:
        return result
    lines = result.content.splitlines()
    clipped = "\n".join(lines[:n] if n >= 0 else lines[n:])
    return dataclasses.replace(result, content=clipped)


def _build_opts(
    *,
    mode: CatMode,
    pipeline: str | Path | Iterable[str | Path] | None,
    encode: dict[str, Any] | None,
    generate: dict[str, Any] | None,
    no_cache: bool,
    no_generate: bool,
    dry_run: bool,
    verbose: bool,
) -> CatOpts:
    """Translate keyword arguments into the internal options bag."""
    specs: dict[str, PipelineSpec] = {}
    if pipeline is not None:
        from mm.pipelines.pipelines_utils import load_pipeline_args

        args = [pipeline] if isinstance(pipeline, (str, Path)) else list(pipeline)
        specs = load_pipeline_args([str(a) for a in args])

    return CatOpts(
        mode=mode,
        no_cache=no_cache,
        no_generate=no_generate,
        encode_overrides=dict(encode or {}),
        generate_overrides={k: str(v) for k, v in (generate or {}).items()},
        pipelines=specs,
        dry_run=dry_run,
        verbose=verbose,
    )


def run(path: Path, opts: CatOpts) -> Extraction:
    """Pipeline-driven extraction dispatch with unified extraction caching.

    The shared core behind :func:`extract` and ``mm cat``. Prefer
    :func:`extract` unless you already hold a :class:`CatOpts`.
    """
    kind = file_kind(path)
    ext = path.suffix.lower()
    if opts.dry_run:
        return Extraction(path, kind, opts.mode, dry_run_preview(path, kind, ext, opts))

    if is_passthrough(kind, ext, opts.mode):
        from mm.cat_utils.extract_meta import extract_text

        assert kind in ("document", "text")
        content, cached = extract_text(path, kind)  # type: ignore[arg-type]
        return Extraction(path, kind, opts.mode, content, cached=bool(cached))

    kind = cast("BinaryFileKind", kind)
    from mm.constants import OFFICE_EXTS
    from mm.encoders.auto_strategy import resolve_auto_strategy
    from mm.pipelines import apply_overrides
    from mm.pipelines.pipelines_utils import resolve_pipeline
    from mm.profile import get_profile
    from mm.store.utils import get_content_hash, shared_db

    db = shared_db()
    profile = get_profile()

    # Resolve & merge the pipeline spec exactly once so the cache key reflects
    # the effective model and the merged extra_body — required for correct
    # invalidation on `--model` / `--generate.extra-body` changes.
    spec = resolve_pipeline(opts, kind)
    spec = apply_overrides(spec, opts.encode_overrides or None, opts.generate_overrides or None)
    spec = resolve_auto_strategy(path, spec, opts)

    eff_model = effective_model(spec, profile.model)
    extra = override_extra(
        opts.encode_overrides,
        opts.generate_overrides,
        opts.pipelines,
    )

    extraction_id: str | None = None
    content_hash = get_content_hash(path)
    if content_hash:
        from mm.store.utils import get_extraction_id

        extraction_id = get_extraction_id(
            content_hash,
            profile.name,
            eff_model,
            opts.mode,
            False,
            extra=extra,
        )

        if not opts.no_cache:
            cached = db.get_extraction(extraction_id)
            if cached is not None:
                if opts.verbose:
                    meta = db.get_extraction_metadata(extraction_id)
                    suffix = meta.get("verbose_suffix") if meta else None
                    if suffix:
                        return Extraction(
                            path, kind, opts.mode, f"{cached}\n\n{suffix}", cached=True
                        )
                return Extraction(path, kind, opts.mode, cached, cached=True)
        else:
            db.evict_extraction(extraction_id)

    if ext in OFFICE_EXTS and opts.mode == "accurate":
        with tempfile.TemporaryDirectory(prefix="mm-office-") as tmpdir:
            from mm._mm import office_to_pdf

            tmp_pdf = Path(tmpdir) / f"{path.stem}.pdf"
            office_to_pdf(str(path), str(tmp_pdf))
            result = run_accurate(tmp_pdf, kind, spec, opts, meta_path=path)
    elif opts.mode == "accurate":
        result = run_accurate(path, kind, spec, opts)
    else:
        result = run_fast(path, kind, spec, opts)

    if content_hash and result.content and not result.content.startswith("["):
        extract_meta(path, kind)
        uri = str(path.resolve())
        meta = {"verbose_suffix": result.verbose_suffix} if result.verbose_suffix else None
        try:
            db.put_extraction(
                uri=uri,
                content_hash=content_hash,
                profile=profile.name,
                model=eff_model,
                content=result.content,
                mode=opts.mode,
                detail=False,
                extra=extra,
                metadata=meta,
            )
        except RuntimeError:
            return Extraction(path, kind, opts.mode, format_run(result, opts.verbose))
    return Extraction(path, kind, opts.mode, format_run(result, opts.verbose))


def is_passthrough(kind: str, ext: str, mode: str) -> bool:
    """Return True when the file should bypass the encode→generate pipeline."""
    from mm.constants import OFFICE_EXTS

    return kind == "text" or (
        kind == "document"
        and (
            (ext != ".pdf" and ext not in OFFICE_EXTS)
            or (ext in OFFICE_EXTS and mode != "accurate")
        )
    )


def format_run(run: RunResult, verbose: bool) -> str:
    """Render a :class:`RunResult` for display, conditionally including the suffix."""
    if verbose and run.verbose_suffix:
        return f"{run.content}\n\n{run.verbose_suffix}"
    return run.content


def run_fast(path: Path, kind: BinaryFileKind, spec: PipelineSpec, opts: CatOpts) -> RunResult:
    """Fast mode: run the kind's fast pipeline."""
    from mm.cat_utils.run_encoder import run_encoder

    if getattr(opts, "no_generate", False):
        import dataclasses

        spec = dataclasses.replace(spec, generate=None)
    if spec.encode.strategy:
        return run_encoder(path, kind, spec, opts)

    return RunResult(content=extract_meta(path, kind, no_cache=opts.no_cache))


def run_accurate(
    path: Path,
    kind: BinaryFileKind,
    spec: PipelineSpec,
    opts: CatOpts,
    *,
    meta_path: Path | None = None,
) -> RunResult:
    """Accurate mode: LLM-powered semantic extraction.

    ``spec`` is the merged (YAML + CLI) pipeline spec resolved by
    :func:`run`; this function does no further override application.
    ``meta_path`` references the original office file.
    """
    if getattr(opts, "no_generate", False):
        import dataclasses

        spec = dataclasses.replace(spec, generate=None)

    extract_meta(meta_path or path, kind, no_cache=opts.no_cache)

    return accurate_dispatch(path, kind, spec, opts)


def accurate_dispatch(
    path: Path, kind: BinaryFileKind, spec: PipelineSpec, opts: CatOpts
) -> RunResult:
    """Dispatch accurate-mode extraction based on file kind."""
    from mm.cat_utils.accurate_audio import accurate_audio
    from mm.cat_utils.accurate_image import accurate_image
    from mm.cat_utils.accurate_video import accurate_video

    if kind == "image":
        return accurate_image(path, spec, opts)
    if kind == "video":
        return accurate_video(path, spec, opts)
    if kind == "audio":
        return accurate_audio(path, spec, opts)

    from mm.cat_utils.run_encoder import run_encoder

    if spec.encode.strategy:
        return run_encoder(path, kind, spec, opts)

    return RunResult(content=extract_meta(path, kind))


def dry_run_preview(path: Path, kind: str, ext: str, opts: CatOpts) -> str:
    """Render the resolved pipeline for ``path × opts.mode`` without invoking it.

    For passthrough kinds (``kind=text``, or non-PDF/non-office documents),
    emit a short header/info block.
    """
    from mm.constants import OFFICE_EXTS
    from mm.display import format_size
    from mm.encoders.auto_strategy import resolve_auto_strategy
    from mm.pipelines import apply_overrides
    from mm.pipelines.pipelines_utils import resolve_pipeline
    from mm.profile import get_profile

    if is_passthrough(kind, ext, opts.mode):
        size_str = format_size(path.stat().st_size)
        header = f"\n# {path} (kind={kind}, mode={opts.mode}) — passthrough preview (--dry-run)"
        info_lines = [
            f"  ├─ size: {size_str}",
            "  └─ passthrough: content emitted as-is \\[skipped via --dry-run]",
        ]
        return "\n".join(["[dim]", header, "passthrough", *info_lines, "[/dim]"])

    spec = resolve_pipeline(opts, kind)
    spec = apply_overrides(spec, opts.encode_overrides or None, opts.generate_overrides or None)
    autoencode = spec.encode.strategy == "auto" or (
        spec.encode.strategy is None and spec.generate is not None
    )
    spec = resolve_auto_strategy(path, spec, opts)
    header = f"\n# {path} (kind={kind}, mode={opts.mode}) — pipeline preview (--dry-run)"

    encode = spec.encode
    strategy = encode.strategy or "<unspecified>"
    strategy = f"auto → {strategy}" if autoencode else strategy
    enc_opts = encode.strategy_opts or {}
    enc_opts_str = (
        ", ".join(f"{k}={v}" for k, v in sorted(enc_opts.items())) if enc_opts else "<defaults>"
    )

    if spec.generate is not None:
        gen = spec.generate
        lines = (gen.prompt or "").strip().splitlines()
        first_line = lines[0] if lines else ""
        if len(first_line) > 60:
            first_line = first_line[:60] + "..."

        prompt_part = f' · prompt="{first_line}"' if first_line else ""
        profile = get_profile()
        eff = gen.model or profile.model
        gen_line = (
            f"generate: profile={profile.name} · model={eff}{prompt_part}  [skipped via --dry-run]"
        )
    else:
        gen_line = "generate: <none>  [encode-only pipeline]"

    if ext in OFFICE_EXTS and opts.mode == "accurate":
        header += " [routes through office→PDF before encode]"

    middle: list[str] = [f"  ├─ encode: {strategy} · {enc_opts_str}"]
    if encode.pyfunc:
        middle.append(f"  ├─ pyfunc: {encode.pyfunc}")

    return "\n".join(["[dim]", header, "pipeline", *middle, f"  └─ {gen_line}", "[/dim]"])
