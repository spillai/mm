"""Variant 1 — mm as a multimodal processor feeding the prompt directly.

No tool calling. mm compresses every file — video, audio, PDF, image —
into text *before* the agent sees it, so even a tiny local model can
answer questions across a mixed-media directory.

This is the highest-leverage integration for small local models: the
agent never has to decide to call a tool, and the whole directory
arrives pre-digested.

    uv run python examples/pydantic-ai-harness/01_context_builder.py

Works with any model, including non-tool-calling ones.
"""

from __future__ import annotations

from pathlib import Path

from _model import DATA_DIR, describe_profile, mm_model
from pydantic_ai import Agent

import mm

agent = Agent(
    mm_model(),
    instructions=(
        "You answer questions about a directory of media files. Each file's "
        "content has already been extracted for you. Cite files by name."
    ),
)


def build_context(directory: str) -> tuple[list[str], int, int]:
    """Extract every file in ``directory``, returning prompt blocks and sizes.

    Returns:
        The per-file text blocks, total bytes on disk, and total characters
        of extracted text — the compression mm achieved.
    """
    root = Path(directory).expanduser()
    files = sorted(p for p in root.iterdir() if p.is_file())

    extractions = mm.cat_many(files)
    blocks = [f"## {e.path.name}  ({e.kind})\n{e.content}" for e in extractions]

    raw_bytes = sum(p.stat().st_size for p in files)
    text_chars = sum(len(e.content) for e in extractions)
    return blocks, raw_bytes, text_chars


def main() -> None:
    print(f"[{describe_profile()}]\n")

    blocks, raw_bytes, text_chars = build_context(DATA_DIR)
    print(f"mm compressed {raw_bytes / 1e6:.1f} MB of media → {text_chars:,} chars of text")
    print(f"  ({raw_bytes / max(text_chars, 1):,.0f}x reduction across {len(blocks)} files)\n")

    result = agent.run_sync(
        [
            "Here is the extracted content of every file in the directory:",
            *blocks,
            "\nWhich file is about a car, and which is about food? Answer in two lines.",
        ]
    )
    print(result.output)
    print(f"\n--- {result.usage} ---")


if __name__ == "__main__":
    main()
