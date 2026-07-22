"""Variant 2 — mm as an encoder, handing native media to a VLM.

Variant 1 gives the agent mm's *text*. This one keeps the pixels: mm's
encoders compress media into something a vision model can actually
ingest, and pydantic-ai passes it through as native image content.

A 29 MB, 4-minute video becomes five JPEG contact sheets (76 sampled
frames). The VLM never sees a video container — it sees pictures.

    uv run python examples/pydantic-ai-harness/02_vlm_encoder.py

Needs a vision-capable model; no tool calling required.
"""

from __future__ import annotations

from pathlib import Path

from _model import DATA_DIR, describe_profile, mm_model
from mm_encode import encode
from pydantic_ai import Agent, BinaryContent

agent = Agent(mm_model())


def main() -> None:
    print(f"[{describe_profile()}]\n")

    video = Path(DATA_DIR).expanduser() / "bakery.mp4"
    parts = encode(video, "mosaic")

    images = [p for p in parts if isinstance(p, BinaryContent)]
    encoded_bytes = sum(len(p.data) for p in images)
    print(
        f"mm encoded {video.stat().st_size / 1e6:.1f} MB of video → "
        f"{len(images)} mosaic image(s), {encoded_bytes / 1e6:.1f} MB "
        f"({video.stat().st_size / max(encoded_bytes, 1):.0f}x smaller)\n"
    )

    result = agent.run_sync(
        ["These frames are sampled from one video. What is happening in it?", *parts]
    )
    print(result.output)
    print(f"\n--- {result.usage} ---")


if __name__ == "__main__":
    main()
