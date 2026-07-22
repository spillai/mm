"""Bridge mm's encoders to pydantic-ai prompt parts.

mm encoders emit OpenAI-shaped message dicts (``{"type": "image_url",
"image_url": {"url": "data:image/jpeg;base64,..."}}``). pydantic-ai wants
:class:`BinaryContent`. This is the ~20 lines that join them — copy it
into your own project if you need it.

The value is in what the encoder does before this point: ``mosaic`` turns
a 4-minute 29 MB video into five JPEG contact sheets, so a VLM with no
video support can still watch it.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Iterable

from pydantic_ai import BinaryContent

_DATA_URL = re.compile(r"^data:([^;]+);base64,(.*)$", re.DOTALL)


def parts_from_messages(messages: Iterable[dict[str, Any]]) -> list[str | BinaryContent]:
    """Flatten mm encoder messages into pydantic-ai prompt parts.

    Args:
        messages: Message dicts as yielded by any ``mm.encoders`` encoder.

    Returns:
        Text strings and :class:`BinaryContent` items, in encoder order.
    """
    parts: list[str | BinaryContent] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content or []:
            if part.get("type") == "text":
                parts.append(part["text"])
            elif part.get("type") == "image_url":
                match = _DATA_URL.match(part["image_url"]["url"])
                if match:
                    media_type, payload = match.groups()
                    parts.append(
                        BinaryContent(data=base64.b64decode(payload), media_type=media_type)
                    )
    return parts


def encode(path: str | Path, strategy: str, **kwargs: Any) -> list[str | BinaryContent]:
    """Encode one file with a named mm encoder, ready for ``agent.run_sync``.

    Args:
        path: File to encode.
        strategy: Encoder name — ``mosaic``/``keyframes`` (video),
            ``resize``/``tile`` (image), ``rasterize`` (document).
        **kwargs: Forwarded to the encoder, e.g. ``max_width=768``.

    Returns:
        Prompt parts to pass alongside your question.

    Examples:
        >>> agent.run_sync(["What happens here?", *encode("clip.mp4", "mosaic")])
    """
    from mm.encoders import get
    from mm.utils import file_kind

    p = Path(path).expanduser()
    encoder = get(strategy, file_kind(p))
    return parts_from_messages(encoder.encode(p, **kwargs))
