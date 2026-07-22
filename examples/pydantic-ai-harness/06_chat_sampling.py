"""Variant 6 — `chat`, where the server borrows *your* model.

The other variants either call mm's own LLM or hand content back for your
agent to reason over. MCP sampling inverts that: the server asks the
*client* to run inference. mm does what it is good at — turning a video or
an 8-page PDF into text — and your model answers the question.

The server needs no API key for this, and the answer comes from whatever
model the client already has.

    mm mcp serve                                                  # terminal 1
    uv run python examples/pydantic-ai-harness/06_chat_sampling.py # terminal 2

Any tool-calling model works as the *agent*; the sampling handler below is
what actually answers, so this example drives the tool directly to keep the
sampling path in focus.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _model import DATA_DIR, describe_profile
from fastmcp import Client

from mm.mcp import mcp


def sampling_handler(messages, params, context) -> str:
    """Answer a sampling request with the client's own LLM.

    This is the piece an MCP client supplies. Anything can go here — a
    local model, a hosted API, a human in the loop.
    """
    from openai import OpenAI

    from mm.profile import get_profile

    profile = get_profile()
    client = OpenAI(base_url=profile.base_url, api_key=profile.api_key)

    chat_messages = []
    if params.systemPrompt:
        chat_messages.append({"role": "system", "content": params.systemPrompt})
    chat_messages += [{"role": m.role, "content": m.content.text} for m in messages]

    sent = sum(len(m["content"]) for m in chat_messages)
    print(f"  [sampling] server asked the client to answer — {sent:,} chars of context")

    response = client.chat.completions.create(
        model=profile.model,
        messages=chat_messages,
        max_tokens=params.maxTokens or 512,
    )
    return response.choices[0].message.content or ""


async def main() -> None:
    print(f"[{describe_profile()}]\n")
    data = Path(DATA_DIR).expanduser()

    asks = [
        ("Summarize this document in 3 bullet points.", data / "BillDownload-8pg.pdf"),
        ("What vehicle is shown, and what colour is it?", data / "1-vqa-car.jpg"),
    ]

    async with Client(mcp, sampling_handler=sampling_handler) as client:
        for instruction, path in asks:
            print(f"> {instruction}  ({path.name})")
            result = await client.call_tool("chat", {"instruction": instruction, "path": str(path)})
            print(f"{result.data}\n")


if __name__ == "__main__":
    asyncio.run(main())
