"""Check what your active mm profile's model can actually do.

Not every model can drive every example. Small local models often handle
vision fine but cannot emit tool calls at all — in which case variants 3-5
will silently produce prose instead of calling mm.

    uv run python examples/pydantic-ai-harness/check_model.py
"""

from __future__ import annotations

from _model import describe_profile

PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _client():
    from openai import OpenAI

    from mm.profile import get_profile

    p = get_profile()
    return OpenAI(base_url=p.base_url, api_key=p.api_key), p.model


def supports_tools() -> bool:
    """True if the model emits a tool call when offered one."""
    client, model = _client()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Weather in Paris? Use the tool."}],
            tools=PROBE_TOOL,
        )
    except Exception as e:
        print(f"  tool probe failed: {type(e).__name__}: {e}")
        return False
    return bool(r.choices[0].message.tool_calls)


def supports_vision() -> bool:
    """True if the model accepts image content without erroring."""
    client, model = _client()
    try:
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour?"},
                        {"type": "image_url", "image_url": {"url": PIXEL}},
                    ],
                }
            ],
            max_tokens=8,
        )
    except Exception as e:
        print(f"  vision probe failed: {type(e).__name__}: {e}")
        return False
    return True


def main() -> None:
    print(f"[{describe_profile()}]\n")

    vision = supports_vision()
    tools = supports_tools()

    print(f"  vision (image input)  : {'yes' if vision else 'no'}")
    print(f"  tool calling          : {'yes' if tools else 'no'}\n")

    runnable = ["01_context_builder.py"]
    if vision:
        runnable.append("02_vlm_encoder.py")
    if tools:
        runnable += ["03_direct_tools.py", "04_mcp_server.py", "05_code_mode.py"]

    print("Examples you can run:")
    for name in runnable:
        print(f"  - {name}")

    if not tools:
        print(
            "\nNo tool calling: variants 3-5 need a bigger model.\n"
            "  mm profile use ollama     # then: ollama pull qwen3:8b\n"
            "  mm profile use openrouter # set an api key with `mm profile update`"
        )


if __name__ == "__main__":
    main()
