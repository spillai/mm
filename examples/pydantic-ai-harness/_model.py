"""Build a pydantic-ai model from mm's active profile.

Every example imports this, so switching the model behind all of them is
one command — no code edits::

    mm profile use ollama       # fully local
    mm profile use gateway      # self-hosted / remote gateway
    mm --profile ollama ...     # or per-invocation

mm already resolves ``--profile`` > ``MM_PROFILE`` > active profile, so
the agent and the media pipeline always talk to the same endpoint.
"""

from __future__ import annotations

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

DATA_DIR = "~/data/mmbench-tiny"
"""Sample media directory used by the examples (image, video, audio, PDF)."""


def mm_model() -> OpenAIChatModel:
    """Return an OpenAI-compatible model pointed at mm's active profile."""
    from mm.profile import get_profile

    profile = get_profile()
    return OpenAIChatModel(
        profile.model,
        provider=OpenAIProvider(base_url=profile.base_url, api_key=profile.api_key),
    )


def describe_profile() -> str:
    """One-line summary of where inference is going, for example banners."""
    from mm.profile import get_profile

    p = get_profile()
    return f"profile={p.name} model={p.model} base_url={p.base_url}"
