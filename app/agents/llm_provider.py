from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_llm() -> Optional[object]:
    """Return a configured LangChain chat LLM instance if available.

    Prefers OpenAI when `OPENAI_API_KEY` is set.
    Returns None if no LLM configuration is available.
    """

    # Allow disabling LLM usage explicitly
    if os.getenv("USE_LLM", "true").lower() in {"0", "false", "no"}:
        return None

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        # Default lightweight, cost-effective model
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            return ChatOpenAI(model=model, api_key=openai_api_key, temperature=0.3)
        except Exception as e:
            # Graceful fallback if environment/proxy issues occur
            print(f"[llm_provider] OpenAI LLM unavailable, falling back: {e}")
            return None

    return None


def llm_available() -> bool:
    return get_llm() is not None

