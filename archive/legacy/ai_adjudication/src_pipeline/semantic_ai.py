"""Optional OpenAI-based semantic explanations (not required for training).

Call from tooling or future UI; batch ``enrich`` uses rule-based semantic features only
unless you extend this module.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def enrich_with_llm(url: str, title: str, visible_snippet: str) -> Dict[str, Any]:
    """Return extra string fields for reporting; no-op without ``OPENAI_API_KEY``."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {}
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "Summarize likely brand and user action for this URL in one sentence. "
            "URL: {u}\nTitle: {t}\nText snippet: {v}\n"
            "Reply JSON with keys: brand_guess, action_guess, explanation."
        ).format(u=url, t=title[:500], v=(visible_snippet or "")[:1500])
        r = client.chat.completions.create(
            model=os.environ.get("PHISH_SEMANTIC_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = (r.choices[0].message.content or "").strip()
        return {"llm_semantic_raw": text}
    except Exception as e:
        return {"llm_semantic_error": str(e)}
