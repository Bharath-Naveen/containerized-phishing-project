"""Rule-based semantic signals from title + visible text."""

from __future__ import annotations

import re
from typing import Any, Dict

_BRANDS = ("amazon", "google", "microsoft", "paypal", "apple", "facebook", "netflix", "stripe")


def extract_semantic_features(title: str, visible_text: str) -> Dict[str, Any]:
    blob = f"{title or ''} {visible_text or ''}".lower()
    out: Dict[str, Any] = {}
    out["login_keyword_present"] = int(bool(re.search(r"\b(login|sign in|sign-in|log in)\b", blob)))
    out["verify_keyword_present"] = int(bool(re.search(r"\b(verify|verification|validate)\b", blob)))
    out["payment_keyword_present"] = int(bool(re.search(r"\b(payment|checkout|invoice|billing)\b", blob)))
    out["support_keyword_present"] = int(bool(re.search(r"\b(support|help desk|customer service)\b", blob)))
    out["recovery_keyword_present"] = int(
        bool(re.search(r"\b(recover|reset password|forgot password|account recovery)\b", blob))
    )
    out["urgency_keyword_present"] = int(
        bool(re.search(r"\b(urgent|immediately|within 24 hours|act now|suspended)\b", blob))
    )
    out["brand_mention_count"] = sum(blob.count(b) for b in _BRANDS)

    action = "unknown"
    if out["login_keyword_present"]:
        action = "login"
    elif out["payment_keyword_present"]:
        action = "payment"
    elif out["verify_keyword_present"]:
        action = "verify"
    elif out["recovery_keyword_present"]:
        action = "recovery"
    out["inferred_action_rule_based"] = action

    brand = "unknown"
    for b in _BRANDS:
        if b in blob:
            brand = b
            break
    out["inferred_brand_rule_based"] = brand
    out["semantic_missing"] = int(not blob.strip())
    return out
