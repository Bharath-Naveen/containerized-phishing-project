"""Three-way verdict policy on a combined phishing score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

Verdict3 = Literal["likely_legitimate", "uncertain", "likely_phishing"]


@dataclass(frozen=True)
class Verdict3WayConfig:
    """Wider uncertain band reduces overconfident false positives on tricky legit URLs."""

    combined_high: float = 0.56  # >= likely phishing
    combined_low: float = 0.38  # <= likely legitimate


def verdict_3way(combined_score: float, cfg: Verdict3WayConfig | None = None) -> Tuple[Verdict3, str]:
    c = cfg or Verdict3WayConfig()
    x = float(combined_score)
    if x <= c.combined_low:
        return "likely_legitimate", f"combined={x:.3f} <= low={c.combined_low}"
    if x >= c.combined_high:
        return "likely_phishing", f"combined={x:.3f} >= high={c.combined_high}"
    return "uncertain", f"low={c.combined_low} < combined={x:.3f} < high={c.combined_high}"
