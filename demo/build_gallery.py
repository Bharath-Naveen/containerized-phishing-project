"""Replay-gallery and honesty-panel data for the portfolio demo (rebuild Phase 6).

Everything numeric comes from files already in the repo; nothing is re-scored on the test split:
  * verdicts, layer signals and hard blockers: metrics/results/tuning/test_after.json (final rules) and
    test_before.json (rules before tuning), the frozen-snapshot replay of the test split,
  * page facts (title, where the visit ended, redirects, capture status): the frozen snapshot itself,
    data/evaluation/frozen/live_snapshot_20260930.jsonl.gz (read only),
  * the URL-model score: the shipped model's calibrated probability (predict_layer1, URL text only;
    no page is fetched). test_after.json stores the score after the official-brand display cap, so the
    uncapped value is recomputed here and both are kept.
  * honesty panel: metrics/results/evaluation.json and the summaries of the two replay files.

The short "why" text for each case is written by hand from those signals and is labeled as such.

Usage: python demo/build_gallery.py   -> demo/build/gallery.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFTER = ROOT / "metrics" / "results" / "tuning" / "test_after.json"
BEFORE = ROOT / "metrics" / "results" / "tuning" / "test_before.json"
SNAP = ROOT / "data" / "evaluation" / "frozen" / "live_snapshot_20260930.jsonl.gz"
EVAL = ROOT / "metrics" / "results" / "evaluation.json"
OUT = ROOT / "demo" / "build" / "gallery.json"

# (snapshot id, category, hand-written explanation of the recorded signals)
CASES = [
    (11, "correct", "Brand site, called legitimate",
     "The URL model alone scores this address 0.646, on the phishing side. With the page loaded, the brand on the page matches the domain, the visit stayed on wellsfargo.com, there is no password form, no form posts off-site, and the domain is in the official-domain prior. The only phishing signal is scripts from many unrelated third-party domains, which is weak next to that legitimacy evidence."),
    (43, "correct", "Real sign-in page, called legitimate",
     "A genuine sign-in page is the hardest kind of legitimate page, because a phishing kit copies exactly this. The URL model scores it 0.888. The page evidence wins: the visit stayed inside apple.com, the brand matches the domain, the domain is in the official-domain prior, and no form posts off-site."),
    (107, "false_alarm", "Brand site, wrongly called phishing",
     "One of the 4 official brand pages (of 136) that the full system flags. The brand check reports a strong brand-to-domain mismatch on this homepage, and the page loads scripts from many unrelated third-party domains. Together with a high URL score, that crosses the phishing threshold. Tuning removed the wrapper hard blocker it had before, but the score alone still gets there. The /about page on the same site (first card) is cleared, so this is a weakness in the rules, not in the domain."),
    (266, "fixed_by_tuning", "Sign-in page fixed by the rule tuning",
     "/home sends you to Dropbox's login page. Before tuning, its password form, which is submitted by JavaScript and so has no form action, tripped the credential-harvesting hard blocker, and the page was called phishing. The tuning, done on the validation half only, made the judge treat that as context on a coherent first-party page: same domain, the page names the site, valid HTTPS, nothing posted off-site."),
    (420, "caught", "Phishing caught by the page, not the URL",
     "The URL model gives 0.646, which is not enough on its own. The page gives it away: it presents itself as Google Docs on an unrelated domain and shows a login form, which fires the credential-harvesting hard blocker and a strong brand-to-domain mismatch."),
    (421, "caught", "Phishing caught by URL and page together",
     "A Google sign-in clone served from a bare IP address over plain HTTP. The URL model is near certain and the host itself is a suspicious pattern. The page adds a credential-harvesting form, obfuscated script around the login, and a sign-in surface without HTTPS."),
    (553, "missed", "Missed: clean-looking page on a free host",
     "This is the known weakness. The URL model scores it 0.985, but the captured page had no password form, no off-site form and plenty of content, so the legitimacy signals outweighed the URL score. The rules cannot turn 'high URL score plus a clean page' into a phishing verdict without also flagging popular legitimate homepages that the URL model also scores high."),
    (362, "missed", "Missed: the phishing content was never seen",
     "The capture got a page titled 'Just a moment...', a bot-check interstitial, not the phishing page. Kits like this most likely show harmless content to data-center browsers, so the page looked clean. The URL, a random domain with a query string, did not look suspicious to the model either (0.176)."),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def defang(url: str) -> str:
    u = re.sub(r"^http", "hxxp", url, flags=re.I)
    m = re.match(r"^(hxxps?://)([^/?#]*)(.*)$", u, flags=re.I)
    if not m:
        return u.replace(".", "[.]")
    return m.group(1) + m.group(2).replace(".", "[.]") + m.group(3)


def host_of(url: str) -> str:
    from urllib.parse import urlsplit

    try:
        return (urlsplit(url).hostname or "")
    except ValueError:
        return ""


def main() -> None:
    from phishguard.app.ml_layer1 import predict_layer1, runtime_models_dir

    want = json.loads((ROOT / "models" / "layer1" / "MANIFEST.json").read_text())["files"]["layer1_bundle.joblib"]["sha256"]
    assert sha256(runtime_models_dir() / "layer1_bundle.joblib") == want, "not the shipped model"
    after = json.loads(AFTER.read_text())
    before = {r["id"]: r for r in json.loads(BEFORE.read_text())["rows"]}
    rows = {r["id"]: r for r in after["rows"]}
    want = {c[0] for c in CASES}
    snap = {}
    with gzip.open(SNAP, "rt", encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        for line in fh:
            d = json.loads(line)
            if d.get("id") in want:
                assert d.get("split") == "test"
                c = d.get("capture") or {}
                snap[d["id"]] = {
                    "title": (c.get("title") or "").strip(),
                    "final_url": c.get("final_url") or "",
                    "redirect_count": c.get("redirect_count"),
                    "uses_https": c.get("uses_https"),
                    "capture_ok": d.get("capture_ok"),
                }
    cases = []
    for sid, cat, headline, why in CASES:
        r, s = rows[sid], snap[sid]
        assert r["url"] and s
        phishing = r["set"] == "fresh_phishing"
        ml = predict_layer1(r["url"])
        final_host = host_of(s["final_url"])
        input_host = host_of(r["url"])
        cases.append({
            "id": sid,
            "category": cat,
            "headline": headline,
            "set": r["set"],
            "truth": "phishing (reported to the PhishStats feed)" if phishing else "legitimate (official brand URL)",
            "url_display": defang(r["url"]) if phishing else r["url"],
            "defanged": phishing,
            "verdict": r["verdict"],
            "verdict_before_tuning": before[sid]["verdict"],
            "passes": r["passes"],
            "url_model_calibrated": ml["phish_proba_calibrated"],
            "url_model_display_recorded": r["layer1_p"],
            "hard_blockers": r["hard_blockers"],
            "phishing_signals": r["evidence"]["evidence_phishing_signals"],
            "legitimacy_signals": r["evidence"]["evidence_legitimacy_signals"],
            "ambiguity_signals": r["evidence"]["evidence_ambiguity_signals"],
            "phishing_score": r["evidence"]["evidence_phishing_score"],
            "legitimacy_score": r["evidence"]["evidence_legitimacy_score"],
            "page": {
                "title": s["title"],
                "ended_on": (defang(final_host) if phishing else final_host) if final_host else "",
                "same_host_as_input": final_host == input_host,
                "redirects": s["redirect_count"],
                "https": s["uses_https"],
                "captured": s["capture_ok"],
            },
            "why_hand_written": why,
        })

    ev = json.loads(EVAL.read_text())
    summ_after = after["summary"]
    summ_before = json.loads(BEFORE.read_text())["summary"]
    ext = ev["external_sets"] if isinstance(ev["external_sets"], dict) else {}
    dash = ev["dashboard_ml_only"]["sets"]
    honesty = {
        "brand_full_system": {"flagged": summ_after["per_set"]["official_brand"]["verdicts"]["likely_phishing"], "n": summ_after["per_set"]["official_brand"]["n"],
                               "source": "metrics/results/tuning/test_after.json (summary.per_set.official_brand)"},
        "brand_full_system_before_tuning": {"flagged": summ_before["per_set"]["official_brand"]["verdicts"]["likely_phishing"], "n": summ_before["per_set"]["official_brand"]["n"],
                                            "source": "metrics/results/tuning/test_before.json"},
        "brand_url_model_flag_rate": {"value": dash["official_brand_test_half"]["layer1_flag_rate"], "n": dash["official_brand_test_half"]["n"],
                                      "source": "metrics/results/evaluation.json (dashboard_ml_only.sets.official_brand_test_half.layer1_flag_rate)"},
        "brand_ml_only_dashboard": {"verdicts": dash["official_brand_test_half"]["verdicts"], "source": "metrics/results/evaluation.json (dashboard_ml_only)"},
        "fresh_phishing_full_system": {"caught": summ_after["phishing_recall"]["likely_phishing"], "n": summ_after["phishing_recall"]["n"],
                                       "called_legitimate": summ_after["phishing_recall"]["likely_legitimate"],
                                       "source": "metrics/results/tuning/test_after.json (summary.phishing_recall)"},
        "phishstats_ml_only_dashboard": {"verdicts": dash["phishstats"]["verdicts"], "n": dash["phishstats"]["n"], "layer1_flag_rate": dash["phishstats"]["layer1_flag_rate"],
                                         "source": "metrics/results/evaluation.json (dashboard_ml_only.sets.phishstats)"},
        "legit_pages_full_system": {"flagged": summ_after["legit_false_alarms"]["likely_phishing"], "n": summ_after["legit_false_alarms"]["n"],
                                    "source": "metrics/results/tuning/test_after.json (summary.legit_false_alarms)"},
    }
    out = {
        "about": "Replay gallery and honesty-panel data for the portfolio demo, built by demo/build_gallery.py.",
        "replay": {
            "file": "metrics/results/tuning/test_after.json",
            "sha256": sha256(AFTER),
            "command": after.get("command"),
            "split": after.get("split"),
            "snapshot_file": after["snapshot"]["file"],
            "snapshot_sha256": after["snapshot"]["sha256"],
            "snapshot_created_utc": after["snapshot"]["created_utc"],
            "rules_commit": after["provenance"]["git_commit"],
            "model_bundle_sha256": after["provenance"]["runtime_model"]["bundle_sha256"],
            "login_interaction": header.get("login_interaction"),
        },
        "cases": cases,
        "honesty": honesty,
        "evaluation_commit": ev["provenance"]["git_commit"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cases": [(c["id"], c["category"], c["verdict"], c["url_model_calibrated"]) for c in cases], "honesty": {k: {kk: vv for kk, vv in v.items() if kk != "source"} for k, v in honesty.items()}}, indent=1))


if __name__ == "__main__":
    main()
