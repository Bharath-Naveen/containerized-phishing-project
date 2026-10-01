"""Frozen snapshot: freeze with a fake browser, then replay offline twice and get identical verdicts."""

from pathlib import Path
from unittest.mock import patch

import phishguard.evaluation.snapshot as snap
from phishguard.app.schemas import CaptureResult

HTML = """<html><head><title>Sign in</title></head><body><form action="https://collect.example.net/p" method="post">
<input name="email"><input type="password" name="pw"><button>Sign in</button></form></body></html>"""


def _fake_capture(tmp: Path):
    def cap(url, cfg, namespace="suspicious"):
        h = tmp / "page.html"
        h.write_text(HTML, encoding="utf-8")
        return CaptureResult(original_url=url, final_url=url, title="Sign in", screenshot_path="",
                             fullpage_screenshot_path="", html_path=str(h), visible_text="Sign in",
                             capture_strategy="playwright_headless", settled_successfully=True, uses_https=True)
    return cap


ITEMS = [
    {"set": "fresh_phishing", "url": "https://secure-login-verify.example-bank.top/signin", "expected": "phishing",
     "split": "val", "registered_domain": "example-bank.top", "popular_domain": False},
    {"set": "official_brand", "url": "https://www.wikipedia.org/", "expected": "not_phishing", "split": "test"},
]


def test_freeze_and_replay_are_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(snap, "project_root", lambda: tmp_path)
    out = tmp_path / "snap.jsonl.gz"
    with patch.object(snap, "_items", return_value=(ITEMS, {"note": "test"})), \
            patch("phishguard.app.capture.capture_url", side_effect=_fake_capture(tmp_path)):
        snap.freeze(out=out)
    header, items = snap.load_snapshot(out)
    assert header["n_items"] == 2 and len(items) == 2
    assert items[0]["html"].startswith("<html>") and items[0]["capture_ok"]

    r1 = snap.replay("all", out, tmp_path / "r1.json")
    r2 = snap.replay("all", out, tmp_path / "r2.json")
    assert [r["verdict"] for r in r1["rows"]] == [r["verdict"] for r in r2["rows"]]
    assert r1["summary"]["phishing_recall"]["n"] == 1
    assert r1["summary"]["legit_false_alarms"]["n"] == 1

    val = snap.replay("val", out, tmp_path / "v.json")
    assert [r["split"] for r in val["rows"]] == ["val"]
    assert not (tmp_path / "metrics" / "results" / "test_access_log.jsonl").exists()
    snap.replay("test", out, tmp_path / "t.json")
    log = (tmp_path / "metrics" / "results" / "test_access_log.jsonl").read_text().splitlines()
    assert len(log) == 1


def test_hash_split_is_stable():
    assert snap.hash_split("example.com") == snap.hash_split("example.com")
    splits = {snap.hash_split(f"d{i}.com") for i in range(50)}
    assert splits == {"val", "test"}
