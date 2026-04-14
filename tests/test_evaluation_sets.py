"""Curated evaluation URL lists."""

from src.pipeline.evaluation_sets import load_hard_legit_rows, load_simple_legit_rows, load_url_suites


def test_hard_legit_load_default() -> None:
    rows = load_hard_legit_rows()
    assert len(rows) >= 5
    assert all("url" in r for r in rows)


def test_url_suites_load() -> None:
    suites = load_url_suites()
    assert "tricky_legit" in suites
    assert len(suites["tricky_legit"]) >= 3
    assert "obvious_legit" in suites
    for u in suites["obvious_legit"]:
        assert "wiki/Phishing" not in u


def test_simple_legit_jsonl_loads() -> None:
    rows = load_simple_legit_rows()
    assert len(rows) >= 500
