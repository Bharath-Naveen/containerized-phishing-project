from phishguard.evaluation.live import _passes


def test_pass_rules():
    assert _passes("phishing", "likely_phishing")
    assert not _passes("phishing", "uncertain")
    assert _passes("not_phishing", "uncertain") and _passes("not_phishing", "likely_legitimate")
    assert not _passes("not_phishing", "likely_phishing")
    assert _passes("phishing_or_uncertain", "uncertain") and not _passes("phishing_or_uncertain", "likely_legitimate")


def test_edge_case_file_is_valid():
    import json
    from phishguard.paths import project_root

    cases = json.loads((project_root() / "data" / "evaluation" / "eal_edge_cases.json").read_text())["cases"]
    assert len(cases) == 15
    assert {c["expected"] for c in cases} <= {"phishing", "not_phishing", "phishing_or_uncertain"}
