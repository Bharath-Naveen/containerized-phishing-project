"""Tests for optional Layer-1 multi-model agreement and EAL wiring."""

from unittest.mock import MagicMock, patch

from src.app_v1.analyze_dashboard import _apply_evidence_adjudication_layer, build_dashboard_analysis
from src.app_v1.ml_layer1 import build_model_agreement_from_outputs


def _row(name: str, p: float) -> dict:
    return {
        "model_name": name,
        "phish_probability": p,
        "predicted_phishing": bool(p >= 0.5),
    }


def _primary_ml(path: str = "/models/layer1_primary.joblib", predicted: bool = False) -> dict:
    return {
        "model_path": path,
        "phish_proba": 0.42,
        "phish_proba_calibrated": 0.42,
        "predicted_phishing": predicted,
    }


def test_missing_agreement_models_unavailable() -> None:
    out = build_model_agreement_from_outputs([], ml_primary_prob=0.4, primary_ml=_primary_ml())
    assert out["ml_consensus"] == "unavailable"
    assert out["ml_model_votes_phishing"] == 0
    assert out["ml_models_available"] == []


def test_single_witness_unavailable() -> None:
    out = build_model_agreement_from_outputs(
        [_row("xgboost", 0.9)],
        ml_primary_prob=0.5,
        primary_ml=_primary_ml(),
    )
    assert out["ml_consensus"] == "unavailable"


def test_strong_phishing_consensus_three_of_four() -> None:
    rows = [
        _row("logistic_regression", 0.9),
        _row("random_forest", 0.8),
        _row("xgboost", 0.85),
        _row("lightgbm", 0.2),
    ]
    out = build_model_agreement_from_outputs(rows, ml_primary_prob=0.9, primary_ml=_primary_ml())
    assert out["ml_consensus"] == "strong_phishing"
    assert out["ml_model_votes_phishing"] == 3


def test_strong_legitimate_consensus() -> None:
    rows = [
        _row("logistic_regression", 0.1),
        _row("random_forest", 0.2),
        _row("xgboost", 0.15),
        _row("lightgbm", 0.05),
    ]
    out = build_model_agreement_from_outputs(rows, ml_primary_prob=0.1, primary_ml=_primary_ml())
    assert out["ml_consensus"] == "strong_legitimate"
    assert out["ml_model_votes_legitimate"] == 4


def test_boosted_only_pattern() -> None:
    rows = [
        _row("logistic_regression", 0.2),
        _row("random_forest", 0.3),
        _row("xgboost", 0.9),
        _row("lightgbm", 0.1),
    ]
    out = build_model_agreement_from_outputs(rows, ml_primary_prob=0.25, primary_ml=_primary_ml())
    assert out["ml_consensus"] == "boosted_only"
    assert out["boosted_models_phishing"] is True


def test_split_vote() -> None:
    rows = [
        _row("logistic_regression", 0.9),
        _row("random_forest", 0.1),
        _row("xgboost", 0.85),
        _row("lightgbm", 0.15),
    ]
    out = build_model_agreement_from_outputs(rows, ml_primary_prob=0.5, primary_ml=_primary_ml())
    assert out["ml_consensus"] == "split"
    assert out["ml_model_votes_phishing"] == 2


def test_eal_strong_phishing_evidence_signal() -> None:
    ml = {
        "phish_proba": 0.5,
        "phish_proba_calibrated": 0.5,
        "model_agreement": build_model_agreement_from_outputs(
            [_row("logistic_regression", 0.9), _row("random_forest", 0.9), _row("xgboost", 0.9), _row("lightgbm", 0.2)],
            ml_primary_prob=0.5,
            primary_ml=_primary_ml(),
        ),
    }
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.5, "reasons": []},
        ml=ml,
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 0, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "ml_consensus_strong_phishing" in (out.get("evidence_phishing_signals") or [])


def test_eal_strong_legitimate_evidence_signal() -> None:
    ml = {
        "phish_proba": 0.5,
        "phish_proba_calibrated": 0.5,
        "model_agreement": build_model_agreement_from_outputs(
            [_row("logistic_regression", 0.1), _row("random_forest", 0.1), _row("xgboost", 0.1), _row("lightgbm", 0.1)],
            ml_primary_prob=0.1,
            primary_ml=_primary_ml(),
        ),
    }
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.5, "reasons": []},
        ml=ml,
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 0, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "ml_consensus_strong_legitimate" in (out.get("evidence_legitimacy_signals") or [])


def test_eal_boosted_only_adds_ambiguity_not_auto_phish() -> None:
    ml = {
        "phish_proba": 0.35,
        "phish_proba_calibrated": 0.35,
        "model_agreement": build_model_agreement_from_outputs(
            [
                _row("logistic_regression", 0.2),
                _row("random_forest", 0.3),
                _row("xgboost", 0.9),
                _row("lightgbm", 0.1),
            ],
            ml_primary_prob=0.35,
            primary_ml=_primary_ml(),
        ),
    }
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.45, "reasons": []},
        ml=ml,
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 0, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "boosted_models_only_flag_phishing" in (out.get("evidence_ambiguity_signals") or [])
    assert "ml_consensus_strong_phishing" not in (out.get("evidence_phishing_signals") or [])


def test_eal_split_adds_ambiguity() -> None:
    ml = {
        "phish_proba": 0.5,
        "phish_proba_calibrated": 0.5,
        "model_agreement": build_model_agreement_from_outputs(
            [
                _row("logistic_regression", 0.9),
                _row("random_forest", 0.1),
                _row("xgboost", 0.85),
                _row("lightgbm", 0.15),
            ],
            ml_primary_prob=0.5,
            primary_ml=_primary_ml(),
        ),
    }
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.5, "reasons": []},
        ml=ml,
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 0, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "ml_consensus_split" in (out.get("evidence_ambiguity_signals") or [])


def test_hard_blocker_dominates_with_disagreement() -> None:
    ml = {
        "phish_proba": 0.2,
        "phish_proba_calibrated": 0.2,
        "model_agreement": build_model_agreement_from_outputs(
            [
                _row("logistic_regression", 0.1),
                _row("random_forest", 0.9),
                _row("xgboost", 0.1),
                _row("lightgbm", 0.9),
            ],
            ml_primary_prob=0.2,
            primary_ml=_primary_ml(),
        ),
    }
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.4, "reasons": []},
        ml=ml,
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 1,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
        },
        html_structure_risk=0.5,
        html_dom_risk=0.4,
        host_path_reasoning={},
        platform_context={},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "cross_domain_credential_form_action" in (out.get("evidence_hard_blockers") or [])


def test_high_probability_spread_ambiguity() -> None:
    ml = {
        "phish_proba": 0.5,
        "phish_proba_calibrated": 0.5,
        "model_agreement": {
            **build_model_agreement_from_outputs(
                [_row("logistic_regression", 0.05), _row("random_forest", 0.95)],
                ml_primary_prob=0.5,
                primary_ml=_primary_ml(),
            ),
            "ml_prob_spread": 0.95,
        },
    }
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.5, "reasons": []},
        ml=ml,
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 0, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "high_model_probability_spread" in (out.get("evidence_ambiguity_signals") or [])


def test_no_probability_averaging_in_dashboard_pipeline() -> None:
    fake_ml = {
        "phish_proba": 0.42,
        "phish_proba_model_raw": 0.42,
        "phish_proba_calibrated": 0.42,
        "predicted_phishing": False,
        "canonical_url": "https://www.example.com",
        "brand_structure_features": {},
        "error": None,
    }
    fake_agreement = build_model_agreement_from_outputs(
        [
            _row("logistic_regression", 0.99),
            _row("random_forest", 0.01),
            _row("xgboost", 0.5),
            _row("lightgbm", 0.5),
        ],
        ml_primary_prob=0.42,
        primary_ml=fake_ml,
    )
    cfg = MagicMock()
    cfg.enable_click_probe = False
    fake = MagicMock()
    fake.as_json.return_value = {
        "final_url": "https://www.example.com",
        "html_path": "",
        "visible_text": "",
        "title": "Example",
        "capture_failed": False,
        "input_registered_domain": "example.com",
        "final_registered_domain": "example.com",
    }
    fake.error = ""
    fake.capture_blocked = False
    fake.capture_strategy = "http_fallback"
    with patch("src.app_v1.analyze_dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("src.app_v1.analyze_dashboard.capture_url", return_value=fake):
            with patch("src.app_v1.analyze_dashboard.predict_layer1", return_value=fake_ml):
                with patch(
                    "src.app_v1.analyze_dashboard.compute_layer1_model_agreement",
                    return_value=fake_agreement,
                ):
                    out, _ = build_dashboard_analysis("https://www.example.com", reinforcement=True)
    layer1 = out.get("layer1_ml") or {}
    assert layer1.get("phish_proba") == 0.42
    assert (layer1.get("model_agreement") or {}).get("ml_primary_prob") == 0.42


def test_primary_only_build_dashboard_still_runs() -> None:
    fake_ml = {
        "phish_proba": 0.4,
        "phish_proba_model_raw": 0.4,
        "phish_proba_calibrated": 0.4,
        "predicted_phishing": False,
        "canonical_url": "https://www.example.com",
        "brand_structure_features": {},
        "error": None,
    }
    empty_agreement = build_model_agreement_from_outputs([], ml_primary_prob=0.4, primary_ml=fake_ml)
    cfg = MagicMock()
    cfg.enable_click_probe = False
    fake = MagicMock()
    fake.as_json.return_value = {
        "final_url": "https://www.example.com",
        "html_path": "",
        "visible_text": "",
        "title": "Example",
        "capture_failed": False,
        "input_registered_domain": "example.com",
        "final_registered_domain": "example.com",
    }
    fake.error = ""
    fake.capture_blocked = False
    fake.capture_strategy = "http_fallback"
    with patch("src.app_v1.analyze_dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("src.app_v1.analyze_dashboard.capture_url", return_value=fake):
            with patch("src.app_v1.analyze_dashboard.predict_layer1", return_value=fake_ml):
                with patch(
                    "src.app_v1.analyze_dashboard.compute_layer1_model_agreement",
                    return_value=empty_agreement,
                ):
                    out, _ = build_dashboard_analysis("https://www.example.com", reinforcement=True)
    assert (out.get("layer1_ml") or {}).get("model_agreement", {}).get("ml_consensus") == "unavailable"
    assert (out.get("verdict") or {}).get("evidence_adjudication_applied") is True
