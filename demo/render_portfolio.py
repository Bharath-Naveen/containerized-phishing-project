"""Render the demo section of projects/phishing-detection-system.html (rebuild Phase 6).

Reads demo/build/gallery.json and demo/parity/results/parity_full_features_models.json and parity_dashboard_verdicts.json and writes
demo/build/demo-section.html: the "Try any URL" box (filled in by demo.js), the replay gallery and the
honesty panel. Every number in the HTML is taken from those two files; none is typed here.

Usage: python demo/render_portfolio.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "demo" / "build"
REPO = "https://github.com/Bharath-Naveen/containerized-phishing-project"

VERDICT = {"likely_phishing": ("Likely phishing", "pd-phishing"), "uncertain": ("Uncertain", "pd-uncertain"), "likely_legitimate": ("Likely legitimate", "pd-legitimate")}
CATEGORY = {
    "correct": ("Correct", "ok"), "false_alarm": ("False alarm", "bad"), "fixed_by_tuning": ("Fixed by tuning", "ok"),
    "caught": ("Caught", "ok"), "missed": ("Missed", "bad"),
}
SIGNALS = {
    "high_ml_score": "URL score 0.85 or higher", "elevated_ml_score": "URL score 0.70 to 0.85", "moderate_ml_score": "URL score 0.50 to 0.70",
    "ml_consensus_strong_phishing": "3+ of 4 models vote phishing", "ml_consensus_strong_legitimate": "3+ of 4 models vote legitimate",
    "boosted_models_only_flag_phishing": "only the boosted models vote phishing", "high_model_probability_spread": "models disagree by more than 0.4",
    "suspicious_host_pattern": "suspicious host pattern", "strong_brand_domain_mismatch": "strong brand-to-domain mismatch",
    "auth_context_on_non_official_domain": "login page on a non-official domain", "many_unrelated_third_party_domains": "scripts from many unrelated third-party domains",
    "unrelated_third_party_network_domains": "requests to unrelated third-party domains",
    "high_js_obfuscation_auth_context": "obfuscated script around a login", "insecure_auth_surface": "sign-in without valid HTTPS",
    "same_domain_consistency": "visit stayed on the same domain", "brand_domain_coherence": "page brand matches the domain",
    "coherent_brand_host_identity": "host identity fits the brand", "valid_https_transport": "valid HTTPS",
    "official_domain_trust_prior": "domain in the official-domain prior", "no_credential_capture": "no password form",
    "no_cross_domain_forms": "no form posts off-site", "rich_nav_footer_support": "full site navigation and help links",
    "first_party_auth_flow_consistency": "first-party sign-in flow", "content_rich_page": "content-rich page", "low_html_dom_risk": "low HTML risk",
    "first_party_script_submitted_login": "login submitted by the site's own script", "official_content_wrapper_pattern": "official content wrapper",
    "coherent_brand_wrapper_pattern": "brand-coherent wrapper", "first_party_login_ml_disagreement": "URL model disagrees with a first-party login",
    "ml_structural_disagreement": "URL and page evidence disagree", "same_domain_on_suspicious_registrable": "same domain, but a suspicious one",
    "url_tracking_or_length_noise": "encoded or tracking-style URL",
    "credential_harvesting_pattern": "credential-harvesting form", "wrapper_or_interstitial_redirect_pattern": "wrapper or interstitial redirect",
}


def e(s) -> str:
    return html.escape(str(s), quote=True)


def chip(text: str, kind: str) -> str:
    return f'<span class="pd-chip {kind}">{e(text)}</span>'


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    g = json.loads((BUILD / "gallery.json").read_text(encoding="utf-8"))
    pres = ROOT / "demo" / "parity" / "results"
    parity = json.loads((pres / "parity_full_features_models.json").read_text(encoding="utf-8"))
    pdash = json.loads((pres / "parity_dashboard_verdicts.json").read_text(encoding="utf-8"))
    h = g["honesty"]
    rp = g["replay"]
    out = []
    out.append('<section class="psec pd" id="demo">')
    out.append('<h2>Try it<span class="k">live demo</span></h2>')
    out.append('<p class="pd-lede">Paste any URL and the URL model and the rules that can run without visiting the page will score it here, in your browser. '
               'Below that are replays of the full system, page capture included, on real brand sites and real phishing pages, including the ones it gets wrong.</p>')

    # Try any URL
    out.append('<h3>Try any URL</h3>')
    out.append('<div class="pd-box" id="pd-try" data-assets="../assets/phishing-demo/">')
    out.append('<form class="pd-form" autocomplete="off"><label for="pd-url" class="sr-only" style="position:absolute;left:-9999px">URL to check</label>'
               '<input id="pd-url" type="text" inputmode="url" spellcheck="false" placeholder="https://example.com/login" maxlength="2048">'
               '<button type="submit">Check URL</button></form>')
    examples = ["https://www.paypal.com/signin", "http://google-login-secure.xyz/signin", "https://en.wikipedia.org/wiki/Phishing", "http://192.168.0.1/login.php"]
    out.append('<div class="pd-examples"><span>Try:</span>' + "".join(f'<button type="button" data-example="{e(x)}">{e(x)}</button>' for x in examples) + "</div>")
    out.append('<p class="pd-privacy">The URL never leaves your browser. Nothing is fetched from it, and it is not sent to any server. This is the URL model only, without live page capture.</p>')
    out.append('<p class="pd-status" aria-live="polite"></p><div class="pd-out" aria-live="polite"></div>')
    out.append("</div>")
    n_rows = parity["rows"]
    held = parity["by_set"].get("heldout_test", 0)
    md = parity["max_abs_diff"]
    pmax = max(md[k] for k in ("p_raw", "p_cal", "logistic_regression", "random_forest", "xgboost", "lightgbm"))
    same_feat = n_rows - parity["feature_rows_mismatch"]
    same_cal = n_rows - parity["p_cal_rounded_mismatch"]
    same_cons = n_rows - parity["consensus_mismatch"]
    nd = pdash["dashboard_rows"]
    same_v = nd - pdash["verdict_mismatch"]
    out.append(
        f'<p class="pd-src">Checked against the Python app on {n_rows:,} URLs ({held:,} from the held-out test file plus the curated evaluation sets): '
        f'identical features for {same_feat:,} of {n_rows:,}, the same calibrated score (to the 6 decimals the app reports) for {same_cal:,}, and the same 4-model vote for {same_cons:,}. '
        f'On {nd:,} of them the real dashboard was also run, and the ML-only verdict matched for {same_v:,} of {nd:,}. '
        f'Largest probability difference: {pmax:.1e} (tolerance 1e-6). '
        f'<a href="{REPO}/tree/main/demo" target="_blank" rel="noopener">Parity test and export code</a>.</p>'
    )

    # Replay gallery
    out.append('<h3>Replays of the full system</h3>')
    out.append(f'<p class="pd-lede">Each page was captured once by GitHub Actions on {e(rp["snapshot_created_utc"][:10])} and saved, then replayed through all five layers with the final rules. '
               f'These cases are from the test half, which was not used to tune the rules. Phishing addresses are shown defanged (hxxp, [.]) and are never links.</p>')
    out.append('<div class="pd-gallery">')
    for c in g["cases"]:
        vt, vc = VERDICT[c["verdict"]]
        ct, ck = CATEGORY[c["category"]]
        url = c["url_display"]
        if len(url) > 110:
            url = url[:107] + "..."
        p = c["page"]
        facts = [f'<b>Page title:</b> {e(p["title"]) if p["title"] else "(none)"}']
        if p["ended_on"]:
            facts.append(f'<b>Ended on:</b> {e(p["ended_on"])}' + ("" if p["same_host_as_input"] else " (another host)"))
        facts.append(f'<b>URL model alone:</b> {c["url_model_calibrated"]:.3f}')
        if c["verdict_before_tuning"] != c["verdict"]:
            facts.append(f'<b>Before tuning:</b> {e(VERDICT[c["verdict_before_tuning"]][0])}')
        chips = [chip(SIGNALS.get(b, b), "pd-phishing") for b in c["hard_blockers"]]
        chips += [chip(SIGNALS.get(s, s), "pd-phishing") for s in c["phishing_signals"][:3]]
        chips += [chip(SIGNALS.get(s, s), "pd-legitimate") for s in c["legitimacy_signals"][:3]]
        allsig = "".join(
            f"<li>{e(label)}: {e(', '.join(SIGNALS.get(s, s) for s in lst) or 'none')}</li>"
            for label, lst in (("Hard blockers", c["hard_blockers"]), ("Phishing signals", c["phishing_signals"]),
                               ("Legitimacy signals", c["legitimacy_signals"]), ("Ambiguity signals", c["ambiguity_signals"]))
        )
        out.append(
            f'<article class="pd-case"><div class="pd-ctop"><span class="pd-tag {ck}">{e(ct)}</span><span class="pd-chip {vc}">{e(vt)}</span></div>'
            f'<h4>{e(c["headline"])}</h4><div class="pd-url" title="{e(c["truth"])}">{e(url)}</div>'
            f'<p class="pd-facts">{" &middot; ".join(facts)}</p><div class="pd-chips">{"".join(chips)}</div>'
            f'<p class="pd-why">{e(c["why_hand_written"])}<small>Truth: {e(c["truth"])}. Explanation written by me from the recorded signals.</small></p>'
            f'<details><summary>All recorded signals (score {c["phishing_score"]} phishing vs {c["legitimacy_score"]} legitimacy)</summary><ul>{allsig}</ul></details></article>'
        )
    out.append("</div>")

    # Honesty panel
    b, bt, fp = h["brand_full_system"], h["brand_full_system_before_tuning"], h["fresh_phishing_full_system"]
    bu, lp, ps = h["brand_url_model_flag_rate"], h["legit_pages_full_system"], h["phishstats_ml_only_dashboard"]
    out.append('<h3>The honest numbers</h3>')
    out.append('<div class="pd-honest">')
    out.append(f'<div class="pd-stat"><div class="n">{b["flagged"]} / {b["n"]}</div><div class="l">real brand sites the full system calls phishing ({bt["flagged"]} before the rule tuning). The URL model alone flags {pct(bu["value"])} of them.</div><div class="s">test half, frozen snapshot replay</div></div>')
    out.append(f'<div class="pd-stat weak"><div class="n">{fp["caught"]} / {fp["n"]}</div><div class="l">fresh phishing pages caught by the full system; {fp["called_legitimate"]} were called legitimate. <b>Recall on fresh phishing is the known weakness.</b></div><div class="s">PhishStats feed, captured pages, test half</div></div>')
    ml_p = ps["verdicts"]["likely_phishing"]
    out.append(f'<div class="pd-stat"><div class="n">{ml_p} / {ps["n"]}</div><div class="l">PhishStats URLs that ML-only mode (the box above) calls likely phishing; the other {ps["n"] - ml_p} stay uncertain, because without the page it never says legitimate.</div><div class="s">ML-only dashboard, no page capture</div></div>')
    out.append("</div>")
    out.append(
        f'<p class="pd-src">What this means: the system is good at not crying wolf on real sites, and it is not yet good at catching brand-new phishing. '
        f'Across all {lp["n"]} legitimate pages in the test replay, {lp["flagged"]} were labeled phishing. Every number here is generated from '
        f'<a href="{REPO}/blob/main/metrics/VERIFIED_METRICS.md" target="_blank" rel="noopener">VERIFIED_METRICS.md</a> and the replay results in '
        f'<a href="{REPO}/tree/main/metrics/results/tuning" target="_blank" rel="noopener">metrics/results/tuning</a>, not typed by hand.</p>'
    )
    out.append("</section>")
    (BUILD / "demo-section.html").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {BUILD / 'demo-section.html'}")


if __name__ == "__main__":
    main()
