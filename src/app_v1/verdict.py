"""Conservative, explainable verdict from comparison features and page heuristics."""

from __future__ import annotations

from .legit_lookup import is_url_on_trusted_brand_root
from .schemas import ComparisonResult, FeatureResult, VerdictResult


def _is_known_first_party_product_domain(final_url: str) -> bool:
    u = (final_url or "").lower()
    return any(
        d in u
        for d in (
            "outlook.office.com",
            "lens.google",
            "drive.google.com",
            "workspace.google.com",
            "mail.google.com",
            "photos.google.com",
        )
    )


def _trusted_brand_benign_landing(comparison: ComparisonResult, features: FeatureResult) -> bool:
    """Known-good brand root in URL, no password-collection signal, no bad post-submit redirect."""
    brand = (comparison.brand_guess or "").strip().lower()
    if brand == "unknown" or not brand:
        return False
    if not is_url_on_trusted_brand_root(brand, features.final_url):
        return False
    if comparison.suspicious_password_field_present:
        return False
    if comparison.post_submit_left_trusted_domain is True:
        return False
    return True


def generate_verdict(
    ai_unknown: bool,
    comparison: ComparisonResult,
    features: FeatureResult,
) -> VerdictResult:
    """Return a deterministic verdict with plain-English reasons.

    When ``trusted_reference_found`` is false, we avoid strong phishing penalties that
    depend on reference alignment; only light structural heuristics apply—except that
    recognized brands on official roots without password-collection signals are not
    marked suspicious solely for missing a (brand, task) mapping.
    """
    reasons: list[str] = []
    score = 0

    trusted_benign = _trusted_brand_benign_landing(comparison, features)
    blocked_reason = (features.capture_block_reason or "").lower()
    blocked_evidence = (features.capture_block_evidence or "").strip()
    browser_blocked = "playwright" in blocked_reason or "navigation_aborted" in blocked_reason
    http_blocked = "http_" in blocked_reason
    http_tls_denied = "http_tls_access_denied" in blocked_reason
    is_known_first_party = _is_known_first_party_product_domain(features.final_url)

    if features.capture_blocked:
        if browser_blocked and http_blocked and not trusted_benign:
            score += 4
            reasons.append("The site blocked automated browser navigation.")
            if http_tls_denied:
                reasons.append("Direct HTTP retrieval also failed due to TLS/access denial.")
            else:
                reasons.append("Direct HTTP retrieval also failed during fallback access attempts.")
            reasons.append(
                "This combination is consistent with evasive phishing infrastructure."
            )
        else:
            score += 1 if trusted_benign else 3
            reasons.append(
                "Target site blocked automated browsing attempts, which is commonly associated "
                "with phishing or evasive infrastructure."
            )
        if blocked_evidence:
            reasons.append(f"Capture-block evidence: {blocked_evidence}")

    # Stronger lexical/non-visual scoring when visual capture artifacts are unavailable.
    if features.visual_capture_unavailable:
        if features.trusted_brand_root_mismatch:
            score += 2
            reasons.append(
                "The URL references a trusted-brand keyword but does not resolve to that brand's trusted root domain."
            )
        if features.brand_lookalike_signal:
            score += 2
            maybe = f" ({features.brand_lookalike_to})" if features.brand_lookalike_to else ""
            reasons.append(
                "The domain contains a lookalike brand token consistent with typosquatting."
                + maybe
            )
        if features.url_contains_auth_tokens:
            score += 1
            reasons.append(
                "The URL includes authentication-related tokens (login/verify/auth), which increases risk when page evidence is unavailable."
            )
    if features.url_first_party_plausibility == "suspicious_shape":
        score += 2
        reasons.append(
            "URL shape appears suspicious for the inferred brand/product (non-first-party structure)."
        )
    elif features.url_first_party_plausibility == "likely_first_party":
        score = max(0, score - 1)
        reasons.append(
            "URL structure looks plausible for a first-party brand/product surface (supporting legitimate signal)."
        )
    if features.redirect_count >= 3:
        score += 1
        reasons.append(
            "The URL underwent multiple redirects before settling, which can indicate obfuscation."
        )
    if features.cross_domain_redirect_count >= 2:
        score += 2
        reasons.append(
            "Redirect flow crossed multiple domains before settling, which increases risk."
        )
    if features.redirect_count >= 5:
        score += 1
        reasons.append(
            "Redirect chain is unusually long for a normal user flow."
        )
    if features.language_match is False:
        reasons.append(
            "Suspicious and reference pages appear to use different languages (weak signal only)."
        )
    if features.legit_reference_quality in {"error_page", "interstitial", "partial"}:
        reasons.append(
            f"Trusted reference quality is {features.legit_reference_quality}, so visual/text comparison is treated with lower confidence."
        )
        score = max(0, score - 1)
    if features.legit_reference_matches_intended_task is False:
        reasons.append(
            "The trusted reference did not appear to match the intended task, reducing comparison reliability."
        )
        score = max(0, score - 1)
    if features.legit_reference_match_tier not in {"exact_surface", "same_product", "unknown"}:
        reasons.append(
            "Comparison may not be apples-to-apples due to different product surface; confidence reduced."
        )
        score = max(0, score - 1)
        if is_known_first_party:
            score = max(0, score - 2)
            reasons.append(
                "Known first-party product domain detected; weak fallback mismatch alone is not treated as strong phishing evidence."
            )
    if features.url_shape_reasons:
        reasons.append(
            "URL-intel notes: " + "; ".join(features.url_shape_reasons[:3])
        )

    if ai_unknown:
        if not trusted_benign:
            score += 1
            reasons.append(
                "The classifier was uncertain about brand or task, so any match to a trusted site is less reliable."
            )
        else:
            reasons.append(
                "The classifier had some uncertainty, but the URL sits on a known official domain for the "
                "recognized brand without risky login signals, so that alone does not add risk."
            )

    if comparison.trusted_reference_found:
        reasons.append(
            "A trusted reference page exists for the guessed brand and task; we scored how well this page aligns with it."
        )

        # Strongest behavioral signal: post-submit leaves trusted host for an unknown / non-OAuth destination.
        if comparison.post_submit_left_trusted_domain is True:
            score += 3
            reasons.append(
                "After a dummy login submit, the page moved to a different host than the trusted reference—"
                "a strong sign of a deceptive flow."
            )

        # Domain vs trusted reference for this brand/task.
        if comparison.action_match_score < 1.0 and not (
            is_known_first_party
            and comparison.legit_reference_match_tier in {"same_brand_fallback", "weak_fallback"}
        ):
            score += 2
            reasons.append(
                "The page's domain does not match the trusted reference domain for this brand and task."
            )

        # HTML shape vs reference.
        dom_unavailable = any(
            x in (comparison.reasons or [])
            for x in (
                "dom_similarity_skipped_missing_suspicious_html",
                "dom_similarity_skipped_missing_legit_html",
            )
        )
        if comparison.dom_similarity_score < 0.4 and not dom_unavailable:
            score += 2
            reasons.append(
                "The page's HTML structure differs strongly from the trusted reference page."
            )

        # Password / login-shape alignment for login-like tasks (from ComparisonResult).
        if not comparison.task_aligned_with_legit_reference:
            score += 2
            reasons.append(
                "Login-related cues (such as password fields) do not line up with what we expect from the trusted reference."
            )

        # Aggregate gap from titles, text, form parity, and post-submit (computed in compare.py).
        bg = comparison.behavior_gap_score
        if bg >= 0.55:
            score += 2
            reasons.append(
                "Titles, visible text, and form behavior together diverge a lot from the legitimate reference."
            )
        elif bg >= 0.32:
            score += 1
            reasons.append(
                "There is a moderate mismatch versus the legitimate page in content and form-related signals."
            )

        # If the aggregate gap is still low, surface very low title/text explicitly (avoids missing edge cases).
        title_unavailable = any(
            x in (comparison.reasons or [])
            for x in (
                "title_comparison_unavailable_missing_one_side",
                "title_comparison_unavailable_missing_suspicious_side",
                "title_comparison_unavailable_missing_both_sides",
            )
        )
        text_unavailable = any(
            x in (comparison.reasons or [])
            for x in (
                "visible_text_comparison_unavailable_missing_one_side",
                "visible_text_comparison_unavailable_missing_suspicious_side",
                "visible_text_comparison_unavailable_missing_both_sides",
            )
        )
        if bg < 0.4:
            if comparison.title_similarity < 0.28 and not title_unavailable:
                score += 1
                reasons.append(
                    "The page title is very different from the trusted reference title."
                )
            if comparison.visible_text_similarity < 0.22 and not text_unavailable:
                score += 1
                reasons.append(
                    "The visible text on the page is very different from the trusted reference."
                )

        # Login-capable suspicious page on the wrong domain (clear narrative, small extra weight).
        if (
            comparison.suspicious_login_interaction_possible
            and comparison.action_match_score < 1.0
        ):
            score += 1
            reasons.append(
                "The page exposes what looks like a full sign-in form while the host does not match the trusted brand site."
            )
        elif (
            comparison.suspicious_password_field_present
            and comparison.action_match_score < 1.0
            and not comparison.suspicious_login_interaction_possible
        ):
            score += 1
            reasons.append(
                "A password field appears on a host that does not match the trusted reference domain."
            )

    else:
        if trusted_benign:
            reasons.append(
                "The page is on a domain we treat as an official root for the recognized brand, and we did not "
                "see password-collection fields in the capture or an off-domain redirect after a dummy submit. "
                "Missing a specific (brand, task) reference entry alone is not treated as suspicious here."
            )
        else:
            reasons.append(
                "No trusted reference was found for this brand and task, so we did not score alignment to a known-good page."
            )
            # Light heuristics only; do not treat missing reference as proof of phishing.
            if features.has_form:
                score += 1
                reasons.append(
                    "The page contains a form; without a reference we cannot compare it to an expected layout."
                )
            if features.external_link_ratio > 0.5:
                score += 1
                reasons.append(
                    "Many links point outside this site's domain, which deserves a closer look."
                )

    # Forms / links when we *do* have a reference: add weight only when domain already mismatches or structure is weak.
    if comparison.trusted_reference_found:
        if features.has_form and comparison.action_match_score < 1.0:
            # Avoid double-counting if we already added the login-form-on-wrong-domain reason.
            if not comparison.suspicious_login_interaction_possible:
                score += 1
                reasons.append(
                    "The page includes a form on a host that does not match the trusted reference."
                )
        if features.external_link_ratio > 0.5:
            score += 1
            reasons.append(
                "A high share of links goes off-domain, which adds risk in addition to reference comparison."
            )

    # Federated login: redirects to Google / Apple / Microsoft or OAuth-shaped URLs are normal.
    if comparison.trusted_oauth_redirect:
        score = max(0, score - 2)
        reasons.append(
            "Redirected to trusted OAuth provider (e.g., Google, Apple, or Microsoft) — expected behavior."
        )
    elif comparison.is_oauth_flow and comparison.post_submit_left_trusted_domain is not True:
        score = max(0, score - 1)
        reasons.append(
            "Post-submit URL shows OAuth-style patterns; cross-domain navigation weighted less heavily."
        )

    # Strong alignment + known IdP redirect → treat as likely legitimate (e.g. GitHub → Google sign-in).
    if (
        comparison.trusted_reference_found
        and comparison.trusted_oauth_redirect
        and comparison.post_submit_left_trusted_domain is not True
        and comparison.title_similarity >= 0.25
        and comparison.visible_text_similarity >= 0.2
        and comparison.dom_similarity_score >= 0.3
        and score <= 4
    ):
        return VerdictResult(verdict="likely_legit", confidence="high", reasons=reasons)

    # Thresholds: conservative; "likely_phishing" needs several strong signals.
    if score >= 7:
        return VerdictResult(verdict="likely_phishing", confidence="high", reasons=reasons)
    if score >= 4:
        return VerdictResult(verdict="likely_phishing", confidence="medium", reasons=reasons)
    if score >= 2:
        return VerdictResult(verdict="suspicious", confidence="medium", reasons=reasons)
    if score >= 1:
        return VerdictResult(verdict="suspicious", confidence="low", reasons=reasons)
    return VerdictResult(verdict="likely_legit", confidence="low", reasons=reasons)
