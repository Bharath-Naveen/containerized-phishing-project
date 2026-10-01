/*
 * "Try any URL" box for the phishing detection demo (bharathnaveen.com).
 * Loads the model files on first use and runs everything in this browser tab.
 * The URL you type is never sent anywhere: no request below contains it.
 */
(function () {
  "use strict";
  var root = document.getElementById("pd-try");
  if (!root) return;
  var BASE = root.getAttribute("data-assets") || "../assets/phishing-demo/";
  var form = root.querySelector("form"), input = root.querySelector("input"), out = root.querySelector(".pd-out");
  var status = root.querySelector(".pd-status"), btn = root.querySelector("button[type=submit]");
  var engine = null, loading = null;

  var FEATURE_LABELS = {
    url_length: "URL length", hostname_length: "Hostname length", path_length: "Path length", query_length: "Query length",
    num_dots: "Dots in the hostname", num_hyphens: "Hyphens in the URL", num_digits: "Digits in the URL",
    num_special_chars: "Unusual characters in the hostname", subdomain_count: "Subdomain depth", has_ip_address: "Host is an IP address",
    has_at_symbol: "“@” in the URL", hostname_entropy: "Hostname randomness (entropy)", path_segment_count: "Path depth",
    path_shallow_le1: "Path at most 1 level deep", path_shallow_le2: "Path at most 2 levels deep",
    suspicious_redirect_query_flag: "Redirect-style query parameter", suspicious_keyword_count: "Lure words (verify, secure, update, ...)",
    kw_login: "“login” words in path or query", kw_verify: "“verify” words in path or query", kw_secure: "“secure” words in path or query",
    kw_update: "“update” words in path or query", kw_account: "“account” words in path or query", kw_payment: "“payment” words in path or query",
    kw_confirm: "“confirm” words in path or query", kw_password: "“password” words in path or query",
    path_query_authish_keyword_hits: "Sign-in style words in path or query", no_authish_path_query_tokens: "No sign-in words in path or query",
    port_present: "Explicit port number", punycode_flag: "Punycode (xn--) hostname", free_hosting_flag: "Free hosting domain",
    cloud_hosting_flag: "Cloud hosting domain", private_host_flag: "Private or local address",
    official_registrable_anchor: "Known official brand domain", official_domain_family: "Brand-family domain name",
    brand_hostname_exact_label_match: "Brand name as its own host label", brand_hyphenated_deception_label: "Brand name with hyphenated extras (brand-login-...)",
    brand_typosquat_embedded_in_label: "Brand name glued to other text", path_brand_token_present: "Brand name in the path",
    path_brand_without_official_host: "Brand name in the path of a non-brand host", brand_on_free_hosting: "Brand cues on free hosting",
    brand_on_cloud_placeholder: "Brand cues on cloud hosting", host_brand_substring_not_official: "Brand text in a host that is not the brand's domain",
    num_brand_tokens_in_host: "Brand names in the host", num_brand_tokens_in_path: "Brand names in the path",
    domain_hash_bucket: "Domain hash bucket (a learned ID, not a readable signal)", layer1_brand_trust_score: "Official-brand trust score",
    legit_auth_surface_on_official_anchor: "Sign-in path on an official brand domain", legit_admin_like_path_on_official_anchor: "Admin path on an official brand domain",
    legit_checkout_like_path_on_official_anchor: "Checkout path on an official brand domain", simple_public_web_shape: "Simple public page shape",
    simple_official_homepage_shape: "Official brand homepage shape", dns_features_skipped: "DNS lookups skipped (always)",
    url_features_missing: "URL features missing", hosting_features_missing: "No parsable host",
  };
  var SIGNAL_LABELS = {
    high_ml_score: "URL model score is 0.85 or higher", elevated_ml_score: "URL model score is between 0.70 and 0.85",
    moderate_ml_score: "URL model score is between 0.50 and 0.70", ml_consensus_strong_phishing: "At least 3 of the 4 models vote phishing",
    ml_consensus_strong_legitimate: "At least 3 of the 4 models vote legitimate", ml_consensus_split: "The 4 models split their votes",
    boosted_models_only_flag_phishing: "Only the boosted models (XGBoost, LightGBM) vote phishing",
    high_model_probability_spread: "The models disagree by more than 0.4", suspicious_host_pattern: "The host itself looks suspicious",
    high_risk_missing_evidence: "High URL risk on a low-trust host, with no page to check",
    behavior_analysis_unavailable: "No page was loaded, so no page behavior", html_dom_unavailable: "No page was loaded, so no HTML evidence",
    ml_structural_disagreement: "URL and other evidence disagree",
  };
  var HOST_REASONS = {
    userinfo_in_netloc: "text with “@” before the host", punycode_label_present: "a punycode (xn--) label",
    numeric_heavy_hostname: "5 or more digits in a row", double_hyphen_subdomain: "a double hyphen in a subdomain",
    very_deep_subdomain_chain: "4 or more dots in the host", encoded_netloc_sequence: "% encoding in the host",
    ip_literal_host: "an IP address instead of a name", brand_token_in_subdomain_not_matching_registrable: "a brand name in a subdomain of another domain",
    brand_token_in_registrable_not_official: "a brand name inside a non-brand domain name",
    credential_lure_tokens_in_registrable_label: "two or more lure words (login, secure, account, ...) in the domain name",
  };
  var HOST_CLASS = {
    suspicious_host_pattern: "Suspicious host pattern", official_brand_auth: "Known official brand domain",
    public_docs_or_reference: "Documentation-style path", generic_unknown_host: "Ordinary host, nothing known about it",
  };
  var VERDICT_TEXT = {
    likely_phishing: "Likely phishing", uncertain: "Uncertain: needs the page to decide", likely_legitimate: "Likely legitimate",
    error: "No verdict: the Python app stops on this URL",
  };

  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function fmt(x) { return (Math.round(x * 1000) / 1000).toFixed(3); }

  function fetchBytes(url) {
    return fetch(url).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status + " for " + url); return r.arrayBuffer(); });
  }
  function gunzipIfNeeded(buf) {
    var b = new Uint8Array(buf);
    if (b.length < 2 || b[0] !== 0x1f || b[1] !== 0x8b) return Promise.resolve(buf); // server already decoded it
    if (typeof DecompressionStream === "undefined") return Promise.reject(new Error("This browser cannot unpack the Random Forest file."));
    var stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).arrayBuffer();
  }
  function load() {
    if (loading) return loading;
    status.textContent = "Loading the models (first use only)...";
    loading = Promise.all([
      fetch(BASE + "model-core.json").then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }),
      fetchBytes(BASE + "model-rf.bin.gz").then(gunzipIfNeeded),
    ]).then(function (res) {
      engine = window.PhishGuardEngine.createEngine(res[0]);
      engine.setRandomForest(res[1]);
      status.textContent = "Models loaded. Everything runs in this tab.";
    }).catch(function (e) {
      loading = null;
      status.textContent = "Could not load the models: " + e.message;
      throw e;
    });
    return loading;
  }

  function render(a) {
    out.innerHTML = "";
    var v = a.verdict, s = a.score;
    var head = el("div", "pd-verdict pd-" + v.label.replace("likely_", ""));
    head.appendChild(el("span", "pd-vlabel", VERDICT_TEXT[v.label] || v.label));
    head.appendChild(el("span", "pd-vmode", "ML-only verdict (URL text only, no page visited)"));
    out.appendChild(head);

    if (v.label === "error") {
      out.appendChild(el("p", "pd-note", "The dashboard's own code raises a URL parsing error on this input (" + v.error + "), so it gives no verdict. The demo shows the model score anyway."));
    }

    var score = el("div", "pd-score");
    var lab = el("div", "pd-score-top");
    lab.appendChild(el("span", "", "URL model (XGBoost), calibrated probability of phishing"));
    lab.appendChild(el("b", "", s.p_cal_rounded.toFixed(6)));
    score.appendChild(lab);
    var bar = el("div", "pd-bar"), fill = el("i", "");
    fill.style.width = Math.max(1, Math.min(100, s.p_cal * 100)) + "%";
    bar.appendChild(fill);
    score.appendChild(bar);
    var votes = el("p", "pd-votes");
    votes.textContent = "4-model vote: " + s.agreement.votes_phishing + " of 4 say phishing (" +
      s.witnesses.map(function (w) { return { logistic_regression: "Logistic Regression", random_forest: "Random Forest", xgboost: "XGBoost", lightgbm: "LightGBM" }[w.model_name] + " " + w.phish_probability.toFixed(6); }).join(", ") + "). Scores are shown as the app reports them, rounded to 6 decimals.";
    score.appendChild(votes);
    out.appendChild(score);

    var grid = el("div", "pd-grid");
    // Top contributing URL features
    var c1 = el("div", "pd-card");
    c1.appendChild(el("h4", "", "What moved the URL score"));
    var feats = engine.core.features, rows = [];
    for (var i = 0; i < feats.length; i++) rows.push({ k: feats[i], c: s.contribs[i], v: a.row[feats[i]] });
    rows.sort(function (x, y) { return Math.abs(y.c) - Math.abs(x.c); });
    var ul = el("ul", "pd-contrib");
    rows.slice(0, 6).forEach(function (r) {
      var li = el("li", r.c > 0 ? "up" : "down");
      li.appendChild(el("span", "pd-arrow", r.c > 0 ? "↑" : "↓"));
      var t = el("span", "pd-ctext", FEATURE_LABELS[r.k] || r.k);
      li.appendChild(t);
      li.appendChild(el("span", "pd-cval", "value " + (Number.isInteger(r.v) ? r.v : fmt(r.v))));
      ul.appendChild(li);
    });
    c1.appendChild(ul);
    c1.appendChild(el("p", "pd-fine", "↑ pushes toward phishing, ↓ toward legitimate. Per-feature contributions to the XGBoost log-odds (the path attribution XGBoost reports as approx_contribs), largest first."));
    grid.appendChild(c1);

    // The judge
    var c2 = el("div", "pd-card");
    c2.appendChild(el("h4", "", "Why the judge decided this"));
    var hp = a.host_path;
    var hostLine = el("p", "pd-host");
    if (hp) {
      hostLine.textContent = "Host check: " + (HOST_CLASS[hp.host_identity_class] || hp.host_identity_class) + " (" + hp.host_legitimacy_confidence + " trust)";
      if (hp.host_reasons.length) hostLine.textContent += ": " + hp.host_reasons.map(function (r) { return HOST_REASONS[r] || r; }).join("; ");
    } else hostLine.textContent = "Host check: the URL could not be parsed.";
    c2.appendChild(hostLine);
    var sl = el("ul", "pd-signals");
    v.phishing_signals.forEach(function (sig) { sl.appendChild(el("li", "ph", SIGNAL_LABELS[sig] || sig)); });
    v.legitimacy_signals.forEach(function (sig) { sl.appendChild(el("li", "lg", SIGNAL_LABELS[sig] || sig)); });
    v.ambiguity_signals.forEach(function (sig) { if (sig !== "behavior_analysis_unavailable") sl.appendChild(el("li", "am", SIGNAL_LABELS[sig] || sig)); });
    c2.appendChild(sl);
    var why = {
      missing_evidence_high_risk: "With no page to look at, a very high URL score (or a 3 of 4 phishing vote) on a low-trust host is enough for a phishing verdict.",
      phishing_score_threshold: "The phishing signals add up to at least 0.70 (here " + v.phishing_score + "), so the verdict is phishing.",
      default_uncertain: "Without the page, the judge does not treat missing evidence as proof of safety. Anything short of strong phishing evidence stays uncertain, so ML-only mode never says “likely legitimate”.",
      app_raises_on_url: "",
    }[v.rule];
    if (why) c2.appendChild(el("p", "pd-fine", why));
    grid.appendChild(c2);
    out.appendChild(grid);
    out.appendChild(el("p", "pd-note", "This is the URL model and the rules that can run without visiting the page. The full system also loads the page (redirects, forms, TLS, scripts) before it decides; the replays below show that part on real pages."));
  }

  function run() {
    var url = (input.value || "").trim();
    if (!url) { status.textContent = "Type or paste a URL first."; return; }
    if (url.length > 2048) { status.textContent = "That is longer than 2,048 characters; please shorten it."; return; }
    btn.disabled = true;
    load().then(function () {
      var t0 = performance.now();
      var a = engine.analyze(url);
      render(a);
      status.textContent = "Analyzed in " + Math.max(1, Math.round(performance.now() - t0)) + " ms, in this tab. Nothing was sent anywhere.";
    }).catch(function () { /* status already says why */ }).then(function () { btn.disabled = false; });
  }
  form.addEventListener("submit", function (e) { e.preventDefault(); run(); });
  root.querySelectorAll("[data-example]").forEach(function (b) {
    b.addEventListener("click", function () { input.value = b.getAttribute("data-example"); run(); });
  });
  input.addEventListener("focus", function () { load().catch(function () {}); }, { once: true });
})();
