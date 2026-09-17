# Build story: Evidence Adjudication Layer (EAL)

## What this system is (one paragraph for interviews)

This is a **containerized phishing detection dashboard** that does not treat a single ML score as the verdict. Layer 1 is URL/host ML (primary Random Forest trained at ~500k scale, plus witness models for **model agreement**). Layer 2 is **Playwright** live capture (redirects, TLS, network requests). Layer 3 is HTML/DOM structure, host/path reasoning, platform context, JS/network behavior, brand-domain coherence, and a small **official-domain trust-prior** registry—not a whitelist. The **Evidence Adjudication Layer (EAL)** in `analyze_dashboard.py` aggregates evidence into phishing, legitimacy, and ambiguity signals, applies **hard blockers**, then picks a 3-way label: `likely_phishing`, `uncertain`, or `likely_legitimate`. Runtime **AI adjudication was removed**; deployment is deterministic and test-driven.

The commits below are not isolated diffs—they are the visible peaks of a longer tuning arc after deploying a stronger primary model and adding model agreement, JS/network signals, and false-positive fixes on real URLs.

---

## Timeline context (git order)

From `.git/logs/HEAD`, the five commits appear in this order:

1. `966fa8f` — Stable version before Evidence Adjudication Layer refactor
2. `6989107` — Refine EAL precedence and restore platform context cache
3. `0cb7b95` — Handle security block pages in evidence adjudication
4. `8270452` — Use brand coherence to demote clean wrapper patterns
5. `4bfca61` — Fix official authwall EAL suppression

There is also a local branch `pre-eal-stable` pointing at the same snapshot as the pre-rebase stable commit—useful as a comparison point when EAL behavior regressed during rapid patches.

---

## Commit stories (interview depth)

### `966fa8f` — Stable version before Evidence Adjudication Layer refactor

**What was going on**

By this point the project had already moved from optional AI adjudication to a **deterministic EAL** with categories like hard blockers, capture-failure handling ("absence of phishing evidence is not safety"), and deployment-oriented cleanup. The team had also **deployed a 500k-trained Random Forest** as `layer1_primary` and wired **multi-model agreement** into the dashboard JSON.

**Why this commit exists**

The message is a **checkpoint**, not "EAL didn't exist yet." It marks a line in the sand **before the next wave of EAL precedence work**: authwall false positives, free-hosted clone under-conviction, platform-context tests, JS/network integration, security-vendor block pages, brand-context tuning, creator platforms (Podia), brand-domain coherence, wrapper demotion, and LinkedIn regressions. A `pre-eal-stable` branch was created so you could diff or roll back if stacked changes made the adjudication graph too hard to reason about.

**How you'd diagnose problems at this stage**

Failures were mostly **precedence** and **context**, not missing features: ML and agreement were strong, but **downstream rules disagreed** (wrapper patterns, platform labels, legitimacy from empty HTML). The stable tag let you say: "Does this bug exist before or after the tuning sprint?"

**Why not a different fix**

Splitting EAL into its own module was discussed in spirit (large `analyze_dashboard.py`), but the priority was **correct verdicts under a new primary model**, not a structural refactor—hence a git checkpoint instead of a big file move.

---

### `6989107` — Refine EAL precedence and restore platform context cache

**What was broken**

Two **production-shaped** failures showed up right after the stronger RF + agreement deployment:

1. **LinkedIn official authwall / profile** (`linkedin.com`, same registrable domain, same-domain forms, `host_identity_class = official_brand_auth`, ML **strong_legitimate**): EAL still produced **`likely_phishing`** because `wrapper_or_interstitial_redirect_pattern` was a **hard blocker** with no exception for first-party auth flows.

2. **GitHub Pages Amazon clone** (`*.github.io`, strong brand mismatch, **strong_phishing** consensus): verdict stayed **`uncertain`** because **content-rich / low DOM risk / no credential capture** legitimacy signals were allowed to **outweigh** free-hosted brand impersonation when nothing looked like a classic harvester page.

Separately, **`test_hosting_domain_trust_layer.py`** started failing: `platform_context_type = unknown` for known user-hosting suffixes (`vercel.app`, `github.io`, `netlify.app`, `framer.app`, `weebly.com`).

**How you diagnosed it**

- LinkedIn: inspect `evidence_hard_blockers` vs capture/DOM flags—wrapper/interstitial fired on **official** hosts despite clean forms and high host legitimacy.
- GitHub clone: inspect evidence lists—phishing signals were present but **legitimacy rescue from "clean looking" pages** dominated.
- Platform tests: trace `_load_platform_domain_registry()`—a **global cache** could be filled once with `[]` (missing/wrong path in an earlier test) and then **every later lookup returned empty**, so classification fell through to `unknown`.

**Why you chose that fix**

- **Authwall**: Treat official first-party wrapper/interstitial as **ambiguity** (`official_authwall_wrapper_pattern`) when same domain, no brand mismatch, no cross-domain credential forms, strong legitimate consensus or moderate ML, and high host legitimacy—**not** a solo hard blocker.
- **Free-hosted clone**: Add an explicit escalation when **free_hosted_brand_impersonation + strong_brand_domain_mismatch + ml_consensus_strong_phishing**; **suppress** `content_rich_page` / `low_html_dom_risk` legitimacy in impersonation-hosting contexts.
- **Platform cache**: Key cache by **resolved CSV path** (`_PLATFORM_DOMAIN_REGISTRY_CACHE_PATH`) so pytest order and alternate configs cannot poison runtime classification.

**Design principle**

Hard blockers should mean **corroborated abuse**, not "this big company uses interstitials." Free-hosted **brand theater** can look like a marketing page and still be staging/phishing.

---

### `0cb7b95` — Handle security block pages in evidence adjudication

**What was broken**

After **JS/network monitoring** in capture, a PayPal-related URL **redirected to a Lionic security block page** (`block.cloud.lionic.com/.../malicious.html`). Playwright faithfully captured **the block page's HTML**, not the original target. EAL then credited:

- `no_credential_capture`
- `no_cross_domain_forms`
- `content_rich_page`
- `low_html_dom_risk`

...while ML/model agreement and brand mismatch on the **input URL story** still said phishing. Result: **`likely_legitimate`** on a case that should stay **`likely_phishing`**.

This is the same **"absence is not innocence"** class as capture/HTML unavailable (Netflix-on-`2ndstage.app`-style cases), but the failure mode was **misleading positive HTML** from a third-party vendor page, not empty DOM.

**How you diagnosed it**

Compare `final_url` / title / visible text to input registrable domain: **redirect_domain_mismatch**, strong ML, strong brand mismatch, but structural signals described **Lionic's warning page**. Legitimacy signals were tied to **final document**, not **investigation context**.

**Why you chose that fix**

1. **`_detect_security_block_page()`** with vendor signatures (Lionic URL fragments and warning copy).
2. Emit `security_block_page_detected`, vendor, and matched reasons on layer-2 capture.
3. In EAL: **do not** award absence-based legitimacy when a block page is detected; add `security_block_page_observed` and `security_vendor_blocked_as_malicious`.
4. **Escalate** to `likely_phishing` when block page + strong ML consensus / high ML / strong brand mismatch; **never** `likely_legitimate` on block pages—at least **`uncertain`**.

**Why not other approaches**

- Whitelisting Lionic globally would break "final page is not the brand" logic elsewhere.
- Ignoring all redirects would lose real phishing redirect chains.
- The fix is **page-type detection** plus **turning off bogus legitimacy credit**, while keeping input-side brand/ML suspicion.

---

### `8270452` — Use brand coherence to demote clean wrapper patterns

**What was broken**

**Brand-domain coherence** (title/H1 vs registrable domain tokens—e.g. Virgin Atlantic, Coursera) was working: high coherence, legitimacy signals, and **`ml_brand_coherence_disagreement`** could cap disagreement at **`uncertain`**. But **`wrapper_or_interstitial_redirect_pattern` remained a hard blocker** when DOM still flagged interstitial/wrapper behavior **without** rich nav/footer counts.

Virgin Atlantic (`virginatlantic.com/en-US`) is a concrete case: coherent brand, same domain, no credentials, no exfiltration, high legitimacy score—yet **`likely_phishing`** because wrapper trumped coherence.

**How you diagnosed it**

Evidence showed **coherence match + wrapper hard blocker** at the same time. Wrapper demotion already existed for **rich nav/footer** or **official_domain_trust_prior**; Virgin-like pages could fail the nav/footer threshold while still being obviously the real airline site in title vs domain.

**Why you chose that fix**

Extend the **clean wrapper demotion gate** in both `_compute_phishing_blockers` and `_apply_evidence_adjudication_layer` so **`brand_domain_coherence_match`** (via `coherent_brand_host_identity_candidate`) counts as a trust anchor alongside rich nav or official prior. When the gate passes: no wrapper hard blocker; ambiguity signals like `coherent_brand_wrapper_pattern` / `official_content_wrapper_pattern`; verdict capped away from phishing.

**Safety**

Gates still require same registrable domain, weak/no brand mismatch, no credential capture, no cross-domain forms, no exfiltration, no security block page, not free-hosted/user-hosted impersonation contexts—so **`paypal-login.vercel.app`** does not get coherence rescue (`test_paypal_vercel_clone_does_not_get_coherence_rescue`).

---

### `4bfca61` — Fix official authwall EAL suppression

**What was broken (regression)**

After coherence, wrapper demotion, **ML overconfidence relaxation** for clean official domains, and richer enrichment rules, **LinkedIn official authwall/profile** broke again:

- Same domain, official auth host identity, same-domain forms, **brand_domain_coherence_match = true**, ML **strong_legitimate**
- EAL still added **`strong_brand_domain_mismatch`**, **`auth_context_on_non_official_domain`**, and **wrapper hard blocker** -> **`likely_phishing`**

**How you diagnosed it**

Compared to the earlier `6989107` authwall fix: **`wrapper_official_authwall_safe`** was **narrow** (e.g. ML not high + high host confidence). New upstream rules flagged **OAuth provider names** ("Sign in with Google") as brand mismatch on official login surfaces. **`auth_context_on_non_official_domain`** fired on login page family even on **`linkedin.com`**. Wrapper demotion paths did not all share one **official first-party auth** definition.

**Why you chose that fix**

Introduce a unified **`official_auth_same_domain_safe`** gate used in EAL (and aligned blocker paths):

- Same registrable domain
- Official host/auth context (`official_brand_auth`, apex/family helpers)
- Same-domain forms, no password external action, no exfiltration, no security block
- Not free-hosted / user-hosted / cloud impersonation
- **Brand coherence or official domain family**

When true:

1. Suppress **`strong_brand_domain_mismatch`** in EAL scoring
2. Skip **`auth_context_on_non_official_domain`** (see guard around `official_auth_same_domain_safe` in `_apply_evidence_adjudication_layer`)
3. Do **not** add wrapper as hard blocker; use **`official_authwall_wrapper_pattern`** ambiguity
4. Credit **`first_party_auth_flow_consistency`**

Upstream in `_enrich_capture_and_html_signals`: for same-domain **official platform login** candidates, **OAuth mentions alone** must not set `brand_domain_mismatch` without host/path/form-action impersonation evidence.

**Why a second fix instead of reverting coherence**

Coherence and Virgin/Framer fixes were still needed; the bug was **incomplete suppression of auth-surface false positives**, not coherence itself. One consolidated gate scales to LinkedIn, Handshake-style logins, and Amazon official auth expectations without reopening Vercel clones.

---

## How the pieces fit together (mental model)

Think of EAL as a **scoring court** with **veto players**:

| Layer | Role |
|--------|------|
| Hard blockers | Instant phishing if corroborated (cross-domain credentials, exfiltration + forms, cloud-hosted impersonation, etc.) |
| `no_phishing_evidence_guard` | Post-blend guard: if no red flags, EAL must not return phishing without hard blockers |
| Security block page | "Wrong document" — strip fake cleanliness from vendor HTML |
| Free-host + strong mismatch + ML consensus | Escalate despite pretty pages |
| Brand coherence + official prior | Push high-ML false positives toward uncertain/legitimate on **same-domain** clean pages |
| Official auth same-domain safe | Stop authwalls/OAuth from looking like impersonation |

Precedence order in `_apply_evidence_adjudication_layer` (e.g. `no_phishing_guard` before generic high-ML paths) was refined repeatedly because **each new legitimacy path accidentally reintroduced an old failure mode**.

---

## Oral history: discussed but not fully documented

These come from prior project chat sessions and code review; many are **not** in `docs/DEPLOYMENT_NOTES.md` (which still focuses mainly on **legitimacy rescue**, trusted CSV, and blockers—not the full EAL graph).

### Edge cases considered and rejected

- **Ignoring brand mismatch on any content-rich page** — rejected; only **weak/incidental** mismatch when terms appear in ads, footers, or resources (`idtech.com/courses`).
- **Hard whitelist from `official_domains.json`** — rejected; **trust-prior** and caps only, never forced legitimate.
- **Treating every 404 as phishing** — rejected; creator platforms (Podia) get **`platform_404_or_inactive`** and **`uncertain`** when a strict safety checklist passes, not **`likely_phishing` from ML alone**.
- **Treating security block pages as automatic phishing** without ML/brand context — partially rejected; weak ML still floors at **`uncertain`**, not legitimate.
- **Using AI/LLM for brand coherence** — rejected; deterministic token/title matching only.

### Approaches tried and abandoned

- **AI adjudication as verdict driver** — removed from runtime; optional AI was phased out entirely for deployment.
- **Global platform registry cache without path key** — caused flaky `unknown` platform context; fixed in `6989107`.
- **Narrow authwall exception (`wrapper_official_authwall_safe` only)** — insufficient after enrichment grew; replaced by **`official_auth_same_domain_safe`** in `4bfca61`.
- **Demoting wrappers only with rich nav/footer** — insufficient for Virgin; extended with coherence in `8270452`.
- **Separate EAL module refactor** — deferred in favor of regression tests and precedence fixes (`966fa8f` checkpoint instead).

### Known gaps / stale docs

- **`docs/DEPLOYMENT_NOTES.md`** does not document: security block pages, brand coherence, creator-platform 404 rules, model agreement, JS/network behavior, EAL precedence vs `no_phishing_evidence_guard`, or official-auth suppression—worth extending beyond legitimacy rescue.
- **`docs/CHANGELOG_PROJECT_EVOLUTION.md`** still describes Phase 4 "bounded AI adjudication" as current-adjacent; runtime has moved on.
- **`docs/TESTING.md`** still mentions `tests/test_ai_adjudicator.py` as optional—AI is not part of the active deployment story.
- **Security block vendor list** is intentionally small (Lionic-first); extending to other SWG vendors was implied but not catalogued in docs.
- **`_load_trusted_domain_registry`** may use similar global caching patterns as platform registry—platform path was fixed; trusted registry wasn't part of the same commit message (verify if tests ever flake).
- **Poster / eval artifacts**: full ROC curve point data was **MISSING** from saved outputs (`outputs/poster_visuals/MISSING_DATA_NOTES.txt`); metrics bars use `metrics_summary.csv`, not fabricated curves.
- **Logistic regression feature names** in coefficient exports marked **MISSING** for human-readable poster importances.
- **`network_request_urls` dropped in `build_dashboard_analysis`** caused `network_request_domain_count == 0` until wired through—easy regression when adding new capture fields.
- **ML overconfidence relaxation** (`ml_overconfidence_relaxed_due_to_strong_legitimacy`) deliberately **does not apply to login/auth pages**—only content-like official pages; not spelled out in DEPLOYMENT_NOTES.
- **Click-probe / form-submit exfiltration** paths depend on capture configuration—not all runs enable credential-like submit probes.

### Regression URLs worth naming in interviews

- LinkedIn authwall/profile — official auth suppression
- `muhammadbilal0011.github.io` Amazon clone — free-host escalation
- PayPal -> Lionic block page — security block handling
- `www.virginatlantic.com/en-US` — coherence + wrapper demotion
- `www.idtech.com/courses` — incidental brand terms
- `unifiedmentor.podia.com/sessions` — creator platform 404
- `www.framer.com/?utm_source=microsoft...` — official platform + `no_phishing_evidence_guard` precedence
- `paypal-login.vercel.app` / Netflix Vercel clones — must stay phishing

---

## Suggested interview closing line

"We built a deterministic evidence court on top of ML because URL models over-fire on modern JS-heavy sites and under-explain free-hosted impersonation. Each of these commits fixes a **class** of mistake—wrong document (block page), wrong host class (authwall vs phish), wrong precedence (wrapper vs coherence), or poisoned global state (platform cache)—and we locked them in with URL-specific regression tests so deployment tuning didn't become whack-a-mole."
