# Deployment Notes

## Legitimacy Rescue Layer

The legitimacy rescue layer is a generalized false-positive reduction guard that runs **after ML scoring and before final verdict**.

Why it exists:

- some legitimate login/account flows can look suspicious to URL/host ML (long URLs, app subdomains, auth-like paths),
- rescue reduces overconfident phishing outcomes when domain/form/HTML evidence is internally consistent and low-risk.

What it does:

- can downgrade `likely_phishing` to a safer review state (default target: `uncertain`),
- caps effective ML influence after rescue,
- writes transparent rescue reasons and blocker signals to verdict/dashboard output.

## Configuration

Runtime flags (via environment / `PipelineConfig`):

- `legitimacy_rescue_enabled`
- `legitimacy_rescue_max_html_structure_risk`
- `legitimacy_rescue_max_html_dom_anomaly_risk`
- `legitimacy_rescue_ml_cap_after_rescue`
- `legitimacy_rescue_target_verdict`
- `trusted_domains_csv_path`

## Trusted Domain Registry

Default path:

- `data/reference/trusted_domains.csv`

Schema:

- `registered_domain`
- `allowed_hosts` (`|`-delimited)
- `organization`
- `notes`

Example row:

- `joinhandshake.com,app.joinhandshake.com|www.joinhandshake.com|app.joinhandshake.co.uk,Handshake,University career platform`

## Safety Behavior

Important:

- trusted-domain registry **does not force** a legitimate verdict.
- it only supports downgrade to `uncertain`/review when strong phishing blockers are absent.

Rescue blockers (downgrade disabled when present):

- cross-domain credential form action
- brand-domain mismatch
- free-hosting impersonation
- punycode/non-ASCII host
- sparse credential harvester pattern
- wrapper/interstitial redirect pattern
- strong anchor/domain mismatch
