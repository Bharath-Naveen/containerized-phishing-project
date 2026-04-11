# Legitimate URL dataset design (Amazon, Google, Microsoft, PayPal)

> Sample `legit-*.txt` / `phish-*.txt` files for challenge workflows live in **`archive/sample_data/challenge/`** (copy into `data/raw/` locally; not committed under `data/raw/`).

## Goals

- Mirror **phishing feed coverage** with first-party URLs in the same brand ecosystems.
- Tag each URL with an **action / surface category** so we can audit balance (not only homepages).
- Keep **label = 0** for all rows in `data/raw/legit-*.txt` and **label = 1** for `phish-*.txt`.

## File format (`data/raw/legit-<brand>.txt`)

- One URL per line; lines starting with `#` are comments.
- Prefer tagged lines so enrichment reports can group by intent:

```text
[homepage] https://www.amazon.com/
[login_auth] https://www.amazon.com/ap/signin?...
```

- If a line has **no** `[category]` prefix, it is stored as `action_category=uncategorized`.

### Recommended categories (snake_case)

| Category | Use for |
|----------|---------|
| `homepage` | Main marketing / landing |
| `login_auth` | Sign-in, OAuth start, SSO |
| `account_dashboard` | Signed-in account hub, settings hub |
| `recovery_verification` | Password reset, verify identity, security check |
| `support_help` | Help, contact, safety, phishing reporting |
| `product_service` | Product detail, app hub, workspace product |
| `checkout_payment` | Cart, checkout, pay, wallet (where brand-owned) |
| `redirect_query_heavy` | Real URLs with long redirects or query strings |
| `regional_localized` | Country TLD or `/en-gb/`-style paths |
| `subdomain_product` | mail.*, drive.*, outlook.*, etc. |

## Per-brand notes

### Amazon (`legit-amazon.txt`)

- **login_auth** / **checkout_payment**: real `amazon.com` / `amazon.co.uk` sign-in and cart flows.
- **regional_localized**: `amazon.de`, `amazon.co.jp`, etc.
- **support_help**: customer service / help hubs.

### Google (`legit-google.txt`)

- **subdomain_product**: `accounts.`, `mail.`, `drive.`, `photos.`, `myaccount.`.
- **login_auth**: Accounts sign-in with real query params.
- **redirect_query_heavy**: OAuth-style `continue=` URLs on `accounts.google.com`.

### Microsoft (`legit-microsoft.txt`)

- **login_auth**: `login.microsoftonline.com`, `login.live.com`, `www.office.com/login`.
- **subdomain_product**: `outlook.office.com`, `www.onedrive.com`, SharePoint-style paths on first-party hosts.
- **account_dashboard**: Microsoft account / security pages.

### PayPal (`legit-paypal.txt`)

- **login_auth**: `paypal.com/signin`, `www.paypal.com/signin`.
- **checkout_payment**: checkout / pay flows on `paypal.com`.
- **support_help**: resolution center / smarthelp / security.

## Quality checks (automated)

Run after ingest / split:

```bash
python -m src.pipeline.dataset_report
```

Review **legit share of `login_auth` vs `homepage`** on the training split and **category entropy** vs phishing.
