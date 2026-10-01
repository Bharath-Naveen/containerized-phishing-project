# Rule tuning plan (pre-registered 2026-09-30, before any tuning)

Why: the second live run (see REBUILD_LOG) showed the full system calls 25 of 233 legitimate pages phishing (mostly two hard blockers firing on real sign-in flows) and catches only 22 of 60 fresh phishing URLs, calling 25 of them legitimate. This file fixes the rules of the tuning pass **before** any rule is changed, so the result cannot be steered by looking at the test data.

## 1. Frozen data (one snapshot, used for every before/after number)

Live pages change by the hour, so tuning on live runs would compare different pages. Instead one GitHub Actions run captures every URL once and saves the capture record and page HTML (`phishguard snapshot-live`, workflow `.github/workflows/live-snapshot.yml`). Analysis needs nothing else: once it has the capture it makes no network calls, so replaying a snapshot gives the same verdict every time for the same code.

| Set | Size | Split |
|---|---|---|
| Official brand URLs | 298 | existing `val` (162) / `test` (136) by registered domain |
| Fresh phishing (PhishStats, OpenPhish if empty) | up to 240, one URL per host | `val` / `test` by sha256(host), even = val |
| Popular homepages (pinned Tranco list, resolving only) | 160 | `val` / `test` by sha256(registered domain) |
| Hard legit, curated suites, edge cases | 48 | `val` only (they were already seen while the rules were written) |

## 2. Label hygiene (decided now, applied mechanically)

A feed entry is "reported as phishing", not proof. For the fresh-phishing set, before scoring:

- capture failed: excluded (a verdict on a dead page is not a detection);
- the submitted URL's host (without `www.`) is itself a Tranco top-10,000 entry (a real site's own page, for example `www.att.com`, `yandex.ru`): excluded from the main number and reported separately. Subdomains of big platforms (`x.godaddysites.com`, `y.sharepoint.com`) stay in: that is where much real phishing lives.

Nothing else is removed, even if a page looks harmless after the fact.

Amendment (2026-09-30, before any snapshot was captured): the first draft grouped fresh phishing by registered domain and excluded any URL on a top-10K registered domain. A dry run of URL selection (no captures) showed that would drop or collapse every page on godaddysites.com, sharepoint.com and similar platforms, so grouping and exclusion now use the host.

## 3. What may change

- Allowed: rule logic and thresholds in `app/eal.py`, `app/verdict_rules.py`, `app/verdict_policy.py`, and the hard-blocker detection in `app/capture_signals.py` / `app/html_*_signals.py`.
- Not allowed: adding any domain or brand to a registry or allowlist (that would just memorize the evaluation sites); retraining or changing the Layer 1 model; editing the frozen snapshot.

## 4. Decision rule

Measured on the `val` split only:

- **legit false alarms** = `likely_phishing` on official brand val + popular homepages val + hard legit + curated legit;
- **phishing recall** = `likely_phishing` on fresh phishing val (after section 2).

A change set is kept only if val phishing recall does not go down **and** val legit false alarms do not go up, and at least one of them improves. Among kept candidates, the one with the most improvement in (recall gain + false-alarm reduction, both as rates) wins; ties go to the smaller code change.

## 5. Test split

`phishguard replay --split test` is run **once**, after the final rules are chosen, and every run of it is appended to `metrics/results/test_access_log.jsonl` (time, commit). Before and after numbers on the test split are reported as they come out, with counts, whether they improve or not.

## 6. What gets reported

Before/after table on val and on test, with counts. Golden dashboard tests are re-recorded only for the intended rule changes, and the diff of changed golden cases is listed in the log.
