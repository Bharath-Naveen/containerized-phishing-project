# Audit baseline (before the rebuild)

This folder is the 2026-09-29 audit of the original code, kept as the "before" record. It is also
tagged in git as `audit-baseline-2026-09-29`; check out that tag to re-run these scripts against the
code they measured. On later commits the scripts import the rebuilt `phishguard` package, so their
output would describe the new code, not the old one.

Do not quote these numbers for the current system: they include a leaky split (55% of the shipped
models' test rows shared a domain with training), the http/https dataset artifact, and a calibrator
fit for a different model. Current numbers: `../VERIFIED_METRICS.md`.
