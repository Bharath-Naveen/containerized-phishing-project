"""Build the fixed URL set for the JS/Python parity test (rebuild Phase 6).

Sets (each row: id, set, url):
  heldout_test     every row of the verified held-out test split (data/processed/kaggle_test.csv after
                   `phishguard train --full`; its sha256 must match metrics/results/evaluation.json)
  official_brand, phishstats, hard_legit, url_suites, eal_edge_cases
                   the curated evaluation files in data/evaluation/
  replay_test      the URLs of the frozen-snapshot test split replay (metrics/results/tuning/test_after.json)
  port_edge_cases  hand-written URLs that exercise parsing corners of the port (Unicode, IPv6, ports,
                   percent-encoding, punycode, wildcard suffix rules); not part of any evaluation

Usage: python demo/parity/build_urls.py [--heldout data/processed/kaggle_test.csv] --out demo/parity/out/urls.jsonl
       (also writes urls_dashboard.jsonl next to it: the subset checked against the real dashboard)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "data" / "evaluation"

PORT_EDGE_CASES = [
    "", "   ", "example.com", "EXAMPLE.COM/Path/", "http://example.com//", "https://user:pw@example.com:8080/a;b/c;d?x=1&y=#f",
    "http://[::1]/", "http://[2001:db8::1]:443/x", "http://[fe80::1%25eth0]/", "http://[::ffff:192.168.1.1]/", "http://[v1.fe]/",
    "http://[1.2.3.4]/", "http://[bad/", "http://192.168.0.1/login.php", "http://10.0.0.1:8080/", "http://8.8.8.8/",
    "http://256.1.1.1/", "http://01.2.3.4/", "http://١٢٣.١.١.١/", "http://localhost/admin", "http://printer.local/",
    "http://xn--pple-43d.com/", "http://xn--80ak6aa92e.com/login", "http://пример.рф/вход", "http://例え.テスト/",
    "http://bücher.de/", "http://ⓖoogle.com/", "http://ＥＸＡＭＰＬＥ.com/", "http://exa​mple.com/", "http://a b.com/",
    "http://ΣΑΣ.gr/ΟΔΟΣ", "http://example.com/%E2%82%AC?q=%zz&r=%E2%82&s=a+b&t=&&u", "http://example.com/?a=1&a=2&b=%20%2B",
    "http://example.com/?redirect=http%3A%2F%2Fevil.com&next=/", "http://example.com/?=x&=", "http://example.com/#?q=1",
    "http://sub.example.co.uk/x", "http://foo.bar.ck/", "http://www.ck/", "http://a.b.c.d.e.example.com/", "http://12345678.example.com/",
    "http://login-secure-account.com/verify", "http://google-login-secure.xyz/signin", "http://paypa1.com", "http://paypalx.com/login",
    "http://accounts.google.com.evil.com/", "http://evil.com/www.paypal.com/signin", "https://paypal.github.io/login",
    "http://my-amazon-update.weebly.com/", "http://s3.amazonaws.com/bucket/apple/id", "http://example.com:abc/", "http://example.com:/",
    "http://example.com.:80/", "http://.example.com/", "http://exam_ple.com/", "http://exa--mple.com/", "ftp://example.com/file",
    "javascript:alert(1)", "mailto:a@b.com", "//example.com/path", "http:///nohost", "http://@example.com", "http://a@b@c.com/",
    "http://example.com/" + "a/" * 40, "http://example.com/?" + "&".join(f"k{i}=v{i}" for i in range(30)),
    "HTTPS://WWW.PAYPAL.COM/SIGNIN", "https://www.paypal.com/signin?country.x=US&locale.x=en_US", "http://www.wikipedia.org/wiki/Main_Page",
    "http://docs.example.com/docs/api/", "http://example.com/℀/", "http://ex℀ample.com/", "http://exa%41mple.com/",
    "http://example.com/a b c", "http://example.com/😀?emoji=😀", "http://😀.example.com/",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=str(ROOT / "data" / "processed" / "kaggle_test.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-heldout", action="store_true", help="development only: leave out the held-out split")
    args = ap.parse_args()

    rows = []

    def add(set_name, url):
        rows.append({"id": len(rows), "set": set_name, "url": url})

    held = Path(args.heldout)
    got = None
    if not args.skip_heldout:
        ev = json.loads((ROOT / "metrics" / "results" / "evaluation.json").read_text())
        want = ev["data"]["test_csv_sha256"]
        got = sha256(held)
        if got != want:
            raise SystemExit(f"held-out test file sha256 {got} != evaluation.json {want}")
        for u in pd.read_csv(held, usecols=["url"], dtype=str, keep_default_na=False)["url"]:
            add("heldout_test", u)
    for name, f in [("official_brand", "official_brand_urls.jsonl"), ("phishstats", "phishstats_urls.jsonl"), ("hard_legit", "hard_legit_urls.jsonl")]:
        for line in open(EV / f, encoding="utf-8"):
            if line.strip():
                add(name, json.loads(line)["url"])
    for bucket, urls in json.loads((EV / "url_suites.json").read_text()).items():
        for u in urls:
            add("url_suites", u)
    for c in json.loads((EV / "eal_edge_cases.json").read_text())["cases"]:
        add("eal_edge_cases", c["url"])
    for r in json.loads((ROOT / "metrics" / "results" / "tuning" / "test_after.json").read_text())["rows"]:
        add("replay_test", r["url"])
    for u in PORT_EDGE_CASES:
        add("port_edge_cases", u)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not args.skip_heldout:
        # Subset for the slower check against the real dashboard: 20,000 seeded held-out rows + every other row.
        import random

        held = [r for r in rows if r["set"] == "heldout_test"]
        rest = [r for r in rows if r["set"] != "heldout_test"]
        pick = sorted(random.Random(42).sample(range(len(held)), 20000))
        with open(out.with_name("urls_dashboard.jsonl"), "w", encoding="utf-8") as fh:
            for r in [held[i] for i in pick] + rest:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["set"]] = counts.get(r["set"], 0) + 1
    print(json.dumps({"rows": len(rows), "by_set": counts, "heldout_sha256": got}))


if __name__ == "__main__":
    main()
