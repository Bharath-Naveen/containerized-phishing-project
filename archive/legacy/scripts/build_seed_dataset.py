import csv
import hashlib
from urllib.parse import urlparse
import requests

PHISHSTATS_URL = "https://api.phishstats.info/api/phishing?_sort=-id&_size=100"
OPENPHISH_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"

OUTPUT_CSV = "seed_phish_urls.csv"


def normalize_url(url: str) -> str:
    return url.strip()


def safe_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def fetch_phishstats():
    rows = []
    r = requests.get(PHISHSTATS_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    for item in data:
        url = normalize_url(item.get("url", ""))
        if not url:
            continue
        rows.append({
            "url": url,
            "source": "phishstats",
            "brand_hint": item.get("brand", ""),
            "domain": safe_domain(url),
            "countrycode": item.get("countrycode", ""),
            "phishscore": item.get("phish_score", ""),
        })
    return rows


def fetch_openphish():
    rows = []
    r = requests.get(OPENPHISH_URL, timeout=30)
    r.raise_for_status()
    for line in r.text.splitlines():
        url = normalize_url(line)
        if not url:
            continue
        rows.append({
            "url": url,
            "source": "openphish",
            "brand_hint": "",
            "domain": safe_domain(url),
            "countrycode": "",
            "phishscore": "",
        })
    return rows


def dedupe(rows):
    seen = set()
    out = []
    for row in rows:
        key = hashlib.sha256(row["url"].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    rows = []
    rows.extend(fetch_phishstats())
    rows.extend(fetch_openphish())
    rows = dedupe(rows)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["url", "source", "brand_hint", "domain", "countrycode", "phishscore"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()