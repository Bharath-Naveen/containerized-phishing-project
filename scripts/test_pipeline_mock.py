"""
test_pipeline_mock.py
=====================
Validates the full pipeline using synthetic mock data — no internet required.
Run this first to confirm everything works before pointing at live APIs.

    python test_pipeline_mock.py

This generates a realistic mock dataset and runs all pipeline stages:
dedup → balance → split → report → CSV output.
"""

import random
import string
import pandas as pd
from datetime import datetime, timedelta, timezone
from build_phishing_dataset import (
    BRANDS, OUTPUT_DIR,
    get_registered_domain, clean_url,
    deduplicate, deduplicate_by_domain,
    balance_dataset, make_splits, write_report,
)
import os, logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MOCK DATA GENERATORS
# ─────────────────────────────────────────────
TLDS = [".com", ".net", ".xyz", ".info", ".top", ".live", ".site", ".online", ".shop", ".cc"]
RANDOM_WORDS = ["secure", "login", "verify", "update", "account", "access",
                "portal", "signin", "auth", "confirm", "alert", "support"]


def rand_str(n=8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def mock_phishing_url(brand_key: str) -> str:
    """Generate a realistic-looking phishing URL for a brand."""
    pattern = random.choice([
        f"https://{brand_key}-{rand_str(6)}{random.choice(TLDS)}/login",
        f"https://{rand_str(5)}-{brand_key}.{rand_str(4)}.com/account/verify",
        f"https://{rand_str(4)}.{rand_str(6)}{random.choice(TLDS)}/{brand_key}/signin",
        f"http://{brand_key}.{rand_str(8)}.com/secure/login.php",
        f"https://{rand_str(3)}{brand_key}{rand_str(3)}.com/update",
        f"https://{brand_key}-secure-{rand_str(4)}.net/auth",
    ])
    return pattern


def mock_phishing_record(brand_key: str, days_ago_max: int = 180) -> dict:
    """Generate a mock phishing record dict."""
    url = mock_phishing_url(brand_key)
    days_ago = random.randint(0, days_ago_max)
    cdate = (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "url": url,
        "label": 1,
        "brand_target": brand_key,
        "source": random.choice(["phishstats", "phishtank", "openphish"]),
        "collection_date": cdate,
        "registered_domain": get_registered_domain(url),
        "notes": f"score={round(random.uniform(2.5, 4.0), 2)} country={random.choice(['US','CN','RU','DE','BR'])}",
    }


def mock_legitimate_record(brand_key: str, brand_cfg: dict) -> dict:
    """Generate a mock legitimate record from curated brand URLs."""
    url = clean_url(random.choice(brand_cfg["legit_urls"]))
    return {
        "url": url + f"?v={rand_str(4)}",   # add param to avoid exact dups
        "label": 0,
        "brand_target": brand_key,
        "source": "brand_official",
        "collection_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "registered_domain": get_registered_domain(url),
        "notes": "curated official brand URL (mock)",
    }


def generate_mock_data(brands: dict, per_brand: int = 550) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate mock phishing and legitimate DataFrames."""
    phish_rows, legit_rows = [], []

    for brand_key, brand_cfg in brands.items():
        for _ in range(per_brand):
            phish_rows.append(mock_phishing_record(brand_key))

        legit_urls_pool = brand_cfg["legit_urls"]
        for i in range(per_brand):
            url = clean_url(legit_urls_pool[i % len(legit_urls_pool)])
            legit_rows.append({
                "url": url + f"/{rand_str(4)}",
                "label": 0,
                "brand_target": brand_key,
                "source": "brand_official",
                "collection_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "registered_domain": get_registered_domain(url),
                "notes": "curated official brand URL (mock)",
            })

    return pd.DataFrame(phish_rows), pd.DataFrame(legit_rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("MOCK PIPELINE TEST - generating synthetic data")
    log.info("=" * 55)

    # Generate
    log.info(f"\nGenerating mock data for {len(BRANDS)} brands x 550 samples each...")
    phish_df, legit_df = generate_mock_data(BRANDS, per_brand=550)
    log.info(f"  Raw phishing : {len(phish_df):,}")
    log.info(f"  Raw legitimate: {len(legit_df):,}")

    # Dedup
    phish_df = deduplicate(phish_df)
    phish_df = deduplicate_by_domain(phish_df, max_per_domain=10)
    legit_df = deduplicate(legit_df)

    # Balance
    full_df = balance_dataset(phish_df, legit_df)
    full_df.to_csv(f"{OUTPUT_DIR}/mock_dataset_full.csv", index=False)

    # Split
    train_df, test_df, holdout_df = make_splits(full_df)
    train_df.to_csv(f"{OUTPUT_DIR}/mock_dataset_train.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/mock_dataset_test_standard.csv", index=False)
    holdout_df.to_csv(f"{OUTPUT_DIR}/mock_dataset_test_recent.csv", index=False)

    # Report
    report = write_report(full_df, train_df, test_df, holdout_df)
    with open(f"{OUTPUT_DIR}/mock_collection_report.txt", "w") as f:
        f.write(report)

    print("\n" + report)

    # Sanity assertions
    assert len(full_df) > 0, "Full dataset is empty!"
    assert full_df["label"].nunique() == 2, "Missing a class!"
    assert (full_df["label"] == 1).sum() > 0, "No phishing samples!"
    assert (full_df["label"] == 0).sum() > 0, "No legitimate samples!"
    train_domains = set(train_df["registered_domain"])
    test_domains  = set(test_df["registered_domain"])
    leak = train_domains & test_domains
    assert len(leak) == 0, f"Domain leakage detected! {len(leak)} domains in both train and test."
    log.info("\nAll assertions passed - pipeline is valid.")
    log.info(f"   Outputs in: {OUTPUT_DIR}/")

    # Print sample rows
    print("\n-- Sample rows from full dataset --")
    print(full_df[["url", "label", "brand_target", "source", "collection_date",
                    "registered_domain"]].sample(10, random_state=1).to_string(index=False))


if __name__ == "__main__":
    main()
