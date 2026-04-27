"""
build_phishing_dataset.py
=========================
Phishing Detection Capstone — Dataset Builder
Goal: ~20,000 balanced URLs (10k phishing / 10k legitimate)
      across 20 major brands, with rich metadata.

Run:
    pip install requests tldextract pandas
    python build_phishing_dataset.py

Outputs (in ./phishing_dataset/):
    raw_phishing.csv          — all collected phishing URLs before dedup
    raw_legitimate.csv        — all collected legitimate URLs before dedup
    dataset_full.csv          — deduplicated, balanced, labelled dataset
    dataset_train.csv         — 80% train split (by registered domain)
    dataset_test_recent.csv   — holdout: most recent 10% by date
    dataset_test_standard.csv — standard 10% test split
    collection_report.txt     — per-brand counts and data quality notes
"""

import os
import re
import time
import random
import logging
import requests
import tldextract
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR        = "./phishing_dataset"
TARGET_PER_BRAND  = 500          # phishing URLs to attempt per brand
LEGIT_PER_BRAND   = 500          # legitimate URLs per brand
PHISHTANK_DB_PATH = ""           # optional: path to local phishtank JSON dump
                                 # download from: http://data.phishtank.com/data/<key>/online-valid.json.bz2
TRANCO_CSV_PATH   = ""           # optional: path to local Tranco top-1m CSV
                                 # download from: https://tranco-list.eu/top-1m.csv.zip
REQUEST_DELAY     = 0.4          # seconds between PhishStats requests (be polite)
MAX_RETRIES       = 3
PHISHSTATS_SIZE   = 1000         # items per page (API max is typically 100 — we paginate)
REQUEST_TIMEOUT_S = 15
RETRY_BACKOFF_BASE_S = 2
MAX_FAILED_PAGES_PER_QUERY = 4
MAX_FAILED_KEYWORDS_PER_BRAND = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# BRAND DEFINITIONS
# ─────────────────────────────────────────────
# Each entry:
#   "keywords"     — search terms for PhishStats URL-like filter
#   "official_domain" — canonical registered domain for the brand
#   "legit_urls"   — curated seed URLs of confirmed-legitimate pages
#   "tranco_grep"  — substring to match against Tranco list as fallback
BRANDS = {
    "microsoft": {
        "keywords": ["microsoft", "office365", "onedrive", "outlook", "azure", "microsoftonline"],
        "official_domain": "microsoft.com",
        "legit_urls": [
            "https://login.microsoftonline.com/",
            "https://account.microsoft.com/",
            "https://www.microsoft.com/en-us/microsoft-365",
            "https://www.microsoft.com/en-us/security",
            "https://outlook.live.com/mail/",
            "https://onedrive.live.com/",
            "https://www.microsoft.com/en-us/store/apps/windows",
            "https://support.microsoft.com/",
            "https://www.microsoft.com/en-us/microsoft-teams/",
            "https://azure.microsoft.com/en-us/",
            "https://www.microsoft.com/en-us/windows",
            "https://www.microsoft.com/en-us/surface",
            "https://www.microsoft.com/en-us/microsoft-365/business",
            "https://signup.microsoft.com/",
            "https://portal.office.com/",
        ],
        "tranco_grep": "microsoft",
    },
    "amazon": {
        "keywords": ["amazon", "aws", "amazonaws"],
        "official_domain": "amazon.com",
        "legit_urls": [
            "https://www.amazon.com/ap/signin",
            "https://www.amazon.com/",
            "https://www.amazon.com/gp/cart/view.html",
            "https://www.amazon.com/gp/css/order-history",
            "https://www.amazon.com/hz/mycd/digital-console/alldevices",
            "https://aws.amazon.com/console/",
            "https://sellercentral.amazon.com/",
            "https://affiliate-program.amazon.com/",
            "https://www.amazon.com/prime",
            "https://www.amazon.com/kindle-dbs/storefront",
            "https://music.amazon.com/",
            "https://www.amazon.com/gp/help/customer/display.html",
            "https://payments.amazon.com/",
            "https://www.amazon.com/gp/browse.html?node=16008589011",
            "https://advertising.amazon.com/",
        ],
        "tranco_grep": "amazon",
    },
    "paypal": {
        "keywords": ["paypal", "paypa1"],
        "official_domain": "paypal.com",
        "legit_urls": [
            "https://www.paypal.com/signin",
            "https://www.paypal.com/myaccount/summary",
            "https://www.paypal.com/us/home",
            "https://www.paypal.com/us/webapps/mpp/send-money-online",
            "https://www.paypal.com/us/webapps/mpp/paypal-fees",
            "https://www.paypal.com/us/webapps/mpp/merchant",
            "https://www.paypal.com/us/webapps/mpp/mobile-apps",
            "https://developer.paypal.com/home",
            "https://www.paypal.com/us/webapps/mpp/buyer-protection",
            "https://www.paypal.com/us/webapps/mpp/personal",
            "https://www.paypal.com/us/smarthelp/home",
            "https://www.paypal.com/us/webapps/mpp/business",
            "https://www.paypal.com/us/webapps/mpp/paypal-credit",
            "https://venmo.com/",
            "https://www.paypal.com/us/webapps/mpp/about",
        ],
        "tranco_grep": "paypal",
    },
    "google": {
        "keywords": ["google", "gmail", "googledocs", "googlelogin", "g00gle"],
        "official_domain": "google.com",
        "legit_urls": [
            "https://accounts.google.com/signin",
            "https://myaccount.google.com/",
            "https://mail.google.com/mail/",
            "https://drive.google.com/",
            "https://docs.google.com/",
            "https://calendar.google.com/",
            "https://meet.google.com/",
            "https://workspace.google.com/",
            "https://www.google.com/maps",
            "https://play.google.com/store",
            "https://cloud.google.com/",
            "https://photos.google.com/",
            "https://www.youtube.com/",
            "https://ads.google.com/home/",
            "https://analytics.google.com/",
        ],
        "tranco_grep": "google",
    },
    "apple": {
        "keywords": ["apple", "icloud", "appleid", "app1e"],
        "official_domain": "apple.com",
        "legit_urls": [
            "https://appleid.apple.com/",
            "https://idmsa.apple.com/IDMSWebAuth/signin",
            "https://www.icloud.com/",
            "https://www.apple.com/shop/sign-in",
            "https://www.apple.com/app-store/",
            "https://www.apple.com/mac/",
            "https://www.apple.com/iphone/",
            "https://www.apple.com/support/",
            "https://discussions.apple.com/",
            "https://developer.apple.com/",
            "https://www.apple.com/apple-pay/",
            "https://www.apple.com/apple-one/",
            "https://tv.apple.com/",
            "https://music.apple.com/",
            "https://www.apple.com/privacy/",
        ],
        "tranco_grep": "apple",
    },
    "netflix": {
        "keywords": ["netflix", "netfl1x", "net-flix"],
        "official_domain": "netflix.com",
        "legit_urls": [
            "https://www.netflix.com/login",
            "https://www.netflix.com/browse",
            "https://www.netflix.com/signup",
            "https://www.netflix.com/account",
            "https://www.netflix.com/watch",
            "https://www.netflix.com/title",
            "https://www.netflix.com/browse/genre/83",
            "https://www.netflix.com/browse/genre/34399",
            "https://www.netflix.com/kids",
            "https://help.netflix.com/en",
            "https://www.netflix.com/us-en/legal/termsofuse",
            "https://jobs.netflix.com/",
            "https://ir.netflix.net/ir/overview/default.aspx",
            "https://www.netflix.com/tudum",
            "https://devices.netflix.com/en/",
        ],
        "tranco_grep": "netflix",
    },
    "facebook": {
        "keywords": ["facebook", "faceb00k", "meta", "fb-login"],
        "official_domain": "facebook.com",
        "legit_urls": [
            "https://www.facebook.com/login",
            "https://www.facebook.com/",
            "https://www.facebook.com/marketplace",
            "https://www.facebook.com/groups",
            "https://business.facebook.com/",
            "https://www.facebook.com/watch",
            "https://www.facebook.com/gaming",
            "https://www.facebook.com/help",
            "https://www.facebook.com/settings",
            "https://developers.facebook.com/",
            "https://www.facebook.com/ads/manager",
            "https://www.meta.com/",
            "https://about.fb.com/",
            "https://m.facebook.com/",
            "https://www.instagram.com/accounts/login/",
        ],
        "tranco_grep": "facebook",
    },
    "instagram": {
        "keywords": ["instagram", "insta-gram", "1nstagram", "instagramm"],
        "official_domain": "instagram.com",
        "legit_urls": [
            "https://www.instagram.com/accounts/login/",
            "https://www.instagram.com/",
            "https://www.instagram.com/explore/",
            "https://www.instagram.com/reels/",
            "https://www.instagram.com/accounts/emailsignup/",
            "https://business.instagram.com/",
            "https://www.instagram.com/p/",
            "https://www.instagram.com/stories/",
            "https://help.instagram.com/",
            "https://www.instagram.com/accounts/password/reset/",
            "https://www.instagram.com/accounts/settings/",
            "https://creator.instagram.com/",
            "https://about.instagram.com/",
            "https://www.instagram.com/direct/inbox/",
            "https://www.instagram.com/accounts/manage_access/",
        ],
        "tranco_grep": "instagram",
    },
    "linkedin": {
        "keywords": ["linkedin", "linke-din", "1inkedin"],
        "official_domain": "linkedin.com",
        "legit_urls": [
            "https://www.linkedin.com/login",
            "https://www.linkedin.com/feed/",
            "https://www.linkedin.com/jobs/",
            "https://www.linkedin.com/in/",
            "https://www.linkedin.com/company/",
            "https://www.linkedin.com/learning/",
            "https://www.linkedin.com/notifications/",
            "https://www.linkedin.com/messaging/",
            "https://www.linkedin.com/mynetwork/",
            "https://premium.linkedin.com/",
            "https://business.linkedin.com/",
            "https://www.linkedin.com/talent/",
            "https://www.linkedin.com/sales/",
            "https://developer.linkedin.com/",
            "https://www.linkedin.com/help/linkedin",
        ],
        "tranco_grep": "linkedin",
    },
    "dropbox": {
        "keywords": ["dropbox", "dr0pbox", "drop-box"],
        "official_domain": "dropbox.com",
        "legit_urls": [
            "https://www.dropbox.com/login",
            "https://www.dropbox.com/home",
            "https://www.dropbox.com/plans",
            "https://www.dropbox.com/business",
            "https://www.dropbox.com/paper",
            "https://www.dropbox.com/sign",
            "https://www.dropbox.com/features",
            "https://help.dropbox.com/",
            "https://www.dropbox.com/terms",
            "https://www.dropbox.com/privacy",
            "https://developers.dropbox.com/",
            "https://www.dropbox.com/paper",
            "https://www.dropbox.com/create-account",
            "https://experience.dropbox.com/",
            "https://www.dropbox.com/transfer",
        ],
        "tranco_grep": "dropbox",
    },
    "adobe": {
        "keywords": ["adobe", "acrobat", "adobe-sign", "adobe-cloud"],
        "official_domain": "adobe.com",
        "legit_urls": [
            "https://auth.services.adobe.com/en_US/index.html",
            "https://account.adobe.com/",
            "https://www.adobe.com/",
            "https://www.adobe.com/products/acrobat.html",
            "https://www.adobe.com/products/photoshop.html",
            "https://www.adobe.com/products/illustrator.html",
            "https://www.adobe.com/sign.html",
            "https://acrobat.adobe.com/",
            "https://www.adobe.com/creativecloud.html",
            "https://www.adobe.com/products/premiere.html",
            "https://www.adobe.com/express/",
            "https://helpx.adobe.com/",
            "https://www.adobe.com/products/stock.html",
            "https://fonts.adobe.com/",
            "https://www.adobe.com/products/firefly.html",
        ],
        "tranco_grep": "adobe",
    },
    "docusign": {
        "keywords": ["docusign", "docu-sign", "docusign-login"],
        "official_domain": "docusign.com",
        "legit_urls": [
            "https://account.docusign.com/",
            "https://www.docusign.com/",
            "https://www.docusign.com/esignature",
            "https://www.docusign.com/products/payments",
            "https://www.docusign.com/products/contracts",
            "https://www.docusign.com/products/identity-verification",
            "https://support.docusign.com/",
            "https://developers.docusign.com/",
            "https://www.docusign.com/pricing",
            "https://www.docusign.com/solutions/small-business",
            "https://www.docusign.com/solutions/enterprise",
            "https://www.docusign.com/trust",
            "https://www.docusign.com/company",
            "https://partners.docusign.com/",
            "https://status.docusign.com/",
        ],
        "tranco_grep": "docusign",
    },
    "chase": {
        "keywords": ["chase", "chase-bank", "jpmorgan", "chaseonline"],
        "official_domain": "chase.com",
        "legit_urls": [
            "https://secure.chase.com/web/auth/",
            "https://www.chase.com/",
            "https://www.chase.com/personal/checking",
            "https://www.chase.com/personal/savings",
            "https://www.chase.com/personal/credit-cards",
            "https://www.chase.com/personal/mortgage",
            "https://www.chase.com/personal/auto-loans",
            "https://www.chase.com/business",
            "https://www.chase.com/personal/investments",
            "https://www.chase.com/personal/sapphire",
            "https://www.chase.com/personal/freedom",
            "https://www.chase.com/personal/ink-business",
            "https://www.jpmorgan.com/",
            "https://www.chase.com/personal/customer-service",
            "https://mobilebanking.chase.com/",
        ],
        "tranco_grep": "chase",
    },
    "bankofamerica": {
        "keywords": ["bankofamerica", "bank-of-america", "bofa", "boa-bank"],
        "official_domain": "bankofamerica.com",
        "legit_urls": [
            "https://www.bankofamerica.com/online-banking/sign-in/",
            "https://www.bankofamerica.com/",
            "https://www.bankofamerica.com/deposits/checking/",
            "https://www.bankofamerica.com/deposits/savings/",
            "https://www.bankofamerica.com/credit-cards/",
            "https://www.bankofamerica.com/mortgage/",
            "https://www.bankofamerica.com/auto-loans/",
            "https://www.bankofamerica.com/smallbusiness/",
            "https://www.bankofamerica.com/investing/",
            "https://www.bankofamerica.com/rewards/",
            "https://www.bankofamerica.com/security-center/",
            "https://www.bankofamerica.com/preferred-rewards/",
            "https://www.bankofamerica.com/digital-banking/",
            "https://www.bankofamerica.com/credit-cards/travel-rewards/",
            "https://newsroom.bankofamerica.com/",
        ],
        "tranco_grep": "bankofamerica",
    },
    "wellsfargo": {
        "keywords": ["wellsfargo", "wells-fargo", "wf-bank"],
        "official_domain": "wellsfargo.com",
        "legit_urls": [
            "https://connect.secure.wellsfargo.com/auth/login/",
            "https://www.wellsfargo.com/",
            "https://www.wellsfargo.com/checking/",
            "https://www.wellsfargo.com/savings/",
            "https://www.wellsfargo.com/credit-cards/",
            "https://www.wellsfargo.com/mortgage/",
            "https://www.wellsfargo.com/auto-loans/",
            "https://www.wellsfargo.com/small-business/",
            "https://www.wellsfargo.com/investing/",
            "https://www.wellsfargo.com/security/",
            "https://www.wellsfargo.com/mobile-banking/",
            "https://www.wellsfargo.com/personal-loans/",
            "https://www.wellsfargo.com/student-loans/",
            "https://www.wellsfargo.com/about/",
            "https://www.wellsfargo.com/help/",
        ],
        "tranco_grep": "wellsfargo",
    },
    "amex": {
        "keywords": ["americanexpress", "amex", "american-express"],
        "official_domain": "americanexpress.com",
        "legit_urls": [
            "https://www.americanexpress.com/en-us/account/login",
            "https://www.americanexpress.com/",
            "https://www.americanexpress.com/us/credit-cards/",
            "https://www.americanexpress.com/en-us/benefits/",
            "https://www.americanexpress.com/en-us/travel/",
            "https://www.americanexpress.com/en-us/shopping/",
            "https://www.americanexpress.com/en-us/business/",
            "https://www.americanexpress.com/en-us/rewards/membership-rewards/",
            "https://global.americanexpress.com/",
            "https://www.americanexpress.com/en-us/customer-service/",
            "https://www.americanexpress.com/en-us/security/",
            "https://developer.americanexpress.com/",
            "https://www.americanexpress.com/en-us/network/",
            "https://www.americanexpress.com/en-us/financial-services/",
            "https://www.americanexpress.com/en-us/about-us/",
        ],
        "tranco_grep": "americanexpress",
    },
    "dhl": {
        "keywords": ["dhl", "dhl-express", "dhlparcel", "dhl-tracking"],
        "official_domain": "dhl.com",
        "legit_urls": [
            "https://www.dhl.com/us-en/home.html",
            "https://www.dhl.com/us-en/home/tracking.html",
            "https://www.dhl.com/us-en/home/shipping.html",
            "https://www.dhl.com/us-en/home/logistics.html",
            "https://www.dhl.com/us-en/home/our-divisions/parcel.html",
            "https://www.dhl.com/us-en/home/our-divisions/express.html",
            "https://www.dhl.com/us-en/home/our-divisions/freight.html",
            "https://www.dhlexpress.com/",
            "https://www.dhl.com/us-en/home/get-a-quote.html",
            "https://www.dhl.com/us-en/home/find-a-location.html",
            "https://www.dhl.com/us-en/home/case-studies.html",
            "https://www.dhl.com/us-en/home/our-divisions/ecommerce-solutions.html",
            "https://www.dhl.com/global-en/home/sustainability.html",
            "https://www.dhl.com/us-en/home/newsletter-signup.html",
            "https://myaccount.dhl.com/",
        ],
        "tranco_grep": "dhl",
    },
    "fedex": {
        "keywords": ["fedex", "fed-ex", "fedex-tracking"],
        "official_domain": "fedex.com",
        "legit_urls": [
            "https://www.fedex.com/en-us/home.html",
            "https://www.fedex.com/en-us/tracking.html",
            "https://www.fedex.com/en-us/shipping.html",
            "https://www.fedex.com/en-us/create-account.html",
            "https://www.fedex.com/en-us/customer-support.html",
            "https://www.fedex.com/en-us/find-a-location.html",
            "https://www.fedex.com/en-us/small-business.html",
            "https://www.fedex.com/en-us/printing.html",
            "https://www.fedex.com/en-us/ecommerce.html",
            "https://www.fedex.com/en-us/health-care.html",
            "https://www.fedex.com/en-us/supply-chain.html",
            "https://developer.fedex.com/",
            "https://www.fedex.com/en-us/rate-finder.html",
            "https://www.fedex.com/en-us/about/company-information.html",
            "https://www.fedex.com/en-us/sustainability.html",
        ],
        "tranco_grep": "fedex",
    },
    "usps": {
        "keywords": ["usps", "usps-tracking", "uspostmail", "postal-service"],
        "official_domain": "usps.com",
        "legit_urls": [
            "https://www.usps.com/",
            "https://tools.usps.com/go/TrackConfirmAction_input",
            "https://reg.usps.com/entreg/LoginAction_input",
            "https://www.usps.com/ship/",
            "https://www.usps.com/manage/",
            "https://informeddelivery.usps.com/",
            "https://www.usps.com/shop/priority-mail.htm",
            "https://www.usps.com/shop/first-class-mail.htm",
            "https://www.usps.com/business/",
            "https://www.usps.com/international/",
            "https://www.usps.com/poboxes/",
            "https://www.usps.com/hold-mail/",
            "https://store.usps.com/store/",
            "https://www.usps.com/help/welcome-center.htm",
            "https://cns.usps.com/labelInformation.shtml",
        ],
        "tranco_grep": "usps",
    },
    "coinbase": {
        "keywords": ["coinbase", "coinbase-wallet", "coinbase-pro"],
        "official_domain": "coinbase.com",
        "legit_urls": [
            "https://www.coinbase.com/signin",
            "https://www.coinbase.com/",
            "https://www.coinbase.com/buy-bitcoin",
            "https://www.coinbase.com/buy-ethereum",
            "https://www.coinbase.com/wallet",
            "https://www.coinbase.com/advanced-trade",
            "https://www.coinbase.com/learn",
            "https://www.coinbase.com/price",
            "https://www.coinbase.com/card",
            "https://www.coinbase.com/earn",
            "https://www.coinbase.com/institutional",
            "https://www.coinbase.com/developer-platform",
            "https://www.coinbase.com/legal/user_agreement",
            "https://help.coinbase.com/",
            "https://pro.coinbase.com/",
        ],
        "tranco_grep": "coinbase",
    },
}


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def get_registered_domain(url: str) -> str:
    """Extract eTLD+1 from a URL using tldextract."""
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return ext.domain or url[:60]
    except Exception:
        return ""


def clean_url(url: str) -> str:
    """Normalise URL — strip trailing whitespace/newlines."""
    return url.strip().rstrip("/").lower() if url else ""


def _empty_dataset_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "url",
            "label",
            "brand_target",
            "source",
            "collection_date",
            "registered_domain",
            "notes",
        ]
    )


def _save_partial_progress(phish_records: list[dict], legit_records: list[dict] | None = None) -> None:
    pd.DataFrame(phish_records).to_csv(f"{OUTPUT_DIR}/raw_phishing.csv", index=False)
    if legit_records is not None:
        pd.DataFrame(legit_records).to_csv(f"{OUTPUT_DIR}/raw_legitimate.csv", index=False)


def phishstats_fetch(keyword: str, max_records: int = 1000) -> list[dict]:
    """
    Fetch phishing URLs from PhishStats API for a given keyword.
    Handles pagination automatically.
    Returns list of dicts with url, score, date, ip, country, etc.
    """
    results = []
    page = 0
    page_size = 100  # PhishStats caps at 100 per request

    failed_pages = 0
    while len(results) < max_records:
        url = (
            f"https://api.phishstats.info/api/phishing"
            f"?_where=(url,like,~{keyword}~)"
            f"&_sort=-id"
            f"&_size={page_size}"
            f"&_p={page}"
        )
        data = []
        page_ok = False
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT_S)
                if r.status_code == 200:
                    data = r.json()
                    page_ok = True
                    if not data:          # empty page = done
                        return results
                    results.extend(data)
                    break
                elif r.status_code == 429:
                    sleep_s = min(30, RETRY_BACKOFF_BASE_S ** attempt + random.uniform(0.0, 0.7))
                    log.warning(
                        f"Rate limited on keyword '{keyword}' page={page}; retry {attempt+1}/{MAX_RETRIES} in {sleep_s:.1f}s"
                    )
                    time.sleep(sleep_s)
                else:
                    log.warning(f"HTTP {r.status_code} for keyword '{keyword}' page {page}")
                    return results
            except requests.RequestException as e:
                sleep_s = min(30, RETRY_BACKOFF_BASE_S ** attempt + random.uniform(0.0, 0.7))
                log.warning(
                    f"Request error keyword='{keyword}' page={page} ({attempt+1}/{MAX_RETRIES}): {e}; "
                    f"backoff {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)

        if not page_ok:
            failed_pages += 1
            log.warning(
                "PhishStats page failed keyword='%s' page=%s (failed_pages=%s/%s).",
                keyword,
                page,
                failed_pages,
                MAX_FAILED_PAGES_PER_QUERY,
            )
            if failed_pages >= MAX_FAILED_PAGES_PER_QUERY:
                log.warning("Fail-fast query stop for keyword '%s' after repeated page failures.", keyword)
                break
        else:
            failed_pages = 0

        page += 1
        time.sleep(REQUEST_DELAY)

        if len(data) < page_size:
            break  # last page reached

    return results[:max_records]


def load_phishtank_db(path: str) -> pd.DataFrame:
    """
    Load a local PhishTank JSON dump and return a DataFrame
    with columns: url, target (brand), verified, submission_time.

    Download from:
    http://data.phishtank.com/data/<your_api_key>/online-valid.json.bz2
    Then: bunzip2 online-valid.json.bz2
    """
    import json
    log.info(f"Loading PhishTank DB from {path}…")
    with open(path) as f:
        data = json.load(f)
    rows = []
    for entry in data:
        rows.append({
            "url": entry.get("url", ""),
            "target": entry.get("target", ""),
            "verified": entry.get("verified", ""),
            "submission_time": entry.get("submission_time", ""),
        })
    return pd.DataFrame(rows)


def load_tranco(path: str) -> list[str]:
    """
    Load Tranco top-1M CSV (rank, domain format).
    Download from: https://tranco-list.eu/top-1m.csv.zip → unzip
    Returns list of domains.
    """
    log.info(f"Loading Tranco list from {path}…")
    df = pd.read_csv(path, header=None, names=["rank", "domain"])
    return df["domain"].tolist()


# ─────────────────────────────────────────────
# COLLECTION FUNCTIONS
# ─────────────────────────────────────────────
def collect_phishing_urls(brands: dict, phishtank_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Collect phishing URLs for all brands from PhishStats (+ optional PhishTank).
    Returns a DataFrame with raw phishing records.
    """
    records: list[dict] = []
    failed_brands: list[str] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for brand_key, brand_cfg in brands.items():
        try:
            log.info(f"[PHISH] Collecting for brand: {brand_key}")
            brand_urls = set()
            brand_records: list[dict] = []
            failed_keywords = 0

            # ── PhishStats primary (try all keywords, stop when we have enough) ──
            for keyword in brand_cfg["keywords"]:
                if len(brand_urls) >= TARGET_PER_BRAND:
                    break
                if failed_keywords >= MAX_FAILED_KEYWORDS_PER_BRAND:
                    log.warning(
                        "Fail-fast: stopping brand '%s' after %s weak/failed keyword pulls.",
                        brand_key,
                        failed_keywords,
                    )
                    break
                log.info(f"  → PhishStats keyword: '{keyword}'")
                raw = phishstats_fetch(keyword, max_records=TARGET_PER_BRAND)
                if not raw:
                    failed_keywords += 1
                    continue
                for item in raw:
                    url = clean_url(item.get("url", ""))
                    if not url or url in brand_urls:
                        continue
                    brand_urls.add(url)
                    brand_records.append({
                        "url": url,
                        "label": 1,
                        "brand_target": brand_key,
                        "source": "phishstats",
                        "collection_date": item.get("date", today)[:10] if item.get("date") else today,
                        "registered_domain": get_registered_domain(url),
                        "notes": f"score={item.get('score','')} country={item.get('countrycode','')}",
                    })

            # ── PhishTank supplement (if local DB provided) ──
            if phishtank_df is not None and len(brand_urls) < TARGET_PER_BRAND:
                needed = TARGET_PER_BRAND - len(brand_urls)
                target_col = phishtank_df["target"].str.lower()
                matches = phishtank_df[target_col.str.contains(brand_key, na=False)].head(needed * 2)
                for _, row in matches.iterrows():
                    url = clean_url(row["url"])
                    if not url or url in brand_urls:
                        continue
                    brand_urls.add(url)
                    brand_records.append({
                        "url": url,
                        "label": 1,
                        "brand_target": brand_key,
                        "source": "phishtank",
                        "collection_date": str(row.get("submission_time", today))[:10],
                        "registered_domain": get_registered_domain(url),
                        "notes": f"verified={row.get('verified','')}",
                    })
                    if len(brand_urls) >= TARGET_PER_BRAND:
                        break

            count = len(brand_records)
            log.info(f"  ✓ {brand_key}: {count} phishing URLs collected")
            if count < TARGET_PER_BRAND // 2:
                log.warning(f"  ⚠ Low yield for {brand_key} ({count}/{TARGET_PER_BRAND}). "
                            f"Consider adding PhishTank or OpenPhish for this brand.")
            records.extend(brand_records)
            _save_partial_progress(records)
        except Exception as exc:  # noqa: BLE001
            failed_brands.append(brand_key)
            log.exception("Brand collection failed for %s: %s", brand_key, exc)
            _save_partial_progress(records)
            continue

    return pd.DataFrame(records), failed_brands


def collect_legitimate_urls(brands: dict, tranco_domains: list[str] | None = None) -> pd.DataFrame:
    """
    Collect legitimate URLs for all brands.
    Uses curated seed URLs from brand definitions + Tranco supplement.
    """
    records = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for brand_key, brand_cfg in brands.items():
        log.info(f"[LEGIT] Collecting for brand: {brand_key}")
        legit_urls = set()
        brand_records = []

        # ── Curated official brand URLs (always first) ──
        for url in brand_cfg["legit_urls"]:
            url = clean_url(url)
            if not url or url in legit_urls:
                continue
            legit_urls.add(url)
            brand_records.append({
                "url": url,
                "label": 0,
                "brand_target": brand_key,
                "source": "brand_official",
                "collection_date": today,
                "registered_domain": get_registered_domain(url),
                "notes": "curated official brand URL",
            })

        # ── Tranco supplement if available and still need more ──
        if tranco_domains and len(legit_urls) < LEGIT_PER_BRAND:
            grep_term = brand_cfg["tranco_grep"]
            matching = [d for d in tranco_domains if grep_term in d.lower()]
            for domain in matching:
                if len(legit_urls) >= LEGIT_PER_BRAND:
                    break
                url = f"https://{domain}"
                url_clean = clean_url(url)
                if url_clean in legit_urls:
                    continue
                legit_urls.add(url_clean)
                brand_records.append({
                    "url": url_clean,
                    "label": 0,
                    "brand_target": brand_key,
                    "source": "tranco",
                    "collection_date": today,
                    "registered_domain": get_registered_domain(url),
                    "notes": "tranco top-1m domain match",
                })

        count = len(brand_records)
        log.info(f"  ✓ {brand_key}: {count} legitimate URLs collected")
        records.extend(brand_records)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# DEDUPLICATION & BALANCING
# ─────────────────────────────────────────────
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact URL duplicates, keep first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    after = len(df)
    log.info(f"Dedup: {before} → {after} rows (removed {before - after} exact duplicates)")
    return df


def deduplicate_by_domain(df: pd.DataFrame, max_per_domain: int = 10) -> pd.DataFrame:
    """
    Cap the number of URLs from any single registered domain
    to prevent a single phishing kit from dominating the dataset.
    """
    before = len(df)
    df = df.groupby(["registered_domain", "label"]).head(max_per_domain).reset_index(drop=True)
    after = len(df)
    log.info(f"Domain cap (max {max_per_domain}/domain): {before} → {after} rows")
    return df


def balance_dataset(phish_df: pd.DataFrame, legit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance overall dataset to equal phishing/legitimate counts.
    Also balance per brand as best as possible.
    """
    brands = list(BRANDS.keys())
    balanced_records = []

    for brand in brands:
        p = phish_df[phish_df["brand_target"] == brand]
        l = legit_df[legit_df["brand_target"] == brand]
        n = min(len(p), len(l), TARGET_PER_BRAND)
        if n == 0:
            log.warning(f"Brand '{brand}' has 0 URLs in one class — skipping balance for it.")
            balanced_records.append(p)
            balanced_records.append(l)
            continue
        balanced_records.append(p.sample(n=min(n, len(p)), random_state=42))
        balanced_records.append(l.sample(n=min(n, len(l)), random_state=42))

    combined = pd.concat(balanced_records, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    log.info(f"Balanced dataset: {len(combined)} rows | "
             f"phishing={combined['label'].sum()} | "
             f"legitimate={(combined['label'] == 0).sum()}")
    return combined


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-way leak-safe split:
      - Recent holdout: most recent 10% of records by collection_date
      - Train: 80% of remaining records (split by registered domain)
      - Standard test: 20% of remaining records

    Splitting by registered domain ensures no domain appears in
    both train and test (prevents URL-level data leakage).
    """
    df = df.copy()
    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")

    # ── 1. Carve out recent holdout (top 10% by date) ──
    cutoff = df["collection_date"].quantile(0.90)
    recent_mask = df["collection_date"] >= cutoff
    recent_holdout = df[recent_mask].copy()
    remaining = df[~recent_mask].copy()
    log.info(f"Recent holdout: {len(recent_holdout)} records (>= {cutoff.date() if pd.notna(cutoff) else 'N/A'})")

    # ── 2. Domain-based 80/20 split on remaining ──
    domains = remaining["registered_domain"].unique()
    random.seed(42)
    random.shuffle(domains)
    split_idx = int(len(domains) * 0.8)
    train_domains = set(domains[:split_idx])

    train = remaining[remaining["registered_domain"].isin(train_domains)].copy()
    test = remaining[~remaining["registered_domain"].isin(train_domains)].copy()

    log.info(f"Train: {len(train)} | Test (standard): {len(test)} | "
             f"Train domains: {len(train_domains)} | Test domains: {len(domains) - split_idx}")

    # Restore collection_date as string for CSV output
    for d in [recent_holdout, train, test]:
        d["collection_date"] = d["collection_date"].dt.strftime("%Y-%m-%d").fillna("")

    return train, test, recent_holdout


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
def write_report(full_df: pd.DataFrame, train_df: pd.DataFrame,
                 test_df: pd.DataFrame, holdout_df: pd.DataFrame) -> str:
    """Generate a human-readable collection report."""
    lines = [
        "=" * 60,
        "PHISHING DATASET COLLECTION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "OVERALL COUNTS",
        f"  Total URLs         : {len(full_df):,}",
        f"  Phishing  (label=1): {(full_df['label'] == 1).sum():,}",
        f"  Legitimate (label=0): {(full_df['label'] == 0).sum():,}",
        "",
        "SPLITS",
        f"  Train              : {len(train_df):,}",
        f"  Test (standard)    : {len(test_df):,}",
        f"  Test (recent HO)   : {len(holdout_df):,}",
        "",
        "SOURCES",
    ]
    for src, cnt in full_df["source"].value_counts().items():
        lines.append(f"  {src:<25}: {cnt:,}")

    lines += ["", "PER-BRAND BREAKDOWN", "-" * 60]
    for brand in sorted(full_df["brand_target"].unique()):
        bdf = full_df[full_df["brand_target"] == brand]
        p = (bdf["label"] == 1).sum()
        l = (bdf["label"] == 0).sum()
        lines.append(f"  {brand:<20} phish={p:>4}  legit={l:>4}  total={len(bdf):>5}")

    lines += [
        "",
        "DATA QUALITY NOTES",
        "  - Exact URL deduplication applied",
        "  - Domain-level cap: max 10 URLs per registered domain per class",
        "  - Split is by registered domain (no domain leakage across splits)",
        "  - Recent holdout = top 10% by collection_date",
        "  - Run head-request liveness check before feature extraction",
        "  - Cross-check legitimate URLs with Google Safe Browsing before use",
        "",
        "NEXT STEPS",
        "  1. Run your feature extractor on dataset_train.csv",
        "  2. Evaluate on dataset_test_standard.csv",
        "  3. Final held-out eval on dataset_test_recent.csv (DO NOT peek!)",
        "  4. For more phishing data: get PhishTank API key from phishtank.org",
        "     and pass PHISHTANK_DB_PATH to load_phishtank_db()",
        "  5. For more legit data: download Tranco top-1m from tranco-list.eu",
        "     and pass TRANCO_CSV_PATH to load_tranco()",
        "=" * 60,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 50)
    log.info("PHISHING DATASET BUILDER — Starting")
    log.info("=" * 50)

    # ── Load optional supplementary data ──
    phishtank_df = None
    if PHISHTANK_DB_PATH and os.path.exists(PHISHTANK_DB_PATH):
        phishtank_df = load_phishtank_db(PHISHTANK_DB_PATH)
        log.info(f"PhishTank DB loaded: {len(phishtank_df):,} entries")

    tranco_domains = None
    if TRANCO_CSV_PATH and os.path.exists(TRANCO_CSV_PATH):
        tranco_domains = load_tranco(TRANCO_CSV_PATH)
        log.info(f"Tranco list loaded: {len(tranco_domains):,} domains")

    phish_df = _empty_dataset_df()
    legit_df = _empty_dataset_df()
    full_df = _empty_dataset_df()
    train_df = _empty_dataset_df()
    test_df = _empty_dataset_df()
    holdout_df = _empty_dataset_df()
    failed_brands: list[str] = []
    try:
        # ── Phase 1: Collect phishing URLs ──
        log.info("\n[ PHASE 1 ] Collecting phishing URLs from PhishStats…")
        phish_df, failed_brands = collect_phishing_urls(BRANDS, phishtank_df)
        phish_df = deduplicate(phish_df)
        phish_df = deduplicate_by_domain(phish_df, max_per_domain=10)
        phish_df.to_csv(f"{OUTPUT_DIR}/raw_phishing.csv", index=False)
        log.info(f"Raw phishing saved: {len(phish_df):,} URLs")

        # ── Phase 2: Collect legitimate URLs ──
        log.info("\n[ PHASE 2 ] Collecting legitimate URLs…")
        legit_df = collect_legitimate_urls(BRANDS, tranco_domains)
        legit_df = deduplicate(legit_df)
        legit_df.to_csv(f"{OUTPUT_DIR}/raw_legitimate.csv", index=False)
        log.info(f"Raw legitimate saved: {len(legit_df):,} URLs")

        # ── Phase 3: Balance ──
        log.info("\n[ PHASE 3 ] Balancing dataset…")
        if not phish_df.empty and not legit_df.empty:
            full_df = balance_dataset(phish_df, legit_df)
        else:
            log.warning("One class is empty (phish=%s legit=%s); using available rows as partial dataset.", len(phish_df), len(legit_df))
            full_df = pd.concat([phish_df, legit_df], ignore_index=True)
        full_df.to_csv(f"{OUTPUT_DIR}/dataset_full.csv", index=False)
        log.info(f"Full balanced dataset saved: {len(full_df):,} rows")

        # ── Phase 4: Split ──
        log.info("\n[ PHASE 4 ] Creating leak-safe splits…")
        if len(full_df) > 0:
            train_df, test_df, holdout_df = make_splits(full_df)
            train_df.to_csv(f"{OUTPUT_DIR}/dataset_train.csv", index=False)
            test_df.to_csv(f"{OUTPUT_DIR}/dataset_test_standard.csv", index=False)
            holdout_df.to_csv(f"{OUTPUT_DIR}/dataset_test_recent.csv", index=False)
        else:
            train_df.to_csv(f"{OUTPUT_DIR}/dataset_train.csv", index=False)
            test_df.to_csv(f"{OUTPUT_DIR}/dataset_test_standard.csv", index=False)
            holdout_df.to_csv(f"{OUTPUT_DIR}/dataset_test_recent.csv", index=False)
    finally:
        # ── Phase 5: Report (always) ──
        report = write_report(full_df, train_df, test_df, holdout_df)
        report_path = f"{OUTPUT_DIR}/collection_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
            if failed_brands:
                f.write("\n\nFAILED BRANDS (PHISH COLLECTION):\n")
                for b in failed_brands:
                    f.write(f"- {b}\n")
        if failed_brands:
            log.warning("Some brands failed during phishing collection: %s", failed_brands)
        print("\n" + report)
        log.info(f"\nAll outputs written to: {OUTPUT_DIR}/")
        log.info("Done.")


if __name__ == "__main__":
    main()
