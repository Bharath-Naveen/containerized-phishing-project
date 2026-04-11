"""Hosting / registration surface features."""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict
from urllib.parse import urlparse

import tldextract

_FREE_HOSTING = (
    "github.io", "gitlab.io", "vercel.app", "netlify.app", "pages.dev",
    "web.app", "firebaseapp.com", "herokuapp.com", "cloudfront.net",
    "azurewebsites.net", "blogspot.com", "wixsite.com", "weebly.com",
)
_CLOUD_HINTS = (
    "amazonaws.com", "cloudflare", "azure", "googleusercontent.com",
    "appspot.com", "digitaloceanspaces.com",
)


def extract_hosting_features(url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (url or "").strip()
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if "@" in host:
        host = host.split("@")[-1]
    port_present = False
    if "]" in host:
        pass
    elif ":" in host:
        hostname_part, port_part = host.rsplit(":", 1)
        if port_part.isdigit():
            port_present = True
            host = hostname_part

    out["port_present"] = int(port_present)
    out["punycode_flag"] = int("xn--" in host)

    ext = tldextract.extract(host)
    out["registered_domain"] = ".".join(p for p in (ext.domain, ext.suffix) if p)
    out["public_suffix"] = ext.suffix or ""

    reg = (out["registered_domain"] or host).lower()
    joined = f"{host} {reg}"
    out["free_hosting_flag"] = int(any(f in joined for f in _FREE_HOSTING))
    out["cloud_hosting_flag"] = int(any(c in joined for c in _CLOUD_HINTS))

    private = 0
    try:
        ip = re.sub(r"^\[|\]$", "", host.split("%")[0])
        if host.startswith("["):
            ipaddress.ip_address(ip)
            private = int(ipaddress.ip_address(ip).is_private)
        elif re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
            private = int(ipaddress.ip_address(host).is_private)
    except ValueError:
        private = 0
    if host in {"localhost", "127.0.0.1"} or host.endswith(".local"):
        private = 1
    out["private_host_flag"] = int(private)
    return out
