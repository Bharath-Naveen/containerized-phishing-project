"""DNS features with timeouts and soft failures."""

from __future__ import annotations

import logging
import socket
from typing import Any, Dict, List

import dns.exception
import dns.resolver

logger = logging.getLogger(__name__)

_LIFETIME = 4.0


def _resolve(name: str, rdtype: str) -> List[Any]:
    for attempt in range(3):
        try:
            ans = dns.resolver.resolve(name, rdtype, lifetime=_LIFETIME)
            return list(ans)
        except (dns.exception.DNSException, OSError) as e:
            logger.debug("DNS %s %s attempt %s: %s", rdtype, name, attempt, e)
    return []


def extract_dns_features(hostname: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    host = (hostname or "").strip().lower().split(":")[0].strip("[]")
    if not host or host == "localhost":
        out["dns_resolves"] = 0
        out["a_record_count"] = 0
        out["mx_present"] = 0
        out["ns_count"] = 0
        out["cname_present"] = 0
        out["txt_present"] = 0
        out["ip_count"] = 0
        out["dns_error"] = 1
        out["dns_missing"] = 1
        return out

    out["dns_error"] = 0
    out["dns_missing"] = 0
    a = _resolve(host, "A")
    aaaa = _resolve(host, "AAAA")
    out["a_record_count"] = len(a) + len(aaaa)
    out["dns_resolves"] = int(out["a_record_count"] > 0)
    mx = _resolve(host, "MX")
    out["mx_present"] = int(len(mx) > 0)
    ns = _resolve(host, "NS")
    out["ns_count"] = len(ns)
    cname = _resolve(host, "CNAME")
    out["cname_present"] = int(len(cname) > 0)
    txt = _resolve(host, "TXT")
    out["txt_present"] = int(len(txt) > 0)
    ips = {str(rr) for rr in a + aaaa}
    out["ip_count"] = len(ips)

    if (
        out["a_record_count"] == 0
        and out["cname_present"] == 0
        and out["mx_present"] == 0
        and out["ns_count"] == 0
    ):
        try:
            socket.getaddrinfo(host, None)
            out["dns_resolves"] = 1
            out["dns_error"] = 0
        except OSError:
            out["dns_error"] = 1
    return out
