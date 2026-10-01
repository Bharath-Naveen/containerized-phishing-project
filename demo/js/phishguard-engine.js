/*
 * Phishing detection system: Layer-1 URL model and the ML-only verdict, running in the browser.
 *
 * A line-by-line port of the Python code it mirrors (paths are in the phishing repo):
 *   URL canonicalization ........ src/phishguard/urls/safe.py, urls/normalize.py, CPython urllib.parse
 *   domain parsing .............. tldextract 5.3.2 (ICANN public suffix rules shipped in model-core.json)
 *   54 model features ........... src/phishguard/features/{layer1,url_features,hosting_features,brand_signals}.py
 *   models ...................... XGBoost (primary) + isotonic calibrator; Logistic Regression, Random Forest,
 *                                 LightGBM and XGBoost as the 4-model agreement (app/ml_layer1.py)
 *   host reasoning .............. src/phishguard/app/host_path_reasoning.py
 *   verdict ..................... src/phishguard/app/eal.py, evaluated with the inputs the dashboard has when
 *                                 there is no live capture (reinforcement=False). With no capture every
 *                                 page-level input is constant, so only the URL-dependent terms remain.
 * demo/parity/ checks this file against the Python code on a fixed URL set.
 *
 * The URL never leaves the browser: nothing here makes a network request except loading the model files.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PhishGuardEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- Python string semantics
  var U = null; // unicode tables from model-core.json
  var LOWER = null;

  function inRanges(r, cp) {
    var lo = 0, hi = (r.length >> 1) - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (cp < r[2 * mid]) hi = mid - 1;
      else if (cp > r[2 * mid + 1]) lo = mid + 1;
      else return true;
    }
    return false;
  }
  function cps(s) { return Array.from(s); }
  function cpLen(s) {
    var n = 0;
    for (var i = 0; i < s.length; i++) {
      var c = s.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
        var d = s.charCodeAt(i + 1);
        if (d >= 0xdc00 && d <= 0xdfff) i++;
      }
      n++;
    }
    return n;
  }
  function isspace(ch) { return inRanges(U.isspace, ch.codePointAt(0)); }
  function isdigit(ch) { return inRanges(U.isdigit, ch.codePointAt(0)); }
  function isdecimal(ch) { return inRanges(U.isdecimal, ch.codePointAt(0)); }
  function isalnum(ch) { return inRanges(U.isalnum, ch.codePointAt(0)); }
  function isAscii(s) { return /^[\x00-\x7f]*$/.test(s); }
  var CASED = /\p{Cased}/u, CASE_IGN = /\p{Case_Ignorable}/u;

  function pyLower(s) {
    if (isAscii(s)) return s.toLowerCase();
    var a = cps(s), out = "";
    for (var i = 0; i < a.length; i++) {
      var ch = a[i], cp = ch.codePointAt(0);
      if (cp < 128) { out += ch.toLowerCase(); continue; }
      if (cp === 0x3a3) { out += capitalSigma(a, i); continue; }
      var m = LOWER[cp];
      out += m === undefined ? ch : m;
    }
    return out;
  }
  function capitalSigma(a, i) { // CPython handle_capital_sigma
    var j, c = "";
    for (j = i - 1; j >= 0; j--) { c = a[j]; if (!CASE_IGN.test(c)) break; }
    var fin = j >= 0 && CASED.test(c);
    if (fin) {
      for (j = i + 1; j < a.length; j++) { c = a[j]; if (!CASE_IGN.test(c)) break; }
      fin = j === a.length || !CASED.test(c);
    }
    return fin ? "ς" : "σ";
  }
  function pyStrip(s, chars) {
    var a = cps(s), i = 0, j = a.length;
    var test = chars === undefined ? isspace : function (c) { return chars.indexOf(c) >= 0; };
    while (i < j && test(a[i])) i++;
    while (j > i && test(a[j - 1])) j--;
    return a.slice(i, j).join("");
  }
  function pyRstrip(s, chars) {
    var a = cps(s), j = a.length;
    while (j > 0 && chars.indexOf(a[j - 1]) >= 0) j--;
    return a.slice(0, j).join("");
  }
  function countSub(s, sub) { // str.count, non-overlapping
    if (!sub) return cpLen(s) + 1;
    var n = 0, i = 0;
    while ((i = s.indexOf(sub, i)) !== -1) { n++; i += sub.length; }
    return n;
  }
  function partition(s, sep) {
    var i = s.indexOf(sep);
    return i < 0 ? [s, "", ""] : [s.slice(0, i), sep, s.slice(i + sep.length)];
  }
  function rpartition(s, sep) {
    var i = s.lastIndexOf(sep);
    return i < 0 ? ["", "", s] : [s.slice(0, i), sep, s.slice(i + sep.length)];
  }
  function splitOnce(s, sep) { var i = s.indexOf(sep); return [s.slice(0, i), s.slice(i + sep.length)]; }
  function allDecimal(s, minLen, maxLen) {
    var a = cps(s);
    if (a.length < minLen || a.length > maxLen) return false;
    for (var i = 0; i < a.length; i++) if (!isdecimal(a[i])) return false;
    return true;
  }
  // Python re: ^\d{1,3}(\.\d{1,3}){3}$ with Unicode \d (str pattern), used for IPv4-looking hosts.
  function looksDottedQuadUnicode(s) {
    if (s.indexOf("\n") >= 0) {
      // "$" also matches before a single trailing newline in Python; such hosts cannot occur here.
      if (s.endsWith("\n")) s = s.slice(0, -1); else return false;
    }
    var parts = s.split(".");
    if (parts.length !== 4) return false;
    for (var i = 0; i < 4; i++) if (!allDecimal(parts[i], 1, 3)) return false;
    return true;
  }
  function hasLoneSurrogate(s) {
    return /[\ud800-\udbff](?![\udc00-\udfff])|(^|[^\ud800-\udbff])[\udc00-\udfff]/.test(s);
  }

  // ---------------------------------------------------------------- ipaddress (CPython 3.11)
  function parseIPv4(s) {
    if (!s) return null;
    if (s.indexOf("/") >= 0) return null;
    var oct = s.split(".");
    if (oct.length !== 4) return null;
    var v = 0;
    for (var i = 0; i < 4; i++) {
      var o = oct[i];
      if (!o || !/^[0-9]+$/.test(o) || o.length > 3) return null;
      if (o !== "0" && o[0] === "0") return null;
      var n = parseInt(o, 10);
      if (n > 255) return null;
      v = v * 256 + n;
    }
    return v;
  }
  function parseIPv6(s) { // returns array of 8 hextets or null
    if (s.indexOf("/") >= 0) return null;
    var p = partition(s, "%");
    if (p[1] && (!p[2] || p[2].indexOf("%") >= 0)) return null;
    var ip = p[0];
    if (!ip) return null;
    if (cpLen(ip) > 45) return null;
    var parts = ip.split(":");
    if (parts.length > 10) parts = parts.slice(0, 9).concat([parts.slice(9).join(":")]); // split(maxsplit=9)
    if (parts.length < 3) return null;
    if (parts[parts.length - 1].indexOf(".") >= 0) {
      var v4 = parseIPv4(parts.pop());
      if (v4 === null) return null;
      parts.push(((v4 >>> 16) & 0xffff).toString(16));
      parts.push((v4 & 0xffff).toString(16));
    }
    if (parts.length > 9) return null;
    var skip = null, i;
    for (i = 1; i < parts.length - 1; i++) {
      if (!parts[i]) { if (skip !== null) return null; skip = i; }
    }
    var hi, lo, skipped;
    if (skip !== null) {
      hi = skip; lo = parts.length - skip - 1;
      if (!parts[0]) { hi -= 1; if (hi) return null; }
      if (!parts[parts.length - 1]) { lo -= 1; if (lo) return null; }
      skipped = 8 - (hi + lo);
      if (skipped < 1) return null;
    } else {
      if (parts.length !== 8) return null;
      if (!parts[0] || !parts[parts.length - 1]) return null;
      hi = parts.length; lo = 0; skipped = 0;
    }
    function hext(h) {
      if (!/^[0-9A-Fa-f]*$/.test(h)) return null;
      if (h.length > 4) return null;
      if (h === "") return null; // int('', 16) raises
      return parseInt(h, 16);
    }
    var out = [];
    for (i = 0; i < hi; i++) { var x = hext(parts[i]); if (x === null) return null; out.push(x); }
    for (i = 0; i < skipped; i++) out.push(0);
    for (i = parts.length - lo; i < parts.length; i++) { var y = hext(parts[i]); if (y === null) return null; out.push(y); }
    return out;
  }
  // ip_address(): IPv4 first, then IPv6. Returns {v:4,int} | {v:6,h:[8]} | null (ValueError).
  function ipAddress(s) {
    var v4 = parseIPv4(s);
    if (v4 !== null) return { v: 4, n: v4 };
    var v6 = parseIPv6(s);
    if (v6 !== null) return { v: 6, h: v6 };
    return null;
  }
  var V4_PRIVATE = [["0.0.0.0", 8], ["10.0.0.0", 8], ["127.0.0.0", 8], ["169.254.0.0", 16], ["172.16.0.0", 12],
    ["192.0.0.0", 24], ["192.0.0.170", 31], ["192.0.2.0", 24], ["192.168.0.0", 16], ["198.18.0.0", 15],
    ["198.51.100.0", 24], ["203.0.113.0", 24], ["240.0.0.0", 4], ["255.255.255.255", 32]];
  var V4_EXC = [["192.0.0.9", 32], ["192.0.0.10", 32]];
  function v4In(n, net) { var base = parseIPv4(net[0]); var span = Math.pow(2, 32 - net[1]); return n >= base && n < base + span; }
  function v4Private(n) {
    return V4_PRIVATE.some(function (x) { return v4In(n, x); }) && !V4_EXC.some(function (x) { return v4In(n, x); });
  }
  var V6_PRIVATE = [["::1", 128], ["::", 128], ["::ffff:0:0", 96], ["64:ff9b:1::", 48], ["100::", 64], ["2001::", 23],
    ["2001:db8::", 32], ["2002::", 16], ["3fff::", 20], ["fc00::", 7], ["fe80::", 10]];
  var V6_EXC = [["2001:1::1", 128], ["2001:1::2", 128], ["2001:3::", 32], ["2001:4:112::", 48], ["2001:20::", 28], ["2001:30::", 28]];
  function v6In(h, net) {
    var base = parseIPv6(net[0]), bits = net[1];
    for (var i = 0; i < 8; i++) {
      var take = Math.max(0, Math.min(16, bits - 16 * i));
      if (take === 0) return true;
      var mask = (0xffff << (16 - take)) & 0xffff;
      if ((h[i] & mask) !== (base[i] & mask)) return false;
    }
    return true;
  }
  function isPrivateIP(a) {
    if (a.v === 4) return v4Private(a.n);
    var h = a.h;
    if (h[0] === 0 && h[1] === 0 && h[2] === 0 && h[3] === 0 && h[4] === 0 && h[5] === 0xffff) {
      return v4Private(h[6] * 65536 + h[7]);
    }
    return V6_PRIVATE.some(function (x) { return v6In(h, x); }) && !V6_EXC.some(function (x) { return v6In(h, x); });
  }

  // ---------------------------------------------------------------- urllib.parse (CPython 3.11)
  var SCHEME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";
  var USES_PARAMS = ["", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu", "sip", "sips", "mms", "sftp", "tel"];
  var USES_NETLOC = ["", "ftp", "http", "gopher", "nntp", "telnet", "imap", "wais", "file", "mms", "https", "shttp", "snews", "prospero", "rtsp", "rtsps", "rtspu", "rsync", "svn", "svn+ssh", "sftp", "nfs", "git", "git+ssh", "ws", "wss"];

  function ValueError(msg) { var e = new Error(msg); e.pyValueError = true; return e; }

  function urlsplit(url) {
    var k = 0;
    while (k < url.length && url.charCodeAt(k) <= 0x20) k++;
    url = url.slice(k).replace(/[\t\r\n]/g, "");
    var scheme = "", netloc = "", query = "", fragment = "";
    var i = url.indexOf(":");
    if (i > 0 && /^[A-Za-z]/.test(url)) {
      var ok = true;
      for (var j = 0; j < i; j++) if (SCHEME_CHARS.indexOf(url[j]) < 0) { ok = false; break; }
      if (ok) { scheme = url.slice(0, i).toLowerCase(); url = url.slice(i + 1); }
    }
    if (url.slice(0, 2) === "//") {
      var delim = url.length;
      ["/", "?", "#"].forEach(function (c) { var w = url.indexOf(c, 2); if (w >= 0) delim = Math.min(delim, w); });
      netloc = url.slice(2, delim); url = url.slice(delim);
      var ob = netloc.indexOf("[") >= 0, cb = netloc.indexOf("]") >= 0;
      if ((ob && !cb) || (cb && !ob)) throw ValueError("Invalid IPv6 URL");
      if (ob && cb) checkBracketedNetloc(netloc);
    }
    if (url.indexOf("#") >= 0) { var f = splitOnce(url, "#"); url = f[0]; fragment = f[1]; }
    if (url.indexOf("?") >= 0) { var q = splitOnce(url, "?"); url = q[0]; query = q[1]; }
    checkNetloc(netloc);
    return { scheme: scheme, netloc: netloc, path: url, query: query, fragment: fragment };
  }
  function checkBracketedNetloc(netloc) {
    var hp = rpartition(netloc, "@")[2];
    var pb = partition(hp, "["), hostname;
    if (pb[1]) {
      if (pb[0]) throw ValueError("Invalid IPv6 URL");
      var pp = partition(pb[2], "]");
      hostname = pp[0];
      if (pp[2] && pp[2][0] !== ":") throw ValueError("Invalid IPv6 URL");
    } else {
      hostname = partition(hp, ":")[0];
    }
    if (hostname.startsWith("v")) {
      if (!/^v[a-fA-F0-9]+\.[^\n]+$/.test(hostname)) throw ValueError("IPvFuture address is invalid");
    } else {
      var ip = ipAddress(hostname);
      if (!ip) throw ValueError("does not appear to be an IPv4 or IPv6 address");
      if (ip.v === 4) throw ValueError("An IPv4 address cannot be in brackets");
    }
  }
  function checkNetloc(netloc) {
    if (!netloc || isAscii(netloc)) return;
    var n = netloc.split("@").join("").split(":").join("").split("#").join("").split("?").join("");
    var n2 = n.normalize("NFKC");
    if (n === n2) return;
    for (var i = 0; i < 5; i++) if (n2.indexOf("/?#@:"[i]) >= 0) throw ValueError("invalid characters under NFKC normalization");
  }
  function urlparse(url) {
    var s = urlsplit(url), path = s.path, params = "";
    if (USES_PARAMS.indexOf(s.scheme) >= 0 && path.indexOf(";") >= 0) {
      var i;
      if (path.indexOf("/") >= 0) {
        i = path.indexOf(";", path.lastIndexOf("/"));
        if (i < 0) i = -1;
      } else i = path.indexOf(";");
      if (i >= 0) { params = path.slice(i + 1); path = path.slice(0, i); }
    }
    return { scheme: s.scheme, netloc: s.netloc, path: path, params: params, query: s.query, fragment: s.fragment };
  }
  function hostnameOf(netloc) { // SplitResult.hostname
    var hostinfo = rpartition(netloc, "@")[2];
    var pb = partition(hostinfo, "["), hostname;
    if (pb[1]) hostname = partition(pb[2], "]")[0];
    else hostname = partition(hostinfo, ":")[0];
    if (!hostname) return null;
    var z = partition(hostname, "%");
    return pyLower(z[0]) + z[1] + z[2];
  }
  function urlunsplit(scheme, netloc, url, query, fragment) {
    if (netloc || (scheme && USES_NETLOC.indexOf(scheme) >= 0) || url.slice(0, 2) === "//") {
      if (url && url[0] !== "/") url = "/" + url;
      url = "//" + (netloc || "") + url;
    }
    if (scheme) url = scheme + ":" + url;
    if (query) url = url + "?" + query;
    if (fragment) url = url + "#" + fragment;
    return url;
  }
  var HEX = "0123456789ABCDEFabcdef";
  var utf8dec = new TextDecoder("utf-8", { fatal: false });
  var utf8enc = new TextEncoder();
  function unquoteToBytes(s) { // ASCII-only input
    var bits = s.split("%"), out = [];
    function pushStr(t) { for (var i = 0; i < t.length; i++) out.push(t.charCodeAt(i)); }
    pushStr(bits[0]);
    for (var i = 1; i < bits.length; i++) {
      var item = bits[i];
      if (item.length >= 2 && HEX.indexOf(item[0]) >= 0 && HEX.indexOf(item[1]) >= 0) {
        out.push(parseInt(item.slice(0, 2), 16)); pushStr(item.slice(2));
      } else { out.push(37); pushStr(item); }
    }
    return new Uint8Array(out);
  }
  function unquote(s) {
    if (s.indexOf("%") < 0) return s;
    var bits = s.split(/([\x00-\x7f]+)/);
    var res = bits[0];
    for (var i = 1; i < bits.length; i += 2) {
      res += utf8dec.decode(unquoteToBytes(bits[i]));
      res += bits[i + 1];
    }
    return res;
  }
  function unquotePlus(s) { return unquote(s.split("+").join(" ")); }
  function parseQsl(qs, keepBlank) {
    if (!qs) return [];
    var r = [];
    qs.split("&").forEach(function (nv) {
      if (!nv) return;
      var p = partition(nv, "=");
      if (p[2] || keepBlank) r.push([unquotePlus(p[0]), unquotePlus(p[2])]);
    });
    return r;
  }
  var ALWAYS_SAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~";
  function quote(s, safe) {
    if (!s) return s;
    if (hasLoneSurrogate(s)) throw new Error("UnicodeEncodeError");
    var b = utf8enc.encode(s), out = "";
    for (var i = 0; i < b.length; i++) {
      var c = b[i];
      var ch = String.fromCharCode(c);
      if (c < 128 && (ALWAYS_SAFE.indexOf(ch) >= 0 || safe.indexOf(ch) >= 0)) out += ch;
      else out += "%" + (c < 16 ? "0" : "") + c.toString(16).toUpperCase();
    }
    return out;
  }
  function quotePlus(s) {
    if (s.indexOf(" ") < 0) return quote(s, "");
    return quote(s, " ").split(" ").join("+");
  }
  function urlencode(pairs) { return pairs.map(function (kv) { return quotePlus(kv[0]) + "=" + quotePlus(kv[1]); }).join("&"); }

  // urls/safe.py canonicalize_url_safe
  function removeSpaces(s) {
    var a = cps(s), out = "";
    for (var i = 0; i < a.length; i++) {
      var c = a[i];
      if (isspace(c) || c === "​" || c === " ") continue;
      out += c;
    }
    return out;
  }
  function canonicalizeUrlSafe(raw) {
    var s = pyStrip(raw || "");
    if (!s) return ["", 1, "empty"];
    s = removeSpaces(s);
    if (s.indexOf("://") < 0) s = "http://" + s;
    var parts;
    try { parts = urlsplit(s); } catch (e) { if (!e.pyValueError) throw e; return [s, 1, "split_error"]; }
    var scheme = (parts.scheme || "http").toLowerCase();
    var netloc = pyLower(parts.netloc || "");
    if (!netloc) return [s, 1, "missing_host"];
    var path = parts.path || "/";
    if (path.endsWith("/") && path !== "/") path = pyRstrip(path, "/");
    var query = parts.query || "";
    var fragment = parts.fragment || "";
    if (query) {
      try { query = urlencode(parseQsl(query, true)); } catch (e) { /* keep query as is */ }
    }
    var canon = urlunsplit(scheme, netloc, path, query, fragment);
    var invalid = 0, err = "";
    if (scheme !== "http" && scheme !== "https") { invalid = 1; err = "unsupported_scheme"; }
    if (path.indexOf("..") >= 0) { invalid = 1; err = "suspicious_path"; }
    return [canon, invalid, err];
  }
  function featureUrl(raw) {
    var canon = canonicalizeUrlSafe(raw)[0];
    if (!canon) return "";
    var rest = canon.indexOf("://") >= 0 ? splitOnce(canon, "://")[1] : canon;
    return "http://" + rest;
  }
  function originalScheme(raw) {
    var s = pyStrip(raw || "");
    if (s.indexOf("://") < 0) return "";
    return pyLower(pyStrip(splitOnce(s, "://")[0]));
  }
  function safeHostname(url) {
    try { return hostnameOf(urlsplit(url || "").netloc) || ""; } catch (e) { if (!e.pyValueError) throw e; return ""; }
  }
  function netlocPathQuery(url) {
    var p;
    try { p = urlparse(url || ""); } catch (e) { if (!e.pyValueError) throw e; return ["", "", "", "", "err"]; }
    var netloc = pyLower(p.netloc || "");
    if (netloc.indexOf("@") >= 0) { var sp = netloc.split("@"); netloc = sp[sp.length - 1]; }
    return [netloc, p.path || "", p.query || "", (p.scheme || "").toLowerCase(), null];
  }

  // ---------------------------------------------------------------- tldextract 5.3.2 (ICANN rules only)
  var TRIE = null;
  function buildTrie(rules) {
    var rootNode = { m: Object.create(null), end: false };
    rules.split("\n").forEach(function (suffix) {
      if (!suffix) return;
      var labels = suffix.split(".").reverse(), node = rootNode;
      labels.forEach(function (l) {
        if (!(l in node.m)) node.m[l] = { m: Object.create(null), end: false };
        node = node.m[l];
      });
      node.end = true;
    });
    return rootNode;
  }
  var PUNY_BASE = 36, PUNY_TMIN = 1, PUNY_TMAX = 26, PUNY_SKEW = 38, PUNY_DAMP = 700;
  function punyDecode(input) { // RFC 3492; returns null on error
    var output = [], n = 128, i = 0, bias = 72;
    var b = input.lastIndexOf("-");
    if (b < 0) b = 0;
    for (var j = 0; j < b; j++) { if (input.charCodeAt(j) >= 0x80) return null; output.push(input.charCodeAt(j)); }
    for (var idx = b > 0 ? b + 1 : 0; idx < input.length;) {
      var oldi = i, w = 1;
      for (var k = PUNY_BASE; ; k += PUNY_BASE) {
        if (idx >= input.length) return null;
        var cp = input.charCodeAt(idx++);
        var digit = cp - 48 < 10 ? cp - 22 : cp - 65 < 26 ? cp - 65 : cp - 97 < 26 ? cp - 97 : PUNY_BASE;
        if (digit >= PUNY_BASE) return null;
        i += digit * w;
        var t = k <= bias ? PUNY_TMIN : k >= bias + PUNY_TMAX ? PUNY_TMAX : k - bias;
        if (digit < t) break;
        w *= PUNY_BASE - t;
        if (w > 0x7fffffff) return null;
      }
      var len = output.length + 1, delta = i - oldi;
      delta = oldi === 0 ? Math.floor(delta / PUNY_DAMP) : delta >> 1;
      delta += Math.floor(delta / len);
      var kk = 0;
      while (delta > ((PUNY_BASE - PUNY_TMIN) * PUNY_TMAX) >> 1) { delta = Math.floor(delta / (PUNY_BASE - PUNY_TMIN)); kk += PUNY_BASE; }
      bias = Math.floor(kk + ((PUNY_BASE - PUNY_TMIN + 1) * delta) / (delta + PUNY_SKEW));
      n += Math.floor(i / len);
      if (n > 0x10ffff) return null;
      i %= len;
      output.splice(i++, 0, n);
    }
    return String.fromCodePoint.apply(null, output);
  }
  function decodePunycodeLabel(label) {
    var lowered = pyLower(label);
    if (lowered.startsWith("xn--")) {
      var d = punyDecode(lowered.slice(4));
      if (d !== null && d.length) return d;
    }
    return lowered;
  }
  function suffixIndex(spl) {
    var node = TRIE, suffixIdx = spl.length, labelIdx = spl.length;
    for (var k = spl.length - 1; k >= 0; k--) {
      var dl = decodePunycodeLabel(spl[k]);
      if (dl in node.m) {
        labelIdx -= 1; node = node.m[dl];
        if (node.end) suffixIdx = labelIdx;
        continue;
      }
      if ("*" in node.m) {
        var exc = ("!" + dl) in node.m;
        return exc ? labelIdx : labelIdx - 1;
      }
      break;
    }
    if (suffixIdx === spl.length) return null;
    return suffixIdx;
  }
  function schemelessUrl(url) {
    var d = url.indexOf("//");
    if (d === 0) return url.slice(2);
    if (d < 2 || url[d - 1] !== ":") return url;
    var pre = url.slice(0, d - 1);
    for (var i = 0; i < pre.length; i++) if (SCHEME_CHARS.indexOf(pre[i]) < 0) return url;
    return url.slice(d + 2);
  }
  function lenientNetloc(url) {
    var after = rpartition(partition(partition(partition(schemelessUrl(url), "/")[0], "?")[0], "#")[0], "@")[2];
    if (after && after[0] === "[") {
      var mb = partition(after, "]");
      if (mb[1] === "]") return mb[0] + "]";
    }
    var hostname = pyStrip(partition(after, ":")[0]);
    return pyRstrip(hostname, ".。．｡");
  }
  var IP_RE = /^(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$/;
  function tldExtract(url) { // tldextract.extract(url) -> {subdomain, domain, suffix}
    var netloc = lenientNetloc(url).replace(/[。．｡]/g, ".");
    if (cpLen(netloc) >= 4 && netloc[0] === "[" && netloc[netloc.length - 1] === "]" && parseIPv6(netloc.slice(1, -1)) !== null) {
      return { subdomain: "", domain: netloc, suffix: "" };
    }
    var labels = netloc.split(".");
    var psi = suffixIndex(labels);
    if (psi === null && labels.length === 4 && netloc.length && isdecimal(cps(netloc)[0]) && IP_RE.test(netloc)) {
      return { subdomain: "", domain: netloc, suffix: "" };
    }
    if (psi === null) return { subdomain: labels.slice(0, -1).join("."), domain: labels[labels.length - 1], suffix: "" };
    return {
      subdomain: psi >= 2 ? labels.slice(0, psi - 1).join(".") : "",
      domain: psi > 0 ? labels[psi - 1] : "",
      suffix: labels.slice(psi).join("."),
    };
  }
  function registeredDomain(host) {
    var e = tldExtract(host);
    return [e.domain, e.suffix].filter(function (x) { return x; }).join(".");
  }

  // ---------------------------------------------------------------- BLAKE2s (digest 8 bytes) for domain_hash_bucket
  var B2S_IV = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  var B2S_SIGMA = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4], [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13], [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11], [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5], [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]];
  function rotr(x, n) { return (x >>> n) | (x << (32 - n)); }
  function blake2s8(bytes) {
    var outlen = 8;
    var h = B2S_IV.slice();
    h[0] ^= 0x01010000 ^ outlen;
    var t = 0, nblocks = Math.max(1, Math.ceil(bytes.length / 64));
    for (var b = 0; b < nblocks; b++) {
      var last = b === nblocks - 1;
      var block = new Uint8Array(64);
      block.set(bytes.subarray(b * 64, Math.min(bytes.length, (b + 1) * 64)));
      t = last ? bytes.length : (b + 1) * 64;
      var m = new Array(16);
      for (var i = 0; i < 16; i++) m[i] = block[4 * i] | (block[4 * i + 1] << 8) | (block[4 * i + 2] << 16) | (block[4 * i + 3] << 24);
      var v = h.concat(B2S_IV);
      v[12] ^= t >>> 0; v[13] ^= Math.floor(t / 4294967296) >>> 0;
      if (last) v[14] = ~v[14];
      for (var r = 0; r < 10; r++) {
        var s = B2S_SIGMA[r];
        var G = function (a, bb, c, d, x, y) {
          v[a] = (v[a] + v[bb] + x) | 0; v[d] = rotr(v[d] ^ v[a], 16);
          v[c] = (v[c] + v[d]) | 0; v[bb] = rotr(v[bb] ^ v[c], 12);
          v[a] = (v[a] + v[bb] + y) | 0; v[d] = rotr(v[d] ^ v[a], 8);
          v[c] = (v[c] + v[d]) | 0; v[bb] = rotr(v[bb] ^ v[c], 7);
        };
        G(0, 4, 8, 12, m[s[0]], m[s[1]]); G(1, 5, 9, 13, m[s[2]], m[s[3]]);
        G(2, 6, 10, 14, m[s[4]], m[s[5]]); G(3, 7, 11, 15, m[s[6]], m[s[7]]);
        G(0, 5, 10, 15, m[s[8]], m[s[9]]); G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        G(2, 7, 8, 13, m[s[12]], m[s[13]]); G(3, 4, 9, 14, m[s[14]], m[s[15]]);
      }
      for (var q = 0; q < 8; q++) h[q] = (h[q] ^ v[q] ^ v[q + 8]) >>> 0;
    }
    return [h[0] >>> 0, h[1] >>> 0]; // little-endian 8 bytes = h0 | h1 << 32
  }
  function stableDomainBucket(reg) {
    if (!reg) return 0;
    var hh = blake2s8(utf8enc.encode(reg));
    // (h1 * 2^32 + h0) mod 2048 == h0 mod 2048
    return hh[0] % 2048;
  }

  // ---------------------------------------------------------------- features (features/*.py)
  var KEYWORD_FLAGS = [
    ["kw_login", ["login", "signin", "sign-in", "log-in"]], ["kw_verify", ["verify", "verification", "validate"]],
    ["kw_secure", ["secure", "ssl", "safe"]], ["kw_update", ["update", "upgrade"]], ["kw_account", ["account", "profile", "billing"]],
    ["kw_payment", ["payment", "checkout", "invoice"]], ["kw_confirm", ["confirm", "confirmation"]],
    ["kw_password", ["password", "passwd", "reset-password"]]];
  var RISKY_KEYS = ["url", "redirect", "redirect_uri", "return", "return_url", "next", "continue", "dest", "destination", "goto", "rurl", "target", "redir", "link", "out", "forward"];
  var SUSPICIOUS_TOKENS = ["verify", "secure", "update", "suspend", "locked", "unusual", "confirm", "wallet", "invoice", "tax", "refund", "prize", "winner"];
  var FREE_HOSTING = ["github.io", "gitlab.io", "vercel.app", "netlify.app", "pages.dev", "web.app", "firebaseapp.com", "herokuapp.com", "cloudfront.net", "azurewebsites.net", "blogspot.com", "wixsite.com", "weebly.com"];
  var CLOUD_HINTS = ["amazonaws.com", "cloudflare", "azure", "googleusercontent.com", "appspot.com", "digitaloceanspaces.com"];
  var BRAND_TOKENS = ["amazon", "apple", "facebook", "google", "instagram", "linkedin", "microsoft", "microsoftonline", "netflix", "paypal", "whatsapp", "icloud", "outlook", "meta"];
  var OFFICIAL_DOMAIN_LABELS = ["google", "amazon", "apple", "facebook", "paypal", "netflix", "microsoftonline", "live", "office", "outlook", "icloud", "instagram", "whatsapp", "linkedin", "microsoft", "meta", "fb", "okta", "auth0", "duosecurity", "cloudflare", "github", "gitlab", "stripe", "shopify", "atlassian", "slack", "zoom", "box", "docusign", "dropbox", "salesforce"];
  var OFFICIAL_ANCHORS = ["amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.ca", "amazon.in", "amazon.com.au", "amazonaws.com", "apple.com", "icloud.com", "facebook.com", "fb.com", "meta.com", "google.com", "google.de", "google.co.uk", "google.fr", "microsoft.com", "microsoftonline.com", "live.com", "office.com", "outlook.com", "windows.net", "paypal.com", "netflix.com", "linkedin.com", "instagram.com", "whatsapp.com", "okta.com", "auth0.com", "duosecurity.com", "cloudflare.com", "github.com", "gitlab.com", "stripe.com", "shopify.com", "atlassian.net", "slack.com", "zoom.us", "box.com", "docusign.com", "dropbox.com", "salesforce.com"];

  function shannonEntropy(s) {
    var a = cps(s);
    if (!a.length) return 0;
    var counts = new Map();
    a.forEach(function (c) { counts.set(c, (counts.get(c) || 0) + 1); });
    var n = a.length, sum = 0;
    counts.forEach(function (c) { sum += (c / n) * Math.log2(c / n); });
    return -sum;
  }
  function pyRound(x, nd) { // Python round(float, nd): correctly rounded, ties to even on the exact binary value
    if (!isFinite(x) || Math.abs(x) >= 1e21) return x;
    var exact = Math.abs(x).toFixed(100), dot = exact.indexOf(".");
    if (/^50*$/.test(exact.slice(dot + 1 + nd))) {
      var kept = exact.slice(0, dot + 1 + nd), last = kept.charAt(kept.length - 1);
      if (last === ".") last = kept.charAt(kept.length - 2);
      if (parseInt(last, 10) % 2 === 0) return (x < 0 ? -1 : 1) * Number(kept);
    }
    return Number(x.toFixed(nd));
  }
  function pathSegmentCount(path) {
    var p = pyStrip(path || "", "/");
    if (!p) return 0;
    return p.split("/").filter(function (x) { return x; }).length;
  }
  function suspiciousRedirectQueryFlag(query) {
    if (!query) return 0;
    var keys;
    try { keys = parseQsl(query, false).map(function (kv) { return pyStrip(pyLower(kv[0])); }); } catch (e) { return 0; }
    return keys.some(function (k) { return RISKY_KEYS.indexOf(k) >= 0; }) ? 1 : 0;
  }
  function extractUrlFeatures(url) {
    var out = {};
    var raw = pyStrip(url || "");
    out.url_length = cpLen(raw);
    var npq = netlocPathQuery(raw), host = npq[0], path = npq[1], query = npq[2], scheme = npq[3];
    out.hostname_length = cpLen(host);
    out.path_length = cpLen(path);
    out.query_length = cpLen(query);
    out.num_dots = countSub(host, ".");
    out.num_hyphens = countSub(raw, "-");
    var rc = cps(raw), nd = 0;
    rc.forEach(function (c) { if (isdigit(c)) nd++; });
    out.num_digits = nd;
    var ns = 0;
    cps(host).forEach(function (c) { if (!isalnum(c) && c !== "." && c !== "-") ns++; });
    out.num_special_chars = ns;
    var labels = host ? host.split(".") : [];
    out.subdomain_count = labels.length >= 2 ? Math.max(0, labels.length - 2) : 0;
    out.has_ip_address = looksDottedQuadUnicode(host.split(":")[0]) ? 1 : 0;
    out.has_at_symbol = raw.indexOf("@") >= 0 ? 1 : 0;
    out.has_https = scheme === "https" ? 1 : 0;
    out.hostname_entropy = pyRound(shannonEntropy(host), 4);
    var nseg = pathSegmentCount(path);
    out.path_segment_count = nseg;
    out.path_shallow_le1 = nseg <= 1 ? 1 : 0;
    out.path_shallow_le2 = nseg <= 2 ? 1 : 0;
    out.suspicious_redirect_query_flag = suspiciousRedirectQueryFlag(query);
    var blob = pyLower(host + " " + path + " " + query);
    var pqBlob = pyLower(path + " " + query);
    var skc = 0;
    SUSPICIOUS_TOKENS.forEach(function (t) { skc += countSub(blob, t); });
    out.suspicious_keyword_count = skc;
    var hits = 0;
    KEYWORD_FLAGS.forEach(function (kw) {
      out[kw[0]] = kw[1].some(function (w) { return pqBlob.indexOf(w) >= 0; }) ? 1 : 0;
      hits += out[kw[0]];
    });
    out.path_query_authish_keyword_hits = hits;
    out.no_authish_path_query_tokens = hits === 0 ? 1 : 0;
    return out;
  }
  function extractHostingFeatures(url) {
    var out = {};
    var raw = pyStrip(url || "");
    var parsed;
    try { parsed = urlparse(raw); } catch (e) {
      if (!e.pyValueError) throw e;
      return { port_present: 0, punycode_flag: 0, registered_domain: "", public_suffix: "", free_hosting_flag: 0, cloud_hosting_flag: 0, private_host_flag: 0 };
    }
    var host = pyLower(parsed.netloc || "");
    if (host.indexOf("@") >= 0) { var sp = host.split("@"); host = sp[sp.length - 1]; }
    var portPresent = false;
    if (host.indexOf("]") >= 0) { /* pass */ } else if (host.indexOf(":") >= 0) {
      var i = host.lastIndexOf(":"), hp = host.slice(0, i), pp = host.slice(i + 1);
      if (pp.length && cps(pp).every(isdigit)) { portPresent = true; host = hp; }
    }
    out.port_present = portPresent ? 1 : 0;
    out.punycode_flag = host.indexOf("xn--") >= 0 ? 1 : 0;
    var ext = tldExtract(host);
    out.registered_domain = [ext.domain, ext.suffix].filter(function (x) { return x; }).join(".");
    out.public_suffix = ext.suffix || "";
    var reg = pyLower(out.registered_domain || host);
    var joined = host + " " + reg;
    out.free_hosting_flag = FREE_HOSTING.some(function (f) { return joined.indexOf(f) >= 0; }) ? 1 : 0;
    out.cloud_hosting_flag = CLOUD_HINTS.some(function (c) { return joined.indexOf(c) >= 0; }) ? 1 : 0;
    var priv = 0;
    var ip = host.split("%")[0].replace(/^\[/, "").replace(/\]$/, "");
    if (host.startsWith("[")) {
      var a = ipAddress(ip);
      priv = a ? (isPrivateIP(a) ? 1 : 0) : 0;
    } else if (looksDottedQuadUnicode(host)) {
      var b4 = ipAddress(host);
      priv = b4 ? (isPrivateIP(b4) ? 1 : 0) : 0;
    }
    if (host === "localhost" || host === "127.0.0.1" || host.endsWith(".local")) priv = 1;
    out.private_host_flag = priv;
    return out;
  }
  function hostLabels(host) {
    var h = pyStrip(pyLower(host || ""), ".").split(":")[0];
    if (!h || h.startsWith("[")) return [];
    return h.split(".").filter(function (p) { return p; });
  }
  function hyphenatedBrandPrefixDeception(labels) {
    for (var i = 0; i < labels.length; i++) for (var j = 0; j < BRAND_TOKENS.length; j++) {
      var lab = labels[i], b = BRAND_TOKENS[j];
      if (lab.startsWith(b + "-") && cpLen(lab) > cpLen(b) + 1) return 1;
    }
    return 0;
  }
  function typosquatEmbeddedInLabel(labels) {
    for (var i = 0; i < labels.length; i++) {
      var lab = labels[i];
      if (OFFICIAL_DOMAIN_LABELS.indexOf(lab) >= 0) continue;
      for (var j = 0; j < BRAND_TOKENS.length; j++) {
        var b = BRAND_TOKENS[j];
        if (lab === b || lab.startsWith(b + "-")) continue;
        if (lab.startsWith(b) && cpLen(lab) > cpLen(b)) return 1;
      }
    }
    return 0;
  }
  function brandSubstringsIn(s) { var sl = pyLower(s || ""); return BRAND_TOKENS.filter(function (b) { return sl.indexOf(b) >= 0; }); }
  function extractBrandStructureFeatures(host, path, registeredDomainV, freeFlag, cloudFlag) {
    var reg = pyStrip(pyLower(registeredDomainV || ""));
    var pl = pyLower(path || "");
    var labels = hostLabels(host);
    var exact = BRAND_TOKENS.filter(function (b) { return labels.indexOf(b) >= 0; });
    var anchor = reg && OFFICIAL_ANCHORS.indexOf(reg) >= 0 ? 1 : 0;
    var rootLabel = reg && reg.indexOf(".") >= 0 ? reg.split(".")[0] : (reg || "");
    var family = (anchor === 1 || (rootLabel && OFFICIAL_DOMAIN_LABELS.indexOf(rootLabel) >= 0 && reg.split(".").length >= 2)) ? 1 : 0;
    var exactMatch = exact.length > 0 ? 1 : 0;
    var hyph = hyphenatedBrandPrefixDeception(labels);
    var typo = typosquatEmbeddedInLabel(labels);
    var inPath = brandSubstringsIn(pl);
    var pathTok = inPath.length > 0 ? 1 : 0;
    var pathNoOfficial = (pathTok === 1 && anchor === 0 && exactMatch === 0) ? 1 : 0;
    var inHost = brandSubstringsIn(pyLower(host || ""));
    var suspHostCue = (inHost.length > 0 && anchor === 0 && exactMatch === 0 && hyph === 0) ? 1 : 0;
    var any = pathTok || hyph || typo || suspHostCue;
    return {
      official_registrable_anchor: anchor,
      official_domain_family: family,
      brand_hostname_exact_label_match: exactMatch,
      brand_hyphenated_deception_label: hyph,
      brand_typosquat_embedded_in_label: typo,
      path_brand_token_present: pathTok,
      path_brand_without_official_host: pathNoOfficial,
      brand_on_free_hosting: (freeFlag === 1 && any) ? 1 : 0,
      brand_on_cloud_placeholder: (cloudFlag === 1 && any) ? 1 : 0,
      host_brand_substring_not_official: (inHost.length > 0 && anchor === 0) ? 1 : 0,
      num_brand_tokens_in_host: Math.min(3, inHost.length),
      num_brand_tokens_in_path: Math.min(3, inPath.length),
    };
  }
  function hostOnOfficialBrandApex(host) {
    var h = pyStrip(pyLower(host || ""), ".").split(":")[0];
    if (!h) return false;
    var reg = pyLower(registeredDomain(h));
    if (reg && OFFICIAL_ANCHORS.indexOf(reg) >= 0) return true;
    for (var i = 0; i < OFFICIAL_ANCHORS.length; i++) {
      var apex = OFFICIAL_ANCHORS[i];
      if (h === apex || h.endsWith("." + apex)) return true;
    }
    return false;
  }
  // features/layer1.py extract_layer1_features (called with the output of data/clean.canonicalize_url)
  function extractLayer1Features(canonicalInput) {
    var rawIn = pyStrip(canonicalInput || "");
    var c = canonicalizeUrlSafe(rawIn);
    var url = featureUrl(rawIn) || rawIn;
    var row = { canonical_url: c[0] || rawIn };
    Object.assign(row, extractUrlFeatures(url));
    row.has_https = originalScheme(c[0] || rawIn) === "https" ? 1 : 0;
    Object.assign(row, extractHostingFeatures(url));
    var reg = String(row.registered_domain || "");
    delete row.registered_domain;
    var npq = netlocPathQuery(url), host = npq[0], path = npq[1];
    Object.assign(row, extractBrandStructureFeatures(host, path, reg, row.free_hosting_flag | 0, row.cloud_hosting_flag | 0));
    row.domain_hash_bucket = stableDomainBucket(reg);
    var o = row.official_registrable_anchor | 0, e = row.brand_hostname_exact_label_match | 0, hd = row.brand_hyphenated_deception_label | 0;
    row.layer1_brand_trust_score = hd ? o * 2 : o * (2 + e);
    var https = row.has_https | 0;
    row.official_anchor_with_https = https * o;
    row.https_without_official_anchor = https * (1 - o);
    var pl = pyLower(path || "");
    var authAny = ["kw_login", "kw_verify", "kw_account", "kw_password"].some(function (k) { return row[k]; });
    row.legit_auth_surface_on_official_anchor = (o && authAny) ? 1 : 0;
    var adm = ["/admin", "/dashboard", "/console", "/portal", "/wp-admin", "/settings", "/security"].some(function (x) { return pl.indexOf(x) >= 0; });
    row.legit_admin_like_path_on_official_anchor = (o && adm) ? 1 : 0;
    row.legit_checkout_like_path_on_official_anchor = (o && row.kw_payment) ? 1 : 0;
    var nseg = row.path_segment_count;
    var shallow = nseg <= 1 ? 1 : 0;
    var noAuth = row.no_authish_path_query_tokens | 0;
    var noRedir = 1 - (row.suspicious_redirect_query_flag | 0);
    var notIp = 1 - (row.has_ip_address | 0);
    row.simple_public_web_shape = (shallow && noAuth && noRedir && notIp) ? 1 : 0;
    row.simple_official_homepage_shape = (o && shallow && noAuth && noRedir && notIp) ? 1 : 0;
    row.dns_features_skipped = 1;
    row.url_features_missing = 0;
    row.hosting_features_missing = safeHostname(url) ? 0 : 1;
    return row;
  }
  // app/ml_layer1.py build_layer1_frame
  function buildLayer1Row(url) {
    var c = canonicalizeUrlSafe(url);
    var canon = c[0];
    if (c[1] || !canon) canon = pyStrip(url || "");
    var row = extractLayer1Features(canon);
    return { canonical: canon, row: row };
  }

  // ---------------------------------------------------------------- models
  var f32 = Math.fround;
  function scaled(vec, scale) { var o = new Array(vec.length); for (var i = 0; i < vec.length; i++) o[i] = vec[i] / scale[i]; return o; }

  function xgbPredict(model, x64, wantContribs) {
    var x = x64.map(f32), margin = f32(model.base_margin_f32);
    var contrib = wantContribs ? new Float64Array(x.length + 1) : null;
    for (var t = 0; t < model.trees.length; t++) {
      var tr = model.trees[t], n = 0, path = wantContribs ? [] : null;
      while (tr.l[n] !== -1) {
        var fv = x[tr.f[n]], nxt;
        if (fv !== fv) nxt = tr.d[n] ? tr.l[n] : tr.r[n];
        else nxt = fv < tr.v[n] ? tr.l[n] : tr.r[n];
        if (path) path.push([n, nxt]);
        n = nxt;
      }
      margin = f32(margin + tr.v[n]);
      if (contrib) {
        var mv = tr._mean || (tr._mean = nodeMeans(tr));
        contrib[x.length] += mv[0] + (t === 0 ? model.base_margin_f32 : 0);
        for (var k = 0; k < path.length; k++) contrib[tr.f[path[k][0]]] += mv[path[k][1]] - mv[path[k][0]];
      }
    }
    var p = f32(1 / f32(1 + f32(Math.exp(-margin))));
    return { margin: margin, p: p, contribs: contrib };
  }
  function nodeMeans(tr) { // XGBoost FillNodeMeanValues (cover-weighted leaf means)
    var mean = new Float64Array(tr.l.length);
    (function fill(n) {
      if (tr.l[n] === -1) { mean[n] = tr.v[n]; return mean[n]; }
      var r = fill(tr.l[n]) * tr.c[tr.l[n]] + fill(tr.r[n]) * tr.c[tr.r[n]];
      mean[n] = r / tr.c[n];
      return mean[n];
    })(0);
    // The root accumulates as the bias term; children use the same table.
    return mean;
  }
  function lgbmPredict(model, x) {
    var s = 0;
    for (var t = 0; t < model.trees.length; t++) {
      var tr = model.trees[t], n = 0;
      while (tr.f[n] !== -1) {
        var fv = x[tr.f[n]], mt = tr.mt[n];
        if (fv !== fv && mt !== 2) fv = 0.0;
        if ((mt === 1 && fv >= -1e-35 && fv <= 1e-35) || (mt === 2 && fv !== fv)) n = tr.dl[n] ? tr.l[n] : tr.r[n];
        else n = fv <= tr.t[n] ? tr.l[n] : tr.r[n];
      }
      s += tr.v[n];
    }
    return 1 / (1 + Math.exp(-model.sigmoid * s));
  }
  function lrPredict(model, x) {
    var z = 0;
    for (var i = 0; i < x.length; i++) z += x[i] * model.coef[i];
    z += model.intercept;
    return 1 / (1 + Math.exp(-z));
  }
  function decodeRF(buf, header) {
    var off = header.part_offsets, nb = header.part_bytes;
    function view(T, i, size) { return new T(buf, off[i], nb[i] / size); }
    var rf = {
      sizes: view(Uint16Array, 0, 2), kind: view(Uint8Array, 1, 1), feat: view(Uint8Array, 2, 1),
      thr: view(Uint16Array, 3, 2), right: view(Uint16Array, 4, 2), leaf: view(Float64Array, 5, 8), table: view(Float32Array, 6, 4),
    };
    var tabStart = [], acc = 0;
    header.table_sizes.forEach(function (n) { tabStart.push(acc); acc += n; });
    // Per-node pointers into the internal/leaf streams and per-tree node offsets.
    var nNodes = rf.kind.length, iIdx = new Int32Array(nNodes), lIdx = new Int32Array(nNodes), ci = 0, cl = 0;
    for (var n = 0; n < nNodes; n++) { if (rf.kind[n] === 0) iIdx[n] = ci++; else lIdx[n] = cl++; }
    var starts = [], s = 0;
    for (var t = 0; t < rf.sizes.length; t++) { starts.push(s); s += rf.sizes[t]; }
    rf.iIdx = iIdx; rf.lIdx = lIdx; rf.starts = starts; rf.tabStart = tabStart;
    return rf;
  }
  function rfPredict(rf, x64) {
    var x = x64.map(f32), sum = 0;
    for (var t = 0; t < rf.starts.length; t++) {
      var base = rf.starts[t], n = 0;
      while (rf.kind[base + n] === 0) {
        var ii = rf.iIdx[base + n], f = rf.feat[ii];
        var th = rf.table[rf.tabStart[f] + rf.thr[ii]];
        n = x[f] <= th ? n + 1 : n + rf.right[ii];
      }
      sum += rf.leaf[rf.lIdx[base + n]];
    }
    return sum / rf.starts.length;
  }
  function isotonic(cal, p) { // sklearn IsotonicRegression(out_of_bounds="clip") + scipy interp1d linear
    var X = cal.x, Y = cal.y;
    var t = Math.min(Math.max(p, X[0]), X[X.length - 1]);
    var lo = 0, hi = X.length; // searchsorted(side="left")
    while (lo < hi) { var mid = (lo + hi) >> 1; if (X[mid] < t) lo = mid + 1; else hi = mid; }
    var idx = Math.min(Math.max(lo, 1), X.length - 1);
    var xl = X[idx - 1], xh = X[idx], yl = Y[idx - 1], yh = Y[idx];
    var slope = (yh - yl) / (xh - xl);
    var y = slope * (t - xl) + yl;
    return Math.min(Math.max(y, 0), 1);
  }

  // ---------------------------------------------------------------- host_path_reasoning.py (identity + confidence)
  var DOCS_HINTS = ["/docs", "/doc/", "/wiki/", "/kb/", "/knowledge", "/reference/", "/manual/", "/api/"];
  var LURE = ["login", "signin", "logon", "verify", "verification", "secure", "security", "account", "update", "bank", "banking", "wallet", "support", "confirm", "password", "unlock", "billing", "recovery"];
  function hasRunOfDecimals(s, n) {
    var a = cps(s), run = 0;
    for (var i = 0; i < a.length; i++) { if (isdecimal(a[i])) { run++; if (run >= n) return true; } else run = 0; }
    return false;
  }
  function assessHostPath(inputUrl) {
    var target = pyStrip(inputUrl || "");
    var parsed;
    try { parsed = urlparse(target); } catch (e) { if (!e.pyValueError) throw e; return null; }
    var hostname = pyLower(hostnameOf(parsed.netloc) || "");
    var registrable = pyLower(registeredDomain(pyLower(hostname)));
    var h = pyStrip(hostname, "."), r = pyStrip(registrable, ".");
    var sublabels = [];
    if (h && r && h !== r) {
      var left = h.endsWith("." + r) ? h.slice(0, h.length - r.length - 1) : h;
      sublabels = left.split(".").filter(function (x) { return x; });
    }
    var path = parsed.path || "/", pathL = pyLower(path);
    var reasons = [];
    var netloc = pyLower(parsed.netloc || "");
    if (netloc.indexOf("@") >= 0) reasons.push("userinfo_in_netloc");
    if (hostname.startsWith("xn--") || sublabels.some(function (x) { return x.startsWith("xn--"); })) reasons.push("punycode_label_present");
    if (hasRunOfDecimals(hostname, 5)) reasons.push("numeric_heavy_hostname");
    if (sublabels.some(function (x) { return x.indexOf("--") >= 0; })) reasons.push("double_hyphen_subdomain");
    if (countSub(hostname, ".") >= 4) reasons.push("very_deep_subdomain_chain");
    if (netloc.indexOf("%") >= 0) reasons.push("encoded_netloc_sequence");
    if (ipAddress(hostname)) reasons.push("ip_literal_host");
    var reg = pyLower(registrable || hostname);
    for (var i = 0; i < BRAND_TOKENS.length; i++) {
      var tok = BRAND_TOKENS[i];
      if (reg.indexOf(tok) >= 0) continue;
      if (sublabels.some(function (s) { return s.indexOf(tok) >= 0; })) { reasons.push("brand_token_in_subdomain_not_matching_registrable"); break; }
    }
    var regLabel = reg ? reg.split(".")[0] : "";
    if (regLabel && !hostOnOfficialBrandApex(hostname)) {
      if (hyphenatedBrandPrefixDeception([regLabel]) || typosquatEmbeddedInLabel([regLabel])) reasons.push("brand_token_in_registrable_not_official");
      var lure = {};
      regLabel.split("-").forEach(function (t) { if (LURE.indexOf(t) >= 0) lure[t] = 1; });
      if (Object.keys(lure).length >= 2) reasons.push("credential_lure_tokens_in_registrable_label");
    }
    var free = extractHostingFeatures(target).free_hosting_flag === 1;
    var cls = "generic_unknown_host", conf = "medium";
    if (reasons.length) { cls = "suspicious_host_pattern"; conf = "low"; }
    else if (hostOnOfficialBrandApex(hostname)) { cls = "official_brand_auth"; conf = "high"; }
    else if (DOCS_HINTS.some(function (d) { return pathL.indexOf(d) >= 0; })) { cls = "public_docs_or_reference"; conf = free ? "medium" : "high"; }
    else if (registrable && registrable.indexOf(".") >= 0 && !free) conf = "medium";
    else conf = "low";
    return { host_identity_class: cls, host_legitimacy_confidence: conf, host_reasons: reasons, hostname: hostname, registrable_domain: registrable };
  }

  // ---------------------------------------------------------------- eal.py with no live capture
  // Every page-level input is absent in ML-only mode (no capture, no HTML, behavior analysis
  // unavailable, platform context "unknown", hosting trust "unknown"), so the adjudication reduces to
  // the URL-dependent terms below. The names match the signals the Python EAL reports.
  function adjudicateMlOnly(p, cons, spread, hp) {
    var hostIdentity = hp ? hp.host_identity_class : "";
    var hostConf = hp ? hp.host_legitimacy_confidence : "";
    var phish = [], legit = [], amb = [], ps = 0, ls = 0;
    if (p >= 0.85) { ps += 0.35; phish.push("high_ml_score"); }
    else if (p >= 0.70) { ps += 0.20; phish.push("elevated_ml_score"); }
    else if (p >= 0.50) amb.push("moderate_ml_score");
    if (cons === "strong_phishing") { ps += 0.12; phish.push("ml_consensus_strong_phishing"); }
    else if (cons === "strong_legitimate") { ls += 0.12; legit.push("ml_consensus_strong_legitimate"); }
    if (cons === "split") { amb.push("ml_consensus_split"); if (ps < 0.40) ls += 0.06; }
    else if (cons === "boosted_only") amb.push("boosted_models_only_flag_phishing");
    if (typeof spread === "number" && spread > 0.4) amb.push("high_model_probability_spread");
    amb.push("behavior_analysis_unavailable");
    if (hostIdentity === "suspicious_host_pattern") { ps += 0.30; phish.push("suspicious_host_pattern"); }
    amb.push("html_dom_unavailable");
    if (ps >= 0.45 && ls >= 0.45) amb.push("ml_structural_disagreement");
    var evidenceConf = (phish.length ? 1 : 0) + (legit.length ? 1 : 0) + (amb.length ? 1 : 0);
    var missingEvidenceHighRisk = (p >= 0.90 || cons === "strong_phishing") && (hostConf === "low" || hostIdentity === "suspicious_host_pattern");
    if (missingEvidenceHighRisk) { ps += 0.20; phish.push("high_risk_missing_evidence"); }
    var label, rule;
    if (missingEvidenceHighRisk) { label = "likely_phishing"; rule = "missing_evidence_high_risk"; }
    else if (ps >= 0.70 && evidenceConf >= 2) { label = "likely_phishing"; rule = "phishing_score_threshold"; }
    else if (ls >= 0.70 && ps < 0.45) { label = "likely_legitimate"; rule = "legitimacy_score_threshold"; }
    else { label = "uncertain"; rule = "default_uncertain"; }
    return { label: label, rule: rule, phishing_score: Math.round(ps * 1e4) / 1e4, legitimacy_score: Math.round(ls * 1e4) / 1e4, phishing_signals: phish, legitimacy_signals: legit, ambiguity_signals: amb };
  }
  function consensus(outputs) { // ml_layer1.build_model_agreement_from_outputs
    var by = {}; outputs.forEach(function (m) { by[m.model_name] = m; });
    var n = outputs.length, vp = outputs.filter(function (m) { return m.predicted_phishing; }).length, vl = n - vp;
    var q = Math.ceil(0.75 * n);
    var probs = outputs.map(function (m) { return m.phish_probability; });
    var spread = probs.length >= 2 ? Math.max.apply(null, probs) - Math.min.apply(null, probs) : 0;
    if (n < 2) return { consensus: "unavailable", spread: pyRound(spread, 6), votes_phishing: vp, votes_legitimate: vl };
    var boosted = false, lr = by.logistic_regression, rf = by.random_forest;
    if (lr && rf && !lr.predicted_phishing && !rf.predicted_phishing) {
      if ((by.xgboost && by.xgboost.predicted_phishing) || (by.lightgbm && by.lightgbm.predicted_phishing)) boosted = true;
    }
    var c;
    if (boosted) c = "boosted_only";
    else if (vp >= q) c = "strong_phishing";
    else if (vl >= q) c = "strong_legitimate";
    else if (vp >= 1 && vl >= 1) c = "split";
    else c = "unavailable";
    return { consensus: c, spread: pyRound(spread, 6), votes_phishing: vp, votes_legitimate: vl };
  }

  // ---------------------------------------------------------------- engine
  function createEngine(core) {
    U = core.unicode;
    LOWER = core.unicode.lower;
    TRIE = buildTrie(core.psl.rules);
    var feats = core.features, rf = null;
    function vectorOf(row) { return feats.map(function (k) { return Number(row[k]); }); }
    var eng = {
      core: core,
      setRandomForest: function (buf) { rf = decodeRF(buf, core.witnesses.random_forest); },
      hasRandomForest: function () { return rf !== null; },
      features: function (url) { var b = buildLayer1Row(url); b.vector = vectorOf(b.row); return b; },
      scoreVector: function (vec, opts) {
        opts = opts || {};
        var prim = xgbPredict(core.primary.xgboost, scaled(vec, core.scale.primary), !!opts.contribs);
        var pRaw = prim.p;
        var pCal = isotonic(core.primary.calibrator, pRaw);
        var out = { p_raw: pRaw, p_cal: pCal, p_raw_rounded: pyRound(pRaw, 6), p_cal_rounded: pyRound(pCal, 6), margin: prim.margin, contribs: prim.contribs };
        if (opts.witnesses !== false) {
          var w = core.witnesses;
          var lr = lrPredict(w.logistic_regression, scaled(vec, core.scale.logistic_regression));
          var xg = xgbPredict(core.primary.xgboost, scaled(vec, core.scale.xgboost), false).p;
          var lg = lgbmPredict(w.lightgbm, scaled(vec, core.scale.lightgbm));
          var rp = rf ? rfPredict(rf, scaled(vec, core.scale.random_forest)) : null;
          var list = [["logistic_regression", lr], ["random_forest", rp], ["xgboost", xg], ["lightgbm", lg]]
            .filter(function (x) { return x[1] !== null; })
            .map(function (x) { return { model_name: x[0], phish_probability: pyRound(x[1], 6), p_exact: x[1], predicted_phishing: x[1] >= 0.5 }; });
          out.witnesses = list;
          out.agreement = consensus(list);
        }
        return out;
      },
      analyze: function (url) {
        var u = pyStrip(url || "");
        var f = eng.features(u);
        var s = eng.scoreVector(f.vector, { contribs: true });
        var hp = assessHostPath(u);
        // dashboard.py -> capture_signals._enrich_capture_and_html_signals parses the input with urlparse
        // without a guard, so the Python app raises on inputs urlsplit rejects (for example "http://[1.2.3.4]/").
        var appError = null;
        try { urlparse(u.indexOf("://") >= 0 ? u : "https://" + u); } catch (e) { if (!e.pyValueError) throw e; appError = e.message; }
        var verdict = appError !== null ? { label: "error", rule: "app_raises_on_url", error: appError, phishing_signals: [], legitimacy_signals: [], ambiguity_signals: [] } : adjudicateMlOnly(s.p_cal_rounded, s.agreement ? s.agreement.consensus : "", s.agreement ? s.agreement.spread : 0, hp);
        return { input: u, canonical: f.canonical, row: f.row, vector: f.vector, score: s, host_path: hp, verdict: verdict, brand_apex_host: hostOnOfficialBrandApex(safeHostname(f.canonical)) };
      },
      // exposed for tests
      _internals: { canonicalizeUrlSafe: canonicalizeUrlSafe, featureUrl: featureUrl, tldExtract: tldExtract, registeredDomain: registeredDomain, pyRound: pyRound, assessHostPath: assessHostPath, adjudicateMlOnly: adjudicateMlOnly, isotonic: isotonic, pyLower: pyLower, blake2s8: blake2s8 },
    };
    return eng;
  }
  return { createEngine: createEngine };
});
