#!/usr/bin/env node
/*
 * JS vs Python parity check (rebuild Phase 6).
 *
 *   node demo/parity/run_parity.js <urls.jsonl> <reference.jsonl> <summary.json> [build_dir]
 *
 * Compares, for every URL, the browser engine with the reference written by make_reference.py:
 * all 54 features (must be identical), the primary probability raw and calibrated, the four
 * agreement-model probabilities and votes, the consensus, the host identity, and (when the reference
 * has it) the verdict of the real ML-only dashboard. Writes a summary with the largest differences
 * and the first mismatching rows of each kind. Exit code 1 if any check exceeds its tolerance.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const readline = require("readline");
const { createEngine } = require("../js/phishguard-engine.js");

const TOL_PROB = 1e-6; // stated tolerance for probabilities (absolute)

async function main() {
  const [urlsPath, refPath, outPath, buildDirArg] = process.argv.slice(2);
  const buildDir = buildDirArg || path.join(__dirname, "..", "build");
  const core = JSON.parse(fs.readFileSync(path.join(buildDir, "model-core.json"), "utf8"));
  const eng = createEngine(core);
  const rfBuf = zlib.gunzipSync(fs.readFileSync(path.join(buildDir, core.witnesses.random_forest.file)));
  const ab = rfBuf.buffer.slice(rfBuf.byteOffset, rfBuf.byteOffset + rfBuf.byteLength);
  eng.setRandomForest(ab);

  const urls = new Map();
  for (const line of fs.readFileSync(urlsPath, "utf8").split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line);
    urls.set(r.id, r);
  }
  const feats = core.features;
  const s = {
    rows: 0, by_set: {},
    feature_rows_mismatch: 0, feature_cells_mismatch: 0, feature_mismatch_by_name: {},
    canonical_mismatch: 0,
    max_abs_diff: { p_raw: 0, p_cal: 0, logistic_regression: 0, random_forest: 0, xgboost: 0, lightgbm: 0, contribs: 0 },
    rows_with_any_difference: { p_raw: 0, p_cal: 0, logistic_regression: 0, random_forest: 0, xgboost: 0, lightgbm: 0 },
    p_cal_rounded_mismatch: 0, vote_mismatch: 0, consensus_mismatch: 0, spread_mismatch: 0,
    host_class_mismatch: 0, host_conf_mismatch: 0,
    dashboard_rows: 0, verdict_mismatch: 0, verdict_counts_python: {}, verdict_counts_js: {}, signal_mismatch: 0,
    tolerance_prob_abs: TOL_PROB,
    examples: {},
  };
  function ex(kind, obj) { (s.examples[kind] = s.examples[kind] || []).length < 8 && s.examples[kind].push(obj); }
  const t0 = Date.now();
  const rl = readline.createInterface({ input: fs.createReadStream(refPath, "utf8"), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const ref = JSON.parse(line);
    const u = urls.get(ref.id);
    s.rows++;
    s.by_set[ref.set] = (s.by_set[ref.set] || 0) + 1;
    const a = eng.analyze(u.url);
    if (a.canonical !== ref.canonical) { s.canonical_mismatch++; ex("canonical", { id: ref.id, url: u.url, py: ref.canonical, js: a.canonical }); }
    let rowBad = false;
    for (let i = 0; i < feats.length; i++) {
      const py = ref.features[i], js = a.vector[i];
      if (!(py === js || (Number.isNaN(py) && Number.isNaN(js)))) {
        rowBad = true; s.feature_cells_mismatch++;
        s.feature_mismatch_by_name[feats[i]] = (s.feature_mismatch_by_name[feats[i]] || 0) + 1;
        ex("feature:" + feats[i], { id: ref.id, url: u.url, py, js });
      }
    }
    if (rowBad) s.feature_rows_mismatch++;
    const sc = a.score;
    const d = (k, x, y) => { const v = Math.abs(x - y); if (v > 0) s.rows_with_any_difference[k]++; if (v > s.max_abs_diff[k]) s.max_abs_diff[k] = v; if (v > TOL_PROB) ex("prob:" + k, { id: ref.id, url: u.url, py: x, js: y }); };
    d("p_raw", ref.p_raw, sc.p_raw);
    d("p_cal", ref.p_cal, sc.p_cal);
    if (ref.p_cal_rounded !== sc.p_cal_rounded) { s.p_cal_rounded_mismatch++; ex("p_cal_rounded", { id: ref.id, url: u.url, py: ref.p_cal_rounded, js: sc.p_cal_rounded }); }
    for (const w of sc.witnesses) {
      const py = ref.witness[w.model_name];
      d(w.model_name, py, w.p_exact);
      if ((py >= 0.5) !== w.predicted_phishing) { s.vote_mismatch++; ex("vote", { id: ref.id, url: u.url, model: w.model_name, py, js: w.p_exact }); }
    }
    if (ref.consensus !== sc.agreement.consensus) { s.consensus_mismatch++; ex("consensus", { id: ref.id, url: u.url, py: ref.consensus, js: sc.agreement.consensus }); }
    if (ref.spread !== sc.agreement.spread) {
      s.spread_mismatch++;
      ex("spread", { id: ref.id, py: ref.spread, js: sc.agreement.spread, py_witness: ref.witness, js_witness: sc.witnesses.map((w) => [w.model_name, w.p_exact, w.phish_probability]) });
    }
    for (let i = 0; i < ref.contribs.length; i++) {
      const v = Math.abs(ref.contribs[i] - sc.contribs[i]);
      if (v > s.max_abs_diff.contribs) s.max_abs_diff.contribs = v;
    }
    const hp = a.host_path || {};
    if ((ref.host_identity_class || null) !== (hp.host_identity_class || null)) { s.host_class_mismatch++; ex("host_class", { id: ref.id, url: u.url, py: ref.host_identity_class, js: hp.host_identity_class }); }
    if ((ref.host_legitimacy_confidence || null) !== (hp.host_legitimacy_confidence || null)) { s.host_conf_mismatch++; ex("host_conf", { id: ref.id, url: u.url, py: ref.host_legitimacy_confidence, js: hp.host_legitimacy_confidence }); }
    if (ref.dashboard) {
      s.dashboard_rows++;
      const pv = ref.dashboard.verdict, jv = a.verdict.label;
      s.verdict_counts_python[pv] = (s.verdict_counts_python[pv] || 0) + 1;
      s.verdict_counts_js[jv] = (s.verdict_counts_js[jv] || 0) + 1;
      if (pv !== jv) { s.verdict_mismatch++; ex("verdict", { id: ref.id, set: ref.set, url: u.url, py: pv, js: jv, py_signals: ref.dashboard.phishing_signals, js_signals: a.verdict.phishing_signals }); }
      if (pv === "error") continue;
      const same = (x, y) => JSON.stringify(x || []) === JSON.stringify(y || []);
      if (!same(ref.dashboard.phishing_signals, a.verdict.phishing_signals) || !same(ref.dashboard.legitimacy_signals, a.verdict.legitimacy_signals)
          || !same(ref.dashboard.ambiguity_signals, a.verdict.ambiguity_signals)) {
        s.signal_mismatch++; ex("signals", { id: ref.id, url: u.url, py: [ref.dashboard.phishing_signals, ref.dashboard.legitimacy_signals, ref.dashboard.ambiguity_signals], js: [a.verdict.phishing_signals, a.verdict.legitimacy_signals, a.verdict.ambiguity_signals] });
      }
    }
  }
  s.js_seconds = (Date.now() - t0) / 1000;
  s.pass = s.feature_rows_mismatch === 0 && s.canonical_mismatch === 0 && s.p_cal_rounded_mismatch === 0 && s.vote_mismatch === 0 &&
    s.consensus_mismatch === 0 && s.host_class_mismatch === 0 && s.host_conf_mismatch === 0 && s.verdict_mismatch === 0 &&
    ["p_raw", "p_cal", "logistic_regression", "random_forest", "xgboost", "lightgbm"].every((k) => s.max_abs_diff[k] <= TOL_PROB);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(s, null, 1));
  const brief = Object.assign({}, s); delete brief.examples;
  console.log(JSON.stringify(brief, null, 1));
  process.exit(s.pass ? 0 : 1);
}
main().catch((e) => { console.error(e); process.exit(2); });
