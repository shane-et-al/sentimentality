#!/usr/bin/env python3
"""Build an interactive HTML visualization from compare_sentiment.py's CSV output.

Reads the comparison CSV (default sentiment_comparison.csv) and writes a single
self-contained HTML file - no internet, no external libraries - with:

    * summary cards: agreement rates, per-tool label distributions, pairwise
      agreement/opposition/rank-correlation stats, and the most common
      whole-row label patterns (click any of them to filter)
    * a score timeline: one panel per tool over all chunks, with each tool's
      neutral zone shaded, hover values and click-to-inspect
    * a chunk-by-chunk heat strip where every chunk is a row and the tools
      are columns; rows where the tools disagree are highlighted, agreeing rows
      fade back, and a left gutter encodes how many labels split
    * a detail panel with the chunk's full text and each tool's exact scores

Note:
    Works best for up to a few thousand chunks; the heat strip shrinks its row
    height to stay inside the browser's maximum canvas size.

Examples:
    Turn the default CSV into sentiment_comparison.html::

        python make_html.py

    Visualize another CSV, choose the output path, label the source in the
    page header, and open the result in a browser when done::

        python make_html.py other.csv --html out.html --source pg5200.txt --open
"""

import argparse
import csv
import json
import sys
import webbrowser

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment tool comparison — VADER · VADER-C · BERT · SiEBERT · RoBERTa</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚖️</text></svg>">
<style>
*{box-sizing:border-box}
body{margin:0;background:#f8f9fa;color:#212529;font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px 18px 70px}
header h1{margin:0;font-size:22px;letter-spacing:-.01em}
.sub{margin:4px 0 12px;color:#868e96;font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:8px 12px;font-size:12px;margin-bottom:16px}
.legend .li{display:inline-flex;align-items:center;gap:6px;background:#fff;border:1px solid #e9ecef;border-radius:999px;padding:3px 10px;color:#495057}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none}
.labchip{width:14px;height:14px;border-radius:4px;display:inline-flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex:none}
.cards{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));margin-bottom:12px}
.card{background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:13px 15px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.card h2{margin:0 0 9px;font-size:11px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#868e96}
.note,.hint{margin:8px 0 0;color:#868e96;font-size:11.5px}
.big{font-size:30px;font-weight:700;line-height:1.1}
.bigsub{color:#868e96;font-size:12.5px}
.segbar{display:flex;height:15px;border-radius:8px;overflow:hidden;margin:12px 0 7px;background:#f1f3f5}
.segbar>div{cursor:pointer}
.segbar>div:hover{outline:2px solid #495057;outline-offset:-2px}
.seglegend{display:flex;gap:12px;font-size:11.5px;color:#495057;flex-wrap:wrap}
.seglegend .sw{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:4px;vertical-align:-1px}
.dist-row{display:grid;grid-template-columns:minmax(64px,76px) 1fr 58px;gap:8px;align-items:center;margin:8px 0;font-size:12.5px}
.dist-row .name{font-weight:600;white-space:nowrap}
.stacked{display:flex;height:15px;border-radius:8px;overflow:hidden;background:#f1f3f5}
.stacked>div{min-width:1px}
.mean{color:#868e96;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
#pairs-wrap{overflow-x:auto}
@media (min-width:900px){.card.wide{grid-column:span 2}}
table.pairs{border-collapse:collapse;width:100%;font-size:12.5px}
table.pairs th{color:#868e96;font-weight:500;text-align:left;padding:3px 6px;border-bottom:1px solid #e9ecef;white-space:nowrap}
table.pairs td{padding:4.5px 6px;border-bottom:1px solid #f1f3f5;font-variant-numeric:tabular-nums;white-space:nowrap}
table.pairs th:nth-child(n+2),table.pairs td:nth-child(n+2){text-align:right}
table.pairs tbody tr{cursor:pointer}
table.pairs tbody tr:hover{background:#f8f9fa}
.pat{display:flex;align-items:center;gap:8px;width:100%;background:none;border:0;border-bottom:1px solid #f1f3f5;padding:5px 2px;cursor:pointer;font:inherit;font-size:12.5px;color:inherit;text-align:left}
.pat:hover{background:#f8f9fa}
.pat-chips{display:inline-flex;gap:3px}
.chip{width:25px;height:17px;border-radius:4px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex:none}
.pat-n{margin-left:auto;color:#868e96;font-variant-numeric:tabular-nums;white-space:nowrap}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:9px}
.controls select,.controls input[type=search]{font:inherit;font-size:12.5px;padding:5px 8px;border:1px solid #dee2e6;border-radius:8px;background:#fff;color:inherit}
.controls input[type=search]{width:190px}
.controls label{display:inline-flex;gap:5px;align-items:center;font-size:12.5px;color:#495057;cursor:pointer}
#count{color:#868e96;font-size:12px;margin-left:auto;white-space:nowrap}
.pill{display:none;align-items:center;gap:6px;background:#e7f5ff;border:1px solid #74c0fc;border-radius:999px;padding:3px 4px 3px 10px;font-size:12px;width:max-content;margin-bottom:9px}
.pill button{border:0;background:none;cursor:pointer;font:inherit;font-weight:700;color:#1864ab;padding:0 6px;border-radius:999px}
.pill button:hover{background:#d0ebff}
.difflegend{display:flex;gap:6px 16px;align-items:center;font-size:11.5px;color:#868e96;margin-bottom:7px;flex-wrap:wrap}
.difflegend .d{display:inline-flex;align-items:center;gap:5px}
.dotmini{width:8px;height:8px;border-radius:50%;display:inline-block}
.dsev{width:17px;height:9px;border-radius:3px;display:inline-block}
.grid-head{position:relative;height:20px;font-size:11px;color:#868e96}
.grid-head span{position:absolute;top:3px;text-align:center;white-space:nowrap;overflow:visible}
.grid-scroll{position:relative;overflow-y:auto;max-height:560px;border:1px solid #dee2e6;border-radius:8px;background:#fff}
#grid{display:block}
#empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#868e96;font-size:13px;pointer-events:none}
#tooltip{position:fixed;display:none;z-index:80;background:#fff;border:1px solid #dee2e6;border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,.15);padding:8px 10px;font-size:11.5px;max-width:460px;pointer-events:none}
#tooltip .tt-line{font-weight:700;margin-bottom:3px}
#tooltip .tt-text{color:#495057;font-size:11px;margin-bottom:5px}
#tooltip .tt-split{color:#c92a2a;font-weight:600}
#tooltip table.ttt{border-collapse:collapse}
#tooltip table.ttt td{padding:1.5px 8px 1.5px 0;font-variant-numeric:tabular-nums;white-space:nowrap}
#detail{position:fixed;top:14px;right:14px;width:404px;max-width:calc(100vw - 28px);max-height:calc(100vh - 28px);overflow:auto;background:#fff;border:1px solid #dee2e6;border-radius:14px;box-shadow:0 10px 34px rgba(0,0,0,.16);padding:12px 14px;z-index:50;display:none}
#detail.open{display:block}
.dnav{display:flex;align-items:center;gap:6px;margin-bottom:2px}
.dnav .title{font-weight:700;font-size:13px;margin-right:2px}
.dnav button{border:1px solid #dee2e6;background:#fff;border-radius:8px;min-width:26px;height:24px;cursor:pointer;font:inherit;line-height:1;padding:0 6px}
.dnav button:hover{background:#f1f3f5}
.dnav .pos{color:#868e96;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.dnav .close{margin-left:auto;font-size:15px}
#detail-text{margin:9px 0;padding:10px;background:#f8f9fa;border-radius:8px;font-size:13px;line-height:1.55;max-height:190px;overflow:auto;white-space:pre-wrap}
.badge{display:inline-block;background:#fff3bf;border:1px solid #ffe066;color:#7a5c00;border-radius:999px;padding:1px 8px;font-size:10.5px;font-weight:600}
.sb{display:grid;grid-template-columns:66px 1fr 132px;gap:8px;align-items:center;margin:7.5px 0}
.sb-name{font-weight:600;font-size:12px;white-space:nowrap}
.sb-track{position:relative;height:10px;background:#f1f3f5;border-radius:5px}
.sb-track::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#495057;opacity:.55}
.sb-fill{position:absolute;top:0;height:100%;border-radius:5px}
.sb-cut{position:absolute;top:-4px;height:18px;width:2px;background:#868e96;opacity:.55;border-radius:1px}
.sb-val{font-size:11.5px;font-variant-numeric:tabular-nums;white-space:nowrap}
.minichip{display:inline-block;padding:1px 6px;border-radius:999px;font-size:9.5px;font-weight:700;margin-right:4px;vertical-align:1px}
.extras{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:11px}
.mini{border:1px solid #e9ecef;border-radius:8px;padding:8px 9px;font-size:11px}
.mini h3{margin:0 0 7px;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#868e96;font-weight:600}
.mini .mtext{margin-top:6px;color:#495057;font-variant-numeric:tabular-nums}
.starrow{display:flex;gap:6px;align-items:flex-end;height:46px}
.starrow .col{display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;min-width:0}
.starrow .bar{width:75%;background:#ffd8a8;border-radius:3px 3px 0 0;min-height:2px}
.starrow .col.top .bar{background:#e8590c}
.starrow .lbl{font-size:9px;color:#868e96;white-space:nowrap}
.hbar{display:flex;height:12px;border-radius:6px;overflow:hidden;background:#f1f3f5}
.agreechips{display:flex;gap:7px;align-items:center;margin-top:11px;flex-wrap:wrap;font-size:11.5px;color:#868e96}
canvas{display:block}
@media (max-width:760px){.extras{grid-template-columns:1fr}.sb{grid-template-columns:56px 1fr 118px}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Sentiment tool comparison</h1>
  <p class="sub" id="meta-line"></p>
  <div class="legend" id="legend"></div>
</header>

<section class="cards">
  <div class="card"><h2>Do the tools agree?</h2><div id="agreement"></div><p class="note">Click a bar segment to show only those chunks.</p></div>
  <div class="card"><h2>Label distribution per tool</h2><div id="dist"></div><p class="note">Each tool's own cutoffs applied; <span class="mean" style="display:inline">mean = mean score.</span></p></div>
  <div class="card wide"><h2>Pairwise agreement</h2><div id="pairs-wrap"></div><p class="note">ρ = Spearman rank correlation of the scores. Click a row to show only chunks where that pair disagrees.</p></div>
  <div class="card"><h2>Common label patterns</h2><div id="patterns"></div><p class="note">Chunks sharing the same labels from every tool, most common first. Order: VADER, VADER-C, BERT, SiEBERT, RoBERTa. Click to filter.</p></div>
</section>

<section class="card">
  <h2>Scores by chunk</h2>
  <canvas id="chart"></canvas>
  <p class="hint">Shaded band = each tool's neutral zone (its label cutoff; RoBERTa picks its own label, so it has none). Hover for values, click a point to inspect that chunk.</p>
</section>

<section class="card">
  <h2>Chunk-by-chunk</h2>
  <div class="controls">
    <select id="mode">
      <option value="all">All chunks</option>
      <option value="notag">Tools disagree (any split)</option>
      <option value="agree">All tools agree</option>
      <option value="minority">Minority report (all but one agree)</option>
      <option value="split2">Split exactly 2 ways</option>
      <option value="split3">Split exactly 3 ways</option>
      <option value="split4">Split 4 different ways</option>
      <option value="neutral">Any tool says neutral</option>
      <option value="trunc">Truncated (over 512 tokens)</option>
    </select>
    <input type="search" id="search" placeholder="Search text or line number…">
    <select id="sort">
      <option value="line">Sort: line order</option>
      <option value="spread">Sort: biggest score spread first</option>
      <option value="len">Sort: longest text first</option>
    </select>
    <label><input type="checkbox" id="dim" checked> dim agreeing rows</label>
    <span id="count"></span>
  </div>
  <div class="pill" id="pill"></div>
  <div class="difflegend" id="difflegend"></div>
  <div class="grid-head" id="grid-head"></div>
  <div class="grid-scroll" id="grid-scroll"><canvas id="grid"></canvas><div id="empty">no chunks match the current filters</div></div>
  <p class="hint">Hover for exact scores · click a row for details · ✂ marks chunks cut off at 512 tokens for the transformer models.</p>
</section>
</div>

<aside id="detail">
  <div class="dnav">
    <span class="title" id="detail-title"></span>
    <button id="detail-prev" title="Previous chunk (↑)">↑</button>
    <button id="detail-next" title="Next chunk (↓)">↓</button>
    <span class="pos" id="detail-pos"></span>
    <button class="close" id="detail-close" title="Close (Esc)">×</button>
  </div>
  <div id="detail-badges"></div>
  <div id="detail-text"></div>
  <div id="detail-scores"></div>
  <div class="extras" id="detail-extras"></div>
  <div class="agreechips" id="detail-agree"></div>
</aside>

<div id="tooltip"></div>
<noscript>This visualization needs JavaScript.</noscript>

<script id="row-data" type="application/json">__DATA__</script>
<script id="meta-data" type="application/json">__META__</script>
<script id="page-js">
"use strict";
const DATA = JSON.parse(document.getElementById("row-data").textContent);
const META = JSON.parse(document.getElementById("meta-data").textContent);

const TOOLS = [
  { key:"v",  name:"VADER",   color:"#1c7ed6", cutoff:0.05, skey:"v",  lkey:"vl"  },
  { key:"vc", name:"VADER-C", color:"#ae3ec9", cutoff:0.05, skey:"vc", lkey:"vcl" },
  { key:"b",  name:"BERT",    color:"#e8590c", cutoff:0.2,  skey:"b",  lkey:"bl"  },
  { key:"s",  name:"SiEBERT", color:"#0c8599", cutoff:0.5,  skey:"s",  lkey:"sl"  },
  /* cutoff null: the model picks its own label, so there is no neutral zone */
  { key:"r",  name:"RoBERTa", color:"#5f3dc4", cutoff:null, skey:"r",  lkey:"rl"  },
];
const NT = TOOLS.length;
const PAIRS = [];
for (let i = 0; i < NT; i++) for (let j = i + 1; j < NT; j++) PAIRS.push([i, j]);

const LAB = {
  pos: { color:"#2f9e44", chipBg:"#d3f9d8", chipFg:"#2b8a3e", fill:"47,158,68",  sym:"+", word:"positive" },
  neu: { color:"#868e96", chipBg:"#e9ecef", chipFg:"#495057", fill:"134,142,150", sym:"0",  word:"neutral"  },
  neg: { color:"#e03131", chipBg:"#ffe3e3", chipFg:"#c92a2a", fill:"224,49,49",  sym:"−", word:"negative" },
};
const SPLIT_FILL = { 1:"rgba(178,242,187,1)", 2:"rgba(255,224,102,1)", 3:"rgba(255,192,120,1)", 4:"rgba(255,168,168,1)" };
const SPLIT_EDGE = { 1:"#69b57e", 2:"#e0b400", 3:"#e06800", 4:"#e03131" };
const SPLIT_NAME = { 1:"all agree", 2:"split 2 ways", 3:"split 3 ways", 4:"split 4 ways" };

const N = DATA.length;
for (const r of DATA) {
  r.L = TOOLS.map(t => r[t.lkey]);
  r.S = TOOLS.map(t => r[t.skey]);
  r.distinct = new Set(r.L).size;
  r.all = r.distinct === 1;
  r.spread = Math.max(...r.S) - Math.min(...r.S);
}
const byLine = new Map(DATA.map(r => [r.l, r]));
const anyTrunc = DATA.some(r => r.tr);

const state = { mode:"all", pattern:null, search:"", sort:"line", dim:true };
let visible = DATA.slice();
let selectedLine = null;

/* ---------- helpers ---------- */
const $ = id => document.getElementById(id);
const fmt = s => (s >= 0 ? "+" : "") + s.toFixed(3);
const pctText = (n, d) => (d ? Math.round(100 * n / d) : 0) + "%";
function el(tag, cls, text){ const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function dpr(){ return window.devicePixelRatio || 1; }
function setupCanvas(c, w, h){
  const k = dpr();
  c.width = Math.max(1, Math.round(w * k));
  c.height = Math.max(1, Math.round(h * k));
  c.style.width = w + "px";
  c.style.height = h + "px";
  const ctx = c.getContext("2d");
  ctx.setTransform(k, 0, 0, k, 0, 0);
  return ctx;
}
function debounce(fn, ms){ let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/* tooltip */
const tip = $("tooltip");
function placeTip(x, y){
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.max(6, Math.min(x + 14, window.innerWidth - w - 10)) + "px";
  tip.style.top = Math.max(6, Math.min(y + 16, window.innerHeight - h - 10)) + "px";
}
function showTip(html, x, y){
  tip.innerHTML = html;
  tip.style.display = "block";
  placeTip(x, y);
}
function moveTip(x, y){ placeTip(x, y); }
function hideTip(){ tip.style.display = "none"; }

function rowTipHTML(r){
  const snippets = r.t.length > 110 ? r.t.slice(0, 110) + "…" : r.t;
  const split = r.all ? `all ${NT} tools agree` : `<span class="tt-split">⚠ labels split ${r.distinct} ways</span>`;
  return `<div class="tt-line">line ${r.l} · ${split}${r.tr ? " · ✂ truncated" : ""}</div>` +
    `<div class="tt-text">${esc(snippets)}</div>` +
    `<table class="ttt">` +
    TOOLS.map((t, i) => `<tr><td><span class="dot" style="background:${LAB[r.L[i]].color}"></span> ${t.name}</td><td style="text-align:right">${fmt(r.S[i])}</td><td style="text-align:right">${r.L[i]}</td></tr>`).join("") +
    `</table>`;
}

/* ---------- filters ---------- */
function passes(r){
  if (state.pattern && r.L.join(",") !== state.pattern) return false;
  const q = state.search.trim().toLowerCase();
  if (q && !(r.t.toLowerCase().includes(q) || String(r.l) === q)) return false;
  switch (state.mode){
    case "agree": return r.all;
    case "notag": return !r.all;
    case "minority": {
      const c = {};
      for (const l of r.L) c[l] = (c[l] || 0) + 1;
      return Math.max(...Object.values(c)) === NT - 1; // all but one agree
    }
    case "split2": return r.distinct === 2;
    case "split3": return r.distinct === 3;
    case "split4": return r.distinct === 4;
    case "trunc": return r.tr;
    case "neutral": return r.L.includes("neu");
  }
  if (state.mode.startsWith("pair-")){
    const [a, b] = PAIRS[+state.mode.slice(5)];
    return r[TOOLS[a].lkey] !== r[TOOLS[b].lkey];
  }
  return true;
}
function computeVisible(){
  visible = DATA.filter(passes);
  if (state.sort === "spread") visible.sort((a, b) => b.spread - a.spread || a.l - b.l);
  else if (state.sort === "len") visible.sort((a, b) => b.t.length - a.t.length || a.l - b.l);
  else visible.sort((a, b) => a.l - b.l);
}

/* ---------- header / legend ---------- */
function renderHeader(){
  const line = [];
  if (META.source) line.push(esc(META.source));
  line.push(`${N} chunks`);
  if (META.csv) line.push(`generated from ${esc(META.csv)}`);
  $("meta-line").textContent = line.join(" · ");
  const lg = $("legend");
  for (const t of TOOLS){
    const li = el("span", "li");
    const d = el("span", "dot"); d.style.background = t.color;
    li.append(d, el("b", null, t.name), el("span", null, t.cutoff == null ? "own label" : `label cutoff ±${t.cutoff}`));
    lg.append(li);
  }
  for (const lab of ["pos", "neu", "neg"]){
    const li = el("span", "li");
    const c = el("span", "labchip"); c.style.background = LAB[lab].chipBg; c.style.color = LAB[lab].chipFg; c.textContent = LAB[lab].sym;
    li.append(c, el("span", null, LAB[lab].word));
    lg.append(li);
  }
  const dl = $("difflegend");
  dl.append(el("span", "d", "left gutter:"));
  const dot = el("span", "dotmini"); dot.style.background = "#868e96";
  dl.append(mkSpanWrap(dot, "dot = all labels equal"));
  for (const k of [2, 3, 4]){
    const s = el("span", "dsev"); s.style.background = SPLIT_FILL[k]; s.style.boxShadow = `inset 0 0 0 1px ${SPLIT_EDGE[k]}`;
    dl.append(mkSpanWrap(s, `${SPLIT_NAME[k]}`));
  }
  if (anyTrunc) dl.append(el("span", "d", "✂ = transformer saw only the first 512 tokens"));
}
function mkSpanWrap(swatch, text){ const w = el("span", "d"); w.append(swatch, el("span", null, text)); return w; }

/* ---------- dashboard ---------- */
function renderAgreement(){
  const counts = { 1:0, 2:0, 3:0, 4:0 };
  let minority = 0;
  for (const r of DATA){
    counts[r.distinct]++;
    const c = {};
    for (const l of r.L) c[l] = (c[l] || 0) + 1;
    if (Math.max(...Object.values(c)) === NT - 1) minority++; // all but one agree
  }
  const ag = $("agreement");
  ag.innerHTML = "";
  const big = el("div");
  big.append(el("span", "big", pctText(counts[1], N)), el("span", "bigsub", ` of chunks — all ${NT} tools give the same label (${counts[1]} of ${N})`));
  const minSpan = el("span", "bigsub");
  minSpan.style.display = "block";
  const minBtn = el("a");
  minBtn.href = "#";
  minBtn.textContent = `${minority} minority report${minority === 1 ? "" : "s"} — ${NT - 1} tools agree, 1 differs`;
  minBtn.addEventListener("click", e => { e.preventDefault(); setMode("minority"); });
  minSpan.append(minBtn);
  big.append(minSpan);
  ag.append(big);
  const bar = el("div", "segbar");
  for (const k of [1, 2, 3, 4]){
    if (!counts[k]) continue;
    const seg = el("div");
    seg.style.width = (100 * counts[k] / N) + "%";
    seg.style.background = SPLIT_FILL[k];
    seg.style.boxShadow = `inset -1px 0 0 rgba(255,255,255,.7)`;
    seg.title = `${SPLIT_NAME[k]}: ${counts[k]} chunks (${pctText(counts[k], N)}) — click to filter`;
    seg.addEventListener("click", () => { setMode(k === 1 ? "all" : "split" + k); });
    bar.append(seg);
  }
  ag.append(bar);
  const leg = el("div", "seglegend");
  for (const k of [1, 2, 3, 4]){
    const s = el("span");
    const sw = el("span", "sw"); sw.style.background = SPLIT_FILL[k];
    s.append(sw, el("span", null, `${SPLIT_NAME[k]}: ${counts[k]}`));
    leg.append(s);
  }
  ag.append(leg);
}

function renderDist(){
  const host = $("dist");
  host.innerHTML = "";
  for (const t of TOOLS){
    const c = { pos:0, neu:0, neg:0 };
    for (const r of DATA) c[r[t.lkey]]++;
    const mean = DATA.reduce((a, r) => a + r[t.skey], 0) / N;
    const row = el("div", "dist-row");
    const name = el("span", "name", t.name); name.style.color = t.color;
    const st = el("div", "stacked");
    for (const lab of ["pos", "neu", "neg"]){
      if (!c[lab]) continue;
      const seg = el("div");
      seg.style.width = (100 * c[lab] / N) + "%";
      seg.style.background = LAB[lab].color;
      seg.title = `${t.name}: ${c[lab]} ${lab} (${pctText(c[lab], N)})`;
      st.append(seg);
    }
    row.append(name, st, el("span", "mean", "mean " + fmt(mean)));
    host.append(row);
  }
}

function rank(a){
  const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
  const out = new Array(a.length);
  let i = 0;
  while (i < idx.length){
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[idx[k][1]] = r;
    i = j + 1;
  }
  return out;
}
function pearson(x, y){
  const n = x.length;
  if (n < 2) return NaN;
  const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++){
    num += (x[i] - mx) * (y[i] - my);
    dx += (x[i] - mx) ** 2;
    dy += (y[i] - my) ** 2;
  }
  const den = Math.sqrt(dx * dy);
  return den ? num / den : NaN;
}
function spearman(a, b){ return pearson(rank(a), rank(b)); }

function renderPairs(){
  const wrap = $("pairs-wrap");
  wrap.innerHTML = "";
  const table = el("table", "pairs");
  const thead = el("thead");
  thead.innerHTML = "<tr><th>pair</th><th title='chunks with the same label'>agree</th><th title='chunks labeled pos by one and neg by the other'>opposite</th><th title='chunks where exactly one of the pair says neutral'>one neutral</th><th title='Spearman rank correlation of the scores'>ρ</th></tr>";
  table.append(thead);
  const tb = el("tbody");
  for (const [a, b] of PAIRS){
    const ta = TOOLS[a], tb2 = TOOLS[b];
    let agree = 0, opp = 0;
    for (const r of DATA){
      const la = r[ta.lkey], lb = r[tb2.lkey];
      if (la === lb) agree++;
      else if ((la === "pos" && lb === "neg") || (la === "neg" && lb === "pos")) opp++;
    }
    const rho = spearman(DATA.map(r => r[ta.skey]), DATA.map(r => r[tb2.skey]));
    const tr = el("tr");
    tr.innerHTML =
      `<td><b>${esc(ta.name)}</b> / <b>${esc(tb2.name)}</b></td>` +
      `<td>${agree} (${pctText(agree, N)})</td><td>${opp}</td>` +
      `<td>${N - agree - opp}</td><td>${isNaN(rho) ? "—" : (rho >= 0 ? "+" : "") + rho.toFixed(2)}</td>`;
    tr.title = `Show only chunks where ${ta.name} and ${tb2.name} disagree`;
    tr.addEventListener("click", () => setMode("pair-" + PAIRS.findIndex(p => p[0] === a && p[1] === b)));
    tb.append(tr);
  }
  table.append(tb);
  wrap.append(table);
}

function renderPatterns(){
  const host = $("patterns");
  host.innerHTML = "";
  const counts = new Map();
  for (const r of DATA){
    const k = r.L.join(",");
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  for (const [k, cnt] of top){
    const btn = el("button", "pat");
    btn.type = "button";
    const chips = el("span", "pat-chips");
    k.split(",").forEach((lab, t) => {
      const c = el("span", "chip");
      c.style.background = LAB[lab].chipBg;
      c.style.color = LAB[lab].chipFg;
      c.textContent = LAB[lab].sym;
      c.title = `${TOOLS[t].name}: ${lab}`;
      chips.append(c);
    });
    btn.append(chips, el("span", "pat-n", `${cnt} · ${pctText(cnt, N)}`));
    btn.title = k === state.pattern ? "Clear this pattern filter" : "Show only chunks with this pattern";
    btn.addEventListener("click", () => {
      state.pattern = state.pattern === k ? null : k;
      renderPatterns();
      updatePill();
      computeVisible();
      renderGrid();
      updateCount();
    });
    host.append(btn);
  }
}

/* ---------- controls ---------- */
function setMode(m){
  state.mode = m;
  $("mode").value = m;
  computeVisible();
  renderGrid();
  updateCount();
}
function updateCount(){
  $("count").textContent = `Showing ${visible.length} of ${N} chunks`;
  $("empty").style.display = visible.length ? "none" : "flex";
}
function updatePill(){
  const pill = $("pill");
  if (!state.pattern){ pill.style.display = "none"; return; }
  pill.style.display = "flex";
  pill.innerHTML = "<span>pattern filter:</span>";
  state.pattern.split(",").forEach((lab, t) => {
    const c = el("span", "chip");
    c.style.background = LAB[lab].chipBg;
    c.style.color = LAB[lab].chipFg;
    c.textContent = LAB[lab].sym;
    c.title = `${TOOLS[t].name}: ${lab}`;
    pill.append(c);
  });
  const x = el("button", null, "clear ×");
  x.addEventListener("click", () => {
    state.pattern = null;
    renderPatterns();
    updatePill();
    computeVisible();
    renderGrid();
    updateCount();
  });
  pill.append(x);
}

/* ---------- chunk grid ---------- */
const gridScroll = $("grid-scroll"), grid = $("grid"), gridHead = $("grid-head");
let colX = [], colW = 0, gridW = 800, effRowH = 16;

function rowHeightFor(rows){
  const MAX_CANVAS_H = 30000;
  if (!rows.length) return 16;
  return Math.max(10, Math.min(18, Math.floor(MAX_CANVAS_H / rows.length)));
}
function layoutGrid(){
  gridW = Math.max(320, gridScroll.clientWidth);
  effRowH = rowHeightFor(visible);
  const truncW = anyTrunc ? 18 : 0;
  const body = gridW - 46 - 26 - truncW;
  colW = body / NT;
  colX = TOOLS.map((_, i) => 46 + 26 + i * colW);
  gridHead.style.width = gridW + "px";
  gridHead.innerHTML = "";
  const mk = (left, width, text, cls, color) => {
    const s = el("span", cls, text);
    s.style.left = left + "px";
    s.style.width = width + "px";
    if (color) s.style.color = color;
    gridHead.append(s);
    return s;
  };
  mk(0, 46, "line", "gline");
  mk(46, 26, "diff", "");
  TOOLS.forEach((t, i) => mk(colX[i], colW, t.name, "", t.color));
}
function cellFill(lab, s){
  const m = Math.min(1, Math.abs(s));
  if (lab === "neu") return `rgba(${LAB.neu.fill},${(0.30 + 0.34 * m).toFixed(3)})`;
  const a = (0.20 + 0.74 * m).toFixed(3);
  return `rgba(${LAB[lab].fill},${a})`;
}
function renderGrid(){
  const rows = visible;
  const h = rows.length * effRowH;
  const ctx = setupCanvas(grid, gridW, Math.max(h, 1));
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";
  ctx.clearRect(0, 0, gridW, h);
  for (let i = 0; i < rows.length; i++){
    const r = rows[i], y = i * effRowH;
    ctx.save();
    if (state.dim && r.all) ctx.globalAlpha = 0.42;
    if (i % 2 === 1){ ctx.fillStyle = "#fcfcfd"; ctx.fillRect(0, y, gridW, effRowH); }
    if (effRowH >= 12){
      ctx.fillStyle = "#adb5bd";
      ctx.textAlign = "right";
      ctx.fillText(String(r.l), 40, y + effRowH / 2 + 0.5);
    }
    /* left gutter: agreement / split marker */
    if (r.all){
      ctx.fillStyle = `rgba(${LAB[r.L[0]].fill},0.55)`;
      ctx.beginPath();
      ctx.arc(46 + 13, y + effRowH / 2, 2.6, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const k = Math.min(4, r.distinct);
      const w = 7 + (k - 2) * 6;
      ctx.fillStyle = SPLIT_FILL[k];
      ctx.strokeStyle = SPLIT_EDGE[k];
      roundRect(ctx, 46 + 4, y + (effRowH - 9) / 2, w, 9, 3);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    if (r.tr && anyTrunc){
      ctx.fillStyle = "#7048e8";
      ctx.textAlign = "center";
      ctx.fillText("✂", gridW - 10, y + effRowH / 2);
    }
    for (let t = 0; t < NT; t++){
      ctx.fillStyle = cellFill(r.L[t], r.S[t]);
      ctx.fillRect(colX[t] + 1.5, y + 2.5, colW - 3, effRowH - 5);
    }
    ctx.restore();
  }
  if (selectedLine != null){
    const i = rows.findIndex(r => r.l === selectedLine);
    if (i >= 0){
      ctx.strokeStyle = "#1971c2";
      ctx.lineWidth = 2;
      ctx.strokeRect(1, i * effRowH + 1, gridW - 2, effRowH - 2);
    }
  }
}
function roundRect(ctx, x, y, w, h, rr){
  ctx.beginPath();
  if (ctx.roundRect){ ctx.roundRect(x, y, w, h, rr); return; }
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

let hoverIdx = -1;
grid.addEventListener("mousemove", e => {
  const i = Math.floor(e.offsetY / effRowH);
  if (i < 0 || i >= visible.length){ hideTip(); hoverIdx = -1; return; }
  if (i !== hoverIdx){ hoverIdx = i; showTip(rowTipHTML(visible[i]), e.clientX, e.clientY); }
  else moveTip(e.clientX, e.clientY);
});
grid.addEventListener("mouseleave", () => { hideTip(); hoverIdx = -1; });
grid.addEventListener("click", e => {
  const i = Math.floor(e.offsetY / effRowH);
  if (i >= 0 && i < visible.length) selectLine(visible[i].l, false);
});

/* ---------- score timeline ---------- */
const chart = $("chart");
const PANEL_H = 86, PANEL_GAP = 14, TOP_PAD = 4, BOT_PAD = 24;
let chartW = 800, chartHover = -1;

function chartGeom(){
  const H = TOP_PAD + NT * (PANEL_H + PANEL_GAP) - PANEL_GAP + BOT_PAD;
  const x0 = 8, x1 = Math.max(x0 + 1, chartW - 8);
  return { H, x0, x1, X: i => (N <= 1 ? x0 : x0 + (x1 - x0) * (i / (N - 1))) };
}
function drawChart(hoverI){
  const { H, x0, x1, X } = chartGeom();
  const ctx = setupCanvas(chart, chartW, H);
  ctx.clearRect(0, 0, chartW, H);
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  const selI = selectedLine == null ? -1 : DATA.findIndex(r => r.l === selectedLine);
  TOOLS.forEach((t, k) => {
    const top = TOP_PAD + k * (PANEL_H + PANEL_GAP);
    const pTop = top + 17, pBot = top + PANEL_H - 6;
    const yOf = v => pBot - (v + 1) / 2 * (pBot - pTop);
    ctx.fillStyle = "#fbfcfe";
    ctx.fillRect(x0, top, x1 - x0, PANEL_H);
    /* neutral band (per-tool label cutoff) */
    if (t.cutoff != null){
      ctx.fillStyle = "rgba(134,142,150,0.10)";
      ctx.fillRect(x0, yOf(t.cutoff), x1 - x0, yOf(-t.cutoff) - yOf(t.cutoff));
    }
    /* gridlines */
    for (const v of [-1, -0.5, 0, 0.5, 1]){
      ctx.strokeStyle = v === 0 ? "#dee2e6" : "#f1f3f5";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, yOf(v) + 0.25);
      ctx.lineTo(x1, yOf(v) + 0.25);
      ctx.stroke();
    }
    /* labels */
    ctx.textAlign = "left";
    ctx.fillStyle = t.color;
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.fillText(t.name, x0 + 6, top + 12);
    ctx.font = "9.5px system-ui, sans-serif";
    ctx.fillStyle = "#adb5bd";
    ctx.textAlign = "right";
    ctx.fillText(t.cutoff == null ? "label: most likely class" : `neutral: |score| < ${t.cutoff}`, x1 - 6, top + 12);
    /* score line */
    ctx.strokeStyle = t.color;
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (let i = 0; i < N; i++){
      const px = X(i), py = yOf(DATA[i][t.skey]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
  /* hover crosshair */
  if (hoverI >= 0){
    const px = X(hoverI);
    ctx.strokeStyle = "rgba(33,37,41,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, TOP_PAD);
    ctx.lineTo(px, TOP_PAD + NT * PANEL_H + (NT - 1) * PANEL_GAP);
    ctx.stroke();
    TOOLS.forEach((t, k) => {
      const top = TOP_PAD + k * (PANEL_H + PANEL_GAP);
      const pTop = top + 17, pBot = top + PANEL_H - 6;
      const yOf = v => pBot - (v + 1) / 2 * (pBot - pTop);
      ctx.fillStyle = t.color;
      ctx.strokeStyle = "#fff";
      ctx.beginPath();
      ctx.arc(X(hoverI), yOf(DATA[hoverI][t.skey]), 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }
  /* selection marker */
  if (selI >= 0){
    const px = X(selI);
    ctx.strokeStyle = "rgba(25,113,194,0.45)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, TOP_PAD);
    ctx.lineTo(px, TOP_PAD + NT * PANEL_H + (NT - 1) * PANEL_GAP);
    ctx.stroke();
    TOOLS.forEach((t, k) => {
      const top = TOP_PAD + k * (PANEL_H + PANEL_GAP);
      const pTop = top + 17, pBot = top + PANEL_H - 6;
      const yOf = v => pBot - (v + 1) / 2 * (pBot - pTop);
      ctx.fillStyle = "#1971c2";
      ctx.strokeStyle = "#fff";
      ctx.beginPath();
      ctx.arc(px, yOf(DATA[selI][t.skey]), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }
  /* x labels */
  ctx.fillStyle = "#adb5bd";
  ctx.font = "9.5px ui-monospace, monospace";
  ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(N / 7));
  for (let i = 0; i < N; i += step){
    ctx.fillText("line " + DATA[i].l, X(i), H - 7);
  }
}
chart.addEventListener("mousemove", e => {
  const { x0, x1, X } = chartGeom();
  const x = e.clientX - chart.getBoundingClientRect().left;
  let i = N <= 1 ? 0 : Math.round((x - x0) / (x1 - x0) * (N - 1));
  i = Math.max(0, Math.min(N - 1, i));
  if (i !== chartHover){
    chartHover = i;
    drawChart(i);
    const r = DATA[i];
    const sn = r.t.length > 100 ? r.t.slice(0, 100) + "…" : r.t;
    showTip(
      `<div class="tt-line">line ${r.l}</div><div class="tt-text">${esc(sn)}</div>` +
      `<table class="ttt">${TOOLS.map(t => `<tr><td><span class="dot" style="background:${t.color}"></span>${t.name}</td><td style="text-align:right">${fmt(r[t.skey])}</td><td style="text-align:right">${r[t.lkey]}</td></tr>`).join("")}</table>`,
      e.clientX, e.clientY);
  } else moveTip(e.clientX, e.clientY);
});
chart.addEventListener("mouseleave", () => { if (chartHover >= 0){ chartHover = -1; drawChart(); } hideTip(); });
chart.addEventListener("click", () => { if (chartHover >= 0) selectLine(DATA[chartHover].l, true); });

/* ---------- detail panel ---------- */
const detail = $("detail");
function navList(){
  return visible.some(r => r.l === selectedLine) ? visible : DATA;
}
function selectLine(line, scrollTo){
  selectedLine = line;
  openDetail(byLine.get(line));
  renderGrid();
  drawChart();
  if (scrollTo){
    const i = visible.findIndex(r => r.l === line);
    if (i >= 0){
      const y = i * effRowH;
      if (y < gridScroll.scrollTop || y > gridScroll.scrollTop + gridScroll.clientHeight - effRowH)
        gridScroll.scrollTop = Math.max(0, y - gridScroll.clientHeight / 2);
    }
  }
}
function scoreBar(t, r){
  const row = el("div", "sb");
  const name = el("span", "sb-name", t.name);
  name.style.color = t.color;
  const track = el("div", "sb-track");
  for (const c of t.cutoff == null ? [] : [t.cutoff, -t.cutoff]){
    const cut = el("div", "sb-cut");
    cut.style.left = `calc(${50 + c * 50}% - 1px)`;
    cut.title = `${t.name} label cutoff ±${t.cutoff}`;
    track.append(cut);
  }
  const s = r[t.skey];
  const fill = el("div", "sb-fill");
  fill.style.background = LAB[r[t.lkey]].color;
  if (s >= 0){ fill.style.left = "50%"; fill.style.width = (s * 50) + "%"; }
  else { fill.style.left = (50 + s * 50) + "%"; fill.style.width = (-s * 50) + "%"; }
  track.append(fill);
  const val = el("span", "sb-val");
  const chip = el("span", "minichip", r[t.lkey]);
  chip.style.background = LAB[r[t.lkey]].chipBg;
  chip.style.color = LAB[r[t.lkey]].chipFg;
  val.append(chip, el("span", null, fmt(s)));
  row.append(name, track, val);
  return row;
}
function miniMini(titleText, build){
  const m = el("div", "mini");
  m.append(el("h3", null, titleText));
  build(m);
  return m;
}
function openDetail(r){
  if (!r) return;
  $("detail-title").textContent = "line " + r.l;
  const bd = $("detail-badges");
  bd.innerHTML = "";
  if (r.tr){
    const b = el("span", "badge", "✂ truncated: the transformer models saw only the first 512 tokens; VADER read the whole line");
    bd.append(b);
  }
  const tx = $("detail-text");
  tx.textContent = r.t;
  const sc = $("detail-scores");
  sc.innerHTML = "";
  TOOLS.forEach(t => sc.append(scoreBar(t, r)));
  const ex = $("detail-extras");
  ex.innerHTML = "";
  /* BERT stars */
  ex.append(miniMini("BERT star probabilities", m => {
    const wrap = el("div", "starrow");
    const mx = Math.max(...r.bp, 0.05);
    for (let i = 1; i <= 5; i++){
      const col = el("div", "col" + (i === r.st ? " top" : ""));
      const bar = el("div", "bar");
      bar.style.height = Math.max(3, r.bp[i - 1] / mx * 30) + "px";
      bar.title = `${i}★: ${(r.bp[i - 1] * 100).toFixed(1)}%`;
      col.append(bar, el("span", "lbl", i + "★"));
      wrap.append(col);
    }
    m.append(wrap, el("div", "mtext", `top: ${r.st}★ (${(r.bp[r.st - 1] * 100).toFixed(0)}%)`));
  }));
  /* VADER shares */
  ex.append(miniMini("VADER word shares", m => {
    const bar = el("div", "hbar");
    const parts = [["pos", r.vp, "positive words"], ["neu", r.vu, "neutral"], ["neg", r.vn, "negative words"]];
    for (const [lab, v, word] of parts){
      const seg = el("div");
      seg.style.width = (v * 100) + "%";
      seg.style.background = LAB[lab].color;
      seg.title = `${word}: ${(v * 100).toFixed(1)}%`;
      bar.append(seg);
    }
    m.append(bar, el("div", "mtext", `pos ${(r.vp * 100).toFixed(0)}% · neu ${(r.vu * 100).toFixed(0)}% · neg ${(r.vn * 100).toFixed(0)}%`));
  }));
  /* SiEBERT certainty */
  ex.append(miniMini("SiEBERT certainty", m => {
    const bar = el("div", "hbar");
    const pos = el("div"); pos.style.width = (r.sp * 100) + "%"; pos.style.background = LAB.pos.color;
    const neg = el("div"); neg.style.width = ((1 - r.sp) * 100) + "%"; neg.style.background = LAB.neg.color;
    bar.append(pos, neg);
    m.append(bar, el("div", "mtext", `P(positive) = ${(r.sp * 100).toFixed(1)}%`));
  }));
  /* RoBERTa class probabilities */
  ex.append(miniMini("RoBERTa probabilities", m => {
    const bar = el("div", "hbar");
    const parts = [["pos", r.rp[2], "positive"], ["neu", r.rp[1], "neutral"], ["neg", r.rp[0], "negative"]];
    for (const [lab, v, word] of parts){
      const seg = el("div");
      seg.style.width = (v * 100) + "%";
      seg.style.background = LAB[lab].color;
      seg.title = `${word}: ${(v * 100).toFixed(1)}%`;
      bar.append(seg);
    }
    m.append(bar, el("div", "mtext", `pos ${(r.rp[2] * 100).toFixed(0)}% · neu ${(r.rp[1] * 100).toFixed(0)}% · neg ${(r.rp[0] * 100).toFixed(0)}%`));
  }));
  $("detail-agree").innerHTML = "";
  const ac = $("detail-agree");
  TOOLS.forEach((t, idx) => {
    const c = el("span", "minichip", t.name.slice(0, 6));
    c.style.background = LAB[r.L[idx]].chipBg;
    c.style.color = LAB[r.L[idx]].chipFg;
    c.title = `${t.name}: ${r.L[idx]} (${fmt(r.S[idx])})`;
    ac.append(c);
  });
  const note = r.all ? `all ${NT} agree` : `labels split ${r.distinct} ways`;
  const noteEl = el("span", null, r.all ? "✓ " + note : "⚠ " + note);
  noteEl.style.color = r.all ? "#2b8a3e" : "#c92a2a";
  noteEl.style.fontWeight = "600";
  ac.append(noteEl);
  const list = navList();
  const i = list.findIndex(x => x.l === r.l);
  $("detail-pos").textContent = `${i + 1} / ${list.length}` + (list.length < N ? " in view" : "");
  detail.classList.add("open");
}
function closeDetail(){
  detail.classList.remove("open");
  selectedLine = null;
  renderGrid();
  drawChart();
}
function navDetail(step){
  const list = navList();
  if (!list.length || selectedLine == null) return;
  const i = list.findIndex(x => x.l === selectedLine);
  const nxt = list[(i + step + list.length) % list.length];
  selectLine(nxt.l, true);
}
$("detail-close").addEventListener("click", closeDetail);
$("detail-prev").addEventListener("click", () => navDetail(-1));
$("detail-next").addEventListener("click", () => navDetail(1));
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (!detail.classList.contains("open")) return;
  if (e.key === "Escape") closeDetail();
  else if (e.key === "ArrowDown" || e.key === "ArrowRight") navDetail(1);
  else if (e.key === "ArrowUp" || e.key === "ArrowLeft") navDetail(-1);
});

/* ---------- wiring ---------- */
PAIRS.forEach(([a, b], i) => {
  const o = el("option", null, `${TOOLS[a].name} ↔ ${TOOLS[b].name} disagree`);
  o.value = "pair-" + i;
  $("mode").append(o);
});
$("mode").addEventListener("change", e => setMode(e.target.value));
$("search").addEventListener("input", e => { state.search = e.target.value; computeVisible(); renderGrid(); updateCount(); });
$("sort").addEventListener("change", e => { state.sort = e.target.value; computeVisible(); renderGrid(); updateCount(); });
$("dim").addEventListener("change", e => { state.dim = e.target.checked; renderGrid(); });

function layoutAll(){
  chartW = Math.max(320, ((chart.parentElement && chart.parentElement.clientWidth) || chart.clientWidth || 800));
  layoutGrid();
  renderGrid();
  drawChart();
}
window.addEventListener("resize", debounce(layoutAll, 150));

renderHeader();
renderAgreement();
renderDist();
renderPairs();
renderPatterns();
updatePill();
layoutAll();
computeVisible();
renderGrid();
updateCount();
</script>
</body>
</html>
"""


def main():
    """Read a comparison CSV and write the self-contained HTML file.

    The CSV must have the columns written by ``compare_sentiment.py`` (see
    that script's docstring for the full list). The HTML embeds the rows as
    JSON together with the page's styles and JavaScript, so the output works
    offline with no local files or network access.

    Command-line arguments: an optional positional ``csv_path`` naming the CSV
    to read (default sentiment_comparison.csv), ``--html`` for the output path
    (default sentiment_comparison.html), ``--source`` for a name shown in the
    page header, and ``--open`` to launch a browser tab when done.

    Raises:
        SystemExit: If the CSV is missing an expected column or contains no rows.
    """
    ap = argparse.ArgumentParser(
        description="Build a self-contained HTML visualization from compare_sentiment.py's CSV output.",
        epilog="Example: uv run python make_html.py sentiment_comparison.csv --source pg5200.txt --open",
    )
    ap.add_argument("csv_path", nargs="?", default="sentiment_comparison.csv", help="CSV produced by compare_sentiment.py")
    ap.add_argument("--html", default="sentiment_comparison.html", help="where to write the HTML")
    ap.add_argument("--source", default="", help="source text name shown in the page header")
    ap.add_argument("--open", action="store_true", help="open the result in a browser")
    args = ap.parse_args()

    rows = []
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "l": int(r["line"]),
                    "t": r["text"],
                    "v": float(r["vader_score"]),
                    "vc": float(r["vader_claude_score"]),
                    "b": float(r["bert_score"]),
                    "s": float(r["siebert_score"]),
                    "r": float(r["roberta_score"]),
                    "vl": r["vader_label"],
                    "vcl": r["vader_claude_label"],
                    "bl": r["bert_label"],
                    "sl": r["siebert_label"],
                    "rl": r["roberta_label"],
                    "st": int(r["bert_stars"]),
                    "bp": [float(r[f"bert_p{i}"]) for i in range(1, 6)],
                    "sp": float(r["siebert_p_pos"]),
                    "rp": [float(r[f"roberta_p_{c}"]) for c in ("neg", "neu", "pos")],
                    "vn": float(r["vader_neg"]),
                    "vu": float(r["vader_neu"]),
                    "vp": float(r["vader_pos"]),
                    "tr": r["truncated"].strip().lower() == "true",
                })
            except (KeyError, ValueError) as exc:
                sys.exit(f"{args.csv_path}: unexpected row/column ({exc}); "
                         f"is this a CSV written by compare_sentiment.py?")
    if not rows:
        sys.exit(f"{args.csv_path} contains no rows")

    # Escape '<' so embedded text can never close the <script> tag early.
    data_js = json.dumps(rows).replace("<", "\\u003c")
    meta = {"source": args.source, "csv": args.csv_path}
    meta_js = json.dumps(meta).replace("<", "\\u003c")
    html = HTML.replace("__DATA__", data_js).replace("__META__", meta_js)
    with open(args.html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.html} ({len(html) / 1024:.0f} KB, {len(rows)} chunks)")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.html), 1, True)


if __name__ == "__main__":
    import os
    main()
