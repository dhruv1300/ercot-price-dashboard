"""
Interactive dashboard for the ERCOT hub price analysis.

    python dashboard.py            -> writes dashboard.html (open in any browser)

Builds one self-contained HTML file: the cleaned hourly data, the Q6
walk-forward forecasts and a few model outputs are embedded as JSON, and
plotly.js (bundled with the Python plotly package) draws every chart in the
browser, so the global toggles (hub, years, season, day type, price
threshold) re-cut the data live. No server or internet connection needed.

Run ercot_analysis.py first (Q6/Q7) so results/ and predictions.csv exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline as po

from ercot_analysis import (HUBS, PRED_CSV, RES, ROOT, conformal_bands,
                            hourly_effects, load_data)

OUT = ROOT / "dashboard.html"
DAY0 = pd.Timestamp("2022-01-01")


def payload() -> dict:
    df = load_data()
    rows = {
        "d": (df["date"] - DAY0).dt.days,
        "he": df["he"], "month": df["month"], "year": df["year"], "dow": df["dow"],
        "temp": df["temp"].round(1),
        "load": (df["load"] / 1e3).round(2), "wind": (df["wind"] / 1e3).round(2),
        "solar": (df["solar"] / 1e3).round(2), "capout": (df["capout"] / 1e3).round(2),
        "gas": df["gas"].round(2), "north": df["north"].round(2),
        "houston": df["houston"].round(2)}
    rows = {k: v.astype(float if v.dtype.kind == "f" else int).tolist()
            for k, v in rows.items()}

    p = pd.read_csv(PRED_CSV, parse_dates=["dt", "date"])
    if "north_p10c" not in p:
        p = conformal_bands(p)
    preds = {"d": (p["date"] - DAY0).dt.days.tolist(), "he": p["he"].tolist(),
             "year": p["year"].tolist(), "dow": p["dow"].tolist()}
    for h in HUBS:
        for c in ["", "_p50", "_p10", "_p90", "_p10c", "_p90c", "_naive"]:
            preds[f"{h}{c}"] = [None if pd.isna(v) else round(float(v), 2)
                                for v in p[f"{h}{c}"]]

    eff = hourly_effects(df)
    effects = {"he": eff.index.astype(int).tolist(),
               "median_price": eff["median_price"].round(2).tolist()}
    for v in ["wind_gw", "solar_gw"]:
        for s in ["", "_lo", "_hi"]:
            effects[f"{v}{s}"] = [None if pd.isna(x) else round(float(x), 3)
                                  for x in eff[f"{v}{s}"]]

    q6 = json.loads((RES / "q6.json").read_text())
    q7 = json.loads((RES / "q7.json").read_text())
    return {"rows": rows, "preds": preds, "effects": effects,
            "importance": q6["perm_importance_north_2025"],
            "benchmark": q7["benchmark"], "day0": str(DAY0.date())}


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ERCOT Hub Prices</title>
<style>
:root {
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --neutral:#898781;
  --y2022:#86b6ef; --y2023:#3987e5; --y2024:#1c5cab; --y2025:#0d366b;
  --seq0:#f4f8fd; --seq1:#cde2fb; --seq2:#86b6ef; --seq3:#3987e5; --seq4:#1c5cab; --seq5:#0d366b;
  --neg:#e34948; --mid:#f0efec; --pos:#2a78d6; --good:#006300; --bad:#d03b3b;
  --band:rgba(42,120,214,.16); --band2:rgba(42,120,214,.32);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --neutral:#898781;
    --y2022:#256abf; --y2023:#3987e5; --y2024:#6da7ec; --y2025:#b7d3f6;
    --seq0:#202326; --seq1:#184f95; --seq2:#256abf; --seq3:#3987e5; --seq4:#86b6ef; --seq5:#cde2fb;
    --neg:#e66767; --mid:#383835; --pos:#3987e5; --good:#0ca30c; --bad:#e66767;
    --band:rgba(57,135,229,.20); --band2:rgba(57,135,229,.38);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --neutral:#898781;
  --y2022:#256abf; --y2023:#3987e5; --y2024:#6da7ec; --y2025:#b7d3f6;
  --seq0:#202326; --seq1:#184f95; --seq2:#256abf; --seq3:#3987e5; --seq4:#86b6ef; --seq5:#cde2fb;
  --neg:#e66767; --mid:#383835; --pos:#3987e5; --good:#0ca30c; --bad:#e66767;
  --band:rgba(57,135,229,.20); --band2:rgba(57,135,229,.38);
}
* { box-sizing: border-box; }
html, body { margin:0; background:var(--page); color:var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size:14px; }
header { padding:20px 24px 8px; display:flex; gap:16px; align-items:flex-start; justify-content:space-between; }
h1 { font-size:20px; margin:0 0 4px; font-weight:650; }
.sub { color:var(--text2); margin:0; max-width:760px; line-height:1.45; }
.theme-btn { border:1px solid var(--border); background:var(--surface); color:var(--text2);
  border-radius:8px; padding:6px 10px; cursor:pointer; font:inherit; white-space:nowrap; }
.filters { position:sticky; top:0; z-index:5; background:var(--page);
  padding:10px 24px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center;
  border-bottom:1px solid var(--border); }
.f { display:flex; align-items:center; gap:8px; color:var(--text2); }
.f > span.lbl { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.seg { display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.seg button, .chip { border:0; background:var(--surface); color:var(--text2); padding:6px 10px;
  font:inherit; cursor:pointer; }
.seg button + button { border-left:1px solid var(--border); }
.seg button[aria-pressed="true"] { background:var(--s1); color:#fff; }
.chip { border:1px solid var(--border); border-radius:999px; padding:4px 10px; }
.chip[aria-pressed="true"] { background:var(--s1); color:#fff; border-color:var(--s1); }
select, input[type=number], input[type=date] { background:var(--surface); color:var(--text);
  border:1px solid var(--border); border-radius:8px; padding:5px 8px; font:inherit; }
input[type=number] { width:84px; }
nav.tabs { display:flex; gap:4px; padding:12px 24px 0; overflow-x:auto; scrollbar-width:none; }
nav.tabs button { border:0; background:transparent; color:var(--text2); font:inherit; padding:8px 12px;
  border-bottom:2px solid transparent; cursor:pointer; white-space:nowrap; }
nav.tabs button[aria-selected="true"] { color:var(--text); border-bottom-color:var(--s1); font-weight:600; }
main { padding:12px 24px 40px; }
.note { color:var(--text2); margin:4px 0 12px; line-height:1.45; max-width:900px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:10px; margin-bottom:12px; }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
.kpi .k { font-size:12px; color:var(--muted); }
.kpi .v { font-size:22px; font-weight:600; margin-top:2px; }
.kpi .s { font-size:12px; color:var(--text2); margin-top:2px; }
.grid { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:12px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px 12px 6px; min-width:0; }
.card.wide { grid-column:1 / -1; }
.card h3 { margin:0 0 2px; font-size:14px; font-weight:600; }
.card .cap { margin:0 0 4px; color:var(--text2); font-size:12.5px; line-height:1.4; }
.card .ctl { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:4px 0; color:var(--text2); font-size:12.5px; }
.chart { width:100%; height:320px; }
.chart.tall { height:380px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13px; }
th, td { padding:6px 8px; border-bottom:1px solid var(--grid); text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; }
tr.hl td { font-weight:600; }
#q4t, #q6t { overflow-x:auto; }
.pos { color:var(--good); } .neg { color:var(--bad); }
ul.obs { margin:6px 0 8px; padding-left:18px; line-height:1.5; color:var(--text2); }
ul.obs b { color:var(--text); }
@media (max-width: 760px) {
  header, .filters, nav.tabs, main { padding-left:16px; padding-right:16px; }
  .grid { grid-template-columns:1fr; }
  .filters { position:static; }
  .chart { height:280px; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>ERCOT Hub Prices: HB_NORTH and HB_HOUSTON, 2022–2025</h1>
    <p class="sub">Day-ahead prices and grid fundamentals, hourly. The filters below re-cut every chart; the price threshold sets what counts as a "high-price" hour.</p>
  </div>
  <button class="theme-btn" id="themeBtn" type="button" aria-label="Switch colour theme">Theme: auto</button>
</header>

<div class="filters" role="region" aria-label="Filters">
  <div class="f"><span class="lbl">Hub</span>
    <div class="seg" id="hubSeg">
      <button type="button" data-v="north" aria-pressed="true">HB_NORTH</button>
      <button type="button" data-v="houston" aria-pressed="false">HB_HOUSTON</button>
    </div></div>
  <div class="f"><span class="lbl">Years</span><span id="yearChips"></span></div>
  <div class="f"><span class="lbl">Season</span>
    <select id="season" aria-label="Season"><option>All</option><option>Winter</option><option>Spring</option><option>Summer</option><option>Fall</option></select></div>
  <div class="f"><span class="lbl">Days</span>
    <select id="days" aria-label="Day type"><option>All</option><option>Weekdays</option><option>Weekends</option></select></div>
  <div class="f"><span class="lbl">High price ≥ $</span>
    <input type="number" id="thr" value="100" min="10" max="2000" step="10" aria-label="High-price threshold"></div>
</div>

<nav class="tabs" role="tablist" id="tabs"></nav>
<main id="main"></main>

<script>__PLOTLY__</script>
<script>
const DATA = __DATA__;
const R = DATA.rows, P = DATA.preds, N = R.he.length;
const DAY0 = new Date(DATA.day0 + "T00:00:00");
const YEARS = [2022, 2023, 2024, 2025];
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const HUBNAME = {north: "HB_NORTH", houston: "HB_HOUSTON"};
const seasonOf = m => (m === 12 || m <= 2) ? "Winter" : m <= 5 ? "Spring" : m <= 9 ? "Summer" : "Fall";
R.season = R.month.map(seasonOf);
R.netload = R.load.map((l, i) => l - R.wind[i] - R.solar[i]);
R.tight = R.netload.map((x, i) => x + R.capout[i]);
R.spread = R.houston.map((h, i) => h - R.north[i]);
P.month = P.d.map(d => new Date(DAY0.getTime() + d * 864e5).getMonth() + 1);

const state = { hub: "north", years: new Set(YEARS), season: "All", days: "All", thr: 100,
  tab: "q1", mw: 50, kN: 50, kH: 53, quint: "wind", week: "2024-08-17", band: "cal",
  effUnit: "pct", q3heat: "price" };

// ---------------------------------------------------------------- helpers
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN;
function quantile(a, q) {
  if (!a.length) return NaN;
  const s = Float64Array.from(a).sort(); const pos = (s.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
const median = a => quantile(a, 0.5);
function cvar(a, q = 0.95) { const t = quantile(a, q); return mean(a.filter(x => x >= t)); }
function groupIdx(idx, key) {
  const m = new Map();
  for (const i of idx) { const k = key(i); if (!m.has(k)) m.set(k, []); m.get(k).push(i); }
  return m;
}
const pick = (idx, arr) => idx.map(i => arr[i]);
const fmt$ = (x, d = 0) => isFinite(x) ? (x < 0 ? "−$" : "$") + Math.abs(x).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d}) : "–";
const fmtN = (x, d = 0) => isFinite(x) ? x.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d}) : "–";
const pct = (x, d = 0) => isFinite(x) ? (x * 100).toFixed(d) + "%" : "–";
const range = (a, b) => Array.from({length: b - a + 1}, (_, i) => a + i);
const HRS = range(1, 24);

function filtered(useYears = true) {
  const out = [];
  for (let i = 0; i < N; i++) {
    if (useYears && !state.years.has(R.year[i])) continue;
    if (state.season !== "All" && R.season[i] !== state.season) continue;
    const we = R.dow[i] >= 5;
    if (state.days === "Weekdays" && we) continue;
    if (state.days === "Weekends" && !we) continue;
    out.push(i);
  }
  return out;
}
const price = () => R[state.hub];

// ---------------------------------------------------------------- plotting
function baseLayout(extra = {}) {
  const ink2 = css("--text2"), grid = css("--grid"), axis = css("--axis");
  const ax = {gridcolor: grid, linecolor: axis, zeroline: false, tickfont: {color: css("--muted"), size: 11},
    title: {font: {color: ink2, size: 12}}, automargin: true};
  const L = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {family: "system-ui, -apple-system, Segoe UI, sans-serif", color: ink2, size: 12},
    margin: {l: 56, r: 16, t: 8, b: 44}, hovermode: "closest",
    hoverlabel: {bgcolor: css("--surface"), bordercolor: css("--axis"), font: {color: css("--text")}},
    legend: {orientation: "h", y: 1.12, x: 0, font: {size: 11}},
    xaxis: {...ax}, yaxis: {...ax} };
  for (const [k, v] of Object.entries(extra)) L[k] = (typeof v === "object" && L[k] && !Array.isArray(v)) ? {...L[k], ...v} : v;
  return L;
}
const CFG = {displayModeBar: false, responsive: true};
function plot(id, traces, layout) { Plotly.react(id, traces, baseLayout(layout), CFG); }
const line = (x, y, name, color, extra = {}) => ({x, y, name, type: "scatter", mode: "lines+markers",
  line: {color, width: 2}, marker: {size: 5, color}, ...extra});
const yearColor = y => css("--y" + y);
const seqScale = () => [0, .2, .4, .6, .8, 1].map((t, i) => [t, css("--seq" + i)]);
const divScale = () => [[0, css("--neg")], [0.5, css("--mid")], [1, css("--pos")]];
const logTicks = vals => ({type: "log", tickvals: vals, ticktext: vals.map(v => v >= 1000 ? (v / 1000) + "k" : String(v))});

// ---------------------------------------------------------------- UI shell
const TABS = [
  ["q1", "Drivers (Q1)"], ["q2", "Wind & solar (Q2)"], ["q3", "Risk by hour (Q3)"],
  ["q4", "Strip trade (Q4)"], ["q5", "Houston − North (Q5)"], ["q6", "Forecast model (Q6–7)"],
  ["q8", "Observations (Q8)"]];
function card(id, title, cap, opts = {}) {
  return `<section class="card${opts.wide ? " wide" : ""}"><h3>${title}</h3>${cap ? `<p class="cap">${cap}</p>` : ""}${opts.ctl || ""}<div id="${id}" class="chart${opts.tall ? " tall" : ""}"></div></section>`;
}
function kpis(items) {
  return `<div class="kpis">${items.map(([k, v, s]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`).join("")}</div>`;
}

function buildChrome() {
  document.getElementById("yearChips").innerHTML = YEARS.map(y =>
    `<button type="button" class="chip" data-y="${y}" aria-pressed="true">${y}</button>`).join(" ");
  document.querySelectorAll("#yearChips .chip").forEach(b => b.onclick = () => {
    const y = +b.dataset.y;
    if (state.years.has(y)) { if (state.years.size > 1) state.years.delete(y); } else state.years.add(y);
    b.setAttribute("aria-pressed", state.years.has(y)); syncChips(); render();
  });
  document.querySelectorAll("#hubSeg button").forEach(b => b.onclick = () => {
    state.hub = b.dataset.v;
    document.querySelectorAll("#hubSeg button").forEach(x => x.setAttribute("aria-pressed", x === b));
    render();
  });
  document.getElementById("season").onchange = e => { state.season = e.target.value; render(); };
  document.getElementById("days").onchange = e => { state.days = e.target.value; render(); };
  document.getElementById("thr").onchange = e => { state.thr = Math.max(1, +e.target.value || 100); render(); };
  const nav = document.getElementById("tabs");
  nav.innerHTML = TABS.map(([k, l]) => `<button type="button" role="tab" data-t="${k}" aria-selected="${k === state.tab}">${l}</button>`).join("");
  nav.querySelectorAll("button").forEach(b => b.onclick = () => {
    state.tab = b.dataset.t;
    nav.querySelectorAll("button").forEach(x => x.setAttribute("aria-selected", x === b));
    try { localStorage.setItem("ercot_tab", state.tab); } catch (e) {}
    render();
  });
}
function syncChips() {
  document.querySelectorAll("#yearChips .chip").forEach(b => b.setAttribute("aria-pressed", state.years.has(+b.dataset.y)));
}

// ---------------------------------------------------------------- Q1
function renderQ1(main) {
  const idx = filtered(), p = price(), pr = pick(idx, p), thr = state.thr;
  const hi = pr.filter(x => x >= thr).length;
  const gas = pick(idx, R.gas);
  main.innerHTML = `<p class="note">What the grid looks like when prices are high. Price rises with net load (load − wind − solar), gas sets the level, and outages matter near the top of the stack.</p>` +
    kpis([["Average price", fmt$(mean(pr), 2), `median ${fmt$(median(pr), 2)}`],
          [`Hours ≥ $${thr}`, fmtN(hi), pct(hi / pr.length, 1) + " of hours"],
          ["Highest price", fmt$(Math.max(...pr)), ""],
          ["Average gas", fmt$(mean(gas), 2), "Henry Hub, $/MMBtu"],
          ["Avg implied heat rate", fmtN(mean(pr) / mean(gas), 1), "avg price ÷ avg gas"],
          ["Hours in view", fmtN(pr.length), ""]]) +
    `<div class="grid">` +
    card("q1a", "Monthly average price", "Both hubs; the selected hub is drawn first.", {wide: true}) +
    card("q1b", "Median price by net load, split by gas price", "Log scale. Same hockey-stick shape at every gas level; gas shifts it up.") +
    card("q1c", `Chance of an hour ≥ $${thr}`, "By net load, and by net load plus capacity on outage (tightness).") +
    card("q1d", `Share of hours ≥ $${thr} by month and hour`, "Summer evenings dominate; winter mornings are the second risk window.") +
    card("q1e", "Median price by DFW temperature", "U-shaped: cheapest around 60–65°F.") +
    `</div>`;

  const byM = groupIdx(idx, i => R.year[i] * 100 + R.month[i]);
  const keys = [...byM.keys()].sort((a, b) => a - b);
  const xs = keys.map(k => `${Math.floor(k / 100)}-${String(k % 100).padStart(2, "0")}-01`);
  const other = state.hub === "north" ? "houston" : "north";
  plot("q1a", [
    {x: xs, y: keys.map(k => mean(pick(byM.get(k), R[state.hub]))), name: HUBNAME[state.hub], type: "scatter", mode: "lines", line: {color: css("--s1"), width: 2}},
    {x: xs, y: keys.map(k => mean(pick(byM.get(k), R[other]))), name: HUBNAME[other], type: "scatter", mode: "lines", line: {color: css("--s2"), width: 1.5}},
    {x: xs, y: keys.map(k => mean(pick(byM.get(k), R.gas)) * 10), name: "Gas × 10 ($/MMBtu)", type: "scatter", mode: "lines", line: {color: css("--neutral"), width: 1.5}},
  ], {yaxis: {title: {text: "$/MWh"}}, hovermode: "x unified"});

  const regimes = [["Gas < $3", g => g < 3, "--s1"], ["Gas $3–5", g => g >= 3 && g <= 5, "--s3"], ["Gas > $5", g => g > 5, "--s2"]];
  plot("q1b", regimes.map(([name, f, col]) => {
    const m = groupIdx(idx.filter(i => f(R.gas[i])), i => Math.floor((R.netload[i] - 15) / 2.5));
    const ks = [...m.keys()].filter(k => m.get(k).length >= 30).sort((a, b) => a - b);
    return line(ks.map(k => 15 + 2.5 * k + 1.25), ks.map(k => median(pick(m.get(k), p))), name, css(col));
  }), {xaxis: {title: {text: "Net load (GW)"}}, yaxis: {title: {text: "Median $/MWh (log)"}, ...logTicks([5, 10, 20, 50, 100, 200, 500, 1000])}});

  plot("q1c", [["Net load", R.netload, "--s1"], ["Net load + outages", R.tight, "--s2"]].map(([name, arr, col]) => {
    const m = groupIdx(idx, i => Math.floor(arr[i] / 5));
    const ks = [...m.keys()].filter(k => m.get(k).length >= 30).sort((a, b) => a - b);
    return line(ks.map(k => k * 5 + 2.5), ks.map(k => 100 * pick(m.get(k), p).filter(x => x >= thr).length / m.get(k).length), name, css(col));
  }), {xaxis: {title: {text: "GW"}}, yaxis: {title: {text: `% of hours ≥ $${thr}`}, rangemode: "tozero"}});

  const z = MONTHS.map(() => HRS.map(() => [0, 0]));
  for (const i of idx) { const c = z[R.month[i] - 1][R.he[i] - 1]; c[1]++; if (p[i] >= thr) c[0]++; }
  plot("q1d", [{type: "heatmap", x: HRS, y: MONTHS, z: z.map(r => r.map(([a, b]) => b ? 100 * a / b : null)),
    colorscale: seqScale(), zmin: 0, hovertemplate: "%{y} HE%{x}: %{z:.1f}%<extra></extra>",
    colorbar: {thickness: 10, outlinewidth: 0, tickfont: {size: 10}, ticksuffix: "%"}}],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2, showgrid: false}, yaxis: {autorange: "reversed", showgrid: false}});

  const tb = groupIdx(idx, i => Math.floor(R.temp[i] / 5));
  const tk = [...tb.keys()].filter(k => tb.get(k).length >= 30).sort((a, b) => a - b);
  plot("q1e", [["north", "--s1"], ["houston", "--s2"]].map(([h, col]) =>
    line(tk.map(k => k * 5 + 2.5), tk.map(k => median(pick(tb.get(k), R[h]))), HUBNAME[h], css(col))),
    {xaxis: {title: {text: "DFW temperature (°F)"}}, yaxis: {title: {text: "Median $/MWh"}}});
}

// ---------------------------------------------------------------- Q2
function renderQ2(main) {
  const idx = filtered(), p = price(), E = DATA.effects;
  main.innerHTML = `<p class="note">Wind and solar push the expensive end of the supply stack out of the money. Per GW the % effect is similar all day, but in dollars it is largest at HE18–20, when the system is tight and solar is fading.</p>` +
    `<div class="grid">` +
    card("q2a", "Average daily profile", "Load, net load, wind and solar (GW).") +
    card("q2b", "Net load by hour, per year", "The duck curve deepens as solar grows: lower midday, higher evening peak.") +
    card("q2c", "Median implied heat rate by hour, per year", "Price ÷ gas, so years with different gas prices compare fairly.") +
    card("q2d", "Price change per +1 GW, by hour", "Per-hour regressions with month and year effects, all 2022–25 data (not filtered). Shaded: 95% CI.",
      {ctl: `<div class="ctl">Show as <select id="effUnit"><option value="pct">% of price</option><option value="usd">$/MWh at that hour's median</option></select></div>`}) +
    card("q2e", "Hours below $10/MWh", "Two cheap windows: wind overnight, solar at midday.") +
    `</div>`;
  document.getElementById("effUnit").value = state.effUnit;
  document.getElementById("effUnit").onchange = e => { state.effUnit = e.target.value; render(); };

  const byH = groupIdx(idx, i => R.he[i]);
  const prof = arr => HRS.map(h => mean(pick(byH.get(h) || [], arr)));
  plot("q2a", [line(HRS, prof(R.load), "Load", css("--neutral")), line(HRS, prof(R.netload), "Net load", css("--text2")),
    line(HRS, prof(R.wind), "Wind", css("--s1")), line(HRS, prof(R.solar), "Solar", css("--s2"))],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "GW"}}, hovermode: "x unified"});

  const yrs = [...state.years].sort();
  const byYH = groupIdx(idx, i => R.year[i] * 100 + R.he[i]);
  const yLines = (f) => yrs.map(y => ({x: HRS, y: HRS.map(h => { const g = byYH.get(y * 100 + h) || []; return f(g); }),
    name: String(y), type: "scatter", mode: "lines", line: {color: yearColor(y), width: 2}}));
  plot("q2b", yLines(g => mean(pick(g, R.netload))), {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "GW"}}, hovermode: "x unified"});
  plot("q2c", yLines(g => median(g.map(i => p[i] / R.gas[i]))), {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "MMBtu/MWh"}}, hovermode: "x unified"});

  const usd = state.effUnit === "usd";
  const conv = (v, k) => v == null ? null : (usd ? v / 100 * E.median_price[k] : v);
  const effTr = [];
  for (const [v, name, col] of [["wind_gw", "Wind", "--s1"], ["solar_gw", "Solar", "--s2"]]) {
    const ks = E.he.map((h, k) => k).filter(k => E[v][k] != null);
    const xs = ks.map(k => E.he[k]);
    if (!usd) effTr.push({x: [...xs, ...xs.slice().reverse()], y: [...ks.map(k => E[v + "_hi"][k]), ...ks.slice().reverse().map(k => E[v + "_lo"][k])],
      fill: "toself", fillcolor: col === "--s1" ? css("--band") : "rgba(235,104,52,.15)", line: {width: 0}, hoverinfo: "skip", showlegend: false, type: "scatter"});
    effTr.push(line(xs, ks.map(k => conv(E[v][k], k)), name, css(col)));
  }
  plot("q2d", effTr, {xaxis: {title: {text: "Hour ending"}, dtick: 2, range: [0.5, 24.5]},
    yaxis: {title: {text: usd ? "$/MWh per +1 GW" : "% change per +1 GW"}}, shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: {color: css("--axis"), width: 1}}]});

  plot("q2e", [{x: HRS, y: HRS.map(h => { const g = byH.get(h) || []; return g.length ? 100 * pick(g, p).filter(x => x < 10).length / g.length : null; }),
    type: "bar", marker: {color: css("--s1")}, name: "", hovertemplate: "HE%{x}: %{y:.1f}%<extra></extra>"}],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "% of hours"}}, bargap: 0.15});
}

// ---------------------------------------------------------------- Q3
function renderQ3(main) {
  const idx = filtered(), p = price(), thr = state.thr;
  const byH = groupIdx(idx, i => R.he[i]);
  const stats = HRS.map(h => { const v = pick(byH.get(h) || [], p); return {med: median(v), p95: quantile(v, .95), cv: cvar(v), hi: v.filter(x => x >= thr).length / v.length}; });
  const peak = stats.reduce((b, s, k) => s.cv > stats[b].cv ? k : b, 0);
  const quiet = mean(stats.slice(0, 5).map(s => s.cv));
  main.innerHTML = `<p class="note">Risk for a buyer of power is the upper tail. CVaR = average price in the worst 5% of hours for that hour ending. The risk window has moved later and narrowed as solar grew.</p>` +
    kpis([["Riskiest hour", "HE" + (peak + 1), `CVaR ${fmt$(stats[peak].cv)}`],
          ["Overnight CVaR (HE1–5)", fmt$(quiet), ""],
          [`HE${peak + 1} hours ≥ $${thr}`, pct(stats[peak].hi, 1), ""]]) +
    `<div class="grid">` +
    card("q3a", "Price risk by hour", "Median, 95th percentile and CVaR (log scale).") +
    card("q3b", "CVaR by hour, per year", "2022 peaked at HE17; from 2023 the peak is HE20.") +
    card("q3c", "When is risk concentrated? Year × hour", "",
      {ctl: `<div class="ctl">Measure <select id="q3heat"><option value="price">Share of hours ≥ threshold price</option><option value="ihr">Share with heat rate > 20 (gas-neutral)</option></select></div>`}) +
    card("q3d", "CVaR by season and hour", "Summer evenings vs winter mornings (colour on a log scale).") +
    `</div>`;
  document.getElementById("q3heat").value = state.q3heat;
  document.getElementById("q3heat").onchange = e => { state.q3heat = e.target.value; render(); };

  plot("q3a", [line(HRS, stats.map(s => s.med), "Median", css("--neutral")), line(HRS, stats.map(s => s.p95), "95th percentile", css("--s1")),
    line(HRS, stats.map(s => s.cv), "CVaR (mean of top 5%)", css("--s2"))],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "$/MWh (log)"}, ...logTicks([10, 20, 50, 100, 200, 500, 1000, 2000])}, hovermode: "x unified"});

  const yrs = [...state.years].sort();
  const byYH = groupIdx(idx, i => R.year[i] * 100 + R.he[i]);
  plot("q3b", yrs.map(y => ({x: HRS, y: HRS.map(h => cvar(pick(byYH.get(y * 100 + h) || [], p))), name: String(y), type: "scatter", mode: "lines", line: {color: yearColor(y), width: 2}})),
    {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "CVaR $/MWh (log)"}, ...logTicks([20, 50, 100, 200, 500, 1000, 3000])}, hovermode: "x unified"});

  const ihr = state.q3heat === "ihr";
  const zz = yrs.map(y => HRS.map(h => { const g = byYH.get(y * 100 + h) || []; if (!g.length) return null;
    return 100 * g.filter(i => ihr ? p[i] / R.gas[i] > 20 : p[i] >= thr).length / g.length; }));
  plot("q3c", [{type: "heatmap", x: HRS, y: yrs.map(String), z: zz, colorscale: seqScale(), zmin: 0,
    hovertemplate: "%{y} HE%{x}: %{z:.1f}%<extra></extra>", colorbar: {thickness: 10, outlinewidth: 0, ticksuffix: "%", tickfont: {size: 10}}}],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2, showgrid: false}, yaxis: {showgrid: false, type: "category", autorange: "reversed"}});

  const seasons = ["Winter", "Spring", "Summer", "Fall"];
  const bySH = groupIdx(idx, i => R.season[i] + "|" + R.he[i]);
  const zs = seasons.map(s => HRS.map(h => { const g = bySH.get(s + "|" + h); return g && g.length > 20 ? cvar(pick(g, p)) : null; }));
  const lz = zs.map(r => r.map(v => v ? Math.log10(Math.max(v, 1)) : null));
  const tv = [30, 100, 300, 1000, 3000];
  plot("q3d", [{type: "heatmap", x: HRS, y: seasons, z: lz, customdata: zs, colorscale: seqScale(),
    hovertemplate: "%{y} HE%{x}: $%{customdata:,.0f}<extra></extra>",
    colorbar: {thickness: 10, outlinewidth: 0, tickvals: tv.map(Math.log10), ticktext: tv.map(v => "$" + v), tickfont: {size: 10}}}],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2, showgrid: false}, yaxis: {showgrid: false, autorange: "reversed"}});
}

// ---------------------------------------------------------------- Q4
function renderQ4(main) {
  const all = range(0, N - 1);
  const {mw, kN, kH} = state;
  const pnlN = R.north.map(x => (x - kN) * mw), pnlH = R.houston.map(x => (x - kH) * mw);
  const sum = a => a.reduce((s, x) => s + x, 0);
  const totN = sum(pnlN) / 1e6, totH = sum(pnlH) / 1e6;
  const mwh = N * mw;
  main.innerHTML = `<p class="note">A long ATC strip bought on 1 Jan 2021 for 2022–25, settled hourly against the day-ahead hub price: P&amp;L = Σ (price − strike) × MW. Uses the full period; the season/day/year filters don't apply here.</p>
    <div class="ctl" style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;color:var(--text2)">
      <label>Volume (MW) <input type="number" id="mw" value="${mw}" min="1" step="5"></label>
      <label>HB_NORTH strike ($) <input type="number" id="kN" value="${kN}" step="0.5"></label>
      <label>HB_HOUSTON strike ($) <input type="number" id="kH" value="${kH}" step="0.5"></label>
    </div>` +
    kpis([["HB_NORTH P&L", `<span class="${totN >= 0 ? "pos" : "neg"}">${fmt$(totN, 2)}M</span>`, `avg price ${fmt$(mean(R.north), 2)} vs strike ${fmt$(kN, 2)}`],
          ["HB_HOUSTON P&L", `<span class="${totH >= 0 ? "pos" : "neg"}">${fmt$(totH, 2)}M</span>`, `avg price ${fmt$(mean(R.houston), 2)} vs strike ${fmt$(kH, 2)}`],
          ["Houston minus North", `${fmt$(totH - totN, 2)}M`, `spread paid ${fmt$(kH - kN, 2)}, realised ${fmt$(mean(R.spread), 2)}`],
          ["Volume", fmtN(mwh / 1e6, 2) + " TWh", `${fmtN(N)} hours × ${mw} MW`]]) +
    `<div class="grid">` + card("q4a", "Cumulative P&L", "Monthly, $ millions.", {wide: true}) +
    card("q4b", "P&L by year", "$ millions.") +
    `<section class="card"><h3>By year</h3><div id="q4t"></div></section></div>`;
  for (const id of ["mw", "kN", "kH"]) document.getElementById(id).onchange = e => { state[id] = +e.target.value; render(); };

  const byM = groupIdx(all, i => R.year[i] * 100 + R.month[i]);
  const ks = [...byM.keys()].sort((a, b) => a - b);
  const xs = ks.map(k => `${Math.floor(k / 100)}-${String(k % 100).padStart(2, "0")}-01`);
  const cum = arr => { let c = 0; return ks.map(k => (c += sum(pick(byM.get(k), arr))) / 1e6); };
  plot("q4a", [
    {x: xs, y: cum(pnlN), name: `HB_NORTH @ $${kN}`, type: "scatter", mode: "lines", line: {color: css("--s1"), width: 2}},
    {x: xs, y: cum(pnlH), name: `HB_HOUSTON @ $${kH}`, type: "scatter", mode: "lines", line: {color: css("--s2"), width: 2}}],
    {yaxis: {title: {text: "$M"}}, hovermode: "x unified",
     shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: {color: css("--axis"), width: 1}}]});

  const byY = groupIdx(all, i => R.year[i]);
  plot("q4b", [["HB_NORTH", pnlN, "--s1"], ["HB_HOUSTON", pnlH, "--s2"]].map(([name, arr, col]) =>
    ({x: YEARS.map(String), y: YEARS.map(y => sum(pick(byY.get(y), arr)) / 1e6), name, type: "bar", marker: {color: css(col)},
      hovertemplate: "%{x}: $%{y:.2f}M<extra>" + name + "</extra>"})),
    {barmode: "group", bargap: 0.25, bargroupgap: 0.06, yaxis: {title: {text: "$M"}}, xaxis: {type: "category"}});

  document.getElementById("q4t").innerHTML = `<table><thead><tr><th>Year</th><th>North avg</th><th>North P&amp;L</th><th>Houston avg</th><th>Houston P&amp;L</th><th>Spread</th></tr></thead><tbody>` +
    YEARS.map(y => { const g = byY.get(y); const a = sum(pick(g, pnlN)) / 1e6, b = sum(pick(g, pnlH)) / 1e6;
      return `<tr><td>${y}</td><td>${fmt$(mean(pick(g, R.north)), 2)}</td><td class="${a >= 0 ? "pos" : "neg"}">${fmt$(a, 2)}M</td><td>${fmt$(mean(pick(g, R.houston)), 2)}</td><td class="${b >= 0 ? "pos" : "neg"}">${fmt$(b, 2)}M</td><td>${fmt$(mean(pick(g, R.spread)), 2)}</td></tr>`; }).join("") +
    `<tr class="hl"><td>Total</td><td>${fmt$(mean(R.north), 2)}</td><td>${fmt$(totN, 2)}M</td><td>${fmt$(mean(R.houston), 2)}</td><td>${fmt$(totH, 2)}M</td><td>${fmt$(mean(R.spread), 2)}</td></tr></tbody></table>`;
}

// ---------------------------------------------------------------- Q5
function renderQ5(main) {
  const idx = filtered(), s = pick(idx, R.spread);
  const drivers = {wind: ["Wind", R.wind], capout: ["Capacity on outage", R.capout], load: ["Load", R.load], temp: ["Temperature", R.temp], solar: ["Solar", R.solar], netload: ["Net load", R.netload]};
  main.innerHTML = `<p class="note">Spread = HB_HOUSTON − HB_NORTH. Usually small, with fat tails: positive on spring afternoons with high wind and heat, negative on winter cold mornings and peak-summer evenings. The hub toggle doesn't apply here.</p>` +
    kpis([["Mean spread", fmt$(mean(s), 2), `median ${fmt$(median(s), 2)}`], ["Houston above North", pct(s.filter(x => x > 0).length / s.length), "of hours"],
          ["1st / 99th percentile", `${fmt$(quantile(s, .01), 0)} / ${fmt$(quantile(s, .99), 0)}`, ""], ["Std. deviation", fmt$(Math.sqrt(mean(s.map(x => (x - mean(s)) ** 2))), 2), ""]]) +
    `<div class="grid">` +
    card("q5a", "Mean spread by month and hour", "Blue: Houston premium. Red: North premium.") +
    card("q5b", "Spread distribution by hour", "Mean with 5–95th and 1–99th percentile bands.") +
    card("q5c", "Spread by quintile of a fundamental", "Hours sorted into five equal groups, low to high.",
      {ctl: `<div class="ctl">Driver <select id="quint">${Object.entries(drivers).map(([k, [l]]) => `<option value="${k}">${l}</option>`).join("")}</select></div>`}) +
    card("q5d", "Mean spread by year", "Spring congestion made 2022 exceptional.") +
    `</div>`;
  document.getElementById("quint").value = state.quint;
  document.getElementById("quint").onchange = e => { state.quint = e.target.value; render(); };

  const z = MONTHS.map(() => HRS.map(() => []));
  for (const i of idx) z[R.month[i] - 1][R.he[i] - 1].push(R.spread[i]);
  const zm = z.map(r => r.map(v => v.length ? mean(v) : null));
  const flat = zm.flat().filter(v => v != null).map(Math.abs).sort((a, b) => a - b);
  const lim = flat.length ? flat[Math.floor(flat.length * 0.98) - 1] || flat[flat.length - 1] : 10;
  plot("q5a", [{type: "heatmap", x: HRS, y: MONTHS, z: zm, colorscale: divScale(), zmin: -lim, zmax: lim,
    hovertemplate: "%{y} HE%{x}: $%{z:.2f}<extra></extra>", colorbar: {thickness: 10, outlinewidth: 0, tickprefix: "$", tickfont: {size: 10}}}],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2, showgrid: false}, yaxis: {autorange: "reversed", showgrid: false}});

  const byH = groupIdx(idx, i => R.he[i]);
  const q = (h, qq) => quantile(pick(byH.get(h) || [], R.spread), qq);
  const band = (lo, hi, fill, name) => ({x: [...HRS, ...HRS.slice().reverse()], y: [...HRS.map(h => q(h, hi)), ...HRS.slice().reverse().map(h => q(h, lo))],
    fill: "toself", fillcolor: fill, line: {width: 0}, name, type: "scatter", hoverinfo: "skip"});
  plot("q5b", [band(.01, .99, css("--band"), "1–99th pct"), band(.05, .95, css("--band2"), "5–95th pct"),
    line(HRS, HRS.map(h => mean(pick(byH.get(h) || [], R.spread))), "Mean", css("--s1"))],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "$/MWh"}}});

  const [lbl, arr] = drivers[state.quint];
  const sorted = idx.slice().sort((a, b) => arr[a] - arr[b]);
  const qs = range(0, 4).map(k => sorted.slice(Math.floor(k * sorted.length / 5), Math.floor((k + 1) * sorted.length / 5)));
  plot("q5c", [
    {x: qs.map((_, k) => "Q" + (k + 1)), y: qs.map(g => mean(pick(g, R.spread))), name: "Mean", type: "bar", marker: {color: css("--s1")}},
    {x: qs.map((_, k) => "Q" + (k + 1)), y: qs.map(g => quantile(pick(g, R.spread), .95)), name: "95th percentile", type: "bar", marker: {color: css("--s3")}}],
    {barmode: "group", bargap: 0.25, bargroupgap: 0.06, xaxis: {title: {text: `${lbl} quintile (low to high)`}}, yaxis: {title: {text: "$/MWh"}}});

  const byY = groupIdx(idx, i => R.year[i]);
  const yk = [...byY.keys()].sort();
  plot("q5d", [{x: yk.map(String), y: yk.map(y => mean(pick(byY.get(y), R.spread))), type: "bar", marker: {color: css("--s1")}, name: "",
    hovertemplate: "%{x}: $%{y:.2f}<extra></extra>"}], {xaxis: {type: "category"}, yaxis: {title: {text: "Mean $/MWh"}}, bargap: 0.4});
}

// ---------------------------------------------------------------- Q6-7
function renderQ6(main) {
  const h = state.hub, cal = state.band === "cal";
  const lo = cal ? P[h + "_p10c"] : P[h + "_p10"], hi = cal ? P[h + "_p90c"] : P[h + "_p90"];
  const M = P.he.length, idx = [];
  for (let i = 0; i < M; i++) {
    if (!state.years.has(P.year[i])) continue;
    const s = seasonOf(P.month[i]); if (state.season !== "All" && s !== state.season) continue;
    const we = P.dow[i] >= 5; if (state.days === "Weekdays" && we) continue; if (state.days === "Weekends" && !we) continue;
    if (P[h + "_p50"][i] == null) continue;
    idx.push(i);
  }
  const act = P[h], mae = arr => mean(idx.filter(i => arr[i] != null).map(i => Math.abs(arr[i] - act[i])));
  const mm = mae(P[h + "_p50"]), mn = mae(P[h + "_naive"]);
  const ci = idx.filter(i => lo[i] != null);
  const cov = ci.filter(i => act[i] >= lo[i] && act[i] <= hi[i]).length / ci.length;
  const medAE = median(idx.map(i => Math.abs(P[h + "_p50"][i] - act[i])));
  main.innerHTML = `<p class="note">Walk-forward test: refit monthly on all earlier data, forecast the next month only, so every 2024–25 forecast below is out of sample. The model forecasts log(price ÷ gas) with gradient-boosted trees, and price = gas × exp(forecast). Only 2024–25 are forecast.</p>` +
    (idx.length ? kpis([["Model MAE", fmt$(mm, 2), `median error ${fmt$(medAE, 2)}`], ["Same hour yesterday", fmt$(mn, 2), "persistence benchmark"],
          ["Improvement", pct(1 - mm / mn), "lower MAE than benchmark"], [`P10–P90 coverage (${cal ? "calibrated" : "raw"})`, pct(cov, 1), "target 80%"]])
      : `<p class="note"><b>Select 2024 or 2025</b> to see forecast accuracy.</p>`) +
    `<div class="grid">` +
    card("q6a", "Forecast vs actual, one week", "Log scale.",
      {wide: true, ctl: `<div class="ctl">Week starting <input type="date" id="week" min="2024-01-01" max="2025-12-25" value="${state.week}">
        Band <select id="band"><option value="cal">Calibrated P10–P90</option><option value="raw">Raw quantile model</option></select></div>`}) +
    card("q6b", "Error by hour", "Mean absolute error, model vs persistence (log scale).") +
    card("q6c", "Interval calibration", "Share of hours at or below each forecast quantile.") +
    card("q6d", "What the model relies on", "Permutation importance, HB_NORTH model on the 2025 hold-out.", {tall: true}) +
    `<section class="card"><h3>Benchmarks, walk-forward 2024–25</h3><p class="cap">MAE in $/MWh, all hours (not filtered).</p><div id="q6t"></div></section></div>`;
  document.getElementById("week").onchange = e => { if (e.target.value) { state.week = e.target.value; render(); } };
  document.getElementById("band").value = state.band;
  document.getElementById("band").onchange = e => { state.band = e.target.value; render(); };

  const w0 = Math.round((new Date(state.week + "T00:00:00") - DAY0) / 864e5);
  const wk = range(0, M - 1).filter(i => P.d[i] >= w0 && P.d[i] < w0 + 7);
  const xt = wk.map(i => new Date(DAY0.getTime() + P.d[i] * 864e5 + P.he[i] * 36e5));
  plot("q6a", [
    {x: [...xt, ...xt.slice().reverse()], y: [...wk.map(i => hi[i]), ...wk.slice().reverse().map(i => lo[i])], fill: "toself", fillcolor: css("--band"), line: {width: 0}, name: "P10–P90", type: "scatter", hoverinfo: "skip"},
    {x: xt, y: wk.map(i => act[i]), name: "Actual", type: "scatter", mode: "lines", line: {color: css("--text"), width: 1.6}},
    {x: xt, y: wk.map(i => P[h + "_p50"][i]), name: "Model median", type: "scatter", mode: "lines", line: {color: css("--s1"), width: 2}},
    {x: xt, y: wk.map(i => P[h + "_naive"][i]), name: "Same hour yesterday", type: "scatter", mode: "lines", line: {color: css("--s2"), width: 1.3}}],
    {yaxis: {title: {text: `${HUBNAME[h]} $/MWh (log)`}, ...logTicks([5, 10, 30, 100, 300, 1000, 3000])}, hovermode: "x unified"});

  const byH = groupIdx(idx, i => P.he[i]);
  const maeH = arr => HRS.map(hh => { const g = (byH.get(hh) || []).filter(i => arr[i] != null); return g.length ? mean(g.map(i => Math.abs(arr[i] - act[i]))) : null; });
  plot("q6b", [line(HRS, maeH(P[h + "_naive"]), "Same hour yesterday", css("--s2")), line(HRS, maeH(P[h + "_p50"]), "Model", css("--s1"))],
    {xaxis: {title: {text: "Hour ending"}, dtick: 2}, yaxis: {title: {text: "MAE $/MWh (log)"}, ...logTicks([2, 5, 10, 20, 50, 100])}, hovermode: "x unified"});

  const shareBelow = arr => 100 * ci.filter(i => act[i] <= arr[i]).length / ci.length;
  plot("q6c", [
    {x: ["P10", "P50", "P90"], y: [10, 50, 90], name: "Target", type: "bar", marker: {color: css("--grid")}},
    {x: ["P10", "P50", "P90"], y: [shareBelow(P[h + "_p10"]), shareBelow(P[h + "_p50"]), shareBelow(P[h + "_p90"])], name: "Raw", type: "bar", marker: {color: css("--s2")}},
    {x: ["P10", "P50", "P90"], y: [shareBelow(P[h + "_p10c"]), shareBelow(P[h + "_p50"]), shareBelow(P[h + "_p90c"])], name: "Calibrated", type: "bar", marker: {color: css("--s1")}}],
    {barmode: "group", bargap: 0.25, bargroupgap: 0.06, yaxis: {title: {text: "% of hours at or below"}, range: [0, 100]}});

  const NAMES = {north_lag24: "Price yesterday, same hour", north_lag168: "Price last week, same hour", north_lagday: "Yesterday's daily avg", north_lag7d: "7-day avg price",
    net_load: "Net load", tightness: "Net load + outages", he: "Hour of day", nl_daymax: "Daily max net load", tight_daymax: "Daily max tightness",
    solar_frac: "Solar / recent max", renew_share: "Renewable share", solar_cap30: "Solar capacity proxy", wind_cap30: "Wind capacity proxy",
    temp_daymax: "Daily max temp", temp_daymin: "Daily min temp", nl_ramp3: "3-hour net-load ramp", capout: "Outages", dayoff: "Weekend/holiday",
    doy: "Day of year", dow: "Day of week", temp: "Temperature", load: "Load", wind: "Wind", solar: "Solar", gas: "Gas", month: "Month"};
  const imp = Object.entries(DATA.importance).sort((a, b) => a[1] - b[1]).slice(-12);
  plot("q6d", [{y: imp.map(([k]) => NAMES[k] || k), x: imp.map(([, v]) => v), type: "bar", orientation: "h", marker: {color: css("--s1")}, name: "",
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>"}], {xaxis: {title: {text: "Drop in R² when shuffled"}}, yaxis: {automargin: true, showgrid: false}, margin: {l: 8}});

  const B = DATA.benchmark;
  const models = ["Same hour yesterday", "Ridge, fundamentals", "GBM, fundamentals only", "Final GBM without price lags", "Final GBM (monthly refit, weighted)"];
  const get = (m, y, k) => { const r = B.find(b => b.hub === h && b.model === m && b.year === y); return r ? r[k] : NaN; };
  document.getElementById("q6t").innerHTML = `<table><thead><tr><th>${HUBNAME[h]}</th><th>MAE 2024</th><th>MAE 2025</th><th>Bias 2024</th><th>Bias 2025</th></tr></thead><tbody>` +
    models.map(m => `<tr class="${m.startsWith("Final GBM (") ? "hl" : ""}"><td>${m}</td><td>${fmtN(get(m, 2024, "MAE"), 2)}</td><td>${fmtN(get(m, 2025, "MAE"), 2)}</td><td>${fmtN(get(m, 2024, "bias"), 2)}</td><td>${fmtN(get(m, 2025, "bias"), 2)}</td></tr>`).join("") + `</tbody></table>`;
}

// ---------------------------------------------------------------- Q8
function renderQ8(main) {
  const idx = filtered(), p = price();
  const yrs = [...state.years].sort();
  const byY = groupIdx(idx, i => R.year[i]);
  main.innerHTML = `<p class="note">Value is concentrated in a few hours, and the market changed fast: load grew about 4.7% a year while solar tripled.</p>` +
    `<div class="grid">` +
    card("q8a", "Price duration curve", "Hours sorted from most to least expensive (both axes log).") +
    card("q8b", "Share of each year's total price sum", "Earned in the top 1% and top 5% of hours.") +
    card("q8c", "Annual averages", "Load grew, but solar absorbed it, so net load stayed flat.") +
    `<section class="card"><h3>Data notes</h3><ul class="obs">
      <li><b>Hour-ending timestamps:</b> 00:00 is HE24 of the previous day; DST shows as duplicated/missing hours.</li>
      <li><b>Solar scale error:</b> 20 Aug 2024 HE16 – 21 Aug HE16 read ~7× too high (up to 132 GW); 15 daylight hours repaired.</li>
      <li><b>Temperature gaps:</b> 103 hours, one 29-hour gap; straight line under 12 h, same hour on adjacent days for longer gaps.</li>
      <li><b>Mixed timing:</b> wind and solar are forecasts, load is actual; gas is daily and flat Sat–Mon.</li>
      <li><b>Only DFW temperature:</b> no Houston weather.</li>
      <li><b>Few negative prices:</b> 6 hours at North, 3 at Houston.</li></ul></section>` +
    `</div>`;

  plot("q8a", yrs.map(y => { const s = pick(byY.get(y) || [], p).sort((a, b) => b - a); const n = s.length;
    const step = Math.max(1, Math.floor(n / 600)); const ks = range(0, n - 1).filter(k => k < 200 || k % step === 0);
    return {x: ks.map(k => 100 * (k + 1) / n), y: ks.map(k => Math.max(s[k], 1)), name: String(y), type: "scatter", mode: "lines", line: {color: yearColor(y), width: 2}}; }),
    {xaxis: {title: {text: "Share of hours"}, type: "log", tickvals: [0.1, 1, 10, 100], ticktext: ["0.1%", "1%", "10%", "100%"]},
     yaxis: {title: {text: "$/MWh (log)"}, ...logTicks([3, 10, 30, 100, 300, 1000, 3000])}});

  const conc = (y, f) => { const s = pick(byY.get(y) || [], p).sort((a, b) => b - a); const tot = s.reduce((a, b) => a + b, 0);
    return 100 * s.slice(0, Math.floor(s.length * f)).reduce((a, b) => a + b, 0) / tot; };
  plot("q8b", [{x: yrs.map(String), y: yrs.map(y => conc(y, .01)), name: "Top 1% of hours", type: "bar", marker: {color: css("--s1")}},
    {x: yrs.map(String), y: yrs.map(y => conc(y, .05)), name: "Top 5% of hours", type: "bar", marker: {color: css("--s3")}}],
    {barmode: "group", bargap: 0.25, bargroupgap: 0.06, xaxis: {type: "category"}, yaxis: {title: {text: "% of total"}, ticksuffix: "%"}});

  plot("q8c", [["Load", R.load, "--neutral"], ["Net load", R.netload, "--s1"], ["Solar", R.solar, "--s2"], ["Wind", R.wind, "--s3"]].map(([name, arr, col]) =>
    line(yrs.map(String), yrs.map(y => mean(pick(byY.get(y) || [], arr))), name, css(col))),
    {xaxis: {type: "category"}, yaxis: {title: {text: "GW"}, rangemode: "tozero"}, hovermode: "x unified"});
}

// ---------------------------------------------------------------- router + theme
const RENDER = {q1: renderQ1, q2: renderQ2, q3: renderQ3, q4: renderQ4, q5: renderQ5, q6: renderQ6, q8: renderQ8};
function render() {
  const main = document.getElementById("main");
  document.querySelectorAll("#main .chart").forEach(el => { try { Plotly.purge(el); } catch (e) {} });
  RENDER[state.tab](main);
}
const THEMES = ["auto", "light", "dark"];
let theme = "auto";
function applyTheme() {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  document.getElementById("themeBtn").textContent = "Theme: " + theme;
}
document.getElementById("themeBtn").onclick = () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % 3];
  try { localStorage.setItem("ercot_theme", theme); } catch (e) {}
  applyTheme(); render();
};
try { theme = localStorage.getItem("ercot_theme") || "auto"; const t = localStorage.getItem("ercot_tab"); if (t && RENDER[t]) state.tab = t; } catch (e) {}
applyTheme();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (theme === "auto") render(); });
buildChrome();
render();
</script>
</body>
</html>
"""


def main():
    data = payload()
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob).replace("__PLOTLY__", po.get_plotlyjs())
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
