"""
ERCOT hub price analysis: HB_NORTH and HB_HOUSTON day-ahead LMPs, 2022-2025.

Run everything:        python ercot_analysis.py
Run one question:      python ercot_analysis.py --q 1
Refit price models:    python ercot_analysis.py --q 6 7 --refit  (~15 min)

Outputs
  figures/  PNG + PDF charts used in the write-up
  report/tables/  LaTeX tables generated from the data
  results/  JSON with the headline numbers quoted in the text
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Book2.xlsx"
FIG = ROOT / "figures"
TAB = ROOT / "report" / "tables"
RES = ROOT / "results"
for d in (FIG, TAB, RES):
    d.mkdir(parents=True, exist_ok=True)

COLS = {
    "DATETIME": "dt",
    "TX - Dallas-Fort Worth/Intl (TEMP_F)": "temp",
    "ERCOT (RTLOAD)": "load",
    "ERCOT (WIND_STWPF)": "wind",
    "ERCOT (SOLAR_STPPF)": "solar",
    "ERCOT (TOTAL_RESOURCE_CAP_OUT)": "capout",
    "Henry (GASPRICE)": "gas",
    "HB_NORTH (DALMP)": "north",
    "HB_HOUSTON (DALMP)": "houston",
}
HUBS = {"north": "HB_NORTH", "houston": "HB_HOUSTON"}
SPIKE = 100.0  # $/MWh threshold used for "high price" hour counts

# ---------------------------------------------------------------- chart style
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
CRIT = "#d03b3b"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", [C1, "#f0efec", "#e34948"])

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "text.color": INK,
})


def save(fig, name: str):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- data prep
def load_data(path: Path = DATA) -> pd.DataFrame:
    """Load the Dataset sheet, clean known data issues, add derived features."""
    df = pd.read_excel(path, sheet_name="Dataset").rename(columns=COLS)
    return add_features(clean(df))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Solar: a 24-hour block (20 Aug 2024 HE16 to 21 Aug HE16) is ~7x too
    # large - 132 GW peak vs ~20 GW on the days around it. Anchor on the
    # impossible values (>35 GW; the clean-data max is 29.7 GW), then flag
    # every hour within 12h of them that is >3x its same-hour reference
    # (median of the 3 days either side). Replace flagged hours with that
    # reference. The ratio test is only applied near the anchor: elsewhere it
    # would also catch genuine sunny-after-cloudy days.
    hour = df["dt"].dt.hour
    day = df["dt"].dt.normalize()
    piv = df.pivot_table(index=day, columns=hour, values="solar", aggfunc="first")
    ref = (pd.concat([piv.shift(k) for k in (-3, -2, -1, 1, 2, 3)])
           .groupby(level=0).median())
    ref = ref.stack().reindex(pd.MultiIndex.from_arrays([day, hour])).values
    anchor = df["solar"] > 35_000
    bad = pd.Series(False, index=df.index)
    if anchor.any():
        t0, t1 = df.loc[anchor, "dt"].min(), df.loc[anchor, "dt"].max()
        near = df["dt"].between(t0 - pd.Timedelta(hours=12),
                                t1 + pd.Timedelta(hours=12))
        bad = near & (df["solar"] > 3 * ref) & (df["solar"] - ref > 500)
    df["solar_flag"] = bad
    df.loc[bad, "solar"] = ref[bad.values]
    # Temperature: 103 missing hours in 49 gaps, all short except one of 29h
    # (16-17 Oct 2023). Gaps under 12h: straight line between the readings
    # either side. Gaps of 12h+: average of the same hour on the day before
    # and the day after, which keeps the day/night cycle a straight line
    # would erase. 12h is where the two methods' backtest errors cross.
    t = df.set_index("dt")["temp"]
    na = t.isna()
    gap_len = na.groupby((~na).cumsum()).transform("sum")
    linear = t.interpolate(method="time", limit_direction="both")
    day = pd.Timedelta(days=1)
    nbr = pd.concat([t.groupby(level=0).first().shift(freq=s).reindex(t.index)
                     for s in (day, -day)], axis=1).mean(axis=1)
    long_gap = na & (gap_len >= 12) & nbr.notna()
    df["temp_flag"] = na.values
    df["temp"] = linear.where(~long_gap, nbr).values
    # Outages: 4 missing hours, all DST fall-back duplicates -> carry forward.
    df["capout"] = df["capout"].ffill()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamps are hour-ending (01:00 = HE1, 00:00 = HE24 of the prior day)."""
    df = df.copy()
    shifted = df["dt"] - pd.Timedelta(hours=1)
    df["date"] = shifted.dt.normalize()
    df["he"] = shifted.dt.hour + 1
    df["month"] = shifted.dt.month
    df["year"] = shifted.dt.year
    df["dow"] = shifted.dt.dayofweek
    df["weekend"] = df["dow"] >= 5
    df["season"] = df["month"].map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
         5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
         10: "Fall", 11: "Fall"})
    # Peak = HE7-22 weekdays (ERCOT 5x16 convention).
    df["onpeak"] = (~df["weekend"]) & df["he"].between(7, 22)
    df["renew"] = df["wind"] + df["solar"]
    df["net_load"] = df["load"] - df["renew"]
    # Net load plus capacity on outage: how much of the thermal fleet is used
    # up, either by serving demand or by sitting offline.
    df["tightness"] = df["net_load"] + df["capout"]
    df["renew_share"] = df["renew"] / df["load"]
    for h in HUBS:
        df[f"{h}_ihr"] = df[h] / df["gas"]  # implied heat rate, MMBtu/MWh
    df["spread"] = df["houston"] - df["north"]
    return df[df["year"].between(2022, 2025)].reset_index(drop=True)


# -------------------------------------------------------------------- Q1
TIERS = ["< P50", "P50-P90", "P90-P99", "> P99"]
TIER_VARS = [
    ("north", "HB\\_NORTH (\\$/MWh)", "{:,.0f}"),
    ("houston", "HB\\_HOUSTON (\\$/MWh)", "{:,.0f}"),
    ("north_ihr", "Implied heat rate N (MMBtu/MWh)", "{:,.1f}"),
    ("temp", "DFW temperature ($^\\circ$F)", "{:,.0f}"),
    ("load", "Load (GW)", "{:,.1f}"),
    ("wind", "Wind (GW)", "{:,.1f}"),
    ("solar", "Solar (GW)", "{:,.1f}"),
    ("net_load", "Net load (GW)", "{:,.1f}"),
    ("capout", "Capacity on outage (GW)", "{:,.1f}"),
    ("tightness", "Net load + outages (GW)", "{:,.1f}"),
    ("gas", "Henry Hub gas (\\$/MMBtu)", "{:,.2f}"),
]
GW = {"load", "wind", "solar", "net_load", "capout", "tightness"}


def price_tiers(s: pd.Series) -> pd.Series:
    q = s.quantile([0.5, 0.9, 0.99]).values
    return pd.cut(s, [-np.inf, *q, np.inf], labels=TIERS)


def q1(df: pd.DataFrame) -> dict:
    out: dict = {}
    df = df.copy()
    df["tier"] = price_tiers(df["north"])
    out["tier_cuts_north"] = df["north"].quantile([0.5, 0.9, 0.99]).round(1).tolist()
    out["hub_corr"] = round(df["north"].corr(df["houston"]), 3)
    out["n_spike"] = int((df["north"] > SPIKE).sum())

    # 1. Conditions table: median fundamentals by price tier ---------------
    med = df.groupby("tier", observed=True)[[v for v, *_ in TIER_VARS]].median()
    shares = df.groupby("tier", observed=True).agg(
        summer=("season", lambda s: (s == "Summer").mean()),
        winter=("season", lambda s: (s == "Winter").mean()),
        evening=("he", lambda s: s.between(17, 21).mean()),
        hours=("he", "size"))
    out["tier_medians"] = med.round(2).to_dict()
    out["tier_shares"] = shares.round(3).to_dict()
    rows = []
    for v, label, fmt in TIER_VARS:
        vals = med[v] / (1000 if v in GW else 1)
        rows.append(f"{label} & " + " & ".join(fmt.format(x) for x in vals) + r" \\")
    rows.append(r"\midrule")
    for k, label in [("summer", "Share in Jun--Sep"), ("winter", "Share in Dec--Feb"),
                     ("evening", "Share in HE17--21")]:
        rows.append(f"{label} & " + " & ".join(f"{x:.0%}" for x in shares[k])
                    .replace("%", r"\%") + r" \\")
    rows.append("Hours & " + " & ".join(f"{x:,}" for x in shares["hours"]) + r" \\")
    cuts = out["tier_cuts_north"]
    hdr = (f"$<$\\${cuts[0]:.0f} & \\${cuts[0]:.0f}--{cuts[1]:.0f} & "
           f"\\${cuts[1]:.0f}--{cuts[2]:.0f} & $>$\\${cuts[2]:.0f}")
    (TAB / "q1_conditions.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        " & Bottom 50\\% & P50--P90 & P90--P99 & Top 1\\% \\\\\n"
        f"Median value (HB\\_NORTH price band) & {hdr} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # 2. Standardised shift of each fundamental in the top-10% hours --------
    drivers = ["temp", "load", "wind", "solar", "net_load", "capout",
               "tightness", "gas", "renew_share"]
    labels = {"temp": "Temperature", "load": "Load", "wind": "Wind",
              "solar": "Solar", "net_load": "Net load", "capout": "Outages",
              "tightness": "Net load + outages", "gas": "Gas price",
              "renew_share": "Renewable share"}
    hi = df["tier"].isin(["P90-P99", "> P99"])
    shift = ((df.loc[hi, drivers].mean() - df[drivers].mean()) / df[drivers].std())
    # Rank correlation with price and with implied heat rate.
    sp_price = df[drivers + ["north"]].corr("spearman")["north"][drivers]
    sp_ihr = df[drivers + ["north_ihr"]].corr("spearman")["north_ihr"][drivers]
    out["z_shift_top10"] = shift.round(2).to_dict()
    out["spearman_price"] = sp_price.round(2).to_dict()
    out["spearman_ihr"] = sp_ihr.round(2).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4),
                             gridspec_kw={"wspace": 0.55})
    order = sp_price.sort_values().index
    for ax, s, title in [(axes[0], sp_price[order], "Rank correlation with HB_NORTH price"),
                         (axes[1], shift[order], "Shift in top-10% price hours (z-score)")]:
        cols = [C1 if v >= 0 else C2 for v in s]
        ax.barh([labels[i] for i in s.index], s.values, color=cols, height=0.6)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for y_, v in enumerate(s.values):
            ax.text(v + (0.02 if v >= 0 else -0.02) * (abs(s).max() / 0.6), y_,
                    f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7, color=INK2)
        lim = abs(s).max() * 1.35
        ax.set_xlim(-lim, lim)
    save(fig, "q1_drivers")

    # 3. Price = gas x implied heat rate. Split log-price variance between the
    #    two, then explain the heat rate (the grid-conditions part) with a
    #    gradient-boosting model. Net load replaces its collinear parts;
    #    leave-one-year-out so every year is scored out of sample.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    pos = df["north"] > 1
    lp, lg = np.log(df.loc[pos, "north"]), np.log(df.loc[pos, "gas"])
    lh = lp - lg
    out["logvar_share"] = {
        "gas": round(lg.var() / lp.var(), 3),
        "heat_rate": round(lh.var() / lp.var(), 3),
        "covariance": round(2 * np.cov(lg, lh)[0, 1] / lp.var(), 3)}
    # Daily-average view: how much of day-to-day price level is gas?
    d = df[pos].groupby("date")[["north", "gas"]].mean()
    out["daily_logprice_r2_on_loggas"] = round(
        np.corrcoef(np.log(d["north"]), np.log(d["gas"]))[0, 1] ** 2, 3)

    feats = ["net_load", "capout", "temp", "he", "month", "weekend"]
    flabels = {**labels, "he": "Hour of day", "month": "Month",
               "weekend": "Weekend"}
    y = np.log(df["north_ihr"].clip(lower=0.2))
    imps, r2 = [], {}
    for yr in sorted(df["year"].unique()):
        tr, te = df["year"] != yr, df["year"] == yr
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                            max_leaf_nodes=31, random_state=0)
        gbm.fit(df.loc[tr, feats], y[tr])
        r2[int(yr)] = round(gbm.score(df.loc[te, feats], y[te]), 3)
        pi = permutation_importance(gbm, df.loc[te, feats], y[te], n_repeats=5,
                                    random_state=0, scoring="r2")
        imps.append(pi.importances_mean)
    imp = pd.Series(np.mean(imps, axis=0), index=feats).sort_values()
    out["gbm_r2_logihr_by_heldout_year"] = r2
    out["perm_importance"] = imp.round(3).to_dict()

    # 4. Outages only show up once net load is held fixed: in the raw data
    #    they are high in mild shoulder months (planned maintenance).
    df["out_tercile"] = pd.qcut(df["capout"], 3,
                                labels=["Low outages", "Mid outages", "High outages"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    nb = np.arange(15_000, 70_001, 5_000)
    for t, col in zip(df["out_tercile"].cat.categories, [C1, C3, C2]):
        g = (df[df["out_tercile"] == t].groupby(pd.cut(df["net_load"], nb),
                                                observed=True)["north_ihr"]
             .agg(["median", "size"]))
        g = g[g["size"] >= 50]
        ax.plot([b.mid / 1000 for b in g.index], g["median"], color=col, label=t,
                marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80, 160], ["5", "10", "20", "40", "80", "160"])
    ax.set_xlabel("Net load (GW)")
    ax.set_ylabel("Median implied heat rate (MMBtu/MWh)")
    ax.set_title("At the same net load, outages lift price")
    ax.legend(loc="upper left")
    q_out = df.groupby("out_tercile", observed=True)["capout"].agg(["min", "max"])
    out["outage_terciles_gw"] = (q_out / 1000).round(1).to_dict("index")
    hi_nl = df["net_load"].between(45_000, 55_000)
    out["ihr_45_55gw_by_outage"] = (df[hi_nl].groupby("out_tercile", observed=True)
                                    ["north_ihr"].median().round(1).to_dict())

    ax = axes[1]
    ax.barh([flabels[i] for i in imp.index], imp.values, color=C1, height=0.6)
    for y_, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.02, y_, f"{v:.2f}", va="center", fontsize=7,
                color=INK2)
    ax.set_xlim(0, imp.max() * 1.18)
    ax.set_title("What drives the heat rate (leave-one-year-out)")
    ax.set_xlabel("Drop in R$^2$ of log heat rate when shuffled")
    ax.grid(axis="y", visible=False)
    save(fig, "q1_outages_importance")

    # 4. The supply curve: price vs net load, by gas regime ----------------
    bins = np.arange(15_000, 75_001, 2_500)
    df["nl_bin"] = pd.cut(df["net_load"], bins)
    df["gas_reg"] = pd.cut(df["gas"], [0, 3, 5, 99],
                           labels=["Gas < $3", "Gas $3-5", "Gas > $5"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    for reg, col in zip(df["gas_reg"].cat.categories, [C1, C3, C2]):
        g = (df[df["gas_reg"] == reg].groupby("nl_bin", observed=True)["north"]
             .agg(["median", "size"]))
        g = g[g["size"] >= 30]
        ax.plot([b.mid / 1000 for b in g.index], g["median"], color=col, label=reg)
    ax.set_yscale("log")
    ax.set_ylim(5, 1000)
    ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500", "1000"])
    ax.set_xlabel("Net load = load − wind − solar (GW)")
    ax.set_ylabel("Median HB_NORTH price ($/MWh, log)")
    ax.set_title("Price follows net load; gas sets the level")
    ax.legend(loc="upper left")
    ax.set_xlim(15, 70)

    ax = axes[1]
    tb = np.arange(30_000, 100_001, 5_000)
    for col_name, col, lab in [("net_load", C1, "Net load"),
                               ("tightness", C2, "Net load + outages")]:
        g = df.groupby(pd.cut(df[col_name], tb), observed=True)["north"].agg(
            p=lambda s: (s > SPIKE).mean(), n="size")
        g = g[g["n"] >= 30]
        ax.plot([b.mid / 1000 for b in g.index], g["p"] * 100, color=col,
                label=lab, marker="o", ms=4)
    ax.set_xlabel("Net load, or net load + capacity on outage (GW)")
    ax.set_ylabel(f"Hours with HB_NORTH > ${SPIKE:.0f} (%)")
    ax.set_title(f"Chance of a >${SPIKE:.0f} hour")
    ax.legend(loc="upper left")
    save(fig, "q1_supply_curve")

    out["spike_prob_by_tightness"] = (
        df.groupby(pd.cut(df["tightness"], tb), observed=True)["north"]
        .apply(lambda s: round((s > SPIKE).mean(), 3)).astype(float)
        .rename(lambda b: f"{b.left/1000:.0f}-{b.right/1000:.0f}GW").to_dict())

    # 5. When do high prices happen: month x hour-ending heatmap -----------
    p = df.pivot_table(index="month", columns="he", values="north",
                       aggfunc=lambda s: (s > SPIKE).mean() * 100)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6),
                             gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    ax = axes[0]
    im = ax.imshow(p.values, aspect="auto", cmap=BLUES, vmin=0)
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(1, 25, 2)])
    ax.set_yticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                              "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Hour ending")
    ax.grid(False)
    ax.set_title(f"Share of hours with HB_NORTH > ${SPIKE:.0f} (%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED)

    # Temperature U-shape.
    ax = axes[1]
    tbins = np.arange(10, 115, 5)
    g = df.groupby(pd.cut(df["temp"], tbins), observed=True).agg(
        north=("north", "median"), houston=("houston", "median"),
        p=("north", lambda s: (s > SPIKE).mean() * 100), n=("north", "size"))
    g = g[g["n"] >= 30]
    x = [b.mid for b in g.index]
    ax.plot(x, g["north"], color=C1, label="HB_NORTH")
    ax.plot(x, g["houston"], color=C2, label="HB_HOUSTON")
    ax.set_xlabel("DFW temperature (°F)")
    ax.set_ylabel("Median price ($/MWh)")
    ax.set_title("Price vs temperature")
    ax.legend(loc="upper center")
    save(fig, "q1_when")
    out["temp_bins"] = g.round(1).rename(lambda b: f"{b.left}-{b.right}F").drop(
        columns="n").to_dict()
    out["min_price_temp_bin"] = str(g["north"].idxmin())

    # 6. Context: monthly price and gas, stacked panels (no dual axis) -----
    m = df.groupby(df["date"].dt.to_period("M")).agg(
        north=("north", "mean"), houston=("houston", "mean"), gas=("gas", "mean"),
        ihr=("north_ihr", "median"), p99=("north", lambda s: s.quantile(0.99)))
    xm = m.index.to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15})
    ax = axes[0]
    ax.plot(xm, m["north"], color=C1, label="HB_NORTH")
    ax.plot(xm, m["houston"], color=C2, label="HB_HOUSTON")
    ax.set_ylabel("Monthly avg ($/MWh)")
    ax.set_title("Monthly average day-ahead price")
    ax.legend(loc="upper right", ncol=2)
    for t, txt in [("2022-07", "Jul-22 heat +\n$7-9 gas"), ("2022-12", "Storm Elliott"),
                   ("2023-08", "Aug-23 scarcity"), ("2024-01", "Storm Heather")]:
        tt = pd.Period(t).to_timestamp()
        ax.annotate(txt, (tt, m.loc[t, "north"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(0, m["north"].max() * 1.25)
    axes[1].plot(xm, m["gas"], color=MUTED)
    axes[1].set_ylabel("Henry Hub ($/MMBtu)")
    axes[1].set_title("Monthly average gas price")
    save(fig, "q1_context")
    out["monthly"] = {str(k): v for k, v in m.round(2).to_dict("index").items()}
    out["yearly"] = (df.groupby("year")[["north", "houston", "gas", "load",
                                         "net_load", "solar", "wind"]]
                     .mean().round(1).to_dict("index"))
    out["spike_hours_by_year"] = (df.groupby("year")["north"]
                                  .apply(lambda s: int((s > SPIKE).sum())).to_dict())
    out["spike_hours_by_season"] = (df.groupby("season")["north"]
                                    .apply(lambda s: int((s > SPIKE).sum())).to_dict())
    return out


# -------------------------------------------------------------------- Q2
YEAR_COLS = {2022: "#86b6ef", 2023: "#3987e5", 2024: "#1c5cab", 2025: "#0d366b"}
CHEAP = 10.0  # $/MWh threshold for "renewables-depressed" hours


def hourly_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Per hour-ending OLS of log price on fundamentals.

    log(price) = wind + solar + load + outages + log(gas) + month FE + year FE
    Coefficients x100 = % change in price per +1 GW. Year effects stop the
    solar term picking up anything else that grew 2022-25 (e.g. batteries).
    Solar enters only in hours where it averages >2 GW (HE9-19). HAC errors
    (7 lags = one week) allow for day-to-day autocorrelation.
    """
    import statsmodels.formula.api as smf
    d0 = df.assign(lp=np.log(df["north"].clip(lower=1)), lg=np.log(df["gas"]),
                   wind_gw=df["wind"] / 1e3, solar_gw=df["solar"] / 1e3,
                   load_gw=df["load"] / 1e3, cap_gw=df["capout"] / 1e3)
    rows = []
    for h in range(1, 25):
        d = d0[d0["he"] == h]
        has_solar = d["solar_gw"].mean() > 2.0  # HE9-19
        f = ("lp ~ wind_gw + load_gw + cap_gw + lg + C(month) + C(year)"
             + (" + solar_gw" if has_solar else ""))
        m = smf.ols(f, d).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        row = {"he": h, "r2": m.rsquared, "median_price": d["north"].median()}
        for v in ["wind_gw", "solar_gw", "load_gw", "cap_gw"]:
            if v in m.params:
                lo, hi = m.conf_int().loc[v]
                row.update({v: m.params[v] * 100, f"{v}_lo": lo * 100,
                            f"{v}_hi": hi * 100})
        rows.append(row)
    return pd.DataFrame(rows).set_index("he").astype(float)


def q2(df: pd.DataFrame) -> dict:
    out: dict = {}
    hrs = range(1, 25)

    # 1. Shape of the day: when wind and solar show up, and how net load and
    #    the heat rate have reshaped as solar grew.
    prof = df.groupby("he")[["load", "wind", "solar", "net_load"]].mean() / 1e3
    nl_y = df.pivot_table(index="he", columns="year", values="net_load") / 1e3
    ihr_y = df.pivot_table(index="he", columns="year", values="north_ihr",
                           aggfunc="median")
    out["profile_gw"] = prof.round(1).to_dict()
    out["net_load_by_year_he"] = nl_y.round(1).to_dict()
    out["ihr_by_year_he"] = ihr_y.round(1).to_dict()
    out["evening_ramp_gw"] = {int(y): round(nl_y[y].loc[17:21].max()
                                            - nl_y[y].loc[10:14].min(), 1)
                              for y in nl_y}
    out["peak_ihr_he"] = {int(y): int(ihr_y[y].idxmax()) for y in ihr_y}
    out["solar_mean_gw_by_year"] = (df.groupby("year")["solar"].mean() / 1e3).round(1).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    for c, col, lab in [("load", MUTED, "Load"), ("wind", C1, "Wind"),
                        ("solar", C2, "Solar")]:
        ax.plot(hrs, prof[c], color=col, label=lab)
    ax.set_title("Average profile, 2022–25")
    ax.set_ylabel("GW")
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.62))
    for ax, tab, title, ylab in [
            (axes[1], nl_y, "Net load by year", "GW"),
            (axes[2], ihr_y, "Implied heat rate by year", "MMBtu/MWh (median)")]:
        for y, col in YEAR_COLS.items():
            ax.plot(hrs, tab[y], color=col, label=str(y), lw=1.8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Hour ending")
    save(fig, "q2_profiles")

    # 2. Price impact of +1 GW of wind or solar, by hour.
    eff = hourly_effects(df)
    out["effects_pct_per_gw"] = eff[["wind_gw", "solar_gw", "load_gw", "cap_gw",
                                     "r2"]].round(2).to_dict()
    for v in ["wind_gw", "solar_gw"]:
        eff[f"{v}_usd"] = eff[v] / 100 * eff["median_price"]
    out["effects_usd_per_gw_at_median"] = eff[["wind_gw_usd", "solar_gw_usd"]].round(2).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    for v, col, lab in [("wind_gw", C1, "Wind"), ("solar_gw", C2, "Solar")]:
        e = eff[v].dropna()
        ax.fill_between(e.index.to_numpy(float),
                        eff.loc[e.index, f"{v}_lo"].to_numpy(float),
                        eff.loc[e.index, f"{v}_hi"].to_numpy(float),
                        color=col, alpha=0.15, lw=0)
        ax.plot(e.index, e, color=col, label=lab, marker="o", ms=3)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("Price change per +1 GW")
    ax.set_ylabel("% change in price (95% CI)")
    ax.legend(loc="lower left")
    ax = axes[1]
    for v, col, lab in [("wind_gw", C1, "Wind"), ("solar_gw", C2, "Solar")]:
        e = eff[f"{v}_usd"].dropna()
        ax.plot(e.index, e, color=col, label=lab, marker="o", ms=3)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("Same, in $ at that hour's median")
    ax.set_ylabel("$/MWh per +1 GW")
    ax.legend(loc="lower left")
    ax = axes[2]
    cheap = df.groupby("he")["north"].apply(lambda s: (s < CHEAP).mean() * 100)
    ax.bar(cheap.index, cheap.values, color=C1, width=0.75)
    ax.set_title(f"Hours below ${CHEAP:.0f}/MWh")
    ax.set_ylabel("% of hours")
    ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Hour ending")
    save(fig, "q2_effects")

    # 3. The cheap hours are the renewable hours.
    c = df["north"] < CHEAP
    out["cheap_hours"] = int(c.sum())
    out["cheap_by_year"] = df[c].groupby("year").size().to_dict()
    out["cheap_share_by_season"] = df[c]["season"].value_counts(normalize=True).round(2).to_dict()
    out["renew_share_median"] = {"cheap": round(df.loc[c, "renew_share"].median(), 2),
                                 "all": round(df["renew_share"].median(), 2)}
    out["wind_median_gw"] = {"cheap": round(df.loc[c, "wind"].median() / 1e3, 1),
                             "all": round(df["wind"].median() / 1e3, 1)}

    # 4. A GW is worth more when the system is tight: slope of the median
    #    low-gas supply curve (Q1) at different net-load levels.
    lo_gas = df[df["gas"] < 3]
    curve = lo_gas.groupby(pd.cut(lo_gas["net_load"], np.arange(20_000, 72_501, 5_000)),
                           observed=True)["north"].agg(["median", "size"])
    curve = curve[curve["size"] >= 30]
    slope = curve["median"].diff() / 5
    out["supply_curve_slope_usd_per_gw"] = (
        slope.dropna().round(2).rename(lambda b: f"{b.left/1e3:.0f}-{b.right/1e3:.0f}GW").to_dict())
    return out


# -------------------------------------------------------------------- Q3
def cvar(s: pd.Series, q: float = 0.95) -> float:
    """Expected shortfall for a short-power position: mean of the top 5%."""
    return s[s >= s.quantile(q)].mean()


def q3(df: pd.DataFrame) -> dict:
    out: dict = {}
    hrs = range(1, 25)
    g = df.groupby("he")["north"]
    risk = pd.DataFrame({"median": g.median(), "p95": g.quantile(0.95),
                         "cvar95": g.apply(cvar), "std": g.std(),
                         "p_gt100": g.apply(lambda s: (s > SPIKE).mean())})
    out["risk_by_he"] = risk.round(2).to_dict()
    cv_y = df.pivot_table(index="he", columns="year", values="north", aggfunc=cvar)
    ihr20 = df.pivot_table(index="year", columns="he", values="north_ihr",
                           aggfunc=lambda s: (s > 20).mean() * 100)
    cv_s = df.pivot_table(index="season", columns="he", values="north",
                          aggfunc=cvar).loc[["Winter", "Spring", "Summer", "Fall"]]
    out["cvar95_by_year_he"] = cv_y.round(0).to_dict()
    out["peak_cvar_he_by_year"] = {int(y): int(cv_y[y].idxmax()) for y in cv_y}
    out["p_ihr_gt20_by_year_he"] = ihr20.round(1).to_dict("index")
    out["cvar95_by_season_he"] = cv_s.round(0).to_dict("index")
    out["peak_cvar_he_by_season"] = cv_s.idxmax(axis=1).to_dict()
    # Rank correlation of the 24-hour risk profile between years: is the
    # shape of risk stable even if the level moves?
    out["risk_profile_rank_corr"] = cv_y.corr("spearman").round(2).to_dict()
    # Share of each year's >$100 hours falling in each block of the day.
    blocks = pd.cut(df["he"], [0, 6, 9, 13, 17, 21, 24],
                    labels=["HE1-6", "HE7-9", "HE10-13", "HE14-17", "HE18-21", "HE22-24"])
    sp = df["north"] > SPIKE
    out["spike_share_by_block_year"] = (pd.crosstab(df.loc[sp, "year"], blocks[sp],
                                                    normalize="index").round(2)
                                        .to_dict("index"))
    wk = df.pivot_table(index="weekend", columns="he", values="north", aggfunc=cvar)
    out["cvar95_weekday_vs_weekend_he20"] = {"weekday": round(wk.loc[False, 20]),
                                             "weekend": round(wk.loc[True, 20])}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    for c, col, lab in [("median", MUTED, "Median"), ("p95", C1, "95th percentile"),
                        ("cvar95", C2, "Mean of top 5% (CVaR)")]:
        ax.plot(hrs, risk[c], color=col, label=lab, marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_yticks([20, 50, 100, 200, 500, 1000], ["20", "50", "100", "200", "500", "1000"])
    ax.set_ylabel("HB_NORTH $/MWh (log)")
    ax.set_title("Price risk by hour, 2022–25")
    ax.legend(loc="upper left")
    ax = axes[1]
    for y, col in YEAR_COLS.items():
        ax.plot(hrs, cv_y[y], color=col, label=str(y), lw=1.8)
    ax.set_yscale("log")
    ax.set_yticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1000", "3000"])
    ax.set_ylabel("Mean of top 5% hours, $/MWh (log)")
    ax.set_title("CVaR by hour and year")
    ax.legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Hour ending")
    save(fig, "q3_risk_by_hour")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9), gridspec_kw={"wspace": 0.32})
    for i, (ax, tab, title) in enumerate([
            (axes[0], ihr20, "Hours with implied heat rate > 20 (%), gas-neutral"),
            (axes[1], cv_s, "Mean of top 5% prices by season ($/MWh, log colour)")]):
        vals = tab.values.astype(float)
        norm = mpl.colors.LogNorm(vmin=max(vals.min(), 1), vmax=vals.max()) if i else None
        im = ax.imshow(vals, aspect="auto", cmap=BLUES, norm=norm)
        ax.set_yticks(range(len(tab)), [str(r) for r in tab.index])
        ax.set_xticks(range(0, 24, 2), [str(h) for h in range(1, 25, 2)])
        ax.set_xlabel("Hour ending")
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
    save(fig, "q3_heatmaps")
    return out


# -------------------------------------------------------------------- Q4
STRIKES = {"north": 50.0, "houston": 53.0}
MW = 50


def q4(df: pd.DataFrame) -> dict:
    """50 MW ATC 2022-25 bought 1 Jan 2021, settled hourly against DA LMP."""
    out: dict = {"hours": int(len(df)), "mwh": int(len(df) * MW)}
    hubs = list(STRIKES)
    pnl = pd.DataFrame({h: (df[h] - k) * MW for h, k in STRIKES.items()})
    pnl["year"], pnl["month"] = df["year"], df["date"].dt.to_period("M")
    yr = pnl.groupby("year")[hubs].sum() / 1e6
    avg = df.groupby("year")[hubs].mean()
    out["avg_price_by_year"] = avg.round(2).to_dict()
    out["avg_price_total"] = df[hubs].mean().round(2).to_dict()
    out["pnl_musd_by_year"] = yr.round(2).to_dict()
    out["pnl_musd_total"] = (pnl[hubs].sum() / 1e6).round(2).to_dict()
    m = pnl.groupby("month")[hubs].sum() / 1e6
    cum = m.cumsum()
    out["months_positive"] = {h: int((m[h] > 0).sum()) for h in hubs}
    out["cum_peak"] = {h: [str(cum[h].idxmax()), round(cum[h].max(), 2)] for h in hubs}
    out["best_month"] = {h: [str(m[h].idxmax()), round(m[h].max(), 2)] for h in hubs}
    out["worst_month"] = {h: [str(m[h].idxmin()), round(m[h].min(), 2)] for h in hubs}
    out["pnl_from_hours_gt100_musd"] = {
        h: round(pnl.loc[df[h] > SPIKE, h].sum() / 1e6, 2) for h in hubs}
    out["share_hours_gt100"] = round((df["north"] > SPIKE).mean(), 4)
    aug23 = (df["year"] == 2023) & (df["month"] == 8)
    out["pnl_aug2023_musd"] = {h: round(pnl.loc[aug23, h].sum() / 1e6, 2) for h in hubs}
    out["pnl_ex_aug2023_musd"] = {h: round(pnl.loc[~aug23, h].sum() / 1e6, 2) for h in hubs}
    # Houston strip = North strip + a Houston-North spread bought at $3.
    sp = df["spread"]
    out["spread_paid_vs_realised"] = [3.0, round(sp.mean(), 2)]
    out["spread_leg_pnl_musd"] = round((sp - 3).sum() * MW / 1e6, 2)
    out["spread_by_year"] = df.groupby("year")["spread"].mean().round(2).to_dict()
    # Strike expressed as a heat rate against realised gas.
    out["avg_gas"] = round(df["gas"].mean(), 2)
    out["strike_ihr_at_realised_gas"] = round(50 / df["gas"].mean(), 1)
    out["realised_atc_ihr"] = round(df["north"].mean() / df["gas"].mean(), 1)

    rows = []
    for y in sorted(df["year"].unique()):
        rows.append(f"{y} & {avg.loc[y, 'north']:.2f} & {yr.loc[y, 'north']:+.2f} & "
                    f"{avg.loc[y, 'houston']:.2f} & {yr.loc[y, 'houston']:+.2f} & "
                    f"{df.loc[df['year'] == y, 'spread'].mean():.2f} \\\\")
    tot = out["pnl_musd_total"]
    rows += [r"\midrule",
             f"2022--25 & {df['north'].mean():.2f} & {tot['north']:+.2f} & "
             f"{df['houston'].mean():.2f} & {tot['houston']:+.2f} & {sp.mean():.2f} \\\\"]
    (TAB / "q4_pnl.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{HB\\_NORTH @ \\$50} & "
        "\\multicolumn{2}{c}{HB\\_HOUSTON @ \\$53} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Year & Avg price & P\\&L (\\$M) & Avg price & P\\&L (\\$M) & Avg spread \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
    ax = axes[0]
    x = cum.index.to_timestamp()
    for h, col in [("north", C1), ("houston", C2)]:
        ax.plot(x, cum[h], color=col, label=f"{HUBS[h]} @ ${STRIKES[h]:.0f}")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Cumulative P&L ($M)")
    ax.set_title("Cumulative P&L, 50 MW ATC")
    ax.legend(loc="lower left")
    pk = cum["north"].idxmax()
    ax.annotate(f"Peak +${cum['north'].max():.1f}M ({pk.strftime('%b %Y')})",
                (pk.to_timestamp(), cum["north"].max()), xytext=(10, 2),
                textcoords="offset points", fontsize=7, color=INK2)
    ax = axes[1]
    yrs = yr.index.to_numpy(int)
    w = 0.38
    ax.bar(yrs - w / 2 - 0.01, yr["north"], w, color=C1, label="HB_NORTH")
    ax.bar(yrs + w / 2 + 0.01, yr["houston"], w, color=C2, label="HB_HOUSTON")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(yrs, [str(y) for y in yrs])
    ax.set_ylabel("P&L ($M)")
    ax.set_title("P&L by year")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    save(fig, "q4_pnl")
    return out


# -------------------------------------------------------------------- Q5
def q5(df: pd.DataFrame) -> dict:
    """Spread = HB_HOUSTON - HB_NORTH (convention: Houston less North)."""
    out: dict = {}
    sp = df["spread"]
    out["summary"] = sp.describe(percentiles=[.01, .05, .5, .95, .99]).round(2).to_dict()
    out["share_positive"] = round((sp > 0).mean(), 3)
    out["by_year"] = df.groupby("year")["spread"].agg(["mean", "std"]).round(2).to_dict()
    out["by_month"] = df.groupby("month")["spread"].agg(["mean", "std"]).round(2).to_dict()
    byhe = df.groupby("he")["spread"]
    risk = pd.DataFrame({"mean": byhe.mean(), "std": byhe.std(),
                         "p01": byhe.quantile(0.01), "p99": byhe.quantile(0.99),
                         "p05": byhe.quantile(0.05), "p95": byhe.quantile(0.95)})
    out["by_he"] = risk.round(2).to_dict()
    drivers = ["temp", "load", "wind", "solar", "net_load", "capout", "gas",
               "renew_share", "north"]
    out["spearman"] = (df[drivers + ["spread"]].corr("spearman")["spread"][drivers]
                       .round(2).to_dict())
    q = {}
    for v in ["wind", "capout", "load"]:
        b = pd.qcut(df[v], 5, labels=[f"Q{i}" for i in range(1, 6)])
        t = df.groupby(b, observed=True)["spread"].agg(["mean", "median", "std"])
        t["p95"] = df.groupby(b, observed=True)["spread"].quantile(0.95)
        q[v] = t.round(2).to_dict("index")
    out["by_quintile"] = q
    cold = df["temp"] < 32
    out["cold_hours"] = {"n": int(cold.sum()), "mean_spread": round(sp[cold].mean(), 2),
                         "p05": round(sp[cold].quantile(0.05), 2)}
    top = df.loc[sp.abs().nlargest(400).index]
    out["largest_400_abs"] = {
        "positive_share": round((top["spread"] > 0).mean(), 2),
        "months_pos": top.loc[top["spread"] > 0, "month"].value_counts().head(4).to_dict(),
        "months_neg": top.loc[top["spread"] < 0, "month"].value_counts().head(4).to_dict(),
        "he_neg": top.loc[top["spread"] < 0, "he"].value_counts().head(4).to_dict()}
    # Multivariate check: OLS of spread on standardised fundamentals with hour,
    # month and year effects; HAC errors (24 lags = one day).
    import statsmodels.formula.api as smf
    z = df.copy()
    for v in ["wind", "solar", "load", "capout", "gas", "temp"]:
        z[v + "_z"] = (z[v] - z[v].mean()) / z[v].std()
    m = smf.ols("spread ~ wind_z + solar_z + load_z + capout_z + gas_z + temp_z"
                " + C(he) + C(month) + C(year)", z).fit(
        cov_type="HAC", cov_kwds={"maxlags": 24})
    keep = [c for c in m.params.index if c.endswith("_z")]
    out["ols_std_coef"] = m.params[keep].round(2).to_dict()
    out["ols_tstat"] = m.tvalues[keep].round(1).to_dict()
    out["ols_r2"] = round(m.rsquared, 3)

    mh = df.pivot_table(index="month", columns="he", values="spread", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5),
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
    ax = axes[0]
    lim = np.nanpercentile(np.abs(mh.values), 98)
    im = ax.imshow(mh.values, aspect="auto", cmap=DIVERGE.reversed(), vmin=-lim, vmax=lim)
    ax.set_yticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                              "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(1, 25, 2)])
    ax.set_xlabel("Hour ending")
    ax.set_title("Mean Houston − North spread ($/MWh)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax = axes[1]
    hx = risk.index.to_numpy(float)
    ax.fill_between(hx, risk["p01"].to_numpy(float), risk["p99"].to_numpy(float),
                    color=C1, alpha=0.12, lw=0, label="1st–99th pct")
    ax.fill_between(hx, risk["p05"].to_numpy(float), risk["p95"].to_numpy(float),
                    color=C1, alpha=0.3, lw=0, label="5th–95th pct")
    ax.plot(hx, risk["mean"], color=C1, label="Mean")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("Hour ending")
    ax.set_ylabel("$/MWh")
    ax.set_title("Spread distribution by hour")
    ax.legend(loc="upper left")
    save(fig, "q5_when")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (v, lab) in zip(axes, [("wind", "Wind"), ("capout", "Capacity on outage"),
                                   ("load", "Load")]):
        t = pd.DataFrame(q[v]).T
        xs = np.arange(1, 6)
        ax.bar(xs - 0.2, t["mean"], 0.38, color=C1, label="Mean")
        ax.bar(xs + 0.2, t["p95"], 0.38, color=C3, label="95th percentile")
        ax.set_xticks(xs, [f"Q{i}" for i in xs])
        ax.set_xlabel(f"{lab} quintile (low to high)")
        ax.set_title(f"Spread by {lab.lower()}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Houston − North ($/MWh)")
    axes[0].legend(loc="upper left")
    save(fig, "q5_drivers")
    return out


# -------------------------------------------------------------------- Q6
# Day-ahead price model. Everything below is known when D-1 day-ahead bids
# are due, except load, which is actual here (see Q7: in production use
# ERCOT's load forecast; tested with simulated forecast error).
BASE_FEATS = ["temp", "load", "wind", "solar", "capout", "net_load", "tightness",
              "renew_share", "gas", "he", "month", "doy", "dow", "dayoff",
              "nl_daymax", "tight_daymax", "temp_daymax", "temp_daymin",
              "nl_ramp3", "solar_cap30", "wind_cap30", "solar_frac"]
PRED_CSV = RES / "predictions.csv"
REFIT = False  # set by --refit; otherwise Q6/Q7 reuse cached predictions
TEST_START = "2024-01"
HALF_LIFE_DAYS = 365


def model_features(df: pd.DataFrame) -> pd.DataFrame:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    df = df.sort_values("dt").reset_index(drop=True).copy()
    hol = USFederalHolidayCalendar().holidays("2022-01-01", "2025-12-31")
    df["dayoff"] = (df["weekend"] | df["date"].isin(hol)).astype(int)
    df["doy"] = df["date"].dt.dayofyear
    g = df.groupby("date")
    df["nl_daymax"] = g["net_load"].transform("max")
    df["tight_daymax"] = g["tightness"].transform("max")
    df["temp_daymax"] = g["temp"].transform("max")
    df["temp_daymin"] = g["temp"].transform("min")
    df["nl_ramp3"] = df["net_load"] - df["net_load"].shift(3)
    # Installed-capacity proxies: 30-day rolling max output, lagged a day.
    # Lets the model see the solar/wind build-out that shifted prices.
    cap = (df.groupby("date")[["solar", "wind"]].max().rolling(30, min_periods=5)
           .max().shift(1).add_suffix("_cap30"))
    df = df.merge(cap, left_on="date", right_index=True, how="left")
    df["solar_frac"] = df["solar"] / df["solar_cap30"]
    for h in HUBS:
        df[f"{h}_lihr"] = np.log(df[h].clip(lower=1) / df["gas"])
        df[f"{h}_lag24"] = df[f"{h}_lihr"].shift(24)     # same hour yesterday
        df[f"{h}_lag168"] = df[f"{h}_lihr"].shift(168)   # same hour last week
        day = df.groupby("date")[f"{h}_lihr"].mean()
        lag = pd.DataFrame({f"{h}_lagday": day.shift(1),
                            f"{h}_lag7d": day.rolling(7).mean().shift(1)})
        df = df.merge(lag, left_on="date", right_index=True, how="left")
    return df


def lag_feats(hub: str) -> list[str]:
    return [f"{hub}_lag24", f"{hub}_lag168", f"{hub}_lagday", f"{hub}_lag7d"]


def gbm(loss: str = "squared_error", quantile: float | None = None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = {"quantile": quantile} if loss == "quantile" else {}
    return HistGradientBoostingRegressor(
        loss=loss, max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0, **kw)


def walk_forward(df: pd.DataFrame, hub: str, feats: list[str],
                 quantiles=(None,), monthly=True, half_life=HALF_LIFE_DAYS,
                 load_noise: float = 0.0) -> pd.DataFrame:
    """Expanding-window, refit each month, predict the next month only.

    Target is log implied heat rate; price = exp(pred) x gas. Training rows
    are weighted by recency (half-life in days). load_noise > 0 perturbs the
    test-period load (and the features built from it) to mimic forecast error.
    """
    per = df["date"].dt.to_period("M")
    months = pd.period_range(TEST_START, per.max(), freq="M")
    fits = months if monthly else [m for m in months if m.month == 1]
    ok = df[f"{hub}_lag168"].notna() & df["solar_cap30"].notna()
    out = pd.DataFrame(index=df.index)
    rng = np.random.default_rng(0)
    for i, m0 in enumerate(fits):
        end = fits[i + 1] if i + 1 < len(fits) else months[-1] + 1
        tr, te = ok & (per < m0), (per >= m0) & (per < end)
        w = None
        if half_life:
            age = (m0.to_timestamp() - df.loc[tr, "date"]).dt.days
            w = 0.5 ** (age / half_life)
        Xte = df.loc[te, feats].copy()
        if load_noise:
            # Day-level forecast error (persistent within a day), shifting
            # load and the derived net-load / tightness features with it.
            days = df.loc[te, "date"]
            eps = pd.Series(rng.normal(0, load_noise, days.nunique()),
                            index=days.unique())
            d_load = df.loc[te, "load"] * days.map(eps).values
            for c in ["load", "net_load", "tightness", "nl_daymax", "tight_daymax"]:
                if c in Xte:
                    Xte[c] = Xte[c] + d_load.values
        for q in quantiles:
            mdl = (gbm() if q is None else gbm("quantile", q))
            mdl.fit(df.loc[tr, feats], df.loc[tr, f"{hub}_lihr"], sample_weight=w)
            col = "p50" if q is None else f"p{int(q * 100)}"
            out.loc[te, col] = np.exp(mdl.predict(Xte)) * df.loc[te, "gas"].values
    return out.dropna(how="all")


def conformal_bands(p: pd.DataFrame, window_days: int = 90,
                    cover: float = 0.8) -> pd.DataFrame:
    """Recalibrate the raw P10-P90 band (conformal prediction).

    At the start of each month, widen the band in log space by the `cover`
    quantile of the last `window_days` of out-of-sample misses, so only past
    errors are used. Adds {hub}_p10c / {hub}_p90c; the first month is NaN.
    """
    p = p.copy()
    per = p["date"].dt.to_period("M")
    for hub in HUBS:
        ly = np.log(p[hub].clip(lower=1))
        sc = np.maximum(np.log(p[f"{hub}_p10"]) - ly, ly - np.log(p[f"{hub}_p90"]))
        adj = pd.Series(np.nan, index=p.index)
        for m0 in per.unique()[1:]:
            start = m0.to_timestamp()
            past = (p["date"] < start) & (p["date"] >= start - pd.Timedelta(days=window_days))
            n = past.sum()
            adj[per == m0] = np.quantile(sc[past], min(1, cover * (1 + 1 / n)))
        p[f"{hub}_p10c"] = p[f"{hub}_p10"] * np.exp(-adj)
        p[f"{hub}_p90c"] = p[f"{hub}_p90"] * np.exp(adj)
    return p


def score(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    big = y > SPIKE
    return {"MAE": np.abs(e).mean(), "MedAE": np.median(np.abs(e)),
            "RMSE": np.sqrt((e ** 2).mean()), "MAE_normal": np.abs(e[~big]).mean(),
            "MAE_spike": np.abs(e[big]).mean(), "bias": e.mean()}


def q6(df: pd.DataFrame) -> dict:
    out: dict = {}
    d = model_features(df)
    if REFIT or not PRED_CSV.exists():
        # ~10 min: 24 monthly refits x 3 quantiles x 2 hubs.
        preds = d[["dt", "date", "year", "he", "dow", "dayoff", "gas", "north",
                   "houston"]].copy()
        for hub in HUBS:
            feats = BASE_FEATS + lag_feats(hub)
            wf = walk_forward(d, hub, feats, quantiles=(None, 0.1, 0.9))
            for c in wf.columns:
                preds[f"{hub}_{c}"] = wf[c].astype(float)
            preds[f"{hub}_naive"] = np.exp(d[f"{hub}_lag24"]) * d["gas"]
        preds[preds["year"] >= 2024].to_csv(PRED_CSV, index=False)
    preds = pd.read_csv(PRED_CSV, parse_dates=["dt", "date"])
    # Store the recalibrated band alongside the raw quantiles (Q7).
    preds = conformal_bands(preds.drop(columns=[c for c in preds if c.endswith(("p10c", "p90c"))]))
    preds.to_csv(PRED_CSV, index=False)

    # Permutation importance of the final North model (fit on all data up to
    # 2025, scored on 2025 so importance reflects out-of-sample use).
    from sklearn.inspection import permutation_importance
    feats = BASE_FEATS + lag_feats("north")
    ok = d["north_lag168"].notna() & d["solar_cap30"].notna()
    tr, te = ok & (d["year"] <= 2024), d["year"] == 2025
    age = (pd.Timestamp("2025-01-01") - d.loc[tr, "date"]).dt.days
    mdl = gbm().fit(d.loc[tr, feats], d.loc[tr, "north_lihr"],
                    sample_weight=0.5 ** (age / HALF_LIFE_DAYS))
    pi = permutation_importance(mdl, d.loc[te, feats], d.loc[te, "north_lihr"],
                                n_repeats=5, random_state=0, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values()
    out["perm_importance_north_2025"] = imp.round(3).to_dict()
    names = {"north_lag24": "Price yesterday, same hour", "north_lag168": "Price last week, same hour",
             "north_lagday": "Yesterday's daily avg price", "north_lag7d": "7-day avg price",
             "net_load": "Net load", "tightness": "Net load + outages", "he": "Hour of day",
             "nl_daymax": "Daily max net load", "tight_daymax": "Daily max tightness",
             "solar_frac": "Solar / recent max", "renew_share": "Renewable share",
             "solar_cap30": "Solar capacity proxy", "wind_cap30": "Wind capacity proxy",
             "temp_daymax": "Daily max temp", "temp_daymin": "Daily min temp",
             "nl_ramp3": "3-hour net-load ramp", "capout": "Outages", "dayoff": "Weekend/holiday",
             "doy": "Day of year", "dow": "Day of week", "temp": "Temperature",
             "load": "Load", "wind": "Wind", "solar": "Solar", "gas": "Gas", "month": "Month"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.55})
    ax = axes[0]
    top = imp.tail(12)
    ax.barh([names.get(i, i) for i in top.index], top.values, color=C1, height=0.6)
    ax.set_title("Top features (2025 hold-out)")
    ax.set_xlabel("Drop in R$^2$ of log heat rate when shuffled")
    ax.grid(axis="y", visible=False)
    # Example: the August 2024 heat week (largest 2024 spike).
    ax = axes[1]
    p = preds[(preds["dt"] >= "2024-08-17") & (preds["dt"] < "2024-08-24")]
    x = mpl.dates.date2num(p["dt"].to_numpy())
    ax.fill_between(x, p["north_p10"].to_numpy(float), p["north_p90"].to_numpy(float),
                    color=C1, alpha=0.18, lw=0, label="Model P10–P90")
    ax.plot(x, p["north"], color=INK, lw=1.4, label="Actual")
    ax.plot(x, p["north_p50"], color=C1, lw=1.6, label="Model median")
    ax.plot(x, p["north_naive"], color=C2, lw=1.2, label="Same hour yesterday")
    ax.set_yscale("log")
    ax.set_yticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    ax.set_ylabel("HB_NORTH $/MWh (log)")
    ax.set_title("Out-of-sample forecast, 17–23 Aug 2024")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d %b"))
    save(fig, "q6_model")
    return out


# -------------------------------------------------------------------- Q7
def q7(df: pd.DataFrame) -> dict:
    out: dict = {}
    if not PRED_CSV.exists():
        q6(df)
    p = pd.read_csv(PRED_CSV, parse_dates=["dt", "date"])
    d = model_features(df)

    # 1. Benchmark ladder, North and Houston, per test year.
    from sklearn.linear_model import Ridge
    rows = []
    for hub in HUBS:
        variants = {
            "Same hour yesterday": p[f"{hub}_naive"],
            "Final GBM (monthly refit, weighted)": p[f"{hub}_p50"]}
        # Ridge on fundamentals with hour/month dummies, refit yearly.
        X = pd.get_dummies(d[["load", "wind", "solar", "capout", "temp", "dayoff"]]
                           .assign(he=d["he"].astype(str), m=d["month"].astype(str)),
                           columns=["he", "m"]).astype(float)
        X["nl2"], X["nl3"] = (d["net_load"] / 1e4) ** 2, (d["net_load"] / 1e4) ** 3
        rp = pd.Series(np.nan, index=d.index)
        for y in (2024, 2025):
            tr, te = d["year"] < y, d["year"] == y
            r = Ridge(1.0).fit(X[tr], d.loc[tr, f"{hub}_lihr"])
            rp[te] = np.exp(r.predict(X[te])) * d.loc[te, "gas"]
        variants["Ridge, fundamentals"] = rp[d["year"] >= 2024].values
        # GBM, fundamentals only, static yearly fit (no lags, no weighting).
        fo = walk_forward(d, hub, BASE_FEATS, monthly=False, half_life=None)
        variants["GBM, fundamentals only"] = fo["p50"].values
        # Final set-up with only the price-memory features removed, to
        # isolate what the lagged prices add.
        nl = walk_forward(d, hub, BASE_FEATS)
        variants["Final GBM without price lags"] = nl["p50"].values
        for name, pred in variants.items():
            for y in (2024, 2025):
                t = (p["year"] == y).values
                rows.append({"hub": hub, "model": name, "year": y,
                             **score(p.loc[t, hub].values, np.asarray(pred)[t])})
    bench = pd.DataFrame(rows)
    out["benchmark"] = bench.round(2).to_dict("records")

    # 2. Robustness to load-forecast error (2.5% daily-persistent noise).
    noisy = walk_forward(d, "north", BASE_FEATS + lag_feats("north"), load_noise=0.025)
    t = p["year"] >= 2024
    out["mae_north_actual_load"] = round(score(p.loc[t, "north"].values,
                                               p.loc[t, "north_p50"].values)["MAE"], 2)
    out["mae_north_noisy_load"] = round(score(d.loc[noisy.index, "north"].values,
                                              noisy["p50"].values)["MAE"], 2)

    # 3. Error by hour, by regime, daily block accuracy, interval calibration.
    for hub in HUBS:
        e = (p[f"{hub}_p50"] - p[hub]).abs()
        en = (p[f"{hub}_naive"] - p[hub]).abs()
        out[f"{hub}_mae_by_he"] = e.groupby(p["he"]).mean().round(2).to_dict()
        out[f"{hub}_naive_mae_by_he"] = en.groupby(p["he"]).mean().round(2).to_dict()
        cov = ((p[hub] >= p[f"{hub}_p10"]) & (p[hub] <= p[f"{hub}_p90"]))
        out[f"{hub}_p10_p90_coverage"] = round(cov.mean(), 3)
        out[f"{hub}_coverage_by_year"] = cov.groupby(p["year"]).mean().round(3).to_dict()
        out[f"{hub}_share_above_p90"] = round((p[hub] > p[f"{hub}_p90"]).mean(), 3)
        # 5x16 on-peak block (weekday HE7-22): the product traders hedge.
        pk = (p["dayoff"] == 0) & p["he"].between(7, 22)
        blk = p[pk].groupby("date")[[hub, f"{hub}_p50", f"{hub}_naive"]].mean()
        out[f"{hub}_onpeak_daily_mae"] = {
            "model": round((blk[f"{hub}_p50"] - blk[hub]).abs().mean(), 2),
            "naive": round((blk[f"{hub}_naive"] - blk[hub]).abs().mean(), 2)}
        # Spike detection: flag when P90 > $100.
        flag, act = p[f"{hub}_p90"] > SPIKE, p[hub] > SPIKE
        out[f"{hub}_spike_flag"] = {
            "recall": round((flag & act).sum() / act.sum(), 3),
            "precision": round((flag & act).sum() / max(flag.sum(), 1), 3),
            "actual_spikes": int(act.sum()), "flags": int(flag.sum())}
    p = conformal_bands(p)
    for hub in HUBS:
        lo, hi = p[f"{hub}_p10c"], p[f"{hub}_p90c"]
        ok = lo.notna()
        cov = (p[hub] >= lo) & (p[hub] <= hi)
        out[f"{hub}_conformal_coverage"] = round(cov[ok].mean(), 3)
        out[f"{hub}_conformal_below_p10"] = round((p[hub] < lo)[ok].mean(), 3)
        out[f"{hub}_conformal_above_p90"] = round((p[hub] > hi)[ok].mean(), 3)
        out[f"{hub}_raw_below_p10"] = round((p[hub] < p[f"{hub}_p10"]).mean(), 3)
        out[f"{hub}_band_width_median"] = {
            "raw": round((p[f"{hub}_p90"] - p[f"{hub}_p10"]).median(), 2),
            "conformal": round((hi - lo)[ok].median(), 2)}

    # Houston two ways: direct model vs North forecast + yesterday's spread.
    spread = p["houston"] - p["north"]
    sp_direct = p["houston_p50"] - p["north_p50"]
    sp_lag = (p["houston"] - p["north"]).shift(24)
    out["spread_mae"] = {"direct_difference": round((sp_direct - spread).abs().mean(), 2),
                         "yesterday_same_hour": round((sp_lag - spread).abs().mean(), 2),
                         "zero": round(spread.abs().mean(), 2)}
    alt = p["north_p50"] + sp_lag
    t = alt.notna()
    out["houston_mae_direct_vs_north_plus_spread"] = {
        "direct": round((p.loc[t, "houston_p50"] - p.loc[t, "houston"]).abs().mean(), 2),
        "north_plus_lag_spread": round((alt[t] - p.loc[t, "houston"]).abs().mean(), 2)}

    # Table for the report.
    tb = bench.pivot_table(index=["hub", "model"], columns="year",
                           values=["MAE", "MedAE", "bias"]).round(2)
    order = ["Same hour yesterday", "Ridge, fundamentals", "GBM, fundamentals only",
             "Final GBM without price lags", "Final GBM (monthly refit, weighted)"]
    lines = []
    for hub in HUBS:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{HUBS[hub].replace('_', chr(92) + '_')}}}}} \\\\")
        for mname in order:
            r = tb.loc[(hub, mname)]
            bold = mname.startswith("Final GBM (")
            cells = [r[("MAE", 2024)], r[("MAE", 2025)], r[("MedAE", 2024)],
                     r[("MedAE", 2025)], r[("bias", 2024)], r[("bias", 2025)]]
            txt = " & ".join((f"\\textbf{{{c:.2f}}}" if bold else f"{c:.2f}") for c in cells)
            lines.append(f"\\quad {mname} & {txt} \\\\")
    (TAB / "q7_benchmark.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{MAE (\\$/MWh)} & \\multicolumn{2}{c}{Median abs.\\ error} "
        "& \\multicolumn{2}{c}{Bias} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Model & 2024 & 2025 & 2024 & 2025 & 2024 & 2025 \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    hrs = range(1, 25)
    ax.plot(hrs, pd.Series(out["north_naive_mae_by_he"]).values, color=C2,
            label="Same hour yesterday", marker="o", ms=3)
    ax.plot(hrs, pd.Series(out["north_mae_by_he"]).values, color=C1,
            label="Final model", marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("Hour ending")
    ax.set_ylabel("MAE, $/MWh (log)")
    ax.set_title("HB_NORTH error by hour, 2024–25")
    ax.legend(loc="upper left")
    ax = axes[1]
    # Calibration: realised share below each predicted quantile.
    ok = p["north_p10c"].notna()
    q = p[ok]
    groups = {"P10": ("north_p10", "north_p10c"), "P50": ("north_p50", "north_p50"),
              "P90": ("north_p90", "north_p90c")}
    xs = np.arange(len(groups))
    raw = [(q["north"] <= q[a]).mean() * 100 for a, _ in groups.values()]
    cal = [(q["north"] <= q[b]).mean() * 100 for _, b in groups.values()]
    ax.bar(xs - 0.27, [10, 50, 90], 0.26, color=GRID, label="Target")
    ax.bar(xs, raw, 0.26, color=C2, label="Raw quantile model")
    ax.bar(xs + 0.27, cal, 0.26, color=C1, label="After conformal step")
    for x, v in zip(xs, raw):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    for x, v in zip(xs, cal):
        ax.text(x + 0.27, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs, list(groups))
    ax.set_ylabel("% of hours at or below forecast")
    ax.set_title("Interval calibration, HB_NORTH")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "q7_validation")
    return out


# -------------------------------------------------------------------- Q8
def q8(df: pd.DataFrame) -> dict:
    out: dict = {}
    # Price concentration: share of each year's total $ from the top hours.
    rows = {}
    for y, g in df.groupby("year"):
        s = g["north"].sort_values(ascending=False)
        tot = s.sum()
        rows[int(y)] = {"mean": round(s.mean(), 2), "median": round(s.median(), 2),
                        "top1pct_share": round(s.head(int(len(s) * 0.01)).sum() / tot, 3),
                        "top5pct_share": round(s.head(int(len(s) * 0.05)).sum() / tot, 3),
                        "max": round(s.max(), 2)}
    out["concentration_north"] = rows
    out["negative_hours"] = {h: int((df[h] < 0).sum()) for h in HUBS}
    out["hours_below_5"] = int((df["north"] < 5).sum())
    out["load_by_year"] = (df.groupby("year")["load"].agg(["mean", "max"]) / 1e3).round(1).to_dict()
    out["load_growth_pct_pa"] = round(((df.groupby("year")["load"].mean().iloc[-1]
                                        / df.groupby("year")["load"].mean().iloc[0])
                                       ** (1 / 3) - 1) * 100, 2)
    out["peak_load_dt"] = str(df.loc[df["load"].idxmax(), "dt"])
    out["solar_max_by_year"] = (df.groupby("year")["solar"].max() / 1e3).round(1).to_dict()
    out["wind_max_by_year"] = (df.groupby("year")["wind"].max() / 1e3).round(1).to_dict()
    out["hub_corr_by_year"] = df.groupby("year")[["north", "houston"]].apply(
        lambda g: round(g["north"].corr(g["houston"]), 4)).to_dict()
    out["capout_daily_unique_median"] = int(df.groupby("date")["capout"].nunique().median())
    out["gas_daily_constant"] = bool((df.groupby("date")["gas"].nunique() == 1).all())
    # Weekly/holiday effect on the median price.
    out["median_price_weekday_vs_weekend"] = df.groupby("weekend")["north"].median().round(2).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    for y, col in YEAR_COLS.items():
        s = np.sort(df.loc[df["year"] == y, "north"].values)[::-1]
        ax.plot(np.arange(1, len(s) + 1) / len(s) * 100, np.clip(s, 1, None),
                color=col, label=str(y), lw=1.6)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_yticks([3, 10, 30, 100, 300, 1000, 3000],
                  ["3", "10", "30", "100", "300", "1000", "3000"])
    ax.set_xticks([0.1, 1, 10, 100], ["0.1%", "1%", "10%", "100%"])
    ax.set_xlabel("Share of hours (log)")
    ax.set_ylabel("HB_NORTH $/MWh (log)")
    ax.set_title("Price duration curve")
    ax.legend(loc="lower left", ncol=2)
    ax = axes[1]
    yrs = np.array(sorted(rows))
    ax.bar(yrs - 0.2, [rows[y]["top1pct_share"] * 100 for y in yrs], 0.38, color=C1,
           label="Top 1% of hours")
    ax.bar(yrs + 0.2, [rows[y]["top5pct_share"] * 100 for y in yrs], 0.38, color=C3,
           label="Top 5% of hours")
    ax.set_xticks(yrs, [str(y) for y in yrs])
    ax.set_ylabel("% of the year's total price sum")
    ax.set_title("How concentrated is value?")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "q8_concentration")
    return out


# ---------------------------------------------------------------------- main
QUESTIONS = {1: q1, 2: q2, 3: q3, 4: q4, 5: q5, 6: q6, 7: q7, 8: q8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, nargs="*", default=sorted(QUESTIONS))
    ap.add_argument("--refit", action="store_true",
                    help="recompute the Q6 walk-forward predictions (~10 min)")
    args = ap.parse_args()
    global REFIT
    REFIT = args.refit
    df = load_data()
    print(f"Loaded {len(df):,} hours, {df['date'].min():%Y-%m-%d} to "
          f"{df['date'].max():%Y-%m-%d}; solar rows repaired: "
          f"{int(df['solar_flag'].sum())}")
    for q in args.q:
        res = QUESTIONS[q](df)
        (RES / f"q{q}.json").write_text(json.dumps(res, indent=2, default=str))
        print(f"Q{q}: wrote results/q{q}.json")


if __name__ == "__main__":
    main()
