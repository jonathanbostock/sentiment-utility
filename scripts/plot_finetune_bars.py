"""Model-organism bar charts — equivalently shaped to the μ-decisiveness headline: a wide
μ-decisiveness panel + the four agreement-probability probes, as bars over each suite's
baseline series vs its fine-tunes.

A suite is defined entirely in `config/run/plots.yaml` (see the `suites:` block + the README's
"Define your own suite"). Each suite has:
  - `baselines`: the grey reference series (sorted by size, coloured light→dark small→big).
    Each entry is EITHER `{model: <name>}` (read from results/coherence_four_metrics.csv) OR
    `{edges: <path>, params_b: <size>}` (computed from a raw edges.jsonl). Add `base: true` to
    the one that is the fine-tune's starting point — it is hatched and drawn as a dashed
    reference line across every panel.
  - `variants`: the fine-tunes, each `{edges: <path>, code: <1-2 letters>, name: <legend name>}`.
Because baselines can be raw edges, a suite needs NO benchmark CSV — point it straight at your
own tarballs' edges. Metrics are memoised (four_metrics.metrics_cached) so re-plots are fast.
"""
from pathlib import Path
import math
import sys
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/plots"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import metrics_cached
from matplotlib.gridspec import GridSpec
sns.set_theme(style="white", context="talk")   # no gridlines on the bar charts (scatter scripts keep their own grid)

with open(REPO / "config/run/plots.yaml") as f:
    CFG = yaml.safe_load(f)

LEAD = CFG["headline_metric"]
PROBES = CFG["probes"]
METRICS = [LEAD] + PROBES
LAB = {m: CFG["metrics"][m]["label"] for m in METRICS}
FLOOR = {m: CFG["metrics"][m]["floor"] for m in METRICS}
BASE_BLACK = "0.0"

_SCALE4 = None


def _scale_csv():
    """Lazily load results/coherence_four_metrics.csv — only needed for `model:` baselines.
    Suites whose baselines are all `edges:` never touch it."""
    global _SCALE4
    if _SCALE4 is None:
        p = REPO / "results/coherence_four_metrics.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — it is needed for `model:` baselines. Either run "
                "scripts/build_four_metrics.py to (re)generate it, or specify the baseline as "
                "`{edges: ..., params_b: ...}` instead of `{model: ...}`.")
        _SCALE4 = pd.read_csv(p).set_index("model")
    return _SCALE4


def scale_row(model, role, params_b=None):
    """A baseline/base row read from the benchmark CSV (params_b defaults to the CSV value)."""
    df = _scale_csv()
    if model not in df.index:
        raise KeyError(f"'{model}' not in coherence_four_metrics.csv. Add it via "
                       "build_four_metrics.py, or specify this baseline as `edges:` instead.")
    r = df.loc[model]
    pb = float(r["params_b"]) if params_b is None else float(params_b)
    return {"role": role, "params_b": pb, "code": None, "name": None,
            **{m: float(r[m]) for m in METRICS}}


def edge_row(path, role, params_b=None, code=None, name=None):
    """A baseline/base/fine-tune row computed straight from a raw edges.jsonl (memoised)."""
    m = metrics_cached(path)
    return {"role": role, "params_b": params_b, "code": code, "name": name,
            **{k: float(m[k]) for k in METRICS}}


def suite_rows(suite):
    """Build the bar rows for one suite. Baselines (CSV or edges, optionally `base: true`) come
    first, then the fine-tune variants."""
    rows = []
    for b in suite["baselines"]:
        role = "base" if b.get("base") else "baseline"
        if "model" in b:
            rows.append(scale_row(b["model"], role, params_b=b.get("params_b")))
        else:
            rows.append(edge_row(REPO / b["edges"], role, params_b=b["params_b"], name=b.get("name")))
    for v in suite["variants"]:
        rows.append(edge_row(REPO / v["edges"], "fine-tune", code=v["code"], name=v["name"]))
    return pd.DataFrame(rows)


def plot_bars(df, title, fname, series):
    base_part = df[df.role.isin(["baseline", "base"])].sort_values("params_b")
    ft_part = df[df.role == "fine-tune"].sort_values("code")
    df = pd.concat([base_part, ft_part]).reset_index(drop=True)

    n_base = len(base_part)
    greys = np.linspace(0.78, 0.30, n_base) if n_base > 1 else np.array([0.5])   # light(small)→dark(big)
    n_ft = int((df.role == "fine-tune").sum())
    ft_palette = (sns.color_palette("colorblind", n_ft) if n_ft <= 4
                  else sns.color_palette("husl", n_ft))
    colors, hatches, xlabels = [], [], []
    gi = ki = 0
    for _, r in df.iterrows():
        if r.role in ("baseline", "base"):
            colors.append(str(round(float(greys[gi]), 3))); gi += 1
            hatches.append("////" if r.role == "base" else "")
            xlabels.append(f"{r.params_b:g}B")
        else:
            colors.append(ft_palette[ki]); ki += 1
            hatches.append("")
            xlabels.append(r.code)

    def draw_bar(ax, met, big=False):
        bars = ax.bar(range(len(df)), df[met], color=colors, edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
        base = df[df.role == "base"]
        if len(base):
            ax.axhline(float(base[met].iloc[0]), color=BASE_BLACK, ls="--", lw=1.1, alpha=0.8)
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)
        ax.set_title(LAB[met], fontsize=15 if big else 12, fontweight="bold" if big else "normal")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(xlabels, rotation=0, fontsize=12 if big else 10, color="black")
        # Headline panel: sit the zero baseline on the bottom axis (no floating bars).
        # Probe panels keep a little padding below their chance floor.
        lo = 0.0 if big else min(FLOOR[met], df[met].min()) - 0.05
        ax.set_ylim(lo, 1.02)
        ax.margins(x=0.02)

    fig = plt.figure(figsize=(18.5, 9.0))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], hspace=0.30, wspace=0.28)
    draw_bar(fig.add_subplot(gs[:, 0]), LEAD, big=True)
    probe_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                  fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, met in zip(probe_axes, PROBES):
        draw_bar(ax, met, big=False)

    # Legend: grey = baseline series; hatched = the fine-tune base; one colour per fine-tune
    # decoded "code = name"; plus the base-level and chance-floor reference lines.
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", ec="black"),
               plt.Rectangle((0, 0), 1, 1, facecolor="0.32", ec="black", hatch="////")]
    labels = [f"{series} baselines (size →)", "base for fine-tuning"]
    ft = df[df.role == "fine-tune"]
    for i, (_, r) in enumerate(ft.iterrows()):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=ft_palette[i], ec="black"))
        labels.append(f"{r.code} = {r['name']}")
    handles += [plt.Line2D([0], [0], color=BASE_BLACK, ls="--", lw=1.1),
                plt.Line2D([0], [0], color="0.4", ls=":", lw=1.2)]
    labels += ["base model performance", "chance floor"]

    ncol = min(7, len(labels))
    nrows = math.ceil(len(labels) / ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=10, handlelength=1.4, columnspacing=1.4)
    fig.suptitle(title, y=1.0, fontsize=14)
    fig.tight_layout(rect=[0, 0.04 + 0.05 * nrows, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png  ({len(df)} bars, {n_ft} fine-tunes, legend {nrows} rows)")


if __name__ == "__main__":
    for suite in CFG["suites"]:
        plot_bars(suite_rows(suite), suite["title"], suite["fname"], suite["series"])
