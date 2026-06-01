"""Headline figure: the canonical metric (μ-decisiveness — the Case-V fitted preference
strength) as one large panel, with the four agreement-probability probes (self / order /
transitivity / framing) as small panels on the right.

x-axis = "capability index" — our ECI-style placement from published benchmarks (Epoch's fixed
per-benchmark difficulties + a per-model 1-parameter fit). Labelled plainly as "capability
index" rather than "ECI" to avoid implying it is Epoch's official published index.

Big-panel points are labelled with the parameter count (in B) or the size name (nano/mini/full
when params aren't public). Labels are de-collided with adjustText AFTER tight_layout (so the
final axes size is used — adjusting before layout lets the resize re-introduce overlaps).
"""
from pathlib import Path
import sys
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from adjustText import adjust_text

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
sns.set_theme(style="whitegrid", context="talk")

with open(REPO / "config/plots.yaml") as f:
    CFG = yaml.safe_load(f)

METRICS = CFG["metrics"]
FLOOR = {m: meta["floor"] for m, meta in METRICS.items()}
FAMILY_ORDER = CFG["headline"]["families"]
BUDGET_SERIES = CFG["headline"]["budget_series"]
BUDGET_FAMILIES = [s["family"] for s in BUDGET_SERIES]
BUDGET_SIZE = {s["family"]: s["size_label"] for s in BUDGET_SERIES}
GPTOSS20 = BUDGET_FAMILIES[0]
CAP_ORDER = FAMILY_ORDER + BUDGET_FAMILIES
# one colour per family + a SINGLE shared colour for the two GPT-OSS budget series (they are
# distinguished by their budget markers, and their size is labelled on the plot instead).
_cols = sns.color_palette("colorblind", len(FAMILY_ORDER) + 1)
palette = dict(zip(FAMILY_ORDER, _cols))
for family in BUDGET_FAMILIES:
    palette[family] = _cols[-1]
BUDGET_ORDER = {"low": 0, "medium": 1, "high": 2}
BUDGET_MARKER = {"low": "v", "medium": "s", "high": "^"}

XCOL, XLABEL = CFG["x_axis"]["col"], CFG["x_axis"]["label"]
df = pd.read_csv(REPO / "results/coherence_four_metrics.csv")


def _budget(model):
    b = str(model).rsplit("-", 1)[-1]
    return b if b in BUDGET_ORDER else None


def _size_name(model):
    m = str(model).lower()
    return next((s for s in ("nano", "mini") if s in m), "full")


def _point_label(row):
    """Label next to a big-panel point: parameter count in B if known, else the size name."""
    pb = row.get("params_b")
    return f"{pb:g}B" if pd.notna(pb) else _size_name(row["model"])


HEAD = {m: (meta["label"], meta["ylabel"], tuple(meta["ylim"]))
        for m, meta in METRICS.items() if "ylabel" in meta and "ylim" in meta}


def draw(ax, met, big=False, head_meta=None):
    """Plot one panel. Returns (texts, px, py) — the big-panel labels and the point coords to
    repel them from; de-collision is run later (post-layout) in make()."""
    texts, px, py = [], [], []
    for grp in CAP_ORDER:
        sub = df[df.family == grp].dropna(subset=[met, XCOL])
        if sub.empty:
            continue
        s = 230 if big else 110
        if "GPT-OSS" in grp:
            sub = sub.assign(_b=sub["model"].map(lambda m: BUDGET_ORDER.get(_budget(m), 0))).sort_values("_b")
            ax.plot(sub[XCOL], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
            for _, r in sub.iterrows():
                ax.scatter(r[XCOL], r[met], color=palette[grp],
                           marker=BUDGET_MARKER.get(_budget(r["model"]), "o"),
                           s=s, edgecolor="black", linewidth=0.5, zorder=3)
                px.append(r[XCOL]); py.append(r[met])
                if big and _budget(r["model"]) == "medium":      # size on the MIDDLE point only
                    size = BUDGET_SIZE[grp]
                    texts.append(ax.text(r[XCOL], r[met], size, color=palette[grp],
                                         fontsize=10, fontweight="bold", zorder=5))
        else:
            sub = sub.sort_values(XCOL)
            ax.plot(sub[XCOL], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
            ax.scatter(sub[XCOL], sub[met], color=palette[grp], marker="o",
                       s=s, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=2)
            px += list(sub[XCOL]); py += list(sub[met])
            if big:
                for _, r in sub.iterrows():
                    texts.append(ax.text(r[XCOL], r[met], _point_label(r), color=palette[grp],
                                         fontsize=10, fontweight="bold", zorder=5))
    if met in FLOOR:
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)
    ax.set_xlabel(XLABEL, fontsize=13 if big else 10)
    ax.grid(True, alpha=0.3)
    if big:
        title, ylab, ylim = head_meta
        ax.set_ylim(min(ylim[0], df[met].min() - 0.03), max(0.05 + df[met].max(), ylim[1]))
        ax.set_title(title, fontsize=17, pad=10)
        ax.set_ylabel(ylab, fontsize=13)
        ax.tick_params(labelsize=12)
    else:
        ax.set_ylim(min(FLOOR[met] - 0.05, df[met].min() - 0.03), 1.02)
        ax.set_title(METRICS[met]["label"], fontsize=11)
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10)
    return texts, px, py


def make(head_met, fname, head_word):
    fig = plt.figure(figsize=(18.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], wspace=0.28, hspace=0.42)
    ax_big = fig.add_subplot(gs[:, 0])
    big_texts, big_px, big_py = draw(ax_big, head_met, big=True, head_meta=HEAD[head_met])
    small_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                  fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, met in zip(small_axes, CFG["probes"]):
        draw(ax, met, big=False)

    # one legend handle per family + a single GPT-OSS entry (the two budget series share a colour)
    legend_groups = FAMILY_ORDER + ["GPT-OSS (20B/120B)"]
    legend_color = {**{g: palette[g] for g in FAMILY_ORDER}, "GPT-OSS (20B/120B)": palette[GPTOSS20]}
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=legend_color[g],
                      markeredgecolor="black", markersize=11, label=g) for g in legend_groups]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=1.4, label="chance floor (probes)"))
    for b in ["low", "medium", "high"]:
        handles.append(Line2D([0], [0], marker=BUDGET_MARKER[b], color="0.35",
                              markerfacecolor="0.7", markersize=11, linestyle="none",
                              label=f"budget: {b}"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, frameon=False, fontsize=12)
    fig.suptitle(f"Preference coherence vs capability — {head_word}", y=1.0, fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    # De-collide AFTER layout so adjustText uses the final axes size (and after a draw so text
    # extents are real). Strong text-text + point repulsion; thin leader lines to the points.
    if big_texts:
        fig.canvas.draw()
        adjust_text(big_texts, x=big_px, y=big_py, ax=ax_big,
                    force_text=(0.4, 0.6), expand=(1.2, 1.5),
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png")


make(CFG["headline_metric"], CFG["headline"]["fname"],
     METRICS[CFG["headline_metric"]]["headline_word"])           # canonical headline
