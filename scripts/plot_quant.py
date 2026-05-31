"""Gemma-3-27B-it coherence vs quantization precision (bf16 / int8 / nf4-4bit).
Same elicitation config at each precision (logprob, items_2000); the capability axis is
~fixed (quantization barely moves MMLU), so this is a degradation-at-fixed-capability view.
Reads results/coherence_gemma27_quant.csv.
"""
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
sys.path.insert(0, str(REPO / "scripts"))
sns.set_theme(style="whitegrid", context="talk")

from four_metrics import FOUR as METRICS, LAB
FLOOR = {"p_self": 0.5, "p_reversal": 0.5, "p_acyclic": 0.75, "p_crossq": 0.5}
ORDER = ["bf16", "int8", "nf4"]
COLORS = dict(zip(ORDER, sns.color_palette("viridis", 3)))

df = pd.read_csv(REPO / "results/coherence_gemma27_quant.csv").set_index("precision").loc[ORDER]

fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6))
for ax, met in zip(axes.flatten(), METRICS):
    ax.bar(ORDER, df[met], color=[COLORS[p] for p in ORDER], edgecolor="black", linewidth=0.5)
    ax.axhline(df.loc["bf16", met], color="black", ls="--", lw=1.1, alpha=0.7)  # bf16 reference
    ax.set_title(LAB[met], fontsize=13)
    lo, hi = df[met].min(), df[met].max()
    pad = max(0.02, (hi - lo) * 0.6)
    ax.set_ylim(max(0, lo - pad - 0.05), hi + pad)   # zoom so tiny differences are visible
fig.suptitle("Gemma-3-27B-it: coherence vs quantization precision\n"
             "(bf16 → int8 → nf4; dashed = bf16 level; y-axes zoomed — note differences are tiny)",
             y=1.02, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"gemma27_quant.{ext}", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote gemma27_quant.pdf / .png")
