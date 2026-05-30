"""Scaling/emergence plots (seaborn PDF) from results/coherence_scale_all.csv.
A: coherence vs params for instruct families (Qwen/Llama).
B: OLMo-2-7B coherence vs pretraining tokens, with post-training (Instruct) reference.
"""
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
REPO=Path("/home/jonathandbostock/Documents/sentiment-utility")
sns.set_theme(style="whitegrid", context="talk")
df=pd.read_csv(REPO/"results/coherence_scale_all.csv")
METS=["decisiveness","q_agreement","mu_valence_corr"]
LAB={"decisiveness":"decisiveness","q_agreement":"q_agreement (corr)","mu_valence_corr":"μ–valence corr"}

# A: params scaling (Qwen + Llama, x_kind=params_B)
ps=df[df.x_kind=="params_B"].copy().sort_values(["series","x"])
m=ps.melt(id_vars=["series","x","label"],value_vars=METS,var_name="metric",value_name="v")
m["metric"]=m["metric"].map(LAB)
g=sns.relplot(m,x="x",y="v",hue="series",col="metric",kind="line",marker="o",
              height=4.2,aspect=1.0,facet_kws={"sharey":True})
g.set(xscale="log"); g.set_axis_labels("params (B, log)","coherence")
for ax in g.axes.flat: ax.axhline(0,color="grey",lw=0.8,ls=":")
g.figure.suptitle("Sentiment coherence vs model size (instruct families)",y=1.04)
g.savefig(REPO/"results/plots/scale_params.pdf"); plt.close()
print("wrote scale_params.pdf")

# B: OLMo pretraining-token trajectory
ol=df[(df.series=="olmo")&(df.x_kind=="tokens_B")].copy().sort_values("x")
post=df[(df.series=="olmo")&(df.label=="Instruct")]
plt.figure(figsize=(9,6))
for met in METS:
    plt.plot(ol.x, ol[met], marker="o", label=LAB[met], lw=2.4)
    if len(post):
        plt.scatter([ol.x.max()*1.5],[post[met].iloc[0]],marker="*",s=320,zorder=5,
                    edgecolor="black",linewidth=0.6)
plt.xscale("log"); plt.axhline(0,color="grey",lw=0.8,ls=":")
plt.xlabel("pretraining tokens (B, log)  →  ★ = +Instruct (post-trained)")
plt.ylabel("coherence"); plt.legend(fontsize=11,loc="center left")
plt.title("OLMo-2-7B: coherence vs pretraining compute\n(flat through pretraining; jumps only after instruction tuning ★)")
plt.tight_layout(); plt.savefig(REPO/"results/plots/scale_olmo_pretrain.pdf"); plt.close()
print("wrote scale_olmo_pretrain.pdf")
