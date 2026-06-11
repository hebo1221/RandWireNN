#!/usr/bin/env python
"""Figure + paired statistics for H7 (rehabilitation after amputation)."""
import matplotlib
matplotlib.use('Agg')

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
OUT = "./output/h7_rehab"

with open(os.path.join(OUT, "rehab_results.json")) as f:
    runs = json.load(f)

# Paired anchor: PATH_k0 from-scratch accs, same seeds, from the Short Paths Study
study = [json.loads(l) for l in open("./output/wiring_study/results.jsonl")]
scratch = {r["seed"]: r["final_acc"] for r in study if r["tag"] == "PATH_k0"}

seeds = [r["seed"] for r in runs]
rehab = np.array([r["rehab_skips_final"] for r in runs])
anchor = np.array([scratch[s] for s in seeds])
diffs = rehab - anchor
tt = stats.ttest_rel(rehab, anchor)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

# Panel A: the decomposition
ax = axes[0]
conditions = [
    ("chain trained\nfrom scratch", float(anchor.mean()), "#95a5a6"),
    ("amputated chain\n(skip-trained weights)\n+ fine-tune", float(rehab.mean()), "#2980b9"),
    ("intact\nPATH+8skips", float(np.mean([r["baseline"] for r in runs])), "#2c3e50"),
    ("chains cut\n+ fine-tune", float(np.mean([r["rehab_chains_final"] for r in runs])), "#27ae60"),
]
names = [c[0] for c in conditions]
vals = [c[1] for c in conditions]
ax.bar(names, vals, color=[c[2] for c in conditions], alpha=0.85)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=9)
ax.annotate("", xy=(1, vals[1]), xytext=(1, vals[0]),
            arrowprops=dict(arrowstyle="<->", color="#c0392b"))
ax.text(1.08, (vals[0] + vals[1]) / 2, f"taught\n+{vals[1]-vals[0]:.1f}",
        fontsize=8, color="#c0392b")
ax.annotate("", xy=(2, vals[2]), xytext=(2, vals[1]),
            arrowprops=dict(arrowstyle="<->", color="#8e44ad"))
ax.text(2.08, (vals[1] + vals[2]) / 2, f"irreplaceable\n+{vals[2]-vals[1]:.1f}",
        fontsize=8, color="#8e44ad")
ax.set_ylim(65, 84)
ax.set_ylabel("Test accuracy (%)")
ax.set_title(f"Decomposing the skip advantage\n"
             f"paired t (rehab vs scratch, 5 seeds): "
             f"t={tt.statistic:.2f}, p={tt.pvalue:.3f}, 5/5 seeds positive")
ax.tick_params(axis="x", labelsize=7.5)
ax.grid(axis="y", alpha=0.3)

# Panel B: recovery curves
ax = axes[1]
for key, color, label in [("rehab_skips_curve", "#2980b9", "skips cut (chain remnant)"),
                          ("rehab_chains_curve", "#27ae60", "chains cut (skip remnant)")]:
    curves = np.array([r[key] for r in runs])
    epochs = np.arange(1, curves.shape[1] + 1)
    ax.plot(epochs, curves.mean(0), "o-", color=color, label=label)
    ax.fill_between(epochs, curves.mean(0) - curves.std(0),
                    curves.mean(0) + curves.std(0), color=color, alpha=0.15)
ax.axhline(float(anchor.mean()), color="#95a5a6", linestyle="--",
           label=f"scratch chain ({anchor.mean():.1f})")
ax.axhline(float(np.mean([r['baseline'] for r in runs])), color="#2c3e50",
           linestyle=":", label="intact baseline")
ax.set_xlabel("Fine-tuning epoch")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Recovery curves after amputation")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("H7: skips both teach the chain (+3.4 transferable) and "
             "compute (+5.1 irreplaceable)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
path = os.path.join(OUT, "fig_h7_rehab.png")
fig.savefig(path, bbox_inches="tight")
print(f"saved {path}")
print(f"paired t={tt.statistic:.3f} p={tt.pvalue:.4f}, "
      f"diffs={diffs.tolist()}")
