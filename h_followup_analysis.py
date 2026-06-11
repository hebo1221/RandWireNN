#!/usr/bin/env python
"""
Figures and statistics for H4/H4b (scaffolding falsification) and H6 (depth
scaling). H5 produces its own figure in h5_zerocost.py.
"""
import matplotlib
matplotlib.use('Agg')

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
OUT = "./output/followup"
os.makedirs(OUT, exist_ok=True)


def fig_h4(stats_out):
    with open("./output/h4_scaffolding/ablation_results.json") as f:
        runs = json.load(f)
    with open("./output/h4_scaffolding/recalibration_results.json") as f:
        recal = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Panel A: per-edge damage by family and edge type
    ax = axes[0]
    groups, labels = [], []
    for family in ["PATH_k8", "WS4"]:
        for kind, pred in [("skip", lambda e: e["skip_dist"] > 1),
                           ("chain", lambda e: e["skip_dist"] == 1)]:
            vals = [e["delta_acc"] for r in runs if r["family"] == family
                    for e in r["edges"] if pred(e)]
            groups.append(vals)
            labels.append(f"{family}\n{kind} (n={len(vals)})")
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel("Accuracy drop when edge removed (pt)")
    ax.set_title("Per-edge ablation damage\n(sparse chains brittle, dense WS robust)")
    ax.grid(alpha=0.3)

    # Panel B: does the learned weight predict ablation damage?
    ax = axes[1]
    colors = {"PATH_k8": "#2980b9", "WS4": "#27ae60"}
    all_w, all_d = [], []
    for family in ["PATH_k8", "WS4"]:
        w = [e["weight"] for r in runs if r["family"] == family
             for e in r["edges"]]
        d = [e["delta_acc"] for r in runs if r["family"] == family
             for e in r["edges"]]
        ax.scatter(w, d, s=14, alpha=0.5, c=colors[family], label=family)
        all_w += w
        all_d += d
    pr = stats.pearsonr(all_w, all_d)
    sp = stats.spearmanr(all_w, all_d)
    ax.set_xlabel("Learned mixing weight sigmoid(w)")
    ax.set_ylabel("Ablation damage (pt)")
    ax.set_title(f"Weight vs functional importance\n"
                 f"r={pr[0]:.2f} (p={pr[1]:.1e}), rho={sp.statistic:.2f}")
    ax.legend(); ax.grid(alpha=0.3)
    stats_out["H4_weight_vs_damage"] = dict(
        r=round(pr[0], 3), p=float(f"{pr[1]:.2e}"),
        rho=round(float(sp.statistic), 3),
        rho_p=float(f"{sp.pvalue:.2e}"))

    btw_d = stats.spearmanr([e["betweenness"] for r in runs for e in r["edges"]],
                            [e["delta_acc"] for r in runs for e in r["edges"]])
    stats_out["H4_betweenness_vs_damage"] = dict(
        rho=round(float(btw_d.statistic), 3), p=float(f"{btw_d.pvalue:.2e}"))

    # Panel C: cumulative ablation with BN recalibration
    ax = axes[2]
    keys = ["baseline", "intact_recalib", "no_chains_recalib",
            "no_chains_naive", "no_skips_recalib", "no_skips_naive"]
    names = ["intact", "intact\n+recalib", "chains cut\n+recalib",
             "chains cut\nnaive", "skips cut\n+recalib", "skips cut\nnaive"]
    means = [np.mean([r[k] for r in recal]) for k in keys]
    sems = [stats.sem([r[k] for r in recal]) for k in keys]
    bars = ax.bar(names, means, yerr=sems, capsize=4,
                  color=["#2c3e50", "#2c3e50", "#e67e22", "#e67e22",
                         "#c0392b", "#c0392b"], alpha=0.8)
    ax.axhline(70.92, color="gray", linestyle="--",
               label="trained without skips (70.9)")
    ax.axhline(10, color="lightgray", linestyle=":", label="chance (10)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Cumulative ablation, PATH+8skips\n"
                 "skips-cut collapse survives BN recalibration")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    stats_out["H4b_means"] = {k: round(float(m), 2)
                              for k, m in zip(keys, means)}

    fig.suptitle("H4: the scaffolding hypothesis is FALSIFIED — "
                 "networks compute through their skips", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, "fig_h4_ablation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def fig_h6(stats_out):
    with open("./output/h6_scaling/scaling_results.json") as f:
        runs = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ns = sorted({r["n_nodes"] for r in runs})
    colors = {"WS": "#27ae60", "PATH": "#2980b9"}

    # Panel A: accuracy vs depth budget
    ax = axes[0]
    gaps = {}
    for family in ["WS", "PATH"]:
        means, sems = [], []
        for n in ns:
            vals = [r["final_acc"] for r in runs
                    if r["family"] == family and r["n_nodes"] == n]
            means.append(np.mean(vals))
            sems.append(stats.sem(vals))
        ax.errorbar(ns, means, yerr=sems, fmt="o-", capsize=4,
                    color=colors[family], linewidth=2,
                    label=f"{family} ({'random wiring' if family == 'WS' else 'plain chain'})")
        gaps[family] = means
    for n, w, p in zip(ns, gaps["WS"], gaps["PATH"]):
        ax.annotate(f"gap {w - p:+.1f}", (n, (w + p) / 2), fontsize=8,
                    ha="center", color="#7f8c8d")
    ax.set_xlabel("Nodes per stage N (stage depth = N-1 for PATH)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_title("Accuracy vs depth budget")
    ax.legend(); ax.grid(alpha=0.3)
    stats_out["H6_gap_by_N"] = {int(n): round(float(w - p), 2)
                                for n, w, p in zip(ns, gaps["WS"], gaps["PATH"])}

    # Panel B: total gradient attenuation vs N
    ax = axes[1]
    for family in ["WS", "PATH"]:
        means, sems = [], []
        for n in ns:
            vals = [r["total_attenuation"] for r in runs
                    if r["family"] == family and r["n_nodes"] == n]
            means.append(np.mean(vals))
            sems.append(stats.sem(vals))
        ax.errorbar(ns, means, yerr=sems, fmt="s-", capsize=4,
                    color=colors[family], linewidth=2, label=family)
    pa_slopes = [abs(r["grad_slope"]) for r in runs if r["family"] == "PATH"]
    ax.set_xlabel("Nodes per stage N")
    ax.set_ylabel("Total init gradient attenuation\n|slope| × DAG depth (decades)")
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_title(f"Mechanism: PATH attenuates {np.mean(pa_slopes):.3f} "
                 "decades/level (constant)\n→ exponential decay in depth")
    ax.legend(); ax.grid(alpha=0.3)
    stats_out["H6_path_decay_per_level_decades"] = round(float(np.mean(pa_slopes)), 4)

    fig.suptitle("H6: the path penalty grows superlinearly with depth",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, "fig_h6_scaling.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    stats_out = {}
    fig_h4(stats_out)
    fig_h6(stats_out)
    with open(os.path.join(OUT, "followup_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(json.dumps(stats_out, indent=2))
