#!/usr/bin/env python
"""
Analysis for the Short Paths Study: figures + statistics.

Outputs (to output/wiring_study/):
  fig_dose_response.png   H1/H2: controlled interventions on path length
  fig_gradient.png        H3: gradient-flow mechanism and mediation
  fig_edge_weights.png    Exploratory: learned mixing weights vs. centrality
  stats_summary.json      All test statistics used in the report
"""
import matplotlib
matplotlib.use('Agg')

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from wiring_study import STUDY_DIR

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

FAMILY_COLORS = {"ws2": "#c0392b", "path_k": "#2980b9", "ref": "#27ae60"}
FAMILY_LABELS = {"ws2": "WS k=2 rewiring sweep (12 edges fixed)",
                 "path_k": "PATH + k skip edges",
                 "ref": "WS k=4, p=0.75 (reference)"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def partial_corr(y, x, covariates):
    """Pearson correlation of y and x after regressing out covariates."""
    Z = np.column_stack([np.ones(len(y))] + covariates)
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def standardize(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std()


def mediation_bootstrap(x, m, y, n_boot=5000, seed=0):
    """Indirect effect a*b of x -> m -> y with percentile bootstrap CI.

    All variables standardized. Returns total effect c, direct effect c',
    indirect effect a*b and its 95% CI.
    """
    x, m, y = standardize(x), standardize(m), standardize(y)
    n = len(x)

    def paths(xi, mi, yi):
        a = stats.linregress(xi, mi).slope
        X = np.column_stack([np.ones(len(xi)), xi, mi])
        coef = np.linalg.lstsq(X, yi, rcond=None)[0]
        c_prime, b = coef[1], coef[2]
        c = stats.linregress(xi, yi).slope
        return c, c_prime, a * b

    c, c_prime, ab = paths(x, m, y)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        boots.append(paths(x[idx], m[idx], y[idx])[2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(total_c=round(c, 3), direct_c_prime=round(c_prime, 3),
                indirect_ab=round(ab, 3), ab_ci95=[round(lo, 3), round(hi, 3)],
                mediated_fraction=round(ab / c, 3) if abs(c) > 1e-9 else None)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def config_means(runs, key="final_acc"):
    by_tag = defaultdict(list)
    for r in runs:
        by_tag[r["tag"]].append(r)
    out = {}
    for tag, rs in by_tag.items():
        vals = [r[key] for r in rs]
        out[tag] = dict(mean=np.mean(vals), sem=stats.sem(vals),
                        family=rs[0]["family"],
                        L=np.mean([r["g_avg_path_length"] for r in rs]),
                        k=rs[0].get("k_skips"), p=rs[0].get("ws_p"))
    return out


def fig_dose_response(runs, out_path, stats_out):
    cm = config_means(runs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: marginal value of skip connections (H2)
    ax = axes[0]
    ks, means, sems = [], [], []
    for tag, c in sorted(cm.items()):
        if c["family"] == "path_k":
            ks.append(c["k"]); means.append(c["mean"]); sems.append(c["sem"])
    order = np.argsort(ks)
    ks = np.array(ks)[order]; means = np.array(means)[order]
    sems = np.array(sems)[order]
    ax.errorbar(ks, means, yerr=sems, fmt="o-", color="#2980b9",
                capsize=4, linewidth=2)
    ref = cm["WS4_ref"]
    ax.axhline(ref["mean"], color="#27ae60", linestyle="--",
               label=f"WS k=4 p=0.75 reference ({ref['mean']:.1f}%)")
    for k, m, d in zip(ks[1:], means[1:], np.diff(means)):
        ax.annotate(f"+{d:.1f}", (k, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#2c3e50")
    ax.set_xlabel("k skip edges added to a plain 12-node chain")
    ax.set_ylabel("Test accuracy (%)  (mean ± s.e.m., 5 seeds)")
    ax.set_title("H2: marginal value of shortcuts is large then saturates")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel B: dose-response, all runs (H1)
    ax = axes[1]
    for fam in FAMILY_COLORS:
        rs = [r for r in runs if r["family"] == fam]
        ax.scatter([r["g_avg_path_length"] for r in rs],
                   [r["final_acc"] for r in rs],
                   c=FAMILY_COLORS[fam], s=22, alpha=0.55,
                   label=FAMILY_LABELS[fam])
    xs = np.array([r["g_avg_path_length"] for r in runs])
    ys = np.array([r["final_acc"] for r in runs])
    fit = stats.linregress(xs, ys)
    grid = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(grid, fit.intercept + fit.slope * grid, "k--", linewidth=1.5)
    pear = stats.pearsonr(xs, ys)
    spear = stats.spearmanr(xs, ys)
    ax.set_xlabel("Average shortest path length L")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"H1: dose-response  r={pear[0]:.2f} (p={pear[1]:.1e}), "
                 f"rho={spear[0]:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # WS2-only correlation: the edge-count-controlled test
    ws2 = [r for r in runs if r["family"] == "ws2"]
    xs2 = [r["g_avg_path_length"] for r in ws2]
    ys2 = [r["final_acc"] for r in ws2]
    p2 = stats.pearsonr(xs2, ys2)
    s2 = stats.spearmanr(xs2, ys2)

    # Partial correlation controlling edge count and input-node count
    cov = [np.array([r["g_n_edges"] for r in runs], dtype=float),
           np.array([r["g_n_input_nodes"] for r in runs], dtype=float)]
    part = partial_corr(ys, xs, cov)

    stats_out["H1"] = dict(
        all_runs_pearson=dict(r=round(pear[0], 3), p=float(f"{pear[1]:.2e}")),
        all_runs_spearman=dict(rho=round(spear[0], 3), p=float(f"{spear[1]:.2e}")),
        ws2_only_pearson=dict(r=round(p2[0], 3), p=float(f"{p2[1]:.2e}"),
                              note="edge count fixed at 12 - confound-free"),
        ws2_only_spearman=dict(rho=round(s2[0], 3), p=float(f"{s2[1]:.2e}")),
        partial_r_given_edges_inputnodes=dict(r=round(part[0], 3),
                                              p=float(f"{part[1]:.2e}")),
    )
    deltas = {f"k={a}->k={b}": round(float(m2 - m1), 2)
              for (a, m1), (b, m2) in zip(zip(ks, means), zip(ks[1:], means[1:]))}
    stats_out["H2"] = dict(acc_by_k={int(k): round(float(m), 2)
                                     for k, m in zip(ks, means)},
                           marginal_gains=deltas)


def fig_gradient(runs, out_path, stats_out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: gradient profiles at init (stage 2), config means
    ax = axes[0]
    show = ["PATH_k0", "PATH_k8", "WS2_p0.0", "WS2_p1.0", "WS4_ref"]
    palette = ["#2980b9", "#7fb3d5", "#c0392b", "#e6a09b", "#27ae60"]
    for tag, color in zip(show, palette):
        rs = [r for r in runs if r["tag"] == tag]
        prof = defaultdict(list)
        for r in rs:
            for d, g in r["grad_profile_stage2"]:
                prof[d].append(g)
        ds = sorted(prof)
        ax.plot(ds, [np.log10(np.mean(prof[d]) + 1e-12) for d in ds],
                "o-", color=color, label=tag, markersize=4)
    ax.set_xlabel("Node depth in DAG (stage 2)")
    ax.set_ylabel("log10 RMS gradient at initialization")
    ax.set_title("H3a: gradient profile across depth")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: gradient imbalance vs accuracy
    ax = axes[1]
    imb = np.array([abs(r["grad_slope"]) for r in runs])
    acc = np.array([r["final_acc"] for r in runs])
    L = np.array([r["g_avg_path_length"] for r in runs])
    for fam in FAMILY_COLORS:
        sel = [i for i, r in enumerate(runs) if r["family"] == fam]
        ax.scatter(imb[sel], acc[sel], c=FAMILY_COLORS[fam], s=22, alpha=0.6,
                   label=FAMILY_LABELS[fam])
    pr = stats.pearsonr(imb, acc)
    ax.set_xlabel("Gradient imbalance |slope of log10 grad vs depth|")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"H3b: imbalance vs accuracy  r={pr[0]:.2f} (p={pr[1]:.1e})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    med = mediation_bootstrap(L, imb, acc)
    a_path = stats.pearsonr(L, imb)
    stats_out["H3"] = dict(
        imbalance_vs_acc=dict(r=round(pr[0], 3), p=float(f"{pr[1]:.2e}")),
        L_vs_imbalance=dict(r=round(a_path[0], 3), p=float(f"{a_path[1]:.2e}")),
        mediation_L_to_imb_to_acc=med,
    )


def fig_edge_weights(edges, out_path, stats_out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    w = np.array([e["weight"] for e in edges])
    btw = np.array([e["betweenness"] for e in edges])
    skip = np.array([e["skip_dist"] for e in edges])
    ninp = np.array([e["n_inputs"] for e in edges], dtype=float)

    # Panel A: weight vs betweenness
    ax = axes[0]
    ax.scatter(btw, w, s=10, alpha=0.35, c="#8e44ad")
    pr_b = stats.pearsonr(btw, w)
    sp_b = stats.spearmanr(btw, w)
    fit = stats.linregress(btw, w)
    grid = np.linspace(btw.min(), btw.max(), 50)
    ax.plot(grid, fit.intercept + fit.slope * grid, "k--")
    ax.axhline(1 / (1 + np.e ** -1), color="gray", linestyle=":",
               label="initialization (sigmoid(1)=0.73)")
    ax.set_xlabel("Edge betweenness centrality (DAG)")
    ax.set_ylabel("Learned mixing weight sigmoid(w)")
    ax.set_title(f"Weight vs centrality  r={pr_b[0]:.2f} (p={pr_b[1]:.1e})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: weight vs skip distance (binned)
    ax = axes[1]
    bins = defaultdict(list)
    for s, wi in zip(skip, w):
        bins[int(s)].append(wi)
    ds = sorted(bins)
    ax.errorbar(ds, [np.mean(bins[d]) for d in ds],
                yerr=[stats.sem(bins[d]) for d in ds],
                fmt="o-", color="#d35400", capsize=3)
    ax.axhline(1 / (1 + np.e ** -1), color="gray", linestyle=":")
    pr_s = stats.spearmanr(skip, w)
    ax.set_xlabel("Skip distance (dst - src node id)")
    ax.set_ylabel("Learned mixing weight")
    ax.set_title(f"Weight vs skip length  rho={pr_s[0]:.2f} (p={pr_s[1]:.1e})")
    ax.grid(alpha=0.3)

    # Panel C: chain edges vs injected skips within PATH_k networks
    ax = axes[2]
    path_edges = [e for e in edges if e["family"] == "path_k"]
    chain = [e["weight"] for e in path_edges if e["skip_dist"] == 1]
    skips = [e["weight"] for e in path_edges if e["skip_dist"] > 1]
    ax.boxplot([chain, skips], tick_labels=[f"chain edges\n(n={len(chain)})",
                                            f"injected skips\n(n={len(skips)})"])
    tt = stats.ttest_ind(chain, skips)
    mw = stats.mannwhitneyu(chain, skips)
    ax.set_ylabel("Learned mixing weight")
    ax.set_title(f"PATH+k nets: skip vs chain edges\n"
                 f"t={tt.statistic:.2f} (p={tt.pvalue:.1e})")
    ax.grid(alpha=0.3)

    fig.suptitle("Exploratory: what does the network learn about its own wiring?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    part = partial_corr(w, btw, [ninp])
    stats_out["edge_weights"] = dict(
        n_edges=len(edges),
        weight_vs_betweenness=dict(r=round(pr_b[0], 3), p=float(f"{pr_b[1]:.2e}"),
                                   rho=round(sp_b[0], 3)),
        weight_vs_betweenness_partial_ninputs=dict(r=round(part[0], 3),
                                                   p=float(f"{part[1]:.2e}")),
        weight_vs_skipdist_spearman=dict(rho=round(pr_s[0], 3),
                                         p=float(f"{pr_s[1]:.2e}")),
        path_k_chain_vs_skip=dict(
            chain_mean=round(float(np.mean(chain)), 4),
            skip_mean=round(float(np.mean(skips)), 4),
            t=round(float(tt.statistic), 2), p=float(f"{tt.pvalue:.2e}"),
            mannwhitney_p=float(f"{mw.pvalue:.2e}")),
    )


if __name__ == "__main__":
    runs = load_jsonl(os.path.join(STUDY_DIR, "results.jsonl"))
    edges = load_jsonl(os.path.join(STUDY_DIR, "edge_weights.jsonl"))
    print(f"Loaded {len(runs)} runs, {len(edges)} edge records")

    stats_out = {}
    fig_dose_response(runs, os.path.join(STUDY_DIR, "fig_dose_response.png"), stats_out)
    fig_gradient(runs, os.path.join(STUDY_DIR, "fig_gradient.png"), stats_out)
    fig_edge_weights(edges, os.path.join(STUDY_DIR, "fig_edge_weights.png"), stats_out)

    stats_path = os.path.join(STUDY_DIR, "stats_summary.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"saved {stats_path}")
    print(json.dumps(stats_out, indent=2))
