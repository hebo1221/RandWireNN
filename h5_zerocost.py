#!/usr/bin/env python
"""
H5 — Zero-cost wiring selection.

Pre-registered predictions:
  P1: Init-time gradient flatness (no training, one forward/backward pass)
      rank-predicts trained accuracy across a diverse pool of 24 wirings
      (Spearman > 0, p < 0.05).
  P2: The pure graph-theoretic proxy (negative average path length) does so
      too, at zero forward passes.
  Open question: does the gradient proxy beat the graph proxy?

Protocol: each candidate is a FIXED pair of stage graphs. Proxies are computed
without any training; ground truth is the mean final accuracy over 2 training
seeds with the identical recipe used in all previous studies.
"""
import matplotlib
matplotlib.use('Agg')

import json
import os
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from wiring_lab import (EPOCHS, LR, NODES, WEIGHT_DECAY, TinyRandWireNN,
                        evaluate, get_loaders, graph_metrics, make_graph)
from wiring_study import gradient_profile, path_plus_skips

H5_DIR = "./output/h5_zerocost"
TRAIN_SEEDS = [3, 7]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def candidate_pool():
    """24 diverse wirings; each returns a fixed (g1, g2) pair."""
    cands = []

    def from_params(name, params, gseed):
        g1, _ = make_graph(NODES, params, gseed)
        g2, _ = make_graph(NODES, params, gseed + 100)
        cands.append((name, (g1, g2)))

    from_params("ER_p0.2", dict(GRAPH_MODEL="ER", ER_P=0.2), 201)
    from_params("ER_p0.36", dict(GRAPH_MODEL="ER", ER_P=0.36), 202)
    from_params("ER_p0.5", dict(GRAPH_MODEL="ER", ER_P=0.5), 203)
    from_params("WS2_p0.1", dict(GRAPH_MODEL="WS", WS_K=2, WS_P=0.1), 204)
    from_params("WS2_p0.5", dict(GRAPH_MODEL="WS", WS_K=2, WS_P=0.5), 205)
    from_params("WS2_p1.0", dict(GRAPH_MODEL="WS", WS_K=2, WS_P=1.0), 206)
    from_params("WS4_p0.1", dict(GRAPH_MODEL="WS", WS_K=4, WS_P=0.1), 207)
    from_params("WS4_p0.75", dict(GRAPH_MODEL="WS", WS_K=4, WS_P=0.75), 208)
    from_params("WS6_p0.5", dict(GRAPH_MODEL="WS", WS_K=6, WS_P=0.5), 209)
    from_params("BA_m1", dict(GRAPH_MODEL="BA", BA_M=1), 210)
    from_params("BA_m2", dict(GRAPH_MODEL="BA", BA_M=2), 211)
    from_params("BA_m3", dict(GRAPH_MODEL="BA", BA_M=3), 212)
    from_params("RR_d3", dict(GRAPH_MODEL="RR", RR_D=3), 213)
    from_params("RR_d4", dict(GRAPH_MODEL="RR", RR_D=4), 214)
    from_params("NWS_p0.25", dict(GRAPH_MODEL="NWS", NWS_K=4, NWS_P=0.25), 215)
    from_params("PC_m2", dict(GRAPH_MODEL="PC", PC_M=2, PC_P=0.3), 216)

    for k in [0, 1, 2, 4, 8]:
        cands.append((f"PATH_k{k}",
                      (path_plus_skips(NODES, k, 217 + k),
                       path_plus_skips(NODES, k, 317 + k))))
    cands.append(("CYCLE", (nx.cycle_graph(NODES), nx.cycle_graph(NODES))))
    cands.append(("COMPLETE", (nx.complete_graph(NODES), nx.complete_graph(NODES))))
    cands.append(("STAR+PATH",  # hub-and-chain hybrid, deliberately odd
                  (nx.compose(nx.path_graph(NODES), nx.star_graph(NODES - 1)),
                   nx.compose(nx.path_graph(NODES), nx.star_graph(NODES - 1)))))
    return cands


def proxy_scores(graphs, device):
    """Zero-cost proxies. Higher = predicted better."""
    # Graph-only proxy: negative mean path length over both stage graphs
    L = np.mean([nx.average_shortest_path_length(g) for g in graphs])

    # Gradient proxy: negative |slope| of log grad vs depth, averaged
    # over two init seeds (still zero training)
    slopes = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        loader, _ = get_loaders("fashion", seed)
        model = TinyRandWireNN(graphs).to(device)
        slope, _ = gradient_profile(model, loader, device)
        slopes.append(abs(slope))
    return dict(neg_L=-L, grad_flatness=-float(np.mean(slopes)))


def train_acc(graphs, device):
    accs = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        train_loader, test_loader = get_loaders("fashion", seed)
        model = TinyRandWireNN(graphs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        for _ in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            scheduler.step()
        accs.append(evaluate(model, test_loader, device))
    return float(np.mean(accs)), accs


if __name__ == "__main__":
    device = torch.device("cpu")
    os.makedirs(H5_DIR, exist_ok=True)

    cands = candidate_pool()
    print(f"H5 zero-cost selection: {len(cands)} candidate wirings")
    print("=" * 72)

    records = []
    t0 = time.time()
    for i, (name, graphs) in enumerate(cands):
        proxies = proxy_scores(graphs, device)
        acc, accs = train_acc(graphs, device)
        rec = dict(name=name, acc=acc, accs=accs, **proxies,
                   n_edges=float(np.mean([g.number_of_edges() for g in graphs])))
        records.append(rec)
        print(f"  [{i+1:2d}/{len(cands)}] {name:11s} acc={acc:5.2f}% "
              f"gradflat={proxies['grad_flatness']:+.3f} "
              f"negL={proxies['neg_L']:+.2f}")

    with open(os.path.join(H5_DIR, "zerocost_results.json"), "w") as f:
        json.dump(records, f, indent=2)

    # ---- Analysis ----
    acc = np.array([r["acc"] for r in records])
    stats_out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, label in [
            (axes[0], "grad_flatness",
             "Gradient flatness at init  (-|slope|, 1 fwd/bwd pass)"),
            (axes[1], "neg_L",
             "Graph proxy  (-avg shortest path length, no NN at all)")]:
        x = np.array([r[key] for r in records])
        sp = stats.spearmanr(x, acc)
        pr = stats.pearsonr(x, acc)
        ax.scatter(x, acc, s=30, c="#2c3e50", alpha=0.75)
        for r in records:
            ax.annotate(r["name"], (r[key], r["acc"]), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
        ax.set_xlabel(label)
        ax.set_ylabel("Trained accuracy (%, mean of 2 seeds)")
        ax.set_title(f"rho={sp.statistic:.2f} (p={sp.pvalue:.1e}), r={pr[0]:.2f}")
        ax.grid(alpha=0.3)
        stats_out[key] = dict(spearman_rho=round(float(sp.statistic), 3),
                              spearman_p=float(f"{sp.pvalue:.2e}"),
                              pearson_r=round(float(pr[0]), 3))

    # Selection regret: pick top-1/top-5 by proxy, compare to oracle
    oracle = max(acc)
    for key in ["grad_flatness", "neg_L"]:
        order = np.argsort([-r[key] for r in records])
        top1 = records[order[0]]["acc"]
        top5 = max(records[i]["acc"] for i in order[:5])
        stats_out[key]["top1_regret"] = round(float(oracle - top1), 2)
        stats_out[key]["best_of_top5_regret"] = round(float(oracle - top5), 2)
        stats_out[key]["top1_pick"] = records[order[0]]["name"]

    fig.suptitle("H5: can we pick good wirings without training?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(H5_DIR, "fig_zerocost.png"), bbox_inches="tight")

    with open(os.path.join(H5_DIR, "h5_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print("=" * 72)
    print(json.dumps(stats_out, indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
