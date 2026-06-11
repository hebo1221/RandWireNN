#!/usr/bin/env python
"""
H8 — Does the zero-cost gradient proxy survive a 58-candidate pool?

H5 validated the init-gradient-flatness proxy at n=24 (rho=0.44, top-1
regret 0). Pre-registered questions at scale:

  P1: The proxy's rank correlation holds on a much more diverse pool
      (58 wirings incl. trees, wheels, ladders, barbells, deep chains).
  P2: Proxy failures are systematic, not random: residuals concentrate on
      hub-dominated graphs (high max degree), where one node's huge fan-in
      makes the init gradient profile look pathological without harming
      trainability (the BA_m1 false negative from H5).
  P3: Selection metrics that matter in practice: top-1 regret, regret@k.

Budget: 1 training seed per candidate (rank correlation at n=58 tolerates
seed noise), 2 init seeds for the proxy. ~25 min on CPU.
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
                        evaluate, get_loaders, make_graph)
from wiring_study import gradient_profile, path_plus_skips

H8_DIR = "./output/h8_proxy_scale"
TRAIN_SEED = 3
PROXY_SEEDS = [3, 7]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def connectify(g):
    if not nx.is_connected(g):
        comps = list(nx.connected_components(g))
        for a, b in zip(comps, comps[1:]):
            g.add_edge(list(a)[0], list(b)[0])
    return g


def candidate_pool():
    cands = []
    gs = 400  # rolling graph seed

    def P(name, params):
        nonlocal gs
        gs += 1
        g1, _ = make_graph(NODES, params, gs)
        g2, _ = make_graph(NODES, params, gs + 100)
        cands.append((name, (g1, g2)))

    for p in [0.15, 0.2, 0.25, 0.3, 0.36, 0.45, 0.55, 0.65]:
        P(f"ER_p{p}", dict(GRAPH_MODEL="ER", ER_P=p))
    for k in [2, 4]:
        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            P(f"WS{k}_p{p}", dict(GRAPH_MODEL="WS", WS_K=k, WS_P=p))
    for p in [0.25, 0.5, 0.75]:
        P(f"WS6_p{p}", dict(GRAPH_MODEL="WS", WS_K=6, WS_P=p))
    for m in [1, 2, 3, 4]:
        P(f"BA_m{m}", dict(GRAPH_MODEL="BA", BA_M=m))
    for d in [3, 4, 5, 6]:
        P(f"RR_d{d}", dict(GRAPH_MODEL="RR", RR_D=d))
    for p in [0.1, 0.3, 0.5]:
        P(f"NWS_p{p}", dict(GRAPH_MODEL="NWS", NWS_K=4, NWS_P=p))
    for m in [2, 3]:
        for p in [0.1, 0.5]:
            P(f"PC_m{m}_p{p}", dict(GRAPH_MODEL="PC", PC_M=m, PC_P=p))
    for k in [0, 1, 2, 3, 4, 6, 8, 10, 12, 16]:
        cands.append((f"PATH_k{k}", (path_plus_skips(NODES, k, 500 + k),
                                     path_plus_skips(NODES, k, 600 + k))))
    cands.append(("CYCLE", (nx.cycle_graph(NODES), nx.cycle_graph(NODES))))
    cands.append(("COMPLETE", (nx.complete_graph(NODES), nx.complete_graph(NODES))))
    for i, ts in enumerate([701, 702, 703, 704]):
        t1 = connectify(nx.random_labeled_tree(NODES, seed=ts))
        t2 = connectify(nx.random_labeled_tree(NODES, seed=ts + 50))
        cands.append((f"TREE_{i}", (t1, t2)))
    cands.append(("LADDER", (nx.ladder_graph(NODES // 2),
                             nx.ladder_graph(NODES // 2))))
    cands.append(("WHEEL", (nx.wheel_graph(NODES - 1),
                            nx.wheel_graph(NODES - 1))))
    cands.append(("BARBELL", (connectify(nx.barbell_graph(5, 2)),
                              connectify(nx.barbell_graph(5, 2)))))
    cands.append(("STAR+PATH",
                  (nx.compose(nx.path_graph(NODES), nx.star_graph(NODES - 1)),
                   nx.compose(nx.path_graph(NODES), nx.star_graph(NODES - 1)))))
    return cands


def proxies(graphs, device):
    L = float(np.mean([nx.average_shortest_path_length(g) for g in graphs]))
    slopes = []
    for seed in PROXY_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        loader, _ = get_loaders("fashion", seed)
        model = TinyRandWireNN(graphs).to(device)
        s, _ = gradient_profile(model, loader, device)
        slopes.append(abs(s))
    return dict(neg_L=-L, grad_flatness=-float(np.mean(slopes)))


def train_one(graphs, device):
    seed = TRAIN_SEED
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
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
    return evaluate(model, test_loader, device)


if __name__ == "__main__":
    device = torch.device("cpu")
    os.makedirs(H8_DIR, exist_ok=True)
    cands = candidate_pool()
    print(f"H8 proxy at scale: {len(cands)} candidates, 1 train seed")
    print("=" * 72)

    records = []
    t0 = time.time()
    for i, (name, graphs) in enumerate(cands):
        px = proxies(graphs, device)
        acc = train_one(graphs, device)
        records.append(dict(
            name=name, acc=acc, **px,
            n_edges=float(np.mean([g.number_of_edges() for g in graphs])),
            max_degree=float(np.mean([max(d for _, d in g.degree())
                                      for g in graphs]))))
        print(f"  [{i+1:2d}/{len(cands)}] {name:12s} acc={acc:5.2f}% "
              f"gradflat={px['grad_flatness']:+.3f} negL={px['neg_L']:+.2f}")

    with open(os.path.join(H8_DIR, "h8_results.json"), "w") as f:
        json.dump(records, f, indent=2)

    # ---- Analysis ----
    acc = np.array([r["acc"] for r in records])
    oracle = acc.max()
    stats_out = dict(n_candidates=len(records))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, label in [(axes[0], "grad_flatness", "Gradient flatness at init"),
                           (axes[1], "neg_L", "Graph proxy (-L)")]:
        x = np.array([r[key] for r in records])
        sp = stats.spearmanr(x, acc)
        ax.scatter(x, acc, s=24, c="#2c3e50", alpha=0.7)
        worst = np.argsort(acc)[:6]
        best = np.argsort(acc)[-3:]
        for j in list(worst) + list(best):
            ax.annotate(records[j]["name"], (x[j], acc[j]), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
        ax.set_xlabel(label)
        ax.set_ylabel("Trained accuracy (%)")
        ax.set_title(f"rho={sp.statistic:.2f} (p={sp.pvalue:.1e})")
        ax.grid(alpha=0.3)
        order = np.argsort([-r[key] for r in records])
        regret_at_k = [round(float(oracle - max(acc[order[:k]])), 2)
                       for k in [1, 3, 5, 10]]
        stats_out[key] = dict(
            spearman_rho=round(float(sp.statistic), 3),
            spearman_p=float(f"{sp.pvalue:.2e}"),
            top1_pick=records[order[0]]["name"],
            regret_at_k={k: r for k, r in zip([1, 3, 5, 10], regret_at_k)})

    # P2: are proxy failures hub-related?
    ax = axes[2]
    gf = np.array([r["grad_flatness"] for r in records])
    rank_err = (stats.rankdata(gf) - stats.rankdata(acc)) / len(records)
    hub = np.array([r["max_degree"] for r in records])
    sp_hub = stats.spearmanr(hub, -rank_err)  # negative err = proxy underrates
    ax.scatter(hub, -rank_err, s=24, c="#8e44ad", alpha=0.7)
    for j in np.argsort(-hub)[:5]:
        ax.annotate(records[j]["name"], (hub[j], -rank_err[j]), fontsize=6,
                    textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("Max node degree (hub-ness)")
    ax.set_ylabel("Proxy underrating (acc rank - proxy rank, normalized)")
    ax.set_title(f"P2: hubs fool the proxy?  rho={sp_hub.statistic:.2f} "
                 f"(p={sp_hub.pvalue:.1e})")
    ax.grid(alpha=0.3)
    stats_out["hub_underrating"] = dict(
        rho=round(float(sp_hub.statistic), 3),
        p=float(f"{sp_hub.pvalue:.2e}"))

    fig.suptitle(f"H8: zero-cost proxy on a {len(records)}-candidate pool",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(H8_DIR, "fig_h8.png"), bbox_inches="tight")

    with open(os.path.join(H8_DIR, "h8_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print("=" * 72)
    print(json.dumps(stats_out, indent=2))
    print(f"Done in {(time.time() - t0) / 60:.1f} min")
