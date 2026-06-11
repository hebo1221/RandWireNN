#!/usr/bin/env python
"""
H6 — The path penalty scales with depth.

Pre-registered predictions:
  P1: The accuracy gap between randomly wired (WS k=4, p=0.75) and plain-chain
      (PATH) networks grows monotonically with the node budget N (6, 12, 24).
  P2: The total init-time gradient attenuation across a PATH stage
      (|slope| x depth) grows with N, while WS stays comparatively flat.

This is the historical ResNet story as a controlled experiment: shallow plain
nets are fine; the deeper you go, the more you need shortcuts.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from wiring_lab import (EPOCHS, LR, WEIGHT_DECAY, TinyRandWireNN, evaluate,
                        get_loaders, make_graph)
from wiring_study import gradient_profile, path_plus_skips

H6_DIR = "./output/h6_scaling"
SEEDS = [3, 7, 11, 13]
NS = [6, 12, 24]


def build(family, n, seed):
    if family == "PATH":
        import networkx as nx
        return (nx.path_graph(n), nx.path_graph(n))
    params = dict(GRAPH_MODEL="WS", WS_K=4, WS_P=0.75)
    return (make_graph(n, params, seed)[0], make_graph(n, params, seed + 100)[0])


if __name__ == "__main__":
    device = torch.device("cpu")
    os.makedirs(H6_DIR, exist_ok=True)
    results = []
    total = len(NS) * 2 * len(SEEDS)
    done = 0
    print(f"H6 depth scaling: N in {NS}, PATH vs WS, {len(SEEDS)} seeds "
          f"({total} runs)")
    print("=" * 72)

    for n in NS:
        for family in ["PATH", "WS"]:
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                graphs = build(family, n, seed)
                train_loader, test_loader = get_loaders("fashion", seed)
                model = TinyRandWireNN(graphs).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                slope, _ = gradient_profile(model, train_loader, device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                              weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS)
                t0 = time.time()
                for _ in range(EPOCHS):
                    model.train()
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        criterion(model(x), y).backward()
                        optimizer.step()
                    scheduler.step()
                acc = evaluate(model, test_loader, device)

                # Max DAG depth of stage-2 graph for total-attenuation calc
                from wiring_study import node_depths
                depth2 = max(node_depths(model.stage2).values())

                results.append(dict(
                    family=family, n_nodes=n, seed=seed, params=n_params,
                    final_acc=acc, grad_slope=round(slope, 4),
                    stage2_depth=depth2,
                    total_attenuation=round(abs(slope) * depth2, 3),
                    time_s=round(time.time() - t0, 1)))
                done += 1
                print(f"  [{done:2d}/{total}] N={n:2d} {family:4s} seed={seed:2d} "
                      f"acc={acc:5.2f}% params={n_params:,} "
                      f"slope={slope:+.3f} depth={depth2} "
                      f"({results[-1]['time_s']}s)")

    with open(os.path.join(H6_DIR, "scaling_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 72)
    for n in NS:
        ws = [r["final_acc"] for r in results
              if r["n_nodes"] == n and r["family"] == "WS"]
        pa = [r["final_acc"] for r in results
              if r["n_nodes"] == n and r["family"] == "PATH"]
        print(f"  N={n:2d}: WS={np.mean(ws):5.2f} PATH={np.mean(pa):5.2f} "
              f"gap={np.mean(ws) - np.mean(pa):+5.2f}")
