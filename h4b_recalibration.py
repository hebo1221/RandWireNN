#!/usr/bin/env python
"""
H4b — Fair scaffolding test: edge ablation WITH BatchNorm recalibration.

The naive ablation in h4_scaffolding.py confounds two effects:
  (a) loss of the computation carried by the removed edge, and
  (b) stale BatchNorm statistics (the input distribution to every downstream
      node shifts when an edge is removed).

Standard remedy from the pruning literature: after ablation, reset BN running
statistics and re-estimate them with forward passes over the training data
(no gradient updates, no weight changes). Whatever damage remains after
recalibration is the true computational reliance on the removed edges.

Decision rule (pre-registered):
  - If PATH_k8 with all skips removed + BN recalib recovers to near the
    PATH_k0-trained baseline (70.9%), the converged forward computation is
    essentially chain-like and skips were training scaffolding after all.
  - If it stays far below 70.9%, the network genuinely computes through its
    skips, and the low mixing weights were a red herring.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from wiring_lab import NODES, TinyRandWireNN, evaluate, get_loaders
from wiring_study import path_plus_skips
from h4_scaffolding import ablatable_edges, set_edge, train_model

H4B_DIR = "./output/h4_scaffolding"
SEEDS = [3, 7, 11, 13, 17]


def recalibrate_bn(model, loader, device):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
    model.eval()


def condition_acc(model, edges_to_cut, train_loader, test_loader, device,
                  recalib):
    m = copy.deepcopy(model)
    for e in edges_to_cut:
        set_edge(m, e, -1e9)
    if recalib:
        recalibrate_bn(m, train_loader, device)
    return evaluate(m, test_loader, device)


if __name__ == "__main__":
    device = torch.device("cpu")
    os.makedirs(H4B_DIR, exist_ok=True)
    results = []
    print("H4b: cumulative ablation with BatchNorm recalibration (PATH_k8)")
    print("=" * 72)

    for seed in SEEDS:
        t0 = time.time()
        graphs = (path_plus_skips(NODES, 8, seed),
                  path_plus_skips(NODES, 8, seed + 100))
        model, test_loader = train_model(graphs, seed, "fashion", device)
        train_loader, _ = get_loaders("fashion", seed)
        model.eval()

        edges = ablatable_edges(model, graphs)
        skips = [e for e in edges if e["skip_dist"] > 1]
        chains = [e for e in edges if e["skip_dist"] == 1]

        rec = dict(seed=seed, baseline=evaluate(model, test_loader, device))
        for label, cut in [("skips", skips), ("chains", chains)]:
            rec[f"no_{label}_naive"] = condition_acc(
                model, cut, train_loader, test_loader, device, recalib=False)
            rec[f"no_{label}_recalib"] = condition_acc(
                model, cut, train_loader, test_loader, device, recalib=True)
        # Control: recalibration alone, nothing removed
        rec["intact_recalib"] = condition_acc(
            model, [], train_loader, test_loader, device, recalib=True)

        rec["time_s"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"  seed={seed:2d} base={rec['baseline']:5.2f} | "
              f"no_skips naive={rec['no_skips_naive']:5.2f} "
              f"recalib={rec['no_skips_recalib']:5.2f} | "
              f"no_chains naive={rec['no_chains_naive']:5.2f} "
              f"recalib={rec['no_chains_recalib']:5.2f} | "
              f"intact+recal={rec['intact_recalib']:5.2f} "
              f"({rec['time_s']}s)")

    out = os.path.join(H4B_DIR, "recalibration_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 72)
    for key in ["baseline", "no_skips_naive", "no_skips_recalib",
                "no_chains_naive", "no_chains_recalib", "intact_recalib"]:
        vals = [r[key] for r in results]
        print(f"  {key:18s} {np.mean(vals):5.2f} ± {np.std(vals):4.2f}")
    print(f"  (reference: PATH_k0 trained-without-skips = 70.92)")
    print(f"saved {out}")
