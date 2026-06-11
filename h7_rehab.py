#!/usr/bin/env python
"""
H7 — Rehabilitation: is the ablation collapse co-adaptation or irreplaceable
computation?

H4 showed that cutting all skips from a trained PATH+8skip network collapses
it to chance (9.9%) even after BN recalibration. Two readings remain:

  (a) Co-adaptation: the surviving chain weights are fine *for a chain*, they
      just need to re-settle. Fine-tuning the amputated network should reach
      the level of a chain trained from scratch (70.9%).
  (b) Transferable scaffolding: training with skips left useful structure in
      the chain weights. Fine-tuning should EXCEED the from-scratch chain.
  (c) Harmful co-adaptation: weights trained with skips are a bad init for
      chain computation. Fine-tuning should fall SHORT of from-scratch.

Protocol per seed: train PATH_k8 (6 ep) -> cut all skips -> fine-tune 6 ep
with the identical recipe -> compare against the PATH_k0-from-scratch anchor
(70.92, same seeds, Short Paths Study). Symmetric condition: cut chain edges
that have skip alternatives, fine-tune the skip-heavy remnant.

Note: ablated logits (-1e9) cannot resurrect during fine-tuning -- sigmoid is
saturated, so their gradient is exactly zero.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from wiring_lab import EPOCHS, LR, NODES, WEIGHT_DECAY, evaluate, get_loaders
from wiring_study import path_plus_skips
from h4_scaffolding import ablatable_edges, set_edge, train_model

H7_DIR = "./output/h7_rehab"
SEEDS = [3, 7, 11, 13, 17]
SCRATCH_CHAIN_ANCHOR = 70.92  # PATH_k0, same seeds/recipe (Short Paths Study)


def fine_tune(model, seed, device, epochs=EPOCHS):
    train_loader, test_loader = get_loaders("fashion", seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    curve = []
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        curve.append(evaluate(model, test_loader, device))
    return curve


if __name__ == "__main__":
    device = torch.device("cpu")
    os.makedirs(H7_DIR, exist_ok=True)
    results = []
    print("H7 rehabilitation: fine-tune after amputation (PATH_k8)")
    print("=" * 72)

    for seed in SEEDS:
        t0 = time.time()
        graphs = (path_plus_skips(NODES, 8, seed),
                  path_plus_skips(NODES, 8, seed + 100))
        model, test_loader = train_model(graphs, seed, "fashion", device)
        model.eval()
        base = evaluate(model, test_loader, device)
        edges = ablatable_edges(model, graphs)

        rec = dict(seed=seed, baseline=base)
        for label, pred in [("skips", lambda e: e["skip_dist"] > 1),
                            ("chains", lambda e: e["skip_dist"] == 1)]:
            import copy
            m = copy.deepcopy(model)
            for e in [e for e in edges if pred(e)]:
                set_edge(m, e, -1e9)
            m.eval()
            rec[f"cut_{label}_acc"] = evaluate(m, test_loader, device)
            curve = fine_tune(m, seed, device)
            rec[f"rehab_{label}_curve"] = curve
            rec[f"rehab_{label}_final"] = curve[-1]

        rec["time_s"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"  seed={seed:2d} base={base:5.2f} | "
              f"skips: cut={rec['cut_skips_acc']:5.2f} "
              f"rehab={rec['rehab_skips_final']:5.2f} | "
              f"chains: cut={rec['cut_chains_acc']:5.2f} "
              f"rehab={rec['rehab_chains_final']:5.2f} "
              f"({rec['time_s']}s)")

    with open(os.path.join(H7_DIR, "rehab_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 72)
    for key in ["baseline", "cut_skips_acc", "rehab_skips_final",
                "cut_chains_acc", "rehab_chains_final"]:
        vals = [r[key] for r in results]
        print(f"  {key:20s} {np.mean(vals):5.2f} ± {np.std(vals):4.2f}")
    print(f"  scratch-chain anchor  {SCRATCH_CHAIN_ANCHOR:5.2f} "
          f"(PATH_k0, same seeds)")
    rehab = np.mean([r["rehab_skips_final"] for r in results])
    diff = rehab - SCRATCH_CHAIN_ANCHOR
    verdict = ("(b) transferable scaffolding" if diff > 1.5 else
               "(c) harmful co-adaptation" if diff < -1.5 else
               "(a) pure co-adaptation")
    print(f"  rehab - scratch = {diff:+.2f}  ->  {verdict}")
