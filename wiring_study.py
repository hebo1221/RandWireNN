#!/usr/bin/env python
"""
Short Paths Study — from correlation to mechanism.

The Wiring Lab (wiring_lab.py) found that average shortest path length
correlates with accuracy at r = -0.84. That result was correlational and
confounded: edge count, clustering and path length all co-varied.

This study tests three pre-registered hypotheses with controlled interventions:

  H1 (causal dose-response). Holding edge count constant, accuracy increases
      as rewiring shortens average path length.
      Intervention: WS k=2 sweep (12 edges fixed, L: 3.27 -> ~2.2).

  H2 (marginal value of shortcuts). Adding k random skip edges to a plain
      chain (PATH) yields diminishing accuracy returns in k.
      Intervention: PATH + k skips, k in {0, 1, 2, 4, 8}.

  H3 (mechanism). The path-length effect is mediated by uneven gradient flow:
      the per-node gradient magnitude profile across DAG depth is flatter in
      short-path graphs, and gradient imbalance predicts accuracy.
      Measurement: per-node RMS gradient at initialization.

  Exploratory (learned wiring). RandWireNN learns a sigmoid mixing weight per
      edge. Do trained networks assign higher weights to shortcut edges
      (high betweenness, long skip distance)?

Design: 12 configurations x 5 seeds = 60 runs, identical recipe and budget,
node count fixed at 12 per stage so parameter counts are nearly identical.
"""
import matplotlib
matplotlib.use('Agg')

import json
import math
import os
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from scipy import stats

from wiring_lab import (NODES, EPOCHS, LR, WEIGHT_DECAY, TinyRandWireNN,
                        get_loaders, evaluate, graph_metrics, make_graph)

STUDY_DIR = "./output/wiring_study"
SEEDS = [3, 7, 11, 13, 17]
SWEEP_PS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SKIP_KS = [0, 1, 2, 4, 8]


# --------------------------------------------------------------------------
# Graph builders (interventions)
# --------------------------------------------------------------------------

def path_plus_skips(n, k, seed):
    """A plain chain with k random skip edges injected (edge-count = n-1+k)."""
    g = nx.path_graph(n)
    rng = random.Random(seed)
    candidates = [(i, j) for i in range(n) for j in range(i + 2, n)]
    rng.shuffle(candidates)
    for (i, j) in candidates[:k]:
        g.add_edge(i, j)
    return g


def build_config_graphs(config, seed):
    if config["family"] == "ws2":
        params = dict(GRAPH_MODEL="WS", WS_K=2, WS_P=config["p"])
        g1, _ = make_graph(NODES, params, seed)
        g2, _ = make_graph(NODES, params, seed + 100)
    elif config["family"] == "path_k":
        g1 = path_plus_skips(NODES, config["k"], seed)
        g2 = path_plus_skips(NODES, config["k"], seed + 100)
    elif config["family"] == "ref":
        params = dict(GRAPH_MODEL="WS", WS_K=4, WS_P=0.75)
        g1, _ = make_graph(NODES, params, seed)
        g2, _ = make_graph(NODES, params, seed + 100)
    else:
        raise ValueError(config["family"])
    return g1, g2


CONFIGS = (
    [dict(tag=f"WS2_p{p}", family="ws2", p=p) for p in SWEEP_PS]
    + [dict(tag=f"PATH_k{k}", family="path_k", k=k) for k in SKIP_KS]
    + [dict(tag="WS4_ref", family="ref")]
)


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

def node_depths(stage):
    """Longest path from any source for each node of a StageBlock."""
    depth = {}
    for node in stage.nodes:  # ids ascending, inputs all lower
        depth[node.id] = 0 if not node.inputs else 1 + max(depth[i] for i in node.inputs)
    return depth


def gradient_profile(model, loader, device):
    """Per-node RMS gradient at initialization, by DAG depth.

    Returns the mean slope of log10(RMS grad) vs depth across both stages
    (0 = perfectly even gradient flow) and the raw stage-2 profile.
    """
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(loader))
    model.train()
    model.zero_grad()
    criterion(model(x.to(device)), y.to(device)).backward()

    slopes, profiles = [], []
    for stage in (model.stage1, model.stage2):
        depth = node_depths(stage)
        ds, gs = [], []
        for node in stage.nodes:
            op = stage.nodeop[node.id]
            grads = [p.grad.flatten() for p in op.conv.parameters()
                     if p.grad is not None]
            rms = torch.cat(grads).pow(2).mean().sqrt().item()
            ds.append(depth[node.id])
            gs.append(rms)
        log_g = [math.log10(g + 1e-12) for g in gs]
        if len(set(ds)) > 1:
            slopes.append(stats.linregress(ds, log_g).slope)
        profiles.append([[d, round(g, 8)] for d, g in zip(ds, gs)])

    model.zero_grad()
    mean_slope = float(np.mean(slopes)) if slopes else 0.0
    return mean_slope, profiles


def extract_edge_weights(model, graphs, run_meta):
    """Learned sigmoid mixing weights per edge, with graph-centrality context.

    Only nodes with >1 input have learnable weights (single-input edges are
    implicitly weight 1 and excluded).
    """
    records = []
    for si, (stage, graph) in enumerate([(model.stage1, graphs[0]),
                                         (model.stage2, graphs[1])]):
        dag = nx.DiGraph()
        dag.add_nodes_from(range(graph.number_of_nodes()))
        for node in stage.nodes:
            for src in node.inputs:
                dag.add_edge(src, node.id)
        btw = nx.edge_betweenness_centrality(dag)
        depth = node_depths(stage)

        for node in stage.nodes:
            op = stage.nodeop[node.id]
            if op.input_nums > 1:
                weights = torch.sigmoid(op.mean_weight).detach().tolist()
                for src, w in zip(node.inputs, weights):
                    records.append(dict(
                        **run_meta, stage=si,
                        src=src, dst=node.id,
                        weight=round(w, 4),
                        betweenness=round(btw[(src, node.id)], 5),
                        skip_dist=node.id - src,
                        src_depth=depth[src],
                        n_inputs=op.input_nums,
                    ))
    return records


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def run_one(config, seed, dataset_name, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    g1, g2 = build_config_graphs(config, seed)
    m1, m2 = graph_metrics(g1), graph_metrics(g2)
    metrics = {k: (m1[k] + m2[k]) / 2.0 for k in m1}

    train_loader, test_loader = get_loaders(dataset_name, seed)
    model = TinyRandWireNN((g1, g2)).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # H3 measurement before any training
    grad_slope, grad_profiles = gradient_profile(model, train_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    t0 = time.time()
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        history.append(evaluate(model, test_loader, device))
    elapsed = time.time() - t0

    result = dict(
        tag=config["tag"],
        family=config["family"],
        ws_p=config.get("p"),
        k_skips=config.get("k"),
        seed=seed,
        dataset=dataset_name,
        params=n_params,
        final_acc=history[-1],
        best_acc=max(history),
        history=history,
        grad_slope=round(grad_slope, 4),
        grad_profile_stage2=grad_profiles[1],
        train_time_s=round(elapsed, 1),
        **{f"g_{k}": round(v, 4) for k, v in metrics.items()},
    )
    run_meta = dict(tag=config["tag"], family=config["family"], seed=seed)
    edges = extract_edge_weights(model, (g1, g2), run_meta)
    return result, edges


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashion", choices=["mnist", "fashion"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(STUDY_DIR, exist_ok=True)
    results_path = os.path.join(STUDY_DIR, "results.jsonl")
    edges_path = os.path.join(STUDY_DIR, "edge_weights.jsonl")

    if args.quick:
        result, edges = run_one(CONFIGS[0], 3, args.dataset, device)
        result.pop("grad_profile_stage2")
        print(json.dumps(result, indent=2))
        print(f"edge records: {len(edges)}, sample: {edges[0]}")
        return

    for path in (results_path, edges_path):
        if os.path.exists(path):
            os.rename(path, path + f".bak.{int(time.time())}")

    total = len(CONFIGS) * len(SEEDS)
    done = 0
    t0 = time.time()
    print(f"Short Paths Study: {len(CONFIGS)} configs x {len(SEEDS)} seeds "
          f"= {total} runs on {args.dataset}")
    print("=" * 72)

    for config in CONFIGS:
        for seed in SEEDS:
            result, edges = run_one(config, seed, args.dataset, device)
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            with open(edges_path, "a") as f:
                for e in edges:
                    f.write(json.dumps(e) + "\n")
            done += 1
            print(f"  [{done:2d}/{total}] {result['tag']:10s} seed={seed:2d} "
                  f"acc={result['final_acc']:5.2f}% "
                  f"L={result['g_avg_path_length']:.2f} "
                  f"gradslope={result['grad_slope']:+.3f} "
                  f"({result['train_time_s']}s)")

    print("=" * 72)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {results_path}")


if __name__ == "__main__":
    main()
