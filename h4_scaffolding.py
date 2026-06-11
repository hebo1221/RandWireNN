#!/usr/bin/env python
"""
H4 — The Scaffolding Hypothesis (causal test by inference-time edge ablation).

Pre-registered predictions:
  P1: In trained PATH+8skip networks, ablating a skip edge at inference costs
      far less than that edge's training-time value (+1.07 pt/edge, measured
      as (acc_k8 - acc_k0)/8 in the Short Paths Study).
  P2: Removing ALL skip edges at inference keeps accuracy well above networks
      that were trained without skips (PATH_k0-trained = 70.9%).
  P3: Per-edge ablation damage correlates with the edge's learned mixing
      weight and betweenness centrality (in organic WS graphs too).

Mechanism of ablation: Node_OP mixes multi-input edges with sigmoid(w_i);
setting w_i = -1e9 makes sigmoid exactly 0, surgically removing one edge
from the forward pass with no other change.
"""
import matplotlib
matplotlib.use('Agg')

import json
import os
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from wiring_lab import EPOCHS, LR, NODES, WEIGHT_DECAY, TinyRandWireNN, evaluate, get_loaders, make_graph
from wiring_study import node_depths, path_plus_skips

H4_DIR = "./output/h4_scaffolding"
SEEDS = [3, 7, 11, 13, 17]


def build_dag_betweenness(stage, graph):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(graph.number_of_nodes()))
    for node in stage.nodes:
        for src in node.inputs:
            dag.add_edge(src, node.id)
    return nx.edge_betweenness_centrality(dag)


def ablatable_edges(model, graphs):
    """All edges entering multi-input nodes: (stage_idx, node_id, input_idx, src)."""
    edges = []
    for si, (stage, graph) in enumerate([(model.stage1, graphs[0]),
                                         (model.stage2, graphs[1])]):
        btw = build_dag_betweenness(stage, graph)
        for node in stage.nodes:
            op = stage.nodeop[node.id]
            if op.input_nums > 1:
                weights = torch.sigmoid(op.mean_weight).detach().tolist()
                for idx, src in enumerate(node.inputs):
                    edges.append(dict(stage=si, dst=node.id, input_idx=idx,
                                      src=src, skip_dist=node.id - src,
                                      weight=round(weights[idx], 4),
                                      betweenness=round(btw[(src, node.id)], 5)))
    return edges


def set_edge(model, edge, value):
    stage = model.stage1 if edge["stage"] == 0 else model.stage2
    op = stage.nodeop[edge["dst"]]
    old = op.mean_weight.data[edge["input_idx"]].item()
    op.mean_weight.data[edge["input_idx"]] = value
    return old


def train_model(graphs, seed, dataset, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_loader, test_loader = get_loaders(dataset, seed)
    model = TinyRandWireNN(graphs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
    return model, test_loader


def run_family(family, seeds, dataset, device, results):
    for seed in seeds:
        if family == "PATH_k8":
            graphs = (path_plus_skips(NODES, 8, seed),
                      path_plus_skips(NODES, 8, seed + 100))
        else:  # WS4
            params = dict(GRAPH_MODEL="WS", WS_K=4, WS_P=0.75)
            graphs = (make_graph(NODES, params, seed)[0],
                      make_graph(NODES, params, seed + 100)[0])

        t0 = time.time()
        model, test_loader = train_model(graphs, seed, dataset, device)
        model.eval()
        baseline = evaluate(model, test_loader, device)

        # Per-edge ablation
        edges = ablatable_edges(model, graphs)
        for edge in edges:
            old = set_edge(model, edge, -1e9)
            acc = evaluate(model, test_loader, device)
            set_edge(model, edge, old)
            edge["delta_acc"] = round(baseline - acc, 4)

        # Cumulative ablations (PATH_k8 only: skip vs chain dichotomy)
        cumulative = {}
        if family == "PATH_k8":
            for kind, predicate in [("all_skips", lambda e: e["skip_dist"] > 1),
                                    ("all_chain_alternatives",
                                     lambda e: e["skip_dist"] == 1)]:
                group = [e for e in edges if predicate(e)]
                olds = [set_edge(model, e, -1e9) for e in group]
                cumulative[kind] = dict(
                    acc=evaluate(model, test_loader, device),
                    n_edges_removed=len(group))
                for e, old in zip(group, olds):
                    set_edge(model, e, old)

        results.append(dict(family=family, seed=seed,
                            baseline_acc=baseline, edges=edges,
                            cumulative=cumulative,
                            time_s=round(time.time() - t0, 1)))
        skips = [e["delta_acc"] for e in edges if e["skip_dist"] > 1]
        chains = [e["delta_acc"] for e in edges if e["skip_dist"] == 1]
        print(f"  {family} seed={seed:2d} base={baseline:5.2f}% "
              f"| skip dAcc={np.mean(skips) if skips else float('nan'):+.2f} "
              f"| chain dAcc={np.mean(chains) if chains else float('nan'):+.2f} "
              f"| cum={ {k: v['acc'] for k, v in cumulative.items()} } "
              f"({results[-1]['time_s']}s)")


if __name__ == "__main__":
    device = torch.device("cpu")
    os.makedirs(H4_DIR, exist_ok=True)
    results = []
    print("H4 Scaffolding test: inference-time edge ablation")
    print("=" * 72)
    run_family("PATH_k8", SEEDS, "fashion", device, results)
    run_family("WS4", SEEDS, "fashion", device, results)
    out = os.path.join(H4_DIR, "ablation_results.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"saved {out}")
