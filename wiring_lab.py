#!/usr/bin/env python
"""
The Wiring Lab — does the wiring actually matter?

A 2026 mini-replication of the core claim of
"Exploring Randomly Wired Neural Networks for Image Recognition" (Xie et al., 2019):
randomly wired networks are competitive with hand-designed wirings, and the
*generator* (graph family + parameters) matters more than any individual graph.

Experiments
  1. Topology tournament : 9 graph families, same node budget, same recipe
  2. Small-world sweep   : WS rewiring probability p swept from 0 (ring lattice) to 1
  3. Structure metrics   : graph theory metrics recorded for correlation analysis

All runs use the same TinyRandWireNN backbone (only the wiring changes),
the same data subset, optimizer, schedule and epoch budget.
"""
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import os
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.graph import build_graph, get_graph_info
from utils.network import StageBlock, conv_unit

RESULTS_DIR = "./output/wiring_lab"
NODES = 12          # nodes per stage graph (params are ~constant across wirings)
EPOCHS = 6
BATCH_SIZE = 64
TRAIN_SAMPLES = 4000
TEST_SAMPLES = 2000
LR = 1e-3
WEIGHT_DECAY = 0.01

# Mean degree matched to ~4 across families so density is comparable
TOPOLOGIES = {
    "WS":       dict(GRAPH_MODEL="WS", WS_K=4, WS_P=0.75),   # paper's best generator
    "ER":       dict(GRAPH_MODEL="ER", ER_P=0.36),           # p ~= 4/(N-1)
    "BA":       dict(GRAPH_MODEL="BA", BA_M=2),
    "NWS":      dict(GRAPH_MODEL="NWS", NWS_K=4, NWS_P=0.25),
    "PC":       dict(GRAPH_MODEL="PC", PC_M=2, PC_P=0.3),
    "RR":       dict(GRAPH_MODEL="RR", RR_D=4),
    "COMPLETE": dict(GRAPH_MODEL="COMPLETE"),                # DenseNet-like: all skips
    "PATH":     dict(GRAPH_MODEL="PATH"),                    # plain deep chain, no skips
    "CYCLE":    dict(GRAPH_MODEL="CYCLE"),                   # chain + one long skip
}

SWEEP_PS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [3, 7, 11]


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------

def make_graph(n_nodes, params, seed):
    """Build a connected graph; bump the seed until connectivity holds."""
    g_cfg = edict(dict(params))
    for s in range(seed, seed + 50):
        g_cfg.RND_SEED = s
        graph = build_graph(n_nodes, g_cfg)
        if nx.is_connected(graph):
            return graph, s
    raise RuntimeError(f"No connected graph found for {params} from seed {seed}")


def graph_metrics(graph):
    """Undirected metrics + metrics of the DAG actually executed by StageBlock."""
    nodes, input_nodes, output_nodes = get_graph_info(graph)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(graph.number_of_nodes()))
    for node in nodes:
        for src in node.inputs:
            dag.add_edge(src, node.id)
    return dict(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        avg_degree=2.0 * graph.number_of_edges() / graph.number_of_nodes(),
        avg_path_length=nx.average_shortest_path_length(graph),
        clustering=nx.average_clustering(graph),
        dag_depth=nx.dag_longest_path_length(dag),
        n_input_nodes=len(input_nodes),
        n_output_nodes=len(output_nodes),
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class TinyRandWireNN(nn.Module):
    """Two randomly wired stages; only the wiring differs between runs."""

    def __init__(self, graphs, in_ch=1, C=32, num_classes=10):
        super().__init__()
        g1, g2 = graphs
        self.stem = nn.Sequential(conv_unit(in_ch, C // 2, 2), nn.BatchNorm2d(C // 2))
        self.stage1 = StageBlock(g1, C // 2, C)    # 14x14 -> 7x7
        self.stage2 = StageBlock(g2, C, 2 * C)     # 7x7  -> 4x4
        self.head = nn.Sequential(nn.ReLU(True), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(2 * C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.head(x).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def get_loaders(dataset_name, seed):
    if dataset_name == "fashion":
        ds_cls, mean, std = datasets.FashionMNIST, 0.2860, 0.3530
    else:
        ds_cls, mean, std = datasets.MNIST, 0.1307, 0.3081

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    train_full = ds_cls("./dataset/", train=True, download=False, transform=tf)
    test_full = ds_cls("./dataset/", train=False, download=False, transform=tf)

    # Fixed subsets so every run sees identical data
    rng = np.random.RandomState(0)
    train_idx = rng.choice(len(train_full), TRAIN_SAMPLES, replace=False)
    test_idx = rng.choice(len(test_full), TEST_SAMPLES, replace=False)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(train_full, train_idx.tolist()),
                              batch_size=BATCH_SIZE, shuffle=True, generator=gen)
    test_loader = DataLoader(Subset(test_full, test_idx.tolist()),
                             batch_size=256, shuffle=False)
    return train_loader, test_loader


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def run_one(tag, graph_params, seed, dataset_name, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    g1, used_seed1 = make_graph(NODES, graph_params, seed)
    g2, used_seed2 = make_graph(NODES, graph_params, seed + 100)
    metrics = {k: (graph_metrics(g1)[k] + graph_metrics(g2)[k]) / 2.0
               for k in graph_metrics(g1)}

    train_loader, test_loader = get_loaders(dataset_name, seed)

    model = TinyRandWireNN((g1, g2)).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        history.append(acc)
    elapsed = time.time() - t0

    result = dict(
        tag=tag,
        graph_params=dict(graph_params),
        seed=seed,
        graph_seeds=[used_seed1, used_seed2],
        dataset=dataset_name,
        params=n_params,
        final_acc=history[-1],
        best_acc=max(history),
        history=history,
        train_time_s=round(elapsed, 1),
        **{f"g_{k}": round(v, 4) for k, v in metrics.items()},
    )
    return result, (g1, g2)


def append_result(result, path):
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


# --------------------------------------------------------------------------
# Experiment suites
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashion", choices=["mnist", "fashion"])
    parser.add_argument("--quick", action="store_true", help="single short smoke-test run")
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "results.jsonl")

    if args.quick:
        global EPOCHS, TRAIN_SAMPLES
        EPOCHS, TRAIN_SAMPLES = 1, 1000
        result, _ = run_one("WS", TOPOLOGIES["WS"], 3, args.dataset, device)
        print(json.dumps(result, indent=2))
        return

    # Fresh results file for a full suite
    if os.path.exists(results_path):
        os.rename(results_path, results_path + f".bak.{int(time.time())}")

    total_runs = len(TOPOLOGIES) * len(SEEDS) + len(SWEEP_PS) * len(SEEDS)
    done = 0
    suite_t0 = time.time()

    print(f"Dataset: {args.dataset} | {TRAIN_SAMPLES} train / {TEST_SAMPLES} test")
    print(f"Budget per run: {EPOCHS} epochs, N={NODES} nodes/stage")
    print(f"Total runs: {total_runs}")
    print("=" * 72)

    # ---- Experiment 1: topology tournament ----
    print("\n[1/2] TOPOLOGY TOURNAMENT")
    for name, params in TOPOLOGIES.items():
        for seed in SEEDS:
            result, graphs = run_one(name, params, seed, args.dataset, device)
            result["experiment"] = "tournament"
            append_result(result, results_path)
            done += 1
            print(f"  [{done:2d}/{total_runs}] {name:9s} seed={seed} "
                  f"acc={result['final_acc']:5.2f}% "
                  f"depth={result['g_dag_depth']:4.1f} "
                  f"L={result['g_avg_path_length']:.2f} "
                  f"C={result['g_clustering']:.2f} "
                  f"({result['train_time_s']}s)")

    # ---- Experiment 2: small-world sweep ----
    print("\n[2/2] SMALL-WORLD SWEEP (WS, k=4, p: 0 -> 1)")
    for p in SWEEP_PS:
        params = dict(GRAPH_MODEL="WS", WS_K=4, WS_P=p)
        for seed in SEEDS:
            result, graphs = run_one(f"WS_p{p}", params, seed, args.dataset, device)
            result["experiment"] = "sweep"
            result["ws_p"] = p
            append_result(result, results_path)
            done += 1
            print(f"  [{done:2d}/{total_runs}] p={p:<4} seed={seed} "
                  f"acc={result['final_acc']:5.2f}% "
                  f"L={result['g_avg_path_length']:.2f} "
                  f"C={result['g_clustering']:.2f} "
                  f"({result['train_time_s']}s)")

    print("\n" + "=" * 72)
    print(f"Suite complete in {(time.time() - suite_t0) / 60:.1f} min")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
