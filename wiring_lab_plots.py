#!/usr/bin/env python
"""
Visualization and analysis for the Wiring Lab experiments.

Produces:
  fig_topologies.png  - the 9 wiring DAGs drawn in layered (depth) layout
  fig_tournament.png  - accuracy by topology, individual seeds shown
  fig_smallworld.png  - the classic Watts-Strogatz plot with accuracy overlaid
  fig_structure.png   - accuracy vs. graph-theoretic structure metrics
"""
import matplotlib
matplotlib.use('Agg')

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from easydict import EasyDict as edict

from utils.graph import build_graph, get_graph_info
from wiring_lab import TOPOLOGIES, NODES, RESULTS_DIR, make_graph

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def load_results(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def to_dag(graph):
    nodes, input_nodes, output_nodes = get_graph_info(graph)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(graph.number_of_nodes()))
    for node in nodes:
        for src in node.inputs:
            dag.add_edge(src, node.id)
    return dag, set(input_nodes), set(output_nodes)


def node_depths(dag):
    """Longest path from any source, per node (defines the execution layer)."""
    depth = {}
    for n in nx.topological_sort(dag):
        preds = list(dag.predecessors(n))
        depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth


# --------------------------------------------------------------------------

def fig_topologies(out_path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, (name, params) in zip(axes.flat, TOPOLOGIES.items()):
        graph, _ = make_graph(NODES, params, 3)
        dag, inputs, outputs = to_dag(graph)
        depth = node_depths(dag)
        nx.set_node_attributes(dag, depth, "layer")
        pos = nx.multipartite_layout(dag, subset_key="layer")

        colors = ["#2ecc71" if n in inputs else "#e74c3c" if n in outputs
                  else "#3498db" for n in dag.nodes]
        # Curved edges so skip connections stay visible in chain-like layouts
        nx.draw_networkx_edges(dag, pos, ax=ax, arrows=True, arrowsize=9,
                               edge_color="#7f8c8d", width=1.1,
                               connectionstyle="arc3,rad=0.25")
        nx.draw_networkx_nodes(dag, pos, ax=ax, node_color=colors, node_size=260)
        nx.draw_networkx_labels(dag, pos, ax=ax, font_size=7, font_color="white")
        ax.set_title(f"{name}  ({graph.number_of_edges()} edges, "
                     f"depth {max(depth.values())})", fontsize=10)
        ax.axis("off")

    fig.suptitle("The 9 wirings as executed DAGs "
                 "(green=input nodes, red=output nodes)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def fig_tournament(results, out_path):
    runs = [r for r in results if r.get("experiment") == "tournament"]
    by_tag = defaultdict(list)
    for r in runs:
        by_tag[r["tag"]].append(r["final_acc"])

    tags = sorted(by_tag, key=lambda t: -np.mean(by_tag[t]))
    means = [np.mean(by_tag[t]) for t in tags]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#e67e22" if t in ("PATH", "CYCLE", "COMPLETE") else "#3498db"
                  for t in tags]
    ax.bar(tags, means, color=bar_colors, alpha=0.75, zorder=2)
    for i, t in enumerate(tags):
        ax.scatter([i] * len(by_tag[t]), by_tag[t], color="black", s=18,
                   zorder=3, label="individual seed" if i == 0 else None)

    lo = min(min(v) for v in by_tag.values())
    ax.set_ylim(max(0, lo - 3), max(means) + 2)
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(f"Topology tournament — FashionMNIST, {len(runs)} runs\n"
                 "blue = random generators, orange = deterministic wirings")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def fig_smallworld(results, out_path):
    runs = [r for r in results if r.get("experiment") == "sweep"]
    by_p = defaultdict(lambda: dict(acc=[], L=[], C=[]))
    for r in runs:
        by_p[r["ws_p"]]["acc"].append(r["final_acc"])
        by_p[r["ws_p"]]["L"].append(r["g_avg_path_length"])
        by_p[r["ws_p"]]["C"].append(r["g_clustering"])

    ps = sorted(by_p)
    acc_mean = [np.mean(by_p[p]["acc"]) for p in ps]
    acc_all = [(p, a) for p in ps for a in by_p[p]["acc"]]
    L0 = np.mean(by_p[0.0]["L"])
    C0 = np.mean(by_p[0.0]["C"])
    L_norm = [np.mean(by_p[p]["L"]) / L0 for p in ps]
    C_norm = [np.mean(by_p[p]["C"]) / C0 for p in ps]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(ps, L_norm, "o-", color="#8e44ad", label="L(p)/L(0)  avg path length")
    ax1.plot(ps, C_norm, "s-", color="#16a085", label="C(p)/C(0)  clustering")
    ax1.set_xlabel("WS rewiring probability p")
    ax1.set_ylabel("Normalized graph metric")
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(loc="center left")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ps, acc_mean, "D-", color="#c0392b", linewidth=2,
             label="test accuracy")
    ax2.scatter([p for p, _ in acc_all], [a for _, a in acc_all],
                color="#c0392b", s=14, alpha=0.5)
    ax2.set_ylabel("Final test accuracy (%)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.legend(loc="center right")

    ax1.set_title("Small-world sweep: Watts-Strogatz rewiring vs. accuracy\n"
                  "(ring lattice at p=0  →  random graph at p=1)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def fig_structure(results, out_path):
    runs = [r for r in results]  # all runs carry graph metrics
    metrics = [
        ("g_dag_depth", "DAG depth (longest path)"),
        ("g_avg_path_length", "Avg shortest path length"),
        ("g_clustering", "Clustering coefficient"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (key, label) in zip(axes, metrics):
        xs = [r[key] for r in runs]
        ys = [r["final_acc"] for r in runs]
        tournament = [r.get("experiment") == "tournament" for r in runs]
        ax.scatter([x for x, t in zip(xs, tournament) if t],
                   [y for y, t in zip(ys, tournament) if t],
                   c="#3498db", s=26, alpha=0.8, label="tournament")
        ax.scatter([x for x, t in zip(xs, tournament) if not t],
                   [y for y, t in zip(ys, tournament) if not t],
                   c="#c0392b", s=26, alpha=0.8, marker="D", label="WS sweep")
        rho = np.corrcoef(xs, ys)[0, 1]
        ax.set_xlabel(label)
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_title(f"r = {rho:+.2f}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Does graph structure predict accuracy?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def summary_table(results):
    runs = [r for r in results if r.get("experiment") == "tournament"]
    by_tag = defaultdict(list)
    meta = {}
    for r in runs:
        by_tag[r["tag"]].append(r)
        meta[r["tag"]] = r

    print("\nTOURNAMENT SUMMARY (mean over seeds)")
    print(f"{'topology':10s} {'acc%':>7s} {'±':>5s} {'params':>8s} "
          f"{'edges':>6s} {'depth':>6s} {'L':>5s} {'C':>5s}")
    rows = []
    for tag in sorted(by_tag, key=lambda t: -np.mean([x["final_acc"] for x in by_tag[t]])):
        accs = [x["final_acc"] for x in by_tag[tag]]
        m = meta[tag]
        row = (tag, np.mean(accs), np.std(accs), m["params"],
               np.mean([x["g_n_edges"] for x in by_tag[tag]]),
               np.mean([x["g_dag_depth"] for x in by_tag[tag]]),
               np.mean([x["g_avg_path_length"] for x in by_tag[tag]]),
               np.mean([x["g_clustering"] for x in by_tag[tag]]))
        rows.append(row)
        print(f"{row[0]:10s} {row[1]:7.2f} {row[2]:5.2f} {row[3]:8,d} "
              f"{row[4]:6.1f} {row[5]:6.1f} {row[6]:5.2f} {row[7]:5.2f}")
    return rows


if __name__ == "__main__":
    results_path = os.path.join(RESULTS_DIR, "results.jsonl")
    results = load_results(results_path)
    print(f"Loaded {len(results)} runs")

    fig_topologies(os.path.join(RESULTS_DIR, "fig_topologies.png"))
    fig_tournament(results, os.path.join(RESULTS_DIR, "fig_tournament.png"))
    fig_smallworld(results, os.path.join(RESULTS_DIR, "fig_smallworld.png"))
    fig_structure(results, os.path.join(RESULTS_DIR, "fig_structure.png"))
    summary_table(results)
