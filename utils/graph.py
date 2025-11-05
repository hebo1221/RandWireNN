import networkx as nx
import collections
import matplotlib.pyplot as plt
import yaml
import json
from typing import List, Tuple, Any

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

def get_graph_info(graph: nx.Graph) -> Tuple[List[Node], List[int], List[int]]:
  input_nodes = []
  output_nodes = []
  Nodes = []
  for node in range(graph.number_of_nodes()):
    tmp = list(graph.neighbors(node))
    tmp.sort()
    type = -1
    if node < tmp[0]:
      input_nodes.append(node)
      type = 0
    if node > tmp[-1]:
      output_nodes.append(node)
      type = 1
    Nodes.append(Node(node, [n for n in tmp if n < node], type))
  return Nodes, input_nodes, output_nodes

def build_graph(Nodes: int, cfg: Any) -> nx.Graph:
  """
  Build a random graph based on the specified model.

  Supported models:
  - ER: Erdos-Renyi (classical random graph)
  - BA: Barabasi-Albert (scale-free, preferential attachment)
  - WS: Watts-Strogatz (small-world with rewiring)
  - NWS: Newman-Watts-Strogatz (small-world with edge addition)
  - PC: Powerlaw Cluster (scale-free with clustering)
  - RR: Random Regular (all nodes have same degree)
  - COMPLETE: Complete graph (all nodes connected)
  """
  model = cfg.GRAPH_MODEL.upper()

  if model == 'ER':
    # Erdos-Renyi: Random graph with edge probability p
    return nx.erdos_renyi_graph(Nodes, cfg.ER_P, seed=cfg.RND_SEED)

  elif model == 'BA':
    # Barabasi-Albert: Scale-free graph with preferential attachment
    return nx.barabasi_albert_graph(Nodes, cfg.BA_M, seed=cfg.RND_SEED)

  elif model == 'WS':
    # Watts-Strogatz: Small-world graph with rewiring
    return nx.connected_watts_strogatz_graph(Nodes, cfg.WS_K, cfg.WS_P, tries=200, seed=cfg.RND_SEED)

  elif model == 'NWS':
    # Newman-Watts-Strogatz: Small-world with edge addition (no rewiring)
    p = getattr(cfg, 'NWS_P', 0.1)
    k = getattr(cfg, 'NWS_K', 4)
    graph = nx.newman_watts_strogatz_graph(Nodes, k, p, seed=cfg.RND_SEED)
    # Ensure connectivity
    if not nx.is_connected(graph):
      # Add edges to connect components
      components = list(nx.connected_components(graph))
      for i in range(len(components) - 1):
        node_a = list(components[i])[0]
        node_b = list(components[i+1])[0]
        graph.add_edge(node_a, node_b)
    return graph

  elif model == 'PC':
    # Powerlaw Cluster: Scale-free with triangle formation
    m = getattr(cfg, 'PC_M', 3)
    p = getattr(cfg, 'PC_P', 0.1)
    graph = nx.powerlaw_cluster_graph(Nodes, m, p, seed=cfg.RND_SEED)
    # Ensure connectivity
    if not nx.is_connected(graph):
      components = list(nx.connected_components(graph))
      for i in range(len(components) - 1):
        node_a = list(components[i])[0]
        node_b = list(components[i+1])[0]
        graph.add_edge(node_a, node_b)
    return graph

  elif model == 'RR':
    # Random Regular: All nodes have exactly d neighbors
    d = getattr(cfg, 'RR_D', 4)
    # d * n must be even
    if (d * Nodes) % 2 != 0:
      d = d + 1 if d + 1 < Nodes else d - 1
    return nx.random_regular_graph(d, Nodes, seed=cfg.RND_SEED)

  elif model == 'COMPLETE':
    # Complete graph: All nodes connected to all others
    return nx.complete_graph(Nodes)

  elif model == 'PATH':
    # Path graph: Linear chain
    return nx.path_graph(Nodes)

  elif model == 'CYCLE':
    # Cycle graph: Ring topology
    return nx.cycle_graph(Nodes)

  else:
    raise ValueError(f"Unknown graph model: {cfg.GRAPH_MODEL}. "
                    f"Available models: ER, BA, WS, NWS, PC, RR, COMPLETE, PATH, CYCLE")

def save_graph(graph: nx.Graph, path: str) -> None:
  """Save graph to YAML format using node-link data structure."""
  graph_data = nx.node_link_data(graph)
  with open(path, 'w') as f:
    yaml.dump(graph_data, f)

def load_graph(path: str) -> nx.Graph:
  """Load graph from YAML format using node-link data structure."""
  with open(path, 'r') as f:
    graph_data = yaml.safe_load(f)
  return nx.node_link_graph(graph_data)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    __C = edict()
    cfg = __C
    # model config
    __C.GRAPH_MODEL = "WS"

    # Erdos-Renyi  model
    __C.ER_P = 0.2
    # Barabasi-Albert model
    __C.BA_M =  5
    # Watts-Strogatz model
    __C.WS_K = 4
    __C.WS_P = 0.75
    # For reproducibility
    __C.RND_SEED = 3

    graph1 = build_graph(32, cfg)

    options = {
    'node_color': 'Yellow',
    'node_size': 500,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    }

    nx.draw_networkx(graph1, arrows=True, **options)
    plt.show()