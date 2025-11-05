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
  if cfg.GRAPH_MODEL == 'ER':
    return nx.erdos_renyi_graph(Nodes, cfg.ER_P, seed=cfg.RND_SEED)
  elif cfg.GRAPH_MODEL == 'BA':
    return nx.barabasi_albert_graph(Nodes, cfg.BA_M, seed=cfg.RND_SEED)
  elif cfg.GRAPH_MODEL == 'WS':
    return nx.connected_watts_strogatz_graph(Nodes, cfg.WS_K, cfg.WS_P, tries=200, seed=cfg.RND_SEED)

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