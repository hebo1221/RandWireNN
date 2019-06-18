import networkx as nx
import collections
import matplotlib.pyplot as plt

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

def get_graph_info(graph):
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

def build_graph(Nodes, cfg ):
  if cfg.GRAPH_MODEL == 'ER':
    return nx.random_graphs.erdos_renyi_graph(Nodes, cfg.ER_P, cfg.RND_SEED)
  elif cfg.GRAPH_MODEL == 'BA':
    return nx.random_graphs.barabasi_albert_graph(Nodes, cfg.BA_M,cfg.RND_SEED)
  elif cfg.GRAPH_MODEL == 'WS':
    return nx.random_graphs.connected_watts_strogatz_graph(Nodes, cfg.WS_K, cfg.WS_P, tries=200, seed=cfg.RND_SEED)

def save_graph(graph, path):
  nx.write_yaml(graph, path)

def load_graph(path):
  return nx.read_yaml(path)


if __name__ == '__main__':

    graph1 = build_graph(32, 'WS', 0, 4, 0.75)

    options = {
    'node_color': 'Yellow',
    'node_size': 500,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    }

    nx.draw_networkx(graph1, arrows=True, **options)
    plt.show()