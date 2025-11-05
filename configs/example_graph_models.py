"""
Example configurations for different graph models.
Demonstrates the various graph topologies available for RandWireNN.
"""
from RandWireNN_config import cfg

# Try different graph models by uncommenting one of the following:

# 1. Barabasi-Albert (Scale-free, Preferential Attachment)
# Good for: Networks with hub nodes
cfg.GRAPH_MODEL = "BA"
cfg.BA_M = 5

# 2. Newman-Watts-Strogatz (Small-world, Edge Addition)
# Good for: High clustering without rewiring
# cfg.GRAPH_MODEL = "NWS"
# cfg.NWS_K = 4
# cfg.NWS_P = 0.1

# 3. Powerlaw Cluster (Scale-free with Clustering)
# Good for: Networks with both hubs and triangles
# cfg.GRAPH_MODEL = "PC"
# cfg.PC_M = 3
# cfg.PC_P = 0.1

# 4. Random Regular (All nodes same degree)
# Good for: Uniform connectivity
# cfg.GRAPH_MODEL = "RR"
# cfg.RR_D = 4

# 5. Erdos-Renyi (Classical random graph)
# Good for: Baseline comparison
# cfg.GRAPH_MODEL = "ER"
# cfg.ER_P = 0.2

# Enable visualization
cfg.USE_TENSORBOARD = True
cfg.EXPERIMENT_NAME = f"rwnn_{cfg.GRAPH_MODEL.lower()}_demo"

print(f"Graph Model: {cfg.GRAPH_MODEL}")
print(f"Experiment: {cfg.EXPERIMENT_NAME}")
print("\nAvailable graph models:")
print("  ER  - Erdos-Renyi (random)")
print("  BA  - Barabasi-Albert (scale-free)")
print("  WS  - Watts-Strogatz (small-world, rewiring)")
print("  NWS - Newman-Watts-Strogatz (small-world, edge addition)")
print("  PC  - Powerlaw Cluster (scale-free + clustering)")
print("  RR  - Random Regular (uniform degree)")
