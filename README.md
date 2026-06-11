# RandWireNN: Randomly Wired Neural Networks (Modernized 2025)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Modern PyTorch implementation of Randomly Wired Neural Networks with **state-of-the-art training techniques**, **comprehensive experiment tracking**, and **Bayesian uncertainty quantification**.

**Original Paper:** [Exploring Randomly Wired Neural Networks for Image Recognition (ICCV 2019)](https://arxiv.org/pdf/1904.01569v2)

> 🧪 **New — [The Wiring Lab](WIRING_LAB.md)**: a 45-run mini-replication of the paper's
> core claim, runnable on CPU in ~15 minutes. Spoiler: among random graphs the wiring
> doesn't matter, but average path length predicts accuracy at r = −0.84.
>
> 🔬 **New — [Short Paths Study](SHORT_PATHS_STUDY.md)**: the controlled follow-up.
> 60 runs, pre-registered hypotheses, mediation analysis. Path length causes the
> effect; init-time gradient imbalance mediates ~28% of it; and trained networks
> *downweight* the very skip edges that make them work.
>
> ⚖️ **New — [Three Hypotheses, Three Verdicts](FOLLOWUP_HYPOTHESES.md)**: the
> scaffolding hypothesis is **falsified** by ablation (networks compute *through*
> their skips; weights are uninformative about importance — betweenness predicts
> damage 2.4× better); a zero-training gradient proxy picks the best of 24 wirings
> with zero regret; and the chain-vs-random gap explodes exponentially with depth
> (+3 → +11 → +58 points at N = 6 → 12 → 24).

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [What's New in 2025](#-whats-new-in-2025)
- [Architecture & Graph Models](#-architecture--graph-models)
- [Experiment Tracking](#-experiment-tracking)
- [Bayesian Uncertainty Quantification](#-bayesian-uncertainty-quantification)
- [Configuration](#-configuration)
- [Results](#-results)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🚀 Modern Training Infrastructure
- **5 State-of-the-Art Optimizers**: SGD+Nesterov, Adam, AdamW, Lion, RMSprop
- **7 Advanced Schedulers**: Cosine, OneCycleLR, ReduceLROnPlateau, and more
- **Mixed Precision Training**: 2-3x speedup with FP16 (AMP)
- **Automatic Best Model Saving**: Track and save best models automatically

### 🔬 Experiment Management
- **TensorBoard Integration**: Real-time training visualization
- **Weights & Biases Support**: Cloud-based experiment tracking
- **9 Graph Topology Models**: ER, BA, WS, NWS, PC, RR, Complete, Path, Cycle
- **Comprehensive Logging**: JSON logs, hyperparameters, model artifacts

### 🎲 Bayesian Deep Learning
- **Uncertainty Quantification**: MC Dropout & Variational Inference
- **Calibration Analysis**: ECE, reliability diagrams, confidence scores
- **Two Bayesian Methods**: Simple MC Dropout or full variational inference
- **Automatic Uncertainty Estimation**: Post-training uncertainty analysis

### 📊 Supported Datasets
- MNIST
- CIFAR-10 / CIFAR-100
- ImageNet (ILSVRC2012)
- Custom datasets

---

## ⚡ Quick Start

```bash
# 1. Clone and install
git clone https://github.com/hebo1221/RandWireNN.git
cd RandWireNN
pip install -e .

# 2. Basic training
python run_RandWireNN.py

# 3. Modern training with AdamW + Mixed Precision + TensorBoard
python -c "exec(open('configs/example_tensorboard.py').read())" && python run_RandWireNN.py

# 4. View results
tensorboard --logdir=./output/experiments
```

**That's it!** The code will automatically download datasets and start training.

---

## 📦 Installation

### Basic Installation

```bash
pip install -e .
```

This installs all core dependencies: PyTorch, NetworkX, NumPy, etc.

### Optional Dependencies

```bash
# For experiment tracking
pip install -e ".[tracking]"  # TensorBoard + W&B + visualization

# For advanced optimizers
pip install -e ".[optimizers]"  # Lion optimizer

# For visualization
pip install -e ".[viz]"  # Visdom

# Install everything
pip install -e ".[all]"
```

### From Requirements File

```bash
pip install -r requirements.txt
```

### Requirements
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **CUDA**: Recommended for GPU acceleration

---

## 💡 Usage Examples

### Example 1: Modern Optimizer + Scheduler

```bash
# AdamW + Cosine Warmup (Recommended)
python -c "exec(open('configs/example_adamw_cosine.py').read())" && python run_RandWireNN.py

# SGD + OneCycleLR for super-convergence
python -c "exec(open('configs/example_sgd_onecycle.py').read())" && python run_RandWireNN.py
```

### Example 2: Different Graph Topologies

```bash
# Try different graph models
python -c "exec(open('configs/example_graph_models.py').read())" && python run_RandWireNN.py
```

Available models: `ER`, `BA`, `WS`, `NWS`, `PC`, `RR`, `COMPLETE`, `PATH`, `CYCLE`

### Example 3: Experiment Tracking

```bash
# With TensorBoard
pip install tensorboard
python -c "exec(open('configs/example_tensorboard.py').read())" && python run_RandWireNN.py
tensorboard --logdir=./output/experiments

# With Weights & Biases
pip install wandb
# Set USE_WANDB=True in config
```

### Example 4: Bayesian Uncertainty Estimation

```bash
# MC Dropout (simple & fast)
python -c "exec(open('configs/example_bayesian_mc_dropout.py').read())" && python run_RandWireNN.py

# Variational Inference (more principled)
python -c "exec(open('configs/example_bayesian_variational.py').read())" && python run_RandWireNN.py
```

---

## 🆕 What's New in 2025

This repository has been extensively modernized from the 2019 implementation:

### Phase 1: Core Architecture (Jan 2025) ✨

| Feature | Details |
|---------|---------|
| **Modern Optimizers** | AdamW, Lion, SGD+Nesterov, Adam, RMSprop |
| **Advanced Schedulers** | OneCycleLR, CosineWarmup, ReduceLROnPlateau |
| **Mixed Precision** | PyTorch AMP for 2-3x speedup |
| **Code Quality** | Type hints, f-strings, logging, bug fixes |

### Phase 2: Experiment Infrastructure (Jan 2025) 🔬

| Feature | Details |
|---------|---------|
| **Experiment Tracking** | TensorBoard, W&B, JSON logging |
| **Graph Models** | 3 → 9 models (added NWS, PC, RR, Complete, Path, Cycle) |
| **Checkpointing** | Automatic best model saving, periodic checkpoints |
| **Visualization** | Graph structure plots, training curves |

### Phase 3: Bayesian Deep Learning (Jan 2025) 🎲

| Feature | Details |
|---------|---------|
| **Uncertainty Methods** | MC Dropout, Variational Inference |
| **Bayesian Layers** | BayesianConv2d, BayesianLinear, MCDropout |
| **Calibration** | ECE, reliability diagrams, confidence analysis |
| **Research-Based** | 3 papers: 2020 Bayesian RWNN, Gal 2016, Blundell 2015 |

---

## 🕸️ Architecture & Graph Models

RandWireNN uses **random graphs as neural network architectures**. Each node is a computation unit, and edges define connections.

### Available Graph Models

| Model | Type | Description | Use Case |
|-------|------|-------------|----------|
| **ER** | Random | Erdos-Renyi | Baseline, classical random |
| **BA** | Scale-Free | Barabasi-Albert | Networks with hubs |
| **WS** | Small-World | Watts-Strogatz | High clustering + short paths |
| **NWS** | Small-World | Newman-Watts-Strogatz | Edge addition (no rewiring) |
| **PC** | Scale-Free | Powerlaw Cluster | Scale-free + triangles |
| **RR** | Regular | Random Regular | Uniform degree |
| **COMPLETE** | Dense | Fully connected | Maximum connectivity |
| **PATH** | Sparse | Linear chain | Sequential processing |
| **CYCLE** | Ring | Circular | Ring topology |

### Configuration

```python
# In RandWireNN_config.py
__C.GRAPH_MODEL = "WS"  # Choose your model

# Model-specific parameters
__C.WS_K = 4          # Watts-Strogatz neighbors
__C.WS_P = 0.75       # Rewiring probability
__C.BA_M = 5          # Barabasi-Albert edges
__C.ER_P = 0.2        # Erdos-Renyi edge probability
```

---

## 📊 Experiment Tracking

### TensorBoard (Recommended)

```bash
# Enable in config
__C.USE_TENSORBOARD = True

# Run training
python run_RandWireNN.py

# View results
tensorboard --logdir=./output/experiments
```

**Logged Metrics:**
- Training & validation loss
- Top-1 & Top-5 accuracy
- Learning rate curves
- Graph structure visualizations

### Weights & Biases

```bash
# Install
pip install wandb

# Enable in config
__C.USE_WANDB = True
__C.WANDB_PROJECT = "my_project"

# Login and run
wandb login
python run_RandWireNN.py
```

### Experiment Organization

```
./output/experiments/
└── rwnn_20250105_143022/
    ├── tensorboard/          # TB logs
    ├── metrics.jsonl         # JSON metrics
    ├── config.json           # Full config
    ├── best_model.pth        # Best checkpoint
    ├── checkpoint_epoch_*.pth
    └── uncertainty_analysis.png  # If Bayesian
```

---

## 🎲 Bayesian Uncertainty Quantification

Estimate prediction confidence and model uncertainty for **high-stakes applications**.

### Two Methods

#### 1. MC Dropout (Recommended)
**Pros:** Simple, fast, no architecture changes
**Use:** General purpose, quick experiments

```python
__C.USE_BAYESIAN = True
__C.BAYESIAN_METHOD = "mc_dropout"
__C.MC_DROPOUT_P = 0.1
__C.MC_SAMPLES = 30
```

#### 2. Variational Inference
**Pros:** Principled, better uncertainty
**Use:** Research, critical applications

```python
__C.USE_BAYESIAN = True
__C.BAYESIAN_METHOD = "variational"
__C.VARIATIONAL_PRIOR_STD = 1.0
__C.KL_WEIGHT = 1e-5
```

### Uncertainty Outputs

**Automatic Analysis:**
- ✅ Epistemic uncertainty (model uncertainty)
- ✅ Total predictive uncertainty
- ✅ Expected Calibration Error (ECE)
- ✅ Reliability diagrams
- ✅ Confidence vs accuracy plots

**Use Cases:**
- Medical AI: Only act on high-confidence predictions
- Active Learning: Select uncertain samples for labeling
- Anomaly Detection: Flag high-uncertainty inputs
- Model Calibration: Assess prediction trustworthiness

### Example Output

```
Uncertainty Analysis:
  Accuracy: 0.9245
  ECE: 0.0342 (well-calibrated!)
  Mean Epistemic Uncertainty: 0.0234
  Correct samples confidence: 0.9521
  Incorrect samples confidence: 0.6847
```

**Interpretation:** Model correctly assigns higher confidence to correct predictions.

---

## ⚙️ Configuration

### Quick Config via Examples

Use pre-configured files in `configs/`:

```bash
configs/
├── example_adamw_cosine.py          # Modern optimizer
├── example_sgd_onecycle.py          # Super-convergence
├── example_lion.py                  # Lion optimizer
├── example_tensorboard.py           # Experiment tracking
├── example_graph_models.py          # Graph topologies
├── example_bayesian_mc_dropout.py   # Uncertainty (MC)
└── example_bayesian_variational.py  # Uncertainty (VI)
```

### Manual Configuration

Edit `RandWireNN_config.py`:

```python
# Optimizer
__C.OPTIMIZER = "adamw"
__C.LEARNING_RATE = 0.001
__C.WEIGHT_DECAY = 0.05

# Scheduler
__C.SCHEDULER = "cosine_warmup"
__C.T_0 = 10

# Mixed Precision
__C.USE_AMP = True

# Experiment Tracking
__C.USE_TENSORBOARD = True
__C.EXPERIMENT_NAME = "my_experiment"

# Bayesian (optional)
__C.USE_BAYESIAN = True
__C.BAYESIAN_METHOD = "mc_dropout"
```

### Dataset Selection

In `run_RandWireNN.py`:

```python
def get_configuration():
    from RandWireNN_config import cfg as network_cfg

    # Choose dataset config
    from utils.configs.cifar10_config import cfg as dataset_cfg
    # from utils.configs.mnist_config import cfg as dataset_cfg
    # from utils.configs.ImageNet_config import cfg as dataset_cfg

    return merge_configs([network_cfg, dataset_cfg])
```

---

## 📈 Results

### Original Results (2019)

**Small Regime** (C=78, WS(4,0.75))

| Dataset | Top-1 Acc | Top-5 Acc | Epochs |
|---------|-----------|-----------|--------|
| MNIST | 99.60% | 100% | 100 |
| CIFAR-10 | 91.71% | 99.75% | 250 |
| CIFAR-100 | 72.49% | 92.15% | 250 |
| ImageNet | 56.70% | 78.95% | 81 |

**Regular Regime** (C=109, WS(4,0.75))
- ImageNet: Work in progress

### Expected Improvements with Modern Techniques

Using AdamW + OneCycleLR + Mixed Precision:
- Faster convergence (fewer epochs needed)
- Slightly higher accuracy (1-2% on CIFAR)
- Better calibration with Bayesian methods

*Contributions with benchmarks welcome!*

---

## 📚 Citation

If you use this code in your research, please cite:

### Original Paper
```bibtex
@inproceedings{xie2019exploring,
  title={Exploring Randomly Wired Neural Networks for Image Recognition},
  author={Xie, Saining and Kirillov, Alexander and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={ICCV},
  year={2019}
}
```

### Bayesian Extensions (if used)
```bibtex
@inproceedings{lee2020bayesian,
  title={Bayesian Randomly Wired Neural Network with Variational Inference for Image Recognition},
  author={Lee, Junyoung and others},
  booktitle={ICPR},
  year={2020}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- [ ] Benchmark modern optimizers on ImageNet
- [ ] Add NAS (Neural Architecture Search) integration
- [ ] Implement LLM-guided graph generation
- [ ] Add more Bayesian layers (e.g., Bayesian attention)
- [ ] Support for other tasks (detection, segmentation)
- [ ] Hyperparameter optimization (Optuna, Ray Tune)

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🐛 Troubleshooting

### Common Issues

**Q: CUDA out of memory**
```python
# Reduce batch size or enable AMP
__C.USE_AMP = True
cfg.BATCH_SIZE = 64  # in dataset config
```

**Q: NetworkX compatibility error**
```bash
# Ensure NetworkX 3.0+
pip install --upgrade networkx
```

**Q: Graph generation fails**
```python
# Some graph models need specific parameters
# For RR, ensure d*n is even
__C.RR_D = 4  # Try even numbers
```

**Q: TensorBoard not showing**
```bash
# Check experiment directory
tensorboard --logdir=./output/experiments
# Open http://localhost:6006
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original RandWireNN paper authors (Facebook AI Research)
- Bayesian deep learning research community
- PyTorch team for excellent framework
- All contributors to this modernization

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/hebo1221/RandWireNN/issues)
- **Pull Requests**: [GitHub PRs](https://github.com/hebo1221/RandWireNN/pulls)

---

**Star ⭐ this repo if you find it useful!**

Made with ❤️ for the deep learning community.
