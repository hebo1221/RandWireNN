# RandWireNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-randomly-wired-neural-networks-for/image-classification-imagenet-image-reco)](https://paperswithcode.com/sota/image-classification-imagenet-image-reco?p=exploring-randomly-wired-neural-networks-for)

![Datasets](https://img.shields.io/badge/Dataset-MNIST-lightgray.svg) ![Datasets](https://img.shields.io/badge/Dataset-CIFAR--10,100-green.svg) ![Datasets](https://img.shields.io/badge/Dataset-ImageNet--12-yellow.svg)

## Results

In small regime, C=78, WS(4,0.75)

| datasets    | top-1 acc. | top-5 acc. | epoch |
| ----------- | ---------- | ---------- | ----- |
| MNIST       | 99.60      | 100.       | 100   |
| CIFAR-10    | 91.71      | 99.75      | 250   |
| CIFAR-100   | 72.49      | 92.15      | 250   | 
| ImageNet-12 | 56.70      | 78.95      | 81    |

In regular regime, C=109, WS(4,0.75)

| datasets    | top-1 acc.       | top-5 acc. | epoch |
| ----------- | ---------------- | ---------- | ----- |
| ImageNet-12 | work in progress |            |       |

*- Because my computer does not have powerful computing power, it will take some time to update.*

## Running the example

### Setup
Requirements:
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/get-started/locally/)

**Recommended:** Python 3.10+ and PyTorch 2.0+

Clone the repository and install:

```bash
git clone https://github.com/hebo1221/RandWireNN.git
cd RandWireNN

# Install as a package (recommended)
pip install -e .

# Or install requirements only
pip install -r requirements.txt
```

### Running the demo
Basic usage:
```bash
python run_RandWireNN.py
```

**Using example configurations with modern optimizers:**
```bash
# AdamW + Cosine Warmup + Mixed Precision (Recommended)
python -c "exec(open('configs/example_adamw_cosine.py').read())" && python run_RandWireNN.py

# SGD + OneCycleLR (Super-convergence)
python -c "exec(open('configs/example_sgd_onecycle.py').read())" && python run_RandWireNN.py

# Lion optimizer (Install: pip install lion-pytorch)
python -c "exec(open('configs/example_lion.py').read())" && python run_RandWireNN.py
```

**Configuration options:**
- Dataset selection: Edit `run_RandWireNN.py`, `get_configuration()`. MNIST, CIFAR-10/100, ImageNet available
- Datasets auto-download if not found
- Custom dataset directory: Set `__C.DATASET_DIR` in `RandWireNN_config.py`
- Loss visualization: Set `__C.VISDOM = True` in config (requires visdom: `pip install visdom`)
- All hyperparameters: See `*_config.py` files

### Advanced Features (NEW!)

**Modern Optimizers:**
- `sgd`: Classic SGD with optional Nesterov momentum
- `adam`: Standard Adam optimizer
- `adamw`: AdamW with decoupled weight decay (recommended)
- `lion`: Lion optimizer (requires `pip install lion-pytorch`)
- `rmsprop`: RMSprop optimizer

**Learning Rate Schedulers:**
- `cosine`: Cosine annealing
- `cosine_warmup`: Cosine with warm restarts
- `onecycle`: OneCycleLR for super-convergence
- `step`: Step decay
- `multistep`: Multi-step decay
- `exponential`: Exponential decay
- `reduce_on_plateau`: Reduce on plateau

**Mixed Precision Training:**
Enable automatic mixed precision (FP16) for 2-3x speedup:
```python
__C.USE_AMP = True
```

Configure in `RandWireNN_config.py`:
```python
__C.OPTIMIZER = "adamw"  # Choose your optimizer
__C.SCHEDULER = "cosine_warmup"  # Choose your scheduler
__C.USE_AMP = True  # Enable mixed precision
__C.LEARNING_RATE = 0.001  # Adjust learning rate
```


## Recent Updates (2025)

This repository has been extensively modernized with state-of-the-art improvements:

### Phase 1: Core Architecture Improvements ✨

- **Modern Optimizers:**
  - AdamW with decoupled weight decay
  - Lion optimizer support (state-of-the-art)
  - SGD with Nesterov momentum
  - Comprehensive optimizer factory pattern

- **Advanced Learning Rate Schedulers:**
  - OneCycleLR for super-convergence
  - Cosine Annealing with Warm Restarts
  - ReduceLROnPlateau
  - Multiple scheduler options for different use cases

- **Mixed Precision Training (AMP):**
  - FP16 automatic mixed precision support
  - 2-3x training speedup on modern GPUs
  - Reduced memory usage

- **Example Configurations:**
  - Pre-configured setups for different use cases
  - Easy-to-use configuration files in `/configs`

### Phase 2: Experiment Infrastructure 🔬

- **Experiment Tracking:**
  - TensorBoard integration for real-time visualization
  - Weights & Biases support for cloud-based tracking
  - JSON logging for offline analysis
  - Automatic hyperparameter logging
  - Graph structure visualization

- **Model Checkpointing:**
  - Automatic best model saving based on validation metrics
  - Periodic checkpoint saving
  - Resume training from checkpoints

- **Extended Graph Models:**
  - Newman-Watts-Strogatz (NWS): Small-world with edge addition
  - Powerlaw Cluster (PC): Scale-free with high clustering
  - Random Regular (RR): Uniform degree distribution
  - Plus ER, BA, WS from original implementation
  - Total of 9 graph generation models

**Using Experiment Tracking:**
```bash
# With TensorBoard (recommended)
pip install tensorboard
python -c "exec(open('configs/example_tensorboard.py').read())" && python run_RandWireNN.py

# View results
tensorboard --logdir=./output/experiments

# With Weights & Biases (optional)
pip install wandb
# Set USE_WANDB=True in config
```

**Trying Different Graph Models:**
```python
# In RandWireNN_config.py or use example_graph_models.py
__C.GRAPH_MODEL = "BA"  # Barabasi-Albert (scale-free)
__C.GRAPH_MODEL = "NWS"  # Newman-Watts-Strogatz (small-world)
__C.GRAPH_MODEL = "PC"  # Powerlaw Cluster (scale-free + clustering)
__C.GRAPH_MODEL = "RR"  # Random Regular (uniform degree)
```

### Phase 3: Bayesian RWNN & Uncertainty Quantification 🎲

- **Uncertainty Estimation:**
  - MC Dropout: Simple and efficient (Gal & Ghahramani, 2016)
  - Variational Inference: Full Bayesian treatment (Blundell et al., 2015)
  - Epistemic uncertainty (model uncertainty)
  - Aleatoric uncertainty (data uncertainty)
  - Total predictive uncertainty

- **Bayesian Layers:**
  - BayesianConv2d: Convolutional layers with weight distributions
  - BayesianLinear: Fully connected Bayesian layers
  - MCDropout: Monte Carlo Dropout for uncertainty
  - Automatic conversion of standard models to Bayesian

- **Uncertainty Analysis:**
  - Expected Calibration Error (ECE)
  - Reliability diagrams
  - Confidence vs accuracy analysis
  - Uncertainty distribution visualization
  - Per-sample confidence scores

**Using Bayesian RWNN:**

```bash
# Method 1: MC Dropout (Recommended - Simple & Fast)
python -c "exec(open('configs/example_bayesian_mc_dropout.py').read())" && python run_RandWireNN.py

# Method 2: Variational Inference (More Principled)
python -c "exec(open('configs/example_bayesian_variational.py').read())" && python run_RandWireNN.py
```

**Configuration:**
```python
# Enable Bayesian mode
__C.USE_BAYESIAN = True
__C.BAYESIAN_METHOD = "mc_dropout"  # or "variational"
__C.MC_DROPOUT_P = 0.1  # Dropout probability
__C.MC_SAMPLES = 30  # Number of MC samples
__C.ESTIMATE_UNCERTAINTY = True  # Compute uncertainty metrics

# For variational inference
__C.VARIATIONAL_PRIOR_STD = 1.0
__C.KL_WEIGHT = 1e-5
```

**Uncertainty Output:**
- Reliability diagrams showing calibration quality
- ECE scores for model confidence assessment
- Per-sample uncertainty estimates
- Separation of correct vs incorrect predictions by uncertainty

Based on research:
- "Bayesian Randomly Wired Neural Network with Variational Inference" (2020)
- "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
- "Weight Uncertainty in Neural Networks" (Blundell et al., 2015)

### Foundation Updates

- **Updated Dependencies:**
  - Python 3.10+ support (with type hints)
  - PyTorch 2.0+ compatibility
  - NetworkX 3.0+ (with updated graph APIs)
  - All major dependencies updated to latest stable versions

- **Code Quality Improvements:**
  - Migrated from `.format()` to f-strings
  - Added type hints for better IDE support
  - Replaced print statements with proper logging
  - Fixed critical indentation bug in Node_OP.forward()
  - Modern `.gitignore` with comprehensive patterns

- **Package Structure:**
  - Added `pyproject.toml` for modern Python packaging
  - Can now be installed with `pip install -e .`
  - Support for development dependencies and optional extras

- **Developer Experience:**
  - Better logging for debugging
  - Improved error messages
  - Type hints for IDE autocomplete
  - Comprehensive documentation

### Reference

All details regarding the Randomly Wired Neural Networks can be found in the original research paper: [https://arxiv.org/pdf/1904.01569v2](https://arxiv.org/pdf/1904.01569v2).


## License

All materials in this repository are released under the Apache License 2.0.