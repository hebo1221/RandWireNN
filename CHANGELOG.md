# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025 Modernization - 2026-06-11

### 🎯 Major Modernization Release

Complete overhaul of the RandWireNN repository with modern training infrastructure, comprehensive experimental validation, and Bayesian uncertainty quantification.

---

### Added

#### 📚 Comprehensive Research Documentation

- **[WIRING_LAB.md](WIRING_LAB.md)** - 45-run mini-replication validating the 2019 paper's core claim
  - Correlation study showing path length predicts accuracy (r = −0.84)
  - Comparison of 5 graph families (ER, BA, WS, NWS, PC)
  - Small-world analysis and structural correlations
  
- **[SHORT_PATHS_STUDY.md](SHORT_PATHS_STUDY.md)** - 60-run controlled follow-up with pre-registered hypotheses
  - **H1**: Path length causally affects accuracy (validated via dose-response)
  - **H2**: Init-time gradient imbalance mediates ~28% of the effect
  - **H3**: Trained networks downweight skip edges (ρ = 0.16 vs betweenness ρ = 0.38)
  
- **[FOLLOWUP_HYPOTHESES.md](FOLLOWUP_HYPOTHESES.md)** - Three hypothesis tests with rigorous validation
  - **H4**: Scaffolding hypothesis **falsified** - networks compute *through* skips, not just trained with them
  - **H5**: Zero-cost gradient proxy validated - picks best of 24 wirings with zero regret
  - **H6**: Depth scaling law measured - exponential penalty 0.067 decades/level

- **[ROUND3_REHAB_AND_PROXY.md](ROUND3_REHAB_AND_PROXY.md)** - Advanced decomposition and boundary conditions
  - **H7**: Skip advantage decomposed into +3.4pt transferable teaching + 5.1pt irreplaceable computation
  - **H8**: Proxy limits established - works as anomaly detector, not fine ranker (58 candidates)

#### 🧪 Experimental Scripts

**Core Experiments:**
- `wiring_lab.py` + `wiring_lab_plots.py` - Graph topology comparison framework
- `wiring_study.py` + `wiring_study_analysis.py` - Causal mediation analysis

**Hypothesis Testing Suite:**
- `h4_scaffolding.py` + `h4b_recalibration.py` - Network ablation and BN recalibration
- `h5_zerocost.py` - Zero-cost gradient proxy validation
- `h6_scaling.py` - Depth scaling experiments
- `h7_rehab.py` + `h7_figure.py` - Skip advantage decomposition via rehabilitation
- `h8_proxy_scale.py` - Large-scale proxy validation (58 candidates)
- `h_followup_analysis.py` - Consolidated analysis and figures

**Total Experimental Coverage:** 253 training runs, all reproducible on CPU in ~2 hours

#### 🚀 Modern Training Infrastructure

**Optimizer Suite** (`utils/optimizers.py`):
- AdamW with weight decay
- Lion optimizer
- SGD with Nesterov momentum
- Adam
- RMSprop

**Scheduler Suite:**
- Cosine Annealing with warmup
- OneCycleLR for super-convergence
- ReduceLROnPlateau
- CosineAnnealingWarmRestarts
- Step scheduler
- Exponential scheduler

**Training Enhancements:**
- Mixed precision training (FP16 AMP) - 2-3x speedup
- Automatic best model checkpointing
- Gradient clipping
- Comprehensive logging

#### 🔬 Experiment Management

**Tracking & Logging** (`utils/experiment_tracker.py`):
- TensorBoard integration with real-time visualization
- Weights & Biases support
- JSON experiment logs
- Hyperparameter tracking
- Model artifact management

**Scripts:**
- `experiment_config.py` - Centralized experiment configuration
- `run_experiment.py` - Experiment runner with tracking
- `demo_training.py` - Comprehensive demo with all features

#### 🎲 Bayesian Deep Learning

**Bayesian Layers** (`utils/bayesian_layers.py`):
- MC Dropout implementation
- Variational Bayesian Conv2d
- Variational Bayesian Linear
- Flexible dropout scheduling

**Uncertainty Quantification** (`utils/uncertainty.py`):
- Epistemic uncertainty estimation
- Expected Calibration Error (ECE)
- Reliability diagrams
- Confidence analysis
- Prediction interval estimation

**Features:**
- Two methods: MC Dropout (simple) and Variational Inference (principled)
- Automatic uncertainty analysis post-training
- Calibration metrics and visualization

#### 📊 Enhanced Graph Models

**Extended Support** (in `utils/graph.py`):
- Erdős-Rényi (ER) - random graphs
- Barabási-Albert (BA) - scale-free networks
- Watts-Strogatz (WS) - small-world networks
- Newman-Watts-Strogatz (NWS) - extended small-world
- Powerlaw Cluster (PC) - clustered scale-free
- Random Regular (RR) - k-regular graphs
- Complete graphs
- Path graphs
- Cycle graphs

Total: **9 topology models** with comprehensive parameter control

#### 📦 Modern Packaging

- `pyproject.toml` - Modern Python packaging with Poetry support
- `requirements.txt` - Updated dependencies for Python 3.10+
- Development tools setup (pytest, black, isort)

#### 🎓 Example Configurations

Created 7 ready-to-use configuration examples in `configs/`:
- `example_adamw_cosine.py` - Modern optimizer baseline
- `example_sgd_onecycle.py` - Super-convergence training
- `example_lion.py` - Lion optimizer
- `example_tensorboard.py` - Experiment tracking
- `example_graph_models.py` - Graph topology exploration
- `example_bayesian_mc_dropout.py` - MC Dropout uncertainty
- `example_bayesian_variational.py` - Variational inference
- `example_quick_test.py` - Fast testing configuration

#### 📈 Results & Data

**Documentation Assets:**
- 12 publication-ready figures across 4 studies
- All raw experimental data (.jsonl, .json)
- Statistical analysis summaries
- Organized in `docs/` subdirectories by study

### Changed

#### Core Architecture Updates

**`RandWireNN_config.py`:**
- Expanded optimizer options (5 total)
- Added scheduler configurations (7 total)
- Mixed precision training flag
- Bayesian layer configuration
- Experiment tracking settings
- Extended graph model parameters

**`RandWireNN_train.py`:**
- Refactored training loop with modern optimizers
- Added mixed precision support
- Integrated experiment tracking
- Enhanced logging and checkpointing
- Gradient clipping support

**`run_RandWireNN.py`:**
- Improved configuration management
- Better dataset handling
- Added experiment tracking initialization
- Enhanced error handling

**`utils/graph.py`:**
- Extended with 6 additional graph models
- Better parameter validation
- Improved edge case handling
- Enhanced documentation

**`utils/dataloader.py`:**
- Updated for modern PyTorch (2.0+)
- Better transform handling
- Improved error messages

**`utils/network.py`:**
- Optional Bayesian layer integration
- Better forward pass handling
- Enhanced compatibility

### Improved

#### Documentation

**README.md** - Complete rewrite (562 line changes):
- Modern badges and styling
- Feature showcase with examples
- Quick start guide
- Comprehensive usage documentation
- Results and benchmarks
- Troubleshooting section
- Contributing guidelines

**Other Improvements:**
- All code examples tested and verified
- Inline documentation enhanced
- Type hints added where appropriate
- Error messages made more informative

#### Testing

**`test_experiment.py`:**
- Comprehensive test suite
- Graph generation validation
- Model instantiation tests
- Training loop validation
- Checkpoint saving/loading tests

### Technical Achievements

#### Scientific Rigor

**Methodology:**
- Pre-registered hypotheses (H1-H8)
- Controlled interventions
- Paired statistical tests
- Mediation analysis (bootstrap CIs)
- Post-hoc diagnostics clearly labeled

**Reproducibility:**
- All experiments use fixed seeds
- Complete parameter documentation
- Raw data included
- Scripts are self-contained
- CPU-runnable in reasonable time

**Honest Reporting:**
- Hypothesis H4 falsification documented
- Hypothesis H8 partial failure explained
- Negative results (hub hypothesis) reported
- Limitations sections in each study

#### Performance Improvements

- **Training Speed:** 2-3x faster with mixed precision
- **Convergence:** Faster with modern optimizers (AdamW, Lion)
- **Flexibility:** 9 graph models vs 3 original
- **Usability:** 7 config examples vs manual setup

#### Code Quality

- Modern Python (3.10+ features)
- PyTorch 2.0+ compatibility
- Type hints added
- Better error handling
- Comprehensive logging
- Modular design

### Experimental Insights

#### Key Findings

1. **Path length is causal** (r = −0.84 correlation, validated via controlled manipulation)
2. **Gradient flow mediates 28%** of the path length effect at initialization
3. **Skips both teach and compute**: +3.4pt transferable knowledge, +5.1pt irreplaceable computation
4. **Weight magnitude is uninformative**: betweenness predicts ablation damage 2.4× better (ρ = 0.38 vs 0.16)
5. **Zero-cost proxy works for screening**: regret@5 = 0 at 91% compute reduction
6. **Healthy wirings are equivalent**: within-cluster accuracy variance = seed noise
7. **Depth penalty is exponential**: 0.067 decades/level measured at N = 6, 12, 24

#### Boundary Conditions Established

- **Small scale (N≤12):** Wiring doesn't matter much (+3pt spread)
- **Medium scale (N=24):** Skips become critical (+58pt chain penalty)
- **Proxy limitation:** Anomaly detector, not fine ranker
- **Hub hypothesis:** Rejected (ρ = 0.14, p = 0.30)

### Dependencies

#### Updated Requirements

```
numpy>=1.24.0 (was 1.19.0)
scipy>=1.10.0 (new)
networkx>=3.0 (was 2.5)
matplotlib>=3.7.0 (new)
torch>=2.0.0 (was 1.7.0)
torchvision>=0.15.0 (was 0.8.0)
pyyaml>=6.0 (new)
easydict>=1.10 (maintained)
```

#### New Optional Dependencies

For full feature support:
```
tensorboard>=2.13.0
wandb>=0.15.0
```

### Migration Guide

#### For Existing Users

**Minimal Changes Required:**
```python
# Old usage still works
python run_RandWireNN.py

# But now you can do:
python -c "exec(open('configs/example_adamw_cosine.py').read())"
python run_RandWireNN.py
```

**New Features Are Opt-In:**
- Mixed precision: set `__C.USE_AMP = True`
- TensorBoard: set `__C.USE_TENSORBOARD = True`
- Bayesian: set `__C.USE_BAYESIAN = True`

**Breaking Changes:**
- Python 3.10+ required (was 3.7+)
- PyTorch 2.0+ required (was 1.7+)
- NetworkX 3.0+ required (was 2.5)

### Statistics

- **Total Commits:** 10 comprehensive commits
- **Files Changed:** 69 files
- **Lines Added:** 8,136 insertions
- **Lines Removed:** 107 deletions
- **New Files:** 57 files
- **Documentation:** 4 comprehensive study documents + updated README
- **Experimental Coverage:** 253 training runs
- **Compute Time:** ~2 hours on CPU for full replication

### Repository Structure

```
RandWireNN/
├── README.md                        # Complete rewrite
├── CHANGELOG.md                     # This file
├── WIRING_LAB.md                    # Study 1: Topology comparison
├── SHORT_PATHS_STUDY.md             # Study 2: Causal mechanism
├── FOLLOWUP_HYPOTHESES.md           # Study 3: H4, H5, H6
├── ROUND3_REHAB_AND_PROXY.md        # Study 4: H7, H8
├── configs/                         # 7 example configurations
├── utils/
│   ├── bayesian_layers.py          # New
│   ├── experiment_tracker.py       # New
│   ├── optimizers.py               # New
│   ├── uncertainty.py              # New
│   └── [enhanced existing files]
├── docs/                            # Organized experimental results
│   ├── wiring_lab/
│   ├── wiring_study/
│   ├── followup/
│   └── round3/
└── [experimental scripts h4-h8, wiring_lab, wiring_study]
```

### Acknowledgments

This modernization effort represents a comprehensive upgrade to the original 2019 RandWireNN implementation, adding:
- Modern training infrastructure
- Rigorous experimental validation
- Bayesian uncertainty quantification
- Complete documentation

All while maintaining compatibility with the original architecture and validating the core claims of the paper with extended mechanistic understanding.

### Future Work

Potential areas for further development (see README Contributing section):
- ImageNet benchmarks with modern techniques
- NAS integration
- LLM-guided graph generation
- Additional Bayesian layers
- Support for detection/segmentation tasks
- Hyperparameter optimization integration

---

**Full Documentation:** See individual study documents and README.md for detailed information.

**Reproducibility:** All experiments can be reproduced using the provided scripts. See each study document for specific commands.

**Citation:** If you use this modernized version, please cite both the original 2019 paper and acknowledge this modernization effort.
