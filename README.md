# DMC-GGAD: Density-Aware and Micro-Environment Camouflage for Semi-Supervised Graph Anomaly Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

This repository contains the official implementation of **DMC-GGAD**, a semi-supervised graph anomaly detection framework that addresses the *topological uniformity assumption* in generative methods by synthesizing topology-adaptive pseudo anomalies with density-aware generation, micro-environment camouflage calibration, and quality-aware score fusion.

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Method Overview](#method-overview)
- [Code Structure](#code-structure)
- [Datasets](#datasets)
- [Environment Setup](#environment-setup)
- [Reproducing Experiments](#reproducing-experiments)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Running DMC-GGAD](#step-2-running-dmc-ggad)
  - [Step 3: Running Baselines](#step-3-running-baselines)
  - [Step 4: Hyperparameter Configuration](#step-4-hyperparameter-configuration)
  - [Step 5: Evaluation Metrics](#step-5-evaluation-metrics)
  - [Step 6: Reproducing All Results](#step-6-reproducing-all-results)
- [Expected Results](#expected-results)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

## Background & Motivation

Graph anomaly detection (GAD) plays a critical role in financial fraud detection, network security monitoring, and rumor source tracing. Semi-supervised generative methods—exemplified by GGAD—use a small set of known normal nodes to synthesize pseudo anomalies, providing the discriminator with learnable negative samples.

However, existing GGAD-style methods implicitly assume **topological uniformity**: they apply a single anomaly generation rule across all regions of the graph. This causes two fundamental problems:

1. **In dense communities**, synthetic anomalies deviate excessively from the local manifold, becoming trivially identifiable artificial negatives.
2. **In sparse regions**, the lack of reliable structural context leads to overly conspicuous and distributionally narrow pseudo anomalies.

Furthermore, real-world anomalies are not always isolated outliers—many embed themselves in high-homophily micro-environments to mimic normal behavior. Topology-insensitive generation thus produces **pseudo anomaly distribution mismatch**, degrading the discriminative boundary.

DMC-GGAD addresses this gap by unifying three mechanisms under the central theme of **pseudo anomaly quality**: density-aware synthesis, micro-environment camouflage calibration, and quality-aware score fusion.

---

## Method Overview

DMC-GGAD adopts a **two-stage decoupled architecture**:

### Stage 1: Static REC Weight Precomputation (Zero-Cost)

Before training, for each labeled normal anchor node $v_i$, we precompute a micro-environment camouflage weight $r_i$ based solely on raw input features $\mathbf{X}$ and sparse topology $\mathbf{A}$:

- Extract the **pure neighborhood subgraph** (retaining only edges where both endpoints are normal nodes).
- Compute local homophily $H(v_i)$ via cosine similarity of raw features.
- Normalize via Z-score and map through $\tanh(\cdot)$ to obtain $r_i \in [-1, 1]$, which controls whether the pseudo anomaly is pushed away from or pulled toward the anchor.
- Store in a lookup table for $O(1)$ access during training.

This stage is executed once and does not participate in gradient computation.

### Stage 2: Density-Aware Generation & Adversarial Training (End-to-End)

A multi-layer GCN encoder extracts high-order embeddings $\mathbf{H}$. The density-aware structural generator adaptively selects a synthesis path based on local topological density $\rho_i$:

| Density Condition | Strategy | Description |
|:--|:--|:--|
| High density ($\rho_i > \tau$) | Feature-driven attention aggregation | Attention over 2-hop neighborhood candidates, injected with Gaussian noise to diversify generation patterns |
| Low density ($\rho_i \leq \tau$) | Random latent feature mixing | Softmax-normalized convex combination of candidate embeddings |

**Joint training objective:**

$$\mathcal{L} = \lambda_1 \mathcal{L}_{margin} + \lambda_2 \mathcal{L}_{bce} + \lambda_{scale} \mathcal{L}_{rec}$$

- $\mathcal{L}_{margin}$: Sparse edge-level local affinity margin loss (computed over actual edges, avoiding $O(N^2)$ dense similarity matrices).
- $\mathcal{L}_{bce}$: Binary cross-entropy classification loss.
- $\mathcal{L}_{rec}$: REC push-pull reconstruction loss, using precomputed $r_i$ to calibrate the distance between generated pseudo anomalies and their anchors.

### Inference: TAF-QA Fusion

During inference, the **Topology-Adaptive Fusion via Quality Assessment** (TAF-QA) mechanism calibrates the contribution of neural discriminative scores and training-set graph prior scores using an unsupervised quality metric computed on labeled normal nodes only. Crucially, it uses **no validation set labels, no test set labels, and no offline regression targets**—maintaining strict information isolation under the semi-supervised protocol.

![Architecture Overview](aijiagoutu.png)

---

## Code Structure

```
dmc-ggad/
├── dmc-ggad/                          # Core method implementation
│   ├── model.py                       # DMC-GGAD model (GCN encoder, density-aware generator, classifier)
│   ├── run.py                         # Training pipeline & evaluation entry point
│   └── utils.py                       # Data loading, preprocessing, normalization utilities
├── baselines/                         # Baseline implementations
│   ├── dominant/                      # DOMINANT baseline
│   ├── anomalyDAE/                    # AnomalyDAE baseline
│   ├── ocgnn/                         # OCGNN baseline
│   ├── aegis/                         # AEGIS baseline
│   ├── gaan/                          # GAAN baseline
│   ├── tam/                           # TAM baseline
│   ├── GGAD/                          # GGAD baseline
│   ├── CoLA/                          # CoLA baseline
│   ├── RHO/                           # RHO baseline
│   └── utils/                         # Shared utilities for baselines
├── datasets/                          # Benchmark datasets (10 graphs)
├── environment.yml                    # Conda environment specification
├── main.tex                           # Full LaTeX source of the paper
└── README.md                          # This file
```

### Main File Descriptions

| File | Description |
|:--|:--|
| `dmc-ggad/model.py` | Defines `Model` (GCN encoder + discriminator), `DensityAwareStructuralGenerator` (adaptive pseudo anomaly synthesis with attention aggregation and random mixing branches), and `GCN` layer. |
| `dmc-ggad/run.py` | Main entry point. Implements: argument parsing, dataset-specific hyperparameter configuration (`BEST_PARAMS`), data loading, graph prior score construction (IsolationForest, LOF, multiple graph/attribute statistics), REC weight precomputation, training loop with sparse affinity loss, TAF-QA fusion weight calculation, and CSV result logging. |
| `dmc-ggad/utils.py` | Utilities for dataset loading (`.mat` and `.npz`), adjacency normalization, feature preprocessing, and sparse tensor conversion. |
| `baselines/GGAD/ggad.py` | Reference GGAD implementation following the original paper. |
| `baselines/utils/utils.py` | Shared dataset loading and preprocessing functions used by all baselines. |

---

## Datasets

We conduct experiments on **10 real-world graph datasets** spanning citation networks, social networks, co-purchase graphs, and crowdsourcing platforms:

| Dataset | Nodes | Edges | Features | Anomaly Ratio | Domain |
|:--|--:|--:|--:|--:|:--|
| **ACM** | 16,484 | 71,980 | 8,337 | 3.6% | Citation network |
| **Citeseer** | 3,327 | 4,732 | 3,703 | 4.5% | Citation network |
| **Cora** | 2,708 | 5,429 | 1,433 | 5.5% | Citation network |
| **DBLP** | 5,484 | 8,117 | 6,775 | 5.6% | Citation network |
| **Facebook** | 4,039 | 88,234 | 1,283 | 6.1% | Social network |
| **Flickr** | 7,575 | 239,738 | 12,047 | 5.9% | Social network |
| **Photo** | 7,650 | 119,081 | 745 | 5.7% | Co-purchase |
| **Pubmed** | 19,717 | 44,338 | 500 | 2.5% | Citation network |
| **Weibo** | 8,405 | 407,963 | 400 | 10.3% | Social network |
| **Tolokers** | 11,758 | 519,000 | 10 | 21.8% | Crowdsourcing |

**Anomaly injection protocol**: Following standard GAD literature, anomalies are injected via structural perturbation (adding/deleting edges) and attribute perturbation (adding Gaussian noise to features) to simulate both structural and contextual anomalies. The anomaly ratio reflects the proportion of injected anomalies in each dataset.

**Data format**: Datasets are stored as `.mat` files (with `tolokers` using `.npz`). Each file contains adjacency matrix, node features, and ground-truth labels. The data loader in `dmc-ggad/utils.py` handles both formats automatically.

**Data split**: In the semi-supervised setting, 25% of normal nodes are used as labeled training data (`train_rate=0.25`). All abnormal nodes are held out entirely from training and are only used for testing. Within the labeled normal set, 50% are designated as anchors for REC weight computation, and 15% of those serve as abnormal label anchors.

---

## Environment Setup

### Requirements

- **Python**: 3.10
- **PyTorch**: 2.0+ (CUDA 11.8 recommended)
- **Operating System**: Linux / Windows / macOS

### Option 1: Conda Environment (Recommended)

```bash
# Create environment from the provided specification
conda env create -f environment.yml

# Activate environment
conda activate ggad_new

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Option 2: Manual Installation

```bash
# Create a fresh environment
conda create -n ggad_new python=3.10
conda activate ggad_new

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy scipy scikit-learn matplotlib networkx
```

---

## Reproducing Experiments

The strict protocol uses the paper seed set (`[0, 1, 3, 5, 42, 123, 456, 789, 3407, 2026]`) and reports mean +/- standard deviation over 10 independent runs. Models access only labeled normal nodes during training. By default, validation/test labels are not used for hyperparameter tuning, early stopping, score fusion, or epoch selection; test labels are evaluated only at the final epoch for reporting.

### Step 1: Data Preparation

Place all dataset files in the `datasets/` directory:

```
dmc-ggad/datasets/
├── acm.mat
├── citeseer.mat
├── cora.mat
├── dblp.mat
├── facebook.mat
├── flickr.mat
├── photo.mat
├── pubmed.mat
├── tolokers.npz
└── weibo.mat
```

No additional preprocessing is required—the loader handles feature normalization, adjacency normalization (symmetric $D^{-1/2}AD^{-1/2}$), and self-loop addition automatically.

Current repository note: the public dataset folder may not include `dblp.mat`. Add the DBLP file before claiming a full 10-dataset reproduction; otherwise DBLP will fail and only the available datasets can be verified.

### Step 2: Running DMC-GGAD

#### Single Dataset, Single Seed

```bash
cd dmc-ggad/dmc-ggad
python run.py --dataset cora --seed 123
```

#### Available Datasets, All 10 Paper Seeds

```bash
cd dmc-ggad/dmc-ggad
python run.py --strict_3_7
```

This runs all datasets listed in `DATASET_LIST` with all seeds in `SEED_LIST` and saves rows to `dmc-ggad/results.csv` by default.

#### Custom Configurations

```bash
python run.py --dataset facebook --seed 3 --device cpu --num_epoch 500 --strict_3_7
```

### Step 3: Running Baselines

Each baseline resides in its own subdirectory under `baselines/`. Navigate to the method's directory and execute its `run.py` or equivalent entry point:

```bash
# Example: run GGAD baseline
cd baselines/GGAD
python ggad.py --dataset cora --seed 123

# Example: run DOMINANT baseline
cd baselines/dominant
python dominant.py --dataset cora --seed 123

# Example: run CoLA baseline
cd baselines/CoLA
python run.py --dataset cora --seed 123
```

**Note**: Some baselines (e.g., AnomalyDAE, TAM) may encounter memory issues on edge-dense datasets such as Flickr and Tolokers. DMC-GGAD's sparse message-passing design ensures it completes on all 10 datasets.

#### Strong Classical NA Baselines

For higher-standard comparisons, the repository includes a strict runner for
classical neighborhood-aggregation baselines:

- `XGBoost+NA`
- `RandomForest+NA`
- `ExtraTrees+NA`
- `MLP+NA`

These baselines use the same strict label-visibility protocol as DMC-GGAD:
25% of normal nodes are labeled for training, all real anomaly labels are held
out until final reporting, and no validation/test labels are used for model
selection. Since these are binary classifiers, the runner trains them with
normal training nodes plus pseudo-negative samples generated from the normal
training distribution only. Results record this explicitly as
`label_visibility=normal_only_pseudo_negative`.

```bash
# Smoke test on one dataset and one seed
python baselines/strong_na/run_strong_na.py \
  --datasets cora \
  --seeds 123 \
  --methods rf_na,extratrees_na,mlp_na \
  --output_csv baselines/strong_na/strong_na_results.csv

# Full strict run on the configured datasets and paper seed set
python baselines/strong_na/run_strong_na.py \
  --methods xgboost_na,rf_na,extratrees_na,mlp_na \
  --skip_completed

# Summarize mean +/- std across seeds
python baselines/strong_na/summarize_strong_na.py \
  --input_csv baselines/strong_na/strong_na_results.csv \
  --output_csv baselines/strong_na/strong_na_summary.csv
```

`xgboost_na` requires the optional `xgboost` package. If it is unavailable, the
runner records the failure for that method without stopping the remaining
baselines.

### Step 4: Hyperparameter Configuration

DMC-GGAD uses dataset-specific hyperparameters pre-configured in `BEST_PARAMS` (see `run.py`). The following table lists the optimal configuration for each dataset:

| Dataset | `embedding_dim` | `fixed_weight_margin` ($\lambda_1$) | `global_sample_rate` |
|:--|:--|:--|:--|
| ACM | 512 | 2.0 | 0.01 |
| Citeseer | 256 | 0.5 | 0.01 |
| Cora | 128 | 1.0 | 0.10 |
| DBLP | 512 | 0.5 | 0.01 |
| Facebook | 512 | 4.0 | 0.10 |
| Flickr | 300 | 2.0 | 0.01 |
| Photo | 300 | 2.0 | 0.10 |
| Pubmed | 256 | 0.5 | 0.30 |
| Weibo | 300 | 0.5 | 0.01 |
| Tolokers | 256 | 1.0 | 0.10 |

**Fixed hyperparameters across all datasets:**

| Parameter | Value | Description |
|:--|:--|:--|
| `num_layers` | 3 | GCN encoder depth |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 0.0 | L2 regularization coefficient |
| `num_epoch` | 200 | Fixed training epochs used by the default script |
| `patience` | 100 | Legacy-only early stopping patience when `--allow_test_metric_selection` is explicitly enabled |
| `train_rate` | 0.25 | Proportion of normal nodes used as labeled anchors |
| $\lambda_2$ (BCE) | 1.0 | Binary cross-entropy loss weight |
| $\lambda_{scale}$ (REC) | 5.0 | Reconstruction push-pull loss weight |
| `density_threshold` ($\tau$) | 1.0 | Density partition threshold |
| `hop` | 2 | Multi-hop neighborhood range for candidate pool |

**Hyperparameter protocol**: The checked-in script uses fixed dataset-specific hyperparameters from `run.py`. Under the strict protocol, validation and test labels are not used for hyperparameter tuning, early stopping, score fusion, or epoch selection. The legacy behavior that selects the best epoch by test AUC is available only through `--allow_test_metric_selection` for backward compatibility and should not be used for strict paper-style reporting.

### Step 5: Evaluation Metrics

We evaluate all methods using two standard metrics:

- **AUROC** (Area Under the Receiver Operating Characteristic Curve): Primary reporting metric. Measures ranking quality across all threshold levels.
- **AP** (Average Precision): Measures the quality of top-ranked predictions, particularly sensitive to early-precision performance.

Both metrics are computed using `sklearn.metrics.roc_auc_score` and `sklearn.metrics.average_precision_score`. The implementation is in the `evaluate()` function of `run.py`.

### Step 6: Reproducing All Results

To reproduce the complete set of experiments reported in the paper:

**1. DMC-GGAD main results (Table 1 & 2):**

```bash
cd dmc-ggad/dmc-ggad
python run.py --strict_3_7
```

**2. Ablation study:**

Use CLI flags instead of manually editing training code:

| Variant | Modification |
|:--|:--|
| `w/o Density` | `--disable_density` |
| `w/o REC` | `--disable_rec` |
| `w/o TAF-QA` | `--disable_taf_qa` |
| `Model-only` | `--disable_graph_prior` |
| `Prior-only` | `--prior_only` |
| Three-way quick ablation | `--three_ablation` |

**3. Sensitivity analysis:**

Vary the controlled parameter while keeping all others fixed:

- Fake anomaly ratio: `{0.05, 0.10, 0.15, 0.25, 0.50}`
- Density threshold $\tau$: `{0.25, 0.50, 0.75, 1.00, 1.25, 1.50}`
- REC-density joint weight: `{1.0, 2.0, 3.0, 4.0, 5.0}` (with $\tau$ adjusted accordingly)

**4. Robustness experiments:**

Three perturbation types are applied to test robustness:

- **Feature noise**: Add zero-mean Gaussian noise with $\sigma \in \{0.01, 0.05, 0.10, 0.15, 0.20\}$ to node features.
- **Structure perturbation**: Randomly add/remove $\{1\%, 5\%, 10\%, 15\%, 20\%\}$ of edges.
- **Label contamination**: Flip $\{5\%, 10\%, 15\%, 20\%, 25\%\}$ of labeled normal nodes to anomalous labels.

---

## Expected Results

### Main Results (AUROC)

Note: the DMC-GGAD row below uses the 10-seed values from the paper PDF. Regenerate this table from the corrected script before using it as a submission artifact.

The following table is a paper-result snapshot. Values should be regenerated with the fixed strict protocol before submission.

| Method | ACM | Citeseer | Cora | DBLP | Facebook | Flickr | Photo | Pubmed | Weibo | Tolokers |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| DOMINANT | 0.7479 | 0.7673 | 0.7525 | 0.7161 | 0.5085 | 0.7431 | 0.5239 | 0.7658 | 0.9361 | 0.5804 |
| AnomalyDAE | – | 0.7623 | 0.7441 | 0.7454 | 0.5059 | – | 0.5225 | 0.7768 | 0.9369 | 0.6452 |
| OCGNN | 0.7109 | 0.4857 | 0.4796 | 0.5897 | 0.6089 | 0.7693 | 0.5609 | 0.6876 | **0.9693** | 0.4920 |
| AEGIS | 0.7558 | 0.6128 | 0.5975 | 0.5309 | 0.5504 | 0.7704 | 0.6090 | 0.5109 | 0.9666 | 0.6514 |
| GAAN | 0.7578 | 0.7699 | 0.7704 | 0.7535 | 0.5803 | 0.7635 | 0.4314 | 0.7678 | 0.9282 | 0.5146 |
| TAM | **0.8639** | 0.5000 | 0.8228 | 0.6111 | 0.8751 | 0.7228 | 0.5983 | 0.8773 | 0.7224 | 0.4981 |
| GGAD | 0.3751 | 0.5859 | 0.4838 | 0.5138 | 0.7626 | 0.4481 | 0.6325 | 0.3293 | 0.1184 | 0.4459 |
| CoLA | 0.6051 | 0.7640 | 0.6899 | 0.3608 | 0.7811 | 0.5771 | 0.6153 | 0.6503 | 0.2854 | 0.5519 |
| RHO | 0.3411 | 0.4449 | 0.3096 | 0.3410 | 0.5177 | 0.6076 | 0.7149 | 0.2874 | – | 0.5110 |
| **DMC-GGAD** | 0.8409 | 0.8348 | 0.7989 | 0.8271 | 0.8898 | 0.7820 | 0.8448 | 0.8753 | 0.8691 | 0.6889 |

### Ablation Study Summary

| Variant | Avg AUROC (10 datasets) | $\Delta$ from Full |
|:--|:--|:--|
| Full DMC-GGAD | 0.870 | — |
| w/o Density | 0.841 | –0.029 |
| w/o REC + Density | 0.806 | –0.064 |
| w/o REC + TAF-QA | 0.725 | –0.145 |
| w/o Density + REC + TAF-QA | 0.682 | –0.189 |

> "–" indicates the method could not complete due to memory or stability issues on that dataset.

---

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{dmcggad2025,
  title={DMC-GGAD: Density-Aware and Micro-Environment Camouflage for Semi-Supervised Graph Anomaly Detection},
  author={xx},
  journal={},
  year={2025}
}
```

---

## Contact

For questions, bug reports, or collaboration inquiries, please contact:

- **Author**: xx
- **Institution**: School of Computer Science, China West Normal University, Nanchong, Sichuan, China
- **Email**: @.com
- **GitHub Issues**: Please open an issue in this repository for code-related questions.

---

## License

This project is released under the [MIT License](./LICENSE).
