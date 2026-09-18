# Stability of Local Feature Explanations for Graph Neural Networks under Controlled Perturbations

An independent research study investigating the stability of local GraphLIME feature explanations for a frozen Graph Convolutional Network (GCN) under controlled feature perturbations on the Cora citation network.

## Research Report

The complete independent research study is available as a PDF:

**[Stability of Local Feature Explanations for Graph Neural Networks under Controlled Perturbations](./GraphLIME_Stability_Research_Report.pdf)**

The report presents the research motivation, related work, methodology, experimental results, statistical analysis, discussion, limitations, and supplementary experimental details for this repository.

---

## Overview

Post-hoc explanations are often interpreted as descriptions of why a trained model produced a prediction. However, prediction stability does not necessarily imply explanation stability.

This study investigates the question:

> **How stable are GraphLIME local feature explanations for a frozen graph neural network under controlled perturbations of the input node features?**

The experiment evaluates whether a GNN can preserve its predicted class while the corresponding GraphLIME explanation changes substantially.

The study uses:

- the Cora citation-network dataset;
- a frozen two-layer Graph Convolutional Network;
- GraphLIME-style local explanations based on Gaussian kernels and non-negative HSIC-Lasso;
- controlled masking of active feature entries within each target node's 2-hop neighbourhood;
- masking rates of 1%, 5%, 10%, and 20%;
- 10 deterministic perturbation repetitions per node and masking rate;
- prediction stability, explanation availability, and top-10 Jaccard similarity as primary measurements.

The completed experimental pipeline is frozen. Subsequent repository changes are limited to documentation, packaging, and explicitly separate follow-up studies.

---

## Research Question

Let

- $f(v)$ denote the baseline prediction for target node $v$;
- $f_r(v)$ denote its prediction after perturbation at masking rate $r$;
- $E_0(v)$ denote the baseline GraphLIME top-$K$ explanation;
- $E_r(v)$ denote the corresponding perturbed explanation.

The central phenomenon of interest is

```math
f(v) = f_r(v)
\quad\text{while}\quad
E_0(v) \neq E_r(v).
```

The primary explanation-stability metric is top-$K$ Jaccard similarity:

```math
J(E_0,E_r)
=
\frac{|E_0 \cap E_r|}
{|E_0 \cup E_r|}.
```

The primary analysis also evaluates explanation stability conditional on the predicted class remaining unchanged.

The complete pre-specified research questions and hypotheses are documented in [`research/`](research/).

---

## Experimental Design

### Dataset

The experiment uses **Cora** through PyTorch Geometric's `Planetoid` dataset loader with `NormalizeFeatures`.

The standard Planetoid split contains:

| Quantity | Value |
|---|---:|
| Nodes | 2,708 |
| Edges | 10,556 |
| Features | 1,433 |
| Classes | 7 |
| Training nodes | 140 |
| Validation nodes | 500 |
| Test nodes | 1,000 |

### Baseline GCN

The baseline model is a two-layer GCN:

```text
GCNConv(1433 -> 16)
ReLU
Dropout(p=0.5)
GCNConv(16 -> 7)
```

Training uses Adam with learning rate `0.01` and weight decay `0.0005`.

Model selection is based exclusively on validation accuracy. The test set is not used for checkpoint selection.

The frozen checkpoint achieved:

| Metric | Value |
|---|---:|
| Best epoch | 125 |
| Training accuracy | 0.9929 |
| Validation accuracy | 0.7840 |
| Test accuracy | 0.8060 |

### Target Nodes

One hundred correctly classified test nodes were selected using a deterministic seeded random permutation.

Three selected nodes produced locally degenerate GraphLIME output kernels and were retained as diagnostic failures rather than replaced.

The primary perturbation cohort therefore contains **97 baseline-explainable target nodes**.

### GraphLIME

For each target node, the implementation:

1. extracts its 2-hop neighbourhood;
2. obtains the frozen GCN probability vectors for local nodes;
3. constructs a Gaussian kernel for every locally varying input feature;
4. constructs a Gaussian kernel over the local GNN probability vectors;
5. centers and Frobenius-normalizes the kernels;
6. solves a non-negative HSIC-Lasso objective;
7. ranks features by their learned coefficients.

The implemented objective is

```math
\min_{\beta \geq 0}
\frac{1}{2}
\left\|
\bar{L}
-
\sum_k \beta_k \bar{K}^{(k)}
\right\|_F^2
+
\rho \|\beta\|_1.
```

The implementation uses non-negative proximal-gradient optimization with a step size derived from the spectral norm of the flattened kernel design matrix.

The regularization parameter was calibrated before the perturbation study and frozen at

```math
\rho = 0.03.
```

The implementation optimizes the GraphLIME/HSIC-Lasso objective used in this study; it should not be interpreted as an exact reproduction of every optimization detail of the original GraphLIME implementation.

### Controlled Feature Perturbation

Perturbations are restricted to originally non-zero feature entries within the frozen 2-hop neighbourhood of each target node.

The nominal masking rates are:

```text
1%, 5%, 10%, 20%
```

with 10 perturbation seeds per rate.

The number of entries masked is

```math
n_{\mathrm{mask}}
=
\min\left(
n_{\mathrm{active}},
\max\left(
1,
\left\lfloor r n_{\mathrm{active}} + 0.5 \right\rfloor
\right)
\right).
```

Each observation starts from the original feature matrix. Masks are sampled separately across rates rather than constructed as nested perturbations.

The experiment does **not**:

- retrain the GCN;
- modify graph structure;
- renormalize feature rows after masking;
- replace failed target nodes;
- treat unavailable explanations as Jaccard similarity zero.

The complete perturbation protocol is documented in `research/perturbation_protocol.md`.

---

## Experimental Grid

The frozen perturbation study contains

```math
97 \times 4 \times 10 = 3880
```

node-rate-seed observations.

Of these:

- 3,821 produced valid complete top-10 explanations;
- 59 produced unavailable explanations;
- unavailable explanations remain missing in explanation-overlap analyses.

The statistical unit for across-rate inference is the **target node**. Perturbation seeds are repeated observations within each node-rate condition.

---

## Main Results

The principal aggregate results are:

| Masking rate | Prediction stability | Explanation availability | Mean top-10 Jaccard | Mean Jaccard given prediction unchanged |
|---:|---:|---:|---:|---:|
| 1% | 0.9969 | 0.9876 | 0.9015 | 0.9014 |
| 5% | 0.9876 | 0.9835 | 0.7253 | 0.7266 |
| 10% | 0.9835 | 0.9825 | 0.6045 | 0.6063 |
| 20% | 0.9588 | 0.9856 | 0.4541 | 0.4560 |

Across the frozen masking-rate range, explanation overlap decreased substantially while prediction agreement remained comparatively high.

For the 1%-to-20% endpoint comparison:

- mean paired explanation-overlap change: **-0.4475**;
- 95% node-level bootstrap CI: **[-0.4660, -0.4282]**;
- mean prediction-stability change: **-0.0381**;
- 95% node-level bootstrap CI: **[-0.0649, -0.0155]**.

For 86 of 97 target nodes, the node-level prediction-stability proportion was unchanged between the 1% and 20% endpoints while mean explanation overlap decreased.

This endpoint statistic concerns prediction-stability proportions across repeated perturbations; it does not imply that every individual perturbed prediction remained unchanged.

Explanation unavailability was sparse and did not increase monotonically with masking severity.

---

## Interpretation

The experiment provides evidence, within this controlled Cora/GCN/GraphLIME setting, that **prediction stability and local explanation stability can be substantially decoupled**.

In particular, a stable predicted class should not automatically be interpreted as evidence that the feature-level explanation is stable.

The result should not be interpreted as establishing universal instability of GraphLIME or GNN explainers. The study uses one graph dataset, one baseline GCN architecture, one explanation method, and one controlled feature-perturbation protocol.

---

## Statistical Analysis

The primary statistical unit is the target node.

The analysis uses:

- node-level aggregation over perturbation repetitions;
- paired comparisons across masking rates;
- 10,000-resample paired node-level percentile bootstrap confidence intervals;
- two-sided Wilcoxon signed-rank tests as secondary analyses;
- Holm correction within each six-contrast outcome family;
- matched-pairs rank-biserial effect sizes;
- descriptive Pearson and Spearman correlations for prediction/explanation endpoint coupling.

Because all target nodes belong to a single graph, node observations should not be interpreted as independent population samples. Correlations used for the coupling analysis are therefore descriptive rather than accompanied by IID inferential significance claims.

---

## Repository Structure

```text
graphlime_stability/
|
|-- configs/
|   `-- baseline.yaml
|
|-- experiments/
|   |-- 01_train_baseline.py
|   |-- 02_baseline_explanations.py
|   |-- 02b_rho_calibration.py
|   |-- 02c_generate_explanations.py
|   |-- 02d_validate_baseline_solutions.py
|   |-- 03_feature_perturbation.py
|   |-- 03b_validate_perturbation_results.py
|   |-- 04_analyze_stability.py
|   |-- 04b_hypothesis_analysis.py
|   |-- 04c_finalize_hypotheses.py
|   `-- 05_plot_stability.py
|
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- explain.py
|   |-- metrics.py
|   |-- model.py
|   |-- perturb.py
|   |-- train.py
|   `-- utils.py
|
|-- research/
|   |-- research_question.md
|   |-- hypotheses.md
|   |-- perturbation_protocol.md
|   `-- experiment_log.md
|
|-- models/
|   `-- cora_gcn_seed42.pt
|
|-- results/
|   |-- baseline/
|   |-- perturbations/
|   |-- analysis/
|   `-- figures/
|
|-- GraphLIME_Stability_Research_Report.pdf
|-- README.md
|-- requirements.txt
|-- requirements-frozen.txt
`-- .gitignore
```

---

## Reproduction

### Environment

The frozen experiment was run with Python 3.13 and a CPU PyTorch/PyTorch Geometric environment.

Create and activate a virtual environment before installing dependencies.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the complete package snapshot of the original experimental environment:

```powershell
pip install -r requirements-frozen.txt
```

### Canonical Pipeline

Run commands from the repository root.

```powershell
python -m experiments.01_train_baseline
python -m experiments.02_baseline_explanations
python -m experiments.02b_rho_calibration
python -m experiments.02c_generate_explanations
python -m experiments.02d_validate_baseline_solutions
python -m experiments.03_feature_perturbation
python -m experiments.03b_validate_perturbation_results
python -m experiments.04_analyze_stability
python -m experiments.04b_hypothesis_analysis
python -m experiments.04c_finalize_hypotheses
python -m experiments.05_plot_stability
```

The scripts contain validation checks intended to detect violations of the frozen experimental protocol.

`configs/baseline.yaml` is a declarative provenance record finalized after the experiment. The current scripts do not use it as their runtime configuration source.

---

## Research Documentation

The `research/` directory separates the pre-specified study design from the final experimental record:

- `research_question.md` — research question, operational definitions, scope, and interpretation boundaries;
- `hypotheses.md` — frozen hypotheses;
- `perturbation_protocol.md` — controlled perturbation design;
- `experiment_log.md` — chronological experimental decisions, validation results, findings, limitations, and freeze status.

The detailed configuration and provenance record is stored in:

```text
configs/baseline.yaml
```

---

## Frozen Study Policy

Phases 0-11 of the canonical experiment are complete and frozen.

The following are not permitted as modifications of the existing frozen study:

- post-hoc parameter tuning;
- replacement of failed target nodes;
- manual alteration of raw results;
- conversion of unavailable explanations to zero Jaccard similarity;
- retrospective changes to the perturbation protocol;
- presentation of newly generated results as if they belonged to the original frozen experiment.

Future extensions should be recorded as separate experiments while preserving the current artifacts.

---

## Status

**Experimental pipeline: complete, validated, analyzed, visualized, and frozen.**

**Independent research report: complete and available in this repository.**

The study artifacts, frozen experimental results, statistical analyses, figures, research documentation, and final research report are retained in this repository for reproducibility and independent inspection.
