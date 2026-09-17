# Experiment Log

## Study

**GraphLIME Explanation Stability under Controlled Local Feature Masking**

This log records the experimental development, validation, perturbation study,
statistical analysis, and visualization stages of the project.

The central methodological principle is that experimental choices are frozen
before downstream outcomes are inspected. Model parameters, target nodes,
GraphLIME settings, perturbation settings, and statistical procedures are not
retuned in response to explanation-stability results.

---

## E001 — Baseline Cora GCN

**Date:** 2026-09-16

### Objective

Establish a reproducible baseline Graph Convolutional Network (GCN) for Cora
node classification before conducting GraphLIME explanation-stability
experiments.

### Dataset

- Dataset: Cora
- Nodes: 2,708
- Edges: 10,556
- Features: 1,433
- Classes: 7
- Training nodes: 140
- Validation nodes: 500
- Test nodes: 1,000
- Feature preprocessing: PyTorch Geometric `NormalizeFeatures`

The standard Planetoid split is retained.

### Model

Architecture: two-layer Graph Convolutional Network (GCN).

- Hidden channels: 16
- Dropout: 0.5
- Output: unnormalized class logits

### Training

- Seed: 42
- Maximum epochs: 200
- Learning rate: 0.01
- Weight decay: 0.0005
- Optimizer: Adam
- Loss: cross-entropy
- Device: CPU

### Model-Selection Protocol

Validation accuracy is evaluated after every training epoch.

The checkpoint with the highest validation accuracy is retained. If multiple
epochs achieve the same highest validation accuracy, the earliest epoch is
retained through a strict `>` comparison.

The test set is not used for model selection.

After training, the best-validation checkpoint is restored before final
training, validation, and test evaluation.

### Initial Fixed-Epoch Diagnostic

The initial implementation evaluated the model after 200 epochs without
validation-based checkpoint selection.

Observed results:

- Train accuracy: 1.0000
- Validation accuracy: 0.7800
- Test accuracy: 0.7990

Training accuracy approached 1.0 while validation accuracy plateaued around
0.76–0.78. Validation-based checkpoint selection was therefore implemented
before freezing the research baseline.

### Frozen Baseline Results

- Best epoch: 125
- Train accuracy: 0.9929
- Validation accuracy: 0.7840
- Test accuracy: 0.8060

### Frozen Checkpoint

`models/cora_gcn_seed42.pt`

The checkpoint contains the parameters of the model selected using validation
accuracy rather than necessarily those of the epoch-200 model.

### Interpretation

Epoch 125 was the earliest epoch attaining the maximum observed validation
accuracy of 0.7840.

The difference between training and validation performance indicates a
generalization gap, while later training did not improve the validation
criterion.

The epoch-125 checkpoint was frozen for all subsequent experiments. The test
set was not used for model selection, and no later explanation-stability result
was used to retune the model.

### Status

**PASSED — baseline GCN validated and frozen.**

---

## E002 — GraphLIME Local Input Diagnostic

**Date:** 2026-09-16

### Objective

Validate construction of the local data required for a GraphLIME explanation
before implementing kernel construction and HSIC-Lasso.

### Diagnostic Target

- Target node: 1903
- Neighborhood definition: 2-hop
- Neighborhood nodes: 26
- Subgraph edges: 78
- Target local index: 18
- Local feature matrix shape: `[26, 1433]`
- Local GNN output matrix shape: `[26, 7]`
- Target true label: 5
- Target predicted label: 5
- Target confidence: 0.738728

### Validation

- Target present at expected local index: PASS
- Local feature/output sample dimensions agree: PASS
- Feature dimension equals Cora feature count: PASS
- Output dimension equals Cora class count: PASS
- Local output probability vectors sum to one: PASS

### Status

**PASSED — GraphLIME local input construction validated.**

---

## E003 — GraphLIME Kernel Diagnostic

**Date:** 2026-09-16

### Objective

Validate construction, centering, normalization, and numerical properties of
the GraphLIME feature and output kernels before solving the HSIC-Lasso
objective.

### Diagnostic Configuration

- Target node: 1903
- Neighborhood: 2-hop
- Neighborhood nodes: 26
- Input features: 1,433
- Active local features: 260
- Degenerate local features: 1,173
- Local feature matrix shape: `[26, 1433]`
- Local GNN output matrix shape: `[26, 7]`
- Output kernel shape: `[26, 26]`
- Output bandwidth: 0.195251
- Output centered Frobenius norm before normalization: 6.978704

### Validation

- Active + degenerate features = 1,433: PASS
- Feature-kernel shape validation: PASS
- Feature-kernel finite-value validation: PASS
- Feature-kernel Frobenius normalization: PASS
- Output-kernel shape validation: PASS
- Output-kernel finite-value validation: PASS
- Output-kernel Frobenius normalization: PASS
- Output-kernel centering: PASS
- Degenerate-feature handling: PASS

### Implementation Decisions

Gaussian-kernel bandwidths use the median positive pairwise Euclidean
distance.

Locally degenerate features are excluded from the active feature-kernel set.

These are explicit implementation decisions and should be distinguished from
requirements imposed directly by the GraphLIME methodology.

### Status

**PASSED — GraphLIME kernel construction validated.**

---

## E004 — Single-Node HSIC-Lasso Diagnostic

**Date:** 2026-09-16

### Objective

Validate the numerical behavior of the non-negative HSIC-Lasso implementation
on one predetermined GraphLIME target node before running the optimizer across
multiple nodes.

The implemented optimization problem is

```text
minimize

    0.5 * ||L_bar - sum_k beta_k K_bar_k||_F^2
    + rho * ||beta||_1

subject to

    beta_k >= 0
```

The Frobenius-norm objective is flattened into an equivalent vectorized
least-squares problem and solved using non-negative proximal-gradient
optimization.

The implementation therefore solves the stated convex GraphLIME HSIC-Lasso
objective, but it does not exactly reproduce the original GraphLIME paper's
non-negative LARS optimization procedure.

### Diagnostic Configuration

- Target node: 1903
- Neighborhood: 2-hop
- Neighborhood nodes: 26
- Active local features: 260
- `TOP_K = 10`
- Maximum optimizer iterations: 10,000
- Coefficient-change tolerance: `1e-8`
- Zero-coefficient tolerance: `1e-8`

Diagnostic rho values:

- 0.001
- 0.003
- 0.010
- 0.030
- 0.100
- 0.300

### Expected Initial Objective

The output kernel is Frobenius-normalized:

```text
||L_bar||_F = 1
```

The optimizer begins from:

```text
beta = 0
```

Therefore, the expected initial objective is:

```text
0.5 * ||L_bar||_F^2 = 0.5
```

All six diagnostic runs produced an initial objective of exactly 0.50000000.

### Results

| rho | Converged | Iterations | Nonzero coefficients | Final objective | Maximum coefficient |
|---:|:---:|---:|---:|---:|---:|
| 0.001 | Yes | 331 | 59 | 0.23096293 | 0.36050606 |
| 0.003 | Yes | 329 | 59 | 0.23499548 | 0.35962400 |
| 0.010 | Yes | 315 | 46 | 0.24865085 | 0.35661221 |
| 0.030 | Yes | 292 | 29 | 0.28426510 | 0.34644812 |
| 0.100 | Yes | 257 | 22 | 0.37886539 | 0.30335858 |
| 0.300 | Yes | 177 | 2 | 0.48454487 | 0.14573692 |

### Numerical Validation

For all six rho values:

- Optimization convergence: PASS
- Initial objective approximately 0.5: PASS
- Final objective no greater than initial objective: PASS
- All coefficients finite: PASS
- All coefficients non-negative: PASS
- Configured coefficient-change convergence criterion satisfied: PASS

### Sparsity Observation

Increasing rho generally produced stronger sparsity:

```text
rho = 0.001 -> 59 nonzero coefficients
rho = 0.003 -> 59 nonzero coefficients
rho = 0.010 -> 46 nonzero coefficients
rho = 0.030 -> 29 nonzero coefficients
rho = 0.100 -> 22 nonzero coefficients
rho = 0.300 ->  2 nonzero coefficients
```

This behavior is consistent with the role of the L1 regularization term in the
HSIC-Lasso objective.

### Leading Features

Across rho values from 0.001 through 0.100, several high-ranking feature IDs
remained similar, including:

- 1149
- 40
- 1332
- 750
- 1336

This observation applies only to the diagnostic target node and is not treated
as evidence of general GraphLIME explanation stability.

At `rho = 0.300`, only two coefficients remained above the numerical zero
threshold, so a complete top-10 explanation was unavailable.

### Decision

No rho value was selected using node 1903 alone.

Selecting rho from one inspected target could tune the explainer to that
particular diagnostic example. A predetermined multi-node calibration was
therefore used for the actual parameter-selection decision.

### Status

**PASSED — single-node HSIC-Lasso behavior validated.**

---

## E005 — Multi-Node HSIC-Lasso Rho Calibration

**Date:** 2026-09-16

### Objective

Select a fixed HSIC-Lasso regularization parameter before generating the full
baseline explanation set and before observing any perturbation-stability
outcomes.

The calibration is intended to select a numerically reliable rho without using
future explanation-stability results.

### Calibration Cohort

The calibration subset consists of the first ten nodes from the already frozen
100-node explanation sample.

No new random sampling was performed.

Calibration node IDs:

1. 1903
2. 1749
3. 1726
4. 2499
5. 2075
6. 1971
7. 1965
8. 2002
9. 2691
10. 2685

Number of calibration nodes: 10.

Number of candidate rho values: 6.

Total HSIC-Lasso runs:

```text
10 nodes * 6 rho values = 60 runs
```

### Frozen Calibration Configuration

- Seed: 42
- Neighborhood: 2-hop
- `TOP_K = 10`
- Maximum optimizer iterations: 10,000
- Coefficient-change tolerance: `1e-8`
- Zero-coefficient tolerance: `1e-8`

Candidate rho values:

- 0.001
- 0.003
- 0.010
- 0.030
- 0.100
- 0.300

### Pre-Specified Selection Rule

A candidate rho is eligible only if:

1. HSIC-Lasso converges for all ten calibration nodes; and
2. every calibration node retains at least `TOP_K = 10` nonzero coefficients.

Among eligible candidates, the largest rho is selected.

The rule therefore prefers the strongest tested regularization that still
provides a complete top-10 explanation for every calibration node and
satisfies the convergence criterion.

The rule was specified before inspecting the multi-node calibration results.

### Calibration Summary

| rho | Converged | Full top-10 | Minimum nonzero | Mean nonzero | Maximum nonzero | Eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.001 | 10/10 | 10/10 | 31 | 132.90 | 293 | Yes |
| 0.003 | 10/10 | 10/10 | 31 | 123.70 | 252 | Yes |
| 0.010 | 9/10 | 10/10 | 31 | 92.10 | 143 | No |
| 0.030 | 10/10 | 10/10 | 16 | 51.70 | 102 | Yes |
| 0.100 | 10/10 | 7/10 | 2 | 22.60 | 43 | No |
| 0.300 | 10/10 | 4/10 | 0 | 6.60 | 21 | No |

### Non-Converged Calibration Observation

One optimization run did not satisfy the predefined convergence criterion:

- Target node: 2499
- `rho = 0.010`
- Neighborhood size: 7
- Active local features: 71
- Degenerate local features: 1,362
- Iterations: 10,000
- Converged: No
- Nonzero coefficients: 54
- Complete top-10 available: Yes
- Final objective: 0.122340

The optimizer reached the configured maximum of 10,000 iterations without
satisfying the coefficient-change convergence threshold of `1e-8`.

The observation was retained rather than discarded or reclassified. The
convergence tolerance and maximum iteration count were not changed in response
to this result.

Consequently, `rho = 0.010` fails the pre-specified calibration rule despite
retaining more than ten nonzero coefficients.

### Observed Neighborhood Variation

The calibration cohort exhibits substantial variation in local 2-hop
neighborhood size.

Examples include:

- Node 2691: 4 neighborhood nodes, 59 active features
- Node 2499: 7 neighborhood nodes, 71 active features
- Node 1903: 26 neighborhood nodes, 260 active features
- Node 2002: 80 neighborhood nodes, 570 active features
- Node 1726: 190 neighborhood nodes, 820 active features
- Node 1749: 194 neighborhood nodes, 839 active features

This variation supports calibrating rho across multiple local neighborhoods
rather than selecting it from one target node.

### Sparsity Behavior

Increasing rho generally reduced the number of nonzero HSIC-Lasso
coefficients.

At `rho = 0.100`, only 7 of 10 calibration nodes retained at least ten
nonzero coefficients.

At `rho = 0.300`, only 4 of 10 calibration nodes retained at least ten
nonzero coefficients, and at least one calibration node produced zero
nonzero coefficients.

These values therefore fail the pre-specified complete-top-10 requirement.

### Calibration Decision

Eligible rho values:

```text
0.001
0.003
0.030
```

The largest eligible value was:

```text
RHO = 0.03
```

The frozen initial GraphLIME configuration is therefore:

```text
SEED = 42
NUM_HOPS = 2
TOP_K = 10
RHO = 0.03
```

`rho = 0.03` was selected because all ten calibration optimizations converged
and every calibration node retained at least ten nonzero coefficients.

The selection did not use explanation-stability measurements, perturbation
results, or behavior outside the predetermined calibration subset.

The parameter was not subsequently retuned using the remaining selected
nodes.

### Saved Artifacts

- `results/baseline/rho_calibration.csv`
- `results/baseline/rho_calibration_summary.csv`

These files are retained as experimental artifacts.

### Status

**PASSED — `RHO = 0.03` selected and frozen.**

---

## E006 — Frozen Baseline GraphLIME Explanation Generation

**Date:** 2026-09-17

### Objective

Generate baseline GraphLIME explanations for the complete predetermined set
of 100 correctly classified Cora test nodes using the frozen model and
GraphLIME configuration.

These explanations define the baseline explanation \(E_0(v)\) against which
perturbed explanations are subsequently compared.

### Frozen Configuration

- Seed: 42
- Baseline model: `models/cora_gcn_seed42.pt`
- Selected-node artifact: `results/baseline/selected_nodes.csv`
- Predetermined selected nodes: 100
- `NUM_HOPS = 2`
- `TOP_K = 10`
- `RHO = 0.03`
- `MAX_ITER = 10000`
- Optimizer tolerance: `1e-8`
- Zero tolerance: `1e-8`

`RHO = 0.03` was selected during the preceding predetermined ten-node
calibration and was not retuned using the remaining selected nodes.

### Frozen-Model Validation

- Checkpoint training seed: 42
- Best training epoch: 125
- Checkpoint validation accuracy: 0.7840
- Checkpoint test accuracy: 0.8060
- Reconstructed test accuracy: 0.8060
- Selected nodes loaded: 100
- Unique selected nodes: 100
- All selected nodes correctly classified by the frozen model: PASS

### Baseline Explanation Results

Of the 100 predetermined selected nodes:

- Valid GraphLIME output kernels: 97/100
- Baseline explanations available: 97/100
- Output-degenerate neighborhoods: 3/100

Among the 97 nodes for which a valid output kernel could be constructed:

- HSIC-Lasso convergence: 97/97
- Finite coefficients: 97/97
- Non-negative coefficients: 97/97
- Objective decrease: 97/97
- Initial objective approximately 0.5: 97/97
- At least ten nonzero coefficients: 97/97
- Complete top-10 explanation: 97/97

Therefore:

```text
97 explainable nodes * 10 features = 970 explanation rows
```

- Expected explanation rows: 970
- Observed explanation rows: 970
- Per-node explanation-row counts: PASS

### Output-Degenerate Nodes

Three predetermined selected nodes did not admit a normalized GraphLIME output
kernel under the frozen 2-hop configuration.

| Node ID | True label | Predicted label | Confidence | Neighborhood size | Subgraph edges | Active features | Degenerate features |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2697 | 3 | 3 | 0.621617 | 2 | 2 | 30 | 1,403 |
| 2625 | 3 | 3 | 0.616579 | 2 | 2 | 28 | 1,405 |
| 2411 | 3 | 3 | 0.404153 | 2 | 2 | 35 | 1,398 |

All three output-degenerate nodes had a 2-hop neighborhood containing only
two nodes.

For these neighborhoods, the local GNN probability vectors contained no
positive pairwise distance. Consequently, the centered GraphLIME output
kernel contained no usable local output variation and could not be
Frobenius-normalized.

No artificial output bandwidth was introduced.

No GraphLIME explanation was fabricated for these nodes.

### Treatment of Output-Degenerate Nodes

The three output-degenerate nodes remain part of the original 100-node
selected sample and remain present in the diagnostic record.

They were not replaced with alternative nodes.

`NUM_HOPS` was not selectively changed for these nodes.

`RHO` was not changed in response to their output degeneracy.

For explanation-stability analyses requiring a baseline explanation, the
analysis populations are defined as:

```text
S_selected =
    the 100 predetermined correctly classified test nodes

S_explainable =
    {v in S_selected :
     a valid baseline GraphLIME output kernel exists and
     a complete top-10 baseline explanation is available}
```

Observed cohort sizes:

```text
|S_selected| = 100
|S_explainable| = 97
```

The three output-degenerate nodes are retained as baseline diagnostic
observations but are not assigned an artificial explanation-overlap score.

### Optimization Statistics

Statistics below apply to the 97 nodes for which HSIC-Lasso optimization was
performed.

Nonzero-coefficient statistics:

- Minimum: 16
- Mean: 51.41
- Median: 50.00
- Maximum: 108

Optimizer-iteration statistics:

- Mean: 708.49
- Maximum: 9,724

Neighborhood-size statistics across all 100 selected nodes:

- Minimum: 2
- Mean: 40.98
- Maximum: 194

### High-Iteration Observation

Node 2477 required 9,724 iterations before satisfying the predetermined
coefficient-change convergence criterion.

The configured maximum was 10,000 iterations.

Node 2477 nevertheless:

- converged under the frozen stopping criterion;
- retained 18 nonzero coefficients;
- produced a complete top-10 explanation;
- produced finite and non-negative coefficients; and
- reduced the optimization objective.

No optimizer setting was changed after observing this case.

### Saved Artifacts

- `results/baseline/explanations.csv`
- `results/baseline/explanation_diagnostics.csv`

The explanation artifact contains 970 rows corresponding to the 97
baseline-explainable nodes.

The diagnostic artifact contains one row for every original selected node,
including the three output-degenerate nodes.

These artifacts are retained without manual modification.

### Status

**PASSED WITH DOCUMENTED OUTPUT-DEGENERATE EXCLUSIONS.**

Frozen baseline explanation cohort:

```text
n = 97
```

---

## E007 — KKT Validation of Baseline HSIC-Lasso Solutions

**Date:** 2026-09-17

### Objective

Independently reconstruct and numerically validate the 97 baseline
HSIC-Lasso solutions before using their GraphLIME explanations as reference
explanations in perturbation experiments.

This experiment is validation-only.

No GraphLIME hyperparameter, optimizer setting, model parameter, selected
node, or baseline explanation is changed as a consequence of this validation.

### Frozen Configuration

- Baseline model: `models/cora_gcn_seed42.pt`
- Baseline explanation cohort: 97 nodes
- `NUM_HOPS = 2`
- `TOP_K = 10`
- `RHO = 0.03`
- `MAX_ITER = 10000`
- Optimizer tolerance: `1e-8`
- Zero tolerance: `1e-8`
- KKT diagnostic tolerance: `1e-4`

Output-degenerate baseline nodes excluded from KKT optimization:

- 2697
- 2625
- 2411

The KKT tolerance is a numerical validation tolerance and is distinct from the
coefficient-change stopping tolerance used by the optimizer.

### KKT Conditions

For the non-negative HSIC-Lasso problem:

```text
minimize

    0.5 * ||X beta - y||_2^2
    + rho * 1^T beta

subject to

    beta >= 0
```

define the smooth gradient:

```text
g = X^T (X beta - y)
```

and:

```text
r = g + rho
```

At an optimum, active coefficients should approximately satisfy:

```text
beta_j > 0  =>  r_j = 0
```

while inactive coefficients should satisfy:

```text
beta_j = 0  =>  r_j >= 0
```

Primal feasibility additionally requires:

```text
beta_j >= 0
```

### Reproduction Validation

All 97 baseline-explainable optimization problems were reconstructed using
the frozen model, graph, neighborhood definition, kernels, rho value, and
optimizer settings.

Results:

- Solver convergence reproduced: 97/97
- Top-10 explanations reproduced: 97/97
- Iteration counts reproduced: 97/97
- Nonzero coefficient counts reproduced: 97/97
- Complete reproduction validation: 97/97

The independently reconstructed solutions therefore reproduced the saved
baseline explanations exactly under the implemented pipeline.

### KKT Validation Results

Pre-specified tolerance:

```text
1e-4
```

Results:

- Active-coordinate stationarity: 97/97 PASS
- Inactive-coordinate dual feasibility: 97/97 PASS
- Primal feasibility: 97/97 PASS
- Complete KKT validation: 97/97 PASS

Worst observed active-coordinate stationarity residual:

```text
6.593764e-07
```

Maximum inactive-coordinate violation:

```text
0.000000e+00
```

Minimum inactive-coordinate slack:

```text
2.796948e-05
```

Maximum complementarity residual:

```text
3.786142e-07
```

The largest active-coordinate residual was substantially below the
pre-specified KKT tolerance of `1e-4`.

No inactive coefficient violated dual feasibility.

### Highest-Iteration Solution

Node ID:

```text
2477
```

Iterations:

```text
9724
```

Maximum active-coordinate stationarity residual:

```text
5.681068e-07
```

Maximum inactive-coordinate violation:

```text
0.000000e+00
```

KKT validation:

```text
PASS
```

Although node 2477 required substantially more optimizer iterations than most
nodes and approached the configured maximum of 10,000, its reconstructed
solution satisfied both the optimizer convergence criterion and the
independent KKT validation.

The optimizer configuration was therefore not changed.

### Frozen Baseline Artifacts

- `models/cora_gcn_seed42.pt`
- `results/baseline/selected_nodes.csv`
- `results/baseline/rho_calibration.csv`
- `results/baseline/rho_calibration_summary.csv`
- `results/baseline/explanations.csv`
- `results/baseline/explanation_diagnostics.csv`
- `results/baseline/kkt_validation.csv`

The primary explanation-stability cohort is frozen at:

```text
n = 97
```

No baseline node is replaced because of later perturbation behavior.

No baseline GraphLIME hyperparameter is retuned using later perturbation
outcomes.

### Phase-8 Decision

Canonical Phase 8 — GraphLIME baseline construction and validation:

**COMPLETE.**

The baseline side of the explanation-stability comparison is frozen and
numerically validated.

### Status

**PASSED — 97/97 baseline solutions reproduced and numerically validated.**

---

## E008 — Controlled Feature-Perturbation Experiment and Integrity Validation

**Date:** 2026-09-17

### Objective

Evaluate GraphLIME explanation stability under controlled local active-feature
masking while keeping the trained GCN, target-node cohort, GraphLIME
configuration, and graph structure fixed.

### Frozen Experimental Configuration

- Dataset: Cora
- Model checkpoint: `models/cora_gcn_seed42.pt`
- Baseline explanation cohort: 97 nodes
- GNN retraining during perturbation: none
- GraphLIME neighborhood: 2-hop
- `TOP_K = 10`
- `RHO = 0.03`
- Perturbation type: local active-entry feature masking
- Post-mask feature renormalization: none
- Graph perturbation: none
- Nominal masking rates: 0.01, 0.05, 0.10, 0.20
- Perturbation seeds: 101–110
- Perturbations per node: 40
- Planned observations: 3,880

Each perturbation starts independently from the original feature matrix rather
than being applied cumulatively across masking rates.

The frozen GCN is not retrained.

### Data Collection

Completed observations:

```text
3,880 / 3,880
```

The full experimental grid is:

```text
97 target nodes * 4 masking rates * 10 perturbation seeds
= 3,880 observations
```

Coverage:

- Target nodes: 97
- Masking rates: 4
- Perturbation seeds: 10
- Observations per masking rate: 970
- Observations per perturbation seed: 388
- Observations per target node: 40
- Duplicate node-rate-seed observations: 0

### Integrity Validation

Validation script:

`experiments/03b_validate_perturbation_results.py`

Results:

- Row-level validation: 3,880/3,880 PASS
- Aggregate validation checks: 46/46 PASS
- Duplicate observation keys: 0
- Missing node-rate-seed combinations: 0
- Mask-count arithmetic mismatches: 0
- Realized-rate arithmetic mismatches: 0
- Prediction bookkeeping mismatches: 0
- Jaccard arithmetic mismatches: 0
- Invalid top-10 Jaccard values: 0

The perturbation cohort exactly matches both the frozen 97-node baseline
explanation cohort and the 97-node KKT-validated cohort.

### Perturbed Explanation Availability

Valid perturbed GraphLIME explanations:

- 3,821/3,880 observations
- approximately 98.48%

Unavailable perturbed explanations:

- 59/3,880 observations
- approximately 1.52%

Recorded failure reasons:

- Fewer than `TOP_K` nonzero HSIC-Lasso coefficients: 35
- Optimizer non-convergence: 24

The failure counts account for all 59 unavailable explanations.

Unavailable explanations are retained as experimental outcomes.

Their Jaccard similarity, intersection size, and union size remain missing.
Unavailable explanations are not assigned `Jaccard = 0` and are not removed
from the raw experiment.

### Frozen Artifacts

- `results/perturbations/feature_mask.csv`
- `results/perturbations/feature_mask_validation.csv`
- `results/perturbations/feature_mask_validation_summary.csv`

These artifacts are frozen and are not manually modified.

### Phase-9 Decision

Canonical Phase 9 — controlled feature perturbation and integrity validation:

**COMPLETE.**

### Status

**PASSED, COMPLETE, AND FROZEN.**

---

## E009 — Statistical and Hypothesis-Oriented Analysis

**Date:** 2026-09-17

### Objective

Aggregate the frozen Phase-9 perturbation results at the target-node level,
perform paired hypothesis-oriented analyses across masking rates, and finalize
evidence for the five pre-specified hypotheses without modifying the frozen
experimental data or retuning any model, explainer, perturbation, or
statistical parameter.

### Frozen Experimental Basis

- Dataset: Cora
- Frozen model: two-layer GCN
- Baseline explanation cohort: 97 nodes
- GraphLIME neighborhood: 2-hop
- `TOP_K = 10`
- `RHO = 0.03`
- Feature perturbation: local active-entry masking
- Masking rates: 0.01, 0.05, 0.10, 0.20
- Perturbation seeds: 101–110
- Raw observations: 3,880
- Available perturbed explanations: 3,821
- Unavailable perturbed explanations: 59
- Missing-explanation policy: unavailable explanations remain missing;
  Jaccard is never assigned zero

### Statistical Design

The target node is the primary across-rate statistical unit.

The ten perturbation seeds are treated as repeated stochastic observations
within each node-rate combination.

Methods:

- Node-rate aggregation before across-rate comparison
- Same target nodes evaluated across all masking rates
- 95% paired node-level bootstrap confidence intervals
- 10,000 bootstrap resamples
- Bootstrap seed: 20260917
- Secondary two-sided Wilcoxon signed-rank analysis
- Local NumPy/Python Wilcoxon implementation
- Tie-corrected normal approximation
- No continuity correction
- Holm correction within each six-contrast outcome family
- Matched-pairs rank-biserial correlation as effect size

H4 correlation analysis is treated as descriptive because the target nodes
belong to a single graph rather than representing independent graph-level
samples.

### Four-Rate Descriptive Results

| Mask rate | Prediction stability | Explanation availability | Mean Jaccard | Mean Jaccard given prediction unchanged |
|---:|---:|---:|---:|---:|
| 0.01 | 0.9969 | 0.9876 | 0.9015 | 0.9014 |
| 0.05 | 0.9876 | 0.9835 | 0.7253 | 0.7266 |
| 0.10 | 0.9835 | 0.9825 | 0.6045 | 0.6063 |
| 0.20 | 0.9588 | 0.9856 | 0.4541 | 0.4560 |

### H1 — Explanation Overlap Declines with Masking Severity

For the 1% to 20% endpoint comparison:

- Mean paired Jaccard difference: -0.4475
- 95% paired bootstrap CI: `[-0.4660, -0.4282]`
- Holm-adjusted Wilcoxon p-value: `7.302144e-17`
- Rank-biserial correlation: -1.0000
- Fraction of complete node trajectories monotonically non-increasing: 0.9375

Within the frozen experiment, GraphLIME top-10 feature overlap shows a large
and highly consistent decline as local active-feature masking increases.

### H2 — Explanation Changes Despite Preserved Predicted Class

Conditioning on observations for which the predicted class remains unchanged:

- Endpoint mean paired Jaccard difference: -0.4454
- 95% paired bootstrap CI: `[-0.4637, -0.4269]`
- Holm-adjusted Wilcoxon p-value: `7.304334e-17`
- Rank-biserial correlation: -1.0000
- Fraction of complete trajectories monotonically non-increasing: 0.9271

Explanation overlap therefore declines substantially even among perturbations
that preserve the predicted class.

This provides direct evidence within the frozen experiment that prediction
stability and explanation stability are distinct quantities.

### H3 — Prediction Agreement Declines with Masking Severity

For the 1% to 20% endpoint comparison:

- Mean paired prediction-stability difference: -0.0381
- 95% paired bootstrap CI: `[-0.0649, -0.0155]`
- Holm-adjusted Wilcoxon p-value: `1.712077e-02`
- Fraction of complete trajectories monotonically non-increasing: 0.9485

Prediction agreement decreases with increasing masking severity, although the
observed magnitude of the decline is substantially smaller than the decline
in explanation overlap.

### H4 — Prediction and Explanation Stability Are Not Perfectly Coupled

Endpoint change analysis compares:

```text
20% masking - 1% masking
```

Results:

- Valid endpoint nodes: 97/97
- Mean prediction-stability change: -0.0381
- Mean explanation-stability change: -0.4475
- Spearman correlation between endpoint changes: -0.0271
- Pearson correlation between endpoint changes: 0.0927
- Nodes with unchanged endpoint prediction-stability proportion but decreased
  explanation overlap: 86/97 (88.66%)

The endpoint changes provide descriptive evidence that prediction stability
and GraphLIME explanation stability do not move together deterministically in
this experiment.

The correlation quantities are treated descriptively. No correlation
p-value is reported.

### H5 — Explanation Unavailability May Increase with Masking Severity

Unavailable explanation counts:

| Mask rate | Unavailable | Total | Unavailability rate |
|---:|---:|---:|---:|
| 0.01 | 12 | 970 | 0.0124 |
| 0.05 | 16 | 970 | 0.0165 |
| 0.10 | 17 | 970 | 0.0175 |
| 0.20 | 14 | 970 | 0.0144 |

The sequence is not monotonically increasing.

Explanation availability remains high across all four perturbation rates.
The observed data therefore do not show a monotonic increase in explanation
unavailability with masking severity.

This non-monotonic result is retained rather than being modified to support
the pre-specified hypothesis.

### Phase-10 Validation

- Node-rate rows: 388/388 PASS
- Phase-9 explanation accounting: 3,821 available / 59 unavailable PASS
- H5 rate-level counts reproduced: PASS
- H4 endpoint nodes: 97/97 PASS
- H1/H2/H3 endpoint results reproduced: PASS
- Final pre-specified H1–H5 table: 5/5 PASS
- Pre-specified H3 restored correctly: PASS
- Overall Phase-10 final validation: PASS

### Frozen Phase-10 Artifacts

- `results/analysis/rate_summary.csv`
- `results/analysis/node_rate_summary.csv`
- `results/analysis/prediction_conditioned_summary.csv`
- `results/analysis/explanation_availability_summary.csv`
- `results/analysis/failure_summary.csv`
- `results/analysis/paired_rate_contrasts.csv`
- `results/analysis/monotonicity_summary.csv`
- `results/analysis/final_hypothesis_summary.csv`
- `results/analysis/h4_coupling_summary.csv`
- `results/analysis/h4_node_endpoint_changes.csv`
- `results/analysis/h5_availability_by_rate.csv`

### Phase-10 Decision

Canonical Phase 10 — statistical and hypothesis-oriented analysis:

**COMPLETE.**

No further parameter tuning or modification of the frozen experimental
results is warranted.

### Status

**PASSED, COMPLETE, AND FROZEN.**

---

## E010 — Phase-11 Research Visualization and Figure Freeze

**Date:** 2026-09-17

### Objective

Convert the frozen Phase-10 statistical results into research figures without
retraining the model, rerunning GraphLIME, regenerating perturbations, or
recomputing statistical inference.

Visualization script:

`experiments/05_plot_stability.py`

### Visualization Principle

Phase 11 is a presentation stage, not a new inferential experiment.

All figures are generated programmatically from frozen analysis artifacts.

No model parameter, GraphLIME parameter, perturbation setting, hypothesis,
confidence interval, effect size, or statistical test is modified to improve
the visual appearance of the results.

### Frozen Inputs

The visualization stage consumes frozen Phase-10 artifacts, including:

- `results/analysis/rate_summary.csv`
- `results/analysis/node_rate_summary.csv`
- `results/analysis/prediction_conditioned_summary.csv`
- `results/analysis/explanation_availability_summary.csv`
- `results/analysis/h4_node_endpoint_changes.csv`
- `results/analysis/h4_coupling_summary.csv`
- `results/analysis/h5_availability_by_rate.csv`

No numerical result is manually transcribed into a figure.

### Plot-Data Validation

Before figure generation, the visualization script validates the frozen input
artifacts.

Validated quantities include:

- Rate-level rows: 4/4 PASS
- Node-rate rows: 388/388 PASS
- H4 endpoint nodes: 97/97 PASS
- H4 coupling summary: 1/1 PASS
- H5 rate rows: 4/4 PASS
- Frozen endpoint results reproduced: PASS
- H4 frozen quantities reproduced: PASS
- H5 frozen quantities reproduced: PASS

Overall plot-data validation:

**PASS**

### Frozen Quantities Visualized

| Mask rate | Prediction stability | Explanation availability | Mean Jaccard | Mean Jaccard given prediction unchanged |
|---:|---:|---:|---:|---:|
| 0.01 | 0.9969 | 0.9876 | 0.9015 | 0.9014 |
| 0.05 | 0.9876 | 0.9835 | 0.7253 | 0.7266 |
| 0.10 | 0.9835 | 0.9825 | 0.6045 | 0.6063 |
| 0.20 | 0.9588 | 0.9856 | 0.4541 | 0.4560 |

H4 quantities reproduced during visualization:

- Spearman rho: -0.0271
- Pearson r: 0.0927
- Direct-decoupling nodes: 86/97

H5 unavailable-explanation counts:

```text
[12, 16, 17, 14]
```

### Figure 1 — Main Stability Figure

Artifacts:

- `results/figures/figure_01_main_stability.png`
- `results/figures/figure_01_main_stability.pdf`

Panels:

- A — GraphLIME explanation overlap
- B — Explanation overlap when prediction is unchanged
- C — Prediction stability
- D — Explanation availability

Points show means across target nodes.

Error bars show the frozen 95% node-level bootstrap confidence intervals
computed during Phase 10.

The four-panel design separates explanation overlap, prediction-conditioned
explanation overlap, prediction stability, and explanation availability rather
than implying that they represent the same quantity.

### Figure 2 — Prediction and Explanation Stability

Artifacts:

- `results/figures/figure_02_prediction_vs_explanation.png`
- `results/figures/figure_02_prediction_vs_explanation.pdf`

The figure places prediction stability and mean GraphLIME Jaccard overlap on
their shared numerical `[0, 1]` scale while explicitly noting that they
represent different notions of stability.

The figure visually contrasts the comparatively small decline in predicted
class agreement with the substantially larger decline in explanation overlap.

It is intended as a descriptive comparison rather than as a mathematical
equivalence between the two metrics.

### Figure 3 — H4 Endpoint Coupling

Artifacts:

- `results/figures/figure_03_h4_endpoint_coupling.png`
- `results/figures/figure_03_h4_endpoint_coupling.pdf`

The figure displays node-level endpoint changes between 1% and 20% masking.

Axes:

- x-axis: change in prediction stability, `20% - 1%`
- y-axis: change in mean Jaccard similarity, `20% - 1%`

Exact endpoint coordinates are retained without artificial jitter.

The dense vertical concentration at zero prediction-stability change reflects
the discrete structure of the observed data rather than a plotting artifact.

Nodes satisfying the direct-decoupling pattern—unchanged endpoint prediction
stability with decreased explanation overlap—are visually distinguished using
open markers.

The displayed Pearson and Spearman correlations are descriptive quantities
from the frozen Phase-10 analysis.

### Figure 4 — Explanation Unavailability

Artifacts:

- `results/figures/figure_04_explanation_unavailability.png`
- `results/figures/figure_04_explanation_unavailability.pdf`

The figure displays the number and proportion of unavailable GraphLIME
explanations at each masking rate.

Observed unavailable counts:

- 1%: 12/970
- 5%: 16/970
- 10%: 17/970
- 20%: 14/970

No fitted trend is imposed because the observed sequence is non-monotonic.

### Visual Inspection

The generated PNG figures were manually inspected after rendering.

The final versions were checked for:

- text clipping;
- overlapping annotations;
- legend placement;
- axis readability;
- confidence-interval visibility;
- point occlusion;
- misleading graphical transformations; and
- consistency with the frozen Phase-10 quantities.

An initial Figure-2 legend/text overlap was corrected during the visualization
stage.

Figure 3 retained exact discrete endpoint coordinates rather than introducing
jitter. Marker size, transparency, and styling were adjusted only to improve
readability and did not alter the underlying observations.

The final four figures were judged visually interpretable without modifying
their underlying data or statistical quantities.

### Output Validation

Expected figure artifacts:

- 4 PNG files
- 4 PDF files
- 8 total files

Generated figure artifacts:

```text
8 / 8
```

Output validation:

**PASS**

### Figure Freeze Decision

The final Phase-11 figures are frozen as the primary visualization artifacts
for the completed study.

No Phase-9 or Phase-10 data were modified during figure development.

No statistical quantity was recomputed to improve figure appearance.

No GCN, GraphLIME, perturbation, or inferential parameter was changed.

### Phase-11 Decision

Canonical Phase 11 — research visualization and figure generation:

**COMPLETE.**

### Status

**PASSED, COMPLETE, AND FROZEN.**

---

# Final Research-Pipeline Status

The canonical Phase 0–11 experimental pipeline is complete.

| Phase | Description | Status |
|---:|---|:---:|
| 0 | Research repository | COMPLETE |
| 1 | Python environment | COMPLETE |
| 2 | Cora loading and validation | COMPLETE |
| 3 | Baseline GNN implementation | COMPLETE |
| 4 | Reproducibility controls | COMPLETE |
| 5 | Baseline GNN training and freeze | COMPLETE |
| 6 | Baseline recording | COMPLETE |
| 7 | Fixed explanation-node selection | COMPLETE |
| 8 | GraphLIME implementation, calibration, baseline generation, and numerical validation | COMPLETE |
| 9 | Controlled feature-perturbation experiment and integrity validation | COMPLETE |
| 10 | Statistical and hypothesis-oriented analysis | COMPLETE |
| 11 | Research visualization and figure freeze | COMPLETE |

---

# Frozen Study Scope

The completed evidence applies specifically to the experimental configuration
documented in this log:

- Cora dataset;
- standard Planetoid split;
- normalized Cora node features;
- frozen two-layer GCN baseline;
- seed 42 baseline model;
- predetermined 100-node correctly classified test-node sample;
- 97-node baseline-explainable analysis cohort;
- GraphLIME using a 2-hop local neighborhood;
- `TOP_K = 10`;
- `RHO = 0.03`;
- Gaussian feature and output kernels;
- median positive pairwise-distance bandwidth heuristic;
- exclusion of locally degenerate feature kernels;
- objective-faithful non-negative proximal-gradient HSIC-Lasso solver;
- local active-entry feature masking;
- no post-mask feature renormalization;
- no GNN retraining during perturbation;
- unchanged graph structure;
- masking rates of 1%, 5%, 10%, and 20%;
- ten perturbation seeds per node-rate combination;
- 3,880 total perturbation observations;
- missing rather than zero Jaccard for unavailable explanations; and
- the statistical procedures documented in E009.

The results should not be generalized automatically to other datasets,
architectures, explanation methods, perturbation operators, GraphLIME
hyperparameters, optimization procedures, or graph-learning tasks.

---

# Central Empirical Findings

Within the frozen experimental setting, increasing local active-feature
masking is associated with a substantial decline in GraphLIME top-10 feature
overlap.

Mean Jaccard similarity decreases from:

```text
0.9015 at 1% masking
```

to:

```text
0.4541 at 20% masking
```

corresponding to a paired endpoint difference of:

```text
-0.4475
```

with frozen 95% paired node-level bootstrap confidence interval:

```text
[-0.4660, -0.4282]
```

Over the same endpoint comparison, prediction stability changes by only:

```text
-0.0381
```

from a mean of 0.9969 at 1% masking to 0.9588 at 20% masking.

The explanation-overlap decline remains similar when conditioning on
observations for which the predicted class is unchanged:

```text
endpoint conditional Jaccard difference = -0.4454
```

with frozen 95% paired bootstrap confidence interval:

```text
[-0.4637, -0.4269]
```

At the node-level endpoint comparison, 86 of 97 nodes exhibit unchanged
prediction-stability proportion while explanation overlap decreases.

Explanation availability remains high across all masking rates:

```text
1%  -> 0.9876
5%  -> 0.9835
10% -> 0.9825
20% -> 0.9856
```

The corresponding unavailable-explanation counts are:

```text
12, 16, 17, 14
```

and therefore do not form a monotonic increasing sequence.

---

# Interpretation Boundary

The completed experiment supports a specific empirical conclusion:

> Under the frozen Cora–GCN–GraphLIME configuration, GraphLIME top-10 feature
> overlap is substantially more sensitive to controlled local active-feature
> masking than predicted-class agreement.

The experiment also shows that substantial explanation changes can occur while
the predicted class remains unchanged.

These observations demonstrate an empirical distinction between prediction
stability and explanation stability within the frozen experimental setting.

They do **not** establish:

- a universal instability property of GraphLIME;
- causal relationships between individual input features and predictions;
- equivalent behavior on other graph datasets;
- equivalent behavior for other GNN architectures;
- equivalent behavior for structural graph perturbations;
- equivalent behavior for other explanation methods;
- equivalent behavior under different GraphLIME hyperparameters; or
- population-level independence between prediction and explanation stability.

The H4 Pearson and Spearman correlations are descriptive node-level summaries
within one graph and are not treated as independent graph-level population
inference.

---

# Known Methodological Limitations

The completed study intentionally uses a restricted initial experimental scope.

First, the experiment uses one dataset, Cora. Replication on additional graph
datasets would be required to evaluate whether the observed stability pattern
generalizes beyond this setting.

Second, the frozen predictive model is a two-layer GCN. Although GraphLIME is
model-agnostic in its local explanation construction, the present experiment
does not establish equivalent behavior for GraphSAGE, GAT, or other GNN
architectures.

Third, the initial perturbation study modifies node features only. Graph
structure remains unchanged. Structural perturbations such as controlled edge
deletion therefore remain a separate research question.

Fourth, the study uses one frozen GraphLIME configuration:

```text
NUM_HOPS = 2
TOP_K = 10
RHO = 0.03
```

The experiment does not establish invariance of the results to neighborhood
radius, explanation size, kernel-bandwidth choices, or HSIC-Lasso
regularization.

Fifth, the implementation solves the stated non-negative HSIC-Lasso objective
using proximal-gradient optimization rather than exactly reproducing the
non-negative LARS optimization procedure used in the original GraphLIME work.
The implemented solutions were independently subjected to reproduction and
KKT diagnostics, but this optimizer distinction should remain explicit in any
research report.

Sixth, the 97 target nodes belong to the same Cora graph. Node-level bootstrap
confidence intervals and paired analyses provide useful within-study
uncertainty summaries, but graph dependence limits interpretation as ordinary
IID population inference.

Seventh, active-feature masking is performed without post-mask
renormalization. This is intentional because renormalization would modify
unselected feature values, but it means the perturbation also changes the
affected rows' normalized feature mass.

Finally, masks at different perturbation rates are independently generated
rather than constructed as nested prefixes. Across-rate comparisons are
therefore paired by target node, not by identical nested feature-mask sets.

---

# Reproducibility and Freeze Policy

The study follows a freeze-before-analysis workflow.

The following classes of decisions were fixed before downstream interpretation:

1. baseline model selection;
2. target-node selection;
3. GraphLIME neighborhood size;
4. explanation size;
5. HSIC-Lasso regularization;
6. optimizer settings;
7. baseline explanation cohort definition;
8. perturbation operator;
9. perturbation rates;
10. perturbation seeds;
11. missing-explanation treatment; and
12. primary statistical unit and analysis procedures.

Unexpected or unfavorable observations were retained rather than silently
removed.

Examples include:

- the three baseline output-degenerate nodes;
- the non-converged `rho = 0.010` calibration case;
- the 59 unavailable perturbed explanations;
- the 24 perturbation optimizer non-convergence cases;
- the 35 perturbation cases with fewer than ten nonzero coefficients; and
- the non-monotonic explanation-unavailability pattern.

Unavailable explanations were not converted to zero-overlap observations.

The frozen raw and derived artifacts should not be manually modified.

---

# Research Freeze

The canonical **Phase 0–11 experimental pipeline is complete**.

The completed artifacts should now be treated as a frozen empirical record.

Subsequent research writing should report and critically interpret these
results rather than retuning the experiment to produce a preferred narrative.

Any future extension—such as:

- PubMed or another graph dataset;
- GraphSAGE or GAT;
- structural edge perturbations;
- alternative explanation methods;
- `TOP_K` sensitivity;
- neighborhood-radius sensitivity;
- HSIC-Lasso regularization sensitivity;
- kernel-bandwidth sensitivity; or
- alternative explanation-stability metrics

should be defined and logged as a **new follow-up experiment** rather than
silently incorporated into the frozen Phase 0–11 study.

The current experimental record is therefore:

**COMPLETE, VALIDATED, ANALYZED, VISUALIZED, AND FROZEN.**
