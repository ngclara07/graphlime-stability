# Research Question

## Study Title

**Stability of Local Feature Explanations for Graph Neural Networks under Controlled Perturbations**

## Research Motivation

Local explanation methods for graph neural networks are intended to identify
which input features are important for a model's prediction around a target
node.

A useful explanation, however, should be considered not only in terms of the
features it selects, but also in terms of how reliably those selections behave
under small, controlled changes to the input.

In particular, prediction stability and explanation stability are distinct
properties. A graph neural network may preserve its predicted class after an
input perturbation while the explanation of that prediction changes
substantially.

This study investigates that distinction using GraphLIME explanations for a
frozen graph convolutional network under controlled local feature masking.

---

## Primary Research Question

**How stable are GraphLIME local feature explanations for a frozen graph neural
network under controlled perturbations of the input node features?**

The study focuses specifically on whether increasing local feature
perturbation changes the features identified by GraphLIME and whether such
changes can occur even when the model's predicted class remains unchanged.

---

## Central Research Problem

For a target node \(v\), let:

- \(f(v)\) denote the prediction produced by the original frozen GNN;
- \(f_r(v)\) denote the prediction after a perturbation at masking rate \(r\);
- \(E_0(v)\) denote the baseline GraphLIME top-\(K\) explanation; and
- \(E_r(v)\) denote the GraphLIME top-\(K\) explanation after perturbation.

Prediction stability and explanation stability need not be equivalent.

The phenomenon of interest is therefore:

\[
f(v) = f_r(v)
\]

while simultaneously:

\[
E_0(v) \neq E_r(v).
\]

More generally, the study asks whether explanation similarity can decline
substantially while predicted-class agreement remains comparatively stable.

---

## Operational Research Questions

The primary question is decomposed into four experimentally measurable
questions.

### RQ1 — Explanation Stability

How does GraphLIME top-10 feature overlap change as the local active-feature
masking rate increases?

### RQ2 — Explanation Stability Conditional on Prediction Stability

When the perturbed input preserves the original predicted class, how does
GraphLIME top-10 feature overlap change as masking severity increases?

### RQ3 — Prediction Stability

How does predicted-class agreement change as the local active-feature masking
rate increases?

### RQ4 — Explanation Availability

How frequently does GraphLIME fail to provide a valid complete top-10
explanation under increasing feature-masking severity?

Together, these questions allow prediction behavior, explanation behavior, and
explanation availability to be analyzed separately.

---

## Explanation-Stability Measure

Explanation stability is measured using top-\(K\) Jaccard similarity.

For the frozen study:

\[
K = 10.
\]

For target node \(v\) and masking rate \(r\):

\[
J(E_0(v), E_r(v))
=
\frac{|E_0(v)\cap E_r(v)|}
{|E_0(v)\cup E_r(v)|}.
\]

Interpretation:

- \(J = 1\): the baseline and perturbed top-10 feature sets are identical;
- \(J = 0\): the two valid top-10 feature sets have no features in common;
- intermediate values indicate partial feature-set overlap.

If a perturbed GraphLIME explanation is unavailable, Jaccard similarity is
treated as missing rather than artificially assigned a value of zero.

This distinction separates explanation unavailability from genuine
zero-overlap between two valid explanations.

---

## Prediction-Stability Measure

Prediction stability is evaluated using predicted-class agreement between the
original and perturbed inputs.

For an individual perturbation:

\[
A_r(v)
=
\mathbf{1}
\left[
\hat{y}_0(v)=\hat{y}_r(v)
\right],
\]

where:

- \(\hat{y}_0(v)\) is the original predicted class; and
- \(\hat{y}_r(v)\) is the predicted class after perturbation.

Node-level prediction stability at each masking rate is obtained from repeated
perturbations of the same target node.

Prediction stability is analyzed separately from explanation overlap.

---

## Pre-Specified Hypotheses

The study evaluates the following hypotheses.

### H1 — Explanation Overlap

GraphLIME top-10 Jaccard similarity tends to decrease as local active-feature
masking severity increases.

### H2 — Explanation Change despite Preserved Prediction

A non-trivial subset of perturbations preserves the predicted class while
changing the GraphLIME top-10 explanation.

The primary conditional explanation analysis therefore considers observations
for which the predicted class remains unchanged.

### H3 — Prediction Agreement

Predicted-class agreement tends to decrease as local active-feature masking
severity increases.

### H4 — Prediction–Explanation Coupling

Prediction stability and explanation stability are not perfectly coupled.

In particular, explanation overlap may decrease substantially for nodes whose
prediction-stability behavior changes little or remains unchanged across the
endpoint masking rates.

### H5 — Explanation Availability

GraphLIME explanation unavailability may increase as perturbation severity
increases.

Unavailable explanations are recorded explicitly and are not assigned
Jaccard similarity equal to zero.

---

## Experimental Scope

The frozen initial study is restricted to the following configuration.

### Dataset

- Cora citation-network dataset
- Standard Planetoid train/validation/test split
- PyTorch Geometric feature normalization

### Predictive Model

- Two-layer Graph Convolutional Network
- Frozen trained checkpoint
- Training seed: 42
- No model retraining during perturbation experiments

### Explanation Method

- GraphLIME
- 2-hop local neighborhoods
- Gaussian input and output kernels
- Median positive pairwise-distance bandwidth heuristic
- HSIC-Lasso feature selection
- Non-negative proximal-gradient optimization of the implemented
  HSIC-Lasso objective
- `TOP_K = 10`
- `RHO = 0.03`

### Target Cohort

- 100 predetermined correctly classified Cora test nodes
- 97 nodes with valid baseline GraphLIME explanations
- The three baseline output-degenerate nodes are retained diagnostically and
  are not replaced

### Perturbation

- Local active-entry feature masking
- Only originally active/nonzero feature entries are eligible for masking
- Masking rates: 1%, 5%, 10%, and 20%
- Ten perturbation seeds per node-rate combination
- No post-mask feature renormalization
- No graph-structure perturbation
- No GNN retraining
- Each perturbation begins from the original feature matrix
- Masks at different rates are generated independently rather than as nested
  perturbation sequences

### Experimental Size

The primary explanation-stability experiment contains:

\[
97 \times 4 \times 10 = 3880
\]

node-rate-seed observations.

---

## Statistical Unit

The target node is the primary across-rate statistical unit.

Perturbation seeds are treated as repeated stochastic observations within each
node-rate combination rather than as independent target nodes.

Across-rate comparisons therefore operate on node-level summaries.

The frozen analysis uses:

- paired node-level comparisons;
- 95% paired node-level bootstrap confidence intervals;
- 10,000 bootstrap resamples;
- two-sided Wilcoxon signed-rank tests as secondary checks;
- Holm adjustment within the corresponding contrast families; and
- matched-pairs rank-biserial correlation as an effect-size measure.

Because all target nodes belong to a single graph, node-level dependence is an
important interpretation limitation. Correlations used to examine
prediction–explanation coupling are therefore treated descriptively rather
than as independent graph-level population inference.

---

## Controlled Variables

The following components remain fixed during the primary perturbation
experiment:

- trained GCN parameters;
- graph structure;
- target-node cohort;
- GraphLIME neighborhood radius;
- GraphLIME top-\(K\);
- HSIC-Lasso regularization;
- optimizer configuration; and
- baseline explanations.

The primary manipulated variable is the local active-feature masking rate.

This design isolates the empirical relationship between controlled feature
perturbation and explanation/prediction stability under the frozen
configuration.

---

## Interpretation Boundary

The study is designed to characterize explanation stability under one
controlled experimental setting.

It does **not** attempt to establish that:

- GraphLIME is universally unstable;
- explanation changes necessarily indicate prediction errors;
- selected features have a causal relationship with the prediction;
- the same behavior occurs for every GNN architecture;
- the same behavior occurs on every graph dataset;
- feature perturbations and structural perturbations behave equivalently; or
- the observed node-level results constitute independent graph-level
  population evidence.

The appropriate interpretation is narrower:

> The experiment evaluates how GraphLIME top-10 feature explanations and GCN
> predictions respond to increasing controlled local active-feature masking
> under a frozen Cora–GCN–GraphLIME configuration.

---

## Research Contribution of the Experiment

The study provides a controlled empirical framework for examining prediction
stability and explanation stability as separate quantities.

Its central comparison is not simply whether the GNN prediction changes after
perturbation, but whether the explanation can change substantially even when
prediction behavior remains comparatively stable.

The experimental design therefore supports investigation of the distinction:

\[
\text{prediction stability}
\neq
\text{explanation stability}.
\]

Any broader claims about GraphLIME, GNN explainability, or explanation
robustness require replication across additional datasets, models,
perturbation mechanisms, and explanation methods.
