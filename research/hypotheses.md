# Research Questions and Hypotheses

Date frozen: 2026-09-17

## Study

Stability of Local Feature Explanations for Graph Neural Networks under
Controlled Feature Perturbations.

## Primary Research Question

How stable are GraphLIME local feature explanations when the observed
node features within the explained node's local graph neighbourhood are
subjected to controlled perturbations?

The study distinguishes two forms of stability:

1. prediction stability; and
2. explanation stability.

These quantities are not assumed to be equivalent.

A prediction may remain unchanged while the set of features identified
by GraphLIME changes substantially.

## Experimental Setting

Dataset:

```
Cora
```

Frozen predictive model:

```
two-layer GCN
```

Frozen model checkpoint:

```
models/cora_gcn_seed42.pt
```

Baseline test accuracy:

```
0.8060
```

GraphLIME locality:

```
2-hop neighbourhood
```

Baseline explanation size:

```
TOP_K = 10
```

Frozen HSIC-Lasso regularization:

```
RHO = 0.03
```

Baseline selected sample:

```
100 predetermined correctly classified test nodes
```

Baseline-explainable cohort:

```
97 nodes
```

Baseline output-degenerate nodes:

```
2697
2625
2411
```

The three output-degenerate nodes are retained in the baseline
diagnostic record but are not part of analyses requiring a valid
baseline explanation E_0(v).

## Notation

For target node v, let:

```
X
```

denote the original Cora feature matrix.

Let:

```
X_(v,r,s)
```

denote a perturbed copy of X for target node v, perturbation severity r,
and perturbation seed s.

Let the frozen GNN be:

```
f_theta
```

where theta remains unchanged throughout the initial perturbation
experiment.

The original predicted class is:

```
y_hat_0(v)
```

and the perturbed predicted class is:

```
y_hat_(r,s)(v).
```

Let the original GraphLIME top-K explanation be:

```
E_0(v)
```

and the corresponding perturbed explanation be:

```
E_(r,s)(v).
```

For valid top-K explanations, explanation overlap is measured using
Jaccard similarity:

```
J(E_0, E_(r,s))
    =
|E_0 intersection E_(r,s)|
/
|E_0 union E_(r,s)|.
```

For two identical top-10 explanation sets:

```
J = 1.
```

For two disjoint valid top-10 explanation sets:

```
J = 0.
```

An unavailable explanation is not assigned J = 0 because an undefined
explanation and two valid non-overlapping explanations represent
different experimental outcomes.

## RQ1 — Perturbation Severity and Explanation Stability

How does GraphLIME explanation stability change as the severity of
controlled feature masking increases?

### H1

GraphLIME top-10 explanation overlap will tend to decrease as the
feature-masking rate increases.

Formally, the expected trend is:

```
increasing perturbation severity
    ->
decreasing mean Jaccard similarity.
```

This is a directional research hypothesis rather than an assumption
that every individual node or perturbation seed will exhibit a
monotonic trajectory.

## RQ2 — Prediction Stability versus Explanation Stability

Can GraphLIME explanations change substantially even when the frozen
GNN's predicted class remains unchanged?

### H2

A non-trivial subset of perturbation observations will preserve the
target node's predicted class while producing a changed GraphLIME
top-10 explanation.

The phenomenon of particular interest is:

```
y_hat_0(v) = y_hat_(r,s)(v)
```

while:

```
E_0(v) != E_(r,s)(v).
```

The magnitude of explanation change will be quantified rather than
classified using an arbitrary post-hoc threshold.

The primary analysis will therefore examine the distribution of
Jaccard similarity conditional on:

```
pred_same = True.
```

## RQ3 — Perturbation Severity and Prediction Stability

How does controlled local feature masking affect the frozen GNN's
predictions?

### H3

Prediction agreement will tend to decrease as the feature-masking rate
increases.

Prediction stability will be measured using:

```
pred_same =
    1 if perturbed predicted class equals baseline predicted class
    0 otherwise.
```

Changes in target-class confidence will also be retained as a
continuous measure.

## RQ4 — Relationship between Prediction and Explanation Stability

Does explanation stability closely track prediction stability?

### H4

Prediction stability and explanation stability will not be perfectly
coupled.

In particular, explanation Jaccard similarity is expected to show
meaningful variation among observations for which:

```
pred_same = True.
```

This hypothesis is central to the study because identical predicted
classes do not necessarily imply identical local feature-dependence
patterns.

## RQ5 — Explanation Availability under Perturbation

Does controlled feature masking cause GraphLIME explanations that are
valid at baseline to become unavailable under perturbation?

### H5

The rate of unavailable perturbed explanations will be low at small
perturbation severities but may increase as perturbation severity
increases.

Possible causes include:

* locally degenerate GNN output vectors;
* insufficient nonzero HSIC-Lasso coefficients for a complete top-10
  explanation;
* optimizer non-convergence; or
* another explicitly recorded numerical failure.

Explanation unavailability will be reported as an outcome in its own
right and will not be converted to Jaccard similarity zero.

## Primary Outcomes

The primary perturbation outcomes are:

1. predicted-class agreement;
2. target-class confidence change;
3. top-10 GraphLIME Jaccard similarity when both explanations exist;
4. perturbed explanation availability.

## Secondary Outcomes

Secondary diagnostic outcomes include:

* number of active feature entries before perturbation;
* number of masked entries;
* realized masking fraction;
* perturbed GraphLIME active-feature count;
* perturbed HSIC-Lasso nonzero-coefficient count;
* optimizer convergence;
* optimizer iterations; and
* failure reason when a complete perturbed explanation is unavailable.

## Initial Perturbation Severities

The initial feature-masking rates are frozen as:

```
0.01
0.05
0.10
0.20
```

These correspond nominally to:

```
1%
5%
10%
20%
```

of active feature entries within the target node's frozen 2-hop
neighbourhood.

The exact integer masking rule and realized masking fraction are
defined in `research/perturbation_protocol.md`.

## Perturbation Repetitions

Ten predetermined perturbation seeds will be used for each
node-rate combination:

```
101
102
103
104
105
106
107
108
109
110
```

For 97 baseline-explainable nodes, four perturbation rates, and ten
seeds, the planned experiment contains:

```
97 * 4 * 10 = 3880
```

perturbation observations.

## Interpretation Constraints

The study will not infer explanation instability merely because the
prediction changes.

Prediction change and explanation change will be reported separately.

The study will not assign zero Jaccard similarity to undefined
perturbed explanations.

The study will not replace difficult nodes or failed perturbation runs
with more convenient observations.

The study will not retune RHO, NUM_HOPS, TOP_K, the GCN checkpoint, or
the baseline node cohort using perturbation outcomes.

The initial study concerns feature perturbation only.

Structural edge perturbation, alternative GraphLIME neighbourhood
sizes, alternative TOP_K values, additional GNN seeds, and additional
datasets are reserved for later robustness or ablation experiments
after the primary experiment is analysed.

## Status

Research questions and initial hypotheses:

FROZEN BEFORE PRIMARY PERTURBATION RESULTS.

Next methodological artifact:

`research/perturbation_protocol.md`
