# Controlled Feature Perturbation Protocol

Date frozen: 2026-09-17

## Purpose

This document defines the initial controlled feature-perturbation
experiment for the GraphLIME explanation-stability study.

The protocol is specified before examining the primary perturbation
results.

Its purpose is to prevent post-hoc changes to the intervention,
perturbation severity, sampling scheme, explanation eligibility rules,
or outcome definitions based on whether the observed results appear
favourable.

## Phase

Canonical project roadmap:

```
Phase 9 — Controlled Feature Perturbation
```

Phase 8 has been completed.

The frozen baseline explanation cohort contains:

```
97 nodes
```

with valid, numerically validated top-10 GraphLIME baseline
explanations.

## Fixed Baseline Components

The following components remain unchanged during the initial
perturbation experiment.

Dataset:

```
Cora
```

Graph structure:

```
unchanged
```

GCN parameters:

```
frozen
```

GCN checkpoint:

```
models/cora_gcn_seed42.pt
```

GraphLIME neighbourhood:

```
NUM_HOPS = 2
```

GraphLIME explanation size:

```
TOP_K = 10
```

HSIC-Lasso regularization:

```
RHO = 0.03
```

HSIC-Lasso maximum iterations:

```
MAX_ITER = 10000
```

HSIC-Lasso coefficient-change tolerance:

```
OPTIMIZER_TOLERANCE = 1e-8
```

HSIC-Lasso zero threshold:

```
ZERO_TOLERANCE = 1e-8
```

Baseline explanations:

```
results/baseline/explanations.csv
```

Baseline diagnostics:

```
results/baseline/explanation_diagnostics.csv
```

## Experimental Population

The original selected population contains 100 predetermined correctly
classified test nodes.

Three nodes did not admit a valid baseline GraphLIME output kernel:

```
2697
2625
2411
```

The primary feature-perturbation cohort therefore contains the 97
nodes with valid baseline top-10 explanations.

These 97 nodes are not resampled or replaced.

## Perturbation Type

The initial intervention is:

```
local active-entry feature masking.
```

For a target node v, let N_2(v) denote the same frozen 2-hop
neighbourhood used by the baseline GraphLIME explanation.

Let:

```
A_v =
    {(i,j) :
     i belongs to N_2(v)
     and X[i,j] != 0}
```

be the set of originally active feature entries within that
neighbourhood.

Only entries in A_v are eligible for masking.

Arbitrary zero-valued feature cells are not sampled.

This avoids an intervention in which a large proportion of nominally
"masked" entries were already zero.

## Perturbation Severities

The frozen nominal masking rates are:

```
0.01
0.05
0.10
0.20
```

For rate r and n_active = |A_v|, the number of entries masked is:

```
n_mask =
    max(
        1,
        floor(r * n_active + 0.5)
    )
```

for all positive perturbation rates.

This is round-half-up integer rounding with a minimum of one masked
entry.

The minimum-one rule prevents nominal positive perturbations from
becoming no-op interventions in very small local neighbourhoods.

Because integer rounding can make the realized fraction differ from
the nominal rate, every experimental observation must record:

```
realized_mask_rate =
    n_mask / n_active.
```

Analyses may therefore inspect both nominal and realized severity.

## Perturbation Seeds

The frozen perturbation seed identifiers are:

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

Each node-rate-seed combination receives a deterministic derived
sampling seed.

The derived seed depends on:

* target node ID;
* nominal perturbation rate index; and
* perturbation seed identifier.

This prevents different node-rate combinations from mechanically
reusing the same permutation while retaining complete reproducibility.

## Sampling Rule

For each node-rate-seed observation:

1. Start from the original frozen Cora feature matrix X.
2. Reconstruct the target node's frozen 2-hop neighbourhood.
3. Identify all originally nonzero feature entries in that
   neighbourhood.
4. Determine n_mask using the frozen integer rule.
5. Sample n_mask eligible entries uniformly without replacement.
6. Clone X.
7. Set only the sampled entries in the clone to zero.
8. Leave all other feature entries unchanged.
9. Leave all graph edges unchanged.
10. Do not modify the original X tensor.

Every observation therefore represents an independent perturbation of
the same original feature matrix rather than cumulative corruption.

## No Post-Masking Renormalization

The perturbed feature matrix is not row-renormalized after masking.

Reason:

Row renormalization would change values of feature entries that were
not selected for intervention.

The initial experiment is intended to remove selected observed feature
information while leaving all unselected entries numerically
unchanged.

Therefore:

```
sampled active value -> 0
```

and:

```
every unselected feature value -> unchanged.
```

## Frozen Model, Recomputed Predictions

The GCN parameters remain frozen.

The GCN is not retrained for any perturbation observation.

However, GNN predictions are recomputed using the perturbed feature
matrix.

For each perturbation:

```
P_(v,r,s)
    =
softmax(
    f_theta(
        X_(v,r,s),
        A
    )
)
```

where theta and A remain fixed.

Thus:

```
frozen model != frozen predictions.
```

This distinction is required because the experiment measures how a
fixed learned model responds to changes in its observed input
features.

## Perturbed GraphLIME Explanation

For each perturbation observation:

1. use the same target node;
2. use the same 2-hop graph neighbourhood;
3. use the perturbed local feature matrix;
4. use GNN probability outputs recomputed from the perturbed full
   feature matrix;
5. reconstruct GraphLIME feature and output kernels;
6. solve HSIC-Lasso using the frozen rho and optimizer settings; and
7. select the perturbed top-10 features if at least ten nonzero
   coefficients are available and optimization converges.

No GraphLIME hyperparameter is retuned per observation.

## Prediction Outcomes

For target node v:

```
baseline_predicted_label
```

is taken from the frozen unperturbed model.

The perturbed predicted label is recomputed after masking.

Prediction agreement is:

```
pred_same =
    baseline_predicted_label
    ==
    perturbed_predicted_label.
```

Baseline confidence is the baseline probability assigned to the
baseline predicted class.

Perturbed baseline-class confidence is the perturbed probability
assigned to that same baseline class.

The primary continuous confidence change is:

```
baseline_class_confidence_delta =
    perturbed probability of baseline class
    -
    baseline probability of baseline class.
```

A negative value indicates reduced support for the original predicted
class.

For completeness, the maximum perturbed class probability may also be
recorded separately.

## Explanation-Stability Outcome

For observations with both a valid baseline top-10 explanation and a
valid perturbed top-10 explanation:

```
Jaccard =
    |E_0 intersection E_p|
    /
    |E_0 union E_p|.
```

Because both sets contain ten unique feature IDs:

```
Jaccard = 1
```

means identical explanation sets.

```
Jaccard = 0
```

means two valid explanation sets have no feature in common.

Ranking is not used in the primary Jaccard measure.

Rank-sensitive measures may be introduced later as secondary analyses,
but not as replacements for the pre-specified primary metric.

## Perturbed Explanation Failure Handling

A perturbed explanation may be unavailable because:

1. the perturbed local GNN outputs are output-degenerate;
2. no active feature kernels remain;
3. HSIC-Lasso does not converge;
4. fewer than TOP_K nonzero coefficients remain;
5. a numerical error occurs; or
6. another explicitly recorded failure is encountered.

Such observations remain in the raw perturbation dataset.

They are not deleted.

They are not replaced.

They are not assigned:

```
Jaccard = 0.
```

Instead:

```
explanation_available = False
jaccard = missing
```

and an explicit failure reason is stored.

This preserves the distinction between explanation instability and
explanation unavailability.

## Planned Experimental Size

Nodes:

```
97
```

Perturbation rates:

```
4
```

Seeds per node-rate combination:

```
10
```

Total planned observations:

```
97 * 4 * 10 = 3880.
```

Each observation is uniquely identified by:

```
node_id
nominal_mask_rate
perturbation_seed
```

## Raw Output

Primary raw perturbation artifact:

`results/perturbations/feature_mask.csv`

The file is long-format.

Each row corresponds to exactly one node-rate-seed perturbation
observation.

At minimum, each row records:

* node_id
* nominal_mask_rate
* perturbation_seed
* derived_sampling_seed
* neighbourhood_size
* active_entries_before
* num_entries_masked
* realized_mask_rate
* baseline_predicted_label
* perturbed_predicted_label
* pred_same
* baseline_confidence
* perturbed_baseline_class_confidence
* baseline_class_confidence_delta
* perturbed_max_confidence
* baseline_top_k_available
* perturbed_output_kernel_valid
* perturbed_solver_converged
* perturbed_num_nonzero
* perturbed_top_k_available
* explanation_available
* jaccard
* intersection_size
* optimizer_iterations
* failure_reason

Additional diagnostic columns may be stored provided that the
pre-specified primary outcomes are not changed.

## Checkpointing and Resume Behavior

Because 3880 GraphLIME optimizations may require substantial CPU time,
the experiment runner should save progress periodically.

If the experiment is interrupted, rerunning it may resume from the
existing raw CSV.

Already completed node-rate-seed keys must not be recomputed or
duplicated.

Resume behavior is an engineering feature and must not alter the
experimental protocol.

## Preflight Validation

Before the full experiment begins, the implementation must verify that
the perturbation operator:

* does not modify the original feature tensor;
* masks exactly the expected number of entries;
* masks only originally nonzero entries;
* modifies only nodes inside the target 2-hop neighbourhood;
* sets sampled entries exactly to zero;
* leaves all unsampled entries unchanged; and
* is deterministic for the same node-rate-seed combination.

The preflight test must not be used to select a favourable
perturbation outcome.

## Prohibited Post-Hoc Changes

After primary perturbation results begin to be generated, do not
automatically:

* change RHO;
* change NUM_HOPS;
* change TOP_K;
* change the GCN checkpoint;
* replace baseline nodes;
* change masking rates because a result looks weak;
* change perturbation seeds because a result looks inconvenient;
* assign zero Jaccard to unavailable explanations;
* discard prediction-changing perturbations;
* discard non-converged perturbations;
* renormalize only selected observations; or
* retrain the GCN.

Any later methodological extension must be explicitly documented as a
new robustness or ablation experiment.

## Primary Planned Analyses

The primary Phase-10 analysis will estimate, by nominal perturbation
rate:

1. prediction agreement rate;
2. mean and distribution of baseline-class confidence change;
3. perturbed explanation availability rate;
4. mean and distribution of Jaccard similarity among valid explanation
   pairs; and
5. Jaccard similarity conditional on pred_same = True.

The final interpretation will distinguish:

```
prediction stability
```

from:

```
explanation stability.
```

## Status

Phase-9 primary perturbation protocol:

FROZEN BEFORE PRIMARY PERTURBATION RESULTS.

Planned raw observations:

```
3880.
```

Next implementation:

`src/perturb.py`

followed by:

`experiments/03_feature_perturbation.py`
