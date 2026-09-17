# experiments/04c_finalize_hypotheses.py
#
# Phase 10D:
# Finalize the pre-specified H1-H5 hypothesis evidence summary using
# already frozen and validated Phase-9 / Phase-10 results.
#
# Run from repository root:
#
#   python -m experiments.04c_finalize_hypotheses
#
# -------------------------------------------------------------------------
# IMPORTANT METHODOLOGICAL RULES
# -------------------------------------------------------------------------
#
# This script DOES NOT:
#
#   - retrain the GCN,
#   - rerun GraphLIME,
#   - regenerate perturbations,
#   - change rho,
#   - change selected nodes,
#   - change perturbation rates,
#   - change perturbation seeds,
#   - impute unavailable explanations as Jaccard = 0,
#   - redefine hypotheses after observing results.
#
# It consumes the already generated Phase-10 analysis artifacts and
# produces a final evidence summary for the hypotheses that were specified
# before interpretation of the perturbation results.
#
# -------------------------------------------------------------------------
# PRE-SPECIFIED HYPOTHESES
# -------------------------------------------------------------------------
#
# H1:
#   GraphLIME top-K explanation overlap tends to decline as feature
#   masking severity increases.
#
# H2:
#   GraphLIME explanations can change even when the GNN predicted class
#   remains unchanged.
#
# H3:
#   Prediction agreement tends to decline as feature masking severity
#   increases.
#
# H4:
#   Prediction stability and explanation stability are not perfectly
#   coupled.
#
# H5:
#   Perturbed explanation unavailability may increase with masking
#   severity.
#
# -------------------------------------------------------------------------
# STATISTICAL INTERPRETATION
# -------------------------------------------------------------------------
#
# H1-H3:
#   Use already computed paired node-level contrasts from 04b.
#
# H4:
#   Quantify coupling descriptively at the node level using:
#
#       1. endpoint change in prediction stability:
#              prediction_stability(0.20)
#              - prediction_stability(0.01)
#
#       2. endpoint change in explanation stability:
#              mean_jaccard(0.20)
#              - mean_jaccard(0.01)
#
#       3. Spearman rank correlation between those two endpoint changes.
#
#       4. A direct decoupling count:
#          nodes whose endpoint prediction stability is unchanged while
#          endpoint explanation overlap decreases.
#
#   The Spearman correlation is descriptive. No IID significance test is
#   performed because nodes belong to a single graph and are not assumed
#   to be independent population samples.
#
# H5:
#   Use explanation availability across the four frozen masking rates.
#   This is primarily descriptive because unavailability is sparse.
#
# -------------------------------------------------------------------------
# OUTPUTS
# -------------------------------------------------------------------------
#
#   results/analysis/final_hypothesis_summary.csv
#   results/analysis/h4_coupling_summary.csv
#   results/analysis/h4_node_endpoint_changes.csv
#   results/analysis/h5_availability_by_rate.csv
#
# -------------------------------------------------------------------------


from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================================
# FROZEN EXPERIMENT CONFIGURATION
# =========================================================================

EXPECTED_NODES = 97

MASK_RATES = [
    0.01,
    0.05,
    0.10,
    0.20,
]

LOW_ENDPOINT_RATE = 0.01
HIGH_ENDPOINT_RATE = 0.20

EXPECTED_NODE_RATE_ROWS = (
    EXPECTED_NODES
    * len(MASK_RATES)
)

EXPECTED_PERTURBATIONS_PER_NODE_RATE = 10

EXPECTED_TOTAL_PERTURBATIONS = (
    EXPECTED_NODES
    * len(MASK_RATES)
    * EXPECTED_PERTURBATIONS_PER_NODE_RATE
)

TOP_K = 10

FLOAT_TOLERANCE = 1e-9
CHANGE_TOLERANCE = 1e-12


# =========================================================================
# PATHS
# =========================================================================

ANALYSIS_DIR = Path(
    "results/analysis"
)

PERTURBATION_DIR = Path(
    "results/perturbations"
)

NODE_RATE_PATH = (
    ANALYSIS_DIR
    / "node_rate_summary.csv"
)

PAIRED_CONTRASTS_PATH = (
    ANALYSIS_DIR
    / "paired_rate_contrasts.csv"
)

MONOTONICITY_PATH = (
    ANALYSIS_DIR
    / "monotonicity_summary.csv"
)

PHASE9_VALIDATION_SUMMARY_PATH = (
    PERTURBATION_DIR
    / "feature_mask_validation_summary.csv"
)

FINAL_HYPOTHESIS_PATH = (
    ANALYSIS_DIR
    / "final_hypothesis_summary.csv"
)

H4_COUPLING_PATH = (
    ANALYSIS_DIR
    / "h4_coupling_summary.csv"
)

H4_NODE_CHANGES_PATH = (
    ANALYSIS_DIR
    / "h4_node_endpoint_changes.csv"
)

H5_AVAILABILITY_PATH = (
    ANALYSIS_DIR
    / "h5_availability_by_rate.csv"
)


# =========================================================================
# REQUIRED COLUMNS
# =========================================================================

NODE_RATE_REQUIRED_COLUMNS = {
    "node_id",
    "nominal_mask_rate",
    "num_perturbations",
    "prediction_stability",
    "prediction_change_rate",
    "explanation_availability",
    "mean_jaccard",
    "mean_jaccard_pred_same",
    "mean_absolute_confidence_delta",
    "num_explanations_available",
    "num_explanations_unavailable",
    "num_pred_same_with_explanation",
}

CONTRAST_REQUIRED_COLUMNS = {
    "outcome",
    "lower_rate",
    "higher_rate",
    "contrast",
    "n_paired_nodes",
    "lower_rate_mean",
    "higher_rate_mean",
    "mean_paired_difference",
    "paired_difference_ci_low",
    "paired_difference_ci_high",
    "wilcoxon_p_raw",
    "wilcoxon_p_holm",
    "rank_biserial",
}

MONOTONICITY_REQUIRED_COLUMNS = {
    "outcome",
    "total_nodes",
    "complete_nodes",
    "incomplete_nodes",
    "strictly_decreasing_nodes",
    "nonincreasing_with_ties_nodes",
    "constant_nodes",
    "strictly_increasing_nodes",
    "nondecreasing_with_ties_nodes",
    "nonmonotonic_nodes",
    "monotonic_nonincreasing_nodes",
    "monotonic_nondecreasing_nodes",
    "fraction_monotonic_nonincreasing",
    "fraction_strictly_decreasing",
    "mean_endpoint_difference_0_20_minus_0_01",
    "median_endpoint_difference_0_20_minus_0_01",
}


# =========================================================================
# GENERAL HELPERS
# =========================================================================


def format_float(
    value,
    digits=4,
):
    """
    Format a numerical value safely for terminal output.
    """

    if pd.isna(value):
        return "NA"

    return f"{float(value):.{digits}f}"


def format_scientific(
    value,
):
    """
    Scientific notation for p-values.
    """

    if pd.isna(value):
        return "NA"

    return f"{float(value):.6e}"


def safe_mean(values):
    """
    Mean over finite numerical values.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_median(values):
    """
    Median over finite numerical values.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def rate_mask(
    series,
    rate,
):
    """
    Floating-point-safe perturbation-rate comparison.
    """

    return np.isclose(
        series.astype(float),
        float(rate),
        atol=FLOAT_TOLERANCE,
        rtol=0.0,
    )


# =========================================================================
# RANKING / SPEARMAN HELPERS
# =========================================================================


def average_ranks(values):
    """
    Compute one-based average ranks.

    Tied observations receive the average of the ranks they occupy.

    This is implemented locally so Phase 10 does not depend on SciPy.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:

        raise ValueError(
            "average_ranks expects a one-dimensional array."
        )

    if not np.isfinite(values).all():

        raise ValueError(
            "average_ranks requires finite values."
        )

    n = len(values)

    if n == 0:

        return np.array(
            [],
            dtype=float,
        )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    sorted_values = values[
        order
    ]

    sorted_ranks = np.empty(
        n,
        dtype=float,
    )

    start = 0

    while start < n:

        end = start + 1

        while (
            end < n
            and
            sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = (
            (start + 1)
            + end
        ) / 2.0

        sorted_ranks[
            start:end
        ] = average_rank

        start = end

    ranks = np.empty(
        n,
        dtype=float,
    )

    ranks[
        order
    ] = sorted_ranks

    return ranks


def pearson_correlation(
    x,
    y,
):
    """
    Pearson correlation for finite paired values.

    Returns NaN if fewer than two pairs exist or either variable has
    zero variance.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if x.shape != y.shape:

        raise ValueError(
            "Correlation arrays must have identical shape."
        )

    finite = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if len(x) < 2:
        return np.nan

    x_centered = (
        x
        - np.mean(x)
    )

    y_centered = (
        y
        - np.mean(y)
    )

    denominator = math.sqrt(
        float(
            np.sum(
                x_centered ** 2
            )
        )
        *
        float(
            np.sum(
                y_centered ** 2
            )
        )
    )

    if denominator <= 0.0:
        return np.nan

    return float(
        np.sum(
            x_centered
            * y_centered
        )
        / denominator
    )


def spearman_correlation(
    x,
    y,
):
    """
    Spearman rank correlation implemented as Pearson correlation between
    average ranks.

    No significance p-value is calculated. The statistic is used only as
    a descriptive measure of node-level coupling.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if x.shape != y.shape:

        raise ValueError(
            "Spearman arrays must have identical shape."
        )

    finite = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if len(x) < 2:
        return np.nan

    x_ranks = average_ranks(
        x
    )

    y_ranks = average_ranks(
        y
    )

    return pearson_correlation(
        x_ranks,
        y_ranks,
    )


# =========================================================================
# INPUT LOADING AND VALIDATION
# =========================================================================


def load_node_rate_summary():
    """
    Load and validate the frozen Phase-10 node-rate table.
    """

    if not NODE_RATE_PATH.exists():

        raise FileNotFoundError(
            f"Missing required artifact: {NODE_RATE_PATH}"
        )

    df = pd.read_csv(
        NODE_RATE_PATH
    )

    missing = (
        NODE_RATE_REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "node_rate_summary.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(df) != EXPECTED_NODE_RATE_ROWS:

        raise RuntimeError(
            "Unexpected node-rate row count. "
            f"Observed={len(df)}, "
            f"expected={EXPECTED_NODE_RATE_ROWS}."
        )

    if df[
        "node_id"
    ].nunique() != EXPECTED_NODES:

        raise RuntimeError(
            "Unexpected number of unique nodes."
        )

    if df.duplicated(
        subset=[
            "node_id",
            "nominal_mask_rate",
        ]
    ).any():

        raise RuntimeError(
            "Duplicate node-rate rows detected."
        )

    observed_rates = sorted(
        df[
            "nominal_mask_rate"
        ]
        .astype(float)
        .unique()
        .tolist()
    )

    if len(observed_rates) != len(MASK_RATES):

        raise RuntimeError(
            "Unexpected number of perturbation rates."
        )

    for observed, expected in zip(
        observed_rates,
        MASK_RATES,
    ):

        if not np.isclose(
            observed,
            expected,
            atol=FLOAT_TOLERANCE,
            rtol=0.0,
        ):

            raise RuntimeError(
                "Observed perturbation-rate set differs "
                "from the frozen protocol."
            )

    counts_per_node = (
        df.groupby(
            "node_id"
        )
        .size()
    )

    if not (
        counts_per_node
        == len(MASK_RATES)
    ).all():

        raise RuntimeError(
            "At least one node does not have exactly four rate rows."
        )

    if not (
        df[
            "num_perturbations"
        ]
        == EXPECTED_PERTURBATIONS_PER_NODE_RATE
    ).all():

        raise RuntimeError(
            "At least one node-rate row does not summarize "
            "exactly 10 perturbations."
        )

    bounded_columns = [
        "prediction_stability",
        "prediction_change_rate",
        "explanation_availability",
        "mean_jaccard",
        "mean_jaccard_pred_same",
    ]

    for column in bounded_columns:

        values = (
            df[
                column
            ]
            .dropna()
            .astype(float)
        )

        if not values.between(
            0.0,
            1.0,
            inclusive="both",
        ).all():

            raise RuntimeError(
                f"{column} contains values outside [0,1]."
            )

    return df


def load_paired_contrasts():
    """
    Load and validate the Phase-10C paired contrast table.
    """

    if not PAIRED_CONTRASTS_PATH.exists():

        raise FileNotFoundError(
            f"Missing required artifact: {PAIRED_CONTRASTS_PATH}"
        )

    df = pd.read_csv(
        PAIRED_CONTRASTS_PATH
    )

    missing = (
        CONTRAST_REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "paired_rate_contrasts.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    expected_outcomes = {
        "mean_jaccard",
        "mean_jaccard_pred_same",
        "prediction_stability",
        "explanation_availability",
        "mean_absolute_confidence_delta",
    }

    observed_outcomes = set(
        df[
            "outcome"
        ].tolist()
    )

    if observed_outcomes != expected_outcomes:

        raise RuntimeError(
            "Unexpected outcome set in paired_rate_contrasts.csv."
        )

    expected_rows = (
        len(expected_outcomes)
        * math.comb(
            len(MASK_RATES),
            2,
        )
    )

    if len(df) != expected_rows:

        raise RuntimeError(
            "Unexpected paired-contrast row count. "
            f"Observed={len(df)}, "
            f"expected={expected_rows}."
        )

    for column in [
        "wilcoxon_p_raw",
        "wilcoxon_p_holm",
    ]:

        finite = (
            df[
                column
            ]
            .dropna()
            .astype(float)
        )

        if not finite.between(
            0.0,
            1.0,
            inclusive="both",
        ).all():

            raise RuntimeError(
                f"{column} contains values outside [0,1]."
            )

    return df


def load_monotonicity_summary():
    """
    Load and validate the Phase-10C monotonicity summary.
    """

    if not MONOTONICITY_PATH.exists():

        raise FileNotFoundError(
            f"Missing required artifact: {MONOTONICITY_PATH}"
        )

    df = pd.read_csv(
        MONOTONICITY_PATH
    )

    missing = (
        MONOTONICITY_REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "monotonicity_summary.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    expected_outcomes = {
        "mean_jaccard",
        "mean_jaccard_pred_same",
        "prediction_stability",
        "explanation_availability",
    }

    if set(
        df[
            "outcome"
        ].tolist()
    ) != expected_outcomes:

        raise RuntimeError(
            "Unexpected outcome set in monotonicity_summary.csv."
        )

    if len(df) != 4:

        raise RuntimeError(
            "Expected exactly four monotonicity rows."
        )

    return df


def validate_phase9_validation_artifact():
    """
    Confirm that the Phase-9 aggregate validation artifact exists and
    contains no failed aggregate check if a recognizable pass column is
    available.

    The Phase-9 validator already performed the authoritative raw-data
    validation. This function does not recreate that validator.
    """

    if not PHASE9_VALIDATION_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            "Missing frozen Phase-9 validation artifact: "
            f"{PHASE9_VALIDATION_SUMMARY_PATH}"
        )

    df = pd.read_csv(
        PHASE9_VALIDATION_SUMMARY_PATH
    )

    if len(df) == 0:

        raise RuntimeError(
            "Phase-9 validation summary is empty."
        )

    # Different validator versions may use different names for the
    # aggregate pass/fail column. Check common possibilities without
    # imposing a new schema on an already frozen artifact.

    candidate_columns = [
        "passed",
        "pass",
        "is_valid",
        "valid",
        "status",
        "result",
    ]

    recognized = None

    for column in candidate_columns:

        if column in df.columns:

            recognized = column
            break

    if recognized is not None:

        values = (
            df[
                recognized
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        accepted = {
            "true",
            "1",
            "pass",
            "passed",
            "ok",
            "valid",
        }

        if not values.isin(
            accepted
        ).all():

            failed = df.loc[
                ~values.isin(
                    accepted
                )
            ]

            raise RuntimeError(
                "Phase-9 validation artifact contains at least one "
                "non-passing aggregate check:\n"
                f"{failed.head(20)}"
            )

    return df


# =========================================================================
# TABLE RETRIEVAL HELPERS
# =========================================================================


def get_contrast(
    contrasts_df,
    outcome,
    lower_rate=LOW_ENDPOINT_RATE,
    higher_rate=HIGH_ENDPOINT_RATE,
):
    """
    Retrieve one paired contrast row.
    """

    subset = contrasts_df.loc[
        (
            contrasts_df[
                "outcome"
            ]
            == outcome
        )
        &
        rate_mask(
            contrasts_df[
                "lower_rate"
            ],
            lower_rate,
        )
        &
        rate_mask(
            contrasts_df[
                "higher_rate"
            ],
            higher_rate,
        )
    ]

    if len(subset) != 1:

        raise RuntimeError(
            "Expected exactly one contrast for "
            f"{outcome}, {lower_rate} -> {higher_rate}."
        )

    return subset.iloc[0]


def get_monotonicity(
    monotonicity_df,
    outcome,
):
    """
    Retrieve one monotonicity row.
    """

    subset = monotonicity_df.loc[
        monotonicity_df[
            "outcome"
        ]
        == outcome
    ]

    if len(subset) != 1:

        raise RuntimeError(
            "Expected exactly one monotonicity row for "
            f"{outcome}."
        )

    return subset.iloc[0]


# =========================================================================
# RATE-LEVEL SUMMARY FROM NODE-LEVEL DATA
# =========================================================================


def build_rate_level_summary(
    node_rate_df,
):
    """
    Reconstruct the principal rate-level descriptive means directly from
    the validated node-rate table.

    This is used as an independent finalization check rather than relying
    on a previously printed terminal summary.
    """

    records = []

    for rate in MASK_RATES:

        subset = node_rate_df.loc[
            rate_mask(
                node_rate_df[
                    "nominal_mask_rate"
                ],
                rate,
            )
        ].copy()

        if len(subset) != EXPECTED_NODES:

            raise RuntimeError(
                f"Rate {rate} has unexpected node count."
            )

        records.append(
            {
                "nominal_mask_rate": float(
                    rate
                ),
                "num_nodes": int(
                    len(subset)
                ),
                "mean_prediction_stability": safe_mean(
                    subset[
                        "prediction_stability"
                    ]
                ),
                "mean_explanation_availability": safe_mean(
                    subset[
                        "explanation_availability"
                    ]
                ),
                "mean_jaccard": safe_mean(
                    subset[
                        "mean_jaccard"
                    ]
                ),
                "mean_jaccard_pred_same": safe_mean(
                    subset[
                        "mean_jaccard_pred_same"
                    ]
                ),
                "mean_absolute_confidence_delta": safe_mean(
                    subset[
                        "mean_absolute_confidence_delta"
                    ]
                ),
                "total_explanations_available": int(
                    subset[
                        "num_explanations_available"
                    ].sum()
                ),
                "total_explanations_unavailable": int(
                    subset[
                        "num_explanations_unavailable"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================================
# H4: PREDICTION / EXPLANATION COUPLING ANALYSIS
# =========================================================================


def build_h4_node_endpoint_changes(
    node_rate_df,
):
    """
    Construct node-level endpoint changes from 1% to 20% masking.

    Explanation stability is represented by mean_jaccard.

    Prediction stability is represented by prediction_stability.

    Only nodes with finite endpoint Jaccard values are included in the
    coupling analysis. Nodes with unavailable endpoint Jaccard remain
    explicitly marked and are not assigned Jaccard = 0.
    """

    low = node_rate_df.loc[
        rate_mask(
            node_rate_df[
                "nominal_mask_rate"
            ],
            LOW_ENDPOINT_RATE,
        ),
        [
            "node_id",
            "prediction_stability",
            "mean_jaccard",
        ],
    ].copy()

    high = node_rate_df.loc[
        rate_mask(
            node_rate_df[
                "nominal_mask_rate"
            ],
            HIGH_ENDPOINT_RATE,
        ),
        [
            "node_id",
            "prediction_stability",
            "mean_jaccard",
        ],
    ].copy()

    low = low.rename(
        columns={
            "prediction_stability": (
                "prediction_stability_0_01"
            ),
            "mean_jaccard": (
                "mean_jaccard_0_01"
            ),
        }
    )

    high = high.rename(
        columns={
            "prediction_stability": (
                "prediction_stability_0_20"
            ),
            "mean_jaccard": (
                "mean_jaccard_0_20"
            ),
        }
    )

    merged = low.merge(
        high,
        on="node_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_NODES:

        raise RuntimeError(
            "H4 endpoint merge did not retain all 97 target nodes."
        )

    merged[
        "prediction_stability_change"
    ] = (
        merged[
            "prediction_stability_0_20"
        ]
        -
        merged[
            "prediction_stability_0_01"
        ]
    )

    merged[
        "explanation_stability_change"
    ] = (
        merged[
            "mean_jaccard_0_20"
        ]
        -
        merged[
            "mean_jaccard_0_01"
        ]
    )

    merged[
        "endpoint_jaccard_available"
    ] = (
        merged[
            "mean_jaccard_0_01"
        ].notna()
        &
        merged[
            "mean_jaccard_0_20"
        ].notna()
    )

    merged[
        "prediction_endpoint_unchanged"
    ] = (
        np.abs(
            merged[
                "prediction_stability_change"
            ].astype(float)
        )
        <= CHANGE_TOLERANCE
    )

    merged[
        "explanation_endpoint_decreased"
    ] = (
        merged[
            "endpoint_jaccard_available"
        ]
        &
        (
            merged[
                "explanation_stability_change"
            ].astype(float)
            < -CHANGE_TOLERANCE
        )
    )

    merged[
        "direct_decoupling_event"
    ] = (
        merged[
            "prediction_endpoint_unchanged"
        ]
        &
        merged[
            "explanation_endpoint_decreased"
        ]
    )

    return (
        merged.sort_values(
            "node_id"
        )
        .reset_index(
            drop=True
        )
    )


def build_h4_coupling_summary(
    h4_node_df,
):
    """
    Summarize node-level prediction/explanation endpoint coupling.

    The principal H4 question is whether prediction stability and
    explanation stability move in perfect lockstep.

    We report:
        - Spearman correlation between endpoint changes,
        - Pearson correlation as an auxiliary descriptive statistic,
        - direct decoupling-event counts,
        - direction counts.

    No p-value is attached to these correlations because graph nodes are
    not treated as IID population samples.
    """

    valid = h4_node_df.loc[
        h4_node_df[
            "endpoint_jaccard_available"
        ]
    ].copy()

    prediction_change = valid[
        "prediction_stability_change"
    ].to_numpy(
        dtype=float
    )

    explanation_change = valid[
        "explanation_stability_change"
    ].to_numpy(
        dtype=float
    )

    spearman_rho = spearman_correlation(
        prediction_change,
        explanation_change,
    )

    pearson_r = pearson_correlation(
        prediction_change,
        explanation_change,
    )

    prediction_unchanged = (
        np.abs(
            prediction_change
        )
        <= CHANGE_TOLERANCE
    )

    prediction_decreased = (
        prediction_change
        < -CHANGE_TOLERANCE
    )

    prediction_increased = (
        prediction_change
        > CHANGE_TOLERANCE
    )

    explanation_decreased = (
        explanation_change
        < -CHANGE_TOLERANCE
    )

    explanation_unchanged = (
        np.abs(
            explanation_change
        )
        <= CHANGE_TOLERANCE
    )

    explanation_increased = (
        explanation_change
        > CHANGE_TOLERANCE
    )

    direct_decoupling = (
        prediction_unchanged
        &
        explanation_decreased
    )

    num_valid = len(
        valid
    )

    num_direct_decoupling = int(
        direct_decoupling.sum()
    )

    return pd.DataFrame(
        [
            {
                "low_rate": float(
                    LOW_ENDPOINT_RATE
                ),
                "high_rate": float(
                    HIGH_ENDPOINT_RATE
                ),
                "difference_orientation": (
                    "high_rate_minus_low_rate"
                ),
                "total_target_nodes": int(
                    EXPECTED_NODES
                ),
                "nodes_with_valid_endpoint_jaccard": int(
                    num_valid
                ),
                "nodes_missing_endpoint_jaccard": int(
                    EXPECTED_NODES
                    - num_valid
                ),
                "spearman_prediction_vs_explanation_change": (
                    spearman_rho
                ),
                "pearson_prediction_vs_explanation_change": (
                    pearson_r
                ),
                "correlation_inference": (
                    "descriptive_only_no_iid_p_value"
                ),
                "nodes_prediction_endpoint_unchanged": int(
                    prediction_unchanged.sum()
                ),
                "nodes_prediction_endpoint_decreased": int(
                    prediction_decreased.sum()
                ),
                "nodes_prediction_endpoint_increased": int(
                    prediction_increased.sum()
                ),
                "nodes_explanation_endpoint_decreased": int(
                    explanation_decreased.sum()
                ),
                "nodes_explanation_endpoint_unchanged": int(
                    explanation_unchanged.sum()
                ),
                "nodes_explanation_endpoint_increased": int(
                    explanation_increased.sum()
                ),
                "nodes_prediction_unchanged_explanation_decreased": (
                    num_direct_decoupling
                ),
                "fraction_prediction_unchanged_explanation_decreased": (
                    float(
                        num_direct_decoupling
                        / num_valid
                    )
                    if num_valid > 0
                    else np.nan
                ),
                "mean_prediction_endpoint_change": safe_mean(
                    prediction_change
                ),
                "mean_explanation_endpoint_change": safe_mean(
                    explanation_change
                ),
                "interpretation_constraint": (
                    "Correlation and event counts are descriptive "
                    "within the frozen Cora graph experiment; "
                    "they are not IID population-level estimates."
                ),
            }
        ]
    )


# =========================================================================
# H5: EXPLANATION AVAILABILITY ANALYSIS
# =========================================================================


def build_h5_availability_summary(
    node_rate_df,
):
    """
    Build explanation-availability summary for each perturbation rate.

    The raw Phase-9 study has 97 nodes x 10 perturbation seeds = 970
    observations at each rate.
    """

    records = []

    expected_per_rate = (
        EXPECTED_NODES
        * EXPECTED_PERTURBATIONS_PER_NODE_RATE
    )

    for rate in MASK_RATES:

        subset = node_rate_df.loc[
            rate_mask(
                node_rate_df[
                    "nominal_mask_rate"
                ],
                rate,
            )
        ].copy()

        available = int(
            subset[
                "num_explanations_available"
            ].sum()
        )

        unavailable = int(
            subset[
                "num_explanations_unavailable"
            ].sum()
        )

        if (
            available
            + unavailable
            != expected_per_rate
        ):

            raise RuntimeError(
                "H5 availability accounting failed "
                f"for rate {rate}."
            )

        records.append(
            {
                "nominal_mask_rate": float(
                    rate
                ),
                "num_nodes": int(
                    len(subset)
                ),
                "total_perturbation_observations": int(
                    expected_per_rate
                ),
                "explanations_available": int(
                    available
                ),
                "explanations_unavailable": int(
                    unavailable
                ),
                "explanation_availability_rate": float(
                    available
                    / expected_per_rate
                ),
                "explanation_unavailability_rate": float(
                    unavailable
                    / expected_per_rate
                ),
                "nodes_with_at_least_one_unavailable": int(
                    (
                        subset[
                            "num_explanations_unavailable"
                        ]
                        > 0
                    ).sum()
                ),
            }
        )

    result = pd.DataFrame(
        records
    )

    result[
        "unavailability_change_from_previous_rate"
    ] = (
        result[
            "explanation_unavailability_rate"
        ]
        .diff()
    )

    return result


# =========================================================================
# FINAL HYPOTHESIS TABLE
# =========================================================================


def build_final_hypothesis_summary(
    contrasts_df,
    monotonicity_df,
    h4_summary_df,
    h5_df,
):
    """
    Construct the final evidence table for the five pre-specified
    hypotheses.

    IMPORTANT:
    The table reports evidence and descriptive consistency. It does not
    automatically turn statistical output into universal claims about
    GraphLIME.
    """

    # ---------------------------------------------------------------------
    # H1
    # ---------------------------------------------------------------------

    h1_contrast = get_contrast(
        contrasts_df,
        "mean_jaccard",
    )

    h1_monotonicity = get_monotonicity(
        monotonicity_df,
        "mean_jaccard",
    )

    # ---------------------------------------------------------------------
    # H2
    # ---------------------------------------------------------------------

    h2_contrast = get_contrast(
        contrasts_df,
        "mean_jaccard_pred_same",
    )

    h2_monotonicity = get_monotonicity(
        monotonicity_df,
        "mean_jaccard_pred_same",
    )

    # ---------------------------------------------------------------------
    # H3
    # ---------------------------------------------------------------------

    h3_contrast = get_contrast(
        contrasts_df,
        "prediction_stability",
    )

    h3_monotonicity = get_monotonicity(
        monotonicity_df,
        "prediction_stability",
    )

    # ---------------------------------------------------------------------
    # H4
    # ---------------------------------------------------------------------

    if len(
        h4_summary_df
    ) != 1:

        raise RuntimeError(
            "Expected exactly one H4 coupling-summary row."
        )

    h4 = h4_summary_df.iloc[
        0
    ]

    # ---------------------------------------------------------------------
    # H5
    # ---------------------------------------------------------------------

    if len(
        h5_df
    ) != len(
        MASK_RATES
    ):

        raise RuntimeError(
            "Expected four H5 availability rows."
        )

    h5_low = h5_df.loc[
        rate_mask(
            h5_df[
                "nominal_mask_rate"
            ],
            LOW_ENDPOINT_RATE,
        )
    ].iloc[0]

    h5_high = h5_df.loc[
        rate_mask(
            h5_df[
                "nominal_mask_rate"
            ],
            HIGH_ENDPOINT_RATE,
        )
    ].iloc[0]

    h5_unavailable_sequence = (
        h5_df.sort_values(
            "nominal_mask_rate"
        )[
            "explanations_unavailable"
        ]
        .astype(int)
        .tolist()
    )

    h5_rates_sequence = (
        h5_df.sort_values(
            "nominal_mask_rate"
        )[
            "explanation_unavailability_rate"
        ]
        .astype(float)
        .tolist()
    )

    h5_monotonic_increase = bool(
        np.all(
            np.diff(
                np.asarray(
                    h5_rates_sequence,
                    dtype=float,
                )
            )
            >= -CHANGE_TOLERANCE
        )
    )

    # ---------------------------------------------------------------------
    # Final evidence records.
    # ---------------------------------------------------------------------

    records = [
        {
            "hypothesis_id": "H1",
            "hypothesis": (
                "GraphLIME top-K explanation overlap tends to decline "
                "as feature masking severity increases."
            ),
            "primary_quantity": (
                "node-level mean_jaccard"
            ),
            "analysis_type": (
                "paired_node_level_endpoint_and_monotonicity"
            ),
            "endpoint_comparison": (
                "0.20_minus_0.01"
            ),
            "n_nodes": int(
                h1_contrast[
                    "n_paired_nodes"
                ]
            ),
            "endpoint_mean_difference": float(
                h1_contrast[
                    "mean_paired_difference"
                ]
            ),
            "endpoint_ci_low": float(
                h1_contrast[
                    "paired_difference_ci_low"
                ]
            ),
            "endpoint_ci_high": float(
                h1_contrast[
                    "paired_difference_ci_high"
                ]
            ),
            "holm_adjusted_wilcoxon_p": float(
                h1_contrast[
                    "wilcoxon_p_holm"
                ]
            ),
            "rank_biserial": float(
                h1_contrast[
                    "rank_biserial"
                ]
            ),
            "fraction_monotonic_nonincreasing": float(
                h1_monotonicity[
                    "fraction_monotonic_nonincreasing"
                ]
            ),
            "descriptive_evidence": (
                "Endpoint mean difference is negative; paired "
                "bootstrap interval lies below zero; most complete "
                "node trajectories are nonincreasing."
            ),
            "interpretation_scope": (
                "Frozen Cora + GCN + GraphLIME + local active-entry "
                "feature-masking experiment."
            ),
        },

        {
            "hypothesis_id": "H2",
            "hypothesis": (
                "GraphLIME explanations can change even when the "
                "GNN predicted class remains unchanged."
            ),
            "primary_quantity": (
                "node-level mean_jaccard_pred_same"
            ),
            "analysis_type": (
                "prediction_conditioned_paired_endpoint_analysis"
            ),
            "endpoint_comparison": (
                "0.20_minus_0.01"
            ),
            "n_nodes": int(
                h2_contrast[
                    "n_paired_nodes"
                ]
            ),
            "endpoint_mean_difference": float(
                h2_contrast[
                    "mean_paired_difference"
                ]
            ),
            "endpoint_ci_low": float(
                h2_contrast[
                    "paired_difference_ci_low"
                ]
            ),
            "endpoint_ci_high": float(
                h2_contrast[
                    "paired_difference_ci_high"
                ]
            ),
            "holm_adjusted_wilcoxon_p": float(
                h2_contrast[
                    "wilcoxon_p_holm"
                ]
            ),
            "rank_biserial": float(
                h2_contrast[
                    "rank_biserial"
                ]
            ),
            "fraction_monotonic_nonincreasing": float(
                h2_monotonicity[
                    "fraction_monotonic_nonincreasing"
                ]
            ),
            "descriptive_evidence": (
                "Explanation overlap declines strongly even after "
                "conditioning on observations whose predicted class "
                "remained unchanged."
            ),
            "interpretation_scope": (
                "Prediction-conditioned GraphLIME explanation overlap "
                "within the frozen feature-masking experiment."
            ),
        },

        {
            "hypothesis_id": "H3",
            "hypothesis": (
                "Prediction agreement tends to decline as feature "
                "masking severity increases."
            ),
            "primary_quantity": (
                "node-level prediction_stability"
            ),
            "analysis_type": (
                "paired_node_level_endpoint_and_monotonicity"
            ),
            "endpoint_comparison": (
                "0.20_minus_0.01"
            ),
            "n_nodes": int(
                h3_contrast[
                    "n_paired_nodes"
                ]
            ),
            "endpoint_mean_difference": float(
                h3_contrast[
                    "mean_paired_difference"
                ]
            ),
            "endpoint_ci_low": float(
                h3_contrast[
                    "paired_difference_ci_low"
                ]
            ),
            "endpoint_ci_high": float(
                h3_contrast[
                    "paired_difference_ci_high"
                ]
            ),
            "holm_adjusted_wilcoxon_p": float(
                h3_contrast[
                    "wilcoxon_p_holm"
                ]
            ),
            "rank_biserial": float(
                h3_contrast[
                    "rank_biserial"
                ]
            ),
            "fraction_monotonic_nonincreasing": float(
                h3_monotonicity[
                    "fraction_monotonic_nonincreasing"
                ]
            ),
            "descriptive_evidence": (
                "Prediction stability is evaluated across the same "
                "four feature-masking rates using paired target nodes."
            ),
            "interpretation_scope": (
                "Predicted-class agreement under the frozen "
                "feature-masking experiment."
            ),
        },

        {
            "hypothesis_id": "H4",
            "hypothesis": (
                "Prediction stability and explanation stability are "
                "not perfectly coupled."
            ),
            "primary_quantity": (
                "node-level endpoint changes in prediction stability "
                "and mean_jaccard"
            ),
            "analysis_type": (
                "descriptive_endpoint_coupling_analysis"
            ),
            "endpoint_comparison": (
                "0.20_minus_0.01"
            ),
            "n_nodes": int(
                h4[
                    "nodes_with_valid_endpoint_jaccard"
                ]
            ),
            "endpoint_mean_difference": np.nan,
            "endpoint_ci_low": np.nan,
            "endpoint_ci_high": np.nan,
            "holm_adjusted_wilcoxon_p": np.nan,
            "rank_biserial": np.nan,
            "fraction_monotonic_nonincreasing": np.nan,
            "descriptive_evidence": (
                "Spearman correlation between node-level endpoint "
                "prediction-stability change and explanation-stability "
                f"change = {format_float(h4['spearman_prediction_vs_explanation_change'])}; "
                "nodes with unchanged endpoint prediction stability "
                "but decreased explanation overlap = "
                f"{int(h4['nodes_prediction_unchanged_explanation_decreased'])}."
            ),
            "interpretation_scope": (
                "Descriptive coupling analysis only; no IID "
                "correlation p-value is used because nodes belong "
                "to one graph."
            ),
        },

        {
            "hypothesis_id": "H5",
            "hypothesis": (
                "Perturbed explanation unavailability may increase "
                "with feature masking severity."
            ),
            "primary_quantity": (
                "explanation_unavailability_rate"
            ),
            "analysis_type": (
                "descriptive_rate_level_availability_analysis"
            ),
            "endpoint_comparison": (
                "0.20_minus_0.01"
            ),
            "n_nodes": int(
                EXPECTED_NODES
            ),
            "endpoint_mean_difference": float(
                h5_high[
                    "explanation_unavailability_rate"
                ]
                -
                h5_low[
                    "explanation_unavailability_rate"
                ]
            ),
            "endpoint_ci_low": np.nan,
            "endpoint_ci_high": np.nan,
            "holm_adjusted_wilcoxon_p": np.nan,
            "rank_biserial": np.nan,
            "fraction_monotonic_nonincreasing": np.nan,
            "descriptive_evidence": (
                "Unavailable explanation counts across rates "
                f"{MASK_RATES} are {h5_unavailable_sequence}. "
                "Monotonic increase in unavailability = "
                f"{h5_monotonic_increase}."
            ),
            "interpretation_scope": (
                "Descriptive availability analysis; unavailable "
                "explanations remain missing and are never assigned "
                "Jaccard = 0."
            ),
        },
    ]

    result = pd.DataFrame(
        records
    )

    return result


# =========================================================================
# FINAL OUTPUT VALIDATION
# =========================================================================


def validate_final_outputs(
    node_rate_df,
    rate_summary_df,
    contrasts_df,
    monotonicity_df,
    h4_node_df,
    h4_summary_df,
    h5_df,
    hypothesis_df,
):
    """
    Final integrity checks before Phase-10 hypothesis artifacts are saved.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-10D Final Hypothesis Validation ==="
    )

    print(
        "=" * 78
    )

    # ---------------------------------------------------------------------
    # Frozen node-rate table.
    # ---------------------------------------------------------------------

    if len(
        node_rate_df
    ) != EXPECTED_NODE_RATE_ROWS:

        raise RuntimeError(
            "Frozen node-rate table size changed."
        )

    print(
        f"Node-rate rows:              "
        f"{len(node_rate_df)}/{EXPECTED_NODE_RATE_ROWS} PASS"
    )

    # ---------------------------------------------------------------------
    # Rate summary.
    # ---------------------------------------------------------------------

    if len(
        rate_summary_df
    ) != len(
        MASK_RATES
    ):

        raise RuntimeError(
            "Unexpected rate-summary row count."
        )

    print(
        "Four-rate descriptive grid: PASS"
    )

    # ---------------------------------------------------------------------
    # Total explanation availability must reproduce Phase-9 totals.
    # ---------------------------------------------------------------------

    total_available = int(
        h5_df[
            "explanations_available"
        ].sum()
    )

    total_unavailable = int(
        h5_df[
            "explanations_unavailable"
        ].sum()
    )

    if (
        total_available
        + total_unavailable
        != EXPECTED_TOTAL_PERTURBATIONS
    ):

        raise RuntimeError(
            "Overall explanation availability accounting failed."
        )

    # Frozen Phase-9 result:
    # 3821 available + 59 unavailable = 3880.

    if total_available != 3821:

        raise RuntimeError(
            "Expected 3821 available explanations from frozen "
            "Phase-9 results."
        )

    if total_unavailable != 59:

        raise RuntimeError(
            "Expected 59 unavailable explanations from frozen "
            "Phase-9 results."
        )

    print(
        f"Explanation availability:   "
        f"{total_available} available / "
        f"{total_unavailable} unavailable PASS"
    )

    # ---------------------------------------------------------------------
    # H5 rate-level counts should reproduce the previously validated
    # counts: 12, 16, 17, 14 unavailable.
    # ---------------------------------------------------------------------

    expected_unavailable = [
        12,
        16,
        17,
        14,
    ]

    observed_unavailable = (
        h5_df.sort_values(
            "nominal_mask_rate"
        )[
            "explanations_unavailable"
        ]
        .astype(int)
        .tolist()
    )

    if observed_unavailable != expected_unavailable:

        raise RuntimeError(
            "H5 rate-level unavailable counts differ from "
            "the frozen Phase-10 results. "
            f"Observed={observed_unavailable}, "
            f"expected={expected_unavailable}."
        )

    print(
        "H5 unavailable counts:      "
        "12 / 16 / 17 / 14 PASS"
    )

    # ---------------------------------------------------------------------
    # H4 endpoint table.
    # ---------------------------------------------------------------------

    if len(
        h4_node_df
    ) != EXPECTED_NODES:

        raise RuntimeError(
            "H4 node endpoint table must contain all 97 target nodes."
        )

    if len(
        h4_summary_df
    ) != 1:

        raise RuntimeError(
            "H4 coupling summary must contain exactly one row."
        )

    h4 = h4_summary_df.iloc[
        0
    ]

    valid_h4_nodes = int(
        h4[
            "nodes_with_valid_endpoint_jaccard"
        ]
    )

    if valid_h4_nodes <= 0:

        raise RuntimeError(
            "H4 has no valid endpoint nodes."
        )

    if valid_h4_nodes > EXPECTED_NODES:

        raise RuntimeError(
            "H4 valid endpoint node count exceeds 97."
        )

    print(
        f"H4 valid endpoint nodes:    "
        f"{valid_h4_nodes}/{EXPECTED_NODES} PASS"
    )

    # ---------------------------------------------------------------------
    # Correlation bounds.
    # ---------------------------------------------------------------------

    for column in [
        "spearman_prediction_vs_explanation_change",
        "pearson_prediction_vs_explanation_change",
    ]:

        value = h4[
            column
        ]

        if pd.notna(
            value
        ):

            if not (
                -1.0
                - FLOAT_TOLERANCE
                <= float(value)
                <= 1.0
                + FLOAT_TOLERANCE
            ):

                raise RuntimeError(
                    f"{column} is outside [-1,1]."
                )

    print(
        "H4 correlation ranges:      PASS"
    )

    # ---------------------------------------------------------------------
    # H4 direction accounting.
    # ---------------------------------------------------------------------

    prediction_direction_total = (
        int(
            h4[
                "nodes_prediction_endpoint_unchanged"
            ]
        )
        +
        int(
            h4[
                "nodes_prediction_endpoint_decreased"
            ]
        )
        +
        int(
            h4[
                "nodes_prediction_endpoint_increased"
            ]
        )
    )

    explanation_direction_total = (
        int(
            h4[
                "nodes_explanation_endpoint_decreased"
            ]
        )
        +
        int(
            h4[
                "nodes_explanation_endpoint_unchanged"
            ]
        )
        +
        int(
            h4[
                "nodes_explanation_endpoint_increased"
            ]
        )
    )

    if prediction_direction_total != valid_h4_nodes:

        raise RuntimeError(
            "H4 prediction direction accounting failed."
        )

    if explanation_direction_total != valid_h4_nodes:

        raise RuntimeError(
            "H4 explanation direction accounting failed."
        )

    print(
        "H4 direction accounting:    PASS"
    )

    # ---------------------------------------------------------------------
    # Endpoint consistency with 04b.
    # ---------------------------------------------------------------------

    h1_contrast = get_contrast(
        contrasts_df,
        "mean_jaccard",
    )

    h2_contrast = get_contrast(
        contrasts_df,
        "mean_jaccard_pred_same",
    )

    h3_contrast = get_contrast(
        contrasts_df,
        "prediction_stability",
    )

    expected_h1 = -0.4475
    expected_h2 = -0.4454
    expected_h3 = -0.0381

    if not np.isclose(
        float(
            h1_contrast[
                "mean_paired_difference"
            ]
        ),
        expected_h1,
        atol=5e-4,
        rtol=0.0,
    ):

        raise RuntimeError(
            "H1 endpoint result differs unexpectedly "
            "from the validated Phase-10C result."
        )

    if not np.isclose(
        float(
            h2_contrast[
                "mean_paired_difference"
            ]
        ),
        expected_h2,
        atol=5e-4,
        rtol=0.0,
    ):

        raise RuntimeError(
            "H2 endpoint result differs unexpectedly "
            "from the validated Phase-10C result."
        )

    if not np.isclose(
        float(
            h3_contrast[
                "mean_paired_difference"
            ]
        ),
        expected_h3,
        atol=5e-4,
        rtol=0.0,
    ):

        raise RuntimeError(
            "H3 endpoint result differs unexpectedly "
            "from the validated Phase-10C result."
        )

    print(
        "H1/H2/H3 endpoint replay:   PASS"
    )

    # ---------------------------------------------------------------------
    # Monotonicity artifact still contains expected outcomes.
    # ---------------------------------------------------------------------

    if len(
        monotonicity_df
    ) != 4:

        raise RuntimeError(
            "Unexpected monotonicity table size."
        )

    print(
        "Monotonicity artifact:       PASS"
    )

    # ---------------------------------------------------------------------
    # Final hypothesis table.
    # ---------------------------------------------------------------------

    if len(
        hypothesis_df
    ) != 5:

        raise RuntimeError(
            "Final hypothesis table must contain exactly five rows."
        )

    expected_hypotheses = {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
    }

    observed_hypotheses = set(
        hypothesis_df[
            "hypothesis_id"
        ].tolist()
    )

    if observed_hypotheses != expected_hypotheses:

        raise RuntimeError(
            "Final hypothesis table does not contain exactly H1-H5."
        )

    if hypothesis_df[
        "hypothesis_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate hypothesis IDs detected."
        )

    print(
        "Final hypothesis rows:      5/5 PASS"
    )

    # ---------------------------------------------------------------------
    # Ensure the temporary structural-perturbation H3 has disappeared.
    # ---------------------------------------------------------------------

    h3_text = hypothesis_df.loc[
        hypothesis_df[
            "hypothesis_id"
        ]
        == "H3",
        "hypothesis",
    ].iloc[0].lower()

    if (
        "structural perturbation"
        in h3_text
    ):

        raise RuntimeError(
            "H3 still contains the temporary structural-perturbation "
            "definition. The pre-specified H3 has not been restored."
        )

    if (
        "prediction agreement"
        not in h3_text
    ):

        raise RuntimeError(
            "H3 does not match the pre-specified prediction-agreement "
            "hypothesis."
        )

    print(
        "Pre-specified H3 restored:  PASS"
    )

    print(
        "\nFinal Phase-10 hypothesis validation: PASSED"
    )


# =========================================================================
# TERMINAL REPORTING
# =========================================================================


def print_rate_level_summary(
    rate_summary_df,
):
    """
    Print the principal four-rate descriptive trajectories.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Frozen Four-Rate Descriptive Summary ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nRate   Pred.Stab.   Expl.Avail.   Mean J   Mean J | Pred.Same"
    )

    print(
        "-" * 68
    )

    for _, row in rate_summary_df.iterrows():

        print(
            f"{row['nominal_mask_rate']:.2f}"
            f"   "
            f"{row['mean_prediction_stability']:.4f}"
            f"       "
            f"{row['mean_explanation_availability']:.4f}"
            f"        "
            f"{row['mean_jaccard']:.4f}"
            f"     "
            f"{row['mean_jaccard_pred_same']:.4f}"
        )


def print_h4_summary(
    h4_summary_df,
):
    """
    Print H4 coupling evidence.
    """

    row = h4_summary_df.iloc[
        0
    ]

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== H4: Prediction / Explanation Coupling ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nEndpoint:"
    )

    print(
        "  20% masking - 1% masking"
    )

    print(
        "\nValid endpoint nodes:"
    )

    print(
        f"  {int(row['nodes_with_valid_endpoint_jaccard'])}"
    )

    print(
        "\nMean endpoint changes:"
    )

    print(
        "  prediction stability: "
        f"{format_float(row['mean_prediction_endpoint_change'])}"
    )

    print(
        "  explanation stability: "
        f"{format_float(row['mean_explanation_endpoint_change'])}"
    )

    print(
        "\nDescriptive coupling:"
    )

    print(
        "  Spearman rho: "
        f"{format_float(row['spearman_prediction_vs_explanation_change'])}"
    )

    print(
        "  Pearson r:    "
        f"{format_float(row['pearson_prediction_vs_explanation_change'])}"
    )

    print(
        "\nDirect decoupling event:"
    )

    print(
        "  prediction endpoint unchanged AND "
        "explanation overlap decreased"
    )

    print(
        "  nodes: "
        f"{int(row['nodes_prediction_unchanged_explanation_decreased'])}"
    )

    print(
        "  fraction of valid endpoint nodes: "
        f"{format_float(row['fraction_prediction_unchanged_explanation_decreased'])}"
    )

    print(
        "\nNo correlation p-value is reported because this is a "
        "descriptive within-graph analysis."
    )


def print_h5_summary(
    h5_df,
):
    """
    Print explanation availability by rate.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== H5: Explanation Availability by Masking Rate ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nRate   Available   Unavailable   Availability   Unavailability"
    )

    print(
        "-" * 72
    )

    for _, row in h5_df.iterrows():

        print(
            f"{row['nominal_mask_rate']:.2f}"
            f"   "
            f"{int(row['explanations_available']):9d}"
            f"   "
            f"{int(row['explanations_unavailable']):11d}"
            f"   "
            f"{row['explanation_availability_rate']:.4f}"
            f"         "
            f"{row['explanation_unavailability_rate']:.4f}"
        )

    unavailable = (
        h5_df[
            "explanations_unavailable"
        ]
        .astype(int)
        .tolist()
    )

    monotonic = bool(
        np.all(
            np.diff(
                np.asarray(
                    unavailable,
                    dtype=float,
                )
            )
            >= 0.0
        )
    )

    print(
        "\nUnavailable-count sequence:"
    )

    print(
        f"  {unavailable}"
    )

    print(
        "\nMonotonic increase in unavailable count:"
    )

    print(
        f"  {monotonic}"
    )


def print_final_hypothesis_summary(
    hypothesis_df,
):
    """
    Print the restored H1-H5 evidence summary.

    This intentionally avoids automatic binary
    'supported/rejected' labels.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Final Pre-Specified H1-H5 Evidence Summary ==="
    )

    print(
        "=" * 78
    )

    for _, row in hypothesis_df.iterrows():

        hypothesis_id = row[
            "hypothesis_id"
        ]

        print(
            f"\n{hypothesis_id}: "
            f"{row['hypothesis']}"
        )

        print(
            "  primary quantity: "
            f"{row['primary_quantity']}"
        )

        print(
            "  analysis type: "
            f"{row['analysis_type']}"
        )

        if hypothesis_id in {
            "H1",
            "H2",
            "H3",
        }:

            print(
                "  endpoint mean difference: "
                f"{format_float(row['endpoint_mean_difference'])}"
            )

            print(
                "  95% paired bootstrap CI: "
                "["
                f"{format_float(row['endpoint_ci_low'])}, "
                f"{format_float(row['endpoint_ci_high'])}"
                "]"
            )

            print(
                "  Holm-adjusted Wilcoxon p: "
                f"{format_scientific(row['holm_adjusted_wilcoxon_p'])}"
            )

            print(
                "  rank-biserial: "
                f"{format_float(row['rank_biserial'])}"
            )

            print(
                "  fraction monotonic nonincreasing: "
                f"{format_float(row['fraction_monotonic_nonincreasing'])}"
            )

        print(
            "  evidence: "
            f"{row['descriptive_evidence']}"
        )

        print(
            "  scope: "
            f"{row['interpretation_scope']}"
        )


# =========================================================================
# MAIN
# =========================================================================


def main():

    print(
        "=" * 78
    )

    print(
        "=== Phase 10D: Finalize Pre-Specified Hypotheses H1-H5 ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nPurpose:"
    )

    print(
        "  Finalize hypothesis-oriented evidence using only already "
        "frozen Phase-9 / Phase-10 artifacts."
    )

    print(
        "\nNo model training, GraphLIME execution, perturbation "
        "generation, or parameter tuning is performed."
    )

    print(
        "\nPre-specified hypotheses:"
    )

    print(
        "  H1 - Explanation overlap declines with masking severity."
    )

    print(
        "  H2 - Explanations can change while predicted class "
        "remains unchanged."
    )

    print(
        "  H3 - Prediction agreement declines with masking severity."
    )

    print(
        "  H4 - Prediction and explanation stability are not "
        "perfectly coupled."
    )

    print(
        "  H5 - Explanation unavailability may increase with "
        "masking severity."
    )

    # =====================================================================
    # Load validated Phase-9 / Phase-10 artifacts.
    # =====================================================================

    print(
        "\n=== Loading Frozen Analysis Artifacts ==="
    )

    node_rate_df = (
        load_node_rate_summary()
    )

    print(
        f"Loaded node-rate summary:       "
        f"{len(node_rate_df)} rows"
    )

    contrasts_df = (
        load_paired_contrasts()
    )

    print(
        f"Loaded paired contrasts:        "
        f"{len(contrasts_df)} rows"
    )

    monotonicity_df = (
        load_monotonicity_summary()
    )

    print(
        f"Loaded monotonicity summary:    "
        f"{len(monotonicity_df)} rows"
    )

    phase9_validation_df = (
        validate_phase9_validation_artifact()
    )

    print(
        f"Loaded Phase-9 validation:      "
        f"{len(phase9_validation_df)} aggregate rows"
    )

    print(
        "Frozen artifact loading:        PASSED"
    )

    # =====================================================================
    # Reconstruct rate-level descriptive summary.
    # =====================================================================

    print(
        "\n=== Reconstructing Four-Rate Descriptive Summary ==="
    )

    rate_summary_df = (
        build_rate_level_summary(
            node_rate_df
        )
    )

    print(
        f"Constructed {len(rate_summary_df)} rate rows."
    )

    # =====================================================================
    # H4 coupling analysis.
    # =====================================================================

    print(
        "\n=== Building H4 Prediction/Explanation Coupling Analysis ==="
    )

    h4_node_df = (
        build_h4_node_endpoint_changes(
            node_rate_df
        )
    )

    h4_summary_df = (
        build_h4_coupling_summary(
            h4_node_df
        )
    )

    print(
        f"Constructed {len(h4_node_df)} "
        "node-level endpoint rows."
    )

    print(
        "Constructed 1 H4 coupling-summary row."
    )

    # =====================================================================
    # H5 explanation availability.
    # =====================================================================

    print(
        "\n=== Building H5 Explanation-Availability Analysis ==="
    )

    h5_df = (
        build_h5_availability_summary(
            node_rate_df
        )
    )

    print(
        f"Constructed {len(h5_df)} rate-level availability rows."
    )

    # =====================================================================
    # Final H1-H5 table.
    # =====================================================================

    print(
        "\n=== Building Final H1-H5 Evidence Table ==="
    )

    hypothesis_df = (
        build_final_hypothesis_summary(
            contrasts_df=contrasts_df,
            monotonicity_df=monotonicity_df,
            h4_summary_df=h4_summary_df,
            h5_df=h5_df,
        )
    )

    print(
        f"Constructed {len(hypothesis_df)} hypothesis rows."
    )

    # =====================================================================
    # Validate everything BEFORE saving.
    # =====================================================================

    validate_final_outputs(
        node_rate_df=node_rate_df,
        rate_summary_df=rate_summary_df,
        contrasts_df=contrasts_df,
        monotonicity_df=monotonicity_df,
        h4_node_df=h4_node_df,
        h4_summary_df=h4_summary_df,
        h5_df=h5_df,
        hypothesis_df=hypothesis_df,
    )

    # =====================================================================
    # Save final artifacts.
    # =====================================================================

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hypothesis_df.to_csv(
        FINAL_HYPOTHESIS_PATH,
        index=False,
    )

    h4_summary_df.to_csv(
        H4_COUPLING_PATH,
        index=False,
    )

    h4_node_df.to_csv(
        H4_NODE_CHANGES_PATH,
        index=False,
    )

    h5_df.to_csv(
        H5_AVAILABILITY_PATH,
        index=False,
    )

    # =====================================================================
    # Terminal reporting.
    # =====================================================================

    print_rate_level_summary(
        rate_summary_df
    )

    print_h4_summary(
        h4_summary_df
    )

    print_h5_summary(
        h5_df
    )

    print_final_hypothesis_summary(
        hypothesis_df
    )

    # =====================================================================
    # Saved artifacts.
    # =====================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Saved Phase-10D Finalization Artifacts ==="
    )

    print(
        "=" * 78
    )

    print(
        FINAL_HYPOTHESIS_PATH
    )

    print(
        H4_COUPLING_PATH
    )

    print(
        H4_NODE_CHANGES_PATH
    )

    print(
        H5_AVAILABILITY_PATH
    )

    # =====================================================================
    # Final status.
    # =====================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-10D Status ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nFinal pre-specified H1-H5 evidence analysis completed "
        "and internally validated."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No frozen experimental data were modified."
    )

    print(
        "No unavailable explanation was assigned Jaccard = 0."
    )

    print(
        "No model, explainer, perturbation, or statistical "
        "hyperparameter was retuned."
    )

    print(
        "\nScientific interpretation remains restricted to the "
        "frozen Cora + GCN + GraphLIME + local active-entry "
        "feature-masking experiment."
    )

    print(
        "\nNext methodological step:"
    )

    print(
        "Review the Phase-10D terminal output and final artifacts."
    )

    print(
        "If all validation checks pass, record E009 in "
        "research/experiment_log.md, formally freeze Phase 10, "
        "and begin canonical Phase 11 visualization."
    )


if __name__ == "__main__":
    main()
