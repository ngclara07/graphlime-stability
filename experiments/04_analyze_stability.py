# experiments/04_analyze_stability.py
#
# Phase 10:
# Statistical aggregation and hypothesis-oriented descriptive analysis
# of the frozen Phase-9 GraphLIME feature-perturbation experiment.
#
# Run from repository root:
#
#   python -m experiments.04_analyze_stability
#
# Required input:
#
#   results/perturbations/feature_mask.csv
#   results/perturbations/feature_mask_validation_summary.csv
#
# Outputs:
#
#   results/analysis/rate_summary.csv
#   results/analysis/node_rate_summary.csv
#   results/analysis/prediction_conditioned_summary.csv
#   results/analysis/explanation_availability_summary.csv
#   results/analysis/failure_summary.csv
#
# IMPORTANT METHODOLOGICAL PRINCIPLES
# -----------------------------------
#
# 1. The raw perturbation observations are NOT treated as 3,880
#    independent experimental units.
#
# 2. Each target node is the primary unit for across-rate summaries.
#    The 10 perturbation seeds are repeated perturbations within a node.
#
# 3. Jaccard similarity is summarized only when a perturbed GraphLIME
#    explanation is actually available.
#
# 4. Unavailable explanations are NOT assigned Jaccard = 0.
#
# 5. Explanation availability is analyzed separately because conditioning
#    Jaccard on available explanations can otherwise conceal systematic
#    explanation failure at stronger perturbation levels.
#
# 6. Prediction-conditioned explanation stability is calculated only for
#    observations where:
#
#       pred_same == True
#       AND
#       explanation_available == True
#
# 7. This script performs descriptive statistical aggregation.
#    It does NOT claim causal effects and does NOT automatically declare
#    hypotheses supported/rejected.
#
# 8. Confidence intervals are node-level bootstrap confidence intervals:
#    nodes, rather than individual perturbation observations, are resampled.
#
# Frozen experiment:
#
#   Dataset: Cora
#   Baseline explainable nodes: 97
#   NUM_HOPS: 2
#   TOP_K: 10
#   RHO: 0.03
#   Mask rates: 0.01, 0.05, 0.10, 0.20
#   Perturbation seeds: 101-110


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FROZEN PHASE-9 CONFIGURATION
# ============================================================

EXPECTED_NODES = 97

MASK_RATES = [
    0.01,
    0.05,
    0.10,
    0.20,
]

PERTURBATION_SEEDS = [
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
]

NUM_HOPS = 2
TOP_K = 10
RHO = 0.03

EXPECTED_OBSERVATIONS = (
    EXPECTED_NODES
    * len(MASK_RATES)
    * len(PERTURBATION_SEEDS)
)

EXPECTED_OBSERVATIONS_PER_RATE = (
    EXPECTED_NODES
    * len(PERTURBATION_SEEDS)
)

EXPECTED_OBSERVATIONS_PER_NODE_RATE = (
    len(PERTURBATION_SEEDS)
)


# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================

# Reproducible node-level bootstrap.
BOOTSTRAP_SEED = 20260917

# 10,000 bootstrap samples is sufficiently large for the current
# 97-node analysis while remaining practical for this small dataset.
N_BOOTSTRAP = 10_000

CONFIDENCE_LEVEL = 0.95

FLOAT_TOLERANCE = 1e-9


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "results/perturbations/feature_mask.csv"
)

VALIDATION_SUMMARY_PATH = Path(
    "results/perturbations/"
    "feature_mask_validation_summary.csv"
)

OUTPUT_DIR = Path(
    "results/analysis"
)

RATE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "rate_summary.csv"
)

NODE_RATE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "node_rate_summary.csv"
)

PREDICTION_CONDITIONED_SUMMARY_PATH = (
    OUTPUT_DIR
    / "prediction_conditioned_summary.csv"
)

EXPLANATION_AVAILABILITY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "explanation_availability_summary.csv"
)

FAILURE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "failure_summary.csv"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "node_id",
    "true_label",
    "nominal_mask_rate",
    "perturbation_seed",
    "num_hops",
    "top_k",
    "rho",
    "active_entries_before",
    "num_entries_masked",
    "realized_mask_rate",
    "baseline_predicted_label",
    "perturbed_predicted_label",
    "pred_same",
    "baseline_confidence",
    "perturbed_baseline_class_confidence",
    "baseline_class_confidence_delta",
    "perturbed_max_confidence",
    "perturbed_output_kernel_valid",
    "perturbed_solver_converged",
    "optimizer_iterations",
    "perturbed_num_nonzero",
    "perturbed_top_k_available",
    "explanation_available",
    "jaccard",
    "intersection_size",
    "union_size",
    "failure_reason",
}


# ============================================================
# BOOLEAN PARSING
# ============================================================


def parse_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Safely parse a CSV boolean column.

    This deliberately avoids astype(bool), because a non-empty
    string such as "False" would otherwise evaluate to True.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        1.0: True,
        0.0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "TRUE": True,
        "FALSE": False,
        "1": True,
        "0": False,
    }

    converted = series.map(mapping)

    invalid = (
        converted.isna()
        & series.notna()
    )

    if invalid.any():

        invalid_values = (
            series.loc[invalid]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains "
            f"unrecognized boolean values: "
            f"{invalid_values[:10]}"
        )

    if converted.isna().any():

        raise ValueError(
            f"Column '{column_name}' contains "
            "missing boolean values."
        )

    return converted.astype(bool)


# ============================================================
# VALIDATION-SUMMARY PARSING
# ============================================================


def validation_summary_all_pass(
    validation_df: pd.DataFrame,
) -> bool:
    """
    Verify that the Phase-9 aggregate validation artifact
    contains only passing checks.
    """

    required = {
        "check_name",
        "passed",
    }

    missing = (
        required
        - set(validation_df.columns)
    )

    if missing:

        raise RuntimeError(
            "Validation-summary artifact is missing "
            f"required columns: {sorted(missing)}"
        )

    passed = parse_boolean_series(
        validation_df["passed"],
        "validation summary passed",
    )

    return bool(passed.all())


# ============================================================
# BASIC STATISTICAL HELPERS
# ============================================================


def safe_mean(
    values,
):
    """
    Mean after removing non-finite values.

    Returns NaN if no finite observations remain.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return np.nan

    return float(
        np.mean(array)
    )


def safe_std(
    values,
):
    """
    Sample standard deviation after removing non-finite values.

    Returns NaN when fewer than two finite observations exist.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) < 2:
        return np.nan

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def safe_median(
    values,
):
    """
    Median after removing non-finite values.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return np.nan

    return float(
        np.median(array)
    )


def safe_min(
    values,
):
    """
    Minimum after removing non-finite values.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return np.nan

    return float(
        np.min(array)
    )


def safe_max(
    values,
):
    """
    Maximum after removing non-finite values.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if len(array) == 0:
        return np.nan

    return float(
        np.max(array)
    )


# ============================================================
# NODE-LEVEL BOOTSTRAP
# ============================================================


def bootstrap_mean_ci(
    values,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
):
    """
    Bootstrap confidence interval for a mean.

    IMPORTANT:
    The input values must already represent NODE-LEVEL
    statistics.

    Therefore, the resampling unit here is the target node,
    not an individual perturbation observation.

    Returns:
        mean,
        lower_ci,
        upper_ci,
        n_finite
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    n = len(array)

    if n == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
        )

    mean_value = float(
        np.mean(array)
    )

    if n == 1:

        return (
            mean_value,
            mean_value,
            mean_value,
            1,
        )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(
        n_bootstrap
    ):

        sample_indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        bootstrap_sample = (
            array[
                sample_indices
            ]
        )

        bootstrap_means[i] = (
            np.mean(
                bootstrap_sample
            )
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower_percentile = (
        100.0
        * alpha
        / 2.0
    )

    upper_percentile = (
        100.0
        * (
            1.0
            - alpha / 2.0
        )
    )

    lower = float(
        np.percentile(
            bootstrap_means,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            upper_percentile,
        )
    )

    return (
        mean_value,
        lower,
        upper,
        n,
    )


# ============================================================
# FORMATTING HELPERS
# ============================================================


def format_float(
    value,
    digits=4,
):
    """
    Terminal-safe floating-point formatting.
    """

    if pd.isna(value):
        return "NA"

    return f"{float(value):.{digits}f}"


def print_rate_header():
    """
    Common terminal header.
    """

    print(
        "rate   "
        "pred_stability   "
        "explain_avail   "
        "mean_J   "
        "mean_J|pred_same"
    )

    print(
        "-" * 69
    )


# ============================================================
# LOAD AND REVALIDATE FROZEN INPUT
# ============================================================


def load_and_validate_input():
    """
    Load the frozen Phase-9 artifact and perform conservative
    prerequisite checks before any statistical aggregation.
    """

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Required perturbation artifact not found: "
            f"{INPUT_PATH}"
        )

    if not VALIDATION_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            "Required Phase-9 validation artifact not found: "
            f"{VALIDATION_SUMMARY_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_SUMMARY_PATH
    )

    # --------------------------------------------------------
    # Confirm Phase-9 validation passed.
    # --------------------------------------------------------

    if not validation_summary_all_pass(
        validation_df
    ):

        raise RuntimeError(
            "The Phase-9 validation summary contains at "
            "least one failed check. Statistical analysis "
            "must not proceed."
        )

    # --------------------------------------------------------
    # Required raw columns.
    # --------------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "feature_mask.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Safe boolean conversion.
    # --------------------------------------------------------

    boolean_columns = [
        "pred_same",
        "perturbed_output_kernel_valid",
        "perturbed_solver_converged",
        "perturbed_top_k_available",
        "explanation_available",
    ]

    for column in boolean_columns:

        df[column] = (
            parse_boolean_series(
                df[column],
                column,
            )
        )

    # --------------------------------------------------------
    # Basic frozen-design checks.
    # --------------------------------------------------------

    if len(df) != EXPECTED_OBSERVATIONS:

        raise RuntimeError(
            "Unexpected observation count. "
            f"Observed={len(df)}, "
            f"expected={EXPECTED_OBSERVATIONS}."
        )

    if df["node_id"].nunique() != EXPECTED_NODES:

        raise RuntimeError(
            "Unexpected number of target nodes."
        )

    observed_rates = set(
        df[
            "nominal_mask_rate"
        ]
        .astype(float)
        .unique()
        .tolist()
    )

    if observed_rates != set(MASK_RATES):

        raise RuntimeError(
            "Observed perturbation rates do not match "
            "the frozen Phase-9 protocol."
        )

    observed_seeds = set(
        df[
            "perturbation_seed"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    if observed_seeds != set(
        PERTURBATION_SEEDS
    ):

        raise RuntimeError(
            "Observed perturbation seeds do not match "
            "the frozen Phase-9 protocol."
        )

    if not (
        df["num_hops"]
        == NUM_HOPS
    ).all():

        raise RuntimeError(
            "NUM_HOPS differs from frozen protocol."
        )

    if not (
        df["top_k"]
        == TOP_K
    ).all():

        raise RuntimeError(
            "TOP_K differs from frozen protocol."
        )

    if not np.allclose(
        df["rho"].astype(float),
        RHO,
        atol=FLOAT_TOLERANCE,
        rtol=0.0,
    ):

        raise RuntimeError(
            "RHO differs from frozen protocol."
        )

    # --------------------------------------------------------
    # Ensure unique experimental keys.
    # --------------------------------------------------------

    key_columns = [
        "node_id",
        "nominal_mask_rate",
        "perturbation_seed",
    ]

    if df.duplicated(
        subset=key_columns
    ).any():

        raise RuntimeError(
            "Duplicate node-rate-seed observations found."
        )

    return (
        df,
        validation_df,
    )


# ============================================================
# NODE-RATE AGGREGATION
# ============================================================


def build_node_rate_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the primary repeated-measure summary.

    Each row represents one:
        target node x perturbation rate

    The 10 perturbation seeds are aggregated within that row.

    Expected rows:
        97 nodes x 4 rates = 388
    """

    records = []

    grouped = df.groupby(
        [
            "node_id",
            "nominal_mask_rate",
        ],
        sort=True,
    )

    for (
        node_id,
        rate,
    ), group in grouped:

        if len(group) != (
            EXPECTED_OBSERVATIONS_PER_NODE_RATE
        ):

            raise RuntimeError(
                f"Node {node_id}, rate {rate} has "
                f"{len(group)} observations; expected "
                f"{EXPECTED_OBSERVATIONS_PER_NODE_RATE}."
            )

        # ----------------------------------------------------
        # Prediction stability.
        # ----------------------------------------------------

        prediction_stability = float(
            group[
                "pred_same"
            ].mean()
        )

        prediction_change_rate = (
            1.0
            - prediction_stability
        )

        num_prediction_same = int(
            group[
                "pred_same"
            ].sum()
        )

        num_prediction_changed = int(
            len(group)
            - num_prediction_same
        )

        # ----------------------------------------------------
        # Explanation availability.
        # ----------------------------------------------------

        available_mask = (
            group[
                "explanation_available"
            ]
        )

        num_explanations_available = int(
            available_mask.sum()
        )

        num_explanations_unavailable = int(
            len(group)
            - num_explanations_available
        )

        explanation_availability = float(
            available_mask.mean()
        )

        # ----------------------------------------------------
        # Jaccard among valid explanations.
        #
        # IMPORTANT:
        # unavailable explanations remain missing.
        # ----------------------------------------------------

        available_jaccard = (
            group.loc[
                available_mask,
                "jaccard",
            ]
            .astype(float)
        )

        mean_jaccard = safe_mean(
            available_jaccard
        )

        std_jaccard = safe_std(
            available_jaccard
        )

        median_jaccard = safe_median(
            available_jaccard
        )

        min_jaccard = safe_min(
            available_jaccard
        )

        max_jaccard = safe_max(
            available_jaccard
        )

        # ----------------------------------------------------
        # Prediction-conditioned explanation stability.
        #
        # Only observations satisfying BOTH:
        #
        #   prediction unchanged
        #   explanation available
        #
        # contribute.
        # ----------------------------------------------------

        conditioned_mask = (
            group[
                "pred_same"
            ]
            &
            group[
                "explanation_available"
            ]
        )

        conditioned_jaccard = (
            group.loc[
                conditioned_mask,
                "jaccard",
            ]
            .astype(float)
        )

        num_conditioned = int(
            conditioned_mask.sum()
        )

        mean_jaccard_pred_same = (
            safe_mean(
                conditioned_jaccard
            )
        )

        std_jaccard_pred_same = (
            safe_std(
                conditioned_jaccard
            )
        )

        median_jaccard_pred_same = (
            safe_median(
                conditioned_jaccard
            )
        )

        # ----------------------------------------------------
        # Jaccard when prediction changed.
        #
        # This is secondary/descriptive but useful for
        # distinguishing explanation changes associated with
        # class changes from explanation changes occurring
        # while the class is stable.
        # ----------------------------------------------------

        changed_condition_mask = (
            (~group["pred_same"])
            &
            group[
                "explanation_available"
            ]
        )

        changed_jaccard = (
            group.loc[
                changed_condition_mask,
                "jaccard",
            ]
            .astype(float)
        )

        num_prediction_changed_with_explanation = int(
            changed_condition_mask.sum()
        )

        mean_jaccard_pred_changed = (
            safe_mean(
                changed_jaccard
            )
        )

        # ----------------------------------------------------
        # Confidence changes.
        #
        # baseline_class_confidence_delta =
        #
        # perturbed probability assigned to original baseline
        # class - baseline probability assigned to that class.
        # ----------------------------------------------------

        confidence_delta = (
            group[
                "baseline_class_confidence_delta"
            ]
            .astype(float)
        )

        mean_confidence_delta = (
            safe_mean(
                confidence_delta
            )
        )

        mean_absolute_confidence_delta = (
            safe_mean(
                np.abs(
                    confidence_delta
                )
            )
        )

        median_confidence_delta = (
            safe_median(
                confidence_delta
            )
        )

        # ----------------------------------------------------
        # Masking metadata.
        # ----------------------------------------------------

        mean_realized_mask_rate = (
            safe_mean(
                group[
                    "realized_mask_rate"
                ]
            )
        )

        mean_num_entries_masked = (
            safe_mean(
                group[
                    "num_entries_masked"
                ]
            )
        )

        # ----------------------------------------------------
        # Failure categories.
        # ----------------------------------------------------

        unavailable_group = (
            group.loc[
                ~available_mask
            ]
        )

        num_fewer_than_top_k = int(
            (
                unavailable_group[
                    "failure_reason"
                ]
                == "fewer_than_top_k_nonzero"
            ).sum()
        )

        num_optimizer_nonconvergence = int(
            (
                unavailable_group[
                    "failure_reason"
                ]
                == "optimizer_nonconvergence"
            ).sum()
        )

        # ----------------------------------------------------
        # Node-level baseline metadata.
        # ----------------------------------------------------

        true_labels = (
            group[
                "true_label"
            ]
            .unique()
        )

        baseline_predictions = (
            group[
                "baseline_predicted_label"
            ]
            .unique()
        )

        baseline_confidences = (
            group[
                "baseline_confidence"
            ]
            .unique()
        )

        active_entries = (
            group[
                "active_entries_before"
            ]
            .unique()
        )

        if len(true_labels) != 1:

            raise RuntimeError(
                f"Node {node_id} has inconsistent true labels."
            )

        if len(baseline_predictions) != 1:

            raise RuntimeError(
                f"Node {node_id} has inconsistent "
                "baseline predictions."
            )

        if len(baseline_confidences) != 1:

            raise RuntimeError(
                f"Node {node_id} has inconsistent "
                "baseline confidence."
            )

        if len(active_entries) != 1:

            raise RuntimeError(
                f"Node {node_id} has inconsistent "
                "active-entry counts."
            )

        records.append(
            {
                "node_id": int(
                    node_id
                ),
                "nominal_mask_rate": float(
                    rate
                ),
                "true_label": int(
                    true_labels[0]
                ),
                "baseline_predicted_label": int(
                    baseline_predictions[0]
                ),
                "baseline_confidence": float(
                    baseline_confidences[0]
                ),
                "active_entries_before": int(
                    active_entries[0]
                ),
                "num_perturbations": int(
                    len(group)
                ),
                "mean_realized_mask_rate": (
                    mean_realized_mask_rate
                ),
                "mean_num_entries_masked": (
                    mean_num_entries_masked
                ),
                "num_prediction_same": (
                    num_prediction_same
                ),
                "num_prediction_changed": (
                    num_prediction_changed
                ),
                "prediction_stability": (
                    prediction_stability
                ),
                "prediction_change_rate": (
                    prediction_change_rate
                ),
                "num_explanations_available": (
                    num_explanations_available
                ),
                "num_explanations_unavailable": (
                    num_explanations_unavailable
                ),
                "explanation_availability": (
                    explanation_availability
                ),
                "mean_jaccard": (
                    mean_jaccard
                ),
                "std_jaccard": (
                    std_jaccard
                ),
                "median_jaccard": (
                    median_jaccard
                ),
                "min_jaccard": (
                    min_jaccard
                ),
                "max_jaccard": (
                    max_jaccard
                ),
                "num_pred_same_with_explanation": (
                    num_conditioned
                ),
                "mean_jaccard_pred_same": (
                    mean_jaccard_pred_same
                ),
                "std_jaccard_pred_same": (
                    std_jaccard_pred_same
                ),
                "median_jaccard_pred_same": (
                    median_jaccard_pred_same
                ),
                "num_pred_changed_with_explanation": (
                    num_prediction_changed_with_explanation
                ),
                "mean_jaccard_pred_changed": (
                    mean_jaccard_pred_changed
                ),
                "mean_confidence_delta": (
                    mean_confidence_delta
                ),
                "mean_absolute_confidence_delta": (
                    mean_absolute_confidence_delta
                ),
                "median_confidence_delta": (
                    median_confidence_delta
                ),
                "num_fewer_than_top_k_nonzero": (
                    num_fewer_than_top_k
                ),
                "num_optimizer_nonconvergence": (
                    num_optimizer_nonconvergence
                ),
            }
        )

    result = pd.DataFrame(
        records
    )

    expected_rows = (
        EXPECTED_NODES
        * len(MASK_RATES)
    )

    if len(result) != expected_rows:

        raise RuntimeError(
            "Unexpected node-rate summary size. "
            f"Observed={len(result)}, "
            f"expected={expected_rows}."
        )

    # Every node should occur once at every rate.
    node_counts = (
        result.groupby(
            "node_id"
        )
        .size()
    )

    if not (
        node_counts
        == len(MASK_RATES)
    ).all():

        raise RuntimeError(
            "At least one node does not have all "
            "four rate-level summaries."
        )

    return result


# ============================================================
# RATE-LEVEL PRIMARY SUMMARY
# ============================================================


def build_rate_summary(
    df: pd.DataFrame,
    node_rate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the primary across-node descriptive summary.

    Each rate-level mean is calculated from the 97 node-level
    summaries rather than directly treating all 970 raw
    observations as independent.

    Confidence intervals are obtained by bootstrapping nodes.
    """

    records = []

    # Separate reproducible RNG stream for this output.
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    for rate in MASK_RATES:

        raw_rate = df.loc[
            np.isclose(
                df[
                    "nominal_mask_rate"
                ].astype(float),
                rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            )
        ].copy()

        node_rate = node_rate_df.loc[
            np.isclose(
                node_rate_df[
                    "nominal_mask_rate"
                ].astype(float),
                rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            )
        ].copy()

        if len(raw_rate) != (
            EXPECTED_OBSERVATIONS_PER_RATE
        ):

            raise RuntimeError(
                f"Rate {rate} has {len(raw_rate)} raw "
                f"observations; expected "
                f"{EXPECTED_OBSERVATIONS_PER_RATE}."
            )

        if len(node_rate) != EXPECTED_NODES:

            raise RuntimeError(
                f"Rate {rate} has {len(node_rate)} node "
                f"summaries; expected {EXPECTED_NODES}."
            )

        # ----------------------------------------------------
        # Prediction stability.
        # ----------------------------------------------------

        (
            prediction_mean,
            prediction_ci_low,
            prediction_ci_high,
            prediction_n,
        ) = bootstrap_mean_ci(
            node_rate[
                "prediction_stability"
            ],
            rng,
        )

        # ----------------------------------------------------
        # Explanation availability.
        # ----------------------------------------------------

        (
            availability_mean,
            availability_ci_low,
            availability_ci_high,
            availability_n,
        ) = bootstrap_mean_ci(
            node_rate[
                "explanation_availability"
            ],
            rng,
        )

        # ----------------------------------------------------
        # Explanation stability.
        #
        # Nodes with zero available explanations at a rate
        # have NaN mean_jaccard and are excluded from the
        # Jaccard mean, but remain represented in the
        # availability analysis.
        # ----------------------------------------------------

        (
            jaccard_mean,
            jaccard_ci_low,
            jaccard_ci_high,
            jaccard_n,
        ) = bootstrap_mean_ci(
            node_rate[
                "mean_jaccard"
            ],
            rng,
        )

        # ----------------------------------------------------
        # Prediction-conditioned explanation stability.
        # ----------------------------------------------------

        (
            conditioned_mean,
            conditioned_ci_low,
            conditioned_ci_high,
            conditioned_n,
        ) = bootstrap_mean_ci(
            node_rate[
                "mean_jaccard_pred_same"
            ],
            rng,
        )

        # ----------------------------------------------------
        # Absolute confidence change.
        # ----------------------------------------------------

        (
            absolute_confidence_delta_mean,
            absolute_confidence_delta_ci_low,
            absolute_confidence_delta_ci_high,
            absolute_confidence_delta_n,
        ) = bootstrap_mean_ci(
            node_rate[
                "mean_absolute_confidence_delta"
            ],
            rng,
        )

        # ----------------------------------------------------
        # Raw counts retained for transparency.
        # ----------------------------------------------------

        num_prediction_same_raw = int(
            raw_rate[
                "pred_same"
            ].sum()
        )

        num_prediction_changed_raw = int(
            len(raw_rate)
            - num_prediction_same_raw
        )

        num_explanations_available_raw = int(
            raw_rate[
                "explanation_available"
            ].sum()
        )

        num_explanations_unavailable_raw = int(
            len(raw_rate)
            - num_explanations_available_raw
        )

        conditioned_raw_mask = (
            raw_rate[
                "pred_same"
            ]
            &
            raw_rate[
                "explanation_available"
            ]
        )

        num_pred_same_with_explanation_raw = int(
            conditioned_raw_mask.sum()
        )

        num_fewer_than_top_k = int(
            (
                raw_rate[
                    "failure_reason"
                ]
                == "fewer_than_top_k_nonzero"
            ).sum()
        )

        num_optimizer_nonconvergence = int(
            (
                raw_rate[
                    "failure_reason"
                ]
                == "optimizer_nonconvergence"
            ).sum()
        )

        records.append(
            {
                "nominal_mask_rate": float(
                    rate
                ),
                "num_nodes": int(
                    len(node_rate)
                ),
                "num_raw_observations": int(
                    len(raw_rate)
                ),
                "mean_realized_mask_rate": (
                    safe_mean(
                        raw_rate[
                            "realized_mask_rate"
                        ]
                    )
                ),
                "prediction_stability_mean": (
                    prediction_mean
                ),
                "prediction_stability_ci_low": (
                    prediction_ci_low
                ),
                "prediction_stability_ci_high": (
                    prediction_ci_high
                ),
                "prediction_stability_nodes": (
                    prediction_n
                ),
                "num_prediction_same_raw": (
                    num_prediction_same_raw
                ),
                "num_prediction_changed_raw": (
                    num_prediction_changed_raw
                ),
                "explanation_availability_mean": (
                    availability_mean
                ),
                "explanation_availability_ci_low": (
                    availability_ci_low
                ),
                "explanation_availability_ci_high": (
                    availability_ci_high
                ),
                "explanation_availability_nodes": (
                    availability_n
                ),
                "num_explanations_available_raw": (
                    num_explanations_available_raw
                ),
                "num_explanations_unavailable_raw": (
                    num_explanations_unavailable_raw
                ),
                "mean_jaccard": (
                    jaccard_mean
                ),
                "mean_jaccard_ci_low": (
                    jaccard_ci_low
                ),
                "mean_jaccard_ci_high": (
                    jaccard_ci_high
                ),
                "jaccard_contributing_nodes": (
                    jaccard_n
                ),
                "mean_jaccard_pred_same": (
                    conditioned_mean
                ),
                "mean_jaccard_pred_same_ci_low": (
                    conditioned_ci_low
                ),
                "mean_jaccard_pred_same_ci_high": (
                    conditioned_ci_high
                ),
                "pred_same_jaccard_contributing_nodes": (
                    conditioned_n
                ),
                "num_pred_same_with_explanation_raw": (
                    num_pred_same_with_explanation_raw
                ),
                "mean_absolute_confidence_delta": (
                    absolute_confidence_delta_mean
                ),
                "mean_absolute_confidence_delta_ci_low": (
                    absolute_confidence_delta_ci_low
                ),
                "mean_absolute_confidence_delta_ci_high": (
                    absolute_confidence_delta_ci_high
                ),
                "confidence_delta_contributing_nodes": (
                    absolute_confidence_delta_n
                ),
                "num_fewer_than_top_k_nonzero": (
                    num_fewer_than_top_k
                ),
                "num_optimizer_nonconvergence": (
                    num_optimizer_nonconvergence
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# PREDICTION-CONDITIONED SUMMARY
# ============================================================


def build_prediction_conditioned_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize explanation stability separately for:

        prediction unchanged
        prediction changed

    Jaccard is still restricted to observations with an
    available perturbed explanation.

    To preserve node-aware inference, each node is first
    summarized within rate x prediction condition. The
    across-node mean and bootstrap CI are then computed.
    """

    node_records = []

    for (
        node_id,
        rate,
        pred_same,
    ), group in df.groupby(
        [
            "node_id",
            "nominal_mask_rate",
            "pred_same",
        ],
        sort=True,
    ):

        available_group = group.loc[
            group[
                "explanation_available"
            ]
        ]

        node_records.append(
            {
                "node_id": int(
                    node_id
                ),
                "nominal_mask_rate": float(
                    rate
                ),
                "prediction_unchanged": bool(
                    pred_same
                ),
                "num_raw_observations": int(
                    len(group)
                ),
                "num_available_explanations": int(
                    len(available_group)
                ),
                "explanation_availability": float(
                    group[
                        "explanation_available"
                    ].mean()
                ),
                "mean_jaccard": (
                    safe_mean(
                        available_group[
                            "jaccard"
                        ]
                    )
                ),
                "mean_absolute_confidence_delta": (
                    safe_mean(
                        np.abs(
                            group[
                                "baseline_class_confidence_delta"
                            ].astype(float)
                        )
                    )
                ),
            }
        )

    node_condition_df = pd.DataFrame(
        node_records
    )

    records = []

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + 1
    )

    for rate in MASK_RATES:

        for pred_same in [
            True,
            False,
        ]:

            raw_subset = df.loc[
                np.isclose(
                    df[
                        "nominal_mask_rate"
                    ].astype(float),
                    rate,
                    atol=FLOAT_TOLERANCE,
                    rtol=0.0,
                )
                &
                (
                    df[
                        "pred_same"
                    ]
                    == pred_same
                )
            ]

            node_subset = (
                node_condition_df.loc[
                    np.isclose(
                        node_condition_df[
                            "nominal_mask_rate"
                        ].astype(float),
                        rate,
                        atol=FLOAT_TOLERANCE,
                        rtol=0.0,
                    )
                    &
                    (
                        node_condition_df[
                            "prediction_unchanged"
                        ]
                        == pred_same
                    )
                ]
            )

            # A condition may be absent at a particular rate.
            if len(raw_subset) == 0:

                records.append(
                    {
                        "nominal_mask_rate": float(
                            rate
                        ),
                        "prediction_unchanged": bool(
                            pred_same
                        ),
                        "num_raw_observations": 0,
                        "num_nodes_with_condition": 0,
                        "num_available_explanations": 0,
                        "raw_explanation_availability": np.nan,
                        "mean_node_explanation_availability": np.nan,
                        "explanation_availability_ci_low": np.nan,
                        "explanation_availability_ci_high": np.nan,
                        "jaccard_contributing_nodes": 0,
                        "mean_jaccard": np.nan,
                        "mean_jaccard_ci_low": np.nan,
                        "mean_jaccard_ci_high": np.nan,
                        "mean_absolute_confidence_delta": np.nan,
                        "mean_absolute_confidence_delta_ci_low": np.nan,
                        "mean_absolute_confidence_delta_ci_high": np.nan,
                    }
                )

                continue

            (
                availability_mean,
                availability_ci_low,
                availability_ci_high,
                _,
            ) = bootstrap_mean_ci(
                node_subset[
                    "explanation_availability"
                ],
                rng,
            )

            (
                jaccard_mean,
                jaccard_ci_low,
                jaccard_ci_high,
                jaccard_n,
            ) = bootstrap_mean_ci(
                node_subset[
                    "mean_jaccard"
                ],
                rng,
            )

            (
                confidence_mean,
                confidence_ci_low,
                confidence_ci_high,
                _,
            ) = bootstrap_mean_ci(
                node_subset[
                    "mean_absolute_confidence_delta"
                ],
                rng,
            )

            num_available = int(
                raw_subset[
                    "explanation_available"
                ].sum()
            )

            records.append(
                {
                    "nominal_mask_rate": float(
                        rate
                    ),
                    "prediction_unchanged": bool(
                        pred_same
                    ),
                    "num_raw_observations": int(
                        len(raw_subset)
                    ),
                    "num_nodes_with_condition": int(
                        node_subset[
                            "node_id"
                        ].nunique()
                    ),
                    "num_available_explanations": (
                        num_available
                    ),
                    "raw_explanation_availability": (
                        float(
                            raw_subset[
                                "explanation_available"
                            ].mean()
                        )
                    ),
                    "mean_node_explanation_availability": (
                        availability_mean
                    ),
                    "explanation_availability_ci_low": (
                        availability_ci_low
                    ),
                    "explanation_availability_ci_high": (
                        availability_ci_high
                    ),
                    "jaccard_contributing_nodes": (
                        jaccard_n
                    ),
                    "mean_jaccard": (
                        jaccard_mean
                    ),
                    "mean_jaccard_ci_low": (
                        jaccard_ci_low
                    ),
                    "mean_jaccard_ci_high": (
                        jaccard_ci_high
                    ),
                    "mean_absolute_confidence_delta": (
                        confidence_mean
                    ),
                    "mean_absolute_confidence_delta_ci_low": (
                        confidence_ci_low
                    ),
                    "mean_absolute_confidence_delta_ci_high": (
                        confidence_ci_high
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# EXPLANATION-AVAILABILITY SUMMARY
# ============================================================


def build_explanation_availability_summary(
    df: pd.DataFrame,
    node_rate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dedicated summary of explanation availability.

    This table exists because missing explanations are a
    scientifically relevant outcome and must not disappear
    inside a Jaccard-only analysis.
    """

    records = []

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + 2
    )

    for rate in MASK_RATES:

        raw_subset = df.loc[
            np.isclose(
                df[
                    "nominal_mask_rate"
                ].astype(float),
                rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            )
        ]

        node_subset = node_rate_df.loc[
            np.isclose(
                node_rate_df[
                    "nominal_mask_rate"
                ].astype(float),
                rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            )
        ]

        (
            availability_mean,
            availability_ci_low,
            availability_ci_high,
            availability_n,
        ) = bootstrap_mean_ci(
            node_subset[
                "explanation_availability"
            ],
            rng,
        )

        num_available = int(
            raw_subset[
                "explanation_available"
            ].sum()
        )

        num_unavailable = int(
            len(raw_subset)
            - num_available
        )

        fewer_than_top_k = int(
            (
                raw_subset[
                    "failure_reason"
                ]
                == "fewer_than_top_k_nonzero"
            ).sum()
        )

        optimizer_nonconvergence = int(
            (
                raw_subset[
                    "failure_reason"
                ]
                == "optimizer_nonconvergence"
            ).sum()
        )

        nodes_with_any_unavailable = int(
            node_subset.loc[
                node_subset[
                    "num_explanations_unavailable"
                ]
                > 0,
                "node_id",
            ].nunique()
        )

        nodes_with_all_available = int(
            node_subset.loc[
                node_subset[
                    "num_explanations_unavailable"
                ]
                == 0,
                "node_id",
            ].nunique()
        )

        records.append(
            {
                "nominal_mask_rate": float(
                    rate
                ),
                "num_nodes": int(
                    len(node_subset)
                ),
                "num_raw_observations": int(
                    len(raw_subset)
                ),
                "num_available": (
                    num_available
                ),
                "num_unavailable": (
                    num_unavailable
                ),
                "raw_availability_rate": (
                    float(
                        num_available
                        / len(raw_subset)
                    )
                ),
                "mean_node_availability": (
                    availability_mean
                ),
                "mean_node_availability_ci_low": (
                    availability_ci_low
                ),
                "mean_node_availability_ci_high": (
                    availability_ci_high
                ),
                "availability_contributing_nodes": (
                    availability_n
                ),
                "nodes_with_all_explanations_available": (
                    nodes_with_all_available
                ),
                "nodes_with_any_unavailable_explanation": (
                    nodes_with_any_unavailable
                ),
                "fewer_than_top_k_nonzero": (
                    fewer_than_top_k
                ),
                "optimizer_nonconvergence": (
                    optimizer_nonconvergence
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# FAILURE SUMMARY
# ============================================================


def build_failure_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize explanation-unavailability reasons by rate.

    Includes an ALL-rates row for transparent accounting.
    """

    unavailable = df.loc[
        ~df[
            "explanation_available"
        ]
    ].copy()

    records = []

    # --------------------------------------------------------
    # Rate-specific failure rows.
    # --------------------------------------------------------

    for rate in MASK_RATES:

        rate_all = df.loc[
            np.isclose(
                df[
                    "nominal_mask_rate"
                ].astype(float),
                rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            )
        ]

        rate_unavailable = (
            unavailable.loc[
                np.isclose(
                    unavailable[
                        "nominal_mask_rate"
                    ].astype(float),
                    rate,
                    atol=FLOAT_TOLERANCE,
                    rtol=0.0,
                )
            ]
        )

        if len(rate_unavailable) == 0:

            records.append(
                {
                    "nominal_mask_rate": float(
                        rate
                    ),
                    "failure_reason": "none",
                    "failure_count": 0,
                    "rate_total_observations": int(
                        len(rate_all)
                    ),
                    "failure_fraction_of_rate": 0.0,
                    "failure_fraction_among_unavailable": np.nan,
                    "unique_nodes_affected": 0,
                }
            )

            continue

        counts = (
            rate_unavailable[
                "failure_reason"
            ]
            .value_counts(
                dropna=False
            )
        )

        for (
            failure_reason,
            count,
        ) in counts.items():

            reason_subset = (
                rate_unavailable.loc[
                    rate_unavailable[
                        "failure_reason"
                    ]
                    == failure_reason
                ]
            )

            records.append(
                {
                    "nominal_mask_rate": float(
                        rate
                    ),
                    "failure_reason": str(
                        failure_reason
                    ),
                    "failure_count": int(
                        count
                    ),
                    "rate_total_observations": int(
                        len(rate_all)
                    ),
                    "failure_fraction_of_rate": float(
                        count
                        / len(rate_all)
                    ),
                    "failure_fraction_among_unavailable": float(
                        count
                        / len(rate_unavailable)
                    ),
                    "unique_nodes_affected": int(
                        reason_subset[
                            "node_id"
                        ].nunique()
                    ),
                }
            )

    # --------------------------------------------------------
    # Overall failure rows.
    # --------------------------------------------------------

    if len(unavailable) == 0:

        records.append(
            {
                "nominal_mask_rate": "ALL",
                "failure_reason": "none",
                "failure_count": 0,
                "rate_total_observations": int(
                    len(df)
                ),
                "failure_fraction_of_rate": 0.0,
                "failure_fraction_among_unavailable": np.nan,
                "unique_nodes_affected": 0,
            }
        )

    else:

        overall_counts = (
            unavailable[
                "failure_reason"
            ]
            .value_counts(
                dropna=False
            )
        )

        for (
            failure_reason,
            count,
        ) in overall_counts.items():

            reason_subset = (
                unavailable.loc[
                    unavailable[
                        "failure_reason"
                    ]
                    == failure_reason
                ]
            )

            records.append(
                {
                    "nominal_mask_rate": "ALL",
                    "failure_reason": str(
                        failure_reason
                    ),
                    "failure_count": int(
                        count
                    ),
                    "rate_total_observations": int(
                        len(df)
                    ),
                    "failure_fraction_of_rate": float(
                        count
                        / len(df)
                    ),
                    "failure_fraction_among_unavailable": float(
                        count
                        / len(unavailable)
                    ),
                    "unique_nodes_affected": int(
                        reason_subset[
                            "node_id"
                        ].nunique()
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# OUTPUT VALIDATION
# ============================================================


def validate_analysis_outputs(
    df: pd.DataFrame,
    node_rate_df: pd.DataFrame,
    rate_summary_df: pd.DataFrame,
    conditioned_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    failure_df: pd.DataFrame,
):
    """
    Validate generated analysis tables before saving them.

    This is not hypothesis testing. It verifies aggregation
    accounting and statistical-output integrity.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-10 Analysis Output Validation ==="
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Node-rate table.
    # --------------------------------------------------------

    expected_node_rate_rows = (
        EXPECTED_NODES
        * len(MASK_RATES)
    )

    if len(node_rate_df) != (
        expected_node_rate_rows
    ):

        raise RuntimeError(
            "node_rate_summary row-count validation failed."
        )

    print(
        f"Node-rate rows:              "
        f"{len(node_rate_df)}/"
        f"{expected_node_rate_rows} PASS"
    )

    # --------------------------------------------------------
    # Rate table.
    # --------------------------------------------------------

    if len(rate_summary_df) != len(
        MASK_RATES
    ):

        raise RuntimeError(
            "rate_summary row-count validation failed."
        )

    print(
        f"Rate-summary rows:           "
        f"{len(rate_summary_df)}/"
        f"{len(MASK_RATES)} PASS"
    )

    # --------------------------------------------------------
    # Availability table.
    # --------------------------------------------------------

    if len(availability_df) != len(
        MASK_RATES
    ):

        raise RuntimeError(
            "availability-summary row-count validation failed."
        )

    print(
        f"Availability-summary rows:   "
        f"{len(availability_df)}/"
        f"{len(MASK_RATES)} PASS"
    )

    # --------------------------------------------------------
    # Prediction-conditioned table.
    #
    # Expected:
    # 4 rates x 2 conditions = 8 rows.
    # --------------------------------------------------------

    expected_condition_rows = (
        len(MASK_RATES)
        * 2
    )

    if len(conditioned_df) != (
        expected_condition_rows
    ):

        raise RuntimeError(
            "prediction-conditioned summary row-count "
            "validation failed."
        )

    print(
        f"Condition-summary rows:      "
        f"{len(conditioned_df)}/"
        f"{expected_condition_rows} PASS"
    )

    # --------------------------------------------------------
    # Raw explanation accounting.
    # --------------------------------------------------------

    raw_available = int(
        df[
            "explanation_available"
        ].sum()
    )

    raw_unavailable = int(
        len(df)
        - raw_available
    )

    summary_available = int(
        availability_df[
            "num_available"
        ].sum()
    )

    summary_unavailable = int(
        availability_df[
            "num_unavailable"
        ].sum()
    )

    if (
        raw_available
        != summary_available
    ):

        raise RuntimeError(
            "Available-explanation accounting mismatch."
        )

    if (
        raw_unavailable
        != summary_unavailable
    ):

        raise RuntimeError(
            "Unavailable-explanation accounting mismatch."
        )

    print(
        f"Available explanation count: "
        f"{summary_available}/"
        f"{raw_available} PASS"
    )

    print(
        f"Unavailable explanation count: "
        f"{summary_unavailable}/"
        f"{raw_unavailable} PASS"
    )

    # --------------------------------------------------------
    # Failure accounting.
    # --------------------------------------------------------

    overall_failure_rows = (
        failure_df.loc[
            failure_df[
                "nominal_mask_rate"
            ].astype(str)
            == "ALL"
        ]
    )

    if raw_unavailable == 0:

        overall_failure_count = 0

    else:

        overall_failure_count = int(
            overall_failure_rows[
                "failure_count"
            ].sum()
        )

    if (
        overall_failure_count
        != raw_unavailable
    ):

        raise RuntimeError(
            "Failure-summary accounting mismatch."
        )

    print(
        f"Failure accounting:          "
        f"{overall_failure_count}/"
        f"{raw_unavailable} PASS"
    )

    # --------------------------------------------------------
    # Range checks.
    # --------------------------------------------------------

    bounded_columns = [
        "prediction_stability_mean",
        "explanation_availability_mean",
        "mean_jaccard",
        "mean_jaccard_pred_same",
    ]

    for column in bounded_columns:

        values = (
            rate_summary_df[
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

    print(
        "Primary probability/Jaccard ranges: PASS"
    )

    # --------------------------------------------------------
    # CI ordering.
    # --------------------------------------------------------

    ci_triplets = [
        (
            "prediction_stability_mean",
            "prediction_stability_ci_low",
            "prediction_stability_ci_high",
        ),
        (
            "explanation_availability_mean",
            "explanation_availability_ci_low",
            "explanation_availability_ci_high",
        ),
        (
            "mean_jaccard",
            "mean_jaccard_ci_low",
            "mean_jaccard_ci_high",
        ),
        (
            "mean_jaccard_pred_same",
            "mean_jaccard_pred_same_ci_low",
            "mean_jaccard_pred_same_ci_high",
        ),
    ]

    for (
        mean_column,
        low_column,
        high_column,
    ) in ci_triplets:

        valid = rate_summary_df[
            [
                mean_column,
                low_column,
                high_column,
            ]
        ].dropna()

        if not (
            (
                valid[
                    low_column
                ]
                <= valid[
                    mean_column
                ]
            )
            &
            (
                valid[
                    mean_column
                ]
                <= valid[
                    high_column
                ]
            )
        ).all():

            raise RuntimeError(
                f"Confidence-interval ordering failed for "
                f"{mean_column}."
            )

    print(
        "Bootstrap CI ordering:      PASS"
    )

    print(
        "\nAnalysis-output validation:  PASSED"
    )


# ============================================================
# TERMINAL DESCRIPTIVE SUMMARY
# ============================================================


def print_descriptive_summary(
    rate_summary_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    failure_df: pd.DataFrame,
):
    """
    Print descriptive results without automatically converting
    them into hypothesis verdicts.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Primary Descriptive Rate Summary ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nMeans are node-level means across the 97 "
        "target nodes."
    )

    print(
        "Jaccard means exclude unavailable explanations; "
        "availability is reported separately.\n"
    )

    print_rate_header()

    for _, row in (
        rate_summary_df
        .sort_values(
            "nominal_mask_rate"
        )
        .iterrows()
    ):

        print(
            f"{row['nominal_mask_rate']:0.2f}   "
            f"{format_float(row['prediction_stability_mean']):>8}         "
            f"{format_float(row['explanation_availability_mean']):>8}       "
            f"{format_float(row['mean_jaccard']):>7}   "
            f"{format_float(row['mean_jaccard_pred_same']):>8}"
        )

    print(
        "\n=== 95% Node-Level Bootstrap Confidence Intervals ==="
    )

    for _, row in (
        rate_summary_df
        .sort_values(
            "nominal_mask_rate"
        )
        .iterrows()
    ):

        rate = (
            row[
                "nominal_mask_rate"
            ]
        )

        print(
            f"\nMask rate {rate:0.2f}"
        )

        print(
            "  Prediction stability: "
            f"{format_float(row['prediction_stability_mean'])} "
            "["
            f"{format_float(row['prediction_stability_ci_low'])}, "
            f"{format_float(row['prediction_stability_ci_high'])}"
            "]"
        )

        print(
            "  Explanation availability: "
            f"{format_float(row['explanation_availability_mean'])} "
            "["
            f"{format_float(row['explanation_availability_ci_low'])}, "
            f"{format_float(row['explanation_availability_ci_high'])}"
            "]"
        )

        print(
            "  Mean Jaccard: "
            f"{format_float(row['mean_jaccard'])} "
            "["
            f"{format_float(row['mean_jaccard_ci_low'])}, "
            f"{format_float(row['mean_jaccard_ci_high'])}"
            "]"
        )

        print(
            "  Mean Jaccard | prediction unchanged: "
            f"{format_float(row['mean_jaccard_pred_same'])} "
            "["
            f"{format_float(row['mean_jaccard_pred_same_ci_low'])}, "
            f"{format_float(row['mean_jaccard_pred_same_ci_high'])}"
            "]"
        )

    print(
        "\n=== Explanation Availability Counts ==="
    )

    for _, row in (
        availability_df
        .sort_values(
            "nominal_mask_rate"
        )
        .iterrows()
    ):

        print(
            f"rate={row['nominal_mask_rate']:0.2f} | "
            f"available={int(row['num_available'])}/"
            f"{int(row['num_raw_observations'])} | "
            f"unavailable={int(row['num_unavailable'])} | "
            f"nodes with >=1 unavailable="
            f"{int(row['nodes_with_any_unavailable_explanation'])}"
        )

    print(
        "\n=== Explanation-Unavailability Accounting ==="
    )

    overall_failures = (
        failure_df.loc[
            failure_df[
                "nominal_mask_rate"
            ].astype(str)
            == "ALL"
        ]
    )

    if len(overall_failures) == 0:

        print(
            "No unavailable explanations recorded."
        )

    else:

        for _, row in (
            overall_failures
            .sort_values(
                "failure_count",
                ascending=False,
            )
            .iterrows()
        ):

            print(
                f"{row['failure_reason']}: "
                f"{int(row['failure_count'])}"
            )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "=" * 78
    )

    print(
        "=== Phase 10: GraphLIME Stability Analysis ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nStatistical unit:"
    )

    print(
        "  Target node is the primary across-rate unit."
    )

    print(
        "  Perturbation seeds are repeated measurements "
        "within each node."
    )

    print(
        "\nMissing-explanation policy:"
    )

    print(
        "  Unavailable explanations remain missing."
    )

    print(
        "  They are NOT converted to Jaccard = 0."
    )

    print(
        "\nUncertainty:"
    )

    print(
        f"  {int(CONFIDENCE_LEVEL * 100)}% node-level "
        f"bootstrap confidence intervals"
    )

    print(
        f"  Bootstrap resamples: {N_BOOTSTRAP}"
    )

    print(
        f"  Bootstrap seed: {BOOTSTRAP_SEED}"
    )

    # ========================================================
    # Load validated frozen data.
    # ========================================================

    (
        df,
        validation_df,
    ) = load_and_validate_input()

    print(
        "\n=== Loaded Validated Phase-9 Dataset ==="
    )

    print(
        f"Raw observations:           {len(df)}"
    )

    print(
        f"Target nodes:               "
        f"{df['node_id'].nunique()}"
    )

    print(
        f"Perturbation rates:         "
        f"{df['nominal_mask_rate'].nunique()}"
    )

    print(
        f"Perturbation seeds:         "
        f"{df['perturbation_seed'].nunique()}"
    )

    print(
        f"Available explanations:     "
        f"{int(df['explanation_available'].sum())}"
    )

    print(
        f"Unavailable explanations:   "
        f"{int((~df['explanation_available']).sum())}"
    )

    print(
        f"Phase-9 aggregate checks:   "
        f"{len(validation_df)}/{len(validation_df)} PASS"
    )

    # ========================================================
    # Construct analysis tables.
    # ========================================================

    print(
        "\n=== Building Node-Rate Summary ==="
    )

    node_rate_df = (
        build_node_rate_summary(
            df
        )
    )

    print(
        f"Constructed {len(node_rate_df)} "
        "node-rate rows."
    )

    print(
        "\n=== Building Rate Summary ==="
    )

    rate_summary_df = (
        build_rate_summary(
            df,
            node_rate_df,
        )
    )

    print(
        f"Constructed {len(rate_summary_df)} "
        "rate-level rows."
    )

    print(
        "\n=== Building Prediction-Conditioned Summary ==="
    )

    conditioned_df = (
        build_prediction_conditioned_summary(
            df
        )
    )

    print(
        f"Constructed {len(conditioned_df)} "
        "rate-condition rows."
    )

    print(
        "\n=== Building Explanation-Availability Summary ==="
    )

    availability_df = (
        build_explanation_availability_summary(
            df,
            node_rate_df,
        )
    )

    print(
        f"Constructed {len(availability_df)} "
        "availability rows."
    )

    print(
        "\n=== Building Failure Summary ==="
    )

    failure_df = (
        build_failure_summary(
            df
        )
    )

    print(
        f"Constructed {len(failure_df)} "
        "failure-summary rows."
    )

    # ========================================================
    # Validate generated tables before saving.
    # ========================================================

    validate_analysis_outputs(
        df=df,
        node_rate_df=node_rate_df,
        rate_summary_df=rate_summary_df,
        conditioned_df=conditioned_df,
        availability_df=availability_df,
        failure_df=failure_df,
    )

    # ========================================================
    # Save outputs.
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_rate_df.to_csv(
        NODE_RATE_SUMMARY_PATH,
        index=False,
    )

    rate_summary_df.to_csv(
        RATE_SUMMARY_PATH,
        index=False,
    )

    conditioned_df.to_csv(
        PREDICTION_CONDITIONED_SUMMARY_PATH,
        index=False,
    )

    availability_df.to_csv(
        EXPLANATION_AVAILABILITY_SUMMARY_PATH,
        index=False,
    )

    failure_df.to_csv(
        FAILURE_SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # Print descriptive results.
    # ========================================================

    print_descriptive_summary(
        rate_summary_df,
        availability_df,
        failure_df,
    )

    # ========================================================
    # Final status.
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Saved Phase-10 Analysis Artifacts ==="
    )

    print(
        "=" * 78
    )

    print(
        RATE_SUMMARY_PATH
    )

    print(
        NODE_RATE_SUMMARY_PATH
    )

    print(
        PREDICTION_CONDITIONED_SUMMARY_PATH
    )

    print(
        EXPLANATION_AVAILABILITY_SUMMARY_PATH
    )

    print(
        FAILURE_SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-10 Analysis Status ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nPrimary descriptive aggregation completed "
        "and internally validated."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not infer hypothesis support solely from this "
        "script's terminal output."
    )

    print(
        "The generated tables should now be reviewed "
        "systematically before formal hypothesis-oriented "
        "interpretation and figure generation."
    )

    print(
        "\nNext methodological step:"
    )

    print(
        "Review the complete Phase-10 statistical summaries, "
        "then determine the appropriate paired/node-level "
        "hypothesis analyses before Phase 11 visualization."
    )


if __name__ == "__main__":
    main()
