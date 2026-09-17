# experiments/04b_hypothesis_analysis.py
#
# Phase 10C:
# Paired, node-level hypothesis-oriented analysis of GraphLIME
# explanation stability under controlled feature perturbations.
#
# Run from repository root:
#
#   python -m experiments.04b_hypothesis_analysis
#
# Required input:
#
#   results/analysis/node_rate_summary.csv
#
# Outputs:
#
#   results/analysis/paired_rate_contrasts.csv
#   results/analysis/monotonicity_summary.csv
#   results/analysis/hypothesis_summary.csv
#
# -------------------------------------------------------------------------
# PRE-SPECIFIED STATISTICAL PRINCIPLES
# -------------------------------------------------------------------------
#
# 1. The target node is the primary statistical unit.
#
# 2. The 10 perturbation seeds have already been aggregated within each
#    node x perturbation-rate combination by 04_analyze_stability.py.
#
# 3. Rate comparisons are paired because the same target nodes are
#    evaluated at every perturbation rate.
#
# 4. The primary explanation-stability outcome is:
#
#       mean_jaccard
#
#    calculated at node x rate level among available explanations.
#
# 5. Prediction-conditioned explanation stability is analyzed separately:
#
#       mean_jaccard_pred_same
#
# 6. Missing explanations remain missing. They are never assigned
#    Jaccard = 0.
#
# 7. Pairwise uncertainty is quantified using paired node-level bootstrap
#    confidence intervals for mean differences.
#
# 8. Wilcoxon signed-rank tests are secondary inferential checks.
#    They are implemented locally using average tied ranks and a
#    tie-corrected normal approximation, without SciPy.
#
# 9. Holm correction is applied across the six pairwise tests separately
#    within each outcome family.
#
# 10. Effect size is reported using matched-pairs rank-biserial
#     correlation:
#
#         r_rb = (W_positive - W_negative)
#                / (W_positive + W_negative)
#
#     where the paired difference is defined as:
#
#         higher-rate outcome - lower-rate outcome
#
#     Therefore:
#
#         negative r_rb -> outcome tends to decrease at the higher rate
#         positive r_rb -> outcome tends to increase at the higher rate
#
# 11. Monotonicity is assessed descriptively at the node level.
#
# 12. No hypothesis is automatically declared "supported" or "rejected"
#     solely because a p-value crosses an arbitrary threshold.
#
# -------------------------------------------------------------------------


from itertools import combinations
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

EXPECTED_NODE_RATE_ROWS = (
    EXPECTED_NODES
    * len(MASK_RATES)
)

EXPECTED_PAIRWISE_CONTRASTS = math.comb(
    len(MASK_RATES),
    2,
)

TOP_K = 10


# =========================================================================
# STATISTICAL CONFIGURATION
# =========================================================================

BOOTSTRAP_SEED = 20260917

N_BOOTSTRAP = 10_000

CONFIDENCE_LEVEL = 0.95

ALPHA = 0.05

FLOAT_TOLERANCE = 1e-9

# Tolerance used only for deciding whether two floating-point node-level
# summaries should count as equal in descriptive monotonicity checks and
# zero-difference handling.
MONOTONICITY_TOLERANCE = 1e-12


# =========================================================================
# PATHS
# =========================================================================

INPUT_PATH = Path(
    "results/analysis/node_rate_summary.csv"
)

OUTPUT_DIR = Path(
    "results/analysis"
)

PAIRED_CONTRASTS_PATH = (
    OUTPUT_DIR
    / "paired_rate_contrasts.csv"
)

MONOTONICITY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "monotonicity_summary.csv"
)

HYPOTHESIS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "hypothesis_summary.csv"
)


# =========================================================================
# REQUIRED INPUT COLUMNS
# =========================================================================

REQUIRED_COLUMNS = {
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


# =========================================================================
# BASIC HELPERS
# =========================================================================


def finite_array(values):
    """
    Convert values to a one-dimensional float array and retain
    only finite observations.
    """

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    return array[
        np.isfinite(array)
    ]


def safe_mean(values):
    """
    Mean of finite values, or NaN when no finite values exist.
    """

    array = finite_array(values)

    if len(array) == 0:
        return np.nan

    return float(
        np.mean(array)
    )


def safe_std(values):
    """
    Sample standard deviation of finite values.
    """

    array = finite_array(values)

    if len(array) < 2:
        return np.nan

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def safe_median(values):
    """
    Median of finite values.
    """

    array = finite_array(values)

    if len(array) == 0:
        return np.nan

    return float(
        np.median(array)
    )


def format_float(
    value,
    digits=4,
):
    """
    Safe terminal formatting.
    """

    if pd.isna(value):
        return "NA"

    return f"{float(value):.{digits}f}"


def format_scientific(
    value,
):
    """
    Scientific notation for p-values and small quantities.
    """

    if pd.isna(value):
        return "NA"

    return f"{float(value):.6e}"


# =========================================================================
# INPUT VALIDATION
# =========================================================================


def load_and_validate_node_rate_summary():
    """
    Load the frozen Phase-10 node-rate summary and validate its
    structural integrity before inferential analysis.
    """

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Required Phase-10 artifact not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    missing = (
        REQUIRED_COLUMNS
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

    if df["node_id"].nunique() != EXPECTED_NODES:

        raise RuntimeError(
            "Unexpected number of unique target nodes. "
            f"Observed={df['node_id'].nunique()}, "
            f"expected={EXPECTED_NODES}."
        )

    # ---------------------------------------------------------------------
    # Duplicate node-rate keys must not exist.
    # ---------------------------------------------------------------------

    if df.duplicated(
        subset=[
            "node_id",
            "nominal_mask_rate",
        ]
    ).any():

        duplicates = df.loc[
            df.duplicated(
                subset=[
                    "node_id",
                    "nominal_mask_rate",
                ],
                keep=False,
            ),
            [
                "node_id",
                "nominal_mask_rate",
            ],
        ]

        raise RuntimeError(
            "Duplicate node-rate rows detected:\n"
            f"{duplicates.head(20)}"
        )

    # ---------------------------------------------------------------------
    # Frozen perturbation-rate set.
    # ---------------------------------------------------------------------

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
                "Perturbation-rate set differs from "
                "the frozen protocol."
            )

    # ---------------------------------------------------------------------
    # Every node must occur exactly once at every rate.
    # ---------------------------------------------------------------------

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
            "At least one node does not have exactly "
            "one row for every perturbation rate."
        )

    for rate in MASK_RATES:

        rate_subset = df.loc[
            np.isclose(
                df[
                    "nominal_mask_rate"
                ].astype(float),
                rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            )
        ]

        if len(rate_subset) != EXPECTED_NODES:

            raise RuntimeError(
                f"Rate {rate} has {len(rate_subset)} nodes; "
                f"expected {EXPECTED_NODES}."
            )

    # ---------------------------------------------------------------------
    # Expected number of perturbations per node-rate.
    # ---------------------------------------------------------------------

    if not (
        df[
            "num_perturbations"
        ]
        == 10
    ).all():

        raise RuntimeError(
            "At least one node-rate row does not summarize "
            "exactly 10 perturbation seeds."
        )

    # ---------------------------------------------------------------------
    # Bounded quantities.
    # ---------------------------------------------------------------------

    bounded_columns = [
        "prediction_stability",
        "prediction_change_rate",
        "explanation_availability",
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

    for column in [
        "mean_jaccard",
        "mean_jaccard_pred_same",
    ]:

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

    # ---------------------------------------------------------------------
    # Explanation availability accounting.
    # ---------------------------------------------------------------------

    if not (
        (
            df[
                "num_explanations_available"
            ]
            +
            df[
                "num_explanations_unavailable"
            ]
        )
        == df[
            "num_perturbations"
        ]
    ).all():

        raise RuntimeError(
            "Explanation availability accounting is inconsistent."
        )

    # ---------------------------------------------------------------------
    # Jaccard must be defined only when at least one explanation
    # contributes to that node-rate mean.
    # ---------------------------------------------------------------------

    invalid_jaccard_presence = (
        (
            df[
                "num_explanations_available"
            ]
            == 0
        )
        &
        df[
            "mean_jaccard"
        ].notna()
    )

    if invalid_jaccard_presence.any():

        raise RuntimeError(
            "mean_jaccard is present for at least one node-rate "
            "with zero available explanations."
        )

    invalid_jaccard_missing = (
        (
            df[
                "num_explanations_available"
            ]
            > 0
        )
        &
        df[
            "mean_jaccard"
        ].isna()
    )

    if invalid_jaccard_missing.any():

        raise RuntimeError(
            "mean_jaccard is missing for at least one node-rate "
            "with available explanations."
        )

    # ---------------------------------------------------------------------
    # Prediction-conditioned Jaccard must be defined only when at least
    # one observation has both unchanged prediction and available
    # explanation.
    # ---------------------------------------------------------------------

    invalid_conditioned_presence = (
        (
            df[
                "num_pred_same_with_explanation"
            ]
            == 0
        )
        &
        df[
            "mean_jaccard_pred_same"
        ].notna()
    )

    if invalid_conditioned_presence.any():

        raise RuntimeError(
            "mean_jaccard_pred_same is present where no "
            "prediction-preserving available explanation exists."
        )

    invalid_conditioned_missing = (
        (
            df[
                "num_pred_same_with_explanation"
            ]
            > 0
        )
        &
        df[
            "mean_jaccard_pred_same"
        ].isna()
    )

    if invalid_conditioned_missing.any():

        raise RuntimeError(
            "mean_jaccard_pred_same is missing despite at least "
            "one qualifying observation."
        )

    return df


# =========================================================================
# PAIRED DATA EXTRACTION
# =========================================================================


def extract_paired_values(
    df: pd.DataFrame,
    outcome: str,
    lower_rate: float,
    higher_rate: float,
):
    """
    Extract paired node-level observations for a given outcome.

    Only nodes with finite values at BOTH rates are retained.

    Returns:
        node_ids
        lower_values
        higher_values
    """

    lower = (
        df.loc[
            np.isclose(
                df[
                    "nominal_mask_rate"
                ].astype(float),
                lower_rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            ),
            [
                "node_id",
                outcome,
            ],
        ]
        .rename(
            columns={
                outcome: "lower_value",
            }
        )
    )

    higher = (
        df.loc[
            np.isclose(
                df[
                    "nominal_mask_rate"
                ].astype(float),
                higher_rate,
                atol=FLOAT_TOLERANCE,
                rtol=0.0,
            ),
            [
                "node_id",
                outcome,
            ],
        ]
        .rename(
            columns={
                outcome: "higher_value",
            }
        )
    )

    paired = lower.merge(
        higher,
        on="node_id",
        how="inner",
        validate="one_to_one",
    )

    finite_mask = (
        np.isfinite(
            paired[
                "lower_value"
            ].astype(float)
        )
        &
        np.isfinite(
            paired[
                "higher_value"
            ].astype(float)
        )
    )

    paired = (
        paired.loc[
            finite_mask
        ]
        .sort_values(
            "node_id"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        paired[
            "node_id"
        ].to_numpy(
            dtype=int
        ),
        paired[
            "lower_value"
        ].to_numpy(
            dtype=float
        ),
        paired[
            "higher_value"
        ].to_numpy(
            dtype=float
        ),
    )


# =========================================================================
# PAIRED BOOTSTRAP
# =========================================================================


def paired_bootstrap_mean_difference_ci(
    lower_values,
    higher_values,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
):
    """
    Paired node-level bootstrap CI for:

        higher-rate outcome - lower-rate outcome

    Nodes are resampled as intact pairs.

    Returns:
        mean_difference
        ci_low
        ci_high
        n_pairs
    """

    lower = np.asarray(
        lower_values,
        dtype=float,
    )

    higher = np.asarray(
        higher_values,
        dtype=float,
    )

    if lower.shape != higher.shape:

        raise ValueError(
            "Paired arrays must have identical shape."
        )

    finite_mask = (
        np.isfinite(lower)
        &
        np.isfinite(higher)
    )

    lower = lower[
        finite_mask
    ]

    higher = higher[
        finite_mask
    ]

    n = len(lower)

    if n == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
        )

    differences = (
        higher
        - lower
    )

    mean_difference = float(
        np.mean(
            differences
        )
    )

    if n == 1:

        return (
            mean_difference,
            mean_difference,
            mean_difference,
            1,
        )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(
        n_bootstrap
    ):

        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        bootstrap_means[i] = float(
            np.mean(
                differences[
                    indices
                ]
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

    ci_low = float(
        np.percentile(
            bootstrap_means,
            lower_percentile,
        )
    )

    ci_high = float(
        np.percentile(
            bootstrap_means,
            upper_percentile,
        )
    )

    return (
        mean_difference,
        ci_low,
        ci_high,
        n,
    )


# =========================================================================
# RANKING / NORMAL-DISTRIBUTION HELPERS
# =========================================================================


def average_ranks(values):
    """
    Compute one-based average ranks with exact tie handling.

    Tied observations receive the mean of the ranks that they occupy.

    Example:

        values = [1, 2, 2, 4]
        ranks  = [1, 2.5, 2.5, 4]

    Returns:
        numpy.ndarray of float ranks
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

    # Stable sorting ensures deterministic handling of ties.
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

        # Statistical ranks are one-based.
        #
        # A tie group occupying zero-based positions:
        #
        #     start, ..., end - 1
        #
        # occupies statistical ranks:
        #
        #     start + 1, ..., end
        #
        # whose average is:
        #
        #     ((start + 1) + end) / 2
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


def standard_normal_cdf(value):
    """
    Standard normal cumulative distribution function.

        Phi(z) = 0.5 * [1 + erf(z / sqrt(2))]

    Uses only Python's standard-library math module.
    """

    return 0.5 * (
        1.0
        +
        math.erf(
            float(value)
            /
            math.sqrt(2.0)
        )
    )


# =========================================================================
# WILCOXON SIGNED-RANK TEST -- LOCAL IMPLEMENTATION, NO SCIPY
# =========================================================================


def wilcoxon_signed_rank(
    lower_values,
    higher_values,
):
    """
    Two-sided Wilcoxon signed-rank test implemented without SciPy.

    Difference orientation:

        d = higher - lower

    Procedure:
        1. Retain finite paired observations.
        2. Remove differences effectively equal to zero.
        3. Rank absolute nonzero differences using average ranks for ties.
        4. Compute positive and negative signed-rank sums.
        5. Use min(W+, W-) as the conventional two-sided statistic.
        6. Compute a two-sided normal-approximation p-value using
           tie-corrected variance.

    No continuity correction is applied.

    The Wilcoxon test is a secondary inferential check in this study;
    paired node-level bootstrap confidence intervals remain the primary
    uncertainty analysis.

    Returns:
        statistic
        p_value
        n_pairs
        n_nonzero_differences
    """

    lower = np.asarray(
        lower_values,
        dtype=float,
    )

    higher = np.asarray(
        higher_values,
        dtype=float,
    )

    if lower.shape != higher.shape:

        raise ValueError(
            "Paired arrays must have identical shape."
        )

    finite_mask = (
        np.isfinite(lower)
        &
        np.isfinite(higher)
    )

    lower = lower[
        finite_mask
    ]

    higher = higher[
        finite_mask
    ]

    differences = (
        higher
        - lower
    )

    n_pairs = len(
        differences
    )

    if n_pairs == 0:

        return (
            np.nan,
            np.nan,
            0,
            0,
        )

    nonzero_mask = (
        np.abs(
            differences
        )
        > MONOTONICITY_TOLERANCE
    )

    nonzero_differences = differences[
        nonzero_mask
    ]

    n_nonzero = len(
        nonzero_differences
    )

    if n_nonzero == 0:

        # Every paired difference is effectively zero.
        return (
            0.0,
            1.0,
            n_pairs,
            0,
        )

    absolute_differences = np.abs(
        nonzero_differences
    )

    ranks = average_ranks(
        absolute_differences
    )

    positive_rank_sum = float(
        ranks[
            nonzero_differences > 0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            nonzero_differences < 0
        ].sum()
    )

    statistic = min(
        positive_rank_sum,
        negative_rank_sum,
    )

    # ---------------------------------------------------------------------
    # Null distribution.
    #
    # For W+ with n nonzero differences:
    #
    #     E[W+] = n(n+1)/4
    #
    # Without ties:
    #
    #     Var(W+) = n(n+1)(2n+1)/24
    #
    # For tie groups among |d_i| having sizes t_j, the variance is:
    #
    #     Var(W+) =
    #         n(n+1)(2n+1)/24
    #         - sum_j[t_j(t_j+1)(2t_j+1)]/48
    #
    # Zero differences have already been removed.
    # ---------------------------------------------------------------------

    n = n_nonzero

    expected_positive_rank_sum = (
        n
        * (n + 1)
        / 4.0
    )

    base_variance = (
        n
        * (n + 1)
        * (2 * n + 1)
        / 24.0
    )

    _, tie_counts = np.unique(
        absolute_differences,
        return_counts=True,
    )

    tie_correction = 0.0

    for tie_count in tie_counts:

        if tie_count > 1:

            t = float(
                tie_count
            )

            tie_correction += (
                t
                * (t + 1.0)
                * (2.0 * t + 1.0)
                / 48.0
            )

    variance = (
        base_variance
        - tie_correction
    )

    if variance <= 0.0:

        # Defensive handling for a pathological variance-degenerate case.
        #
        # Returning p=1 is conservative and avoids producing an invalid
        # z statistic.
        return (
            float(statistic),
            1.0,
            n_pairs,
            n_nonzero,
        )

    standard_error = math.sqrt(
        variance
    )

    z = (
        positive_rank_sum
        - expected_positive_rank_sum
    ) / standard_error

    two_sided_p = math.erfc(
        abs(z)
        / math.sqrt(2.0)
    )

    # Guard against tiny floating-point excursions outside [0, 1].
    two_sided_p = min(
        1.0,
        max(
            0.0,
            float(two_sided_p),
        ),
    )

    return (
        float(statistic),
        two_sided_p,
        n_pairs,
        n_nonzero,
    )


# =========================================================================
# MATCHED-PAIRS RANK-BISERIAL EFFECT SIZE
# =========================================================================


def matched_pairs_rank_biserial(
    lower_values,
    higher_values,
):
    """
    Matched-pairs rank-biserial correlation.

    Difference orientation:

        d = higher - lower

    r_rb =
        (sum ranks of positive d - sum ranks of negative d)
        /
        (sum ranks of positive d + sum ranks of negative d)

    Zero differences are excluded from ranking.

    Interpretation of sign:

        negative -> higher rate tends to have smaller outcome
        positive -> higher rate tends to have larger outcome

    Returns:
        r_rb
        positive_rank_sum
        negative_rank_sum
        n_positive
        n_negative
        n_zero
    """

    lower = np.asarray(
        lower_values,
        dtype=float,
    )

    higher = np.asarray(
        higher_values,
        dtype=float,
    )

    if lower.shape != higher.shape:

        raise ValueError(
            "Paired arrays must have identical shape."
        )

    finite_mask = (
        np.isfinite(lower)
        &
        np.isfinite(higher)
    )

    differences = (
        higher[
            finite_mask
        ]
        -
        lower[
            finite_mask
        ]
    )

    positive_mask = (
        differences
        > MONOTONICITY_TOLERANCE
    )

    negative_mask = (
        differences
        < -MONOTONICITY_TOLERANCE
    )

    zero_mask = ~(
        positive_mask
        |
        negative_mask
    )

    nonzero_mask = (
        positive_mask
        |
        negative_mask
    )

    nonzero_differences = differences[
        nonzero_mask
    ]

    if len(
        nonzero_differences
    ) == 0:

        return (
            0.0,
            0.0,
            0.0,
            0,
            0,
            int(
                zero_mask.sum()
            ),
        )

    ranks = average_ranks(
        np.abs(
            nonzero_differences
        )
    )

    signs = np.sign(
        nonzero_differences
    )

    positive_rank_sum = float(
        ranks[
            signs > 0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            signs < 0
        ].sum()
    )

    denominator = (
        positive_rank_sum
        +
        negative_rank_sum
    )

    if denominator == 0.0:

        rank_biserial = 0.0

    else:

        rank_biserial = float(
            (
                positive_rank_sum
                -
                negative_rank_sum
            )
            /
            denominator
        )

    return (
        rank_biserial,
        positive_rank_sum,
        negative_rank_sum,
        int(
            positive_mask.sum()
        ),
        int(
            negative_mask.sum()
        ),
        int(
            zero_mask.sum()
        ),
    )


# =========================================================================
# HOLM MULTIPLE-COMPARISON CORRECTION
# =========================================================================


def holm_adjust(
    p_values,
):
    """
    Holm step-down adjusted p-values.

    NaN p-values remain NaN.

    The adjusted p-values are returned in the original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    finite_indices = np.where(
        np.isfinite(
            p_values
        )
    )[0]

    if len(
        finite_indices
    ) == 0:

        return adjusted

    finite_p = p_values[
        finite_indices
    ]

    order = np.argsort(
        finite_p
    )

    sorted_p = finite_p[
        order
    ]

    m = len(
        sorted_p
    )

    sorted_adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank_index, p_value in enumerate(
        sorted_p
    ):

        multiplier = (
            m
            - rank_index
        )

        candidate = min(
            1.0,
            multiplier
            * p_value,
        )

        running_max = max(
            running_max,
            candidate,
        )

        sorted_adjusted[
            rank_index
        ] = min(
            1.0,
            running_max,
        )

    # Map from sorted finite positions back to finite input positions.
    finite_adjusted = np.empty(
        m,
        dtype=float,
    )

    finite_adjusted[
        order
    ] = sorted_adjusted

    adjusted[
        finite_indices
    ] = finite_adjusted

    return adjusted


# =========================================================================
# SINGLE PAIRED CONTRAST
# =========================================================================


def analyze_paired_contrast(
    df: pd.DataFrame,
    outcome: str,
    lower_rate: float,
    higher_rate: float,
    rng: np.random.Generator,
):
    """
    Analyze one paired rate contrast for one node-level outcome.
    """

    (
        node_ids,
        lower_values,
        higher_values,
    ) = extract_paired_values(
        df=df,
        outcome=outcome,
        lower_rate=lower_rate,
        higher_rate=higher_rate,
    )

    differences = (
        higher_values
        - lower_values
    )

    (
        mean_difference,
        ci_low,
        ci_high,
        n_pairs,
    ) = paired_bootstrap_mean_difference_ci(
        lower_values=lower_values,
        higher_values=higher_values,
        rng=rng,
    )

    (
        wilcoxon_statistic,
        wilcoxon_p,
        wilcoxon_n_pairs,
        wilcoxon_n_nonzero,
    ) = wilcoxon_signed_rank(
        lower_values=lower_values,
        higher_values=higher_values,
    )

    (
        rank_biserial,
        positive_rank_sum,
        negative_rank_sum,
        n_positive,
        n_negative,
        n_zero,
    ) = matched_pairs_rank_biserial(
        lower_values=lower_values,
        higher_values=higher_values,
    )

    if (
        n_pairs
        != wilcoxon_n_pairs
    ):

        raise RuntimeError(
            "Paired bootstrap and Wilcoxon pair counts disagree."
        )

    if (
        n_positive
        +
        n_negative
        +
        n_zero
        != n_pairs
    ):

        raise RuntimeError(
            "Paired direction accounting failed inside contrast analysis."
        )

    return {
        "outcome": outcome,
        "lower_rate": float(
            lower_rate
        ),
        "higher_rate": float(
            higher_rate
        ),
        "contrast": (
            f"{lower_rate:.2f}_to_{higher_rate:.2f}"
        ),
        "difference_orientation": (
            "higher_rate_minus_lower_rate"
        ),
        "n_paired_nodes": int(
            n_pairs
        ),
        "lower_rate_mean": safe_mean(
            lower_values
        ),
        "higher_rate_mean": safe_mean(
            higher_values
        ),
        "mean_paired_difference": (
            mean_difference
        ),
        "median_paired_difference": (
            safe_median(
                differences
            )
        ),
        "std_paired_difference": (
            safe_std(
                differences
            )
        ),
        "paired_difference_ci_low": (
            ci_low
        ),
        "paired_difference_ci_high": (
            ci_high
        ),
        "num_nodes_higher_value": int(
            n_positive
        ),
        "num_nodes_lower_value": int(
            n_negative
        ),
        "num_nodes_equal_value": int(
            n_zero
        ),
        "wilcoxon_statistic": (
            wilcoxon_statistic
        ),
        "wilcoxon_p_raw": (
            wilcoxon_p
        ),
        "wilcoxon_nonzero_pairs": int(
            wilcoxon_n_nonzero
        ),
        "wilcoxon_method": (
            "local_tie_corrected_normal_approximation"
        ),
        "wilcoxon_continuity_correction": False,
        "rank_biserial": (
            rank_biserial
        ),
        "positive_rank_sum": (
            positive_rank_sum
        ),
        "negative_rank_sum": (
            negative_rank_sum
        ),
        "bootstrap_resamples": int(
            N_BOOTSTRAP
        ),
        "confidence_level": float(
            CONFIDENCE_LEVEL
        ),
    }


# =========================================================================
# BUILD PAIRED RATE CONTRAST TABLE
# =========================================================================


def build_paired_rate_contrasts(
    df: pd.DataFrame,
):
    """
    Construct paired rate contrasts.

    Primary family:
        mean_jaccard

    Secondary descriptive/inferential families:
        mean_jaccard_pred_same
        prediction_stability
        explanation_availability
        mean_absolute_confidence_delta

    Holm correction is applied separately within each outcome family.
    """

    outcomes = [
        "mean_jaccard",
        "mean_jaccard_pred_same",
        "prediction_stability",
        "explanation_availability",
        "mean_absolute_confidence_delta",
    ]

    rate_pairs = list(
        combinations(
            MASK_RATES,
            2,
        )
    )

    if len(
        rate_pairs
    ) != EXPECTED_PAIRWISE_CONTRASTS:

        raise RuntimeError(
            "Unexpected number of pairwise rate contrasts."
        )

    records = []

    # Independent deterministic RNG stream for each outcome.
    for outcome_index, outcome in enumerate(
        outcomes
    ):

        rng = np.random.default_rng(
            BOOTSTRAP_SEED
            + 1000
            * outcome_index
        )

        outcome_records = []

        for (
            lower_rate,
            higher_rate,
        ) in rate_pairs:

            record = analyze_paired_contrast(
                df=df,
                outcome=outcome,
                lower_rate=lower_rate,
                higher_rate=higher_rate,
                rng=rng,
            )

            outcome_records.append(
                record
            )

        raw_p_values = [
            record[
                "wilcoxon_p_raw"
            ]
            for record in outcome_records
        ]

        adjusted_p_values = holm_adjust(
            raw_p_values
        )

        for (
            record,
            adjusted_p,
        ) in zip(
            outcome_records,
            adjusted_p_values,
        ):

            record[
                "wilcoxon_p_holm"
            ] = float(
                adjusted_p
            ) if np.isfinite(
                adjusted_p
            ) else np.nan

            record[
                "holm_family"
            ] = outcome

            record[
                "holm_family_size"
            ] = int(
                len(
                    outcome_records
                )
            )

            record[
                "holm_alpha"
            ] = float(
                ALPHA
            )

            record[
                "holm_reject_at_alpha"
            ] = bool(
                np.isfinite(
                    adjusted_p
                )
                and
                adjusted_p
                < ALPHA
            )

            records.append(
                record
            )

    result = pd.DataFrame(
        records
    )

    expected_rows = (
        len(outcomes)
        * EXPECTED_PAIRWISE_CONTRASTS
    )

    if len(result) != expected_rows:

        raise RuntimeError(
            "Unexpected paired-contrast table size. "
            f"Observed={len(result)}, "
            f"expected={expected_rows}."
        )

    return result


# =========================================================================
# MONOTONICITY ANALYSIS
# =========================================================================


def classify_monotonic_pattern(
    values,
    tolerance=MONOTONICITY_TOLERANCE,
):
    """
    Classify a four-rate node-level sequence.

    Categories:
        strictly_decreasing
        nonincreasing_with_ties
        strictly_increasing
        nondecreasing_with_ties
        constant
        nonmonotonic
        incomplete
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    if len(array) != len(
        MASK_RATES
    ):

        return "incomplete"

    if not np.isfinite(
        array
    ).all():

        return "incomplete"

    differences = np.diff(
        array
    )

    all_zero = np.all(
        np.abs(
            differences
        )
        <= tolerance
    )

    if all_zero:
        return "constant"

    strictly_decreasing = np.all(
        differences
        < -tolerance
    )

    if strictly_decreasing:
        return "strictly_decreasing"

    nonincreasing = np.all(
        differences
        <= tolerance
    )

    if nonincreasing:
        return "nonincreasing_with_ties"

    strictly_increasing = np.all(
        differences
        > tolerance
    )

    if strictly_increasing:
        return "strictly_increasing"

    nondecreasing = np.all(
        differences
        >= -tolerance
    )

    if nondecreasing:
        return "nondecreasing_with_ties"

    return "nonmonotonic"


def node_level_monotonicity_table(
    df: pd.DataFrame,
    outcome: str,
):
    """
    Build one row per node describing its four-rate trajectory.
    """

    records = []

    for node_id, group in df.groupby(
        "node_id",
        sort=True,
    ):

        rate_to_value = {}

        for rate in MASK_RATES:

            subset = group.loc[
                np.isclose(
                    group[
                        "nominal_mask_rate"
                    ].astype(float),
                    rate,
                    atol=FLOAT_TOLERANCE,
                    rtol=0.0,
                )
            ]

            if len(subset) != 1:

                raise RuntimeError(
                    f"Node {node_id} has invalid row count "
                    f"for rate {rate}."
                )

            value = subset.iloc[0][
                outcome
            ]

            rate_to_value[
                rate
            ] = (
                float(value)
                if pd.notna(value)
                else np.nan
            )

        ordered_values = [
            rate_to_value[
                rate
            ]
            for rate in MASK_RATES
        ]

        pattern = (
            classify_monotonic_pattern(
                ordered_values
            )
        )

        endpoint_difference = np.nan

        if (
            np.isfinite(
                ordered_values[0]
            )
            and
            np.isfinite(
                ordered_values[-1]
            )
        ):

            endpoint_difference = float(
                ordered_values[-1]
                - ordered_values[0]
            )

        records.append(
            {
                "outcome": outcome,
                "node_id": int(
                    node_id
                ),
                "rate_0_01": (
                    ordered_values[0]
                ),
                "rate_0_05": (
                    ordered_values[1]
                ),
                "rate_0_10": (
                    ordered_values[2]
                ),
                "rate_0_20": (
                    ordered_values[3]
                ),
                "pattern": pattern,
                "endpoint_difference_0_20_minus_0_01": (
                    endpoint_difference
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def build_monotonicity_summary(
    df: pd.DataFrame,
):
    """
    Summarize node-level monotonicity for the principal trajectories.

    We report:
        mean_jaccard
        mean_jaccard_pred_same
        prediction_stability
        explanation_availability
    """

    outcomes = [
        "mean_jaccard",
        "mean_jaccard_pred_same",
        "prediction_stability",
        "explanation_availability",
    ]

    summary_records = []

    for outcome in outcomes:

        node_table = (
            node_level_monotonicity_table(
                df=df,
                outcome=outcome,
            )
        )

        pattern_counts = (
            node_table[
                "pattern"
            ]
            .value_counts()
            .to_dict()
        )

        complete_mask = (
            node_table[
                "pattern"
            ]
            != "incomplete"
        )

        complete_nodes = int(
            complete_mask.sum()
        )

        incomplete_nodes = int(
            (~complete_mask).sum()
        )

        strictly_decreasing = int(
            pattern_counts.get(
                "strictly_decreasing",
                0,
            )
        )

        nonincreasing_with_ties = int(
            pattern_counts.get(
                "nonincreasing_with_ties",
                0,
            )
        )

        constant = int(
            pattern_counts.get(
                "constant",
                0,
            )
        )

        strictly_increasing = int(
            pattern_counts.get(
                "strictly_increasing",
                0,
            )
        )

        nondecreasing_with_ties = int(
            pattern_counts.get(
                "nondecreasing_with_ties",
                0,
            )
        )

        nonmonotonic = int(
            pattern_counts.get(
                "nonmonotonic",
                0,
            )
        )

        monotonic_nonincreasing_total = (
            strictly_decreasing
            +
            nonincreasing_with_ties
            +
            constant
        )

        monotonic_nondecreasing_total = (
            strictly_increasing
            +
            nondecreasing_with_ties
            +
            constant
        )

        endpoint_values = (
            node_table[
                "endpoint_difference_0_20_minus_0_01"
            ]
            .dropna()
            .astype(float)
        )

        summary_records.append(
            {
                "outcome": outcome,
                "total_nodes": int(
                    len(node_table)
                ),
                "complete_nodes": (
                    complete_nodes
                ),
                "incomplete_nodes": (
                    incomplete_nodes
                ),
                "strictly_decreasing_nodes": (
                    strictly_decreasing
                ),
                "nonincreasing_with_ties_nodes": (
                    nonincreasing_with_ties
                ),
                "constant_nodes": (
                    constant
                ),
                "strictly_increasing_nodes": (
                    strictly_increasing
                ),
                "nondecreasing_with_ties_nodes": (
                    nondecreasing_with_ties
                ),
                "nonmonotonic_nodes": (
                    nonmonotonic
                ),
                "monotonic_nonincreasing_nodes": (
                    monotonic_nonincreasing_total
                ),
                "monotonic_nondecreasing_nodes": (
                    monotonic_nondecreasing_total
                ),
                "fraction_monotonic_nonincreasing": (
                    float(
                        monotonic_nonincreasing_total
                        / complete_nodes
                    )
                    if complete_nodes > 0
                    else np.nan
                ),
                "fraction_strictly_decreasing": (
                    float(
                        strictly_decreasing
                        / complete_nodes
                    )
                    if complete_nodes > 0
                    else np.nan
                ),
                "mean_endpoint_difference_0_20_minus_0_01": (
                    safe_mean(
                        endpoint_values
                    )
                ),
                "median_endpoint_difference_0_20_minus_0_01": (
                    safe_median(
                        endpoint_values
                    )
                ),
            }
        )

    return pd.DataFrame(
        summary_records
    )


# =========================================================================
# HYPOTHESIS-ORIENTED SUMMARY
# =========================================================================


def get_contrast_row(
    contrasts_df: pd.DataFrame,
    outcome: str,
    lower_rate: float,
    higher_rate: float,
):
    """
    Retrieve exactly one contrast row.
    """

    subset = contrasts_df.loc[
        (
            contrasts_df[
                "outcome"
            ]
            == outcome
        )
        &
        np.isclose(
            contrasts_df[
                "lower_rate"
            ].astype(float),
            lower_rate,
            atol=FLOAT_TOLERANCE,
            rtol=0.0,
        )
        &
        np.isclose(
            contrasts_df[
                "higher_rate"
            ].astype(float),
            higher_rate,
            atol=FLOAT_TOLERANCE,
            rtol=0.0,
        )
    ]

    if len(subset) != 1:

        raise RuntimeError(
            "Expected exactly one matching contrast row."
        )

    return subset.iloc[0]


def get_monotonicity_row(
    monotonicity_df: pd.DataFrame,
    outcome: str,
):
    """
    Retrieve exactly one monotonicity summary row.
    """

    subset = monotonicity_df.loc[
        monotonicity_df[
            "outcome"
        ]
        == outcome
    ]

    if len(subset) != 1:

        raise RuntimeError(
            "Expected exactly one matching monotonicity row."
        )

    return subset.iloc[0]


def build_hypothesis_summary(
    contrasts_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
):
    """
    Build a compact hypothesis-oriented evidence table.

    IMPORTANT:
    This table reports evidence relevant to each hypothesis.
    It deliberately does NOT assign automatic binary labels such as
    "supported" or "rejected".

    H1:
        Explanation overlap declines as feature perturbation magnitude
        increases.

    H2:
        Explanation stability can deteriorate even when the predicted
        class remains unchanged.

    H3:
        Feature and structural perturbations affect explanation stability
        differently.

        H3 is NOT testable from the present feature-masking experiment
        because structural perturbations have not yet been run.
    """

    h1_endpoint = get_contrast_row(
        contrasts_df=contrasts_df,
        outcome="mean_jaccard",
        lower_rate=0.01,
        higher_rate=0.20,
    )

    h1_monotonicity = get_monotonicity_row(
        monotonicity_df=monotonicity_df,
        outcome="mean_jaccard",
    )

    h2_endpoint = get_contrast_row(
        contrasts_df=contrasts_df,
        outcome="mean_jaccard_pred_same",
        lower_rate=0.01,
        higher_rate=0.20,
    )

    h2_monotonicity = get_monotonicity_row(
        monotonicity_df=monotonicity_df,
        outcome="mean_jaccard_pred_same",
    )

    records = [
        {
            "hypothesis_id": "H1",
            "hypothesis": (
                "GraphLIME top-K explanation overlap declines "
                "as feature perturbation magnitude increases."
            ),
            "analysis_status": "testable_with_current_experiment",
            "primary_outcome": "mean_jaccard",
            "primary_contrast": "0.01_to_0.20",
            "n_paired_nodes": int(
                h1_endpoint[
                    "n_paired_nodes"
                ]
            ),
            "endpoint_mean_difference": float(
                h1_endpoint[
                    "mean_paired_difference"
                ]
            ),
            "endpoint_ci_low": float(
                h1_endpoint[
                    "paired_difference_ci_low"
                ]
            ),
            "endpoint_ci_high": float(
                h1_endpoint[
                    "paired_difference_ci_high"
                ]
            ),
            "wilcoxon_p_raw": float(
                h1_endpoint[
                    "wilcoxon_p_raw"
                ]
            ),
            "wilcoxon_p_holm": float(
                h1_endpoint[
                    "wilcoxon_p_holm"
                ]
            ),
            "rank_biserial": float(
                h1_endpoint[
                    "rank_biserial"
                ]
            ),
            "fraction_monotonic_nonincreasing": float(
                h1_monotonicity[
                    "fraction_monotonic_nonincreasing"
                ]
            ),
            "interpretation_rule": (
                "Assess direction, magnitude, paired bootstrap CI, "
                "node-level monotonicity, and corrected Wilcoxon result "
                "together; do not base interpretation on p-value alone."
            ),
        },
        {
            "hypothesis_id": "H2",
            "hypothesis": (
                "GraphLIME explanation stability can decline under "
                "feature perturbation even when the predicted class "
                "remains unchanged."
            ),
            "analysis_status": "testable_with_current_experiment",
            "primary_outcome": "mean_jaccard_pred_same",
            "primary_contrast": "0.01_to_0.20",
            "n_paired_nodes": int(
                h2_endpoint[
                    "n_paired_nodes"
                ]
            ),
            "endpoint_mean_difference": float(
                h2_endpoint[
                    "mean_paired_difference"
                ]
            ),
            "endpoint_ci_low": float(
                h2_endpoint[
                    "paired_difference_ci_low"
                ]
            ),
            "endpoint_ci_high": float(
                h2_endpoint[
                    "paired_difference_ci_high"
                ]
            ),
            "wilcoxon_p_raw": float(
                h2_endpoint[
                    "wilcoxon_p_raw"
                ]
            ),
            "wilcoxon_p_holm": float(
                h2_endpoint[
                    "wilcoxon_p_holm"
                ]
            ),
            "rank_biserial": float(
                h2_endpoint[
                    "rank_biserial"
                ]
            ),
            "fraction_monotonic_nonincreasing": float(
                h2_monotonicity[
                    "fraction_monotonic_nonincreasing"
                ]
            ),
            "interpretation_rule": (
                "Assess explanation-overlap decline specifically among "
                "prediction-preserving observations, together with "
                "prediction-stability results; do not equate unchanged "
                "class with unchanged explanation."
            ),
        },
        {
            "hypothesis_id": "H3",
            "hypothesis": (
                "Feature perturbations and structural perturbations "
                "affect explanation stability differently."
            ),
            "analysis_status": "not_testable_with_current_experiment",
            "primary_outcome": "not_available",
            "primary_contrast": "requires_structural_perturbation_data",
            "n_paired_nodes": np.nan,
            "endpoint_mean_difference": np.nan,
            "endpoint_ci_low": np.nan,
            "endpoint_ci_high": np.nan,
            "wilcoxon_p_raw": np.nan,
            "wilcoxon_p_holm": np.nan,
            "rank_biserial": np.nan,
            "fraction_monotonic_nonincreasing": np.nan,
            "interpretation_rule": (
                "Do not infer H3 from feature-masking data alone. "
                "A separately pre-specified structural perturbation "
                "experiment is required."
            ),
        },
    ]

    return pd.DataFrame(
        records
    )


# =========================================================================
# OUTPUT VALIDATION
# =========================================================================


def validate_outputs(
    contrasts_df: pd.DataFrame,
    monotonicity_df: pd.DataFrame,
    hypothesis_df: pd.DataFrame,
):
    """
    Integrity checks for the Phase-10C statistical outputs.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-10C Statistical Output Validation ==="
    )

    print(
        "=" * 78
    )

    # ---------------------------------------------------------------------
    # Contrast table size.
    # ---------------------------------------------------------------------

    expected_outcomes = 5

    expected_contrast_rows = (
        expected_outcomes
        * EXPECTED_PAIRWISE_CONTRASTS
    )

    if len(
        contrasts_df
    ) != expected_contrast_rows:

        raise RuntimeError(
            "Unexpected number of paired contrast rows."
        )

    print(
        f"Paired contrast rows:        "
        f"{len(contrasts_df)}/"
        f"{expected_contrast_rows} PASS"
    )

    # ---------------------------------------------------------------------
    # Exactly six contrasts per outcome.
    # ---------------------------------------------------------------------

    counts = (
        contrasts_df.groupby(
            "outcome"
        )
        .size()
    )

    if not (
        counts
        == EXPECTED_PAIRWISE_CONTRASTS
    ).all():

        raise RuntimeError(
            "At least one outcome does not have all "
            "six pairwise rate contrasts."
        )

    print(
        "Six contrasts per outcome:  PASS"
    )

    # ---------------------------------------------------------------------
    # Pair counts must be sensible.
    # ---------------------------------------------------------------------

    if (
        contrasts_df[
            "n_paired_nodes"
        ]
        > EXPECTED_NODES
    ).any():

        raise RuntimeError(
            "A contrast has more paired nodes than expected."
        )

    if (
        contrasts_df[
            "n_paired_nodes"
        ]
        <= 0
    ).any():

        raise RuntimeError(
            "At least one contrast has no paired observations."
        )

    print(
        "Paired-node counts:         PASS"
    )

    # ---------------------------------------------------------------------
    # Bootstrap CI ordering.
    # ---------------------------------------------------------------------

    valid_ci = contrasts_df[
        [
            "mean_paired_difference",
            "paired_difference_ci_low",
            "paired_difference_ci_high",
        ]
    ].dropna()

    if not (
        (
            valid_ci[
                "paired_difference_ci_low"
            ]
            <= valid_ci[
                "mean_paired_difference"
            ]
        )
        &
        (
            valid_ci[
                "mean_paired_difference"
            ]
            <= valid_ci[
                "paired_difference_ci_high"
            ]
        )
    ).all():

        raise RuntimeError(
            "Paired bootstrap CI ordering failed."
        )

    print(
        "Paired bootstrap CI order:  PASS"
    )

    # ---------------------------------------------------------------------
    # p-values.
    # ---------------------------------------------------------------------

    for column in [
        "wilcoxon_p_raw",
        "wilcoxon_p_holm",
    ]:

        finite = (
            contrasts_df[
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

    print(
        "Wilcoxon p-value ranges:    PASS"
    )

    # Holm-adjusted p-values should never be smaller than raw p-values.
    finite_holm = contrasts_df.loc[
        contrasts_df[
            "wilcoxon_p_raw"
        ].notna()
        &
        contrasts_df[
            "wilcoxon_p_holm"
        ].notna()
    ]

    if not (
        finite_holm[
            "wilcoxon_p_holm"
        ]
        + FLOAT_TOLERANCE
        >= finite_holm[
            "wilcoxon_p_raw"
        ]
    ).all():

        raise RuntimeError(
            "At least one Holm-adjusted p-value is smaller "
            "than its raw p-value."
        )

    print(
        "Holm adjustment ordering:   PASS"
    )

    # ---------------------------------------------------------------------
    # Wilcoxon nonzero pair accounting.
    # ---------------------------------------------------------------------

    expected_nonzero = (
        contrasts_df[
            "num_nodes_higher_value"
        ]
        +
        contrasts_df[
            "num_nodes_lower_value"
        ]
    )

    if not (
        expected_nonzero
        == contrasts_df[
            "wilcoxon_nonzero_pairs"
        ]
    ).all():

        raise RuntimeError(
            "Wilcoxon nonzero-pair accounting mismatch."
        )

    print(
        "Wilcoxon pair accounting:   PASS"
    )

    # ---------------------------------------------------------------------
    # Effect size range.
    # ---------------------------------------------------------------------

    effect_sizes = (
        contrasts_df[
            "rank_biserial"
        ]
        .dropna()
        .astype(float)
    )

    if not effect_sizes.between(
        -1.0,
        1.0,
        inclusive="both",
    ).all():

        raise RuntimeError(
            "Rank-biserial effect size outside [-1,1]."
        )

    print(
        "Rank-biserial ranges:       PASS"
    )

    # ---------------------------------------------------------------------
    # Direction accounting.
    # ---------------------------------------------------------------------

    direction_total = (
        contrasts_df[
            "num_nodes_higher_value"
        ]
        +
        contrasts_df[
            "num_nodes_lower_value"
        ]
        +
        contrasts_df[
            "num_nodes_equal_value"
        ]
    )

    if not (
        direction_total
        == contrasts_df[
            "n_paired_nodes"
        ]
    ).all():

        raise RuntimeError(
            "Paired direction accounting mismatch."
        )

    print(
        "Direction accounting:       PASS"
    )

    # ---------------------------------------------------------------------
    # Rank-sum identity.
    #
    # For n nonzero paired differences, the total of the signed-rank
    # absolute ranks must equal:
    #
    #     n(n+1)/2
    #
    # even when average ranks are used for ties.
    # ---------------------------------------------------------------------

    observed_rank_total = (
        contrasts_df[
            "positive_rank_sum"
        ]
        +
        contrasts_df[
            "negative_rank_sum"
        ]
    )

    n_nonzero = (
        contrasts_df[
            "wilcoxon_nonzero_pairs"
        ].astype(float)
    )

    expected_rank_total = (
        n_nonzero
        * (
            n_nonzero
            + 1.0
        )
        / 2.0
    )

    if not np.allclose(
        observed_rank_total.to_numpy(
            dtype=float
        ),
        expected_rank_total.to_numpy(
            dtype=float
        ),
        atol=FLOAT_TOLERANCE,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Signed-rank total identity failed."
        )

    print(
        "Signed-rank total identity: PASS"
    )

    # ---------------------------------------------------------------------
    # Wilcoxon statistic should equal min(W+, W-).
    # ---------------------------------------------------------------------

    expected_statistics = np.minimum(
        contrasts_df[
            "positive_rank_sum"
        ].to_numpy(
            dtype=float
        ),
        contrasts_df[
            "negative_rank_sum"
        ].to_numpy(
            dtype=float
        ),
    )

    if not np.allclose(
        contrasts_df[
            "wilcoxon_statistic"
        ].to_numpy(
            dtype=float
        ),
        expected_statistics,
        atol=FLOAT_TOLERANCE,
        rtol=0.0,
    ):

        raise RuntimeError(
            "Wilcoxon statistic does not equal min(W+, W-)."
        )

    print(
        "Wilcoxon statistic identity: PASS"
    )

    # ---------------------------------------------------------------------
    # Monotonicity table.
    # ---------------------------------------------------------------------

    expected_monotonicity_rows = 4

    if len(
        monotonicity_df
    ) != expected_monotonicity_rows:

        raise RuntimeError(
            "Unexpected monotonicity-summary row count."
        )

    print(
        f"Monotonicity rows:           "
        f"{len(monotonicity_df)}/"
        f"{expected_monotonicity_rows} PASS"
    )

    for _, row in monotonicity_df.iterrows():

        category_total = (
            int(
                row[
                    "strictly_decreasing_nodes"
                ]
            )
            +
            int(
                row[
                    "nonincreasing_with_ties_nodes"
                ]
            )
            +
            int(
                row[
                    "constant_nodes"
                ]
            )
            +
            int(
                row[
                    "strictly_increasing_nodes"
                ]
            )
            +
            int(
                row[
                    "nondecreasing_with_ties_nodes"
                ]
            )
            +
            int(
                row[
                    "nonmonotonic_nodes"
                ]
            )
            +
            int(
                row[
                    "incomplete_nodes"
                ]
            )
        )

        if category_total != EXPECTED_NODES:

            raise RuntimeError(
                "Monotonicity category accounting mismatch "
                f"for {row['outcome']}."
            )

    print(
        "Monotonicity accounting:    PASS"
    )

    # ---------------------------------------------------------------------
    # Hypothesis table.
    # ---------------------------------------------------------------------

    if len(
        hypothesis_df
    ) != 3:

        raise RuntimeError(
            "Hypothesis summary must contain H1, H2, and H3."
        )

    observed_hypotheses = set(
        hypothesis_df[
            "hypothesis_id"
        ].tolist()
    )

    if observed_hypotheses != {
        "H1",
        "H2",
        "H3",
    }:

        raise RuntimeError(
            "Hypothesis summary does not contain exactly H1/H2/H3."
        )

    print(
        "Hypothesis-summary rows:    3/3 PASS"
    )

    print(
        "\nStatistical-output validation: PASSED"
    )


# =========================================================================
# TERMINAL REPORTING
# =========================================================================


def print_primary_jaccard_contrasts(
    contrasts_df: pd.DataFrame,
):
    """
    Print all six primary mean-Jaccard contrasts.
    """

    primary = (
        contrasts_df.loc[
            contrasts_df[
                "outcome"
            ]
            == "mean_jaccard"
        ]
        .sort_values(
            [
                "lower_rate",
                "higher_rate",
            ]
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Primary Outcome: Paired Mean-Jaccard Contrasts ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nDifference orientation:"
    )

    print(
        "  higher perturbation rate - lower perturbation rate"
    )

    print(
        "\nTherefore, a negative difference indicates lower "
        "explanation overlap at the higher perturbation rate.\n"
    )

    for _, row in primary.iterrows():

        print(
            f"{row['lower_rate']:.2f} -> "
            f"{row['higher_rate']:.2f}"
        )

        print(
            "  paired nodes: "
            f"{int(row['n_paired_nodes'])}"
        )

        print(
            "  lower-rate mean: "
            f"{format_float(row['lower_rate_mean'])}"
        )

        print(
            "  higher-rate mean: "
            f"{format_float(row['higher_rate_mean'])}"
        )

        print(
            "  mean paired difference: "
            f"{format_float(row['mean_paired_difference'])}"
        )

        print(
            "  95% paired bootstrap CI: "
            "["
            f"{format_float(row['paired_difference_ci_low'])}, "
            f"{format_float(row['paired_difference_ci_high'])}"
            "]"
        )

        print(
            "  node directions "
            "(higher / lower / equal): "
            f"{int(row['num_nodes_higher_value'])} / "
            f"{int(row['num_nodes_lower_value'])} / "
            f"{int(row['num_nodes_equal_value'])}"
        )

        print(
            "  Wilcoxon raw p: "
            f"{format_scientific(row['wilcoxon_p_raw'])}"
        )

        print(
            "  Wilcoxon Holm p: "
            f"{format_scientific(row['wilcoxon_p_holm'])}"
        )

        print(
            "  rank-biserial: "
            f"{format_float(row['rank_biserial'])}"
        )

        print()


def print_prediction_conditioned_endpoint(
    contrasts_df: pd.DataFrame,
):
    """
    Print the central H2 endpoint contrast.
    """

    row = get_contrast_row(
        contrasts_df=contrasts_df,
        outcome="mean_jaccard_pred_same",
        lower_rate=0.01,
        higher_rate=0.20,
    )

    print(
        "=" * 78
    )

    print(
        "=== Prediction-Conditioned Explanation Stability ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nPrimary endpoint contrast: 0.01 -> 0.20"
    )

    print(
        "Condition: predicted class remained unchanged."
    )

    print(
        f"Paired nodes: "
        f"{int(row['n_paired_nodes'])}"
    )

    print(
        "Mean paired difference: "
        f"{format_float(row['mean_paired_difference'])}"
    )

    print(
        "95% paired bootstrap CI: "
        "["
        f"{format_float(row['paired_difference_ci_low'])}, "
        f"{format_float(row['paired_difference_ci_high'])}"
        "]"
    )

    print(
        "Wilcoxon Holm p: "
        f"{format_scientific(row['wilcoxon_p_holm'])}"
    )

    print(
        "Rank-biserial: "
        f"{format_float(row['rank_biserial'])}"
    )


def print_monotonicity_summary(
    monotonicity_df: pd.DataFrame,
):
    """
    Print node-level trajectory summaries.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Node-Level Monotonicity Summary ==="
    )

    print(
        "=" * 78
    )

    for _, row in monotonicity_df.iterrows():

        print(
            f"\nOutcome: {row['outcome']}"
        )

        print(
            f"  complete nodes: "
            f"{int(row['complete_nodes'])}"
        )

        print(
            f"  incomplete nodes: "
            f"{int(row['incomplete_nodes'])}"
        )

        print(
            f"  strictly decreasing: "
            f"{int(row['strictly_decreasing_nodes'])}"
        )

        print(
            f"  nonincreasing with ties: "
            f"{int(row['nonincreasing_with_ties_nodes'])}"
        )

        print(
            f"  nonmonotonic: "
            f"{int(row['nonmonotonic_nodes'])}"
        )

        print(
            "  fraction monotonic nonincreasing: "
            f"{format_float(row['fraction_monotonic_nonincreasing'])}"
        )

        print(
            "  mean endpoint difference "
            "(0.20 - 0.01): "
            f"{format_float(row['mean_endpoint_difference_0_20_minus_0_01'])}"
        )


def print_hypothesis_evidence(
    hypothesis_df: pd.DataFrame,
):
    """
    Print evidence table without converting it into an automatic
    binary hypothesis verdict.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Hypothesis-Oriented Evidence Summary ==="
    )

    print(
        "=" * 78
    )

    for _, row in hypothesis_df.iterrows():

        print(
            f"\n{row['hypothesis_id']}: "
            f"{row['hypothesis']}"
        )

        print(
            f"  analysis status: "
            f"{row['analysis_status']}"
        )

        if (
            row[
                "analysis_status"
            ]
            == "testable_with_current_experiment"
        ):

            print(
                f"  primary outcome: "
                f"{row['primary_outcome']}"
            )

            print(
                f"  endpoint mean difference: "
                f"{format_float(row['endpoint_mean_difference'])}"
            )

            print(
                "  endpoint 95% paired bootstrap CI: "
                "["
                f"{format_float(row['endpoint_ci_low'])}, "
                f"{format_float(row['endpoint_ci_high'])}"
                "]"
            )

            print(
                f"  Holm-adjusted Wilcoxon p: "
                f"{format_scientific(row['wilcoxon_p_holm'])}"
            )

            print(
                f"  rank-biserial: "
                f"{format_float(row['rank_biserial'])}"
            )

            print(
                "  fraction monotonic nonincreasing: "
                f"{format_float(row['fraction_monotonic_nonincreasing'])}"
            )

        else:

            print(
                "  Current feature-masking experiment does "
                "not provide the required comparison."
            )


# =========================================================================
# MAIN
# =========================================================================


def main():

    print(
        "=" * 78
    )

    print(
        "=== Phase 10C: Paired Hypothesis-Oriented Analysis ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nPrimary statistical unit:"
    )

    print(
        "  Target node"
    )

    print(
        "\nRepeated-measure structure:"
    )

    print(
        "  The same nodes are evaluated at all four perturbation rates."
    )

    print(
        "\nPrimary outcome:"
    )

    print(
        "  Node-level mean GraphLIME top-K Jaccard similarity"
    )

    print(
        "\nPrimary uncertainty procedure:"
    )

    print(
        f"  {int(CONFIDENCE_LEVEL * 100)}% paired node-level "
        f"bootstrap confidence intervals"
    )

    print(
        f"  Bootstrap resamples: {N_BOOTSTRAP}"
    )

    print(
        f"  Bootstrap seed: {BOOTSTRAP_SEED}"
    )

    print(
        "\nSecondary inferential check:"
    )

    print(
        "  Two-sided Wilcoxon signed-rank tests"
    )

    print(
        "  Local NumPy/Python implementation; no SciPy dependency"
    )

    print(
        "  Tie-corrected normal approximation; no continuity correction"
    )

    print(
        "  Holm correction within each six-contrast outcome family"
    )

    print(
        "\nEffect size:"
    )

    print(
        "  Matched-pairs rank-biserial correlation"
    )

    # =====================================================================
    # Load validated Phase-10 node-rate table.
    # =====================================================================

    df = (
        load_and_validate_node_rate_summary()
    )

    print(
        "\n=== Loaded Phase-10 Node-Rate Dataset ==="
    )

    print(
        f"Rows:                       {len(df)}"
    )

    print(
        f"Unique target nodes:        "
        f"{df['node_id'].nunique()}"
    )

    print(
        f"Perturbation rates:         "
        f"{df['nominal_mask_rate'].nunique()}"
    )

    print(
        "Input validation:           PASSED"
    )

    # =====================================================================
    # Paired contrasts.
    # =====================================================================

    print(
        "\n=== Building Paired Rate Contrasts ==="
    )

    contrasts_df = (
        build_paired_rate_contrasts(
            df
        )
    )

    print(
        f"Constructed {len(contrasts_df)} "
        "paired contrast rows."
    )

    # =====================================================================
    # Monotonicity.
    # =====================================================================

    print(
        "\n=== Building Node-Level Monotonicity Summary ==="
    )

    monotonicity_df = (
        build_monotonicity_summary(
            df
        )
    )

    print(
        f"Constructed {len(monotonicity_df)} "
        "monotonicity-summary rows."
    )

    # =====================================================================
    # Hypothesis evidence table.
    # =====================================================================

    print(
        "\n=== Building Hypothesis-Oriented Evidence Summary ==="
    )

    hypothesis_df = (
        build_hypothesis_summary(
            contrasts_df=contrasts_df,
            monotonicity_df=monotonicity_df,
        )
    )

    print(
        f"Constructed {len(hypothesis_df)} "
        "hypothesis-summary rows."
    )

    # =====================================================================
    # Validate outputs BEFORE saving.
    # =====================================================================

    validate_outputs(
        contrasts_df=contrasts_df,
        monotonicity_df=monotonicity_df,
        hypothesis_df=hypothesis_df,
    )

    # =====================================================================
    # Save artifacts.
    # =====================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contrasts_df.to_csv(
        PAIRED_CONTRASTS_PATH,
        index=False,
    )

    monotonicity_df.to_csv(
        MONOTONICITY_SUMMARY_PATH,
        index=False,
    )

    hypothesis_df.to_csv(
        HYPOTHESIS_SUMMARY_PATH,
        index=False,
    )

    # =====================================================================
    # Terminal report.
    # =====================================================================

    print_primary_jaccard_contrasts(
        contrasts_df
    )

    print_prediction_conditioned_endpoint(
        contrasts_df
    )

    print_monotonicity_summary(
        monotonicity_df
    )

    print_hypothesis_evidence(
        hypothesis_df
    )

    # =====================================================================
    # Saved outputs.
    # =====================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Saved Phase-10C Statistical Artifacts ==="
    )

    print(
        "=" * 78
    )

    print(
        PAIRED_CONTRASTS_PATH
    )

    print(
        MONOTONICITY_SUMMARY_PATH
    )

    print(
        HYPOTHESIS_SUMMARY_PATH
    )

    # =====================================================================
    # Final status.
    # =====================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-10C Status ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nPaired node-level hypothesis-oriented analysis "
        "completed and internally validated."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The statistical outputs quantify evidence; they do not "
        "automatically establish a universal claim about GraphLIME."
    )

    print(
        "Interpretation must remain specific to the frozen Cora + "
        "GCN + GraphLIME + local active-entry masking experiment."
    )

    print(
        "\nNext methodological step:"
    )

    print(
        "Review these Phase-10C results, update the experiment log, "
        "then decide whether Phase 10 can be closed before beginning "
        "Phase 11 visualization."
    )


if __name__ == "__main__":
    main()
