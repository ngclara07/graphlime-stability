# experiments/03b_validate_perturbation_results.py
#
# Phase 9 final validation:
# Audit the complete controlled feature-perturbation dataset
# BEFORE hypothesis analysis or plotting.
#
# Run from repository root:
#
#   python -m experiments.03b_validate_perturbation_results
#
# Input:
#
#   results/perturbations/feature_mask.csv
#
# Frozen baseline artifacts:
#
#   results/baseline/explanations.csv
#   results/baseline/explanation_diagnostics.csv
#   results/baseline/kkt_validation.csv
#
# Output:
#
#   results/perturbations/feature_mask_validation.csv
#   results/perturbations/feature_mask_validation_summary.csv
#
# IMPORTANT:
#
# This script validates experimental integrity only.
# It does NOT test H1-H5 and does NOT perform substantive
# statistical analysis of explanation stability.


from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# FROZEN EXPERIMENT CONFIGURATION
# ============================================================

NUM_HOPS = 2
TOP_K = 10
RHO = 0.03

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

EXPECTED_BASELINE_EXPLAINABLE_NODES = 97

EXPECTED_OBSERVATIONS_PER_NODE = (
    len(MASK_RATES)
    * len(PERTURBATION_SEEDS)
)

EXPECTED_OBSERVATIONS_PER_RATE = (
    EXPECTED_BASELINE_EXPLAINABLE_NODES
    * len(PERTURBATION_SEEDS)
)

EXPECTED_OBSERVATIONS_PER_SEED = (
    EXPECTED_BASELINE_EXPLAINABLE_NODES
    * len(MASK_RATES)
)

EXPECTED_TOTAL_OBSERVATIONS = (
    EXPECTED_BASELINE_EXPLAINABLE_NODES
    * len(MASK_RATES)
    * len(PERTURBATION_SEEDS)
)

FLOAT_TOLERANCE = 1e-9
JACCARD_TOLERANCE = 1e-6


# ============================================================
# PATHS
# ============================================================

PERTURBATION_PATH = Path(
    "results/perturbations/feature_mask.csv"
)

BASELINE_EXPLANATIONS_PATH = Path(
    "results/baseline/explanations.csv"
)

BASELINE_DIAGNOSTICS_PATH = Path(
    "results/baseline/explanation_diagnostics.csv"
)

KKT_VALIDATION_PATH = Path(
    "results/baseline/kkt_validation.csv"
)

VALIDATION_OUTPUT_PATH = Path(
    "results/perturbations/"
    "feature_mask_validation.csv"
)

VALIDATION_SUMMARY_PATH = Path(
    "results/perturbations/"
    "feature_mask_validation_summary.csv"
)


# ============================================================
# REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "node_id",
    "true_label",
    "nominal_mask_rate",
    "perturbation_seed",
    "derived_sampling_seed",
    "num_hops",
    "top_k",
    "rho",
    "neighbourhood_size",
    "subgraph_edges",
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
    "baseline_top_k_available",
    "perturbed_active_features",
    "perturbed_degenerate_features",
    "perturbed_output_kernel_valid",
    "output_bandwidth",
    "output_centered_norm",
    "perturbed_solver_converged",
    "optimizer_iterations",
    "perturbed_num_nonzero",
    "perturbed_top_k_available",
    "explanation_available",
    "jaccard",
    "intersection_size",
    "union_size",
    "final_objective",
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
    Convert a CSV boolean column safely.

    Unlike astype(bool), this function does not interpret an
    arbitrary non-empty string such as "False" as True.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):
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

    converted = series.map(
        mapping
    )

    invalid = (
        converted.isna()
        & series.notna()
    )

    if invalid.any():

        bad_values = (
            series[
                invalid
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains "
            f"unrecognized boolean values: "
            f"{bad_values[:10]}"
        )

    if converted.isna().any():

        raise ValueError(
            f"Column '{column_name}' contains "
            "missing boolean values."
        )

    return converted.astype(bool)


# ============================================================
# INTEGER MASKING RULE
# ============================================================


def expected_mask_count(
    num_active_entries: int,
    rate: float,
) -> int:
    """
    Frozen Phase-9 rule:

        max(1, floor(rate * n_active + 0.5))

    with the result bounded above by n_active.
    """

    if num_active_entries <= 0:
        raise ValueError(
            "num_active_entries must be positive."
        )

    if not (
        0.0 < rate <= 1.0
    ):
        raise ValueError(
            "rate must satisfy 0 < rate <= 1."
        )

    count = max(
        1,
        int(
            math.floor(
                rate
                * num_active_entries
                + 0.5
            )
        ),
    )

    return min(
        count,
        num_active_entries,
    )


# ============================================================
# EXPECTED JACCARD
# ============================================================


def expected_top_k_jaccard(
    intersection_size: int,
    top_k: int = TOP_K,
):
    """
    For two valid top-K sets:

        union = 2K - intersection
        J = intersection / union
    """

    union_size = (
        2 * top_k
        - intersection_size
    )

    if union_size <= 0:
        raise ValueError(
            "Invalid top-K union size."
        )

    return (
        intersection_size
        / union_size
    )


# ============================================================
# VALIDATION REPORT HELPERS
# ============================================================


def print_check(
    label: str,
    passed: bool,
    details: str = "",
):
    """
    Standard terminal output for validation checks.
    """

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    if details:

        print(
            f"{label:<48} "
            f"{status:<4} | "
            f"{details}"
        )

    else:

        print(
            f"{label:<48} "
            f"{status}"
        )


def add_check(
    checks,
    check_name,
    passed,
    observed=None,
    expected=None,
    notes=None,
):
    """
    Store one aggregate validation check.
    """

    checks.append(
        {
            "check_name": (
                check_name
            ),
            "passed": bool(
                passed
            ),
            "observed": (
                observed
            ),
            "expected": (
                expected
            ),
            "notes": (
                notes
            ),
        }
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "=" * 78
    )

    print(
        "=== Phase-9 Perturbation Dataset Validation ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nThis is an integrity audit."
    )

    print(
        "No hypothesis testing or stability interpretation "
        "is performed here."
    )

    # ========================================================
    # Required files
    # ========================================================

    required_files = [
        PERTURBATION_PATH,
        BASELINE_EXPLANATIONS_PATH,
        BASELINE_DIAGNOSTICS_PATH,
        KKT_VALIDATION_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    # ========================================================
    # Load artifacts
    # ========================================================

    df = pd.read_csv(
        PERTURBATION_PATH
    )

    baseline_explanations = pd.read_csv(
        BASELINE_EXPLANATIONS_PATH
    )

    baseline_diagnostics = pd.read_csv(
        BASELINE_DIAGNOSTICS_PATH
    )

    kkt_df = pd.read_csv(
        KKT_VALIDATION_PATH
    )

    print(
        "\n=== Loaded Artifacts ==="
    )

    print(
        f"Perturbation rows:          {len(df)}"
    )

    print(
        "Baseline explanation rows: "
        f"{len(baseline_explanations)}"
    )

    print(
        "Baseline diagnostic rows:  "
        f"{len(baseline_diagnostics)}"
    )

    print(
        f"KKT validation rows:        {len(kkt_df)}"
    )

    # ========================================================
    # Column validation
    # ========================================================

    missing_columns = (
        REQUIRED_COLUMNS
        - set(
            df.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            "Perturbation dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    print(
        "\nRequired raw columns: PASSED"
    )

    # ========================================================
    # Parse booleans safely
    # ========================================================

    boolean_columns = [
        "pred_same",
        "baseline_top_k_available",
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

    baseline_diagnostics[
        "explanation_available"
    ] = parse_boolean_series(
        baseline_diagnostics[
            "explanation_available"
        ],
        "baseline diagnostics "
        "explanation_available",
    )

    if "overall_node_pass" in kkt_df.columns:

        kkt_df[
            "overall_node_pass"
        ] = parse_boolean_series(
            kkt_df[
                "overall_node_pass"
            ],
            "KKT overall_node_pass",
        )

    # ========================================================
    # Aggregate validation storage
    # ========================================================

    checks = []

    # ========================================================
    # 1. TOTAL ROW COUNT
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== 1. Experimental Coverage ==="
    )

    print(
        "=" * 78
    )

    row_count_pass = bool(
        len(df)
        == EXPECTED_TOTAL_OBSERVATIONS
    )

    print_check(
        "Total observation count",
        row_count_pass,
        (
            f"observed={len(df)}, "
            f"expected={EXPECTED_TOTAL_OBSERVATIONS}"
        ),
    )

    add_check(
        checks,
        "total_observation_count",
        row_count_pass,
        len(df),
        EXPECTED_TOTAL_OBSERVATIONS,
    )

    # ========================================================
    # 2. UNIQUE OBSERVATION KEYS
    # ========================================================

    key_columns = [
        "node_id",
        "nominal_mask_rate",
        "perturbation_seed",
    ]

    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    duplicate_pass = bool(
        duplicate_count == 0
    )

    print_check(
        "Unique node-rate-seed keys",
        duplicate_pass,
        f"duplicate rows={duplicate_count}",
    )

    add_check(
        checks,
        "unique_observation_keys",
        duplicate_pass,
        duplicate_count,
        0,
    )

    # ========================================================
    # 3. FROZEN BASELINE COHORT
    # ========================================================

    baseline_explainable_nodes = set(
        baseline_diagnostics.loc[
            baseline_diagnostics[
                "explanation_available"
            ],
            "node_id",
        ]
        .astype(int)
        .tolist()
    )

    observed_nodes = set(
        df[
            "node_id"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    node_count_pass = bool(
        len(
            baseline_explainable_nodes
        )
        == EXPECTED_BASELINE_EXPLAINABLE_NODES
        and
        len(
            observed_nodes
        )
        == EXPECTED_BASELINE_EXPLAINABLE_NODES
    )

    cohort_match_pass = bool(
        observed_nodes
        == baseline_explainable_nodes
    )

    print_check(
        "Number of perturbation nodes",
        node_count_pass,
        (
            f"observed={len(observed_nodes)}, "
            f"expected={EXPECTED_BASELINE_EXPLAINABLE_NODES}"
        ),
    )

    print_check(
        "Perturbation cohort matches frozen baseline",
        cohort_match_pass,
        (
            f"missing="
            f"{len(baseline_explainable_nodes - observed_nodes)}, "
            f"unexpected="
            f"{len(observed_nodes - baseline_explainable_nodes)}"
        ),
    )

    add_check(
        checks,
        "perturbation_node_count",
        node_count_pass,
        len(observed_nodes),
        EXPECTED_BASELINE_EXPLAINABLE_NODES,
    )

    add_check(
        checks,
        "baseline_cohort_match",
        cohort_match_pass,
        len(
            observed_nodes
            & baseline_explainable_nodes
        ),
        EXPECTED_BASELINE_EXPLAINABLE_NODES,
    )

    # ========================================================
    # 4. KKT COHORT CONSISTENCY
    # ========================================================

    kkt_nodes = set(
        kkt_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    kkt_cohort_pass = bool(
        kkt_nodes
        == observed_nodes
    )

    print_check(
        "Perturbation cohort matches KKT cohort",
        kkt_cohort_pass,
        (
            f"KKT nodes={len(kkt_nodes)}, "
            f"perturbation nodes={len(observed_nodes)}"
        ),
    )

    add_check(
        checks,
        "kkt_cohort_match",
        kkt_cohort_pass,
        len(
            kkt_nodes
            & observed_nodes
        ),
        EXPECTED_BASELINE_EXPLAINABLE_NODES,
    )

    if "overall_node_pass" in kkt_df.columns:

        all_kkt_pass = bool(
            kkt_df[
                "overall_node_pass"
            ].all()
        )

        print_check(
            "All frozen baseline KKT checks passed",
            all_kkt_pass,
            (
                f"passed="
                f"{int(kkt_df['overall_node_pass'].sum())}"
                f"/{len(kkt_df)}"
            ),
        )

        add_check(
            checks,
            "all_baseline_kkt_pass",
            all_kkt_pass,
            int(
                kkt_df[
                    "overall_node_pass"
                ].sum()
            ),
            len(kkt_df),
        )

    # ========================================================
    # 5. BASELINE EXPLANATION COVERAGE
    # ========================================================

    baseline_explanation_counts = (
        baseline_explanations
        .groupby(
            "node_id"
        )
        .size()
    )

    baseline_top_k_count_pass = bool(
        len(
            baseline_explanation_counts
        )
        == EXPECTED_BASELINE_EXPLAINABLE_NODES
        and
        (
            baseline_explanation_counts
            == TOP_K
        ).all()
    )

    print_check(
        "Frozen baseline has exactly TOP_K rows/node",
        baseline_top_k_count_pass,
        (
            f"nodes="
            f"{len(baseline_explanation_counts)}, "
            f"TOP_K={TOP_K}"
        ),
    )

    add_check(
        checks,
        "baseline_top_k_rows",
        baseline_top_k_count_pass,
        len(
            baseline_explanations
        ),
        (
            EXPECTED_BASELINE_EXPLAINABLE_NODES
            * TOP_K
        ),
    )

    # ========================================================
    # 6. RATE COVERAGE
    # ========================================================

    expected_rate_set = set(
        MASK_RATES
    )

    observed_rate_set = set(
        df[
            "nominal_mask_rate"
        ]
        .astype(float)
        .unique()
        .tolist()
    )

    rate_set_pass = bool(
        observed_rate_set
        == expected_rate_set
    )

    print_check(
        "Frozen perturbation-rate set",
        rate_set_pass,
        f"observed={sorted(observed_rate_set)}",
    )

    add_check(
        checks,
        "mask_rate_set",
        rate_set_pass,
        str(
            sorted(
                observed_rate_set
            )
        ),
        str(
            MASK_RATES
        ),
    )

    rate_counts = (
        df[
            "nominal_mask_rate"
        ]
        .value_counts()
        .sort_index()
    )

    rate_count_pass = bool(
        len(
            rate_counts
        )
        == len(
            MASK_RATES
        )
        and
        (
            rate_counts
            == EXPECTED_OBSERVATIONS_PER_RATE
        ).all()
    )

    print_check(
        "Observations per perturbation rate",
        rate_count_pass,
        (
            f"expected each="
            f"{EXPECTED_OBSERVATIONS_PER_RATE}"
        ),
    )

    for rate in MASK_RATES:

        observed = int(
            rate_counts.get(
                rate,
                0,
            )
        )

        print(
            f"  rate={rate:0.2f}: "
            f"{observed}"
        )

    add_check(
        checks,
        "observations_per_rate",
        rate_count_pass,
        str(
            rate_counts.to_dict()
        ),
        EXPECTED_OBSERVATIONS_PER_RATE,
    )

    # ========================================================
    # 7. SEED COVERAGE
    # ========================================================

    observed_seed_set = set(
        df[
            "perturbation_seed"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    expected_seed_set = set(
        PERTURBATION_SEEDS
    )

    seed_set_pass = bool(
        observed_seed_set
        == expected_seed_set
    )

    print_check(
        "Frozen perturbation-seed set",
        seed_set_pass,
        f"observed={sorted(observed_seed_set)}",
    )

    add_check(
        checks,
        "perturbation_seed_set",
        seed_set_pass,
        str(
            sorted(
                observed_seed_set
            )
        ),
        str(
            PERTURBATION_SEEDS
        ),
    )

    seed_counts = (
        df[
            "perturbation_seed"
        ]
        .value_counts()
        .sort_index()
    )

    seed_count_pass = bool(
        len(
            seed_counts
        )
        == len(
            PERTURBATION_SEEDS
        )
        and
        (
            seed_counts
            == EXPECTED_OBSERVATIONS_PER_SEED
        ).all()
    )

    print_check(
        "Observations per perturbation seed",
        seed_count_pass,
        (
            f"expected each="
            f"{EXPECTED_OBSERVATIONS_PER_SEED}"
        ),
    )

    add_check(
        checks,
        "observations_per_seed",
        seed_count_pass,
        str(
            seed_counts.to_dict()
        ),
        EXPECTED_OBSERVATIONS_PER_SEED,
    )

    # ========================================================
    # 8. OBSERVATIONS PER NODE
    # ========================================================

    node_counts = (
        df.groupby(
            "node_id"
        )
        .size()
    )

    node_coverage_pass = bool(
        len(
            node_counts
        )
        == EXPECTED_BASELINE_EXPLAINABLE_NODES
        and
        (
            node_counts
            == EXPECTED_OBSERVATIONS_PER_NODE
        ).all()
    )

    print_check(
        "Observations per node",
        node_coverage_pass,
        (
            f"expected each="
            f"{EXPECTED_OBSERVATIONS_PER_NODE}"
        ),
    )

    add_check(
        checks,
        "observations_per_node",
        node_coverage_pass,
        (
            f"min={int(node_counts.min())}, "
            f"max={int(node_counts.max())}"
        ),
        EXPECTED_OBSERVATIONS_PER_NODE,
    )

    # ========================================================
    # 9. COMPLETE RATE-SEED GRID PER NODE
    # ========================================================

    expected_rate_seed_pairs = {
        (
            float(rate),
            int(seed),
        )
        for rate in MASK_RATES
        for seed in PERTURBATION_SEEDS
    }

    grid_fail_nodes = []

    for node_id, group in df.groupby(
        "node_id"
    ):

        observed_pairs = {
            (
                float(rate),
                int(seed),
            )
            for rate, seed in zip(
                group[
                    "nominal_mask_rate"
                ],
                group[
                    "perturbation_seed"
                ],
            )
        }

        if (
            observed_pairs
            != expected_rate_seed_pairs
        ):

            grid_fail_nodes.append(
                int(node_id)
            )

    grid_pass = bool(
        len(
            grid_fail_nodes
        )
        == 0
    )

    print_check(
        "Complete 4 x 10 grid for every node",
        grid_pass,
        (
            f"failing nodes="
            f"{len(grid_fail_nodes)}"
        ),
    )

    add_check(
        checks,
        "complete_node_rate_seed_grid",
        grid_pass,
        len(
            grid_fail_nodes
        ),
        0,
    )

    # ========================================================
    # 10. FROZEN CONFIGURATION COLUMNS
    # ========================================================

    hops_pass = bool(
        (
            df[
                "num_hops"
            ]
            == NUM_HOPS
        ).all()
    )

    top_k_pass = bool(
        (
            df[
                "top_k"
            ]
            == TOP_K
        ).all()
    )

    rho_pass = bool(
        np.allclose(
            df[
                "rho"
            ].astype(float),
            RHO,
            atol=FLOAT_TOLERANCE,
            rtol=0.0,
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== 2. Frozen Configuration Integrity ==="
    )

    print(
        "=" * 78
    )

    print_check(
        "NUM_HOPS unchanged",
        hops_pass,
        f"expected={NUM_HOPS}",
    )

    print_check(
        "TOP_K unchanged",
        top_k_pass,
        f"expected={TOP_K}",
    )

    print_check(
        "RHO unchanged",
        rho_pass,
        f"expected={RHO}",
    )

    add_check(
        checks,
        "num_hops_frozen",
        hops_pass,
        str(
            sorted(
                df[
                    "num_hops"
                ].unique()
            )
        ),
        NUM_HOPS,
    )

    add_check(
        checks,
        "top_k_frozen",
        top_k_pass,
        str(
            sorted(
                df[
                    "top_k"
                ].unique()
            )
        ),
        TOP_K,
    )

    add_check(
        checks,
        "rho_frozen",
        rho_pass,
        str(
            sorted(
                df[
                    "rho"
                ].unique()
            )
        ),
        RHO,
    )

    # ========================================================
    # 11. BASELINE TOP-K AVAILABILITY
    # ========================================================

    baseline_available_pass = bool(
        df[
            "baseline_top_k_available"
        ].all()
    )

    print_check(
        "Baseline top-K available for all rows",
        baseline_available_pass,
        (
            f"available="
            f"{int(df['baseline_top_k_available'].sum())}"
            f"/{len(df)}"
        ),
    )

    add_check(
        checks,
        "baseline_top_k_available",
        baseline_available_pass,
        int(
            df[
                "baseline_top_k_available"
            ].sum()
        ),
        len(df),
    )

    # ========================================================
    # 12. MASKING ARITHMETIC
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== 3. Perturbation Arithmetic ==="
    )

    print(
        "=" * 78
    )

    positive_active_pass = bool(
        (
            df[
                "active_entries_before"
            ]
            > 0
        ).all()
    )

    print_check(
        "Positive active-entry counts",
        positive_active_pass,
    )

    add_check(
        checks,
        "positive_active_entries",
        positive_active_pass,
        int(
            (
                df[
                    "active_entries_before"
                ]
                <= 0
            ).sum()
        ),
        0,
    )

    expected_mask_counts = []

    for row in df[
        [
            "active_entries_before",
            "nominal_mask_rate",
        ]
    ].itertuples(
        index=False
    ):

        expected_mask_counts.append(
            expected_mask_count(
                int(
                    row.active_entries_before
                ),
                float(
                    row.nominal_mask_rate
                ),
            )
        )

    expected_mask_counts = np.array(
        expected_mask_counts,
        dtype=np.int64,
    )

    observed_mask_counts = (
        df[
            "num_entries_masked"
        ]
        .astype(int)
        .to_numpy()
    )

    mask_count_matches = (
        observed_mask_counts
        == expected_mask_counts
    )

    mask_count_pass = bool(
        mask_count_matches.all()
    )

    mask_count_failures = int(
        (
            ~mask_count_matches
        ).sum()
    )

    print_check(
        "Integer mask-count rule",
        mask_count_pass,
        (
            f"mismatches="
            f"{mask_count_failures}"
        ),
    )

    add_check(
        checks,
        "mask_count_rule",
        mask_count_pass,
        mask_count_failures,
        0,
    )

    # ========================================================
    # 13. REALIZED MASK RATE
    # ========================================================

    expected_realized_rate = (
        df[
            "num_entries_masked"
        ].astype(float)
        /
        df[
            "active_entries_before"
        ].astype(float)
    )

    realized_rate_matches = np.isclose(
        df[
            "realized_mask_rate"
        ].astype(float),
        expected_realized_rate,
        atol=FLOAT_TOLERANCE,
        rtol=0.0,
    )

    realized_rate_pass = bool(
        realized_rate_matches.all()
    )

    realized_rate_failures = int(
        (
            ~realized_rate_matches
        ).sum()
    )

    print_check(
        "Realized mask-rate arithmetic",
        realized_rate_pass,
        (
            f"mismatches="
            f"{realized_rate_failures}"
        ),
    )

    add_check(
        checks,
        "realized_mask_rate",
        realized_rate_pass,
        realized_rate_failures,
        0,
    )

    # ========================================================
    # 14. NODE-INVARIANT LOCAL STRUCTURE
    # ========================================================

    invariant_columns = [
        "neighbourhood_size",
        "subgraph_edges",
        "active_entries_before",
        "baseline_predicted_label",
        "baseline_confidence",
    ]

    invariant_failures = {}

    for column in invariant_columns:

        unique_per_node = (
            df.groupby(
                "node_id"
            )[column]
            .nunique(
                dropna=False
            )
        )

        failures = int(
            (
                unique_per_node
                != 1
            ).sum()
        )

        invariant_failures[
            column
        ] = failures

        print_check(
            f"Node-invariant {column}",
            failures == 0,
            (
                f"failing nodes="
                f"{failures}"
            ),
        )

        add_check(
            checks,
            f"node_invariant_{column}",
            failures == 0,
            failures,
            0,
        )

    # ========================================================
    # 15. PREDICTION CONSISTENCY
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== 4. Prediction-Outcome Integrity ==="
    )

    print(
        "=" * 78
    )

    expected_pred_same = (
        df[
            "baseline_predicted_label"
        ].astype(int)
        ==
        df[
            "perturbed_predicted_label"
        ].astype(int)
    )

    pred_same_matches = (
        df[
            "pred_same"
        ]
        ==
        expected_pred_same
    )

    pred_same_pass = bool(
        pred_same_matches.all()
    )

    pred_same_failures = int(
        (
            ~pred_same_matches
        ).sum()
    )

    print_check(
        "pred_same agrees with class labels",
        pred_same_pass,
        (
            f"mismatches="
            f"{pred_same_failures}"
        ),
    )

    add_check(
        checks,
        "pred_same_consistency",
        pred_same_pass,
        pred_same_failures,
        0,
    )

    # ========================================================
    # 16. CONFIDENCE BOUNDS
    # ========================================================

    confidence_columns = [
        "baseline_confidence",
        "perturbed_baseline_class_confidence",
        "perturbed_max_confidence",
    ]

    for column in confidence_columns:

        values = (
            df[column]
            .astype(float)
        )

        valid = (
            values.notna()
            &
            values.between(
                0.0,
                1.0,
                inclusive="both",
            )
        )

        failures = int(
            (
                ~valid
            ).sum()
        )

        print_check(
            f"{column} in [0,1]",
            failures == 0,
            (
                f"invalid="
                f"{failures}"
            ),
        )

        add_check(
            checks,
            f"{column}_bounds",
            failures == 0,
            failures,
            0,
        )

    # ========================================================
    # 17. CONFIDENCE DELTA ARITHMETIC
    # ========================================================

    expected_delta = (
        df[
            "perturbed_baseline_class_confidence"
        ].astype(float)
        -
        df[
            "baseline_confidence"
        ].astype(float)
    )

    delta_matches = np.isclose(
        df[
            "baseline_class_confidence_delta"
        ].astype(float),
        expected_delta,
        atol=FLOAT_TOLERANCE,
        rtol=0.0,
    )

    delta_pass = bool(
        delta_matches.all()
    )

    delta_failures = int(
        (
            ~delta_matches
        ).sum()
    )

    print_check(
        "Confidence-delta arithmetic",
        delta_pass,
        (
            f"mismatches="
            f"{delta_failures}"
        ),
    )

    add_check(
        checks,
        "confidence_delta_arithmetic",
        delta_pass,
        delta_failures,
        0,
    )

    # ========================================================
    # 18. EXPLANATION LOGIC
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== 5. Explanation-Outcome Integrity ==="
    )

    print(
        "=" * 78
    )

    available = (
        df[
            "explanation_available"
        ]
    )

    unavailable = (
        ~available
    )

    available_count = int(
        available.sum()
    )

    unavailable_count = int(
        unavailable.sum()
    )

    print(
        f"Available explanations:   "
        f"{available_count}/{len(df)}"
    )

    print(
        f"Unavailable explanations: "
        f"{unavailable_count}/{len(df)}"
    )

    # --------------------------------------------------------
    # Available explanation implies all prerequisite flags.
    # --------------------------------------------------------

    available_prerequisites = (
        df.loc[
            available,
            "perturbed_output_kernel_valid",
        ]
        &
        df.loc[
            available,
            "perturbed_solver_converged",
        ]
        &
        df.loc[
            available,
            "perturbed_top_k_available",
        ]
    )

    available_prerequisite_pass = bool(
        available_prerequisites.all()
    )

    print_check(
        "Available explanations satisfy prerequisites",
        available_prerequisite_pass,
        (
            f"checked="
            f"{available_count}"
        ),
    )

    add_check(
        checks,
        "available_explanation_prerequisites",
        available_prerequisite_pass,
        int(
            available_prerequisites.sum()
        ),
        available_count,
    )

    # --------------------------------------------------------
    # Available explanation => >= TOP_K nonzero.
    # --------------------------------------------------------

    available_nonzero_pass = bool(
        (
            df.loc[
                available,
                "perturbed_num_nonzero",
            ]
            >= TOP_K
        ).all()
    )

    print_check(
        "Available explanations have >= TOP_K nonzero",
        available_nonzero_pass,
    )

    add_check(
        checks,
        "available_nonzero_count",
        available_nonzero_pass,
        int(
            (
                df.loc[
                    available,
                    "perturbed_num_nonzero",
                ]
                < TOP_K
            ).sum()
        ),
        0,
    )

    # ========================================================
    # 19. JACCARD MISSINGNESS SEMANTICS
    # ========================================================

    available_jaccard_present = bool(
        df.loc[
            available,
            "jaccard",
        ]
        .notna()
        .all()
    )

    unavailable_jaccard_missing = bool(
        df.loc[
            unavailable,
            "jaccard",
        ]
        .isna()
        .all()
    )

    print_check(
        "Available explanations have Jaccard",
        available_jaccard_present,
    )

    print_check(
        "Unavailable explanations have missing Jaccard",
        unavailable_jaccard_missing,
    )

    add_check(
        checks,
        "available_jaccard_present",
        available_jaccard_present,
        int(
            df.loc[
                available,
                "jaccard",
            ]
            .notna()
            .sum()
        ),
        available_count,
    )

    add_check(
        checks,
        "unavailable_jaccard_missing",
        unavailable_jaccard_missing,
        int(
            df.loc[
                unavailable,
                "jaccard",
            ]
            .isna()
            .sum()
        ),
        unavailable_count,
    )

    # ========================================================
    # 20. JACCARD BOUNDS
    # ========================================================

    valid_jaccard = (
        df.loc[
            available,
            "jaccard",
        ]
        .astype(float)
    )

    jaccard_bounds_pass = bool(
        valid_jaccard.between(
            0.0,
            1.0,
            inclusive="both",
        ).all()
    )

    print_check(
        "Valid Jaccard values lie in [0,1]",
        jaccard_bounds_pass,
    )

    add_check(
        checks,
        "jaccard_bounds",
        jaccard_bounds_pass,
        int(
            (
                ~valid_jaccard.between(
                    0.0,
                    1.0,
                    inclusive="both",
                )
            ).sum()
        ),
        0,
    )

    # ========================================================
    # 21. INTERSECTION / UNION INTEGER SEMANTICS
    # ========================================================

    valid_intersection = (
        df.loc[
            available,
            "intersection_size",
        ]
    )

    valid_union = (
        df.loc[
            available,
            "union_size",
        ]
    )

    intersection_present = bool(
        valid_intersection
        .notna()
        .all()
    )

    union_present = bool(
        valid_union
        .notna()
        .all()
    )

    print_check(
        "Available intersection sizes present",
        intersection_present,
    )

    print_check(
        "Available union sizes present",
        union_present,
    )

    if (
        intersection_present
        and union_present
    ):

        intersection_values = (
            valid_intersection
            .astype(int)
        )

        union_values = (
            valid_union
            .astype(int)
        )

        intersection_bounds_pass = bool(
            intersection_values.between(
                0,
                TOP_K,
                inclusive="both",
            ).all()
        )

        expected_union_values = (
            2 * TOP_K
            - intersection_values
        )

        union_formula_pass = bool(
            (
                union_values
                == expected_union_values
            ).all()
        )

        print_check(
            "Intersection sizes in [0, TOP_K]",
            intersection_bounds_pass,
        )

        print_check(
            "Union = 2*TOP_K - intersection",
            union_formula_pass,
        )

        add_check(
            checks,
            "intersection_bounds",
            intersection_bounds_pass,
            int(
                (
                    ~intersection_values.between(
                        0,
                        TOP_K,
                        inclusive="both",
                    )
                ).sum()
            ),
            0,
        )

        add_check(
            checks,
            "union_formula",
            union_formula_pass,
            int(
                (
                    union_values
                    != expected_union_values
                ).sum()
            ),
            0,
        )

    else:

        intersection_bounds_pass = False
        union_formula_pass = False

        add_check(
            checks,
            "intersection_bounds",
            False,
            "missing values",
            "all available rows present",
        )

        add_check(
            checks,
            "union_formula",
            False,
            "missing values",
            "all available rows present",
        )

    # ========================================================
    # 22. JACCARD EXACT MATHEMATICAL CONSISTENCY
    # ========================================================

    if (
        intersection_present
        and union_present
    ):

        expected_jaccard_values = (
            valid_intersection.astype(float)
            /
            valid_union.astype(float)
        )

        observed_jaccard_values = (
            valid_jaccard
        )

        jaccard_formula_matches = np.isclose(
            observed_jaccard_values,
            expected_jaccard_values,
            atol=JACCARD_TOLERANCE,
            rtol=0.0,
        )

        jaccard_formula_pass = bool(
            jaccard_formula_matches.all()
        )

        jaccard_formula_failures = int(
            (
                ~jaccard_formula_matches
            ).sum()
        )

    else:

        jaccard_formula_pass = False
        jaccard_formula_failures = (
            available_count
        )

    print_check(
        "Jaccard = intersection / union",
        jaccard_formula_pass,
        (
            f"mismatches="
            f"{jaccard_formula_failures}"
        ),
    )

    add_check(
        checks,
        "jaccard_formula",
        jaccard_formula_pass,
        jaccard_formula_failures,
        0,
    )

    # ========================================================
    # 23. DISCRETE TOP-K JACCARD SUPPORT
    # ========================================================

    allowed_jaccard_values = np.array(
        [
            expected_top_k_jaccard(
                intersection,
                TOP_K,
            )
            for intersection in range(
                0,
                TOP_K + 1,
            )
        ],
        dtype=float,
    )

    discrete_support_failures = 0

    for value in valid_jaccard:

        if not np.any(
            np.isclose(
                value,
                allowed_jaccard_values,
                atol=JACCARD_TOLERANCE,
                rtol=0.0,
            )
        ):

            discrete_support_failures += 1

    discrete_jaccard_pass = bool(
        discrete_support_failures
        == 0
    )

    print_check(
        "Jaccard values lie on valid TOP_K support",
        discrete_jaccard_pass,
        (
            f"invalid="
            f"{discrete_support_failures}"
        ),
    )

    add_check(
        checks,
        "jaccard_discrete_support",
        discrete_jaccard_pass,
        discrete_support_failures,
        0,
    )

    # ========================================================
    # 24. UNAVAILABLE EXPLANATION SEMANTICS
    # ========================================================

    unavailable_intersection_missing = bool(
        df.loc[
            unavailable,
            "intersection_size",
        ]
        .isna()
        .all()
    )

    unavailable_union_missing = bool(
        df.loc[
            unavailable,
            "union_size",
        ]
        .isna()
        .all()
    )

    print_check(
        "Unavailable intersection sizes are missing",
        unavailable_intersection_missing,
    )

    print_check(
        "Unavailable union sizes are missing",
        unavailable_union_missing,
    )

    add_check(
        checks,
        "unavailable_intersection_missing",
        unavailable_intersection_missing,
        int(
            df.loc[
                unavailable,
                "intersection_size",
            ]
            .isna()
            .sum()
        ),
        unavailable_count,
    )

    add_check(
        checks,
        "unavailable_union_missing",
        unavailable_union_missing,
        int(
            df.loc[
                unavailable,
                "union_size",
            ]
            .isna()
            .sum()
        ),
        unavailable_count,
    )

    # ========================================================
    # 25. FAILURE REASON CONSISTENCY
    # ========================================================

    available_failure_missing = bool(
        df.loc[
            available,
            "failure_reason",
        ]
        .isna()
        .all()
    )

    unavailable_failure_present = bool(
        df.loc[
            unavailable,
            "failure_reason",
        ]
        .notna()
        .all()
    )

    print_check(
        "Available rows have no failure reason",
        available_failure_missing,
    )

    print_check(
        "Unavailable rows have failure reason",
        unavailable_failure_present,
    )

    add_check(
        checks,
        "available_failure_reason_missing",
        available_failure_missing,
        int(
            df.loc[
                available,
                "failure_reason",
            ]
            .isna()
            .sum()
        ),
        available_count,
    )

    add_check(
        checks,
        "unavailable_failure_reason_present",
        unavailable_failure_present,
        int(
            df.loc[
                unavailable,
                "failure_reason",
            ]
            .notna()
            .sum()
        ),
        unavailable_count,
    )

    # ========================================================
    # 26. FAILURE REASON DISTRIBUTION
    # ========================================================

    print(
        "\n=== Recorded Explanation-Unavailability Reasons ==="
    )

    if unavailable_count == 0:

        print(
            "None."
        )

    else:

        failure_counts = (
            df.loc[
                unavailable,
                "failure_reason",
            ]
            .fillna(
                "<missing>"
            )
            .value_counts(
                dropna=False
            )
        )

        print(
            failure_counts.to_string()
        )

    # ========================================================
    # 27. SOLVER LOGIC
    # ========================================================

    converged = (
        df[
            "perturbed_solver_converged"
        ]
    )

    converged_iterations_pass = bool(
        (
            df.loc[
                converged,
                "optimizer_iterations",
            ]
            > 0
        ).all()
    )

    print_check(
        "Converged solutions have positive iterations",
        converged_iterations_pass,
    )

    add_check(
        checks,
        "converged_positive_iterations",
        converged_iterations_pass,
        int(
            (
                df.loc[
                    converged,
                    "optimizer_iterations",
                ]
                <= 0
            ).sum()
        ),
        0,
    )

    # ========================================================
    # 28. FINITE OBJECTIVE FOR CONVERGED SOLUTIONS
    # ========================================================

    converged_objectives = (
        df.loc[
            converged,
            "final_objective",
        ]
    )

    finite_objective_pass = bool(
        np.isfinite(
            converged_objectives
            .astype(float)
        ).all()
    )

    print_check(
        "Converged solutions have finite objective",
        finite_objective_pass,
    )

    add_check(
        checks,
        "converged_finite_objective",
        finite_objective_pass,
        int(
            (
                ~np.isfinite(
                    converged_objectives
                    .astype(float)
                )
            ).sum()
        ),
        0,
    )

    # ========================================================
    # 29. DERIVED SAMPLING SEED UNIQUENESS
    # ========================================================

    seed_key_pairs = df[
        [
            "node_id",
            "nominal_mask_rate",
            "perturbation_seed",
            "derived_sampling_seed",
        ]
    ].drop_duplicates()

    derived_seed_mapping_pass = bool(
        len(
            seed_key_pairs
        )
        == len(df)
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== 6. Reproducibility Metadata ==="
    )

    print(
        "=" * 78
    )

    print_check(
        "One derived seed per observation key",
        derived_seed_mapping_pass,
        (
            f"unique mappings="
            f"{len(seed_key_pairs)}"
        ),
    )

    add_check(
        checks,
        "derived_seed_mapping",
        derived_seed_mapping_pass,
        len(
            seed_key_pairs
        ),
        len(df),
    )

    # ========================================================
    # 30. ROW-LEVEL VALIDATION TABLE
    # ========================================================

    validation_df = df[
        key_columns
    ].copy()

    validation_df[
        "mask_count_valid"
    ] = mask_count_matches

    validation_df[
        "realized_rate_valid"
    ] = realized_rate_matches

    validation_df[
        "pred_same_valid"
    ] = pred_same_matches

    validation_df[
        "confidence_delta_valid"
    ] = delta_matches

    # Default row-level explanation validation.
    validation_df[
        "explanation_semantics_valid"
    ] = True

    # Available rows.
    available_index = df.index[
        available
    ]

    validation_df.loc[
        available_index,
        "explanation_semantics_valid",
    ] = (
        df.loc[
            available,
            "perturbed_output_kernel_valid",
        ].to_numpy()
        &
        df.loc[
            available,
            "perturbed_solver_converged",
        ].to_numpy()
        &
        df.loc[
            available,
            "perturbed_top_k_available",
        ].to_numpy()
        &
        (
            df.loc[
                available,
                "perturbed_num_nonzero",
            ].to_numpy()
            >= TOP_K
        )
        &
        df.loc[
            available,
            "jaccard",
        ].notna().to_numpy()
        &
        df.loc[
            available,
            "failure_reason",
        ].isna().to_numpy()
    )

    # Unavailable rows.
    unavailable_index = df.index[
        unavailable
    ]

    validation_df.loc[
        unavailable_index,
        "explanation_semantics_valid",
    ] = (
        df.loc[
            unavailable,
            "jaccard",
        ].isna().to_numpy()
        &
        df.loc[
            unavailable,
            "intersection_size",
        ].isna().to_numpy()
        &
        df.loc[
            unavailable,
            "union_size",
        ].isna().to_numpy()
        &
        df.loc[
            unavailable,
            "failure_reason",
        ].notna().to_numpy()
    )

    validation_df[
        "row_validation_pass"
    ] = (
        validation_df[
            "mask_count_valid"
        ]
        &
        validation_df[
            "realized_rate_valid"
        ]
        &
        validation_df[
            "pred_same_valid"
        ]
        &
        validation_df[
            "confidence_delta_valid"
        ]
        &
        validation_df[
            "explanation_semantics_valid"
        ]
    )

    row_validation_pass_count = int(
        validation_df[
            "row_validation_pass"
        ].sum()
    )

    row_validation_fail_count = int(
        (
            ~validation_df[
                "row_validation_pass"
            ]
        ).sum()
    )

    # ========================================================
    # 31. SAVE VALIDATION ARTIFACTS
    # ========================================================

    VALIDATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    checks_df = pd.DataFrame(
        checks
    )

    checks_df.to_csv(
        VALIDATION_SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # 32. OVERALL STATUS
    # ========================================================

    aggregate_checks_pass = bool(
        checks_df[
            "passed"
        ].all()
    )

    all_rows_pass = bool(
        row_validation_fail_count
        == 0
    )

    overall_pass = bool(
        aggregate_checks_pass
        and all_rows_pass
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-9 Validation Summary ==="
    )

    print(
        "=" * 78
    )

    print(
        f"\nRaw observations:             "
        f"{len(df)}"
    )

    print(
        f"Expected observations:        "
        f"{EXPECTED_TOTAL_OBSERVATIONS}"
    )

    print(
        f"Unique nodes:                 "
        f"{df['node_id'].nunique()}"
    )

    print(
        f"Unique rates:                 "
        f"{df['nominal_mask_rate'].nunique()}"
    )

    print(
        f"Unique perturbation seeds:    "
        f"{df['perturbation_seed'].nunique()}"
    )

    print(
        f"\nAvailable explanations:       "
        f"{available_count}"
    )

    print(
        f"Unavailable explanations:     "
        f"{unavailable_count}"
    )

    print(
        f"\nRow-level validation PASS:    "
        f"{row_validation_pass_count}/{len(df)}"
    )

    print(
        f"Row-level validation FAIL:    "
        f"{row_validation_fail_count}/{len(df)}"
    )

    aggregate_pass_count = int(
        checks_df[
            "passed"
        ].sum()
    )

    print(
        f"\nAggregate checks PASS:        "
        f"{aggregate_pass_count}/{len(checks_df)}"
    )

    print(
        "Aggregate checks FAIL:        "
        f"{len(checks_df) - aggregate_pass_count}"
        f"/{len(checks_df)}"
    )

    print(
        "\n=== Saved Validation Artifacts ==="
    )

    print(
        VALIDATION_OUTPUT_PATH
    )

    print(
        VALIDATION_SUMMARY_PATH
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Final Validation Decision ==="
    )

    print(
        "=" * 78
    )

    if overall_pass:

        print(
            "\nPASS"
        )

        print(
            "\nThe complete 3,880-observation "
            "feature-perturbation dataset passed "
            "the pre-specified structural, arithmetic, "
            "reproducibility, prediction, and explanation "
            "integrity checks."
        )

        print(
            "\nPhase 9 may now be CLOSED."
        )

        print(
            "\nDo not modify feature_mask.csv."
        )

        print(
            "\nNext phase:"
        )

        print(
            "Phase 10 — statistical aggregation and "
            "hypothesis-oriented analysis."
        )

    else:

        print(
            "\nREVIEW REQUIRED"
        )

        print(
            "\nAt least one validation check failed."
        )

        print(
            "Do NOT begin hypothesis analysis or plotting yet."
        )

        print(
            "\nInspect:"
        )

        print(
            VALIDATION_OUTPUT_PATH
        )

        print(
            VALIDATION_SUMMARY_PATH
        )

        print(
            "\nand diagnose the failed checks without "
            "changing the frozen experimental protocol."
        )


if __name__ == "__main__":
    main()
