# experiments/02d_validate_baseline_solutions.py
#
# Milestone 2D / Experiment E007:
# KKT validation of frozen baseline HSIC-Lasso solutions.
#
# Purpose:
#   1. Load the frozen Cora GCN.
#   2. Load the frozen 100-node sample.
#   3. Load the baseline explanation diagnostics generated
#      by experiments/02c_generate_explanations.py.
#   4. Identify the 97 nodes for which a baseline GraphLIME
#      explanation exists.
#   5. Reconstruct each GraphLIME HSIC-Lasso problem using
#      exactly the frozen configuration.
#   6. Re-solve using exactly the frozen optimization
#      configuration.
#   7. Independently evaluate approximate KKT optimality.
#   8. Save one KKT diagnostic row per explainable node.
#
# IMPORTANT:
#
#   This is a VALIDATION experiment.
#
#   It MUST NOT:
#       - retune rho;
#       - change NUM_HOPS;
#       - change TOP_K;
#       - replace selected nodes;
#       - change the GCN;
#       - modify baseline CSV artifacts;
#       - change optimizer settings in response to results.
#
# Frozen GraphLIME configuration:
#
#       NUM_HOPS = 2
#       TOP_K    = 10
#       RHO      = 0.03
#
# Frozen optimization configuration:
#
#       MAX_ITER            = 10000
#       OPTIMIZER_TOLERANCE = 1e-8
#       ZERO_TOLERANCE      = 1e-8
#
# Pre-specified KKT diagnostic tolerance:
#
#       KKT_TOLERANCE = 1e-4
#
# Run from repository root:
#
#   python -m experiments.02d_validate_baseline_solutions


from pathlib import Path

import pandas as pd
import torch

from src.data import load_cora
from src.explain import (
    build_feature_kernels,
    build_output_kernel,
    get_n_hop_neighborhood,
    prepare_hsic_lasso_problem,
    solve_prepared_hsic_lasso,
    top_k_graphlime_features,
)
from src.model import GCN
from src.utils import set_seed


# ============================================================
# FROZEN EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

NUM_SELECTED_NODES = 100

NUM_HOPS = 2

TOP_K = 10

RHO = 0.03


# ============================================================
# FROZEN OPTIMIZATION CONFIGURATION
# ============================================================

MAX_ITER = 10000

OPTIMIZER_TOLERANCE = 1e-8

ZERO_TOLERANCE = 1e-8


# ============================================================
# PRE-SPECIFIED KKT VALIDATION CONFIGURATION
# ============================================================
#
# This is NOT the optimizer stopping tolerance.
#
# The optimization is performed in float32. We therefore use
# a separate, practical first-order residual tolerance for
# diagnostic KKT validation.
#
# This value is specified before inspecting KKT results.
# ============================================================

KKT_TOLERANCE = 1e-4


# ============================================================
# FILE LOCATIONS
# ============================================================

CHECKPOINT_PATH = Path(
    "models/cora_gcn_seed42.pt"
)

SELECTED_NODES_PATH = Path(
    "results/baseline/selected_nodes.csv"
)

EXPLANATIONS_PATH = Path(
    "results/baseline/explanations.csv"
)

BASELINE_DIAGNOSTICS_PATH = Path(
    "results/baseline/explanation_diagnostics.csv"
)

OUTPUT_DIR = Path(
    "results/baseline"
)

KKT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "kkt_validation.csv"
)


# ============================================================
# KKT VALIDATION FUNCTION
# ============================================================


def compute_kkt_diagnostics(
    design_matrix: torch.Tensor,
    output_vector: torch.Tensor,
    beta: torch.Tensor,
    rho: float,
    zero_tolerance: float,
    kkt_tolerance: float,
):
    """
    Evaluate approximate KKT conditions for:

        min_{beta >= 0}
            0.5 ||X beta - y||_2^2
            + rho * 1^T beta

    For the smooth least-squares term:

        g = X^T (X beta - y)

    Define:

        r = g + rho

    KKT conditions:

        beta_j > 0  =>  r_j = 0

        beta_j = 0  =>  r_j >= 0

    Parameters
    ----------
    design_matrix : torch.Tensor
        X with shape [n_samples, n_features].

    output_vector : torch.Tensor
        y with shape [n_samples].

    beta : torch.Tensor
        Non-negative coefficient vector.

    rho : float
        Frozen L1 regularization parameter.

    zero_tolerance : float
        Threshold used to distinguish active from inactive
        coefficients.

    kkt_tolerance : float
        Pre-specified approximate KKT tolerance.

    Returns
    -------
    dict
        Numerical KKT diagnostics.
    """

    if design_matrix.ndim != 2:

        raise ValueError(
            "design_matrix must be 2-dimensional."
        )

    if output_vector.ndim != 1:

        raise ValueError(
            "output_vector must be 1-dimensional."
        )

    if beta.ndim != 1:

        raise ValueError(
            "beta must be 1-dimensional."
        )

    if (
        design_matrix.shape[0]
        != output_vector.shape[0]
    ):

        raise ValueError(
            "Design matrix and output vector "
            "sample dimensions do not agree."
        )

    if (
        design_matrix.shape[1]
        != beta.shape[0]
    ):

        raise ValueError(
            "Design matrix and beta feature "
            "dimensions do not agree."
        )

    if not torch.isfinite(
        design_matrix
    ).all():

        raise RuntimeError(
            "Non-finite design matrix detected."
        )

    if not torch.isfinite(
        output_vector
    ).all():

        raise RuntimeError(
            "Non-finite output vector detected."
        )

    if not torch.isfinite(
        beta
    ).all():

        raise RuntimeError(
            "Non-finite beta detected."
        )

    # --------------------------------------------------------
    # Smooth gradient
    # --------------------------------------------------------

    residual = (
        design_matrix
        @ beta
        - output_vector
    )

    gradient = (
        design_matrix.T
        @ residual
    )

    # --------------------------------------------------------
    # KKT stationarity/slack vector
    # --------------------------------------------------------
    #
    # For beta >= 0 and rho > 0:
    #
    #   active beta:
    #       gradient + rho ~= 0
    #
    #   inactive beta:
    #       gradient + rho >= 0
    # --------------------------------------------------------

    kkt_vector = (
        gradient
        + rho
    )

    active_mask = (
        beta
        > zero_tolerance
    )

    inactive_mask = (
        ~active_mask
    )

    num_active = int(
        active_mask.sum().item()
    )

    num_inactive = int(
        inactive_mask.sum().item()
    )

    # --------------------------------------------------------
    # Active-coordinate stationarity
    # --------------------------------------------------------

    if num_active > 0:

        active_residuals = (
            kkt_vector[
                active_mask
            ]
            .abs()
        )

        max_active_stationarity_residual = float(
            active_residuals.max().item()
        )

        mean_active_stationarity_residual = float(
            active_residuals.mean().item()
        )

        active_kkt_pass = bool(
            max_active_stationarity_residual
            <= kkt_tolerance
        )

    else:

        max_active_stationarity_residual = 0.0

        mean_active_stationarity_residual = 0.0

        active_kkt_pass = True

    # --------------------------------------------------------
    # Inactive-coordinate dual feasibility
    # --------------------------------------------------------

    if num_inactive > 0:

        inactive_slacks = (
            kkt_vector[
                inactive_mask
            ]
        )

        min_inactive_slack = float(
            inactive_slacks.min().item()
        )

        max_inactive_violation = float(
            torch.clamp(
                -inactive_slacks,
                min=0.0,
            )
            .max()
            .item()
        )

        inactive_kkt_pass = bool(
            min_inactive_slack
            >= -kkt_tolerance
        )

    else:

        min_inactive_slack = float("nan")

        max_inactive_violation = 0.0

        inactive_kkt_pass = True

    # --------------------------------------------------------
    # Primal feasibility
    # --------------------------------------------------------

    min_beta = float(
        beta.min().item()
    )

    primal_feasible = bool(
        min_beta
        >= -zero_tolerance
    )

    # --------------------------------------------------------
    # Complementarity diagnostic
    # --------------------------------------------------------
    #
    # For the non-negative constraint, beta_j * r_j should
    # be approximately zero.
    # --------------------------------------------------------

    complementarity = (
        beta
        * kkt_vector
    )

    max_complementarity_residual = float(
        complementarity
        .abs()
        .max()
        .item()
    )

    # This is reported as a diagnostic. The primary KKT pass
    # criterion below is based directly on active stationarity,
    # inactive dual feasibility, and primal feasibility.

    overall_kkt_pass = bool(
        active_kkt_pass
        and inactive_kkt_pass
        and primal_feasible
    )

    return {
        "num_active": (
            num_active
        ),
        "num_inactive": (
            num_inactive
        ),
        "max_active_stationarity_residual": (
            max_active_stationarity_residual
        ),
        "mean_active_stationarity_residual": (
            mean_active_stationarity_residual
        ),
        "min_inactive_slack": (
            min_inactive_slack
        ),
        "max_inactive_violation": (
            max_inactive_violation
        ),
        "min_beta": (
            min_beta
        ),
        "max_complementarity_residual": (
            max_complementarity_residual
        ),
        "active_kkt_pass": (
            active_kkt_pass
        ),
        "inactive_kkt_pass": (
            inactive_kkt_pass
        ),
        "primal_feasible": (
            primal_feasible
        ),
        "overall_kkt_pass": (
            overall_kkt_pass
        ),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(
        SEED
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Validate required artifacts
    # ========================================================

    required_paths = [
        CHECKPOINT_PATH,
        SELECTED_NODES_PATH,
        EXPLANATIONS_PATH,
        BASELINE_DIAGNOSTICS_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                "Required frozen artifact "
                f"not found: {path}"
            )

    # ========================================================
    # Load Cora
    # ========================================================

    dataset, data = load_cora()

    data = data.to(
        device
    )

    # ========================================================
    # Load frozen checkpoint
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if checkpoint["seed"] != SEED:

        raise RuntimeError(
            "Checkpoint seed does not match "
            "the frozen experiment seed."
        )

    print(
        "\n=== Loaded Frozen Baseline ==="
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Training seed: {checkpoint['seed']}"
    )

    print(
        f"Best epoch: {checkpoint['best_epoch']}"
    )

    print(
        "Validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    print(
        "Test accuracy: "
        f"{checkpoint['test_accuracy']:.4f}"
    )

    # ========================================================
    # Reconstruct frozen model
    # ========================================================

    model = GCN(
        in_channels=checkpoint[
            "num_features"
        ],
        hidden_channels=checkpoint[
            "hidden_channels"
        ],
        out_channels=checkpoint[
            "num_classes"
        ],
        dropout=checkpoint[
            "dropout"
        ],
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # Frozen predictions
    # ========================================================

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        probabilities = (
            logits.softmax(
                dim=-1
            )
        )

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

    # ========================================================
    # Revalidate test accuracy
    # ========================================================

    reconstructed_test_accuracy = float(
        (
            predictions[
                data.test_mask
            ]
            ==
            data.y[
                data.test_mask
            ]
        )
        .float()
        .mean()
        .item()
    )

    if abs(
        reconstructed_test_accuracy
        - checkpoint["test_accuracy"]
    ) > 1e-6:

        raise RuntimeError(
            "Reconstructed test accuracy does "
            "not match the frozen checkpoint."
        )

    print(
        "\nReconstructed test accuracy: "
        f"{reconstructed_test_accuracy:.4f}"
    )

    # ========================================================
    # Load frozen artifacts
    # ========================================================

    selected_df = pd.read_csv(
        SELECTED_NODES_PATH
    )

    explanations_df = pd.read_csv(
        EXPLANATIONS_PATH
    )

    baseline_diagnostics_df = pd.read_csv(
        BASELINE_DIAGNOSTICS_PATH
    )

    # ========================================================
    # Validate selected-node artifact
    # ========================================================

    if (
        len(selected_df)
        != NUM_SELECTED_NODES
    ):

        raise RuntimeError(
            "Expected exactly "
            f"{NUM_SELECTED_NODES} selected nodes."
        )

    if not selected_df[
        "node_id"
    ].is_unique:

        raise RuntimeError(
            "Selected-node artifact contains "
            "duplicate node IDs."
        )

    selected_node_ids = (
        selected_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    # ========================================================
    # Validate baseline diagnostics
    # ========================================================

    required_diagnostic_columns = {
        "node_id",
        "output_kernel_valid",
        "explanation_available",
        "failure_reason",
        "converged",
        "num_nonzero",
        "num_top_features",
    }

    missing_diagnostic_columns = (
        required_diagnostic_columns
        - set(
            baseline_diagnostics_df.columns
        )
    )

    if missing_diagnostic_columns:

        raise RuntimeError(
            "Baseline diagnostic artifact is "
            "missing columns: "
            f"{sorted(missing_diagnostic_columns)}"
        )

    if (
        len(
            baseline_diagnostics_df
        )
        != NUM_SELECTED_NODES
    ):

        raise RuntimeError(
            "Baseline diagnostics must contain "
            f"{NUM_SELECTED_NODES} rows."
        )

    diagnostic_node_ids = (
        baseline_diagnostics_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    if (
        diagnostic_node_ids
        != selected_node_ids
    ):

        raise RuntimeError(
            "Baseline diagnostic node ordering "
            "does not match selected_nodes.csv."
        )

    # ========================================================
    # Identify explainable baseline cohort
    # ========================================================

    explainable_df = (
        baseline_diagnostics_df[
            baseline_diagnostics_df[
                "explanation_available"
            ]
        ]
        .copy()
    )

    explainable_node_ids = (
        explainable_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    num_explainable = len(
        explainable_node_ids
    )

    num_output_degenerate = (
        NUM_SELECTED_NODES
        - num_explainable
    )

    print(
        "\n=== Baseline Explanation Cohort ==="
    )

    print(
        f"Selected nodes:          "
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"Explainable nodes:       "
        f"{num_explainable}"
    )

    print(
        f"Unavailable explanations:"
        f" {num_output_degenerate}"
    )

    # --------------------------------------------------------
    # Expected result from E006
    # --------------------------------------------------------

    if num_explainable != 97:

        raise RuntimeError(
            "Expected 97 baseline-explainable "
            "nodes from E006, but found "
            f"{num_explainable}."
        )

    # ========================================================
    # Validate explanation artifact
    # ========================================================

    expected_explanation_rows = (
        num_explainable
        * TOP_K
    )

    if (
        len(explanations_df)
        != expected_explanation_rows
    ):

        raise RuntimeError(
            "Unexpected number of baseline "
            "explanation rows. Expected "
            f"{expected_explanation_rows}, found "
            f"{len(explanations_df)}."
        )

    explanation_node_ids = set(
        explanations_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    if (
        explanation_node_ids
        != set(
            explainable_node_ids
        )
    ):

        raise RuntimeError(
            "Baseline explanation node IDs "
            "do not match the explainable cohort."
        )

    explanation_counts = (
        explanations_df
        .groupby(
            "node_id"
        )
        .size()
    )

    if not (
        explanation_counts
        == TOP_K
    ).all():

        raise RuntimeError(
            "At least one explainable node does "
            "not have exactly TOP_K baseline rows."
        )

    # ========================================================
    # Frozen validation configuration report
    # ========================================================

    print(
        "\n=== Frozen KKT Validation Configuration ==="
    )

    print(
        f"NUM_HOPS: {NUM_HOPS}"
    )

    print(
        f"TOP_K: {TOP_K}"
    )

    print(
        f"RHO: {RHO}"
    )

    print(
        f"MAX_ITER: {MAX_ITER}"
    )

    print(
        "OPTIMIZER_TOLERANCE: "
        f"{OPTIMIZER_TOLERANCE:.1e}"
    )

    print(
        "ZERO_TOLERANCE: "
        f"{ZERO_TOLERANCE:.1e}"
    )

    print(
        "KKT_TOLERANCE: "
        f"{KKT_TOLERANCE:.1e}"
    )

    # ========================================================
    # KKT result container
    # ========================================================

    kkt_records = []

    # ========================================================
    # Validate each explainable baseline solution
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== KKT Validation of Baseline "
        "HSIC-Lasso Solutions ==="
    )

    print(
        "=" * 78
    )

    for node_position, node_id in enumerate(
        explainable_node_ids,
        start=1,
    ):

        # ----------------------------------------------------
        # Extract frozen neighbourhood
        # ----------------------------------------------------

        (
            subset,
            sub_edge_index,
            mapping,
        ) = get_n_hop_neighborhood(
            node_id=node_id,
            edge_index=data.edge_index,
            num_hops=NUM_HOPS,
        )

        neighbourhood_size = int(
            subset.numel()
        )

        target_local_index = int(
            mapping.item()
        )

        mapped_node_id = int(
            subset[
                target_local_index
            ].item()
        )

        if mapped_node_id != node_id:

            raise RuntimeError(
                "Neighbourhood mapping failed "
                f"for node {node_id}."
            )

        # ----------------------------------------------------
        # Reconstruct local GraphLIME inputs
        # ----------------------------------------------------

        local_x = data.x[
            subset
        ]

        local_y = probabilities[
            subset
        ]

        # ----------------------------------------------------
        # Reconstruct feature kernels
        # ----------------------------------------------------

        (
            feature_kernels,
            feature_bandwidths,
            degenerate_features,
        ) = build_feature_kernels(
            local_x
        )

        num_active_features = len(
            feature_kernels
        )

        if num_active_features == 0:

            raise RuntimeError(
                "Explainable baseline node "
                f"{node_id} unexpectedly has "
                "zero active features."
            )

        # ----------------------------------------------------
        # Reconstruct output kernel
        # ----------------------------------------------------

        (
            output_kernel,
            output_bandwidth,
            output_centered_norm,
        ) = build_output_kernel(
            local_y
        )

        # ----------------------------------------------------
        # Reconstruct flattened HSIC-Lasso problem
        # ----------------------------------------------------

        problem = (
            prepare_hsic_lasso_problem(
                feature_kernels=(
                    feature_kernels
                ),
                output_kernel=(
                    output_kernel
                ),
            )
        )

        # ----------------------------------------------------
        # Re-solve using EXACT frozen optimizer settings
        # ----------------------------------------------------

        result = (
            solve_prepared_hsic_lasso(
                problem=problem,
                rho=RHO,
                max_iter=MAX_ITER,
                tolerance=(
                    OPTIMIZER_TOLERANCE
                ),
                zero_tolerance=(
                    ZERO_TOLERANCE
                ),
            )
        )

        beta = result[
            "beta"
        ]

        # ----------------------------------------------------
        # Verify the reconstructed solution reproduces
        # the saved baseline top-K explanation.
        # ----------------------------------------------------

        reconstructed_top_features = (
            top_k_graphlime_features(
                result[
                    "coefficient_by_feature"
                ],
                top_k=TOP_K,
                zero_tolerance=(
                    ZERO_TOLERANCE
                ),
            )
        )

        reconstructed_top_ids = [
            int(feature_id)
            for (
                feature_id,
                coefficient,
            ) in reconstructed_top_features
        ]

        saved_node_explanation = (
            explanations_df[
                explanations_df[
                    "node_id"
                ]
                == node_id
            ]
            .sort_values(
                "rank"
            )
        )

        saved_top_ids = (
            saved_node_explanation[
                "feature_id"
            ]
            .astype(int)
            .tolist()
        )

        top_k_reproduced = bool(
            reconstructed_top_ids
            == saved_top_ids
        )

        # ----------------------------------------------------
        # Extract X and y from the prepared problem
        # ----------------------------------------------------
        #
        # The prepared problem is expected to expose the
        # flattened design matrix and output vector used by
        # solve_prepared_hsic_lasso.
        # ----------------------------------------------------

        if "design_matrix" in problem:

            design_matrix = problem[
                "design_matrix"
            ]

        elif "X" in problem:

            design_matrix = problem[
                "X"
            ]

        else:

            raise KeyError(
                "Prepared HSIC-Lasso problem does "
                "not contain 'design_matrix' or 'X'. "
                "Inspect "
                "prepare_hsic_lasso_problem() in "
                "src/explain.py."
            )

        if "output_vector" in problem:

            output_vector = problem[
                "output_vector"
            ]

        elif "y" in problem:

            output_vector = problem[
                "y"
            ]

        else:

            raise KeyError(
                "Prepared HSIC-Lasso problem does "
                "not contain 'output_vector' or 'y'. "
                "Inspect "
                "prepare_hsic_lasso_problem() in "
                "src/explain.py."
            )

        # ----------------------------------------------------
        # Independent KKT validation
        # ----------------------------------------------------

        kkt = compute_kkt_diagnostics(
            design_matrix=design_matrix,
            output_vector=output_vector,
            beta=beta,
            rho=RHO,
            zero_tolerance=ZERO_TOLERANCE,
            kkt_tolerance=KKT_TOLERANCE,
        )

        # ----------------------------------------------------
        # Baseline diagnostic reproduction
        # ----------------------------------------------------

        saved_diagnostic_row = (
            baseline_diagnostics_df[
                baseline_diagnostics_df[
                    "node_id"
                ]
                == node_id
            ]
            .iloc[0]
        )

        saved_iterations = int(
            saved_diagnostic_row[
                "iterations"
            ]
        )

        saved_num_nonzero = int(
            saved_diagnostic_row[
                "num_nonzero"
            ]
        )

        iterations_reproduced = bool(
            int(
                result[
                    "iterations"
                ]
            )
            == saved_iterations
        )

        nonzero_reproduced = bool(
            int(
                result[
                    "num_nonzero"
                ]
            )
            == saved_num_nonzero
        )

        # ----------------------------------------------------
        # Combined validation
        # ----------------------------------------------------

        solver_converged = bool(
            result[
                "converged"
            ]
        )

        reproduction_pass = bool(
            top_k_reproduced
            and iterations_reproduced
            and nonzero_reproduced
        )

        overall_node_pass = bool(
            solver_converged
            and reproduction_pass
            and kkt[
                "overall_kkt_pass"
            ]
        )

        # ----------------------------------------------------
        # Save diagnostic row
        # ----------------------------------------------------

        kkt_records.append(
            {
                "node_id": (
                    node_id
                ),
                "neighbourhood_size": (
                    neighbourhood_size
                ),
                "active_features": (
                    num_active_features
                ),
                "rho": (
                    RHO
                ),
                "solver_converged": (
                    solver_converged
                ),
                "iterations": int(
                    result[
                        "iterations"
                    ]
                ),
                "saved_iterations": (
                    saved_iterations
                ),
                "iterations_reproduced": (
                    iterations_reproduced
                ),
                "num_nonzero": int(
                    result[
                        "num_nonzero"
                    ]
                ),
                "saved_num_nonzero": (
                    saved_num_nonzero
                ),
                "nonzero_reproduced": (
                    nonzero_reproduced
                ),
                "top_k_reproduced": (
                    top_k_reproduced
                ),
                "num_active_beta": (
                    kkt[
                        "num_active"
                    ]
                ),
                "num_inactive_beta": (
                    kkt[
                        "num_inactive"
                    ]
                ),
                "max_active_stationarity_residual": (
                    kkt[
                        "max_active_stationarity_residual"
                    ]
                ),
                "mean_active_stationarity_residual": (
                    kkt[
                        "mean_active_stationarity_residual"
                    ]
                ),
                "min_inactive_slack": (
                    kkt[
                        "min_inactive_slack"
                    ]
                ),
                "max_inactive_violation": (
                    kkt[
                        "max_inactive_violation"
                    ]
                ),
                "min_beta": (
                    kkt[
                        "min_beta"
                    ]
                ),
                "max_complementarity_residual": (
                    kkt[
                        "max_complementarity_residual"
                    ]
                ),
                "active_kkt_pass": (
                    kkt[
                        "active_kkt_pass"
                    ]
                ),
                "inactive_kkt_pass": (
                    kkt[
                        "inactive_kkt_pass"
                    ]
                ),
                "primal_feasible": (
                    kkt[
                        "primal_feasible"
                    ]
                ),
                "kkt_pass": (
                    kkt[
                        "overall_kkt_pass"
                    ]
                ),
                "reproduction_pass": (
                    reproduction_pass
                ),
                "overall_node_pass": (
                    overall_node_pass
                ),
            }
        )

        # ----------------------------------------------------
        # Terminal report
        # ----------------------------------------------------

        status = (
            "PASS"
            if overall_node_pass
            else "REVIEW"
        )

        print(
            f"[{node_position:03d}/"
            f"{num_explainable:03d}] "
            f"node={node_id:4d} | "
            f"N={neighbourhood_size:3d} | "
            f"iters={result['iterations']:5d} | "
            f"nz={result['num_nonzero']:4d} | "
            f"active_res="
            f"{kkt['max_active_stationarity_residual']:.3e} | "
            f"inactive_violation="
            f"{kkt['max_inactive_violation']:.3e} | "
            f"KKT={str(kkt['overall_kkt_pass']):5s} | "
            f"{status}"
        )

    # ========================================================
    # Construct output DataFrame
    # ========================================================

    kkt_df = pd.DataFrame(
        kkt_records
    )

    if len(kkt_df) != num_explainable:

        raise RuntimeError(
            "Unexpected number of KKT "
            "diagnostic rows."
        )

    # ========================================================
    # Save raw KKT diagnostics BEFORE aggregate decision
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    kkt_df.to_csv(
        KKT_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # Aggregate validation
    # ========================================================

    num_solver_converged = int(
        kkt_df[
            "solver_converged"
        ].sum()
    )

    num_top_k_reproduced = int(
        kkt_df[
            "top_k_reproduced"
        ].sum()
    )

    num_iterations_reproduced = int(
        kkt_df[
            "iterations_reproduced"
        ].sum()
    )

    num_nonzero_reproduced = int(
        kkt_df[
            "nonzero_reproduced"
        ].sum()
    )

    num_reproduction_pass = int(
        kkt_df[
            "reproduction_pass"
        ].sum()
    )

    num_active_kkt_pass = int(
        kkt_df[
            "active_kkt_pass"
        ].sum()
    )

    num_inactive_kkt_pass = int(
        kkt_df[
            "inactive_kkt_pass"
        ].sum()
    )

    num_primal_feasible = int(
        kkt_df[
            "primal_feasible"
        ].sum()
    )

    num_kkt_pass = int(
        kkt_df[
            "kkt_pass"
        ].sum()
    )

    num_overall_pass = int(
        kkt_df[
            "overall_node_pass"
        ].sum()
    )

    maximum_active_residual = float(
        kkt_df[
            "max_active_stationarity_residual"
        ].max()
    )

    maximum_inactive_violation = float(
        kkt_df[
            "max_inactive_violation"
        ].max()
    )

    maximum_complementarity_residual = float(
        kkt_df[
            "max_complementarity_residual"
        ].max()
    )

    minimum_inactive_slack = float(
        kkt_df[
            "min_inactive_slack"
        ].min()
    )

    # ========================================================
    # Overall validation decision
    # ========================================================

    all_pass = bool(
        num_overall_pass
        == num_explainable
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Baseline HSIC-Lasso KKT "
        "Validation Summary ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nCohort:"
    )

    print(
        f"  selected nodes:             "
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  baseline explainable:       "
        f"{num_explainable}"
    )

    print(
        f"  output-degenerate excluded: "
        f"{num_output_degenerate}"
    )

    print(
        "\nReproduction checks:"
    )

    print(
        f"  solver converged:           "
        f"{num_solver_converged}/"
        f"{num_explainable}"
    )

    print(
        f"  top-K reproduced:           "
        f"{num_top_k_reproduced}/"
        f"{num_explainable}"
    )

    print(
        f"  iterations reproduced:      "
        f"{num_iterations_reproduced}/"
        f"{num_explainable}"
    )

    print(
        f"  nonzero counts reproduced:  "
        f"{num_nonzero_reproduced}/"
        f"{num_explainable}"
    )

    print(
        f"  full reproduction pass:     "
        f"{num_reproduction_pass}/"
        f"{num_explainable}"
    )

    print(
        "\nKKT checks:"
    )

    print(
        f"  KKT tolerance:              "
        f"{KKT_TOLERANCE:.1e}"
    )

    print(
        f"  active stationarity pass:   "
        f"{num_active_kkt_pass}/"
        f"{num_explainable}"
    )

    print(
        f"  inactive feasibility pass:  "
        f"{num_inactive_kkt_pass}/"
        f"{num_explainable}"
    )

    print(
        f"  primal feasibility pass:    "
        f"{num_primal_feasible}/"
        f"{num_explainable}"
    )

    print(
        f"  complete KKT pass:          "
        f"{num_kkt_pass}/"
        f"{num_explainable}"
    )

    print(
        "\nWorst observed residuals:"
    )

    print(
        "  max active stationarity "
        "residual: "
        f"{maximum_active_residual:.6e}"
    )

    print(
        "  max inactive violation:    "
        f"{maximum_inactive_violation:.6e}"
    )

    print(
        "  minimum inactive slack:    "
        f"{minimum_inactive_slack:.6e}"
    )

    print(
        "  max complementarity "
        "residual: "
        f"{maximum_complementarity_residual:.6e}"
    )

    print(
        "\nOverall node validation:"
    )

    print(
        f"  PASS:                       "
        f"{num_overall_pass}/"
        f"{num_explainable}"
    )

    # ========================================================
    # Review failures if present
    # ========================================================

    review_df = (
        kkt_df[
            ~kkt_df[
                "overall_node_pass"
            ]
        ]
    )

    if len(review_df) > 0:

        print(
            "\n=== Nodes Requiring Review ==="
        )

        review_columns = [
            "node_id",
            "neighbourhood_size",
            "iterations",
            "top_k_reproduced",
            "iterations_reproduced",
            "nonzero_reproduced",
            "max_active_stationarity_residual",
            "max_inactive_violation",
            "active_kkt_pass",
            "inactive_kkt_pass",
            "primal_feasible",
            "kkt_pass",
        ]

        print(
            review_df[
                review_columns
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nNodes requiring review: none"
        )

    # ========================================================
    # Explicitly report highest-iteration node
    # ========================================================

    highest_iteration_index = (
        kkt_df[
            "iterations"
        ].idxmax()
    )

    highest_iteration_row = (
        kkt_df.loc[
            highest_iteration_index
        ]
    )

    print(
        "\n=== Highest-Iteration Solution ==="
    )

    print(
        "Node ID: "
        f"{int(highest_iteration_row['node_id'])}"
    )

    print(
        "Iterations: "
        f"{int(highest_iteration_row['iterations'])}"
    )

    print(
        "Max active stationarity residual: "
        f"{highest_iteration_row['max_active_stationarity_residual']:.6e}"
    )

    print(
        "Max inactive violation: "
        f"{highest_iteration_row['max_inactive_violation']:.6e}"
    )

    print(
        "KKT pass: "
        f"{bool(highest_iteration_row['kkt_pass'])}"
    )

    # ========================================================
    # Saved artifact
    # ========================================================

    print(
        "\n=== Saved KKT Artifact ==="
    )

    print(
        KKT_OUTPUT_PATH
    )

    # ========================================================
    # Final milestone status
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Milestone 2D Status ==="
    )

    print(
        "=" * 78
    )

    if all_pass:

        print(
            "All reconstructed baseline "
            "HSIC-Lasso solutions PASSED "
            "the pre-specified reproduction "
            "and KKT validation checks."
        )

        print(
            "\nThe 97-node baseline explanation "
            "cohort is numerically validated."
        )

        print(
            "\nNext methodological step:"
        )

        print(
            "Define and freeze the controlled "
            "feature-perturbation operator before "
            "implementing perturbation experiments."
        )

    else:

        print(
            "Baseline HSIC-Lasso KKT validation "
            "requires REVIEW."
        )

        print(
            "\nRaw KKT diagnostics have been saved."
        )

        print(
            "Do NOT change rho automatically."
        )

        print(
            "Do NOT change optimizer tolerances "
            "automatically."
        )

        print(
            "Do NOT begin perturbation experiments "
            "until the failed KKT cases are examined."
        )


if __name__ == "__main__":
    main()
