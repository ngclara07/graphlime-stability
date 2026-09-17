# experiments/02b_rho_calibration.py
#
# Milestone 2B.4:
# Multi-node GraphLIME rho calibration
#
# Purpose:
#   1. Load the frozen GCN checkpoint.
#   2. Load the already frozen 100-node explanation sample.
#   3. Take the first 10 nodes as the predetermined
#      calibration subset.
#   4. Construct GraphLIME kernels independently for each node.
#   5. Prepare the HSIC-Lasso design matrix once per node.
#   6. Evaluate six rho values per node.
#   7. Save all 60 raw calibration runs.
#   8. Apply a pre-specified rho eligibility rule.
#
# IMPORTANT:
#   This experiment does NOT use perturbation stability to
#   choose rho.
#
# Run from repository root:
#
#   python -m experiments.02b_rho_calibration

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
# Fixed experiment configuration
# ============================================================

SEED = 42

NUM_CALIBRATION_NODES = 10

NUM_HOPS = 2

TOP_K = 10

MAX_ITER = 10000

OPTIMIZER_TOLERANCE = 1e-8

ZERO_TOLERANCE = 1e-8


# ============================================================
# Diagnostic rho grid
# ============================================================

RHO_VALUES = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
]


# ============================================================
# File locations
# ============================================================

CHECKPOINT_PATH = Path(
    "models/cora_gcn_seed42.pt"
)

SELECTED_NODES_PATH = Path(
    "results/baseline/selected_nodes.csv"
)

OUTPUT_DIR = Path(
    "results/baseline"
)

CALIBRATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "rho_calibration.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "rho_calibration_summary.csv"
)


# ============================================================
# Pre-specified rho selection rule
# ============================================================
#
# A rho value is ELIGIBLE only when:
#
#   1. every calibration run converges;
#
#   2. every calibration node has at least TOP_K nonzero
#      coefficients.
#
# Among eligible rho values, the largest rho is the
# recommended candidate.
#
# Rationale:
#   We prefer the strongest regularization that still
#   produces a complete top-K explanation for every
#   predetermined calibration node.
#
# IMPORTANT:
#   This rule is specified before looking at the
#   multi-node calibration results.
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
    # Validate required files
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Frozen checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    if not SELECTED_NODES_PATH.exists():

        raise FileNotFoundError(
            "Frozen selected-node file not found: "
            f"{SELECTED_NODES_PATH}"
        )

    # ========================================================
    # Load Cora
    # ========================================================

    dataset, data = load_cora()

    data = data.to(
        device
    )

    # ========================================================
    # Load frozen GCN checkpoint
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    print(
        "\n=== Loaded Frozen Baseline ==="
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Training seed: "
        f"{checkpoint['seed']}"
    )

    print(
        f"Best epoch: "
        f"{checkpoint['best_epoch']}"
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
    # Reconstruct frozen GCN
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
    # Generate frozen-model probabilities
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
    # Load frozen 100-node sample
    # ========================================================

    selected_df = pd.read_csv(
        SELECTED_NODES_PATH
    )

    if (
        len(selected_df)
        < NUM_CALIBRATION_NODES
    ):

        raise RuntimeError(
            "Frozen selected-node file does not "
            "contain enough calibration nodes."
        )

    # --------------------------------------------------------
    # Use first 10 frozen nodes.
    #
    # No resampling is performed.
    # --------------------------------------------------------

    calibration_df = (
        selected_df
        .iloc[
            :NUM_CALIBRATION_NODES
        ]
        .copy()
    )

    calibration_node_ids = (
        calibration_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    # ========================================================
    # Validate calibration nodes
    # ========================================================

    if (
        len(
            set(
                calibration_node_ids
            )
        )
        != NUM_CALIBRATION_NODES
    ):

        raise RuntimeError(
            "Calibration node IDs are not unique."
        )

    for node_id in (
        calibration_node_ids
    ):

        true_label = int(
            data.y[
                node_id
            ].item()
        )

        predicted_label = int(
            predictions[
                node_id
            ].item()
        )

        if (
            true_label
            != predicted_label
        ):

            raise RuntimeError(
                "Calibration node "
                f"{node_id} is not correctly "
                "classified by the frozen model."
            )

    print(
        "\n=== Calibration Sample ==="
    )

    print(
        "Number of calibration nodes: "
        f"{len(calibration_node_ids)}"
    )

    print(
        "Calibration node IDs:"
    )

    print(
        calibration_node_ids
    )

    print(
        "\nNumber of rho values: "
        f"{len(RHO_VALUES)}"
    )

    print(
        "rho values:"
    )

    print(
        RHO_VALUES
    )

    expected_runs = (
        NUM_CALIBRATION_NODES
        * len(
            RHO_VALUES
        )
    )

    print(
        "\nExpected HSIC-Lasso runs: "
        f"{expected_runs}"
    )

    # ========================================================
    # Raw calibration records
    # ========================================================

    calibration_records = []

    # ========================================================
    # Process each predetermined node
    # ========================================================

    for node_position, node_id in enumerate(
        calibration_node_ids,
        start=1,
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Calibration node "
            f"{node_position}/"
            f"{NUM_CALIBRATION_NODES}: "
            f"{node_id}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Extract local neighbourhood
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

        # ----------------------------------------------------
        # Validate target mapping
        # ----------------------------------------------------

        mapped_original_node = int(
            subset[
                target_local_index
            ].item()
        )

        if (
            mapped_original_node
            != node_id
        ):

            raise RuntimeError(
                "Target-node mapping failed for "
                f"node {node_id}."
            )

        # ----------------------------------------------------
        # Local GraphLIME data
        # ----------------------------------------------------

        local_x = data.x[
            subset
        ]

        local_y = probabilities[
            subset
        ]

        if (
            local_x.shape[0]
            != local_y.shape[0]
        ):

            raise RuntimeError(
                "Local X/Y sample dimensions "
                "do not agree."
            )

        # ----------------------------------------------------
        # Validate probabilities
        # ----------------------------------------------------

        probability_sums = (
            local_y.sum(
                dim=-1
            )
        )

        if not torch.allclose(
            probability_sums,
            torch.ones_like(
                probability_sums
            ),
            atol=1e-5,
        ):

            raise RuntimeError(
                "Invalid local GNN probability "
                f"vectors for node {node_id}."
            )

        # ----------------------------------------------------
        # Build GraphLIME feature kernels
        # ----------------------------------------------------

        (
            feature_kernels,
            feature_bandwidths,
            degenerate_features,
        ) = build_feature_kernels(
            local_x
        )

        active_features = len(
            feature_kernels
        )

        degenerate_count = len(
            degenerate_features
        )

        if (
            active_features
            + degenerate_count
            != dataset.num_features
        ):

            raise RuntimeError(
                "Active/degenerate feature accounting "
                f"failed for node {node_id}."
            )

        if (
            active_features
            == 0
        ):

            raise RuntimeError(
                "No active GraphLIME features for "
                f"node {node_id}."
            )

        # ----------------------------------------------------
        # Build output kernel
        # ----------------------------------------------------

        (
            output_kernel,
            output_bandwidth,
            output_centered_norm,
        ) = build_output_kernel(
            local_y
        )

        # ----------------------------------------------------
        # Prepare HSIC-Lasso ONCE for this node
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

        print(
            f"Neighbourhood size: "
            f"{neighbourhood_size}"
        )

        print(
            f"Subgraph edges: "
            f"{sub_edge_index.shape[1]}"
        )

        print(
            f"Active features: "
            f"{active_features}"
        )

        print(
            f"Degenerate features: "
            f"{degenerate_count}"
        )

        print(
            "Output bandwidth: "
            f"{output_bandwidth:.6f}"
        )

        print(
            "HSIC design matrix: "
            f"{tuple(problem['design_matrix'].shape)}"
        )

        print(
            "Lipschitz constant: "
            f"{problem['lipschitz_constant']:.6f}"
        )

        print(
            "Step size: "
            f"{problem['step_size']:.8f}"
        )

        # ====================================================
        # rho sweep for this node
        # ====================================================

        for rho in RHO_VALUES:

            result = (
                solve_prepared_hsic_lasso(
                    problem=problem,
                    rho=rho,
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

            # ------------------------------------------------
            # Numerical checks
            # ------------------------------------------------

            if not torch.isfinite(
                beta
            ).all():

                raise RuntimeError(
                    "Non-finite beta values for "
                    f"node={node_id}, rho={rho}."
                )

            if torch.any(
                beta
                < -ZERO_TOLERANCE
            ):

                raise RuntimeError(
                    "Negative beta value for "
                    f"node={node_id}, rho={rho}."
                )

            if (
                result[
                    "final_objective"
                ]
                >
                result[
                    "initial_objective"
                ]
                + 1e-7
            ):

                raise RuntimeError(
                    "Objective failed to decrease for "
                    f"node={node_id}, rho={rho}."
                )

            # ------------------------------------------------
            # Initial objective should be approximately 0.5
            # because ||L_bar||_F = 1 and beta starts at zero.
            # ------------------------------------------------

            if abs(
                result[
                    "initial_objective"
                ]
                - 0.5
            ) > 1e-5:

                raise RuntimeError(
                    "Unexpected initial objective for "
                    f"node={node_id}, rho={rho}: "
                    f"{result['initial_objective']}"
                )

            # ------------------------------------------------
            # Extract top-K explanation
            # ------------------------------------------------

            top_features = (
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

            top_feature_ids = [
                int(
                    feature_index
                )
                for (
                    feature_index,
                    coefficient,
                ) in top_features
            ]

            top_feature_betas = [
                float(
                    coefficient
                )
                for (
                    feature_index,
                    coefficient,
                ) in top_features
            ]

            # ------------------------------------------------
            # Store top features in a compact reproducible
            # representation.
            # ------------------------------------------------

            top_features_string = (
                ";".join(
                    str(
                        feature_index
                    )
                    for feature_index
                    in top_feature_ids
                )
            )

            top_betas_string = (
                ";".join(
                    f"{coefficient:.10f}"
                    for coefficient
                    in top_feature_betas
                )
            )

            has_full_top_k = (
                result[
                    "num_nonzero"
                ]
                >= TOP_K
            )

            print(
                f"rho={rho:>6.3f} | "
                f"conv="
                f"{str(result['converged']):>5} | "
                f"iters="
                f"{result['iterations']:>4} | "
                f"nonzero="
                f"{result['num_nonzero']:>3} | "
                f"full_top_{TOP_K}="
                f"{has_full_top_k} | "
                f"obj="
                f"{result['final_objective']:.6f}"
            )

            calibration_records.append(
                {
                    "node_id": (
                        node_id
                    ),

                    "rho": (
                        rho
                    ),

                    "neighbourhood_size": (
                        neighbourhood_size
                    ),

                    "subgraph_edges": int(
                        sub_edge_index.shape[
                            1
                        ]
                    ),

                    "active_features": (
                        active_features
                    ),

                    "degenerate_features": (
                        degenerate_count
                    ),

                    "output_bandwidth": (
                        output_bandwidth
                    ),

                    "output_centered_norm": (
                        output_centered_norm
                    ),

                    "lipschitz_constant": (
                        problem[
                            "lipschitz_constant"
                        ]
                    ),

                    "step_size": (
                        problem[
                            "step_size"
                        ]
                    ),

                    "converged": (
                        result[
                            "converged"
                        ]
                    ),

                    "iterations": (
                        result[
                            "iterations"
                        ]
                    ),

                    "initial_objective": (
                        result[
                            "initial_objective"
                        ]
                    ),

                    "final_objective": (
                        result[
                            "final_objective"
                        ]
                    ),

                    "objective_reduction": (
                        result[
                            "initial_objective"
                        ]
                        - result[
                            "final_objective"
                        ]
                    ),

                    "relative_change": (
                        result[
                            "relative_change"
                        ]
                    ),

                    "num_nonzero": (
                        result[
                            "num_nonzero"
                        ]
                    ),

                    "has_full_top_k": (
                        has_full_top_k
                    ),

                    "max_beta": float(
                        beta.max().item()
                    ),

                    "min_beta": float(
                        beta.min().item()
                    ),

                    "num_top_features": (
                        len(
                            top_features
                        )
                    ),

                    "top_10_features": (
                        top_features_string
                    ),

                    "top_10_betas": (
                        top_betas_string
                    ),
                }
            )

    # ========================================================
    # Construct raw calibration DataFrame
    # ========================================================

    calibration_results_df = (
        pd.DataFrame(
            calibration_records
        )
    )

    # ========================================================
    # Validate expected run count
    # ========================================================

    if (
        len(
            calibration_results_df
        )
        != expected_runs
    ):

        raise RuntimeError(
            "Unexpected number of calibration "
            "runs. Expected "
            f"{expected_runs}, obtained "
            f"{len(calibration_results_df)}."
        )

    # ========================================================
    # Save raw results
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_results_df.to_csv(
        CALIBRATION_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # Build rho-level summary
    # ========================================================

    summary_records = []

    for rho in RHO_VALUES:

        rho_df = (
            calibration_results_df[
                calibration_results_df[
                    "rho"
                ]
                == rho
            ]
        )

        num_runs = len(
            rho_df
        )

        num_converged = int(
            rho_df[
                "converged"
            ].sum()
        )

        num_full_top_k = int(
            rho_df[
                "has_full_top_k"
            ].sum()
        )

        min_nonzero = int(
            rho_df[
                "num_nonzero"
            ].min()
        )

        max_nonzero = int(
            rho_df[
                "num_nonzero"
            ].max()
        )

        mean_nonzero = float(
            rho_df[
                "num_nonzero"
            ].mean()
        )

        median_nonzero = float(
            rho_df[
                "num_nonzero"
            ].median()
        )

        mean_iterations = float(
            rho_df[
                "iterations"
            ].mean()
        )

        max_iterations = int(
            rho_df[
                "iterations"
            ].max()
        )

        all_converged = (
            num_converged
            == NUM_CALIBRATION_NODES
        )

        all_have_top_k = (
            num_full_top_k
            == NUM_CALIBRATION_NODES
        )

        eligible = (
            all_converged
            and all_have_top_k
        )

        summary_records.append(
            {
                "rho": (
                    rho
                ),

                "num_runs": (
                    num_runs
                ),

                "num_converged": (
                    num_converged
                ),

                "num_full_top_k": (
                    num_full_top_k
                ),

                "min_nonzero": (
                    min_nonzero
                ),

                "mean_nonzero": (
                    mean_nonzero
                ),

                "median_nonzero": (
                    median_nonzero
                ),

                "max_nonzero": (
                    max_nonzero
                ),

                "mean_iterations": (
                    mean_iterations
                ),

                "max_iterations": (
                    max_iterations
                ),

                "all_converged": (
                    all_converged
                ),

                "all_have_top_k": (
                    all_have_top_k
                ),

                "eligible": (
                    eligible
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_records
    )

    # ========================================================
    # Determine recommended rho using ONLY the
    # pre-specified calibration rule.
    # ========================================================

    eligible_df = (
        summary_df[
            summary_df[
                "eligible"
            ]
        ]
    )

    if len(
        eligible_df
    ) > 0:

        recommended_rho = float(
            eligible_df[
                "rho"
            ].max()
        )

    else:

        recommended_rho = None

    # ========================================================
    # Mark recommended candidate in summary
    # ========================================================

    if recommended_rho is None:

        summary_df[
            "recommended_candidate"
        ] = False

    else:

        summary_df[
            "recommended_candidate"
        ] = (
            summary_df[
                "rho"
            ]
            == recommended_rho
        )

    # ========================================================
    # Save summary
    # ========================================================

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # Terminal summary
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "=== Multi-node rho Calibration Summary ==="
    )

    print(
        "=" * 60
    )

    print(
        f"{'rho':>8} "
        f"{'conv':>8} "
        f"{'top10':>8} "
        f"{'min_nz':>8} "
        f"{'mean_nz':>10} "
        f"{'max_nz':>8} "
        f"{'eligible':>10}"
    )

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['rho']:>8.3f} "
            f"{int(row['num_converged']):>3}/"
            f"{NUM_CALIBRATION_NODES:<4} "
            f"{int(row['num_full_top_k']):>3}/"
            f"{NUM_CALIBRATION_NODES:<4} "
            f"{int(row['min_nonzero']):>8} "
            f"{row['mean_nonzero']:>10.2f} "
            f"{int(row['max_nonzero']):>8} "
            f"{str(bool(row['eligible'])):>10}"
        )

    # ========================================================
    # Pre-specified rule result
    # ========================================================

    print(
        "\n=== Pre-specified Selection Rule ==="
    )

    print(
        "Eligibility requires:"
    )

    print(
        "  1. convergence on all 10 "
        "calibration nodes;"
    )

    print(
        f"  2. at least {TOP_K} nonzero "
        "coefficients on all 10 nodes."
    )

    print(
        "\nAmong eligible rho values, "
        "choose the largest rho."
    )

    if recommended_rho is None:

        print(
            "\nResult: No rho value satisfies "
            "the pre-specified rule."
        )

        print(
            "Do NOT freeze rho yet."
        )

    else:

        print(
            "\nRecommended rho candidate: "
            f"{recommended_rho:.6f}"
        )

        print(
            "\nIMPORTANT:"
        )

        print(
            "This is the result of the "
            "pre-specified calibration rule."
        )

        print(
            "Do not yet generate/freeze the "
            "100-node explanations until this "
            "calibration output has been reviewed."
        )

    # ========================================================
    # Output locations
    # ========================================================

    print(
        "\n=== Saved Calibration Artifacts ==="
    )

    print(
        "Raw 60-run results:"
    )

    print(
        CALIBRATION_OUTPUT_PATH
    )

    print(
        "\nrho-level summary:"
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        "\n=== Milestone Status ==="
    )

    print(
        "Multi-node rho calibration executed."
    )

    print(
        "rho is NOT yet frozen."
    )

    print(
        "100-node baseline explanations have "
        "NOT yet been generated."
    )


if __name__ == "__main__":
    main()
