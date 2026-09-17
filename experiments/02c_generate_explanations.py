# experiments/02c_generate_explanations.py
#
# Milestone 2C:
# Generate frozen baseline GraphLIME explanations.
#
# Purpose:
#   1. Load the frozen Cora GCN checkpoint.
#   2. Load the frozen set of 100 correctly classified test nodes.
#   3. Use the frozen GraphLIME configuration:
#
#          NUM_HOPS = 2
#          TOP_K    = 10
#          RHO      = 0.03
#
#   4. Construct one GraphLIME explanation where the local
#      output kernel is well-defined.
#   5. Explicitly record output-degenerate neighbourhoods.
#   6. Save available explanations in long format.
#   7. Save diagnostics for ALL 100 selected nodes.
#   8. Validate the complete experiment before deciding
#      whether the baseline explanation set can be frozen.
#
# IMPORTANT:
#   rho = 0.03 was selected in the preceding predetermined
#   10-node calibration experiment.
#
#   This script MUST NOT retune rho based on the remaining nodes.
#
#   If a node has locally degenerate GNN outputs, no arbitrary
#   bandwidth is introduced. The condition is recorded and the
#   node receives no GraphLIME explanation.
#
# Run from repository root:
#
#   python -m experiments.02c_generate_explanations


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

# ------------------------------------------------------------
# FROZEN rho
# ------------------------------------------------------------
#
# Selected by the predetermined 10-node calibration rule:
#
#   1. all calibration runs must converge;
#   2. all calibration nodes must retain >= TOP_K
#      nonzero coefficients;
#   3. among eligible candidates, select the largest rho.
#
# Calibration selected:
#
#   rho = 0.03
#
# Do NOT modify this value based on results from the
# remaining nodes.
# ------------------------------------------------------------

RHO = 0.03


# ============================================================
# OPTIMIZATION CONFIGURATION
# ============================================================

MAX_ITER = 10000

OPTIMIZER_TOLERANCE = 1e-8

ZERO_TOLERANCE = 1e-8


# ============================================================
# FILE LOCATIONS
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

EXPLANATIONS_OUTPUT_PATH = (
    OUTPUT_DIR
    / "explanations.csv"
)

DIAGNOSTICS_OUTPUT_PATH = (
    OUTPUT_DIR
    / "explanation_diagnostics.csv"
)


# ============================================================
# SAFE OUTPUT-KERNEL CONSTRUCTION
# ============================================================


def try_build_output_kernel(
    local_y: torch.Tensor,
):
    """
    Attempt to construct the normalized GraphLIME output kernel.

    A local neighbourhood is output-degenerate when the local GNN
    probability vectors contain no positive pairwise distance.

    In that case, the Gaussian output kernel contains no useful
    local output variation after centering and cannot be
    Frobenius-normalized for GraphLIME.

    This condition is treated as an empirical diagnostic outcome.
    No arbitrary bandwidth is introduced.

    Only the known output-degeneracy RuntimeError is intercepted.
    Any other RuntimeError is re-raised.

    Parameters
    ----------
    local_y : torch.Tensor
        Local GNN probability matrix with shape
        [neighbourhood_size, num_classes].

    Returns
    -------
    dict
        Dictionary containing:

        success
            Whether output-kernel construction succeeded.

        output_kernel
            Normalized output kernel, or None.

        output_bandwidth
            Gaussian bandwidth, or None.

        output_centered_norm
            Frobenius norm before normalization, or None.

        reason
            Failure code, or None.
    """

    try:

        (
            output_kernel,
            output_bandwidth,
            output_centered_norm,
        ) = build_output_kernel(
            local_y
        )

        return {
            "success": True,
            "output_kernel": output_kernel,
            "output_bandwidth": output_bandwidth,
            "output_centered_norm": output_centered_norm,
            "reason": None,
        }

    except RuntimeError as error:

        error_message = str(
            error
        )

        expected_message = (
            "Local GNN outputs are degenerate: "
            "no positive pairwise distance exists."
        )

        if expected_message not in error_message:
            # Do not conceal unrelated implementation or
            # numerical errors.
            raise

        return {
            "success": False,
            "output_kernel": None,
            "output_bandwidth": None,
            "output_centered_norm": None,
            "reason": "degenerate_local_gnn_outputs",
        }


# ============================================================
# MAIN EXPERIMENT
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
    # Validate required frozen artifacts
    # ========================================================

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            "Frozen baseline checkpoint not found: "
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
    # Validate frozen experiment configuration
    # ========================================================

    if SEED != checkpoint["seed"]:

        raise RuntimeError(
            "Experiment seed does not match "
            "the frozen checkpoint seed."
        )

    print(
        "\n=== Frozen GraphLIME Configuration ==="
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
    # Generate frozen-model predictions and probabilities
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

        confidences = (
            probabilities.max(
                dim=-1
            ).values
        )

    # ========================================================
    # Validate reconstructed test accuracy
    # ========================================================

    test_predictions = predictions[
        data.test_mask
    ]

    test_labels = data.y[
        data.test_mask
    ]

    reconstructed_test_accuracy = float(
        (
            test_predictions
            == test_labels
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
            "not match frozen checkpoint."
        )

    print(
        "\nReconstructed test accuracy: "
        f"{reconstructed_test_accuracy:.4f}"
    )

    # ========================================================
    # Load frozen 100-node sample
    # ========================================================

    selected_df = pd.read_csv(
        SELECTED_NODES_PATH
    )

    # ========================================================
    # Validate selected-node artifact
    # ========================================================

    required_columns = {
        "node_id",
        "true_label",
        "predicted_label",
        "confidence",
    }

    missing_columns = (
        required_columns
        - set(
            selected_df.columns
        )
    )

    if missing_columns:

        raise RuntimeError(
            "selected_nodes.csv is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        len(selected_df)
        != NUM_SELECTED_NODES
    ):

        raise RuntimeError(
            "Expected exactly "
            f"{NUM_SELECTED_NODES} frozen nodes, "
            f"found {len(selected_df)}."
        )

    if not selected_df[
        "node_id"
    ].is_unique:

        raise RuntimeError(
            "Frozen selected-node file contains "
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
    # Revalidate every frozen node against the model
    # ========================================================

    for _, row in selected_df.iterrows():

        node_id = int(
            row["node_id"]
        )

        saved_true_label = int(
            row["true_label"]
        )

        saved_predicted_label = int(
            row["predicted_label"]
        )

        actual_true_label = int(
            data.y[
                node_id
            ].item()
        )

        actual_predicted_label = int(
            predictions[
                node_id
            ].item()
        )

        if (
            actual_true_label
            != saved_true_label
        ):

            raise RuntimeError(
                "True-label mismatch for "
                f"node {node_id}."
            )

        if (
            actual_predicted_label
            != saved_predicted_label
        ):

            raise RuntimeError(
                "Prediction mismatch for "
                f"node {node_id}."
            )

        if (
            actual_true_label
            != actual_predicted_label
        ):

            raise RuntimeError(
                "Frozen explanation node "
                f"{node_id} is no longer "
                "correctly classified."
            )

    print(
        "\n=== Frozen Explanation Sample ==="
    )

    print(
        "Number of nodes: "
        f"{len(selected_node_ids)}"
    )

    print(
        "Unique nodes: "
        f"{len(set(selected_node_ids))}"
    )

    print(
        "All frozen nodes correctly classified: True"
    )

    print(
        "\nFirst 10 node IDs:"
    )

    print(
        selected_node_ids[:10]
    )

    # ========================================================
    # Result containers
    # ========================================================

    explanation_records = []

    diagnostic_records = []

    # ========================================================
    # Generate one baseline explanation per node
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== Generating Frozen Baseline "
        "GraphLIME Explanations ==="
    )

    print(
        "=" * 70
    )

    for node_position, node_id in enumerate(
        selected_node_ids,
        start=1,
    ):

        # ----------------------------------------------------
        # Original prediction information
        # ----------------------------------------------------

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

        confidence = float(
            confidences[
                node_id
            ].item()
        )

        # ----------------------------------------------------
        # Extract frozen 2-hop neighbourhood
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

        subgraph_edge_count = int(
            sub_edge_index.shape[
                1
            ]
        )

        target_local_index = int(
            mapping.item()
        )

        # ----------------------------------------------------
        # Validate target-node mapping
        # ----------------------------------------------------

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
        # Local GraphLIME input
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
                "Local X/Y size mismatch "
                f"for node {node_id}."
            )

        # ----------------------------------------------------
        # Validate probability vectors
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
                "Invalid probability vectors "
                f"for node {node_id}."
            )

        # ----------------------------------------------------
        # Build local GraphLIME feature kernels
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

        num_degenerate_features = len(
            degenerate_features
        )

        if (
            num_active_features
            + num_degenerate_features
            != dataset.num_features
        ):

            raise RuntimeError(
                "Feature accounting failed "
                f"for node {node_id}."
            )

        if num_active_features == 0:

            raise RuntimeError(
                "No active local features "
                f"for node {node_id}."
            )

        # ----------------------------------------------------
        # Safely build local GNN-output kernel
        # ----------------------------------------------------

        output_kernel_result = (
            try_build_output_kernel(
                local_y
            )
        )

        # ----------------------------------------------------
        # Handle output-degenerate neighbourhood
        # ----------------------------------------------------
        #
        # No GraphLIME explanation is fabricated in this case.
        # The node remains part of the 100-node diagnostic set,
        # but contributes zero rows to explanations.csv.
        # ----------------------------------------------------

        if not output_kernel_result[
            "success"
        ]:

            diagnostic_records.append(
                {
                    "node_id": node_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "rho": RHO,
                    "num_hops": NUM_HOPS,
                    "top_k": TOP_K,
                    "neighbourhood_size": (
                        neighbourhood_size
                    ),
                    "subgraph_edges": (
                        subgraph_edge_count
                    ),
                    "active_features": (
                        num_active_features
                    ),
                    "degenerate_features": (
                        num_degenerate_features
                    ),
                    "output_bandwidth": None,
                    "output_centered_norm": None,
                    "lipschitz_constant": None,
                    "step_size": None,
                    "converged": False,
                    "iterations": 0,
                    "initial_objective": None,
                    "final_objective": None,
                    "objective_reduction": None,
                    "relative_change": None,
                    "num_nonzero": 0,
                    "num_top_features": 0,
                    "has_full_top_k": False,
                    "has_at_least_top_k_nonzero": False,

                    # These are not optimization failures.
                    # There was no beta vector to validate.
                    # False is used so aggregate "all valid"
                    # checks cannot incorrectly pass.
                    "beta_finite": False,
                    "beta_nonnegative": False,
                    "objective_decreased": False,
                    "initial_objective_valid": False,

                    "max_beta": None,
                    "min_beta": None,

                    # Explicit explanation-availability fields.
                    "output_kernel_valid": False,
                    "explanation_available": False,
                    "failure_reason": (
                        output_kernel_result[
                            "reason"
                        ]
                    ),
                }
            )

            print(
                f"[{node_position:03d}/"
                f"{NUM_SELECTED_NODES:03d}] "
                f"node={node_id:4d} | "
                f"N={neighbourhood_size:3d} | "
                f"active={num_active_features:4d} | "
                "OUTPUT-DEGENERATE | "
                "REVIEW_OUTPUT_DEGENERATE"
            )

            # Continue to the next frozen node.
            continue

        # ----------------------------------------------------
        # Successful output-kernel construction
        # ----------------------------------------------------

        output_kernel = (
            output_kernel_result[
                "output_kernel"
            ]
        )

        output_bandwidth = (
            output_kernel_result[
                "output_bandwidth"
            ]
        )

        output_centered_norm = (
            output_kernel_result[
                "output_centered_norm"
            ]
        )

        # ----------------------------------------------------
        # Validate output kernel
        # ----------------------------------------------------

        expected_kernel_shape = (
            neighbourhood_size,
            neighbourhood_size,
        )

        if (
            output_kernel.shape
            != expected_kernel_shape
        ):

            raise RuntimeError(
                "Unexpected output-kernel shape "
                f"for node {node_id}."
            )

        if not torch.isfinite(
            output_kernel
        ).all():

            raise RuntimeError(
                "Non-finite output kernel "
                f"for node {node_id}."
            )

        output_kernel_norm = float(
            torch.linalg.matrix_norm(
                output_kernel,
                ord="fro",
            ).item()
        )

        if abs(
            output_kernel_norm
            - 1.0
        ) > 1e-5:

            raise RuntimeError(
                "Output kernel is not "
                "Frobenius-normalized for "
                f"node {node_id}."
            )

        # ----------------------------------------------------
        # Prepare HSIC-Lasso problem
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
        # Solve at FROZEN rho
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

        # ====================================================
        # Numerical validation
        # ====================================================

        beta_finite = bool(
            torch.isfinite(
                beta
            ).all().item()
        )

        beta_nonnegative = bool(
            torch.all(
                beta
                >= -ZERO_TOLERANCE
            ).item()
        )

        objective_decreased = bool(
            result[
                "final_objective"
            ]
            <=
            result[
                "initial_objective"
            ]
            + 1e-7
        )

        initial_objective_valid = bool(
            abs(
                result[
                    "initial_objective"
                ]
                - 0.5
            )
            <= 1e-5
        )

        if not beta_finite:

            raise RuntimeError(
                "Non-finite HSIC-Lasso "
                f"coefficients for node {node_id}."
            )

        if not beta_nonnegative:

            raise RuntimeError(
                "Negative HSIC-Lasso "
                f"coefficient for node {node_id}."
            )

        if not objective_decreased:

            raise RuntimeError(
                "HSIC-Lasso objective failed "
                f"to decrease for node {node_id}."
            )

        if not initial_objective_valid:

            raise RuntimeError(
                "Unexpected initial HSIC-Lasso "
                f"objective for node {node_id}: "
                f"{result['initial_objective']}"
            )

        # ====================================================
        # Extract top-K explanation
        # ====================================================

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

        num_top_features = len(
            top_features
        )

        has_full_top_k = bool(
            num_top_features
            == TOP_K
        )

        has_at_least_top_k_nonzero = bool(
            result[
                "num_nonzero"
            ]
            >= TOP_K
        )

        # ----------------------------------------------------
        # Validate ranking
        # ----------------------------------------------------

        if num_top_features > TOP_K:

            raise RuntimeError(
                "top_k_graphlime_features returned "
                f"more than {TOP_K} features for "
                f"node {node_id}."
            )

        # ----------------------------------------------------
        # Ensure no duplicate feature IDs
        # ----------------------------------------------------

        top_feature_ids = [
            int(
                feature_id
            )
            for (
                feature_id,
                coefficient,
            ) in top_features
        ]

        if (
            len(top_feature_ids)
            != len(
                set(top_feature_ids)
            )
        ):

            raise RuntimeError(
                "Duplicate top feature IDs "
                f"for node {node_id}."
            )

        # ----------------------------------------------------
        # Validate descending coefficient ordering
        # ----------------------------------------------------

        top_coefficients = [
            float(
                coefficient
            )
            for (
                feature_id,
                coefficient,
            ) in top_features
        ]

        for index in range(
            len(top_coefficients)
            - 1
        ):

            if (
                top_coefficients[
                    index
                ]
                <
                top_coefficients[
                    index + 1
                ]
                - 1e-12
            ):

                raise RuntimeError(
                    "Top features are not ordered "
                    "by descending beta for "
                    f"node {node_id}."
                )

        # ====================================================
        # Store long-format explanation
        # ====================================================

        for rank, (
            feature_id,
            coefficient,
        ) in enumerate(
            top_features,
            start=1,
        ):

            explanation_records.append(
                {
                    "node_id": node_id,
                    "rank": rank,
                    "feature_id": int(
                        feature_id
                    ),
                    "beta": float(
                        coefficient
                    ),
                    "rho": RHO,
                    "num_hops": NUM_HOPS,
                    "top_k": TOP_K,
                }
            )

        # ====================================================
        # Store node-level diagnostics
        # ====================================================

        diagnostic_records.append(
            {
                "node_id": node_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "rho": RHO,
                "num_hops": NUM_HOPS,
                "top_k": TOP_K,
                "neighbourhood_size": (
                    neighbourhood_size
                ),
                "subgraph_edges": (
                    subgraph_edge_count
                ),
                "active_features": (
                    num_active_features
                ),
                "degenerate_features": (
                    num_degenerate_features
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
                    -
                    result[
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
                "num_top_features": (
                    num_top_features
                ),
                "has_full_top_k": (
                    has_full_top_k
                ),
                "has_at_least_top_k_nonzero": (
                    has_at_least_top_k_nonzero
                ),
                "beta_finite": (
                    beta_finite
                ),
                "beta_nonnegative": (
                    beta_nonnegative
                ),
                "objective_decreased": (
                    objective_decreased
                ),
                "initial_objective_valid": (
                    initial_objective_valid
                ),
                "max_beta": float(
                    beta.max().item()
                ),
                "min_beta": float(
                    beta.min().item()
                ),

                # Explicit explanation-availability fields.
                "output_kernel_valid": True,
                "explanation_available": True,
                "failure_reason": None,
            }
        )

        # ====================================================
        # Per-node terminal report
        # ====================================================

        status = (
            "PASS"
            if (
                result[
                    "converged"
                ]
                and has_full_top_k
                and beta_finite
                and beta_nonnegative
                and objective_decreased
                and initial_objective_valid
            )
            else "REVIEW"
        )

        print(
            f"[{node_position:03d}/"
            f"{NUM_SELECTED_NODES:03d}] "
            f"node={node_id:4d} | "
            f"N={neighbourhood_size:3d} | "
            f"active={num_active_features:4d} | "
            f"iters={result['iterations']:5d} | "
            f"nz={result['num_nonzero']:4d} | "
            f"top={num_top_features:2d} | "
            f"conv={str(result['converged']):5s} | "
            f"{status}"
        )

    # ========================================================
    # Construct result DataFrames
    # ========================================================

    explanations_df = pd.DataFrame(
        explanation_records
    )

    diagnostics_df = pd.DataFrame(
        diagnostic_records
    )

    # ========================================================
    # Validate node-level diagnostic count
    # ========================================================

    if (
        len(diagnostics_df)
        != NUM_SELECTED_NODES
    ):

        raise RuntimeError(
            "Expected exactly "
            f"{NUM_SELECTED_NODES} diagnostic rows, "
            f"obtained {len(diagnostics_df)}."
        )

    # ========================================================
    # Validate diagnostic node IDs and ordering
    # ========================================================

    diagnostic_node_ids = (
        diagnostics_df[
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
            "Diagnostic node ordering does not "
            "match the frozen selected-node file."
        )

    # ========================================================
    # Validate explanation rows correspond only to
    # explanation-available nodes
    # ========================================================

    explanation_available_node_ids = set(
        diagnostics_df.loc[
            diagnostics_df[
                "explanation_available"
            ],
            "node_id",
        ]
        .astype(int)
        .tolist()
    )

    explanation_row_node_ids = set(
        explanations_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    if not explanation_row_node_ids.issubset(
        explanation_available_node_ids
    ):

        raise RuntimeError(
            "explanations.csv contains a node "
            "marked as explanation unavailable."
        )

    # ========================================================
    # Save artifacts BEFORE final quality assessment
    # ========================================================
    #
    # This is intentional.
    #
    # Degenerate or otherwise problematic nodes remain in
    # explanation_diagnostics.csv so the experiment preserves
    # the full 100-node diagnostic record.
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanations_df.to_csv(
        EXPLANATIONS_OUTPUT_PATH,
        index=False,
    )

    diagnostics_df.to_csv(
        DIAGNOSTICS_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # Aggregate validation statistics
    # ========================================================

    num_output_kernel_valid = int(
        diagnostics_df[
            "output_kernel_valid"
        ].sum()
    )

    num_explanations_available = int(
        diagnostics_df[
            "explanation_available"
        ].sum()
    )

    num_output_degenerate = int(
        (
            diagnostics_df[
                "failure_reason"
            ]
            == "degenerate_local_gnn_outputs"
        ).sum()
    )

    num_converged = int(
        diagnostics_df[
            "converged"
        ].sum()
    )

    num_full_top_k = int(
        diagnostics_df[
            "has_full_top_k"
        ].sum()
    )

    num_at_least_top_k_nonzero = int(
        diagnostics_df[
            "has_at_least_top_k_nonzero"
        ].sum()
    )

    num_beta_finite = int(
        diagnostics_df[
            "beta_finite"
        ].sum()
    )

    num_beta_nonnegative = int(
        diagnostics_df[
            "beta_nonnegative"
        ].sum()
    )

    num_objective_decreased = int(
        diagnostics_df[
            "objective_decreased"
        ].sum()
    )

    num_initial_objective_valid = int(
        diagnostics_df[
            "initial_objective_valid"
        ].sum()
    )

    # --------------------------------------------------------
    # Statistics over nodes where optimization was actually
    # performed.
    # --------------------------------------------------------

    optimized_df = (
        diagnostics_df[
            diagnostics_df[
                "output_kernel_valid"
            ]
        ]
    )

    if len(optimized_df) > 0:

        min_nonzero = int(
            optimized_df[
                "num_nonzero"
            ].min()
        )

        max_nonzero = int(
            optimized_df[
                "num_nonzero"
            ].max()
        )

        mean_nonzero = float(
            optimized_df[
                "num_nonzero"
            ].mean()
        )

        median_nonzero = float(
            optimized_df[
                "num_nonzero"
            ].median()
        )

        max_iterations_observed = int(
            optimized_df[
                "iterations"
            ].max()
        )

        mean_iterations = float(
            optimized_df[
                "iterations"
            ].mean()
        )

    else:

        min_nonzero = 0
        max_nonzero = 0
        mean_nonzero = 0.0
        median_nonzero = 0.0
        max_iterations_observed = 0
        mean_iterations = 0.0

    min_neighbourhood_size = int(
        diagnostics_df[
            "neighbourhood_size"
        ].min()
    )

    max_neighbourhood_size = int(
        diagnostics_df[
            "neighbourhood_size"
        ].max()
    )

    mean_neighbourhood_size = float(
        diagnostics_df[
            "neighbourhood_size"
        ].mean()
    )

    # ========================================================
    # Explanation-row validation
    # ========================================================

    expected_explanation_rows_if_complete = (
        NUM_SELECTED_NODES
        * TOP_K
    )

    expected_explanation_rows_available = (
        num_explanations_available
        * TOP_K
    )

    actual_explanation_rows = len(
        explanations_df
    )

    available_row_count_correct = (
        actual_explanation_rows
        == expected_explanation_rows_available
    )

    # ========================================================
    # Additional per-node explanation-row validation
    # ========================================================

    if len(explanations_df) > 0:

        explanation_counts = (
            explanations_df
            .groupby(
                "node_id"
            )
            .size()
        )

        invalid_explanation_counts = (
            explanation_counts[
                explanation_counts
                != TOP_K
            ]
        )

        explanation_counts_valid = (
            len(
                invalid_explanation_counts
            )
            == 0
        )

    else:

        explanation_counts_valid = (
            num_explanations_available
            == 0
        )

    # ========================================================
    # Determine overall quality status
    # ========================================================

    all_output_kernels_valid = (
        num_output_kernel_valid
        == NUM_SELECTED_NODES
    )

    all_explanations_available = (
        num_explanations_available
        == NUM_SELECTED_NODES
    )

    all_converged = (
        num_converged
        == NUM_SELECTED_NODES
    )

    all_full_top_k = (
        num_full_top_k
        == NUM_SELECTED_NODES
    )

    all_beta_finite = (
        num_beta_finite
        == NUM_SELECTED_NODES
    )

    all_beta_nonnegative = (
        num_beta_nonnegative
        == NUM_SELECTED_NODES
    )

    all_objectives_decreased = (
        num_objective_decreased
        == NUM_SELECTED_NODES
    )

    all_initial_objectives_valid = (
        num_initial_objective_valid
        == NUM_SELECTED_NODES
    )

    overall_pass = all(
        [
            all_output_kernels_valid,
            all_explanations_available,
            all_converged,
            all_full_top_k,
            all_beta_finite,
            all_beta_nonnegative,
            all_objectives_decreased,
            all_initial_objectives_valid,
            available_row_count_correct,
            explanation_counts_valid,
        ]
    )

    # ========================================================
    # Terminal summary
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== Baseline GraphLIME Explanation Summary ==="
    )

    print(
        "=" * 70
    )

    print(
        "\nFrozen configuration:"
    )

    print(
        f"  rho:                       {RHO}"
    )

    print(
        f"  num_hops:                  {NUM_HOPS}"
    )

    print(
        f"  top_k:                     {TOP_K}"
    )

    print(
        f"  selected nodes:            "
        f"{NUM_SELECTED_NODES}"
    )

    print(
        "\nOutput-kernel validation:"
    )

    print(
        f"  valid output kernels:      "
        f"{num_output_kernel_valid}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  output-degenerate nodes:   "
        f"{num_output_degenerate}"
    )

    print(
        f"  explanations available:    "
        f"{num_explanations_available}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        "\nOptimization validation:"
    )

    print(
        f"  converged:                 "
        f"{num_converged}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  finite beta:               "
        f"{num_beta_finite}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  nonnegative beta:          "
        f"{num_beta_nonnegative}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  objective decreased:       "
        f"{num_objective_decreased}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  initial objective = 0.5:   "
        f"{num_initial_objective_valid}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        "\nExplanation validation:"
    )

    print(
        f"  >= {TOP_K} nonzero beta:         "
        f"{num_at_least_top_k_nonzero}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  complete top-{TOP_K}:            "
        f"{num_full_top_k}/"
        f"{NUM_SELECTED_NODES}"
    )

    print(
        f"  complete-set target rows:  "
        f"{expected_explanation_rows_if_complete}"
    )

    print(
        f"  expected available rows:   "
        f"{expected_explanation_rows_available}"
    )

    print(
        f"  actual explanation rows:   "
        f"{actual_explanation_rows}"
    )

    print(
        f"  available-row count valid: "
        f"{available_row_count_correct}"
    )

    print(
        f"  per-node row counts valid: "
        f"{explanation_counts_valid}"
    )

    print(
        "\nNonzero-coefficient statistics "
        "(optimized nodes only):"
    )

    print(
        f"  minimum:                   "
        f"{min_nonzero}"
    )

    print(
        f"  mean:                      "
        f"{mean_nonzero:.2f}"
    )

    print(
        f"  median:                    "
        f"{median_nonzero:.2f}"
    )

    print(
        f"  maximum:                   "
        f"{max_nonzero}"
    )

    print(
        "\nNeighbourhood-size statistics "
        "(all selected nodes):"
    )

    print(
        f"  minimum:                   "
        f"{min_neighbourhood_size}"
    )

    print(
        f"  mean:                      "
        f"{mean_neighbourhood_size:.2f}"
    )

    print(
        f"  maximum:                   "
        f"{max_neighbourhood_size}"
    )

    print(
        "\nOptimizer-iteration statistics "
        "(optimized nodes only):"
    )

    print(
        f"  mean:                      "
        f"{mean_iterations:.2f}"
    )

    print(
        f"  maximum:                   "
        f"{max_iterations_observed}"
    )

    # ========================================================
    # Report problematic nodes
    # ========================================================

    problematic_df = (
        diagnostics_df[
            ~(
                diagnostics_df[
                    "output_kernel_valid"
                ]
                &
                diagnostics_df[
                    "explanation_available"
                ]
                &
                diagnostics_df[
                    "converged"
                ]
                &
                diagnostics_df[
                    "has_full_top_k"
                ]
                &
                diagnostics_df[
                    "beta_finite"
                ]
                &
                diagnostics_df[
                    "beta_nonnegative"
                ]
                &
                diagnostics_df[
                    "objective_decreased"
                ]
                &
                diagnostics_df[
                    "initial_objective_valid"
                ]
            )
        ]
    )

    if len(problematic_df) > 0:

        print(
            "\n=== Nodes Requiring Review ==="
        )

        review_columns = [
            "node_id",
            "neighbourhood_size",
            "active_features",
            "output_kernel_valid",
            "explanation_available",
            "failure_reason",
            "converged",
            "iterations",
            "num_nonzero",
            "num_top_features",
            "final_objective",
        ]

        print(
            problematic_df[
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
    # Explicit output-degenerate node report
    # ========================================================

    output_degenerate_df = (
        diagnostics_df[
            diagnostics_df[
                "failure_reason"
            ]
            == "degenerate_local_gnn_outputs"
        ]
    )

    if len(output_degenerate_df) > 0:

        print(
            "\n=== Output-Degenerate Nodes ==="
        )

        degenerate_columns = [
            "node_id",
            "true_label",
            "predicted_label",
            "confidence",
            "neighbourhood_size",
            "subgraph_edges",
            "active_features",
            "degenerate_features",
        ]

        print(
            output_degenerate_df[
                degenerate_columns
            ].to_string(
                index=False
            )
        )

        print(
            "\nThese nodes received no GraphLIME "
            "explanation because their local GNN "
            "outputs contained no positive pairwise "
            "distance."
        )

        print(
            "No artificial output bandwidth was "
            "introduced."
        )

    # ========================================================
    # Saved artifacts
    # ========================================================

    print(
        "\n=== Saved Baseline Artifacts ==="
    )

    print(
        "Available long-format explanations:"
    )

    print(
        EXPLANATIONS_OUTPUT_PATH
    )

    print(
        "\nDiagnostics for all selected nodes:"
    )

    print(
        DIAGNOSTICS_OUTPUT_PATH
    )

    # ========================================================
    # Final milestone status
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "=== Milestone 2C Status ==="
    )

    print(
        "=" * 70
    )

    if overall_pass:

        print(
            "All baseline GraphLIME validation "
            "checks PASSED."
        )

        print(
            f"All {NUM_SELECTED_NODES} selected "
            f"nodes produced complete top-{TOP_K} "
            "explanations."
        )

        print(
            "The generated baseline explanation "
            "artifacts are candidates for freezing."
        )

        print(
            "\nDo NOT begin perturbation experiments "
            "until this output has been reviewed."
        )

    else:

        print(
            "Baseline GraphLIME validation "
            "requires REVIEW."
        )

        print(
            "The complete 100-node diagnostic record "
            "has been preserved."
        )

        if num_output_degenerate > 0:

            print(
                f"\nDetected {num_output_degenerate} "
                "output-degenerate selected node(s)."
            )

            print(
                "No explanation was fabricated for "
                "those nodes."
            )

        print(
            "\nDo NOT change rho automatically."
        )

        print(
            "Do NOT change NUM_HOPS automatically."
        )

        print(
            "Do NOT replace selected nodes yet."
        )

        print(
            "Do NOT freeze the explanation set yet."
        )

        print(
            "Do NOT begin perturbation experiments yet."
        )


if __name__ == "__main__":
    main()
