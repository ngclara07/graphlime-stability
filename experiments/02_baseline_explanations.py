# experiments/02_baseline_explanations.py
#
# Purpose:
#   1. Load the frozen Cora GCN checkpoint.
#   2. Reproduce the fixed 100-node explanation sample.
#   3. Validate the 2-hop GraphLIME neighbourhood for one node.
#   4. Construct and validate GraphLIME input/output kernels.
#   5. Run a single-node HSIC-Lasso rho sweep.
#
# Command:
#   python -m experiments.02_baseline_explanations

from pathlib import Path

import pandas as pd
import torch

from src.data import load_cora
from src.explain import (
    build_feature_kernels,
    build_output_kernel,
    get_n_hop_neighborhood,
    solve_hsic_lasso,
    top_k_graphlime_features,
)
from src.model import GCN
from src.utils import set_seed


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42
NUM_SELECTED_NODES = 100

# GraphLIME local-neighbourhood configuration.
NUM_HOPS = 2

# Number of features eventually returned by GraphLIME.
TOP_K = 10

# Diagnostic regularization sweep.
#
# IMPORTANT:
# These values are currently diagnostic only.
# No rho value has been frozen for the final study.
RHO_VALUES = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
]

CHECKPOINT_PATH = Path(
    "models/cora_gcn_seed42.pt"
)

OUTPUT_DIR = Path(
    "results/baseline"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "selected_nodes.csv"
)


def main():

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(SEED)

    # ========================================================
    # Load Cora
    # ========================================================

    dataset, data = load_cora()

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    data = data.to(device)

    # ========================================================
    # Load frozen baseline checkpoint
    # ========================================================

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    print(
        "\n=== Loaded Baseline Checkpoint ==="
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
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
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ========================================================
    # Generate baseline predictions
    # ========================================================

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        probabilities = logits.softmax(
            dim=-1
        )

        predictions = logits.argmax(
            dim=-1
        )

        confidences = probabilities.max(
            dim=-1
        ).values

    # ========================================================
    # Identify correctly classified test nodes
    # ========================================================

    test_node_ids = (
        data.test_mask
        .nonzero(as_tuple=False)
        .view(-1)
    )

    test_predictions = predictions[
        test_node_ids
    ]

    test_labels = data.y[
        test_node_ids
    ]

    correct_mask = (
        test_predictions.eq(
            test_labels
        )
    )

    correct_test_nodes = (
        test_node_ids[
            correct_mask
        ]
    )

    print(
        "\n=== Test-Node Filtering ==="
    )

    print(
        f"Total test nodes: "
        f"{len(test_node_ids)}"
    )

    print(
        "Correctly classified test nodes: "
        f"{len(correct_test_nodes)}"
    )

    correct_fraction = (
        len(correct_test_nodes)
        / len(test_node_ids)
    )

    print(
        f"Correct test fraction: "
        f"{correct_fraction:.4f}"
    )

    # ========================================================
    # Validate checkpoint reconstruction
    # ========================================================

    expected_test_acc = checkpoint[
        "test_accuracy"
    ]

    if abs(
        correct_fraction
        - expected_test_acc
    ) > 1e-6:

        raise RuntimeError(
            "Reconstructed model test accuracy "
            "does not match the saved checkpoint."
        )

    # ========================================================
    # Deterministically select explanation nodes
    # ========================================================

    if (
        len(correct_test_nodes)
        < NUM_SELECTED_NODES
    ):

        raise RuntimeError(
            "Not enough correctly classified "
            "test nodes for requested sample size."
        )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
    )

    correct_test_nodes_cpu = (
        correct_test_nodes
        .detach()
        .cpu()
    )

    permutation = torch.randperm(
        len(correct_test_nodes_cpu),
        generator=generator,
    )

    selected_nodes = (
        correct_test_nodes_cpu[
            permutation[
                :NUM_SELECTED_NODES
            ]
        ]
    )

    # ========================================================
    # Construct selected-node table
    # ========================================================

    selected_records = []

    predictions_cpu = (
        predictions
        .detach()
        .cpu()
    )

    labels_cpu = (
        data.y
        .detach()
        .cpu()
    )

    confidences_cpu = (
        confidences
        .detach()
        .cpu()
    )

    for node_id in (
        selected_nodes.tolist()
    ):

        selected_records.append(
            {
                "node_id": node_id,

                "true_label": int(
                    labels_cpu[
                        node_id
                    ].item()
                ),

                "predicted_label": int(
                    predictions_cpu[
                        node_id
                    ].item()
                ),

                "confidence": float(
                    confidences_cpu[
                        node_id
                    ].item()
                ),
            }
        )

    selected_df = pd.DataFrame(
        selected_records
    )

    # ========================================================
    # Validate fixed sample
    # ========================================================

    assert (
        len(selected_df)
        == NUM_SELECTED_NODES
    )

    assert selected_df[
        "node_id"
    ].is_unique

    assert (
        selected_df["true_label"]
        == selected_df[
            "predicted_label"
        ]
    ).all()

    # ========================================================
    # Save fixed sample
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # Report fixed sample
    # ========================================================

    print(
        "\n=== Fixed Explanation Sample ==="
    )

    print(
        f"Selected nodes: "
        f"{len(selected_df)}"
    )

    print(
        f"Unique nodes: "
        f"{selected_df['node_id'].nunique()}"
    )

    print(
        "All selected nodes correctly classified:",
        bool(
            (
                selected_df[
                    "true_label"
                ]
                == selected_df[
                    "predicted_label"
                ]
            ).all()
        ),
    )

    print(
        "Mean prediction confidence: "
        f"{selected_df['confidence'].mean():.4f}"
    )

    print(
        "\nFirst 10 selected nodes:"
    )

    print(
        selected_df.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\nSelected-node file saved to: "
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # GRAPHLIME LOCAL-INPUT DIAGNOSTIC
    # ========================================================
    #
    # Only the first frozen target node is examined.
    #
    # We intentionally do NOT scale to all 100 nodes yet.
    # ========================================================

    frozen_selected_df = pd.read_csv(
        OUTPUT_PATH
    )

    target_node = int(
        frozen_selected_df.iloc[
            0
        ]["node_id"]
    )

    # --------------------------------------------------------
    # Extract 2-hop neighbourhood
    # --------------------------------------------------------

    (
        subset,
        sub_edge_index,
        mapping,
    ) = get_n_hop_neighborhood(
        node_id=target_node,
        edge_index=data.edge_index,
        num_hops=NUM_HOPS,
    )

    subset_cpu = (
        subset
        .detach()
        .cpu()
    )

    neighbourhood_node_ids = (
        subset_cpu.tolist()
    )

    assert (
        target_node
        in neighbourhood_node_ids
    )

    target_local_index = int(
        mapping.item()
    )

    assert (
        neighbourhood_node_ids[
            target_local_index
        ]
        == target_node
    )

    print(
        "\n=== GraphLIME "
        "Neighbourhood Check ==="
    )

    print(
        f"Target node: "
        f"{target_node}"
    )

    print(
        f"Number of hops: "
        f"{NUM_HOPS}"
    )

    print(
        f"Neighbourhood nodes: "
        f"{len(subset)}"
    )

    print(
        f"Subgraph edges: "
        f"{sub_edge_index.shape[1]}"
    )

    print(
        f"Target local index: "
        f"{target_local_index}"
    )

    print(
        "\nFirst neighbourhood node IDs:"
    )

    print(
        neighbourhood_node_ids[:20]
    )

    # ========================================================
    # Local feature matrix X
    # ========================================================

    local_x = data.x[
        subset
    ]

    print(
        "\n=== Local Feature Matrix ==="
    )

    print(
        f"Shape: "
        f"{local_x.shape}"
    )

    # ========================================================
    # Local GNN-output matrix Y
    # ========================================================

    local_y = probabilities[
        subset
    ]

    print(
        "\n=== Local GNN Outputs ==="
    )

    print(
        f"Shape: "
        f"{local_y.shape}"
    )

    # ========================================================
    # Dimensional validation
    # ========================================================

    assert (
        local_x.shape[0]
        == local_y.shape[0]
    )

    assert (
        local_x.shape[1]
        == dataset.num_features
    )

    assert (
        local_y.shape[1]
        == dataset.num_classes
    )

    assert (
        local_x.shape[0]
        == len(subset)
    )

    # ========================================================
    # Probability validation
    # ========================================================

    probability_sums = (
        local_y.sum(
            dim=-1
        )
    )

    assert torch.allclose(
        probability_sums,
        torch.ones_like(
            probability_sums
        ),
        atol=1e-5,
    )

    # ========================================================
    # Target-node prediction validation
    # ========================================================

    target_true_label = int(
        data.y[
            target_node
        ].item()
    )

    target_predicted_label = int(
        predictions[
            target_node
        ].item()
    )

    target_confidence = float(
        confidences[
            target_node
        ].item()
    )

    print(
        "\n=== Target Node "
        "Prediction Check ==="
    )

    print(
        f"True label: "
        f"{target_true_label}"
    )

    print(
        f"Predicted label: "
        f"{target_predicted_label}"
    )

    print(
        "Prediction confidence: "
        f"{target_confidence:.6f}"
    )

    print(
        "Prediction correct:",
        (
            target_true_label
            == target_predicted_label
        ),
    )

    print(
        "\n=== GraphLIME Input Diagnostic ==="
    )

    print(
        "Neighbourhood extraction: PASSED"
    )

    print(
        "Local feature matrix X: PASSED"
    )

    print(
        "Local prediction matrix Y: PASSED"
    )

    print(
        "Probability validation: PASSED"
    )

    # ========================================================
    # GRAPHLIME KERNEL DIAGNOSTIC
    # ========================================================

    print(
        "\n=== GraphLIME Kernel Diagnostic ==="
    )

    # --------------------------------------------------------
    # Build input-feature kernels
    # --------------------------------------------------------

    (
        feature_kernels,
        feature_bandwidths,
        degenerate_features,
    ) = build_feature_kernels(
        local_x
    )

    num_total_features = (
        local_x.shape[1]
    )

    num_active_features = len(
        feature_kernels
    )

    num_degenerate_features = len(
        degenerate_features
    )

    print(
        "Total input features: "
        f"{num_total_features}"
    )

    print(
        "Active local features: "
        f"{num_active_features}"
    )

    print(
        "Degenerate local features: "
        f"{num_degenerate_features}"
    )

    assert (
        num_active_features
        + num_degenerate_features
        == num_total_features
    )

    # --------------------------------------------------------
    # Validate feature kernels
    # --------------------------------------------------------

    expected_kernel_shape = (
        local_x.shape[0],
        local_x.shape[0],
    )

    for (
        feature_index,
        kernel,
    ) in feature_kernels.items():

        assert (
            kernel.shape
            == expected_kernel_shape
        )

        assert torch.isfinite(
            kernel
        ).all()

        kernel_norm = (
            torch.linalg.matrix_norm(
                kernel,
                ord="fro",
            )
        )

        assert torch.isclose(
            kernel_norm,
            torch.tensor(
                1.0,
                dtype=kernel.dtype,
                device=kernel.device,
            ),
            atol=1e-5,
        )

    print(
        "Feature-kernel shape checks: PASSED"
    )

    print(
        "Feature-kernel finite-value checks: "
        "PASSED"
    )

    print(
        "Feature-kernel norm checks: PASSED"
    )

    # --------------------------------------------------------
    # Build output kernel
    # --------------------------------------------------------

    (
        output_kernel,
        output_bandwidth,
        output_centered_norm,
    ) = build_output_kernel(
        local_y
    )

    print(
        "Output kernel shape: "
        f"{output_kernel.shape}"
    )

    print(
        "Output bandwidth: "
        f"{output_bandwidth:.6f}"
    )

    print(
        "Output centered Frobenius norm: "
        f"{output_centered_norm:.6f}"
    )

    # --------------------------------------------------------
    # Validate output kernel
    # --------------------------------------------------------

    assert (
        output_kernel.shape
        == expected_kernel_shape
    )

    assert torch.isfinite(
        output_kernel
    ).all()

    output_norm = (
        torch.linalg.matrix_norm(
            output_kernel,
            ord="fro",
        )
    )

    assert torch.isclose(
        output_norm,
        torch.tensor(
            1.0,
            dtype=output_kernel.dtype,
            device=output_kernel.device,
        ),
        atol=1e-5,
    )

    print(
        "Output-kernel shape check: PASSED"
    )

    print(
        "Output-kernel finite-value check: "
        "PASSED"
    )

    print(
        "Output-kernel norm check: PASSED"
    )

    # --------------------------------------------------------
    # Validate kernel centering
    # --------------------------------------------------------

    output_row_sums = (
        output_kernel.sum(
            dim=1
        )
    )

    output_col_sums = (
        output_kernel.sum(
            dim=0
        )
    )

    zero_rows = torch.zeros_like(
        output_row_sums
    )

    zero_cols = torch.zeros_like(
        output_col_sums
    )

    assert torch.allclose(
        output_row_sums,
        zero_rows,
        atol=1e-5,
    )

    assert torch.allclose(
        output_col_sums,
        zero_cols,
        atol=1e-5,
    )

    print(
        "Output-kernel centering check: "
        "PASSED"
    )

    # --------------------------------------------------------
    # Report example feature indices
    # --------------------------------------------------------

    active_feature_indices = list(
        feature_kernels.keys()
    )

    print(
        "\nFirst 10 active feature indices:"
    )

    print(
        active_feature_indices[:10]
    )

    print(
        "\nFirst 10 degenerate feature indices:"
    )

    print(
        degenerate_features[:10]
    )

    print(
        "\n=== Kernel Diagnostic Status ==="
    )

    print(
        "Gaussian feature kernels: PASSED"
    )

    print(
        "Gaussian output kernel: PASSED"
    )

    print(
        "Kernel centering: PASSED"
    )

    print(
        "Frobenius normalization: PASSED"
    )

    print(
        "Degenerate-feature handling: PASSED"
    )

    # ========================================================
    # SINGLE-NODE HSIC-LASSO DIAGNOSTIC
    # ========================================================
    #
    # We now solve HSIC-Lasso for node 1903 (the first
    # deterministic target node) across a small rho grid.
    #
    # IMPORTANT:
    # This is a diagnostic sweep.
    #
    # We are NOT selecting/finalizing rho in this script.
    # We are NOT yet generating explanations for all 100
    # target nodes.
    # ========================================================

    print(
        "\n=== HSIC-Lasso Diagnostic: Node "
        f"{target_node} ==="
    )

    hsic_results = []

    for rho in RHO_VALUES:

        print(
            f"\n--- rho = {rho:.6f} ---"
        )

        result = solve_hsic_lasso(
            feature_kernels=feature_kernels,
            output_kernel=output_kernel,
            rho=rho,
            max_iter=10000,
            tolerance=1e-8,
            zero_tolerance=1e-8,
        )

        beta = result[
            "beta"
        ]

        # ----------------------------------------------------
        # Optimizer diagnostics
        # ----------------------------------------------------

        print(
            f"Converged: "
            f"{result['converged']}"
        )

        print(
            f"Iterations: "
            f"{result['iterations']}"
        )

        print(
            "Initial objective: "
            f"{result['initial_objective']:.8f}"
        )

        print(
            "Final objective: "
            f"{result['final_objective']:.8f}"
        )

        objective_reduction = (
            result["initial_objective"]
            - result["final_objective"]
        )

        print(
            "Objective reduction: "
            f"{objective_reduction:.8f}"
        )

        print(
            "Relative coefficient change: "
            f"{result['relative_change']:.3e}"
        )

        print(
            f"Active features: "
            f"{result['num_active_features']}"
        )

        print(
            f"Non-zero coefficients: "
            f"{result['num_nonzero']}"
        )

        print(
            "Minimum beta: "
            f"{float(beta.min().item()):.8f}"
        )

        print(
            "Maximum beta: "
            f"{float(beta.max().item()):.8f}"
        )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        assert torch.isfinite(
            beta
        ).all()

        assert torch.all(
            beta >= -1e-8
        )

        assert (
            result["final_objective"]
            <= result["initial_objective"]
            + 1e-7
        )

        # ----------------------------------------------------
        # Extract top-K original Cora feature indices
        # ----------------------------------------------------

        top_features = (
            top_k_graphlime_features(
                result[
                    "coefficient_by_feature"
                ],
                top_k=TOP_K,
                zero_tolerance=1e-8,
            )
        )

        print(
            f"Top-{TOP_K} features:"
        )

        if len(
            top_features
        ) == 0:

            print(
                "No non-zero features selected."
            )

        else:

            for rank, (
                feature_index,
                coefficient,
            ) in enumerate(
                top_features,
                start=1,
            ):

                print(
                    f"  {rank:02d}. "
                    f"feature={feature_index:4d} "
                    f"beta={coefficient:.8f}"
                )

        # ----------------------------------------------------
        # Store rho-level diagnostic results
        # ----------------------------------------------------

        hsic_results.append(
            {
                "rho": rho,

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

                "num_nonzero": (
                    result[
                        "num_nonzero"
                    ]
                ),

                "max_beta": float(
                    beta.max().item()
                ),

                "top_features": (
                    top_features
                ),
            }
        )

    # ========================================================
    # rho-sweep summary
    # ========================================================

    print(
        "\n=== rho Sweep Summary ==="
    )

    print(
        f"{'rho':>10} "
        f"{'conv':>6} "
        f"{'iters':>8} "
        f"{'nonzero':>9} "
        f"{'objective':>14} "
        f"{'max_beta':>12}"
    )

    for result in hsic_results:

        print(
            f"{result['rho']:>10.6f} "
            f"{str(result['converged']):>6} "
            f"{result['iterations']:>8d} "
            f"{result['num_nonzero']:>9d} "
            f"{result['final_objective']:>14.8f} "
            f"{result['max_beta']:>12.8f}"
        )

    # ========================================================
    # Final HSIC-Lasso diagnostic status
    # ========================================================

    print(
        "\n=== HSIC-Lasso Diagnostic Status ==="
    )

    print(
        "rho sweep completed."
    )

    print(
        "No rho value has been frozen yet."
    )

    print(
        "Baseline explanations for all 100 nodes "
        "have NOT yet been generated."
    )


if __name__ == "__main__":
    main()
