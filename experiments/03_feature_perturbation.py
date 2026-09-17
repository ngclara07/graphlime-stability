# experiments/03_feature_perturbation.py
#
# Phase 9:
# Controlled local active-entry feature perturbation experiment.
#
# Run from repository root:
#
#   python -m experiments.03_feature_perturbation
#
# This experiment:
#
#   - uses the frozen 97-node baseline GraphLIME cohort;
#   - NEVER retrains the GCN;
#   - masks active feature entries only inside each target's
#     frozen 2-hop neighbourhood;
#   - recomputes frozen-GCN predictions after perturbation;
#   - recomputes GraphLIME using perturbed features and outputs;
#   - records prediction stability and explanation stability;
#   - preserves unavailable explanations as missing Jaccard;
#   - periodically checkpoints raw long-format results;
#   - supports deterministic resume after interruption.


from pathlib import Path
import math

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
from src.perturb import (
    derive_sampling_seed,
    mask_local_active_entries,
)
from src.utils import set_seed


# ============================================================
# FROZEN BASELINE CONFIGURATION
# ============================================================

SEED = 42

NUM_HOPS = 2
TOP_K = 10
RHO = 0.03

MAX_ITER = 10000
OPTIMIZER_TOLERANCE = 1e-8
ZERO_TOLERANCE = 1e-8


# ============================================================
# FROZEN PHASE-9 PERTURBATION CONFIGURATION
# ============================================================

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

EXPECTED_TOTAL_OBSERVATIONS = (
    EXPECTED_BASELINE_EXPLAINABLE_NODES
    * len(MASK_RATES)
    * len(PERTURBATION_SEEDS)
)

SAVE_EVERY = 25


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_PATH = Path(
    "models/cora_gcn_seed42.pt"
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

OUTPUT_DIR = Path(
    "results/perturbations"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "feature_mask.csv"
)


# ============================================================
# HELPERS
# ============================================================


def jaccard_similarity(
    first,
    second,
):
    """
    Jaccard similarity between two feature-ID collections.
    """

    first_set = set(
        int(x)
        for x in first
    )

    second_set = set(
        int(x)
        for x in second
    )

    union = (
        first_set
        | second_set
    )

    intersection = (
        first_set
        & second_set
    )

    if len(union) == 0:
        raise ValueError(
            "Jaccard similarity is undefined for "
            "two empty sets."
        )

    score = (
        len(intersection)
        / len(union)
    )

    return (
        float(score),
        int(len(intersection)),
        int(len(union)),
    )


def try_build_output_kernel(
    local_y,
):
    """
    Construct the GraphLIME output kernel while explicitly
    preserving the known output-degenerate case.

    Unrelated RuntimeErrors are re-raised.
    """

    expected_message = (
        "Local GNN outputs are degenerate: "
        "no positive pairwise distance exists."
    )

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
            "output_bandwidth": (
                float(output_bandwidth)
            ),
            "output_centered_norm": (
                float(output_centered_norm)
            ),
            "failure_reason": None,
        }

    except RuntimeError as exc:

        if expected_message not in str(exc):
            raise

        return {
            "success": False,
            "output_kernel": None,
            "output_bandwidth": None,
            "output_centered_norm": None,
            "failure_reason": (
                "degenerate_local_gnn_outputs"
            ),
        }


def save_records(
    records,
):
    """
    Save the complete in-memory raw record set atomically
    enough for ordinary experiment checkpointing.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def make_observation_key(
    node_id,
    nominal_mask_rate,
    perturbation_seed,
):
    """
    Stable key used for resume logic.

    Rate is represented as a fixed decimal string to avoid
    ordinary floating-point equality issues.
    """

    return (
        int(node_id),
        f"{float(nominal_mask_rate):.6f}",
        int(perturbation_seed),
    )


def run_preflight_validation(
    x,
    neighbourhood_nodes,
    node_id,
):
    """
    Validate the perturbation operator before the primary
    experiment begins.

    This checks mechanics only; it does not inspect GraphLIME
    stability outcomes.
    """

    print(
        "\n=== Perturbation Operator Preflight ==="
    )

    test_rate = MASK_RATES[0]
    rate_index = 0
    test_seed = PERTURBATION_SEEDS[0]

    sampling_seed = derive_sampling_seed(
        node_id=node_id,
        rate_index=rate_index,
        perturbation_seed=test_seed,
    )

    original_x = x.clone()

    (
        perturbed_x_1,
        metadata_1,
        masked_nodes_1,
        masked_features_1,
    ) = mask_local_active_entries(
        x=x,
        neighbourhood_nodes=neighbourhood_nodes,
        rate=test_rate,
        sampling_seed=sampling_seed,
    )

    # Repeat exactly to test determinism.
    (
        perturbed_x_2,
        metadata_2,
        masked_nodes_2,
        masked_features_2,
    ) = mask_local_active_entries(
        x=x,
        neighbourhood_nodes=neighbourhood_nodes,
        rate=test_rate,
        sampling_seed=sampling_seed,
    )

    # --------------------------------------------------------
    # Original tensor must remain untouched.
    # --------------------------------------------------------

    if not torch.equal(
        x,
        original_x,
    ):
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: perturbation modified "
            "the original feature tensor."
        )

    # --------------------------------------------------------
    # Determinism.
    # --------------------------------------------------------

    if not torch.equal(
        perturbed_x_1,
        perturbed_x_2,
    ):
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: same seed did not "
            "produce identical perturbation."
        )

    if not torch.equal(
        masked_nodes_1,
        masked_nodes_2,
    ):
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: masked node IDs "
            "are not deterministic."
        )

    if not torch.equal(
        masked_features_1,
        masked_features_2,
    ):
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: masked feature IDs "
            "are not deterministic."
        )

    # --------------------------------------------------------
    # Exact number of changed entries.
    # --------------------------------------------------------

    changed = (
        original_x
        != perturbed_x_1
    ).nonzero(
        as_tuple=False
    )

    actual_changed = int(
        changed.shape[0]
    )

    expected_changed = int(
        metadata_1[
            "num_entries_masked"
        ]
    )

    if actual_changed != expected_changed:
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: changed-entry count "
            "does not equal requested mask count."
        )

    # --------------------------------------------------------
    # All selected values must become exactly zero.
    # --------------------------------------------------------

    selected_after = perturbed_x_1[
        masked_nodes_1,
        masked_features_1,
    ]

    if not torch.equal(
        selected_after,
        torch.zeros_like(
            selected_after
        ),
    ):
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: selected entries "
            "were not set exactly to zero."
        )

    # --------------------------------------------------------
    # All selected values must originally have been active.
    # --------------------------------------------------------

    selected_before = original_x[
        masked_nodes_1,
        masked_features_1,
    ]

    if not (
        selected_before != 0
    ).all():
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: an originally zero "
            "entry was selected."
        )

    # --------------------------------------------------------
    # Changed nodes must belong to neighbourhood.
    # --------------------------------------------------------

    neighbourhood_set = set(
        int(x)
        for x in neighbourhood_nodes
        .detach()
        .cpu()
        .tolist()
    )

    changed_node_set = set(
        int(x)
        for x in changed[:, 0]
        .detach()
        .cpu()
        .tolist()
    )

    if not changed_node_set.issubset(
        neighbourhood_set
    ):
        raise RuntimeError(
            "PRE-FLIGHT FAILURE: perturbation changed "
            "a node outside the target neighbourhood."
        )

    print(
        f"Target node:              {node_id}"
    )

    print(
        f"Nominal test rate:        {test_rate:.2%}"
    )

    print(
        "Active local entries:    "
        f"{metadata_1['active_entries_before']}"
    )

    print(
        "Entries masked:          "
        f"{metadata_1['num_entries_masked']}"
    )

    print(
        "Realized mask rate:      "
        f"{metadata_1['realized_mask_rate']:.6f}"
    )

    print(
        "Original tensor intact:  PASSED"
    )

    print(
        "Deterministic sampling:   PASSED"
    )

    print(
        "Active-only masking:      PASSED"
    )

    print(
        "Local-only masking:       PASSED"
    )

    print(
        "Exact zero replacement:   PASSED"
    )

    print(
        "Preflight status:         PASSED"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Required artifact validation
    # ========================================================

    required_paths = [
        CHECKPOINT_PATH,
        BASELINE_EXPLANATIONS_PATH,
        BASELINE_DIAGNOSTICS_PATH,
        KKT_VALIDATION_PATH,
    ]

    for path in required_paths:

        if not path.exists():
            raise FileNotFoundError(
                f"Required frozen artifact not found: {path}"
            )

    # ========================================================
    # Load Cora
    # ========================================================

    dataset, data = load_cora()

    data = data.to(
        device
    )

    # Keep a frozen reference copy for defensive validation.
    original_x = data.x.clone()

    # ========================================================
    # Load frozen GCN
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if checkpoint["seed"] != SEED:
        raise RuntimeError(
            "Checkpoint seed does not match frozen seed."
        )

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
    # Baseline model outputs
    # ========================================================

    with torch.no_grad():

        baseline_logits = model(
            data.x,
            data.edge_index,
        )

        baseline_probabilities = (
            baseline_logits.softmax(
                dim=-1
            )
        )

        baseline_predictions = (
            baseline_logits.argmax(
                dim=-1
            )
        )

    reconstructed_test_accuracy = float(
        (
            baseline_predictions[
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
            "Reconstructed test accuracy does not "
            "match frozen checkpoint."
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

    print(
        "Reconstructed test accuracy: "
        f"{reconstructed_test_accuracy:.4f}"
    )

    # ========================================================
    # Load frozen baseline artifacts
    # ========================================================

    explanations_df = pd.read_csv(
        BASELINE_EXPLANATIONS_PATH
    )

    diagnostics_df = pd.read_csv(
        BASELINE_DIAGNOSTICS_PATH
    )

    kkt_df = pd.read_csv(
        KKT_VALIDATION_PATH
    )

    # ========================================================
    # Validate KKT freeze
    # ========================================================

    if len(kkt_df) != EXPECTED_BASELINE_EXPLAINABLE_NODES:
        raise RuntimeError(
            "Expected 97 KKT-validated baseline nodes."
        )

    if not kkt_df[
        "overall_node_pass"
    ].astype(bool).all():
        raise RuntimeError(
            "At least one frozen baseline node did not "
            "pass KKT validation."
        )

    # ========================================================
    # Determine frozen explainable cohort
    # ========================================================

    explainable_df = diagnostics_df[
        diagnostics_df[
            "explanation_available"
        ].astype(bool)
    ].copy()

    explainable_node_ids = (
        explainable_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    if (
        len(explainable_node_ids)
        != EXPECTED_BASELINE_EXPLAINABLE_NODES
    ):
        raise RuntimeError(
            "Baseline explainable cohort is not 97 nodes."
        )

    kkt_node_ids = set(
        kkt_df[
            "node_id"
        ]
        .astype(int)
        .tolist()
    )

    if set(
        explainable_node_ids
    ) != kkt_node_ids:
        raise RuntimeError(
            "KKT node set does not match frozen "
            "baseline-explainable cohort."
        )

    # ========================================================
    # Construct frozen baseline explanation map
    # ========================================================

    baseline_explanation_map = {}

    for node_id in explainable_node_ids:

        node_rows = explanations_df[
            explanations_df[
                "node_id"
            ]
            == node_id
        ].sort_values(
            "rank"
        )

        if len(node_rows) != TOP_K:
            raise RuntimeError(
                f"Node {node_id} does not have exactly "
                f"{TOP_K} baseline explanation rows."
            )

        feature_ids = (
            node_rows[
                "feature_id"
            ]
            .astype(int)
            .tolist()
        )

        if len(set(feature_ids)) != TOP_K:
            raise RuntimeError(
                f"Node {node_id} baseline explanation "
                "contains duplicate feature IDs."
            )

        baseline_explanation_map[
            node_id
        ] = feature_ids

    # ========================================================
    # Report frozen Phase-9 protocol
    # ========================================================

    print(
        "\n=== Frozen Phase-9 Protocol ==="
    )

    print(
        f"Baseline explainable nodes: "
        f"{len(explainable_node_ids)}"
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
        f"Mask rates: {MASK_RATES}"
    )

    print(
        f"Perturbation seeds: "
        f"{PERTURBATION_SEEDS}"
    )

    print(
        "Planned observations: "
        f"{EXPECTED_TOTAL_OBSERVATIONS}"
    )

    print(
        "Perturbation type: "
        "local active-entry masking"
    )

    print(
        "Post-mask renormalization: False"
    )

    print(
        "GCN retraining: False"
    )

    # ========================================================
    # Preflight perturbation validation
    # ========================================================

    preflight_node = (
        explainable_node_ids[0]
    )

    (
        preflight_subset,
        _,
        preflight_mapping,
    ) = get_n_hop_neighborhood(
        node_id=preflight_node,
        edge_index=data.edge_index,
        num_hops=NUM_HOPS,
    )

    run_preflight_validation(
        x=data.x,
        neighbourhood_nodes=preflight_subset,
        node_id=preflight_node,
    )

    # Defensive check again after preflight.
    if not torch.equal(
        data.x,
        original_x,
    ):
        raise RuntimeError(
            "Original feature matrix changed during preflight."
        )

    # ========================================================
    # Resume support
    # ========================================================

    records = []
    completed_keys = set()

    if OUTPUT_PATH.exists():

        existing_df = pd.read_csv(
            OUTPUT_PATH
        )

        required_resume_columns = {
            "node_id",
            "nominal_mask_rate",
            "perturbation_seed",
        }

        if not required_resume_columns.issubset(
            existing_df.columns
        ):
            raise RuntimeError(
                "Existing perturbation CSV does not contain "
                "the required resume-key columns."
            )

        if existing_df.duplicated(
            subset=[
                "node_id",
                "nominal_mask_rate",
                "perturbation_seed",
            ]
        ).any():
            raise RuntimeError(
                "Existing perturbation CSV contains "
                "duplicate observation keys."
            )

        records = (
            existing_df
            .to_dict(
                orient="records"
            )
        )

        for row in records:

            completed_keys.add(
                make_observation_key(
                    row["node_id"],
                    row["nominal_mask_rate"],
                    row["perturbation_seed"],
                )
            )

        print(
            "\n=== Resume Mode ==="
        )

        print(
            f"Existing observations: "
            f"{len(records)}"
        )

        print(
            "Completed observations will be skipped."
        )

    else:

        print(
            "\n=== New Experiment ==="
        )

        print(
            "No existing feature_mask.csv found."
        )

    # ========================================================
    # Main perturbation experiment
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Controlled Feature Perturbation Experiment ==="
    )

    print(
        "=" * 78
    )

    newly_completed = 0

    planned_index = 0

    for node_position, node_id in enumerate(
        explainable_node_ids,
        start=1,
    ):

        # ----------------------------------------------------
        # Frozen neighbourhood
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

        if int(
            subset[
                target_local_index
            ].item()
        ) != node_id:
            raise RuntimeError(
                f"Target mapping failed for node {node_id}."
            )

        baseline_predicted_label = int(
            baseline_predictions[
                node_id
            ].item()
        )

        true_label = int(
            data.y[
                node_id
            ].item()
        )

        baseline_confidence = float(
            baseline_probabilities[
                node_id,
                baseline_predicted_label,
            ].item()
        )

        baseline_top_ids = (
            baseline_explanation_map[
                node_id
            ]
        )

        # ----------------------------------------------------
        # Rate loop
        # ----------------------------------------------------

        for rate_index, rate in enumerate(
            MASK_RATES
        ):

            # ------------------------------------------------
            # Seed loop
            # ------------------------------------------------

            for perturbation_seed in PERTURBATION_SEEDS:

                planned_index += 1

                observation_key = (
                    make_observation_key(
                        node_id,
                        rate,
                        perturbation_seed,
                    )
                )

                if observation_key in completed_keys:
                    continue

                sampling_seed = derive_sampling_seed(
                    node_id=node_id,
                    rate_index=rate_index,
                    perturbation_seed=(
                        perturbation_seed
                    ),
                )

                # ============================================
                # Construct independent perturbation
                # ============================================

                (
                    perturbed_x,
                    perturbation_metadata,
                    masked_global_nodes,
                    masked_feature_ids,
                ) = mask_local_active_entries(
                    x=data.x,
                    neighbourhood_nodes=subset,
                    rate=rate,
                    sampling_seed=sampling_seed,
                )

                # Defensive immutability check.
                if not torch.equal(
                    data.x,
                    original_x,
                ):
                    raise RuntimeError(
                        "Original feature matrix was modified."
                    )

                # ============================================
                # Recompute predictions with SAME frozen GCN
                # ============================================

                with torch.no_grad():

                    perturbed_logits = model(
                        perturbed_x,
                        data.edge_index,
                    )

                    perturbed_probabilities = (
                        perturbed_logits.softmax(
                            dim=-1
                        )
                    )

                    perturbed_predictions = (
                        perturbed_logits.argmax(
                            dim=-1
                        )
                    )

                perturbed_predicted_label = int(
                    perturbed_predictions[
                        node_id
                    ].item()
                )

                pred_same = bool(
                    perturbed_predicted_label
                    ==
                    baseline_predicted_label
                )

                perturbed_baseline_class_confidence = float(
                    perturbed_probabilities[
                        node_id,
                        baseline_predicted_label,
                    ].item()
                )

                baseline_class_confidence_delta = float(
                    perturbed_baseline_class_confidence
                    - baseline_confidence
                )

                perturbed_max_confidence = float(
                    perturbed_probabilities[
                        node_id
                    ].max().item()
                )

                # ============================================
                # Perturbed local GraphLIME inputs
                # ============================================

                perturbed_local_x = (
                    perturbed_x[
                        subset
                    ]
                )

                perturbed_local_y = (
                    perturbed_probabilities[
                        subset
                    ]
                )

                # ============================================
                # Default diagnostic state
                # ============================================

                failure_reason = None

                perturbed_output_kernel_valid = False
                perturbed_solver_converged = False
                perturbed_top_k_available = False
                explanation_available = False

                perturbed_active_features = 0
                perturbed_degenerate_features = 0
                perturbed_num_nonzero = 0
                optimizer_iterations = 0

                output_bandwidth = math.nan
                output_centered_norm = math.nan
                final_objective = math.nan

                jaccard = math.nan
                intersection_size = math.nan
                union_size = math.nan

                # ============================================
                # Feature kernels
                # ============================================

                try:

                    (
                        feature_kernels,
                        feature_bandwidths,
                        degenerate_features,
                    ) = build_feature_kernels(
                        perturbed_local_x
                    )

                    perturbed_active_features = int(
                        len(
                            feature_kernels
                        )
                    )

                    perturbed_degenerate_features = int(
                        len(
                            degenerate_features
                        )
                    )

                    if perturbed_active_features == 0:

                        failure_reason = (
                            "no_active_feature_kernels"
                        )

                    else:

                        # ====================================
                        # Output kernel
                        # ====================================

                        output_result = (
                            try_build_output_kernel(
                                perturbed_local_y
                            )
                        )

                        if not output_result[
                            "success"
                        ]:

                            failure_reason = (
                                output_result[
                                    "failure_reason"
                                ]
                            )

                        else:

                            perturbed_output_kernel_valid = True

                            output_bandwidth = (
                                output_result[
                                    "output_bandwidth"
                                ]
                            )

                            output_centered_norm = (
                                output_result[
                                    "output_centered_norm"
                                ]
                            )

                            # ================================
                            # HSIC-Lasso
                            # ================================

                            problem = (
                                prepare_hsic_lasso_problem(
                                    feature_kernels=(
                                        feature_kernels
                                    ),
                                    output_kernel=(
                                        output_result[
                                            "output_kernel"
                                        ]
                                    ),
                                )
                            )

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

                            perturbed_solver_converged = bool(
                                result[
                                    "converged"
                                ]
                            )

                            optimizer_iterations = int(
                                result[
                                    "iterations"
                                ]
                            )

                            perturbed_num_nonzero = int(
                                result[
                                    "num_nonzero"
                                ]
                            )

                            final_objective = float(
                                result[
                                    "final_objective"
                                ]
                            )

                            if not perturbed_solver_converged:

                                failure_reason = (
                                    "optimizer_nonconvergence"
                                )

                            elif (
                                perturbed_num_nonzero
                                < TOP_K
                            ):

                                failure_reason = (
                                    "fewer_than_top_k_nonzero"
                                )

                            else:

                                # ============================
                                # Perturbed top-K
                                # ============================

                                perturbed_top = (
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

                                perturbed_top_ids = [
                                    int(feature_id)
                                    for (
                                        feature_id,
                                        coefficient,
                                    )
                                    in perturbed_top
                                ]

                                if (
                                    len(
                                        perturbed_top_ids
                                    )
                                    != TOP_K
                                ):

                                    failure_reason = (
                                        "incomplete_top_k"
                                    )

                                elif (
                                    len(
                                        set(
                                            perturbed_top_ids
                                        )
                                    )
                                    != TOP_K
                                ):

                                    failure_reason = (
                                        "duplicate_top_k_features"
                                    )

                                else:

                                    perturbed_top_k_available = True
                                    explanation_available = True

                                    (
                                        jaccard,
                                        intersection_size,
                                        union_size,
                                    ) = jaccard_similarity(
                                        baseline_top_ids,
                                        perturbed_top_ids,
                                    )

                except Exception as exc:

                    # Known methodological failures are handled above.
                    # Unexpected failures are preserved in the raw
                    # dataset rather than silently discarded.
                    failure_reason = (
                        "unexpected_error:"
                        + type(exc).__name__
                        + ":"
                        + str(exc)
                    )

                # ============================================
                # Record one raw observation
                # ============================================

                record = {
                    "node_id": (
                        node_id
                    ),
                    "true_label": (
                        true_label
                    ),
                    "nominal_mask_rate": (
                        float(rate)
                    ),
                    "perturbation_seed": (
                        int(
                            perturbation_seed
                        )
                    ),
                    "derived_sampling_seed": (
                        int(
                            sampling_seed
                        )
                    ),
                    "num_hops": (
                        NUM_HOPS
                    ),
                    "top_k": (
                        TOP_K
                    ),
                    "rho": (
                        RHO
                    ),
                    "neighbourhood_size": (
                        neighbourhood_size
                    ),
                    "subgraph_edges": int(
                        sub_edge_index.shape[1]
                    ),
                    "active_entries_before": int(
                        perturbation_metadata[
                            "active_entries_before"
                        ]
                    ),
                    "num_entries_masked": int(
                        perturbation_metadata[
                            "num_entries_masked"
                        ]
                    ),
                    "realized_mask_rate": float(
                        perturbation_metadata[
                            "realized_mask_rate"
                        ]
                    ),
                    "baseline_predicted_label": (
                        baseline_predicted_label
                    ),
                    "perturbed_predicted_label": (
                        perturbed_predicted_label
                    ),
                    "pred_same": (
                        pred_same
                    ),
                    "baseline_confidence": (
                        baseline_confidence
                    ),
                    "perturbed_baseline_class_confidence": (
                        perturbed_baseline_class_confidence
                    ),
                    "baseline_class_confidence_delta": (
                        baseline_class_confidence_delta
                    ),
                    "perturbed_max_confidence": (
                        perturbed_max_confidence
                    ),
                    "baseline_top_k_available": (
                        True
                    ),
                    "perturbed_active_features": (
                        perturbed_active_features
                    ),
                    "perturbed_degenerate_features": (
                        perturbed_degenerate_features
                    ),
                    "perturbed_output_kernel_valid": (
                        perturbed_output_kernel_valid
                    ),
                    "output_bandwidth": (
                        output_bandwidth
                    ),
                    "output_centered_norm": (
                        output_centered_norm
                    ),
                    "perturbed_solver_converged": (
                        perturbed_solver_converged
                    ),
                    "optimizer_iterations": (
                        optimizer_iterations
                    ),
                    "perturbed_num_nonzero": (
                        perturbed_num_nonzero
                    ),
                    "perturbed_top_k_available": (
                        perturbed_top_k_available
                    ),
                    "explanation_available": (
                        explanation_available
                    ),
                    "jaccard": (
                        jaccard
                    ),
                    "intersection_size": (
                        intersection_size
                    ),
                    "union_size": (
                        union_size
                    ),
                    "final_objective": (
                        final_objective
                    ),
                    "failure_reason": (
                        failure_reason
                    ),
                }

                records.append(
                    record
                )

                completed_keys.add(
                    observation_key
                )

                newly_completed += 1

                # ============================================
                # Terminal status
                # ============================================

                if explanation_available:

                    explanation_status = (
                        f"J={jaccard:.3f}"
                    )

                else:

                    explanation_status = (
                        "J=NA "
                        f"({failure_reason})"
                    )

                print(
                    f"[{len(completed_keys):04d}/"
                    f"{EXPECTED_TOTAL_OBSERVATIONS}] "
                    f"node={node_id:4d} | "
                    f"rate={rate:0.2f} | "
                    f"seed={perturbation_seed:3d} | "
                    f"mask="
                    f"{perturbation_metadata['num_entries_masked']:4d}"
                    f"/"
                    f"{perturbation_metadata['active_entries_before']:4d}"
                    f" | "
                    f"pred_same={str(pred_same):5s} | "
                    f"{explanation_status}"
                )

                # ============================================
                # Periodic raw checkpoint
                # ============================================

                if (
                    newly_completed
                    % SAVE_EVERY
                    == 0
                ):

                    save_records(
                        records
                    )

                    print(
                        "  -> checkpoint saved: "
                        f"{OUTPUT_PATH}"
                    )

    # ========================================================
    # Final save
    # ========================================================

    save_records(
        records
    )

    results_df = pd.DataFrame(
        records
    )

    # ========================================================
    # Final structural validation
    # ========================================================

    if results_df.duplicated(
        subset=[
            "node_id",
            "nominal_mask_rate",
            "perturbation_seed",
        ]
    ).any():

        raise RuntimeError(
            "Duplicate perturbation observation keys detected."
        )

    actual_observations = len(
        results_df
    )

    complete_experiment = bool(
        actual_observations
        == EXPECTED_TOTAL_OBSERVATIONS
    )

    # ========================================================
    # Descriptive run-completion summary only
    #
    # This is NOT the Phase-10 statistical analysis.
    # ========================================================

    prediction_same_count = int(
        results_df[
            "pred_same"
        ].astype(bool).sum()
    )

    explanation_available_count = int(
        results_df[
            "explanation_available"
        ].astype(bool).sum()
    )

    unavailable_count = (
        actual_observations
        - explanation_available_count
    )

    valid_jaccard_df = (
        results_df[
            results_df[
                "explanation_available"
            ].astype(bool)
        ]
    )

    if len(valid_jaccard_df) > 0:

        mean_jaccard = float(
            valid_jaccard_df[
                "jaccard"
            ].mean()
        )

    else:

        mean_jaccard = math.nan

    # ========================================================
    # Failure summary
    # ========================================================

    failure_summary = (
        results_df[
            ~results_df[
                "explanation_available"
            ].astype(bool)
        ][
            "failure_reason"
        ]
        .fillna(
            "unspecified"
        )
        .value_counts()
    )

    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-9 Feature Perturbation Run Summary ==="
    )

    print(
        "=" * 78
    )

    print(
        f"\nExpected observations:       "
        f"{EXPECTED_TOTAL_OBSERVATIONS}"
    )

    print(
        f"Observed rows:               "
        f"{actual_observations}"
    )

    print(
        f"Complete experiment:         "
        f"{complete_experiment}"
    )

    print(
        f"\nPrediction class unchanged: "
        f"{prediction_same_count}/"
        f"{actual_observations}"
    )

    print(
        f"Explanation available:       "
        f"{explanation_available_count}/"
        f"{actual_observations}"
    )

    print(
        f"Explanation unavailable:     "
        f"{unavailable_count}/"
        f"{actual_observations}"
    )

    print(
        "\nMean Jaccard over valid explanation pairs "
        "(descriptive only):"
    )

    print(
        f"  {mean_jaccard:.6f}"
    )

    if unavailable_count > 0:

        print(
            "\n=== Explanation-Unavailability Reasons ==="
        )

        print(
            failure_summary.to_string()
        )

    print(
        "\n=== Saved Raw Artifact ==="
    )

    print(
        OUTPUT_PATH
    )

    # ========================================================
    # Final Phase-9 status
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase 9 Status ==="
    )

    print(
        "=" * 78
    )

    if complete_experiment:

        print(
            "All planned controlled feature-perturbation "
            "observations have been recorded."
        )

        print(
            "\nDo NOT alter the frozen perturbation protocol "
            "based on the observed stability values."
        )

        print(
            "\nNext phase:"
        )

        print(
            "Phase 10 — validate, aggregate, and statistically "
            "analyse prediction and explanation stability."
        )

    else:

        print(
            "The perturbation experiment is incomplete."
        )

        print(
            "\nRerun the same command to resume from the "
            "saved CSV."
        )


if __name__ == "__main__":
    main()
