# src/perturb.py

import math

import torch


def derive_sampling_seed(
    node_id: int,
    rate_index: int,
    perturbation_seed: int,
) -> int:
    """
    Construct a deterministic sampling seed for one
    node-rate-seed perturbation observation.

    The constants are fixed and have no statistical meaning;
    they simply separate the integer identifiers reproducibly.
    """

    modulus = 2**63 - 1

    derived_seed = (
        int(perturbation_seed) * 1_000_003
        + int(node_id) * 10_007
        + int(rate_index) * 101
    ) % modulus

    return int(derived_seed)


def compute_mask_count(
    num_active_entries: int,
    rate: float,
) -> int:
    """
    Compute the number of active entries to mask.

    Protocol:
        n_mask = max(1, floor(rate * n_active + 0.5))

    for positive perturbation rates.
    """

    if num_active_entries <= 0:
        raise ValueError(
            "num_active_entries must be positive."
        )

    if not (0.0 < rate <= 1.0):
        raise ValueError(
            "rate must satisfy 0 < rate <= 1."
        )

    num_to_mask = max(
        1,
        int(
            math.floor(
                rate * num_active_entries + 0.5
            )
        ),
    )

    return min(
        num_to_mask,
        num_active_entries,
    )


def mask_local_active_entries(
    x: torch.Tensor,
    neighbourhood_nodes: torch.Tensor,
    rate: float,
    sampling_seed: int,
):
    """
    Mask a deterministic random subset of originally active
    feature entries within a target node's neighbourhood.

    Parameters
    ----------
    x:
        Full node-feature matrix [num_nodes, num_features].

    neighbourhood_nodes:
        Global node IDs belonging to the frozen target
        neighbourhood.

    rate:
        Nominal fraction of active local feature entries to mask.

    sampling_seed:
        Deterministic seed for this perturbation observation.

    Returns
    -------
    perturbed_x:
        Clone of x with sampled active entries set to zero.

    metadata:
        Dictionary describing the intervention.

    masked_global_nodes:
        Global node IDs for each masked matrix entry.

    masked_feature_ids:
        Feature-column IDs for each masked matrix entry.
    """

    if x.ndim != 2:
        raise ValueError(
            "x must be a two-dimensional feature matrix."
        )

    if neighbourhood_nodes.ndim != 1:
        neighbourhood_nodes = (
            neighbourhood_nodes.view(-1)
        )

    if neighbourhood_nodes.numel() == 0:
        raise ValueError(
            "Neighbourhood cannot be empty."
        )

    local_x = x[
        neighbourhood_nodes
    ]

    active_coordinates = (
        local_x != 0
    ).nonzero(
        as_tuple=False
    )

    num_active_entries = int(
        active_coordinates.shape[0]
    )

    if num_active_entries == 0:
        raise RuntimeError(
            "No active feature entries exist in the "
            "target neighbourhood."
        )

    num_to_mask = compute_mask_count(
        num_active_entries=num_active_entries,
        rate=rate,
    )

    # Use a CPU generator so the sampling sequence is
    # independent of whether the model runs on CPU or GPU.
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        int(sampling_seed)
    )

    permutation = torch.randperm(
        num_active_entries,
        generator=generator,
    )

    chosen = permutation[
        :num_to_mask
    ]

    # active_coordinates may live on GPU if x lives on GPU.
    # Move the selected indices to the same device before
    # indexing it.
    chosen_device = chosen.to(
        active_coordinates.device
    )

    selected_coordinates = (
        active_coordinates[
            chosen_device
        ]
    )

    local_row_indices = (
        selected_coordinates[:, 0]
    )

    feature_ids = (
        selected_coordinates[:, 1]
    )

    global_node_ids = (
        neighbourhood_nodes[
            local_row_indices
        ]
    )

    # Defensive check: every selected value must have been
    # nonzero in the original feature matrix.
    original_selected_values = x[
        global_node_ids,
        feature_ids,
    ]

    if not (
        original_selected_values != 0
    ).all():

        raise RuntimeError(
            "Perturbation attempted to mask an entry "
            "that was already zero."
        )

    perturbed_x = x.clone()

    perturbed_x[
        global_node_ids,
        feature_ids,
    ] = 0.0

    realized_rate = (
        num_to_mask
        / num_active_entries
    )

    metadata = {
        "active_entries_before": (
            num_active_entries
        ),
        "num_entries_masked": (
            num_to_mask
        ),
        "nominal_mask_rate": (
            float(rate)
        ),
        "realized_mask_rate": (
            float(realized_rate)
        ),
        "sampling_seed": (
            int(sampling_seed)
        ),
    }

    return (
        perturbed_x,
        metadata,
        global_node_ids,
        feature_ids,
    )
