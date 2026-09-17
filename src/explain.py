# src/explain.py

import torch
from torch_geometric.utils import k_hop_subgraph


# ============================================================
# N-hop neighbourhood extraction
# ============================================================


def get_n_hop_neighborhood(
    node_id: int,
    edge_index: torch.Tensor,
    num_hops: int = 2,
):
    """
    Extract the N-hop neighbourhood of a target node.

    Parameters
    ----------
    node_id : int
        Target node to explain.

    edge_index : torch.Tensor
        Graph connectivity in COO format.

    num_hops : int
        Number of graph hops defining the local neighbourhood.

    Returns
    -------
    subset : torch.Tensor
        Original node IDs in the N-hop neighbourhood.

    sub_edge_index : torch.Tensor
        Relabelled edge index for the extracted subgraph.

    mapping : torch.Tensor
        Position of the target node within the relabelled subset.
    """

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_id,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
    )

    return subset, sub_edge_index, mapping


# ============================================================
# Pairwise squared Euclidean distances
# ============================================================


def pairwise_squared_distances(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the matrix of pairwise squared Euclidean distances.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [n, d] or [n].

    Returns
    -------
    distances : torch.Tensor
        Pairwise squared-distance matrix of shape [n, n].
    """

    if x.ndim == 1:
        x = x.unsqueeze(1)

    if x.ndim != 2:
        raise ValueError(
            "Input must have shape [n] or [n, d]."
        )

    differences = (
        x.unsqueeze(1)
        - x.unsqueeze(0)
    )

    distances = (
        differences.pow(2)
        .sum(dim=-1)
    )

    return distances


# ============================================================
# Median bandwidth
# ============================================================


def median_bandwidth(
    x: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Estimate a Gaussian-kernel bandwidth using the median
    non-zero pairwise Euclidean distance.

    Parameters
    ----------
    x : torch.Tensor
        Input observations with shape [n, d] or [n].

    eps : float
        Numerical tolerance used to identify zero distances.

    Returns
    -------
    sigma : torch.Tensor or None
        Median non-zero pairwise Euclidean distance.

        Returns None when no positive pairwise distance exists,
        which indicates that the observations are locally
        constant or otherwise degenerate.
    """

    squared_distances = pairwise_squared_distances(
        x
    )

    distances = torch.sqrt(
        torch.clamp(
            squared_distances,
            min=0.0,
        )
    )

    n = distances.shape[0]

    upper_triangle = torch.triu(
        torch.ones(
            (n, n),
            dtype=torch.bool,
            device=distances.device,
        ),
        diagonal=1,
    )

    pairwise_values = distances[
        upper_triangle
    ]

    positive_values = pairwise_values[
        pairwise_values > eps
    ]

    if positive_values.numel() == 0:
        return None

    sigma = positive_values.median()

    if (
        not torch.isfinite(sigma)
        or sigma <= eps
    ):
        return None

    return sigma


# ============================================================
# Gaussian kernel
# ============================================================


def gaussian_kernel(
    x: torch.Tensor,
    sigma: torch.Tensor | float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Construct a Gaussian (RBF) Gram matrix.

    K_ij = exp(
        - ||x_i - x_j||^2 / (2 * sigma^2)
    )

    Parameters
    ----------
    x : torch.Tensor
        Input observations of shape [n, d] or [n].

    sigma : torch.Tensor or float
        Positive Gaussian-kernel bandwidth.

    eps : float
        Numerical tolerance.

    Returns
    -------
    kernel : torch.Tensor
        Gaussian Gram matrix of shape [n, n].
    """

    sigma_tensor = torch.as_tensor(
        sigma,
        dtype=x.dtype,
        device=x.device,
    )

    if (
        not torch.isfinite(sigma_tensor)
        or sigma_tensor <= eps
    ):
        raise ValueError(
            "Gaussian kernel bandwidth must be "
            "finite and strictly positive."
        )

    squared_distances = pairwise_squared_distances(
        x
    )

    denominator = (
        2.0 * sigma_tensor.pow(2)
    )

    kernel = torch.exp(
        -squared_distances / denominator
    )

    return kernel


# ============================================================
# Kernel centering
# ============================================================


def center_kernel(
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Center a square Gram matrix.

    Mathematically:

        K_c = H K H

    where:

        H = I - (1/n) 11^T

    Parameters
    ----------
    kernel : torch.Tensor
        Square Gram matrix [n, n].

    Returns
    -------
    centered_kernel : torch.Tensor
        Centered Gram matrix [n, n].
    """

    if kernel.ndim != 2:
        raise ValueError(
            "Kernel must be a two-dimensional matrix."
        )

    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(
            "Kernel matrix must be square."
        )

    n = kernel.shape[0]

    identity = torch.eye(
        n,
        dtype=kernel.dtype,
        device=kernel.device,
    )

    ones = torch.ones(
        (n, n),
        dtype=kernel.dtype,
        device=kernel.device,
    )

    centering_matrix = (
        identity
        - ones / n
    )

    centered_kernel = (
        centering_matrix
        @ kernel
        @ centering_matrix
    )

    return centered_kernel


# ============================================================
# Frobenius normalization
# ============================================================


def frobenius_normalize(
    matrix: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Normalize a matrix by its Frobenius norm.

    Returns None when the Frobenius norm is effectively zero.

    This behaviour is important for locally constant Cora
    features whose centered Gram matrices contain no useful
    variation.

    Parameters
    ----------
    matrix : torch.Tensor
        Input matrix.

    eps : float
        Numerical tolerance.

    Returns
    -------
    normalized_matrix : torch.Tensor or None
        Frobenius-normalized matrix, or None if degenerate.

    norm : torch.Tensor
        Original Frobenius norm.
    """

    norm = torch.linalg.matrix_norm(
        matrix,
        ord="fro",
    )

    if (
        not torch.isfinite(norm)
        or norm <= eps
    ):
        return None, norm

    normalized_matrix = (
        matrix / norm
    )

    return normalized_matrix, norm


# ============================================================
# Center + normalize a Gram matrix
# ============================================================


def prepare_kernel(
    kernel: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Center and Frobenius-normalize a Gram matrix.

    Parameters
    ----------
    kernel : torch.Tensor
        Raw Gram matrix [n, n].

    eps : float
        Numerical tolerance.

    Returns
    -------
    normalized_kernel : torch.Tensor or None
        Centered and normalized Gram matrix.

    frobenius_norm : torch.Tensor
        Frobenius norm of the centered matrix.
    """

    centered_kernel = center_kernel(
        kernel
    )

    normalized_kernel, frobenius_norm = (
        frobenius_normalize(
            centered_kernel,
            eps=eps,
        )
    )

    return normalized_kernel, frobenius_norm


# ============================================================
# GraphLIME input-feature kernels
# ============================================================


def build_feature_kernels(
    local_x: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Construct centered and Frobenius-normalized Gaussian
    kernels for each locally varying input feature.

    GraphLIME associates one Gram matrix with each input
    feature. Features that are constant in the local
    neighbourhood are treated as degenerate and excluded from
    the active kernel set.

    Parameters
    ----------
    local_x : torch.Tensor
        Local feature matrix of shape [n, d].

    eps : float
        Numerical tolerance.

    Returns
    -------
    feature_kernels : dict[int, torch.Tensor]
        Mapping from original feature index to its centered,
        normalized Gram matrix.

    feature_bandwidths : dict[int, float]
        Mapping from original feature index to the Gaussian
        bandwidth used for that feature.

    degenerate_features : list[int]
        Original feature indices that contained no usable local
        variation or produced a zero-norm centered kernel.
    """

    if local_x.ndim != 2:
        raise ValueError(
            "local_x must have shape [n, d]."
        )

    _, num_features = local_x.shape

    feature_kernels = {}
    feature_bandwidths = {}
    degenerate_features = []

    for feature_index in range(
        num_features
    ):

        feature_values = local_x[
            :,
            feature_index
        ]

        sigma = median_bandwidth(
            feature_values,
            eps=eps,
        )

        if sigma is None:
            degenerate_features.append(
                feature_index
            )
            continue

        raw_kernel = gaussian_kernel(
            feature_values,
            sigma=sigma,
            eps=eps,
        )

        normalized_kernel, _ = prepare_kernel(
            raw_kernel,
            eps=eps,
        )

        if normalized_kernel is None:
            degenerate_features.append(
                feature_index
            )
            continue

        if not torch.isfinite(
            normalized_kernel
        ).all():
            raise RuntimeError(
                "Non-finite values encountered in "
                f"feature kernel {feature_index}."
            )

        feature_kernels[
            feature_index
        ] = normalized_kernel

        feature_bandwidths[
            feature_index
        ] = float(
            sigma.item()
        )

    return (
        feature_kernels,
        feature_bandwidths,
        degenerate_features,
    )


# ============================================================
# GraphLIME output kernel
# ============================================================


def build_output_kernel(
    local_y: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Construct the centered and Frobenius-normalized Gaussian
    kernel for local GNN outputs.

    Parameters
    ----------
    local_y : torch.Tensor
        Local GNN probability matrix of shape [n, c].

    eps : float
        Numerical tolerance.

    Returns
    -------
    output_kernel : torch.Tensor
        Centered and Frobenius-normalized output Gram matrix.

    sigma : float
        Gaussian bandwidth used for the output kernel.

    centered_norm : float
        Frobenius norm before normalization.
    """

    if local_y.ndim != 2:
        raise ValueError(
            "local_y must have shape [n, c]."
        )

    sigma = median_bandwidth(
        local_y,
        eps=eps,
    )

    if sigma is None:
        raise RuntimeError(
            "Local GNN outputs are degenerate: "
            "no positive pairwise distance exists."
        )

    raw_kernel = gaussian_kernel(
        local_y,
        sigma=sigma,
        eps=eps,
    )

    output_kernel, centered_norm = (
        prepare_kernel(
            raw_kernel,
            eps=eps,
        )
    )

    if output_kernel is None:
        raise RuntimeError(
            "Centered output kernel has zero "
            "Frobenius norm."
        )

    if not torch.isfinite(
        output_kernel
    ).all():
        raise RuntimeError(
            "Output kernel contains non-finite values."
        )

    return (
        output_kernel,
        float(sigma.item()),
        float(centered_norm.item()),
    )


# ============================================================
# Stack active GraphLIME feature kernels
# ============================================================


def stack_feature_kernels(
    feature_kernels: dict[int, torch.Tensor],
):
    """
    Stack active GraphLIME feature kernels.

    Each centered, normalized n x n kernel is flattened into
    one column of a design matrix.

    If there are p active features and n local nodes:

        design_matrix shape = [n*n, p]
        active_feature_indices shape = [p]

    Parameters
    ----------
    feature_kernels : dict[int, torch.Tensor]
        Mapping from original feature index to its centered,
        normalized Gram matrix.

    Returns
    -------
    design_matrix : torch.Tensor
        Flattened kernel design matrix.

    active_feature_indices : list[int]
        Original feature indices corresponding to the columns
        of design_matrix.
    """

    if len(feature_kernels) == 0:
        raise ValueError(
            "No active feature kernels were provided."
        )

    active_feature_indices = list(
        feature_kernels.keys()
    )

    flattened_kernels = []

    expected_shape = None

    for feature_index in active_feature_indices:

        kernel = feature_kernels[
            feature_index
        ]

        if expected_shape is None:
            expected_shape = kernel.shape

        if kernel.shape != expected_shape:
            raise ValueError(
                "All feature kernels must have "
                "the same shape."
            )

        flattened_kernels.append(
            kernel.reshape(-1)
        )

    design_matrix = torch.stack(
        flattened_kernels,
        dim=1,
    )

    return (
        design_matrix,
        active_feature_indices,
    )


# ============================================================
# HSIC-Lasso objective
# ============================================================


def hsic_lasso_objective(
    design_matrix: torch.Tensor,
    output_vector: torch.Tensor,
    beta: torch.Tensor,
    rho: float,
):
    """
    Evaluate the GraphLIME HSIC-Lasso objective:

        1/2 || y - X beta ||_2^2
        + rho ||beta||_1

    This is equivalent to the Frobenius-norm formulation
    after flattening the normalized Gram matrices.

    Parameters
    ----------
    design_matrix : torch.Tensor
        Matrix of flattened feature kernels [n*n, p].

    output_vector : torch.Tensor
        Flattened output kernel [n*n].

    beta : torch.Tensor
        Non-negative coefficient vector [p].

    rho : float
        L1 regularization parameter.

    Returns
    -------
    objective : torch.Tensor
        Scalar objective value.
    """

    residual = (
        output_vector
        - design_matrix @ beta
    )

    reconstruction_term = (
        0.5
        * torch.sum(
            residual.pow(2)
        )
    )

    regularization_term = (
        rho
        * torch.sum(
            torch.abs(beta)
        )
    )

    objective = (
        reconstruction_term
        + regularization_term
    )

    return objective


# ============================================================
# Non-negative proximal operator
# ============================================================


def nonnegative_soft_threshold(
    values: torch.Tensor,
    threshold: float,
):
    """
    Apply the proximal operator for:

        rho * ||beta||_1
        subject to beta >= 0.

    For non-negative Lasso this reduces to:

        beta = max(0, values - threshold)
    """

    return torch.clamp(
        values - threshold,
        min=0.0,
    )


# ============================================================
# Prepare HSIC-Lasso problem
# ============================================================


def prepare_hsic_lasso_problem(
    feature_kernels: dict[int, torch.Tensor],
    output_kernel: torch.Tensor,
):
    """
    Prepare quantities that are invariant across rho values.

    For one target node, the feature kernels and output kernel
    do not change during a rho sweep. Therefore the flattened
    design matrix, output vector, spectral norm, Lipschitz
    constant, and step size should be computed only once.

    Parameters
    ----------
    feature_kernels : dict[int, torch.Tensor]
        Active centered and Frobenius-normalized feature
        kernels.

    output_kernel : torch.Tensor
        Centered and Frobenius-normalized output kernel.

    Returns
    -------
    problem : dict
        Prepared HSIC-Lasso optimization problem.
    """

    (
        design_matrix,
        active_feature_indices,
    ) = stack_feature_kernels(
        feature_kernels
    )

    output_vector = output_kernel.reshape(
        -1
    )

    if (
        design_matrix.shape[0]
        != output_vector.shape[0]
    ):
        raise ValueError(
            "Feature kernels and output kernel "
            "have incompatible dimensions."
        )

    if not torch.isfinite(
        design_matrix
    ).all():
        raise RuntimeError(
            "Design matrix contains non-finite values."
        )

    if not torch.isfinite(
        output_vector
    ).all():
        raise RuntimeError(
            "Output kernel contains non-finite values."
        )

    # --------------------------------------------------------
    # Gradient Lipschitz constant
    #
    # Smooth term:
    #
    #   f(beta) = 1/2 ||X beta - y||_2^2
    #
    # Gradient:
    #
    #   grad f(beta) = X^T (X beta - y)
    #
    # Lipschitz constant:
    #
    #   L = ||X||_2^2
    # --------------------------------------------------------

    spectral_norm = torch.linalg.matrix_norm(
        design_matrix,
        ord=2,
    )

    lipschitz_constant = (
        spectral_norm.pow(2)
    )

    if (
        not torch.isfinite(
            lipschitz_constant
        )
        or lipschitz_constant <= 0
    ):
        raise RuntimeError(
            "Invalid HSIC-Lasso Lipschitz constant."
        )

    step_size = (
        1.0
        / lipschitz_constant
    )

    return {
        "design_matrix": design_matrix,

        "output_vector": output_vector,

        "active_feature_indices": (
            active_feature_indices
        ),

        "num_active_features": (
            design_matrix.shape[1]
        ),

        "spectral_norm": float(
            spectral_norm.item()
        ),

        "lipschitz_constant": float(
            lipschitz_constant.item()
        ),

        "step_size": float(
            step_size.item()
        ),
    }


# ============================================================
# Solve a prepared HSIC-Lasso problem
# ============================================================


def solve_prepared_hsic_lasso(
    problem: dict,
    rho: float,
    max_iter: int = 10000,
    tolerance: float = 1e-8,
    zero_tolerance: float = 1e-8,
):
    """
    Solve a previously prepared non-negative HSIC-Lasso
    problem for one rho value.

    The prepared problem can be reused across multiple rho
    values for the same target node.

    Parameters
    ----------
    problem : dict
        Output of prepare_hsic_lasso_problem().

    rho : float
        L1 regularization parameter.

    max_iter : int
        Maximum number of proximal-gradient iterations.

    tolerance : float
        Relative coefficient-change convergence tolerance.

    zero_tolerance : float
        Numerical threshold used when counting selected
        coefficients.

    Returns
    -------
    result : dict
        Coefficients and optimization diagnostics.
    """

    if rho < 0:
        raise ValueError(
            "rho must be non-negative."
        )

    design_matrix = problem[
        "design_matrix"
    ]

    output_vector = problem[
        "output_vector"
    ]

    active_feature_indices = problem[
        "active_feature_indices"
    ]

    num_active_features = problem[
        "num_active_features"
    ]

    step_size = problem[
        "step_size"
    ]

    # --------------------------------------------------------
    # Start from beta = 0
    # --------------------------------------------------------

    beta = torch.zeros(
        num_active_features,
        dtype=design_matrix.dtype,
        device=design_matrix.device,
    )

    initial_objective = (
        hsic_lasso_objective(
            design_matrix,
            output_vector,
            beta,
            rho,
        )
    )

    previous_objective = (
        initial_objective
        .detach()
        .clone()
    )

    converged = False

    relative_change = float(
        "inf"
    )

    objective_change = float(
        "inf"
    )

    # --------------------------------------------------------
    # Proximal-gradient optimization
    # --------------------------------------------------------

    for iteration in range(
        1,
        max_iter + 1,
    ):

        gradient = (
            design_matrix.T
            @ (
                design_matrix @ beta
                - output_vector
            )
        )

        gradient_step = (
            beta
            - step_size * gradient
        )

        beta_new = (
            nonnegative_soft_threshold(
                gradient_step,
                threshold=(
                    step_size * rho
                ),
            )
        )

        current_objective = (
            hsic_lasso_objective(
                design_matrix,
                output_vector,
                beta_new,
                rho,
            )
        )

        if not torch.isfinite(
            current_objective
        ):
            raise RuntimeError(
                "HSIC-Lasso objective became "
                "non-finite."
            )

        coefficient_change = (
            torch.linalg.vector_norm(
                beta_new - beta
            )
        )

        coefficient_scale = max(
            float(
                torch.linalg.vector_norm(
                    beta
                ).item()
            ),
            1.0,
        )

        relative_change = (
            float(
                coefficient_change.item()
            )
            / coefficient_scale
        )

        objective_change = abs(
            float(
                current_objective.item()
                - previous_objective.item()
            )
        )

        beta = beta_new

        if (
            relative_change
            < tolerance
        ):
            converged = True
            break

        previous_objective = (
            current_objective
            .detach()
            .clone()
        )

    # --------------------------------------------------------
    # Final objective
    # --------------------------------------------------------

    final_objective = (
        hsic_lasso_objective(
            design_matrix,
            output_vector,
            beta,
            rho,
        )
    )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not torch.isfinite(
        beta
    ).all():
        raise RuntimeError(
            "HSIC-Lasso coefficients contain "
            "non-finite values."
        )

    if torch.any(
        beta < -zero_tolerance
    ):
        raise RuntimeError(
            "HSIC-Lasso produced negative "
            "coefficients."
        )

    nonzero_mask = (
        beta > zero_tolerance
    )

    num_nonzero = int(
        nonzero_mask.sum().item()
    )

    # --------------------------------------------------------
    # Map coefficients back to original feature IDs
    # --------------------------------------------------------

    coefficient_by_feature = {
        feature_index: float(
            beta[position].item()
        )
        for position, feature_index
        in enumerate(
            active_feature_indices
        )
    }

    return {
        "beta": beta,

        "coefficient_by_feature": (
            coefficient_by_feature
        ),

        "active_feature_indices": (
            active_feature_indices
        ),

        "num_active_features": (
            num_active_features
        ),

        "num_nonzero": (
            num_nonzero
        ),

        "initial_objective": float(
            initial_objective.item()
        ),

        "final_objective": float(
            final_objective.item()
        ),

        "iterations": (
            iteration
        ),

        "converged": (
            converged
        ),

        "relative_change": (
            relative_change
        ),

        "objective_change": (
            objective_change
        ),

        "lipschitz_constant": (
            problem[
                "lipschitz_constant"
            ]
        ),

        "step_size": (
            step_size
        ),
    }


# ============================================================
# HSIC-Lasso solver
# ============================================================


def solve_hsic_lasso(
    feature_kernels: dict[int, torch.Tensor],
    output_kernel: torch.Tensor,
    rho: float,
    max_iter: int = 10000,
    tolerance: float = 1e-8,
    zero_tolerance: float = 1e-8,
):
    """
    Convenience wrapper for solving one HSIC-Lasso problem.

    For repeated rho values on the same kernels, prefer:

        prepare_hsic_lasso_problem()
        solve_prepared_hsic_lasso()

    because the expensive spectral norm then only needs to be
    calculated once.
    """

    problem = prepare_hsic_lasso_problem(
        feature_kernels=feature_kernels,
        output_kernel=output_kernel,
    )

    return solve_prepared_hsic_lasso(
        problem=problem,
        rho=rho,
        max_iter=max_iter,
        tolerance=tolerance,
        zero_tolerance=zero_tolerance,
    )


# ============================================================
# Extract top-K GraphLIME features
# ============================================================


def top_k_graphlime_features(
    coefficient_by_feature: dict[int, float],
    top_k: int = 10,
    zero_tolerance: float = 1e-8,
):
    """
    Return the top-K original feature indices ranked by their
    HSIC-Lasso coefficients.

    Only coefficients above zero_tolerance are considered
    selected.

    Parameters
    ----------
    coefficient_by_feature : dict[int, float]
        Mapping from original feature index to coefficient.

    top_k : int
        Maximum number of explanation features.

    zero_tolerance : float
        Numerical threshold defining a non-zero coefficient.

    Returns
    -------
    ranked_features : list[tuple[int, float]]
        List of (original_feature_index, coefficient), sorted
        by descending coefficient.
    """

    selected = [
        (
            feature_index,
            coefficient,
        )
        for (
            feature_index,
            coefficient,
        ) in coefficient_by_feature.items()
        if coefficient > zero_tolerance
    ]

    selected.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    return selected[:top_k]
