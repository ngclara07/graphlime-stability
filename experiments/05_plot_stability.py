"""
experiments/05_plot_stability.py

Canonical Phase 11:
Visualization of the frozen GraphLIME stability experiment.

Run from the repository root:

    python -m experiments.05_plot_stability

Purpose
-------
This script creates research figures exclusively from frozen Phase-10
analysis artifacts.

It DOES NOT:
    - train or modify the GCN,
    - execute GraphLIME,
    - regenerate perturbations,
    - recompute bootstrap confidence intervals,
    - recompute hypothesis tests,
    - change missing-value handling,
    - tune any experimental parameter.

Statistical quantities displayed in figures are loaded from the frozen
Phase-10 CSV artifacts.

Primary figure
--------------
Four-panel stability figure:

    A. Mean GraphLIME top-10 Jaccard similarity
    B. Mean Jaccard conditional on predicted class remaining unchanged
    C. Prediction stability
    D. Explanation availability

All panels use masking rate on the x-axis and frozen 95% node-level
bootstrap confidence intervals where available.

Additional figures
------------------
    1. Prediction versus explanation stability comparison
    2. H4 endpoint prediction/explanation change scatter
    3. Explanation unavailability by masking rate

Outputs
-------
Figures are written to:

    results/figures/

Both PNG and PDF versions are saved.

Dependencies
------------
    numpy
    pandas
    matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so the script works consistently from
# PowerShell, VS Code terminals, and headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================================
# FROZEN EXPERIMENT CONSTANTS
# ============================================================================

EXPECTED_NODES = 97

MASK_RATES = np.array(
    [0.01, 0.05, 0.10, 0.20],
    dtype=float,
)

EXPECTED_RATE_ROWS = 4

EXPECTED_NODE_RATE_ROWS = (
    EXPECTED_NODES
    * EXPECTED_RATE_ROWS
)

EXPECTED_H4_NODE_ROWS = EXPECTED_NODES

TOP_K = 10

FLOAT_TOL = 1e-9


# ============================================================================
# PATHS
# ============================================================================

ANALYSIS_DIR = Path(
    "results/analysis"
)

FIGURE_DIR = Path(
    "results/figures"
)

RATE_SUMMARY_PATH = (
    ANALYSIS_DIR
    / "rate_summary.csv"
)

NODE_RATE_SUMMARY_PATH = (
    ANALYSIS_DIR
    / "node_rate_summary.csv"
)

H4_NODE_CHANGES_PATH = (
    ANALYSIS_DIR
    / "h4_node_endpoint_changes.csv"
)

H4_COUPLING_PATH = (
    ANALYSIS_DIR
    / "h4_coupling_summary.csv"
)

H5_AVAILABILITY_PATH = (
    ANALYSIS_DIR
    / "h5_availability_by_rate.csv"
)


# ============================================================================
# OUTPUT PATHS
# ============================================================================

MAIN_FIGURE_PNG = (
    FIGURE_DIR
    / "figure_01_main_stability.png"
)

MAIN_FIGURE_PDF = (
    FIGURE_DIR
    / "figure_01_main_stability.pdf"
)

COMPARISON_FIGURE_PNG = (
    FIGURE_DIR
    / "figure_02_prediction_vs_explanation.png"
)

COMPARISON_FIGURE_PDF = (
    FIGURE_DIR
    / "figure_02_prediction_vs_explanation.pdf"
)

H4_FIGURE_PNG = (
    FIGURE_DIR
    / "figure_03_h4_endpoint_coupling.png"
)

H4_FIGURE_PDF = (
    FIGURE_DIR
    / "figure_03_h4_endpoint_coupling.pdf"
)

H5_FIGURE_PNG = (
    FIGURE_DIR
    / "figure_04_explanation_unavailability.png"
)

H5_FIGURE_PDF = (
    FIGURE_DIR
    / "figure_04_explanation_unavailability.pdf"
)


# ============================================================================
# VISUAL STYLE
# ============================================================================

COLOR_JACCARD = "#2166AC"
COLOR_CONDITIONAL = "#4393C3"
COLOR_PREDICTION = "#D6604D"
COLOR_AVAILABILITY = "#4D9221"

COLOR_H4_POINTS = "#5E3C99"
COLOR_H4_ZERO = "#666666"

COLOR_FAILURE = "#B2182B"

GRID_COLOR = "#D9D9D9"

MARKER_SIZE = 6.5

LINE_WIDTH = 2.0

ERROR_CAP_SIZE = 4

DPI = 300


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def require_file(path: Path) -> None:
    """
    Raise a clear error if a required frozen artifact is missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Required frozen Phase-10 artifact does not exist:\n"
            f"  {path}"
        )


def check_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    artifact_name: str,
) -> None:
    """
    Validate that a dataframe contains all required columns.
    """

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{artifact_name} is missing required columns:\n"
            f"  {sorted(missing)}"
        )


def rates_match(
    observed,
) -> bool:
    """
    Verify that observed masking rates equal the frozen rate grid.
    """

    observed = np.asarray(
        sorted(
            np.asarray(
                observed,
                dtype=float,
            )
        ),
        dtype=float,
    )

    if observed.shape != MASK_RATES.shape:
        return False

    return bool(
        np.allclose(
            observed,
            MASK_RATES,
            atol=FLOAT_TOL,
            rtol=0.0,
        )
    )


def format_rate_percent(
    rate: float,
) -> str:
    """
    Convert 0.01 -> '1%', etc.
    """

    return f"{100.0 * float(rate):.0f}%"


def save_figure(
    fig,
    png_path: Path,
    pdf_path: Path,
) -> None:
    """
    Save a figure in both high-resolution PNG and vector PDF formats.
    """

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )


def configure_matplotlib() -> None:
    """
    Configure a restrained research-paper visual style.
    """

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": LINE_WIDTH,
            "savefig.dpi": DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(
    ax,
) -> None:
    """
    Apply common axis styling.
    """

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.7,
    )

    ax.set_axisbelow(
        True
    )

    ax.set_xticks(
        MASK_RATES
    )

    ax.set_xticklabels(
        [
            format_rate_percent(rate)
            for rate in MASK_RATES
        ]
    )

    ax.set_xlabel(
        "Local active-feature masking rate"
    )


def add_panel_label(
    ax,
    label: str,
) -> None:
    """
    Add publication-style panel labels A, B, C, ...
    """

    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


# ============================================================================
# COLUMN RESOLUTION
# ============================================================================


def resolve_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Resolve a column from a small list of accepted names.

    This is useful because earlier Phase-10 scripts may use either
    concise or explicit confidence-interval column names.

    This does NOT infer statistics or recompute anything.
    """

    for column in candidates:

        if column in df.columns:
            return column

    raise RuntimeError(
        f"Could not locate {description}.\n"
        f"Tried columns:\n"
        f"  {candidates}\n"
        f"Available columns:\n"
        f"  {list(df.columns)}"
    )


# ============================================================================
# LOAD AND VALIDATE RATE SUMMARY
# ============================================================================


def load_rate_summary() -> pd.DataFrame:
    """
    Load the frozen rate-level Phase-10 summary.

    The function resolves the exact Phase-10 column names and standardizes
    them into a plotting-only dataframe.
    """

    require_file(
        RATE_SUMMARY_PATH
    )

    raw = pd.read_csv(
        RATE_SUMMARY_PATH
    )

    if len(raw) != EXPECTED_RATE_ROWS:
        raise RuntimeError(
            "rate_summary.csv must contain exactly four rows.\n"
            f"Observed: {len(raw)}"
        )

    rate_col = resolve_column(
        raw,
        [
            "nominal_mask_rate",
            "mask_rate",
            "rate",
        ],
        "masking-rate column",
    )

    prediction_mean_col = resolve_column(
        raw,
        [
            "prediction_stability_mean",
            "mean_prediction_stability",
            "prediction_stability",
        ],
        "prediction-stability mean",
    )

    prediction_low_col = resolve_column(
        raw,
        [
            "prediction_stability_ci_low",
            "mean_prediction_stability_ci_low",
            "prediction_ci_low",
        ],
        "prediction-stability lower CI",
    )

    prediction_high_col = resolve_column(
        raw,
        [
            "prediction_stability_ci_high",
            "mean_prediction_stability_ci_high",
            "prediction_ci_high",
        ],
        "prediction-stability upper CI",
    )

    availability_mean_col = resolve_column(
        raw,
        [
            "explanation_availability_mean",
            "mean_explanation_availability",
            "explanation_availability",
        ],
        "explanation-availability mean",
    )

    availability_low_col = resolve_column(
        raw,
        [
            "explanation_availability_ci_low",
            "mean_explanation_availability_ci_low",
            "availability_ci_low",
        ],
        "explanation-availability lower CI",
    )

    availability_high_col = resolve_column(
        raw,
        [
            "explanation_availability_ci_high",
            "mean_explanation_availability_ci_high",
            "availability_ci_high",
        ],
        "explanation-availability upper CI",
    )

    jaccard_mean_col = resolve_column(
        raw,
        [
            "mean_jaccard",
        ],
        "mean Jaccard",
    )

    jaccard_low_col = resolve_column(
        raw,
        [
            "mean_jaccard_ci_low",
            "jaccard_ci_low",
        ],
        "mean-Jaccard lower CI",
    )

    jaccard_high_col = resolve_column(
        raw,
        [
            "mean_jaccard_ci_high",
            "jaccard_ci_high",
        ],
        "mean-Jaccard upper CI",
    )

    conditional_mean_col = resolve_column(
        raw,
        [
            "mean_jaccard_pred_same",
            "mean_jaccard_prediction_unchanged",
        ],
        "prediction-conditioned mean Jaccard",
    )

    conditional_low_col = resolve_column(
        raw,
        [
            "mean_jaccard_pred_same_ci_low",
            "jaccard_pred_same_ci_low",
            "mean_jaccard_prediction_unchanged_ci_low",
        ],
        "prediction-conditioned Jaccard lower CI",
    )

    conditional_high_col = resolve_column(
        raw,
        [
            "mean_jaccard_pred_same_ci_high",
            "jaccard_pred_same_ci_high",
            "mean_jaccard_prediction_unchanged_ci_high",
        ],
        "prediction-conditioned Jaccard upper CI",
    )

    df = pd.DataFrame(
        {
            "rate": raw[
                rate_col
            ].astype(float),

            "prediction_mean": raw[
                prediction_mean_col
            ].astype(float),

            "prediction_ci_low": raw[
                prediction_low_col
            ].astype(float),

            "prediction_ci_high": raw[
                prediction_high_col
            ].astype(float),

            "availability_mean": raw[
                availability_mean_col
            ].astype(float),

            "availability_ci_low": raw[
                availability_low_col
            ].astype(float),

            "availability_ci_high": raw[
                availability_high_col
            ].astype(float),

            "jaccard_mean": raw[
                jaccard_mean_col
            ].astype(float),

            "jaccard_ci_low": raw[
                jaccard_low_col
            ].astype(float),

            "jaccard_ci_high": raw[
                jaccard_high_col
            ].astype(float),

            "conditional_mean": raw[
                conditional_mean_col
            ].astype(float),

            "conditional_ci_low": raw[
                conditional_low_col
            ].astype(float),

            "conditional_ci_high": raw[
                conditional_high_col
            ].astype(float),
        }
    )

    df = (
        df.sort_values(
            "rate"
        )
        .reset_index(
            drop=True
        )
    )

    if not rates_match(
        df[
            "rate"
        ].to_numpy()
    ):
        raise RuntimeError(
            "rate_summary.csv does not contain the frozen "
            "masking rates [0.01, 0.05, 0.10, 0.20]."
        )

    metric_triplets = [
        (
            "prediction_mean",
            "prediction_ci_low",
            "prediction_ci_high",
        ),
        (
            "availability_mean",
            "availability_ci_low",
            "availability_ci_high",
        ),
        (
            "jaccard_mean",
            "jaccard_ci_low",
            "jaccard_ci_high",
        ),
        (
            "conditional_mean",
            "conditional_ci_low",
            "conditional_ci_high",
        ),
    ]

    for (
        mean_col,
        low_col,
        high_col,
    ) in metric_triplets:

        values = df[
            [
                mean_col,
                low_col,
                high_col,
            ]
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"Non-finite values detected in {mean_col} "
                "or its confidence interval."
            )

        if (
            (
                values < -FLOAT_TOL
            ).any()
            or
            (
                values > 1.0 + FLOAT_TOL
            ).any()
        ):
            raise RuntimeError(
                f"{mean_col} or its confidence interval "
                "contains values outside [0,1]."
            )

        if not (
            df[
                low_col
            ]
            <= df[
                mean_col
            ] + FLOAT_TOL
        ).all():
            raise RuntimeError(
                f"Lower confidence interval exceeds {mean_col}."
            )

        if not (
            df[
                mean_col
            ]
            <= df[
                high_col
            ] + FLOAT_TOL
        ).all():
            raise RuntimeError(
                f"Upper confidence interval is below {mean_col}."
            )

    # ---------------------------------------------------------------------
    # Replay the already validated Phase-10 descriptive values.
    # This catches accidental editing of frozen analysis artifacts.
    # ---------------------------------------------------------------------

    expected_prediction = np.array(
        [
            0.9969,
            0.9876,
            0.9835,
            0.9588,
        ]
    )

    expected_availability = np.array(
        [
            0.9876,
            0.9835,
            0.9825,
            0.9856,
        ]
    )

    expected_jaccard = np.array(
        [
            0.9015,
            0.7253,
            0.6045,
            0.4541,
        ]
    )

    expected_conditional = np.array(
        [
            0.9014,
            0.7266,
            0.6063,
            0.4560,
        ]
    )

    replay_checks = [
        (
            df[
                "prediction_mean"
            ].to_numpy(),
            expected_prediction,
            "prediction stability",
        ),
        (
            df[
                "availability_mean"
            ].to_numpy(),
            expected_availability,
            "explanation availability",
        ),
        (
            df[
                "jaccard_mean"
            ].to_numpy(),
            expected_jaccard,
            "mean Jaccard",
        ),
        (
            df[
                "conditional_mean"
            ].to_numpy(),
            expected_conditional,
            "prediction-conditioned Jaccard",
        ),
    ]

    for (
        observed,
        expected,
        name,
    ) in replay_checks:

        if not np.allclose(
            observed,
            expected,
            atol=5e-4,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Frozen Phase-10 replay check failed for {name}.\n"
                f"Observed: {observed}\n"
                f"Expected approximately: {expected}"
            )

    return df


# ============================================================================
# LOAD NODE-RATE DATA
# ============================================================================


def load_node_rate_summary() -> pd.DataFrame:
    """
    Load the frozen 97 x 4 node-rate summary.
    """

    require_file(
        NODE_RATE_SUMMARY_PATH
    )

    df = pd.read_csv(
        NODE_RATE_SUMMARY_PATH
    )

    required = {
        "node_id",
        "nominal_mask_rate",
        "prediction_stability",
        "explanation_availability",
        "mean_jaccard",
        "mean_jaccard_pred_same",
    }

    check_required_columns(
        df,
        required,
        "node_rate_summary.csv",
    )

    if len(df) != EXPECTED_NODE_RATE_ROWS:
        raise RuntimeError(
            "Unexpected node-rate row count.\n"
            f"Observed: {len(df)}\n"
            f"Expected: {EXPECTED_NODE_RATE_ROWS}"
        )

    if (
        df[
            "node_id"
        ].nunique()
        != EXPECTED_NODES
    ):
        raise RuntimeError(
            "node_rate_summary.csv does not contain "
            "exactly 97 unique target nodes."
        )

    if df.duplicated(
        subset=[
            "node_id",
            "nominal_mask_rate",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate node-rate rows detected."
        )

    if not rates_match(
        df[
            "nominal_mask_rate"
        ].unique()
    ):
        raise RuntimeError(
            "Node-rate summary does not use the frozen rate grid."
        )

    return df


# ============================================================================
# LOAD H4 DATA
# ============================================================================


def load_h4_data():
    """
    Load the frozen H4 endpoint-change artifacts.
    """

    require_file(
        H4_NODE_CHANGES_PATH
    )

    require_file(
        H4_COUPLING_PATH
    )

    nodes = pd.read_csv(
        H4_NODE_CHANGES_PATH
    )

    summary = pd.read_csv(
        H4_COUPLING_PATH
    )

    required_node_columns = {
        "node_id",
        "prediction_stability_change",
        "explanation_stability_change",
        "endpoint_jaccard_available",
        "prediction_endpoint_unchanged",
        "explanation_endpoint_decreased",
        "direct_decoupling_event",
    }

    check_required_columns(
        nodes,
        required_node_columns,
        "h4_node_endpoint_changes.csv",
    )

    if len(nodes) != EXPECTED_H4_NODE_ROWS:
        raise RuntimeError(
            "H4 node endpoint table must contain 97 rows."
        )

    if (
        nodes[
            "node_id"
        ].nunique()
        != EXPECTED_NODES
    ):
        raise RuntimeError(
            "H4 endpoint table must contain 97 unique nodes."
        )

    required_summary_columns = {
        "nodes_with_valid_endpoint_jaccard",
        "spearman_prediction_vs_explanation_change",
        "pearson_prediction_vs_explanation_change",
        "nodes_prediction_unchanged_explanation_decreased",
        "fraction_prediction_unchanged_explanation_decreased",
        "mean_prediction_endpoint_change",
        "mean_explanation_endpoint_change",
    }

    check_required_columns(
        summary,
        required_summary_columns,
        "h4_coupling_summary.csv",
    )

    if len(summary) != 1:
        raise RuntimeError(
            "h4_coupling_summary.csv must contain exactly one row."
        )

    if int(
        summary.iloc[0][
            "nodes_with_valid_endpoint_jaccard"
        ]
    ) != EXPECTED_NODES:
        raise RuntimeError(
            "Expected 97 valid H4 endpoint nodes."
        )

    return nodes, summary


# ============================================================================
# LOAD H5 DATA
# ============================================================================


def load_h5_availability() -> pd.DataFrame:
    """
    Load the frozen H5 rate-level explanation availability table.
    """

    require_file(
        H5_AVAILABILITY_PATH
    )

    df = pd.read_csv(
        H5_AVAILABILITY_PATH
    )

    required = {
        "nominal_mask_rate",
        "explanations_available",
        "explanations_unavailable",
        "explanation_availability_rate",
        "explanation_unavailability_rate",
    }

    check_required_columns(
        df,
        required,
        "h5_availability_by_rate.csv",
    )

    if len(df) != EXPECTED_RATE_ROWS:
        raise RuntimeError(
            "H5 availability table must contain four rows."
        )

    df = (
        df.sort_values(
            "nominal_mask_rate"
        )
        .reset_index(
            drop=True
        )
    )

    if not rates_match(
        df[
            "nominal_mask_rate"
        ].to_numpy()
    ):
        raise RuntimeError(
            "H5 availability table does not contain "
            "the frozen rate grid."
        )

    expected_unavailable = np.array(
        [
            12,
            16,
            17,
            14,
        ],
        dtype=int,
    )

    observed_unavailable = df[
        "explanations_unavailable"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        observed_unavailable,
        expected_unavailable,
    ):
        raise RuntimeError(
            "Frozen H5 unavailable counts do not match "
            "the validated Phase-10 results."
        )

    if int(
        df[
            "explanations_available"
        ].sum()
    ) != 3821:
        raise RuntimeError(
            "Expected exactly 3821 available explanations."
        )

    if int(
        df[
            "explanations_unavailable"
        ].sum()
    ) != 59:
        raise RuntimeError(
            "Expected exactly 59 unavailable explanations."
        )

    return df


# ============================================================================
# ERROR-BAR HELPER
# ============================================================================


def asymmetric_yerr(
    mean,
    low,
    high,
):
    """
    Convert confidence interval endpoints into matplotlib asymmetric
    error-bar distances.
    """

    mean = np.asarray(
        mean,
        dtype=float,
    )

    low = np.asarray(
        low,
        dtype=float,
    )

    high = np.asarray(
        high,
        dtype=float,
    )

    lower_distance = (
        mean
        - low
    )

    upper_distance = (
        high
        - mean
    )

    if (
        lower_distance
        < -FLOAT_TOL
    ).any():
        raise RuntimeError(
            "Invalid lower error-bar distance."
        )

    if (
        upper_distance
        < -FLOAT_TOL
    ).any():
        raise RuntimeError(
            "Invalid upper error-bar distance."
        )

    return np.vstack(
        [
            np.maximum(
                lower_distance,
                0.0,
            ),
            np.maximum(
                upper_distance,
                0.0,
            ),
        ]
    )


# ============================================================================
# COMMON RATE PLOT
# ============================================================================


def plot_rate_metric(
    ax,
    rates,
    means,
    ci_low,
    ci_high,
    color,
    title,
    ylabel,
    panel_label,
    ylim=None,
):
    """
    Plot one frozen rate-level metric with 95% bootstrap confidence
    intervals.
    """

    yerr = asymmetric_yerr(
        means,
        ci_low,
        ci_high,
    )

    ax.errorbar(
        rates,
        means,
        yerr=yerr,
        color=color,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        capthick=1.2,
        elinewidth=1.2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markeredgecolor=color,
        zorder=3,
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
    )

    ax.set_ylabel(
        ylabel
    )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    style_axis(
        ax
    )

    add_panel_label(
        ax,
        panel_label,
    )


# ============================================================================
# FIGURE 1 — MAIN FOUR-PANEL STABILITY FIGURE
# ============================================================================


def create_main_stability_figure(
    rate_df: pd.DataFrame,
):
    """
    Create the principal four-panel research figure.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 8.0),
        constrained_layout=True,
    )

    axes = axes.flatten()

    rates = rate_df[
        "rate"
    ].to_numpy(
        dtype=float
    )

    # ---------------------------------------------------------------------
    # Panel A
    # ---------------------------------------------------------------------

    plot_rate_metric(
        ax=axes[0],
        rates=rates,
        means=rate_df[
            "jaccard_mean"
        ],
        ci_low=rate_df[
            "jaccard_ci_low"
        ],
        ci_high=rate_df[
            "jaccard_ci_high"
        ],
        color=COLOR_JACCARD,
        title=(
            "GraphLIME explanation overlap"
        ),
        ylabel=(
            "Mean top-10 Jaccard similarity"
        ),
        panel_label="A",
        ylim=(0.35, 1.00),
    )

    # ---------------------------------------------------------------------
    # Panel B
    # ---------------------------------------------------------------------

    plot_rate_metric(
        ax=axes[1],
        rates=rates,
        means=rate_df[
            "conditional_mean"
        ],
        ci_low=rate_df[
            "conditional_ci_low"
        ],
        ci_high=rate_df[
            "conditional_ci_high"
        ],
        color=COLOR_CONDITIONAL,
        title=(
            "Explanation overlap when prediction is unchanged"
        ),
        ylabel=(
            "Mean top-10 Jaccard similarity"
        ),
        panel_label="B",
        ylim=(0.35, 1.00),
    )

    # ---------------------------------------------------------------------
    # Panel C
    # ---------------------------------------------------------------------

    plot_rate_metric(
        ax=axes[2],
        rates=rates,
        means=rate_df[
            "prediction_mean"
        ],
        ci_low=rate_df[
            "prediction_ci_low"
        ],
        ci_high=rate_df[
            "prediction_ci_high"
        ],
        color=COLOR_PREDICTION,
        title=(
            "Prediction stability"
        ),
        ylabel=(
            "Predicted-class agreement"
        ),
        panel_label="C",
        ylim=(0.90, 1.005),
    )

    # ---------------------------------------------------------------------
    # Panel D
    # ---------------------------------------------------------------------

    plot_rate_metric(
        ax=axes[3],
        rates=rates,
        means=rate_df[
            "availability_mean"
        ],
        ci_low=rate_df[
            "availability_ci_low"
        ],
        ci_high=rate_df[
            "availability_ci_high"
        ],
        color=COLOR_AVAILABILITY,
        title=(
            "Explanation availability"
        ),
        ylabel=(
            "Fraction with valid explanation"
        ),
        panel_label="D",
        ylim=(0.94, 1.005),
    )

    fig.suptitle(
        "GraphLIME stability under controlled local feature masking",
        fontweight="bold",
    )

    # The plotted points are means across target nodes at each rate.
    # The uncertainty intervals are the already frozen Phase-10
    # node-level bootstrap confidence intervals.
    fig.text(
        0.5,
        -0.015,
        (
            "Points show means across target nodes; error bars show frozen "
            "95% node-level bootstrap confidence intervals."
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )

    save_figure(
        fig,
        MAIN_FIGURE_PNG,
        MAIN_FIGURE_PDF,
    )

    plt.close(
        fig
    )


# ============================================================================
# FIGURE 2 — PREDICTION VS EXPLANATION STABILITY
# ============================================================================


def create_prediction_vs_explanation_figure(
    rate_df: pd.DataFrame,
):
    """
    Compare explanation overlap and prediction stability directly.

    Both quantities lie in [0,1], so a shared numerical y-axis is
    appropriate. The metrics nevertheless represent different notions
    of stability, so the figure remains descriptive rather than treating
    them as substantively identical quantities.
    """

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2),
        constrained_layout=True,
    )

    rates = rate_df[
        "rate"
    ].to_numpy(
        dtype=float
    )

    jaccard_yerr = asymmetric_yerr(
        rate_df[
            "jaccard_mean"
        ],
        rate_df[
            "jaccard_ci_low"
        ],
        rate_df[
            "jaccard_ci_high"
        ],
    )

    prediction_yerr = asymmetric_yerr(
        rate_df[
            "prediction_mean"
        ],
        rate_df[
            "prediction_ci_low"
        ],
        rate_df[
            "prediction_ci_high"
        ],
    )

    ax.errorbar(
        rates,
        rate_df[
            "jaccard_mean"
        ],
        yerr=jaccard_yerr,
        color=COLOR_JACCARD,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        label="Explanation overlap (Jaccard)",
        zorder=3,
    )

    ax.errorbar(
        rates,
        rate_df[
            "prediction_mean"
        ],
        yerr=prediction_yerr,
        color=COLOR_PREDICTION,
        marker="s",
        markersize=MARKER_SIZE,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=LINE_WIDTH,
        capsize=ERROR_CAP_SIZE,
        label="Prediction stability",
        zorder=3,
    )

    style_axis(
        ax
    )

    ax.set_ylabel(
        "Stability"
    )

    ax.set_ylim(
        0.35,
        1.02,
    )

    ax.set_title(
        "Prediction and explanation stability under feature masking",
        loc="left",
        fontweight="bold",
    )

    # Place the legend away from the explanatory note and away from the
    # steeply declining Jaccard trajectory.
    ax.legend(
        frameon=False,
        loc="upper right",
    )

    # Presentation-only clarification. This text does not introduce any
    # additional statistical claim.
    ax.text(
        0.02,
        0.035,
        (
            "Metrics share a numerical scale but represent "
            "different notions of stability."
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
        ha="left",
        va="bottom",
    )

    save_figure(
        fig,
        COMPARISON_FIGURE_PNG,
        COMPARISON_FIGURE_PDF,
    )

    plt.close(
        fig
    )


# ============================================================================
# FIGURE 3 — H4 ENDPOINT COUPLING
# ============================================================================


def create_h4_coupling_figure(
    h4_nodes: pd.DataFrame,
    h4_summary: pd.DataFrame,
):
    """
    Plot node-level endpoint changes:

        x = prediction stability change, 20% - 1%
        y = explanation stability change, 20% - 1%

    Points in the x=0, y<0 region directly illustrate nodes whose
    endpoint prediction-stability proportion is unchanged while their
    explanation overlap decreases.

    No jitter is applied. Exact x coordinates are deliberately preserved
    because prediction-stability changes are discrete quantities. Smaller,
    partially transparent markers are used for the direct-decoupling group
    to reduce visual occlusion without inventing artificial variation.
    """

    valid = h4_nodes.loc[
        h4_nodes[
            "endpoint_jaccard_available"
        ].astype(bool)
    ].copy()

    if len(valid) != EXPECTED_NODES:
        raise RuntimeError(
            "Expected all 97 H4 nodes to have valid endpoint Jaccard."
        )

    summary = h4_summary.iloc[
        0
    ]

    x = valid[
        "prediction_stability_change"
    ].to_numpy(
        dtype=float
    )

    y = valid[
        "explanation_stability_change"
    ].to_numpy(
        dtype=float
    )

    direct = valid[
        "direct_decoupling_event"
    ].astype(bool).to_numpy()

    fig, ax = plt.subplots(
        figsize=(7.4, 5.8),
        constrained_layout=True,
    )

    # ---------------------------------------------------------------------
    # General endpoint nodes.
    # ---------------------------------------------------------------------

    ax.scatter(
        x[
            ~direct
        ],
        y[
            ~direct
        ],
        s=46,
        color=COLOR_H4_POINTS,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
        label="Other endpoint patterns",
        zorder=3,
    )

    # ---------------------------------------------------------------------
    # Direct decoupling nodes.
    #
    # Exact coordinates are preserved. We intentionally do not add jitter.
    # The smaller marker size and partial transparency reduce overplotting
    # while retaining the actual discrete endpoint values.
    # ---------------------------------------------------------------------

    ax.scatter(
        x[
            direct
        ],
        y[
            direct
        ],
        s=38,
        facecolor="white",
        edgecolor=COLOR_JACCARD,
        linewidth=1.2,
        alpha=0.72,
        label=(
            "Prediction stability unchanged,\n"
            "explanation overlap decreased"
        ),
        zorder=4,
    )

    ax.axhline(
        0.0,
        color=COLOR_H4_ZERO,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )

    ax.axvline(
        0.0,
        color=COLOR_H4_ZERO,
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        zorder=1,
    )

    ax.grid(
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.65,
    )

    ax.set_axisbelow(
        True
    )

    ax.set_xlabel(
        "Change in prediction stability (20% - 1%)"
    )

    ax.set_ylabel(
        "Change in mean Jaccard similarity (20% - 1%)"
    )

    ax.set_title(
        "Endpoint changes in prediction and explanation stability",
        loc="left",
        fontweight="bold",
    )

    spearman = float(
        summary[
            "spearman_prediction_vs_explanation_change"
        ]
    )

    pearson = float(
        summary[
            "pearson_prediction_vs_explanation_change"
        ]
    )

    direct_count = int(
        summary[
            "nodes_prediction_unchanged_explanation_decreased"
        ]
    )

    direct_fraction = float(
        summary[
            "fraction_prediction_unchanged_explanation_decreased"
        ]
    )

    annotation = (
        f"Spearman ρ = {spearman:.3f}\n"
        f"Pearson r = {pearson:.3f}\n"
        f"Direct decoupling: "
        f"{direct_count}/{EXPECTED_NODES} "
        f"({100.0 * direct_fraction:.1f}%)"
    )

    ax.text(
        0.98,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "alpha": 0.95,
        },
    )

    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=8.5,
    )

    save_figure(
        fig,
        H4_FIGURE_PNG,
        H4_FIGURE_PDF,
    )

    plt.close(
        fig
    )


# ============================================================================
# FIGURE 4 — H5 EXPLANATION UNAVAILABILITY
# ============================================================================


def create_h5_availability_figure(
    h5_df: pd.DataFrame,
):
    """
    Plot the number and rate of unavailable perturbed explanations.

    The figure displays the observed counts directly and does not impose
    a fitted trend.
    """

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0),
        constrained_layout=True,
    )

    rates = h5_df[
        "nominal_mask_rate"
    ].to_numpy(
        dtype=float
    )

    unavailable = h5_df[
        "explanations_unavailable"
    ].to_numpy(
        dtype=int
    )

    bars = ax.bar(
        rates,
        unavailable,
        width=0.018,
        color=COLOR_FAILURE,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )

    for (
        bar,
        count,
        unavailability_rate,
    ) in zip(
        bars,
        unavailable,
        h5_df[
            "explanation_unavailability_rate"
        ].to_numpy(
            dtype=float
        ),
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            bar.get_height() + 0.35,
            (
                f"{count}\n"
                f"({100.0 * unavailability_rate:.2f}%)"
            ),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(
        MASK_RATES
    )

    ax.set_xticklabels(
        [
            format_rate_percent(rate)
            for rate in MASK_RATES
        ]
    )

    ax.set_xlabel(
        "Local active-feature masking rate"
    )

    ax.set_ylabel(
        "Unavailable GraphLIME explanations\n"
        "(out of 970 observations per rate)"
    )

    ax.set_ylim(
        0,
        max(
            unavailable
        ) + 5,
    )

    ax.set_title(
        "GraphLIME explanation unavailability by masking rate",
        loc="left",
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.7,
    )

    ax.set_axisbelow(
        True
    )

    save_figure(
        fig,
        H5_FIGURE_PNG,
        H5_FIGURE_PDF,
    )

    plt.close(
        fig
    )


# ============================================================================
# OUTPUT VALIDATION
# ============================================================================


def validate_generated_files() -> None:
    """
    Confirm that every expected PNG/PDF artifact was created and is nonempty.
    """

    expected_files = [
        MAIN_FIGURE_PNG,
        MAIN_FIGURE_PDF,
        COMPARISON_FIGURE_PNG,
        COMPARISON_FIGURE_PDF,
        H4_FIGURE_PNG,
        H4_FIGURE_PDF,
        H5_FIGURE_PNG,
        H5_FIGURE_PDF,
    ]

    failures = []

    for path in expected_files:

        if not path.exists():

            failures.append(
                f"Missing: {path}"
            )

            continue

        if path.stat().st_size <= 0:

            failures.append(
                f"Empty: {path}"
            )

    if failures:

        raise RuntimeError(
            "Figure-output validation failed:\n"
            + "\n".join(
                failures
            )
        )

    print(
        f"Generated figure files:       "
        f"{len(expected_files)}/{len(expected_files)} PASS"
    )


# ============================================================================
# NUMERICAL PLOT-DATA VALIDATION
# ============================================================================


def validate_plot_data(
    rate_df: pd.DataFrame,
    node_rate_df: pd.DataFrame,
    h4_nodes: pd.DataFrame,
    h4_summary: pd.DataFrame,
    h5_df: pd.DataFrame,
) -> None:
    """
    Validate all frozen numerical inputs before any figure is created.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-11 Plot-Data Validation ==="
    )

    print(
        "=" * 78
    )

    if len(
        rate_df
    ) != 4:
        raise RuntimeError(
            "Rate-level plotting table must contain four rows."
        )

    print(
        "Rate-level rows:             4/4 PASS"
    )

    if len(
        node_rate_df
    ) != EXPECTED_NODE_RATE_ROWS:
        raise RuntimeError(
            "Node-rate plotting table must contain 388 rows."
        )

    print(
        "Node-rate rows:              388/388 PASS"
    )

    if len(
        h4_nodes
    ) != EXPECTED_NODES:
        raise RuntimeError(
            "H4 plotting table must contain 97 nodes."
        )

    print(
        "H4 endpoint nodes:           97/97 PASS"
    )

    if len(
        h4_summary
    ) != 1:
        raise RuntimeError(
            "H4 summary must contain one row."
        )

    print(
        "H4 coupling summary:         1/1 PASS"
    )

    if len(
        h5_df
    ) != 4:
        raise RuntimeError(
            "H5 plotting table must contain four rows."
        )

    print(
        "H5 rate rows:                4/4 PASS"
    )

    # ---------------------------------------------------------------------
    # Explicit replay of the central Phase-10 endpoint results.
    # ---------------------------------------------------------------------

    low = rate_df.iloc[
        0
    ]

    high = rate_df.iloc[
        -1
    ]

    jaccard_change = (
        float(
            high[
                "jaccard_mean"
            ]
        )
        -
        float(
            low[
                "jaccard_mean"
            ]
        )
    )

    prediction_change = (
        float(
            high[
                "prediction_mean"
            ]
        )
        -
        float(
            low[
                "prediction_mean"
            ]
        )
    )

    conditional_change = (
        float(
            high[
                "conditional_mean"
            ]
        )
        -
        float(
            low[
                "conditional_mean"
            ]
        )
    )

    availability_change = (
        float(
            high[
                "availability_mean"
            ]
        )
        -
        float(
            low[
                "availability_mean"
            ]
        )
    )

    if not np.isclose(
        jaccard_change,
        -0.4474,
        atol=5e-4,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected plotted Jaccard endpoint change."
        )

    if not np.isclose(
        prediction_change,
        -0.0381,
        atol=5e-4,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected plotted prediction endpoint change."
        )

    if not np.isclose(
        conditional_change,
        -0.4454,
        atol=5e-4,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected plotted conditioned-Jaccard endpoint change."
        )

    if not np.isclose(
        availability_change,
        -0.0020,
        atol=5e-4,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected plotted explanation-availability endpoint change."
        )

    print(
        "Frozen endpoint replay:      PASS"
    )

    # ---------------------------------------------------------------------
    # H4 replay.
    # ---------------------------------------------------------------------

    summary = h4_summary.iloc[
        0
    ]

    if int(
        summary[
            "nodes_prediction_unchanged_explanation_decreased"
        ]
    ) != 86:
        raise RuntimeError(
            "Expected 86 direct H4 decoupling nodes."
        )

    if not np.isclose(
        float(
            summary[
                "fraction_prediction_unchanged_explanation_decreased"
            ]
        ),
        0.8866,
        atol=5e-4,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected H4 direct-decoupling fraction."
        )

    print(
        "H4 frozen-result replay:     PASS"
    )

    # ---------------------------------------------------------------------
    # H5 replay.
    # ---------------------------------------------------------------------

    expected_unavailable = np.array(
        [
            12,
            16,
            17,
            14,
        ]
    )

    if not np.array_equal(
        h5_df[
            "explanations_unavailable"
        ].to_numpy(
            dtype=int
        ),
        expected_unavailable,
    ):
        raise RuntimeError(
            "Unexpected H5 unavailable-count sequence."
        )

    print(
        "H5 frozen-result replay:     PASS"
    )

    print(
        "\nPhase-11 plot-data validation: PASSED"
    )


# ============================================================================
# TERMINAL SUMMARY
# ============================================================================


def print_plotting_summary(
    rate_df: pd.DataFrame,
    h4_summary: pd.DataFrame,
    h5_df: pd.DataFrame,
) -> None:
    """
    Print exactly what was visualized.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Frozen Quantities Visualized ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nRate   Pred.Stab.   Expl.Avail.   Mean J   Mean J | Pred.Same"
    )

    print(
        "-" * 68
    )

    for _, row in rate_df.iterrows():

        print(
            f"{row['rate']:.2f}"
            f"   "
            f"{row['prediction_mean']:.4f}"
            f"       "
            f"{row['availability_mean']:.4f}"
            f"        "
            f"{row['jaccard_mean']:.4f}"
            f"     "
            f"{row['conditional_mean']:.4f}"
        )

    summary = h4_summary.iloc[
        0
    ]

    print(
        "\nH4 endpoint coupling:"
    )

    print(
        "  Spearman rho: "
        f"{float(summary['spearman_prediction_vs_explanation_change']):.4f}"
    )

    print(
        "  Pearson r:    "
        f"{float(summary['pearson_prediction_vs_explanation_change']):.4f}"
    )

    print(
        "  Direct decoupling nodes: "
        f"{int(summary['nodes_prediction_unchanged_explanation_decreased'])}"
        f"/{EXPECTED_NODES}"
    )

    print(
        "\nH5 unavailable explanations:"
    )

    print(
        "  "
        + str(
            h5_df[
                "explanations_unavailable"
            ]
            .astype(int)
            .tolist()
        )
    )


# ============================================================================
# MAIN
# ============================================================================


def main():

    print(
        "=" * 78
    )

    print(
        "=== Phase 11: GraphLIME Stability Visualization ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nThis script is visualization-only."
    )

    print(
        "It does not retrain the GCN, rerun GraphLIME, regenerate "
        "perturbations, or recompute statistical inference."
    )

    print(
        "\nAll plotted statistics are loaded from frozen "
        "Phase-10 analysis artifacts."
    )

    # ---------------------------------------------------------------------
    # Configure visual style.
    # ---------------------------------------------------------------------

    configure_matplotlib()

    # ---------------------------------------------------------------------
    # Load frozen artifacts.
    # ---------------------------------------------------------------------

    print(
        "\n=== Loading Frozen Phase-10 Artifacts ==="
    )

    rate_df = (
        load_rate_summary()
    )

    print(
        f"Loaded rate summary:          "
        f"{len(rate_df)} rows"
    )

    node_rate_df = (
        load_node_rate_summary()
    )

    print(
        f"Loaded node-rate summary:     "
        f"{len(node_rate_df)} rows"
    )

    (
        h4_nodes,
        h4_summary,
    ) = load_h4_data()

    print(
        f"Loaded H4 endpoint nodes:     "
        f"{len(h4_nodes)} rows"
    )

    print(
        f"Loaded H4 coupling summary:   "
        f"{len(h4_summary)} row"
    )

    h5_df = (
        load_h5_availability()
    )

    print(
        f"Loaded H5 availability:       "
        f"{len(h5_df)} rows"
    )

    # ---------------------------------------------------------------------
    # Validate all plot inputs before generating any figures.
    # ---------------------------------------------------------------------

    validate_plot_data(
        rate_df=rate_df,
        node_rate_df=node_rate_df,
        h4_nodes=h4_nodes,
        h4_summary=h4_summary,
        h5_df=h5_df,
    )

    # ---------------------------------------------------------------------
    # Create output directory only after numerical validation passes.
    # ---------------------------------------------------------------------

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Figure 1
    # ---------------------------------------------------------------------

    print(
        "\n=== Generating Figure 1: Main Stability Figure ==="
    )

    create_main_stability_figure(
        rate_df
    )

    print(
        "Figure 1 generated."
    )

    # ---------------------------------------------------------------------
    # Figure 2
    # ---------------------------------------------------------------------

    print(
        "\n=== Generating Figure 2: Prediction vs Explanation Stability ==="
    )

    create_prediction_vs_explanation_figure(
        rate_df
    )

    print(
        "Figure 2 generated."
    )

    # ---------------------------------------------------------------------
    # Figure 3
    # ---------------------------------------------------------------------

    print(
        "\n=== Generating Figure 3: H4 Endpoint Coupling ==="
    )

    create_h4_coupling_figure(
        h4_nodes=h4_nodes,
        h4_summary=h4_summary,
    )

    print(
        "Figure 3 generated."
    )

    # ---------------------------------------------------------------------
    # Figure 4
    # ---------------------------------------------------------------------

    print(
        "\n=== Generating Figure 4: Explanation Unavailability ==="
    )

    create_h5_availability_figure(
        h5_df
    )

    print(
        "Figure 4 generated."
    )

    # ---------------------------------------------------------------------
    # Validate generated files.
    # ---------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-11 Figure Output Validation ==="
    )

    print(
        "=" * 78
    )

    validate_generated_files()

    # ---------------------------------------------------------------------
    # Print exact frozen quantities represented.
    # ---------------------------------------------------------------------

    print_plotting_summary(
        rate_df=rate_df,
        h4_summary=h4_summary,
        h5_df=h5_df,
    )

    # ---------------------------------------------------------------------
    # Artifact report.
    # ---------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Saved Phase-11 Figure Artifacts ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nMain stability figure:"
    )

    print(
        f"  {MAIN_FIGURE_PNG}"
    )

    print(
        f"  {MAIN_FIGURE_PDF}"
    )

    print(
        "\nPrediction/explanation comparison:"
    )

    print(
        f"  {COMPARISON_FIGURE_PNG}"
    )

    print(
        f"  {COMPARISON_FIGURE_PDF}"
    )

    print(
        "\nH4 endpoint coupling:"
    )

    print(
        f"  {H4_FIGURE_PNG}"
    )

    print(
        f"  {H4_FIGURE_PDF}"
    )

    print(
        "\nExplanation unavailability:"
    )

    print(
        f"  {H5_FIGURE_PNG}"
    )

    print(
        f"  {H5_FIGURE_PDF}"
    )

    # ---------------------------------------------------------------------
    # Final Phase-11 status.
    # ---------------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "=== Phase-11 Visualization Status ==="
    )

    print(
        "=" * 78
    )

    print(
        "\nFrozen Phase-10 results were successfully converted into "
        "research figures."
    )

    print(
        "\nNo experimental or statistical quantities were recomputed."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not treat successful file generation alone as final "
        "figure approval."
    )

    print(
        "The generated PNG figures should next be visually inspected "
        "for clipping, overlapping text, misleading axis scaling, "
        "legend placement, and readability."
    )

    print(
        "\nNext methodological step:"
    )

    print(
        "Open and inspect the four generated PNG figures. "
        "If their visual rendering is correct, record the Phase-11 "
        "visualization run in the experiment log and formally freeze "
        "the Phase-11 figure artifacts."
    )


if __name__ == "__main__":
    main()
