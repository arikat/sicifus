"""
Visualization functions for mutation analysis results.

Provides publication-ready plots for ddG values, energy term breakdowns,
position scans, and alanine scans following matplotlib conventions.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from .mutate import ALL_AMINO_ACIDS


def plot_ddg(
    results_df: pl.DataFrame,
    output_file: Optional[str] = None,
    show_error_bars: bool = True,
    stability_threshold: float = 1.0,
    figsize: Tuple[int, int] = (12, 6),
) -> pl.DataFrame:
    """Plot ddG values as sorted bar chart with stability color-coding.

    Args:
        results_df: DataFrame with columns [mutation, ddg_kcal_mol], optionally [ddg_sd]
        output_file: Path to save figure (if None, shows interactively)
        show_error_bars: Include SD/CI error bars if available
        stability_threshold: ddG threshold for destabilizing (red) vs stabilizing (green)
        figsize: Figure size

    Returns:
        Sorted DataFrame used for plotting
    """
    # Validate required columns
    if "mutation" not in results_df.columns and "ddg_kcal_mol" not in results_df.columns:
        raise ValueError("DataFrame must contain 'mutation' and 'ddg_kcal_mol' columns")

    # Sort by ddG
    sorted_df = results_df.sort("ddg_kcal_mol")

    # Extract data
    mutations = sorted_df["mutation"].to_list()
    ddg_values = sorted_df["ddg_kcal_mol"].to_numpy()

    # Color-code by stability
    colors = ["#d62728" if ddg > stability_threshold else
              "#2ca02c" if ddg < -stability_threshold else
              "#7f7f7f" for ddg in ddg_values]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    y_pos = np.arange(len(mutations))
    bars = ax.barh(y_pos, ddg_values, color=colors, edgecolor="black", linewidth=0.5)

    # Add error bars if available and requested
    if show_error_bars and "ddg_sd" in sorted_df.columns:
        sd_values = sorted_df["ddg_sd"].to_numpy()
        ax.errorbar(ddg_values, y_pos, xerr=sd_values, fmt='none',
                    ecolor='black', capsize=3, linewidth=1)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mutations, fontsize=9)
    ax.set_xlabel("ddG (kcal/mol)", fontsize=12)
    ax.set_ylabel("Mutation", fontsize=12)
    ax.set_title("Mutation Stability Analysis", fontsize=14, fontweight="bold")
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label=f'Destabilizing (>{stability_threshold})'),
        Patch(facecolor='#2ca02c', edgecolor='black', label=f'Stabilizing (<{-stability_threshold})'),
        Patch(facecolor='#7f7f7f', edgecolor='black', label='Neutral')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return sorted_df


def plot_energy_terms(
    energy_terms_df: pl.DataFrame,
    output_file: Optional[str] = None,
    plot_type: str = "grouped",
    figsize: Tuple[int, int] = (10, 6),
) -> pl.DataFrame:
    """Plot energy term breakdown for WT vs mutant.

    Args:
        energy_terms_df: DataFrame from MutationResult.energy_terms
                        with columns [term, wt_energy, mutant_energy, delta]
        output_file: Path to save figure
        plot_type: "grouped" (side-by-side) or "stacked" (cumulative)
        figsize: Figure size

    Returns:
        Processed DataFrame with term contributions
    """
    # Filter out total (plot it separately if needed)
    terms_df = energy_terms_df.filter(pl.col("term") != "total")

    terms = terms_df["term"].to_list()
    wt_energies = terms_df["wt_energy"].to_numpy()
    mut_energies = terms_df["mutant_energy"].to_numpy()

    x = np.arange(len(terms))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "grouped":
        # Side-by-side bars
        ax.bar(x - width/2, wt_energies, width, label='Wild-type',
               color='#1f77b4', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, mut_energies, width, label='Mutant',
               color='#ff7f0e', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Energy Term', fontsize=12)
        ax.set_ylabel('Energy (kcal/mol)', fontsize=12)
        ax.set_title('Energy Term Breakdown: WT vs Mutant', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(terms, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    elif plot_type == "stacked":
        # Stacked bars showing delta contributions
        deltas = terms_df["delta"].to_numpy()
        positive_mask = deltas > 0
        negative_mask = deltas < 0

        # Plot positive and negative separately for color-coding
        if np.any(positive_mask):
            ax.bar(x[positive_mask], deltas[positive_mask], color='#d62728',
                   edgecolor='black', linewidth=0.5, label='Destabilizing')
        if np.any(negative_mask):
            ax.bar(x[negative_mask], deltas[negative_mask], color='#2ca02c',
                   edgecolor='black', linewidth=0.5, label='Stabilizing')

        ax.set_xlabel('Energy Term', fontsize=12)
        ax.set_ylabel('Δ Energy (kcal/mol)', fontsize=12)
        ax.set_title('Energy Term Changes (Mutant - WT)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(terms, rotation=45, ha='right', fontsize=9)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    else:
        raise ValueError(f"plot_type must be 'grouped' or 'stacked', got '{plot_type}'")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return terms_df


def plot_position_scan_heatmap(
    scan_df: pl.DataFrame,
    output_file: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: float = -2.0,
    vmax: float = 2.0,
    figsize: Tuple[int, int] = (15, 8),
) -> pl.DataFrame:
    """Plot position scan as heatmap (20 amino acids × N positions).

    Args:
        scan_df: DataFrame from position_scan() with [position, wt_residue, mut_residue, ddg_kcal_mol]
        output_file: Path to save figure
        cmap: Colormap (diverging recommended)
        vmin/vmax: Color scale limits
        figsize: Figure size

    Returns:
        Pivoted DataFrame (rows=amino acids, cols=positions)
    """
    # Get unique positions
    positions = sorted(scan_df["position"].unique().to_list())

    # Create a matrix: rows = amino acids, cols = positions
    matrix = np.full((len(ALL_AMINO_ACIDS), len(positions)), np.nan)

    # Map amino acid 3-letter to index
    from .mutate import THREE_TO_ONE
    aa_to_idx = {aa: i for i, aa in enumerate(ALL_AMINO_ACIDS)}

    # Fill the matrix
    for row in scan_df.iter_rows(named=True):
        pos = row["position"]
        mut_res = row["mut_residue"]
        ddg = row["ddg_kcal_mol"]

        if mut_res in aa_to_idx:
            aa_idx = aa_to_idx[mut_res]
            pos_idx = positions.index(pos)
            matrix[aa_idx, pos_idx] = ddg

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                   interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='ddG (kcal/mol)')

    # Axis labels
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{p}" for p in positions], fontsize=9)
    ax.set_yticks(range(len(ALL_AMINO_ACIDS)))
    ax.set_yticklabels([f"{aa} ({THREE_TO_ONE[aa]})" for aa in ALL_AMINO_ACIDS], fontsize=9)

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Mutant Residue', fontsize=12)
    ax.set_title('Position Scan Heatmap', fontsize=14, fontweight='bold')

    # Mark wild-type residues
    wt_map = scan_df.select(["position", "wt_residue"]).unique().to_dict()
    wt_by_pos = {wt_map["position"][i]: wt_map["wt_residue"][i] for i in range(len(wt_map["position"]))}

    for pos_idx, pos in enumerate(positions):
        wt_res = wt_by_pos.get(pos)
        if wt_res and wt_res in aa_to_idx:
            aa_idx = aa_to_idx[wt_res]
            # Mark WT with a box
            rect = plt.Rectangle((pos_idx - 0.5, aa_idx - 0.5), 1, 1,
                                fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return pivoted DataFrame
    pivot_data = []
    for aa_idx, aa in enumerate(ALL_AMINO_ACIDS):
        row_dict = {"amino_acid": aa}
        for pos_idx, pos in enumerate(positions):
            row_dict[f"pos_{pos}"] = matrix[aa_idx, pos_idx]
        pivot_data.append(row_dict)

    return pl.DataFrame(pivot_data)


def plot_alanine_scan(
    scan_df: pl.DataFrame,
    output_file: Optional[str] = None,
    highlight_threshold: float = 1.5,
    figsize: Tuple[int, int] = (12, 6),
) -> pl.DataFrame:
    """Plot alanine scan results as sorted bar chart.

    Args:
        scan_df: DataFrame from alanine_scan() with [position, wt_residue, ddg_kcal_mol]
        output_file: Path to save figure
        highlight_threshold: Highlight positions with |ddG| > threshold
        figsize: Figure size

    Returns:
        Sorted DataFrame
    """
    # Sort by ddG
    sorted_df = scan_df.sort("ddg_kcal_mol")

    # Extract data
    positions = sorted_df["position"].to_list()
    wt_residues = sorted_df["wt_residue"].to_list()
    ddg_values = sorted_df["ddg_kcal_mol"].to_numpy()

    # Create labels with position and WT residue
    from .mutate import THREE_TO_ONE
    labels = [f"{THREE_TO_ONE[wt]}{pos}" for pos, wt in zip(positions, wt_residues)]

    # Color-code by magnitude
    colors = []
    for ddg in ddg_values:
        if abs(ddg) > highlight_threshold:
            colors.append('#d62728' if ddg > 0 else '#2ca02c')
        else:
            colors.append('#7f7f7f')

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, ddg_values, color=colors, edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('ddG (kcal/mol)', fontsize=12)
    ax.set_ylabel('Position (WT residue)', fontsize=12)
    ax.set_title('Alanine Scan Results', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label=f'Hotspot destabilizing (|ddG|>{highlight_threshold})'),
        Patch(facecolor='#2ca02c', edgecolor='black', label=f'Hotspot stabilizing (|ddG|>{highlight_threshold})'),
        Patch(facecolor='#7f7f7f', edgecolor='black', label='Minor effect')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return sorted_df


def plot_ddg_distribution(
    results_df: pl.DataFrame,
    output_file: Optional[str] = None,
    bins: int = 20,
    show_stats: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> pl.DataFrame:
    """Plot ddG distribution histogram with statistical overlay.

    Args:
        results_df: DataFrame with ddg_kcal_mol column
        output_file: Path to save figure
        bins: Number of histogram bins
        show_stats: Annotate with mean, median, SD
        figsize: Figure size

    Returns:
        DataFrame with histogram bins and counts
    """
    ddg_values = results_df["ddg_kcal_mol"].to_numpy()

    # Compute statistics
    mean_ddg = float(np.mean(ddg_values))
    median_ddg = float(np.median(ddg_values))
    std_ddg = float(np.std(ddg_values))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    counts, bin_edges, patches = ax.hist(ddg_values, bins=bins, color='#1f77b4',
                                         edgecolor='black', linewidth=0.5, alpha=0.7)

    # Color-code bins
    for i, patch in enumerate(patches):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        if bin_center > 1.0:
            patch.set_facecolor('#d62728')
        elif bin_center < -1.0:
            patch.set_facecolor('#2ca02c')

    # Add vertical lines for mean and median
    ax.axvline(mean_ddg, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_ddg:.2f}')
    ax.axvline(median_ddg, color='orange', linestyle=':', linewidth=2, label=f'Median = {median_ddg:.2f}')

    # Formatting
    ax.set_xlabel('ddG (kcal/mol)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('ddG Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Statistics text box
    if show_stats:
        stats_text = f'n = {len(ddg_values)}\nμ = {mean_ddg:.2f}\nσ = {std_ddg:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return histogram data
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))]
    hist_df = pl.DataFrame({
        "bin_center": bin_centers,
        "count": counts,
    })

    return hist_df
