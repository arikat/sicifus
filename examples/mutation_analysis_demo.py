"""
Demonstration of mutation analysis with experimental validation using Barnase.

This example shows how to:
1. Run mutations with statistical analysis
2. Compare predictions to experimental ΔΔG values from literature
3. Calculate performance metrics (R², RMSE, MAE)
4. Visualize results with publication-ready plots
5. Perform position scanning and alanine scanning

Test System: Barnase (1BNI)
- Small, stable bacterial ribonuclease
- Extensively studied for protein stability
- Rich experimental mutagenesis data (Serrano et al. 1993)

Experimental Data Source:
Serrano, L., Kellis, J.T., Cann, P., Matouschek, A., & Fersht, A.R. (1993).
"Step-wise Mutation of Barnase to Binase"
J. Mol. Biol. 233, 305-312.

Available experimental ΔΔG values from Table 3 (kcal/mol, pH 6.3, 25°C):
    Q15I: -0.96, T16R: -0.53, H18K: +1.19, K19R: -0.21,
    E29Q: +0.01, Q31S: +0.25, D44E: -0.08, I55V: +0.29,
    K62R: +0.48, G65S: -0.51, K66A: -0.25, T79V: -0.29,
    S85A: +0.12, I88L: +0.28, L89V: +0.27, Q104A: +0.21,
    K108R: -0.93

Note: Requires OpenMM and PDBFixer to run.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sicifus import MutationEngine
from sicifus.visualization import plot_ddg, plot_position_scan_heatmap, plot_alanine_scan


def download_barnase():
    """Download and clean Barnase structure if needed."""
    import urllib.request
    from pathlib import Path

    if not Path("1BNI.pdb").exists():
        print("Downloading Barnase structure (1BNI) from RCSB...")
        urllib.request.urlretrieve("https://files.rcsb.org/download/1BNI.pdb", "1BNI_raw.pdb")

        print("Cleaning structure (removing waters and heteroatoms)...")
        with open("1BNI_raw.pdb", "r") as f_in, open("1BNI.pdb", "w") as f_out:
            for line in f_in:
                # Keep only ATOM records (skip HETATM for waters, ligands)
                if line.startswith("ATOM"):
                    f_out.write(line)
                # Keep structural records
                elif line.startswith(("MODEL", "ENDMDL", "END", "TER")):
                    f_out.write(line)

        print("✅ Downloaded and cleaned 1BNI.pdb\n")
    else:
        print("Using existing 1BNI.pdb\n")


def demo_single_mutation_with_validation():
    """Demonstrate single mutation with experimental validation."""
    print("=" * 70)
    print("DEMO 1: Single Mutation with Experimental Validation")
    print("=" * 70)

    engine = MutationEngine()

    # Run mutation with statistics
    # H18K: His18 to Lys - located in first α-helix
    # Experimental ΔΔG = +1.19 kcal/mol (destabilizing)
    result = engine.mutate(
        "1BNI.pdb",
        ["H18K"],
        n_runs=5,
        keep_statistics=True,
        use_mean=False,
        max_iterations=2000
    )

    label = "H18K"
    print(f"\nMutation: {label} (His18 → Lys, α-helix 1)")
    print(f"Wild-type energy: {result.wt_energy:.2f} kcal/mol")
    print(f"Mutant energy (best): {result.mutant_energies[label]:.2f} kcal/mol")
    print(f"ΔΔG (best-based): {result.ddg[label]:+.2f} kcal/mol")

    if result.mean_energy:
        print(f"\nStatistical Summary (n={len(result.all_run_energies[label])} runs):")
        print(f"  Mean energy: {result.mean_energy[label]:.2f} ± {result.sd_energy[label]:.2f} kcal/mol")
        print(f"  ΔΔG (mean-based): {result.ddg_mean[label]:+.2f} ± {result.ddg_sd[label]:.2f} kcal/mol")
        print(f"  95% CI: [{result.ddg_ci_95[label][0]:+.2f}, {result.ddg_ci_95[label][1]:+.2f}]")
        print(f"  Convergence (CV): {result.convergence_metric[label]:.3f}")

    # Compare to experiment (Serrano et al. 1993, Table 3)
    experimental_ddg = 1.19
    predicted_ddg = result.ddg[label]
    error = predicted_ddg - experimental_ddg

    print(f"\nComparison to Experiment (Serrano et al. 1993):")
    print(f"  Experimental ΔΔG: {experimental_ddg:+.2f} kcal/mol (destabilizing)")
    print(f"  Predicted ΔΔG:    {predicted_ddg:+.2f} kcal/mol")
    print(f"  Error:            {error:+.2f} kcal/mol")
    print(f"  Absolute error:   {abs(error):.2f} kcal/mol")
    print("\nComplete\n")


def demo_batch_mutations_with_validation():
    """Demonstrate batch mutations with experimental validation."""
    print("=" * 70)
    print("DEMO 2: Batch Mutations with Experimental Validation")
    print("=" * 70)

    engine = MutationEngine()

    # Define mutations with experimental data from Serrano et al. (1993)
    # Source: J. Mol. Biol. (1993) 233, 305-312, Table 3
    # These are ΔΔG values from urea denaturation at pH 6.3, 25°C
    mutations_df = pl.DataFrame({
        "mutation": ["H18K", "I55V", "K62R", "K66A", "T79V", "S85A", "I88L", "L89V"],
        "chain": ["A"] * 8,
        "experimental_ddg": [1.19, 0.29, 0.48, -0.25, -0.29, 0.12, 0.28, 0.27],
        "notes": [
            "His to Lys - α-helix 1",
            "Ile to Val - smaller hydrophobic",
            "Lys to Arg - surface charge",
            "Lys to Ala - α-helix 2 (stabilizing)",
            "Thr to Val - α-helix 2 (stabilizing)",
            "Ser to Ala - loop",
            "Ile to Leu - conservative surface",
            "Leu to Val - smaller hydrophobic"
        ]
    })

    print(f"\nRunning {len(mutations_df)} mutations with known experimental values...")
    print("Source: Serrano et al. (1993) J Mol Biol 233:305-312, Table 3\n")

    results_df = engine.mutate_batch(
        "1BNI.pdb",
        mutations_df,
        n_runs=3,
        max_iterations=2000,
        keep_statistics=True,
    )

    # Add predicted_ddg column
    results_df = results_df.with_columns(
        pl.col("ddg_kcal_mol").alias("predicted_ddg")
    )

    # Filter valid results
    valid_results = results_df.filter(~pl.col("predicted_ddg").is_nan())

    # Calculate metrics
    experimental = valid_results["experimental_ddg"].to_numpy()
    predicted = valid_results["predicted_ddg"].to_numpy()

    r_squared = stats.pearsonr(experimental, predicted)[0] ** 2
    rmse = np.sqrt(np.mean((predicted - experimental) ** 2))
    mae = np.mean(np.abs(predicted - experimental))
    pearson_r, pearson_p = stats.pearsonr(experimental, predicted)

    print("\n" + "=" * 70)
    print("PREDICTION PERFORMANCE")
    print("=" * 70)
    print(f"Sample size:       {len(experimental)} mutations")
    print(f"R² (R-squared):    {r_squared:.3f}")
    print(f"RMSE:              {rmse:.2f} kcal/mol")
    print(f"MAE:               {mae:.2f} kcal/mol")
    print(f"Pearson r:         {pearson_r:.3f} (p = {pearson_p:.2e})")
    print("=" * 70)

    # Detailed comparison
    comparison_df = valid_results.with_columns(
        (pl.col("predicted_ddg") - pl.col("experimental_ddg")).alias("error"),
        (pl.col("predicted_ddg") - pl.col("experimental_ddg")).abs().alias("abs_error")
    )

    print("\nDetailed Comparison:")
    print(comparison_df.select([
        "mutation", "experimental_ddg", "predicted_ddg", "error", "abs_error"
    ]).sort("abs_error", descending=True))

    # Save results
    comparison_df.write_csv("barnase_validation_results.csv")
    print("\nResults saved to barnase_validation_results.csv")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(experimental, predicted, alpha=0.7, s=150,
               edgecolors='black', linewidth=1.5, c='steelblue')

    # Perfect prediction line
    min_val = min(experimental.min(), predicted.min()) - 0.5
    max_val = max(experimental.max(), predicted.max()) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.5, linewidth=2, label='Perfect prediction')

    # Linear fit
    slope, intercept = np.polyfit(experimental, predicted, 1)
    fit_x = np.linspace(min_val, max_val, 100)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, 'r-', alpha=0.7, linewidth=2,
            label=f'Linear fit (y = {slope:.2f}x + {intercept:+.2f})')

    ax.set_xlabel('Experimental ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_title('Barnase Mutations: Predicted vs Experimental',
                 fontsize=14, fontweight='bold', pad=15)

    stats_text = (
        f"n = {len(experimental)}\n"
        f"R² = {r_squared:.3f}\n"
        f"RMSE = {rmse:.2f} kcal/mol\n"
        f"MAE = {mae:.2f} kcal/mol"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig("barnase_validation.png", dpi=300, bbox_inches='tight')
    print("Scatter plot saved to barnase_validation.png")

    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    mutations = comparison_df["mutation"].to_list()
    exp_values = comparison_df["experimental_ddg"].to_numpy()
    pred_values = comparison_df["predicted_ddg"].to_numpy()

    x = np.arange(len(mutations))
    width = 0.35

    ax.bar(x - width/2, exp_values, width, label='Experimental',
           color='#2ca02c', edgecolor='black', linewidth=1)
    ax.bar(x + width/2, pred_values, width, label='Predicted (Sicifus)',
           color='#1f77b4', edgecolor='black', linewidth=1)

    ax.set_xlabel('Mutation', fontsize=12, fontweight='bold')
    ax.set_ylabel('ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_title('Barnase Mutations: Experimental vs Predicted ΔΔG',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mutations, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("barnase_comparison.png", dpi=300, bbox_inches='tight')
    print("Comparison plot saved to barnase_comparison.png")

    print("\nComplete\n")


def demo_position_scan():
    """Demonstrate position scanning."""
    print("=" * 70)
    print("DEMO 3: Position Scan (Computationally Expensive)")
    print("=" * 70)

    engine = MutationEngine()

    print("\nScanning positions 24, 35, 56 (all 20 amino acids each)...")
    print("This will take several minutes...\n")

    scan_df = engine.position_scan(
        "1BNI.pdb",
        chain="A",
        positions=[24, 35, 56],
        max_iterations=1500,
        constrain_backbone=True,
    )

    print(f"\nCompleted {len(scan_df)} mutations")
    print("\nSample results:")
    print(scan_df.head(10))

    plot_position_scan_heatmap(
        scan_df,
        output_file="barnase_position_scan.png",
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=3.0
    )

    print("\nHeatmap saved to barnase_position_scan.png")
    print("\nComplete\n")


def demo_alanine_scan():
    """Demonstrate alanine scanning."""
    print("=" * 70)
    print("DEMO 4: Alanine Scan")
    print("=" * 70)

    engine = MutationEngine()

    print("\nPerforming alanine scan on positions 20-30...")
    scan_df = engine.alanine_scan(
        "1BNI.pdb",
        chain="A",
        positions=list(range(20, 31)),
        max_iterations=1500,
    )

    print("\nResults (sorted by ΔΔG):")
    print(scan_df.sort("ddg_kcal_mol", descending=True))

    plot_alanine_scan(
        scan_df,
        output_file="barnase_alanine_scan.png",
        highlight_threshold=1.5
    )

    print("\nAlanine scan plot saved to barnase_alanine_scan.png")
    print("\nComplete\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Barnase Mutation Analysis with Experimental Validation")
    print("=" * 70 + "\n")

    try:
        # Download structure
        download_barnase()

        # Run demos
        demo_single_mutation_with_validation()
        demo_batch_mutations_with_validation()

        # Uncomment to run position scan (takes longer):
        # demo_position_scan()
        # demo_alanine_scan()

        print("=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. OpenMM and PDBFixer installed: pip install sicifus[energy]")
        print("  2. Internet connection to download 1BNI.pdb")
        raise
