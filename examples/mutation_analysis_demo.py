"""
Demonstration of industry-standard mutation analysis with statistics and visualization.

This example shows how to:
1. Run mutations with statistical analysis
2. Visualize ddG results with error bars
3. Perform position scanning with heatmap visualization
4. Conduct alanine scanning with sorted bar charts

Note: Requires OpenMM and PDBFixer to run.
"""

import polars as pl
from sicifus import Sicifus, MutationEngine

# For this demo, you'll need a PDB file. Using a small protein like crambin (1CRN)
# Download from: https://files.rcsb.org/download/1CRN.pdb

def demo_statistical_mutation_analysis():
    """Demonstrate mutation analysis with full statistics."""
    print("=" * 70)
    print("DEMO 1: Single Mutation with Statistics")
    print("=" * 70)

    engine = MutationEngine()

    # Run mutation with statistics enabled
    result = engine.mutate(
        "1CRN.pdb",
        ["F13A"],
        n_runs=5,  # More runs for better statistics
        keep_statistics=True,
        use_mean=False,  # Use best energy (default)
        max_iterations=500
    )

    label = "F13A"
    print(f"\nMutation: {label}")
    print(f"Wild-type energy: {result.wt_energy:.2f} kcal/mol")
    print(f"Mutant energy (best): {result.mutant_energies[label]:.2f} kcal/mol")
    print(f"ddG (best-based): {result.ddg[label]:+.2f} kcal/mol")

    if result.mean_energy:
        print(f"\nStatistical Summary (n={len(result.all_run_energies[label])} runs):")
        print(f"  Mean energy: {result.mean_energy[label]:.2f} ± {result.sd_energy[label]:.2f} kcal/mol")
        print(f"  Range: [{result.min_energy[label]:.2f}, {result.max_energy[label]:.2f}]")
        print(f"  ddG (mean-based): {result.ddg_mean[label]:+.2f} ± {result.ddg_sd[label]:.2f} kcal/mol")
        print(f"  95% CI: [{result.ddg_ci_95[label][0]:+.2f}, {result.ddg_ci_95[label][1]:+.2f}]")
        print(f"  Convergence (CV): {result.convergence_metric[label]:.3f}")

    print("\nComplete\n")


def demo_batch_mutations_with_visualization():
    """Demonstrate batch mutations with ddG visualization."""
    print("=" * 70)
    print("DEMO 2: Batch Mutations with Visualization")
    print("=" * 70)

    engine = MutationEngine()

    # Create a batch of mutations
    mutations_df = pl.DataFrame({
        "mutation": ["F13A", "W14L", "I7V", "T1A", "C4S"],
        "chain": ["A"] * 5,
        "notes": ["Core", "Core", "Surface", "N-term", "Disulfide"],
    })

    print(f"\nRunning {len(mutations_df)} mutations...")
    results_df = engine.mutate_batch(
        "1CRN.pdb",
        mutations_df,
        n_runs=3,
        max_iterations=500,
        keep_statistics=True,
    )

    print("\nResults:")
    print(results_df.select(["mutation", "ddg_kcal_mol", "notes"]))

    # Visualize with error bars
    print("\nGenerating ddG plot with error bars...")
    from sicifus.visualization import plot_ddg

    plot_ddg(
        results_df,
        output_file="ddg_batch_results.png",
        show_error_bars=False,  # No SD in batch mode yet
        stability_threshold=1.0
    )

    print("Saved to ddg_batch_results.png\n")


def demo_position_scan_heatmap():
    """Demonstrate position scanning with heatmap visualization."""
    print("=" * 70)
    print("DEMO 3: Position Scan with Heatmap")
    print("=" * 70)

    engine = MutationEngine()

    # Scan a few positions (all 20 amino acids at each)
    print("\nScanning positions 13-15...")
    scan_df = engine.position_scan(
        "1CRN.pdb",
        chain="A",
        positions=[13, 14, 15],
        max_iterations=300,
        constrain_backbone=True,
    )

    print(f"Completed {len(scan_df)} mutations")
    print("\nSample results:")
    print(scan_df.head(10))

    # Create heatmap
    print("\nGenerating position scan heatmap...")
    from sicifus.visualization import plot_position_scan_heatmap

    plot_position_scan_heatmap(
        scan_df,
        output_file="position_scan_heatmap.png",
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0
    )

    print("Saved to position_scan_heatmap.png\n")


def demo_alanine_scan():
    """Demonstrate alanine scanning with sorted visualization."""
    print("=" * 70)
    print("DEMO 4: Alanine Scan")
    print("=" * 70)

    engine = MutationEngine()

    # Scan specific positions
    print("\nPerforming alanine scan on positions 13-17...")
    scan_df = engine.alanine_scan(
        "1CRN.pdb",
        chain="A",
        positions=[13, 14, 15, 16, 17],
        max_iterations=300,
    )

    print("\nResults:")
    print(scan_df.sort("ddg_kcal_mol", descending=True))

    # Visualize
    print("\nGenerating alanine scan plot...")
    from sicifus.visualization import plot_alanine_scan

    plot_alanine_scan(
        scan_df,
        output_file="alanine_scan_results.png",
        highlight_threshold=1.5
    )

    print("Saved to alanine_scan_results.png\n")


def demo_api_integration():
    """Demonstrate using the Sicifus API for mutation analysis."""
    print("=" * 70)
    print("DEMO 5: Sicifus API Integration")
    print("=" * 70)

    # Note: This assumes you have a Sicifus database with structure "1CRN"
    # If not, you can create one:
    # db = Sicifus("demo_db")
    # db.ingest("./pdb_files", protonate=True)

    print("\nUsing Sicifus API for integrated workflow...")
    db = Sicifus("demo_db")

    # Batch mutations
    mutations_df = pl.DataFrame({
        "mutation": ["F13A", "W14L"],
        "chain": ["A", "A"],
    })

    # Run mutations
    results_df = db.mutate_batch("1CRN", mutations_df, n_runs=3, keep_statistics=True)

    # Visualize using API method
    db.plot_mutation_results(results_df, output_file="api_ddg_plot.png", plot_type="ddg")

    print("Complete\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("industry-standard Mutation Analysis Demonstration")
    print("=" * 70 + "\n")

    try:
        # Run demos
        demo_statistical_mutation_analysis()
        demo_batch_mutations_with_visualization()
        # Uncomment to run position scan (takes longer):
        # demo_position_scan_heatmap()
        # demo_alanine_scan()
        # demo_api_integration()

        print("=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. OpenMM and PDBFixer installed: pip install sicifus[energy]")
        print("  2. A PDB file (e.g., 1CRN.pdb) in the current directory")
        raise
