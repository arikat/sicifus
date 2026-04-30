"""
Generate and analyze a library of Barnase variants.

This example demonstrates:
1. Systematic generation of single-point mutations
2. High-throughput ΔΔG prediction (1000+ variants)
3. Ranking mutations by stability
4. Identifying stabilizing/destabilizing hotspots
5. Exporting top variants for further analysis

Use case: Protein engineering for increased stability
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sicifus import MutationEngine
from pathlib import Path
import urllib.request


def download_barnase():
    """Download and clean Barnase structure."""
    if not Path("1BNI.pdb").exists():
        print("Downloading Barnase (1BNI)...")
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/1BNI.pdb",
            "1BNI_raw.pdb"
        )

        # Clean structure
        with open("1BNI_raw.pdb", "r") as f_in, open("1BNI.pdb", "w") as f_out:
            for line in f_in:
                if line.startswith("ATOM"):
                    f_out.write(line)
                elif line.startswith(("MODEL", "ENDMDL", "END", "TER")):
                    f_out.write(line)
        print("✅ Downloaded 1BNI.pdb\n")
    else:
        print("Using existing 1BNI.pdb\n")


def get_barnase_sequence():
    """Get Barnase sequence from structure."""
    # Barnase sequence (110 residues, but crystal starts at residue 3)
    # Actual sequence in crystal: positions 3-110
    return "AKVYNTGIKGVIPEDLLTGRPEWIWLDRDELNVVIEQGYNDISIGNRKILANQYQDITAKRQGVTFQGQILIQNHPFVTPRTPPPSQGKGRPQSNGAKGPGSP"


def generate_all_single_mutants():
    """Generate all possible single-point mutations for Barnase."""
    print("=" * 70)
    print("Generating Barnase Variant Library")
    print("=" * 70)

    sequence = get_barnase_sequence()
    mutations = []

    # Barnase crystal structure starts at residue 3 (not 1!)
    # Positions 3-110 in PDB numbering
    start_residue = 3

    amino_acids = "ARNDCQEGHILKMFPSTWYV"

    for seq_idx, wt_aa in enumerate(sequence):
        pdb_position = start_residue + seq_idx

        for mut_aa in amino_acids:
            if mut_aa == wt_aa:
                continue  # Skip WT

            mutations.append({
                "mutation": f"{wt_aa}{pdb_position}{mut_aa}",
                "chain": "A",
                "position": pdb_position,
                "wt_residue": wt_aa,
                "mut_residue": mut_aa,
            })

    mutations_df = pl.DataFrame(mutations)

    print(f"\nGenerated {len(mutations_df)} single-point mutations:")
    print(f"  Sequence length: {len(sequence)} residues")
    print(f"  PDB positions: {start_residue} to {start_residue + len(sequence) - 1}")
    print(f"  19 mutations/position × {len(sequence)} = {len(mutations_df)} total")
    print(f"  (excluding WT at each position)")

    return mutations_df


def run_stability_screen(mutations_df, fast_mode=True):
    """
    Run high-throughput stability screening.

    Args:
        mutations_df: DataFrame with mutations
        fast_mode: If True, use n_runs=1 for speed. If False, use n_runs=3 for statistics.
    """
    print("\n" + "=" * 70)
    print("Running High-Throughput Stability Screen")
    print("=" * 70)

    engine = MutationEngine()

    n_runs = 1 if fast_mode else 3
    max_iter = 500 if fast_mode else 2000

    print(f"\nSettings:")
    print(f"  Fast mode: {fast_mode}")
    print(f"  Runs per mutation: {n_runs}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Total mutations: {len(mutations_df)}")

    if fast_mode:
        print(f"\n⚠️  Fast mode: Results are approximate (single minimization)")
        print(f"   Use fast_mode=False for production runs with statistics")

    estimated_time = len(mutations_df) * 30 / 60  # ~30 sec per mutation
    print(f"\n⏱️  Estimated time: {estimated_time:.1f} minutes")

    response = input(f"\nProceed with {len(mutations_df)} mutations? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return None

    print("\nRunning mutations...")
    results = engine.mutate_batch(
        "1BNI.pdb",
        mutations_df,
        n_runs=n_runs,
        max_iterations=max_iter,
        keep_statistics=(n_runs > 1)
    )

    print(f"\n✅ Completed {len(results)} mutations")

    # Filter successful mutations
    valid_results = results.filter(~pl.col("ddg_kcal_mol").is_nan())
    failed_results = results.filter(pl.col("ddg_kcal_mol").is_nan())

    print(f"  ✅ Successful: {len(valid_results)}")
    if len(failed_results) > 0:
        print(f"  ❌ Failed: {len(failed_results)}")

    return valid_results


def analyze_variant_library(results_df):
    """Analyze the variant library results."""
    print("\n" + "=" * 70)
    print("Variant Library Analysis")
    print("=" * 70)

    # Summary statistics
    ddg = results_df["ddg_kcal_mol"].to_numpy()

    print(f"\nΔΔG Distribution:")
    print(f"  Mean:   {np.mean(ddg):+.2f} kcal/mol")
    print(f"  Median: {np.median(ddg):+.2f} kcal/mol")
    print(f"  Std:    {np.std(ddg):.2f} kcal/mol")
    print(f"  Min:    {np.min(ddg):+.2f} kcal/mol (most stabilizing)")
    print(f"  Max:    {np.max(ddg):+.2f} kcal/mol (most destabilizing)")

    # Classification
    stabilizing = results_df.filter(pl.col("ddg_kcal_mol") < -0.5)
    neutral = results_df.filter(
        (pl.col("ddg_kcal_mol") >= -0.5) & (pl.col("ddg_kcal_mol") <= 0.5)
    )
    destabilizing = results_df.filter(pl.col("ddg_kcal_mol") > 0.5)

    print(f"\nClassification (threshold = ±0.5 kcal/mol):")
    print(f"  Stabilizing:     {len(stabilizing):4d} ({100*len(stabilizing)/len(results_df):.1f}%)")
    print(f"  Neutral:         {len(neutral):4d} ({100*len(neutral)/len(results_df):.1f}%)")
    print(f"  Destabilizing:   {len(destabilizing):4d} ({100*len(destabilizing)/len(results_df):.1f}%)")

    # Top stabilizing
    print(f"\n🌟 Top 10 Stabilizing Mutations:")
    top_stabilizing = results_df.sort("ddg_kcal_mol").head(10)
    for row in top_stabilizing.iter_rows(named=True):
        print(f"  {row['mutation']:6s}: ΔΔG = {row['ddg_kcal_mol']:+6.2f} kcal/mol")

    # Top destabilizing
    print(f"\n💥 Top 10 Destabilizing Mutations:")
    top_destabilizing = results_df.sort("ddg_kcal_mol", descending=True).head(10)
    for row in top_destabilizing.iter_rows(named=True):
        print(f"  {row['mutation']:6s}: ΔΔG = {row['ddg_kcal_mol']:+6.2f} kcal/mol")

    return {
        "stabilizing": stabilizing,
        "neutral": neutral,
        "destabilizing": destabilizing,
        "top_stabilizing": top_stabilizing,
        "top_destabilizing": top_destabilizing,
    }


def visualize_results(results_df):
    """Create visualization of variant library."""
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. ΔΔG distribution
    ax = axes[0, 0]
    ddg = results_df["ddg_kcal_mol"].to_numpy()
    ax.hist(ddg, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ΔΔG=0 (neutral)')
    ax.axvline(-0.5, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Stabilizing')
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Destabilizing')
    ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('ΔΔG Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. ΔΔG by position (average)
    ax = axes[0, 1]
    position_avg = results_df.group_by("position").agg(
        pl.col("ddg_kcal_mol").mean().alias("avg_ddg")
    ).sort("position")

    positions = position_avg["position"].to_numpy()
    avg_ddg = position_avg["avg_ddg"].to_numpy()

    colors = ['green' if x < -0.5 else 'orange' if x > 0.5 else 'gray' for x in avg_ddg]
    ax.bar(positions, avg_ddg, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_title('ΔΔG by Position (averaged over 19 mutations)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 3. Mutation type preferences
    ax = axes[1, 0]
    mut_type_avg = results_df.group_by("mut_residue").agg(
        pl.col("ddg_kcal_mol").mean().alias("avg_ddg"),
        pl.col("ddg_kcal_mol").count().alias("count")
    ).sort("avg_ddg")

    aa_order = mut_type_avg["mut_residue"].to_list()
    aa_avg = mut_type_avg["avg_ddg"].to_numpy()

    colors = ['green' if x < -0.2 else 'orange' if x > 0.2 else 'gray' for x in aa_avg]
    ax.barh(aa_order, aa_avg, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Average ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mutant Residue Type', fontsize=11, fontweight='bold')
    ax.set_title('Mutation Target Preferences', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 4. Cumulative ΔΔG
    ax = axes[1, 1]
    sorted_ddg = np.sort(ddg)
    cumulative = np.arange(1, len(sorted_ddg) + 1) / len(sorted_ddg) * 100

    ax.plot(sorted_ddg, cumulative, linewidth=2, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ΔΔG=0')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative ΔΔG Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("barnase_variant_library.png", dpi=300, bbox_inches='tight')
    print("\n✅ Saved barnase_variant_library.png")


def export_top_variants(results_df, n_top=20):
    """Export top stabilizing variants for further analysis."""
    print("\n" + "=" * 70)
    print("Exporting Top Variants")
    print("=" * 70)

    # Top stabilizing
    top_stabilizing = results_df.sort("ddg_kcal_mol").head(n_top)
    top_stabilizing.write_csv("top_stabilizing_variants.csv")
    print(f"\n✅ Saved top {n_top} stabilizing mutations to:")
    print(f"   top_stabilizing_variants.csv")

    # Top destabilizing (for understanding what NOT to do)
    top_destabilizing = results_df.sort("ddg_kcal_mol", descending=True).head(n_top)
    top_destabilizing.write_csv("top_destabilizing_variants.csv")
    print(f"\n✅ Saved top {n_top} destabilizing mutations to:")
    print(f"   top_destabilizing_variants.csv")

    # Full results
    results_df.write_csv("barnase_variant_library_full.csv")
    print(f"\n✅ Saved all {len(results_df)} results to:")
    print(f"   barnase_variant_library_full.csv")

    # Summary for engineering
    print(f"\n📊 Engineering Summary:")
    print(f"   Load results: df = pl.read_csv('barnase_variant_library_full.csv')")
    print(f"   Filter: df.filter(pl.col('ddg_kcal_mol') < -1.0)")
    print(f"   Combine: Design combinatorial library from top hits")


def demo_small_library():
    """Demo mode: Test on 5 positions × 19 mutations = 95 mutations."""
    print("=" * 70)
    print("DEMO MODE: Small Variant Library (5 positions)")
    print("=" * 70)

    sequence = get_barnase_sequence()
    mutations = []

    # Just 5 positions for demo
    test_positions = [10, 20, 30, 40, 50]  # PDB numbering
    amino_acids = "ARNDCQEGHILKMFPSTWYV"

    for pdb_pos in test_positions:
        seq_idx = pdb_pos - 3  # Convert PDB to sequence index
        wt_aa = sequence[seq_idx]

        for mut_aa in amino_acids:
            if mut_aa == wt_aa:
                continue

            mutations.append({
                "mutation": f"{wt_aa}{pdb_pos}{mut_aa}",
                "chain": "A",
                "position": pdb_pos,
                "wt_residue": wt_aa,
                "mut_residue": mut_aa,
            })

    mutations_df = pl.DataFrame(mutations)
    print(f"\nGenerated {len(mutations_df)} mutations for demo")
    print(f"Positions: {test_positions}")

    # Run fast screen
    results = run_stability_screen(mutations_df, fast_mode=True)

    if results is not None:
        # Analyze
        analysis = analyze_variant_library(results)

        # Visualize
        visualize_results(results)

        # Export
        export_top_variants(results, n_top=10)

        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run full library: demo_full_library()")
        print("  2. Examine plots: barnase_variant_library.png")
        print("  3. Review top hits: top_stabilizing_variants.csv")


def demo_full_library():
    """Full library: All positions × 19 mutations = ~2000 mutations."""
    print("=" * 70)
    print("FULL LIBRARY MODE")
    print("=" * 70)
    print("\n⚠️  This will run ~2000 mutations!")
    print("   Estimated time: 15-30 minutes (fast mode)")
    print("   For production: 1-3 hours (n_runs=3)\n")

    mutations_df = generate_all_single_mutants()

    # Run screen
    results = run_stability_screen(mutations_df, fast_mode=True)

    if results is not None:
        # Analyze
        analysis = analyze_variant_library(results)

        # Visualize
        visualize_results(results)

        # Export
        export_top_variants(results, n_top=50)

        print("\n" + "=" * 70)
        print("Full Library Complete!")
        print("=" * 70)
        print("\n🎉 You now have a complete mutational landscape of Barnase!")
        print("\nFiles generated:")
        print("  - barnase_variant_library.png (plots)")
        print("  - barnase_variant_library_full.csv (all results)")
        print("  - top_stabilizing_variants.csv (best mutations)")
        print("  - top_destabilizing_variants.csv (worst mutations)")
        print("\nNext: Combine stabilizing mutations for synergistic effects!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Barnase Variant Library Generation")
    print("=" * 70 + "\n")

    # Download structure
    download_barnase()

    # Choose mode
    print("Select mode:")
    print("  1. Demo (5 positions × 19 = 95 mutations, ~5 min)")
    print("  2. Full library (108 positions × 19 = ~2000 mutations, ~30 min)")

    choice = input("\nChoice [1/2]: ").strip()

    if choice == "1":
        demo_small_library()
    elif choice == "2":
        demo_full_library()
    else:
        print("Invalid choice. Run again and select 1 or 2.")
