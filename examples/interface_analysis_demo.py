"""
Demonstration of advanced mutation analysis features:
1. Mutation-to-Binding Pipeline (ΔΔG_binding)
2. Disulfide Bond Analysis
3. Residue Interaction Networks

Test System: Barnase-Barstar Complex (1BRS)
- Well-characterized protein-protein interface
- Barnase (chain A) + Barstar (chain B)
- Experimental interface mutation data available

Note: Requires OpenMM and PDBFixer to run the mutation/binding examples.
"""

import polars as pl
import urllib.request
from pathlib import Path
from sicifus import Sicifus, MutationEngine
from sicifus.analysis import AnalysisToolkit


def download_and_clean_complex(pdb_id: str = "1BRS"):
    """Download and clean Barnase-Barstar complex."""
    pdb_file = f"{pdb_id}.pdb"

    if not Path(pdb_file).exists():
        print(f"Downloading {pdb_id} (Barnase-Barstar complex) from RCSB...")
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            f"{pdb_id}_raw.pdb"
        )

        print("Cleaning structure (removing waters and heteroatoms)...")
        with open(f"{pdb_id}_raw.pdb", "r") as f_in, open(pdb_file, "w") as f_out:
            for line in f_in:
                if line.startswith("ATOM"):
                    f_out.write(line)
                elif line.startswith(("MODEL", "ENDMDL", "END", "TER")):
                    f_out.write(line)

        print(f"✅ Downloaded and cleaned {pdb_file}\n")
    else:
        print(f"Using existing {pdb_file}\n")

    return pdb_file


def setup_database():
    """Create a small Sicifus database for network analysis demo."""
    db_path = "interface_demo_db"
    pdb_file = download_and_clean_complex("1BRS")

    if not Path(db_path).exists():
        print(f"Creating Sicifus database: {db_path}")
        db = Sicifus(db_path)

        # Create a temporary directory with just our structure
        import os
        os.makedirs("temp_pdbs", exist_ok=True)
        import shutil
        shutil.copy(pdb_file, f"temp_pdbs/{pdb_file}")

        db.ingest("temp_pdbs", batch_size=10, file_extension="pdb")
        print(f"✅ Database created\n")

        # Cleanup
        shutil.rmtree("temp_pdbs")
    else:
        print(f"Using existing database: {db_path}\n")

    return db_path


# ---------------------------------------------------------------------------
# Feature 1: Mutation-to-Binding Pipeline
# ---------------------------------------------------------------------------

def demo_interface_mutagenesis():
    """Demonstrate automated interface mutation analysis."""
    print("=" * 70)
    print("DEMO 1: Interface Mutagenesis (ΔΔG_binding)")
    print("=" * 70)

    pdb_file = download_and_clean_complex("1BRS")
    engine = MutationEngine()

    # Barnase-Barstar complex (1BRS)
    # Chain A = Barnase (ribonuclease)
    # Chain B = Barstar (inhibitor)
    # Well-studied interface with experimental data

    print("\nRunning interface mutation analysis...")
    print("Mutation: R59A on Barnase (chain A)")
    print("This is a key interface residue with known destabilizing effect\n")

    result = engine.mutate_interface(
        pdb_file,
        mutations={
            "A": ["R59A"],  # Arg59 on Barnase - critical interface residue
        },
        chains_a=["A"],  # Barnase
        chains_b=["B"],  # Barstar
        max_iterations=1000,
        n_runs=1  # Use 1 run for speed in demo
    )

    print(f"\nResults:")
    print(f"  WT binding energy:     {result.wt_binding_energy:.2f} kcal/mol")
    print(f"  Mutant binding energy: {result.mutant_binding_energy:.2f} kcal/mol")
    print(f"  ΔΔG_binding:           {result.ddg_binding:+.2f} kcal/mol")
    print()
    print(f"  WT complex energy:     {result.wt_complex_energy:.2f} kcal/mol")
    print(f"  Mutant complex energy: {result.mutant_complex_energy:.2f} kcal/mol")
    print()
    print(f"Stability Changes (individual chains):")
    print(f"  Chain A (Barnase) ΔΔG:  {result.ddg_stability_a:+.2f} kcal/mol")
    print(f"  Chain B (Barstar) ΔΔG: {result.ddg_stability_b:+.2f} kcal/mol")
    print()

    # Interpretation
    if result.ddg_binding > 0:
        print(f"⚠️  DESTABILIZING effect on binding ({result.ddg_binding:+.2f} kcal/mol)")
        print("   → Mutation weakens the interaction")
    else:
        print(f"✅ STABILIZING effect on binding ({result.ddg_binding:+.2f} kcal/mol)")
        print("   → Mutation strengthens the interaction")

    # Save mutant structure
    with open("1BRS_R59A_mutant.pdb", "w") as f:
        f.write(result.mutant_pdb)
    print(f"\n💾 Mutant structure saved to: 1BRS_R59A_mutant.pdb")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Feature 2: Disulfide Bond Analysis
# ---------------------------------------------------------------------------

def demo_disulfide_detection():
    """Demonstrate disulfide bond detection and mutation impact."""
    print("=" * 70)
    print("DEMO 2: Disulfide Bond Analysis")
    print("=" * 70)

    pdb_file = download_and_clean_complex("1BRS")
    engine = MutationEngine()

    # --- Part A: Detect existing disulfide bonds ---
    print("\nA. Detecting disulfide bonds in Barnase-Barstar complex...")
    disulfides = engine.detect_disulfides(
        pdb_file,
        distance_cutoff=2.5  # Å (S-S distance)
    )

    print(f"\nFound {len(disulfides)} disulfide bond(s):")
    if len(disulfides) > 0:
        print(disulfides)

        for row in disulfides.iter_rows(named=True):
            print(f"  Chain {row['chain1']}:{row['residue1']} ↔ "
                  f"Chain {row['chain2']}:{row['residue2']} "
                  f"(distance: {row['distance']:.2f} Å)")
    else:
        print("  (No disulfide bonds in this structure)")

    # --- Part B: Analyze mutation impact on disulfides ---
    print("\nB. Testing mutation impact (hypothetical C82A on Barnase)...")
    # Note: Barnase has Cys residues, let's check if mutating them affects anything

    impact = engine.analyze_mutation_disulfide_impact(
        pdb_file,
        mutations=["C82A"],
        chain="A"
    )

    print(f"\nMutation Impact Analysis:")
    print(f"  Affected cysteines: {impact['affected_cysteines']}")
    print(f"  Broken disulfide bonds: {len(impact['broken_bonds'])}")
    print(f"  New disulfide bonds: {len(impact['new_bonds'])}")

    if impact['broken_bonds']:
        print("\n⚠️  WARNING: This mutation would break disulfide bonds:")
        for bond in impact['broken_bonds']:
            print(f"     {bond}")
    else:
        print("\n✅ No disulfide bonds affected by this mutation")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Feature 3: Residue Interaction Networks
# ---------------------------------------------------------------------------

def demo_interaction_network():
    """Demonstrate residue interaction network analysis."""
    print("=" * 70)
    print("DEMO 3: Residue Interaction Networks")
    print("=" * 70)

    # Setup database
    db_path = setup_database()
    db = Sicifus(db_path)
    db.load()

    structure_id = "1BRS"

    # --- Part A: Compute interaction network ---
    print("\nA. Computing residue interaction network...")
    print("   (Residues within 5 Å are considered 'interacting')")

    G = db.compute_interaction_network(
        structure_id,
        distance_cutoff=5.0,
        interaction_types=None
    )

    print(f"\nNetwork Statistics:")
    print(f"  Nodes (residues):    {len(G.nodes())}")
    print(f"  Edges (interactions): {len(G.edges())}")
    print(f"  Average degree:      {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")

    # --- Part B: Identify key residues (hubs) ---
    print("\nB. Identifying key residues via centrality analysis...")
    centrality_df = db.analyze_network_centrality(G, top_n=10)

    print("\nTop 10 Hub Residues (by betweenness centrality):")
    print(centrality_df.select(["chain", "residue_number", "residue_name", "betweenness_centrality"]))

    top = centrality_df.row(0, named=True)
    print(f"\n🎯 Most critical residue: Chain {top['chain']}:"
          f"{top['residue_name']}{top['residue_number']}")
    print(f"   Betweenness centrality: {top['betweenness_centrality']:.3f}")
    print("   → This residue is a key structural connector")

    # --- Part C: Visualize network ---
    print("\nC. Visualizing interaction network...")
    db.plot_interaction_network(
        G,
        output_file="barnase_barstar_network.png",
        node_color_by="chain",
        figsize=(12, 12)
    )
    print("   Network plot saved to: barnase_barstar_network.png")

    # --- Part D: Interface-focused network ---
    print("\nD. Analyzing interface residues only...")
    # Get interface residues
    interface_residues = result.interface_residues if 'result' in locals() else None

    if interface_residues is not None and len(interface_residues) > 0:
        print(f"   Found {len(interface_residues)} interface residues")
    else:
        print("   (Run demo_interface_mutagenesis first to identify interface)")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Feature 4: Direct AnalysisToolkit Usage
# ---------------------------------------------------------------------------

def demo_direct_toolkit_usage():
    """Demonstrate using AnalysisToolkit directly (without database)."""
    print("=" * 70)
    print("DEMO 4: AnalysisToolkit - Direct Usage")
    print("=" * 70)

    toolkit = AnalysisToolkit()

    # Create a simple example DataFrame with residue coordinates
    # (In practice, you'd get this from parsing a PDB or from Sicifus database)
    example_df = pl.DataFrame({
        "chain": ["A"] * 10,
        "residue_number": list(range(1, 11)),
        "residue_name": ["ALA", "VAL", "LEU", "ILE", "PHE",
                        "TRP", "TYR", "SER", "THR", "GLY"],
        "atom_name": ["CA"] * 10,
        "element": ["C"] * 10,
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })

    print("\nComputing interaction network from DataFrame...")
    G = toolkit.compute_residue_interaction_network(
        example_df,
        distance_cutoff=10.0
    )

    print(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")

    centrality = toolkit.analyze_network_centrality(G, top_n=5)
    print("\nTop 5 hub residues:")
    print(centrality)

    toolkit.plot_interaction_network(G, output_file="direct_network.png")
    print("\nNetwork plot saved to: direct_network.png")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Advanced Mutation Analysis Features Demonstration")
    print("=" * 70 + "\n")

    try:
        # Run demos in order of complexity
        print("Running demos with Barnase-Barstar complex (1BRS)...\n")

        # Demo 1: Interface mutagenesis (requires OpenMM)
        demo_interface_mutagenesis()

        # Demo 2: Disulfide analysis (fast)
        demo_disulfide_detection()

        # Demo 3: Network analysis (requires database)
        demo_interaction_network()

        # Demo 4: Direct toolkit usage (fast)
        demo_direct_toolkit_usage()

        print("=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - 1BRS.pdb (cleaned structure)")
        print("  - 1BRS_R59A_mutant.pdb (mutant structure)")
        print("  - barnase_barstar_network.png (interaction network)")
        print("  - direct_network.png (toolkit demo network)")
        print("  - interface_demo_db/ (Sicifus database)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. OpenMM and PDBFixer installed: pip install sicifus[energy]")
        print("  2. NetworkX installed (for network analysis)")
        print("  3. Internet connection (to download 1BRS.pdb)")
        import traceback
        traceback.print_exc()
        raise
