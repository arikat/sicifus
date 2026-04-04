"""
Demonstration of advanced mutation analysis features:
1. Mutation-to-Binding Pipeline
2. Disulfide Bond Analysis
3. Residue Interaction Networks

Note: Requires OpenMM and PDBFixer to run the mutation/binding examples.
"""

import polars as pl
from sicifus import Sicifus, MutationEngine
from sicifus.analysis import AnalysisToolkit

# ---------------------------------------------------------------------------
# Feature 1: Mutation-to-Binding Pipeline
# ---------------------------------------------------------------------------

def demo_interface_mutagenesis():
    """Demonstrate automated interface mutation analysis."""
    print("=" * 70)
    print("DEMO 1: Mutation-to-Binding Pipeline (ΔΔG_binding)")
    print("=" * 70)

    engine = MutationEngine()

    # Example: Antibody-antigen complex with 2 chains
    # This automatically:
    # 1. Calculates WT binding energy
    # 2. Applies mutations to specified chains
    # 3. Calculates mutant binding energy
    # 4. Returns ΔΔG_binding AND ΔΔG_stability for each chain

    result = engine.mutate_interface(
        "antibody_antigen.pdb",  # Complex PDB
        mutations={
            "A": ["F13A", "W25L"],  # Mutations on chain A
            "B": ["Y42F"]           # Mutations on chain B
        },
        chains_a=["A"],  # Antibody
        chains_b=["B"],  # Antigen
        max_iterations=500,
        n_runs=3
    )

    print(f"\nBinding Energy Changes:")
    print(f"  WT binding:     {result.wt_binding_energy:.2f} kcal/mol")
    print(f"  Mutant binding: {result.mutant_binding_energy:.2f} kcal/mol")
    print(f"  ΔΔG_binding:    {result.ddg_binding:+.2f} kcal/mol")

    print(f"\nStability Changes:")
    print(f"  Chain A ΔΔG: {result.ddg_stability_a:+.2f} kcal/mol")
    print(f"  Chain B ΔΔG: {result.ddg_stability_b:+.2f} kcal/mol")

    # Interpretation guide
    if result.ddg_binding > 0:
        print(f"\nDestabilizing effect on binding ({result.ddg_binding:+.2f} kcal/mol)")
    else:
        print(f"\nStabilizing effect on binding ({result.ddg_binding:+.2f} kcal/mol)")

    # Save mutant structure
    with open("mutant_complex.pdb", "w") as f:
        f.write(result.mutant_pdb)

    print("\nComplete\n")


def demo_interface_with_sicifus_api():
    """Demonstrate using the Sicifus API for interface mutagenesis."""
    print("=" * 70)
    print("DEMO 1b: Interface Mutagenesis via Sicifus API")
    print("=" * 70)

    db = Sicifus("my_db")

    # Assuming you have a complex structure "1ABC" in the database
    result = db.mutate_interface(
        "1ABC",
        mutations={"A": ["F13A"], "B": ["Y25F"]},
        chains_a=["A"],
        chains_b=["B"],
        max_iterations=500
    )

    print(f"ΔΔG_binding: {result.ddg_binding:+.2f} kcal/mol")
    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Feature 2: Disulfide Bond Analysis
# ---------------------------------------------------------------------------

def demo_disulfide_detection():
    """Demonstrate disulfide bond detection and mutation impact."""
    print("=" * 70)
    print("DEMO 2: Disulfide Bond Analysis")
    print("=" * 70)

    engine = MutationEngine()

    # --- Part A: Detect existing disulfide bonds ---
    print("\nA. Detecting disulfide bonds in wild-type...")
    disulfides = engine.detect_disulfides(
        "protein.pdb",
        distance_cutoff=2.5  # Å (S-S distance)
    )

    print(f"\nFound {len(disulfides)} disulfide bond(s):")
    if len(disulfides) > 0:
        print(disulfides)

        # Highlight critical disulfides
        for row in disulfides.iter_rows(named=True):
            print(f"  {row['chain1']}:{row['residue1']} ↔ {row['chain2']}:{row['residue2']} "
                  f"(distance: {row['distance']:.2f} Å)")

    # --- Part B: Analyze mutation impact on disulfides ---
    print("\nB. Analyzing mutation impact on disulfide bonds...")
    impact = engine.analyze_mutation_disulfide_impact(
        "protein.pdb",
        mutations=["C42A", "C108S"],  # Break two cysteines
        chain="A"
    )

    print(f"\nMutations affecting cysteines: {impact['affected_cysteines']}")
    print(f"Broken disulfide bonds: {len(impact['broken_bonds'])}")
    print(f"New disulfide bonds formed: {len(impact['new_bonds'])}")

    if impact['broken_bonds']:
        print("\nWARNING: These mutations break existing disulfide bonds:")
        for bond in impact['broken_bonds']:
            print(f"  {bond}")

    print("\nComplete\n")


def demo_disulfide_with_sicifus_api():
    """Demonstrate disulfide analysis via Sicifus API."""
    print("=" * 70)
    print("DEMO 2b: Disulfide Analysis via Sicifus API")
    print("=" * 70)

    db = Sicifus("my_db")

    # Detect disulfides
    disulfides = db.detect_disulfides("1CRN")
    print(f"Found {len(disulfides)} disulfide bond(s)")

    # Analyze mutation impact
    impact = db.analyze_mutation_disulfide_impact("1CRN", ["C42A"])
    print(f"Affected cysteines: {impact['affected_cysteines']}")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Feature 3: Residue Interaction Networks
# ---------------------------------------------------------------------------

def demo_interaction_network():
    """Demonstrate residue interaction network analysis."""
    print("=" * 70)
    print("DEMO 3: Residue Interaction Networks")
    print("=" * 70)

    # You'll need a structure loaded as a DataFrame
    # For the Sicifus API:
    db = Sicifus("my_db")
    structure_id = "1CRN"

    # --- Part A: Compute interaction network ---
    print("\nA. Computing residue interaction network...")
    G = db.compute_interaction_network(
        structure_id,
        distance_cutoff=5.0,  # Å (consider residues within 5 Å as interacting)
        interaction_types=None  # Or filter: ["PHE", "TYR", "TRP"] for aromatics only
    )

    print(f"\nNetwork statistics:")
    print(f"  Nodes (residues): {len(G.nodes())}")
    print(f"  Edges (interactions): {len(G.edges())}")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")

    # --- Part B: Identify key residues (hubs) ---
    print("\nB. Identifying key residues via centrality analysis...")
    centrality_df = db.analyze_network_centrality(G, top_n=10)

    print("\nTop 10 hub residues (by betweenness centrality):")
    print(centrality_df)

    # Interpret: High betweenness = residue connects different parts of structure
    top_residue = centrality_df.row(0, named=True)
    print(f"\nMost critical residue: {top_residue['chain']}:"
          f"{top_residue['residue_name']}{top_residue['residue_number']}")
    print(f"  (Betweenness centrality: {top_residue['betweenness_centrality']:.3f})")

    # --- Part C: Visualize network ---
    print("\nC. Visualizing interaction network...")
    db.plot_interaction_network(
        G,
        output_file="interaction_network.png",
        node_color_by="chain",  # Or "residue_name"
        figsize=(14, 14)
    )

    print("Saved to interaction_network.png")

    # --- Part D: Focused analysis (aromatics only) ---
    print("\nD. Aromatic residue network (PHE, TYR, TRP)...")
    G_aromatic = db.compute_interaction_network(
        structure_id,
        distance_cutoff=6.0,
        interaction_types=["PHE", "TYR", "TRP"]
    )

    print(f"Aromatic network: {len(G_aromatic.nodes())} nodes, "
          f"{len(G_aromatic.edges())} edges")

    # Useful for identifying pi-stacking networks, aromatic clusters
    db.plot_interaction_network(
        G_aromatic,
        output_file="aromatic_network.png",
        figsize=(10, 10)
    )

    print("Saved to aromatic_network.png")
    print("\nComplete\n")


def demo_direct_toolkit_usage():
    """Demonstrate using AnalysisToolkit directly (without database)."""
    print("=" * 70)
    print("DEMO 3b: Interaction Networks via AnalysisToolkit")
    print("=" * 70)

    # If you have a DataFrame directly (e.g., from parsing a PDB)
    toolkit = AnalysisToolkit()

    # Example DataFrame with atomic coordinates
    structure_df = pl.DataFrame({
        "chain": ["A"] * 10,
        "residue_number": list(range(1, 11)),
        "residue_name": ["PHE", "TRP", "GLY", "LEU", "ILE",
                        "VAL", "SER", "THR", "ALA", "PRO"],
        "atom_name": ["CA"] * 10,  # Just alpha carbons for simplicity
        "element": ["C"] * 10,
        "x": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })

    G = toolkit.compute_residue_interaction_network(
        structure_df,
        distance_cutoff=10.0
    )

    print(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")

    centrality = toolkit.analyze_network_centrality(G, top_n=5)
    print("\nTop 5 hub residues:")
    print(centrality)

    toolkit.plot_interaction_network(G, output_file="direct_network.png")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Combined Workflow: All Three Features
# ---------------------------------------------------------------------------

def demo_combined_workflow():
    """Demonstrate using all three features together."""
    print("=" * 70)
    print("DEMO 4: Combined Workflow - Design Interface Mutation")
    print("=" * 70)

    db = Sicifus("my_db")
    structure_id = "antibody_complex"

    # Step 1: Identify interface residues via network analysis
    print("\n1. Identifying interface hub residues...")
    G = db.compute_interaction_network(structure_id, distance_cutoff=5.0)
    hubs = db.analyze_network_centrality(G, top_n=10)

    print("Top interface residues:")
    print(hubs.select(["chain", "residue_number", "residue_name", "betweenness_centrality"]))

    # Step 2: Check for disulfide bonds
    print("\n2. Checking for disulfide bonds...")
    disulfides = db.detect_disulfides(structure_id)
    print(f"Found {len(disulfides)} disulfide bond(s)")

    # Step 3: Design mutations avoiding disulfides
    print("\n3. Designing mutations at hub residues (avoiding cysteines)...")
    candidate_mutations = {"A": ["F13A"], "B": ["Y25F"]}  # Example

    # Check impact on disulfides first
    all_mutations = candidate_mutations.get("A", []) + candidate_mutations.get("B", [])
    impact = db.analyze_mutation_disulfide_impact(structure_id, all_mutations)

    if impact['broken_bonds']:
        print("Warning: Mutations would break disulfide bonds!")
    else:
        print("No disulfide bonds affected")

    # Step 4: Test binding impact
    print("\n4. Testing binding affinity impact...")
    result = db.mutate_interface(
        structure_id,
        mutations=candidate_mutations,
        chains_a=["A"],
        chains_b=["B"]
    )

    print(f"\nResults:")
    print(f"  ΔΔG_binding:  {result.ddg_binding:+.2f} kcal/mol")
    print(f"  ΔΔG_stab (A): {result.ddg_stability_a:+.2f} kcal/mol")
    print(f"  ΔΔG_stab (B): {result.ddg_stability_b:+.2f} kcal/mol")

    # Interpretation
    if result.ddg_binding < 0 and result.ddg_stability_a > -1.0:
        print("\nSUCCESS: Improved binding with minimal stability loss!")
    elif result.ddg_binding < -2.0:
        print("\nSignificant stability loss - reconsider mutations")

    print("\nComplete\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Advanced Mutation Analysis Features Demonstration")
    print("=" * 70 + "\n")

    try:
        # Run individual demos (comment out as needed)
        # demo_interface_mutagenesis()
        # demo_disulfide_detection()
        demo_interaction_network()
        # demo_combined_workflow()

        print("=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. OpenMM and PDBFixer installed: pip install sicifus[energy]")
        print("  2. A Sicifus database or PDB files")
        raise
