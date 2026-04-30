"""
Demonstration of flexible atom-based alignment using PyMOL-like selection syntax.

This example shows how to:
1. Align structures on arbitrary atom selections (not just CA)
2. Use PyMOL-like selection syntax for flexible atom picking
3. Align ligands for docking comparison
4. Overlay binding sites
5. Compare transition states in computational chemistry

Note: This uses Kabsch algorithm for optimal superposition.
"""

import polars as pl
import numpy as np
from sicifus import AtomAligner, SelectionParser, write_pdb


def demo_basic_alignment():
    """Demonstrate basic alignment on CA atoms."""
    print("=" * 70)
    print("DEMO 1: Basic Alignment on CA Atoms")
    print("=" * 70)

    # Create two simple protein structures
    mobile = pl.DataFrame({
        "atom_name": ["CA", "CA", "CA", "CA"],
        "residue_name": ["ALA", "VAL", "GLY", "LEU"],
        "chain": ["A", "A", "A", "A"],
        "residue_number": [1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0],
        "element": ["C", "C", "C", "C"],
    })

    # Rotate and translate mobile structure
    theta = np.pi / 4  # 45 degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    coords = mobile.select(["x", "y", "z"]).to_numpy()
    coords_rotated = coords @ R.T + np.array([5.0, 5.0, 0.0])

    mobile_rotated = mobile.with_columns([
        pl.Series("x", coords_rotated[:, 0]),
        pl.Series("y", coords_rotated[:, 1]),
        pl.Series("z", coords_rotated[:, 2]),
    ])

    # Align on CA atoms
    aligner = AtomAligner()
    result, aligned = aligner.align(
        mobile_rotated,
        mobile,
        selection="name CA",
        apply_to_mobile=mobile_rotated
    )

    print(f"\nAlignment Results:")
    print(f"  Selection: {result.selection}")
    print(f"  Atoms used: {result.n_atoms}")
    print(f"  RMSD: {result.rmsd:.4f} Å")
    print(f"  Rotation matrix:\n{result.rotation_matrix}")
    print(f"  Translation: {result.translation_vector}")
    print("\n✅ Structures aligned successfully!\n")


def demo_ligand_alignment():
    """Demonstrate aligning ligands on pharmacophore atoms."""
    print("=" * 70)
    print("DEMO 2: Ligand Alignment (Docking Pose Comparison)")
    print("=" * 70)

    # Create ATP-like ligand pose 1
    atp_pose1 = pl.DataFrame({
        "atom_name": ["C1", "C2", "C3", "N1", "N2", "O1", "O2", "P1"],
        "residue_name": ["ATP"] * 8,
        "chain": ["B"] * 8,
        "residue_number": [100] * 8,
        "x": [10.0, 11.5, 13.0, 12.0, 14.0, 11.0, 12.5, 15.0],
        "y": [0.0, 0.5, 0.0, 1.5, 0.5, 2.0, 2.5, 1.0],
        "z": [0.0, 0.0, 0.5, 0.0, 1.0, 0.5, 1.5, 0.5],
        "element": ["C", "C", "C", "N", "N", "O", "O", "P"],
    })

    # Create pose 2 (rotated and translated)
    theta = np.pi / 6
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    coords = atp_pose1.select(["x", "y", "z"]).to_numpy()
    coords_pose2 = coords @ R_y.T + np.array([2.0, 1.0, 3.0])

    atp_pose2 = atp_pose1.with_columns([
        pl.Series("x", coords_pose2[:, 0]),
        pl.Series("y", coords_pose2[:, 1]),
        pl.Series("z", coords_pose2[:, 2]),
    ])

    print("\nAligning two ATP docking poses on adenine ring atoms (C1, C2, N1)...")

    aligner = AtomAligner()
    result, aligned_pose2 = aligner.align(
        atp_pose2,
        atp_pose1,
        selection="resn ATP and name C1,C2,N1",
        apply_to_mobile=atp_pose2
    )

    print(f"\nAlignment Results:")
    print(f"  Pharmacophore atoms: C1, C2, N1")
    print(f"  RMSD (aligned): {result.rmsd:.4f} Å")

    # Compute RMSD before alignment (positional)
    rmsd_before = aligner.compute_pairwise_rmsd(
        atp_pose2,
        atp_pose1,
        selection="resn ATP and name C1,C2,N1",
        align=False
    )
    print(f"  RMSD (before alignment): {rmsd_before:.4f} Å")
    print(f"  Improvement: {rmsd_before - result.rmsd:.4f} Å")

    # Save aligned ligands
    write_pdb(atp_pose1, "atp_pose1_reference.pdb")
    write_pdb(aligned_pose2, "atp_pose2_aligned.pdb")
    print("\n✅ Saved atp_pose1_reference.pdb and atp_pose2_aligned.pdb")
    print("   Visualize in PyMOL: load atp_pose*.pdb\n")


def demo_binding_site_alignment():
    """Demonstrate aligning proteins on binding site residues."""
    print("=" * 70)
    print("DEMO 3: Binding Site Alignment")
    print("=" * 70)

    # Create protein with binding site
    protein1 = pl.DataFrame({
        "atom_name": ["CA"] * 10 + ["CB"] * 5,
        "residue_name": ["ALA", "VAL", "ILE", "LEU", "PHE",
                        "GLY", "SER", "THR", "TYR", "TRP"] + ["VAL"] * 5,
        "chain": ["A"] * 15,
        "residue_number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [2, 3, 4, 5, 9],
        "x": list(np.linspace(0, 9, 10)) + [1.5, 2.5, 3.5, 4.5, 8.5],
        "y": [0.0] * 10 + [1.0] * 5,
        "z": [0.0] * 15,
        "element": ["C"] * 15,
    })

    # Create variant with rotated binding site (residues 2-5)
    theta = np.pi / 8
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    coords = protein1.select(["x", "y", "z"]).to_numpy()
    coords_rot = coords @ R_z.T + np.array([1.0, 2.0, 0.5])

    protein2 = protein1.with_columns([
        pl.Series("x", coords_rot[:, 0]),
        pl.Series("y", coords_rot[:, 1]),
        pl.Series("z", coords_rot[:, 2]),
    ])

    print("\nAligning on binding site residues (2-5) using CA and CB atoms...")

    aligner = AtomAligner()
    result, aligned_protein2 = aligner.align(
        protein2,
        protein1,
        selection="resi 2-5 and name CA,CB",
        apply_to_mobile=protein2
    )

    print(f"\nAlignment Results:")
    print(f"  Selection: residues 2-5, atoms CA+CB")
    print(f"  Atoms used: {result.n_atoms}")
    print(f"  RMSD: {result.rmsd:.4f} Å")
    print("\n✅ Binding sites aligned!\n")


def demo_transition_state_overlay():
    """Demonstrate aligning transition states for computational chemistry."""
    print("=" * 70)
    print("DEMO 4: Transition State Overlay (QM/MM)")
    print("=" * 70)

    # Create reaction center geometry (SN2 transition state-like)
    # X- + C-Y → X-C + Y-
    ts1 = pl.DataFrame({
        "atom_name": ["X", "C", "Y", "H1", "H2"],
        "residue_name": ["TS"] * 5,
        "chain": ["A"] * 5,
        "residue_number": [1] * 5,
        "x": [-1.5, 0.0, 1.5, 0.2, 0.2],
        "y": [0.0, 0.0, 0.0, 1.0, -1.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0],
        "element": ["F", "C", "Cl", "H", "H"],
    })

    # Create TS2 with slightly different geometry
    ts2 = ts1.clone()
    coords = ts2.select(["x", "y", "z"]).to_numpy()
    # Asymmetric TS: X closer, Y farther
    coords[0, 0] = -1.3  # X-C distance shorter
    coords[2, 0] = 1.7   # C-Y distance longer

    # Add rotation
    theta = np.pi / 12
    R_xyz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    coords_rot = coords @ R_xyz.T + np.array([2.0, 1.0, 0.5])

    ts2 = ts2.with_columns([
        pl.Series("x", coords_rot[:, 0]),
        pl.Series("y", coords_rot[:, 1]),
        pl.Series("z", coords_rot[:, 2]),
    ])

    print("\nAligning two SN2 transition states on reactive atoms (X, C, Y)...")

    aligner = AtomAligner()
    result, aligned_ts2 = aligner.align(
        ts2,
        ts1,
        selection="name X,C,Y",
        apply_to_mobile=ts2
    )

    print(f"\nAlignment Results:")
    print(f"  Reactive center: X-C-Y")
    print(f"  RMSD: {result.rmsd:.4f} Å")

    # Analyze differences after alignment
    coords_ts1 = ts1.filter(pl.col("atom_name").is_in(["X", "C", "Y"])) \
                    .select(["x", "y", "z"]).to_numpy()
    coords_aligned = aligned_ts2.filter(pl.col("atom_name").is_in(["X", "C", "Y"])) \
                                 .select(["x", "y", "z"]).to_numpy()

    print(f"\n  Atom-by-atom deviations:")
    for atom, coord1, coord2 in zip(["X", "C", "Y"], coords_ts1, coords_aligned):
        dist = np.linalg.norm(coord1 - coord2)
        print(f"    {atom}: {dist:.4f} Å")

    write_pdb(ts1, "ts1_reference.pdb")
    write_pdb(aligned_ts2, "ts2_aligned.pdb")
    print("\n✅ Saved ts1_reference.pdb and ts2_aligned.pdb")
    print("   Visualize in PyMOL to compare geometries\n")


def demo_selection_syntax():
    """Demonstrate various selection syntaxes."""
    print("=" * 70)
    print("DEMO 5: Selection Syntax Examples")
    print("=" * 70)

    # Create diverse structure
    structure = pl.DataFrame({
        "atom_name": ["CA", "CB", "CA", "CB", "C1", "C2", "CA"],
        "residue_name": ["ALA", "ALA", "VAL", "VAL", "ATP", "ATP", "GLY"],
        "chain": ["A", "A", "A", "A", "B", "B", "C"],
        "residue_number": [1, 1, 2, 2, 100, 100, 1],
        "x": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 20.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "element": ["C", "C", "C", "C", "C", "C", "C"],
    })

    parser = SelectionParser()

    examples = [
        ("name CA", "Select all CA atoms"),
        ("chain A", "Select all atoms in chain A"),
        ("resi 1-2", "Select residues 1 and 2"),
        ("resn ATP", "Select ATP residue"),
        ("chain A and name CA", "CA atoms in chain A only"),
        ("resi 1 and name CA,CB", "CA and CB of residue 1"),
        ("resn ATP and name C1,C2", "Specific ATP atoms"),
        ("chain A,B", "Multiple chains"),
    ]

    print("\nTesting PyMOL-like selection syntax:\n")

    for selection, description in examples:
        result = parser.parse(selection, structure)
        print(f"  {selection:30s} → {len(result)} atoms  ({description})")

    print("\n✅ All selections work!\n")


def demo_multiple_structure_alignment():
    """Demonstrate aligning multiple structures to a reference."""
    print("=" * 70)
    print("DEMO 6: Multiple Structure Alignment")
    print("=" * 70)

    # Create reference structure
    reference = pl.DataFrame({
        "atom_name": ["CA"] * 5,
        "residue_name": ["ALA", "VAL", "GLY", "LEU", "ILE"],
        "chain": ["A"] * 5,
        "residue_number": [1, 2, 3, 4, 5],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0],
        "element": ["C"] * 5,
    })

    # Create 3 variants with different orientations
    structures = {"reference": reference}

    for i in range(1, 4):
        theta = np.pi / (4 * i)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        coords = reference.select(["x", "y", "z"]).to_numpy()
        coords_rot = coords @ R.T + np.array([i * 2.0, i * 1.0, 0.0])

        structures[f"variant{i}"] = reference.with_columns([
            pl.Series("x", coords_rot[:, 0]),
            pl.Series("y", coords_rot[:, 1]),
            pl.Series("z", coords_rot[:, 2]),
        ])

    print(f"\nAligning {len(structures)} structures to reference...")

    aligner = AtomAligner()
    results = aligner.align_multiple(
        structures,
        reference_id="reference",
        selection="name CA",
        apply_to_all=True
    )

    print(f"\nAlignment Results:")
    print(f"  {'Structure':<15s} {'RMSD (Å)':<12s} {'Atoms':<8s}")
    print(f"  {'-'*15} {'-'*12} {'-'*8}")

    for struct_id, (result, transformed) in sorted(results.items()):
        print(f"  {struct_id:<15s} {result.rmsd:<12.4f} {result.n_atoms:<8d}")

    print("\n✅ All structures aligned to reference!\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Sicifus Atom-Based Alignment Demo")
    print("=" * 70 + "\n")

    try:
        demo_basic_alignment()
        demo_ligand_alignment()
        demo_binding_site_alignment()
        demo_transition_state_overlay()
        demo_selection_syntax()
        demo_multiple_structure_alignment()

        print("=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\n📖 Key Features:")
        print("  ✓ PyMOL-like selection syntax")
        print("  ✓ Kabsch alignment on arbitrary atoms")
        print("  ✓ Ligand overlay for docking")
        print("  ✓ Binding site comparison")
        print("  ✓ Transition state analysis")
        print("  ✓ Multiple structure alignment")
        print("\n📚 Documentation:")
        print("  from sicifus import AtomAligner")
        print("  help(AtomAligner.align)")
        print("\n💡 Use cases:")
        print("  - Computational chemistry (TS overlay, IRC path analysis)")
        print("  - Drug design (ligand pose comparison)")
        print("  - Structural biology (binding site superposition)")
        print("  - Protein engineering (variant comparison)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
