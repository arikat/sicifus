"""
Tests for atom-based alignment with PyMOL-like selection syntax.
"""

import numpy as np
import polars as pl
import pytest
from sicifus.atom_align import AtomAligner, SelectionParser, AlignmentResult


# Test data
@pytest.fixture
def sample_protein():
    """Create a sample protein DataFrame with 3 residues."""
    return pl.DataFrame({
        "atom_name": ["N", "CA", "C", "O", "CB",
                     "N", "CA", "C", "O", "CG",
                     "N", "CA", "C", "O", "CB"],
        "residue_name": ["ALA", "ALA", "ALA", "ALA", "ALA",
                        "VAL", "VAL", "VAL", "VAL", "VAL",
                        "GLY", "GLY", "GLY", "GLY", "GLY"],
        "chain": ["A"] * 15,
        "residue_number": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "x": [0.0, 1.0, 2.0, 3.0, 1.5, 4.0, 5.0, 6.0, 7.0, 5.5, 8.0, 9.0, 10.0, 11.0, 9.5],
        "y": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "element": ["N", "C", "C", "O", "C", "N", "C", "C", "O", "C", "N", "C", "C", "O", "C"],
    })


@pytest.fixture
def sample_ligand():
    """Create a sample ligand DataFrame."""
    return pl.DataFrame({
        "atom_name": ["C1", "C2", "N1", "O1", "C3"],
        "residue_name": ["ATP", "ATP", "ATP", "ATP", "ATP"],
        "chain": ["B", "B", "B", "B", "B"],
        "residue_number": [100, 100, 100, 100, 100],
        "x": [10.0, 11.0, 12.0, 13.0, 14.0],
        "y": [0.0, 1.0, 0.5, 1.5, 0.0],
        "z": [0.0, 0.0, 1.0, 1.0, 0.5],
        "element": ["C", "C", "N", "O", "C"],
    })


@pytest.fixture
def rotated_protein(sample_protein):
    """Create a rotated version of sample protein."""
    # Simple 45-degree rotation around Z axis
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    coords = sample_protein.select(["x", "y", "z"]).to_numpy()
    coords_rotated = coords @ R.T + np.array([1.0, 2.0, 3.0])  # Rotate and translate

    return sample_protein.with_columns([
        pl.Series("x", coords_rotated[:, 0]),
        pl.Series("y", coords_rotated[:, 1]),
        pl.Series("z", coords_rotated[:, 2]),
    ])


# ===================================================================
# Selection Parser Tests
# ===================================================================

class TestSelectionParser:
    """Test PyMOL-like selection syntax parsing."""

    def test_select_all(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("all", sample_protein)
        assert len(result) == len(sample_protein)

    def test_select_chain(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("chain A", sample_protein)
        assert len(result) == 15
        assert result["chain"].unique().to_list() == ["A"]

    def test_select_multiple_chains(self):
        df = pl.DataFrame({
            "atom_name": ["CA", "CA", "CA"],
            "residue_name": ["ALA", "ALA", "ALA"],
            "chain": ["A", "B", "C"],
            "residue_number": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "element": ["C", "C", "C"],
        })
        parser = SelectionParser()
        result = parser.parse("chain A,B", df)
        assert len(result) == 2
        assert set(result["chain"].to_list()) == {"A", "B"}

    def test_select_resi_single(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("resi 1", sample_protein)
        assert len(result) == 5
        assert result["residue_number"].unique().to_list() == [1]

    def test_select_resi_range(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("resi 1-2", sample_protein)
        assert len(result) == 10

    def test_select_resi_multiple(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("resi 1,2", sample_protein)
        assert len(result) == 10

    def test_select_resn(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("resn ALA", sample_protein)
        assert len(result) == 5
        assert result["residue_name"].unique().to_list() == ["ALA"]

    def test_select_multiple_resn(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("resn ALA,VAL", sample_protein)
        assert len(result) == 10

    def test_select_atom_name(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("name CA", sample_protein)
        assert len(result) == 3
        assert result["atom_name"].unique().to_list() == ["CA"]

    def test_select_multiple_atom_names(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("name CA,CB", sample_protein)
        assert len(result) == 5  # 3 CA + 2 CB (GLY doesn't have CB)

    def test_select_element(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("element C", sample_protein)
        assert len(result) == 9  # CA, C, CB/CG per residue

    def test_select_and_combination(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("chain A and name CA", sample_protein)
        assert len(result) == 3
        assert result["atom_name"].unique().to_list() == ["CA"]
        assert result["chain"].unique().to_list() == ["A"]

    def test_select_complex_combination(self, sample_protein):
        parser = SelectionParser()
        result = parser.parse("resi 1 and name CA,CB", sample_protein)
        assert len(result) == 2
        assert set(result["atom_name"].to_list()) == {"CA", "CB"}
        assert result["residue_number"].unique().to_list() == [1]

    def test_select_ligand(self, sample_ligand):
        parser = SelectionParser()
        result = parser.parse("resn ATP and name C1,C2,N1", sample_ligand)
        assert len(result) == 3
        assert set(result["atom_name"].to_list()) == {"C1", "C2", "N1"}

    def test_select_empty(self, sample_protein):
        parser = SelectionParser()
        with pytest.raises(ValueError, match="matched 0 atoms"):
            parser.parse("chain Z", sample_protein)

    def test_invalid_keyword(self, sample_protein):
        parser = SelectionParser()
        with pytest.raises(ValueError, match="Unknown selection keyword"):
            parser.parse("invalid_keyword test", sample_protein)

    def test_case_insensitive(self, sample_protein):
        parser = SelectionParser()
        result1 = parser.parse("CHAIN A", sample_protein)
        result2 = parser.parse("chain a", sample_protein)
        assert len(result1) == len(result2)


# ===================================================================
# Atom Aligner Tests
# ===================================================================

class TestAtomAligner:
    """Test Kabsch alignment on arbitrary atom selections."""

    def test_align_identical_structures(self, sample_protein):
        """Aligning identical structures should give RMSD ≈ 0."""
        aligner = AtomAligner()
        result, _ = aligner.align(
            sample_protein,
            sample_protein.clone(),
            selection="name CA"
        )
        assert result.rmsd < 1e-6
        assert result.n_atoms == 3

    def test_align_rotated_structure(self, sample_protein, rotated_protein):
        """Aligning rotated structure should recover low RMSD."""
        aligner = AtomAligner()
        result, _ = aligner.align(
            rotated_protein,
            sample_protein,
            selection="name CA"
        )
        # Should align perfectly (within numerical precision)
        assert result.rmsd < 0.01  # Very small RMSD after alignment
        assert result.n_atoms == 3

    def test_align_with_transformation(self, sample_protein, rotated_protein):
        """Test applying transformation to entire structure."""
        aligner = AtomAligner()
        result, transformed = aligner.align(
            rotated_protein,
            sample_protein,
            selection="name CA",
            apply_to_mobile=rotated_protein
        )

        assert transformed is not None
        assert len(transformed) == len(rotated_protein)

        # Check that CA atoms in transformed match target after alignment
        ca_transformed = transformed.filter(pl.col("atom_name") == "CA")
        ca_target = sample_protein.filter(pl.col("atom_name") == "CA")

        coords_transformed = ca_transformed.select(["x", "y", "z"]).to_numpy()
        coords_target = ca_target.select(["x", "y", "z"]).to_numpy()

        diff = np.linalg.norm(coords_transformed - coords_target, axis=1)
        assert np.max(diff) < 0.01

    def test_align_mismatched_atoms(self, sample_protein):
        """Different number of atoms should raise error."""
        aligner = AtomAligner()

        # Create structure with different number of CA atoms
        subset = sample_protein.filter(pl.col("residue_number") == 1)

        with pytest.raises(ValueError, match="same number of atoms"):
            aligner.align(subset, sample_protein, selection="name CA")

    def test_align_too_few_atoms(self, sample_protein):
        """Need at least 3 atoms for alignment."""
        aligner = AtomAligner()

        # Create structure with only 2 atoms
        subset = sample_protein.head(2)

        with pytest.raises(ValueError, match="at least 3 atoms"):
            aligner.align(subset, subset.clone(), selection="name N,CA")

    def test_align_on_backbone(self, sample_protein):
        """Test alignment on backbone atoms."""
        aligner = AtomAligner()
        result, _ = aligner.align(
            sample_protein,
            sample_protein.clone(),
            selection="name N,CA,C"
        )
        assert result.rmsd < 1e-6
        assert result.n_atoms == 9  # 3 atoms per residue × 3 residues

    def test_align_on_residue_range(self, sample_protein):
        """Test alignment on specific residue range."""
        aligner = AtomAligner()
        result, _ = aligner.align(
            sample_protein,
            sample_protein.clone(),
            selection="resi 1-2 and name CA,C"
        )
        assert result.rmsd < 1e-6
        assert result.n_atoms == 4  # 2 residues × 2 atoms (CA, C)

    def test_result_contains_transformation(self, sample_protein, rotated_protein):
        """Check that result contains rotation and translation."""
        aligner = AtomAligner()
        result, _ = aligner.align(
            rotated_protein,
            sample_protein,
            selection="name CA"
        )

        assert result.rotation_matrix.shape == (3, 3)
        assert result.translation_vector.shape == (3,)
        assert isinstance(result.rmsd, float)
        assert result.selection == "name CA"

    def test_align_ligand_atoms(self, sample_ligand):
        """Test alignment on ligand atoms."""
        # Create slightly perturbed ligand
        coords = sample_ligand.select(["x", "y", "z"]).to_numpy()
        noise = np.random.RandomState(42).randn(*coords.shape) * 0.1
        coords_perturbed = coords + noise

        ligand_perturbed = sample_ligand.with_columns([
            pl.Series("x", coords_perturbed[:, 0]),
            pl.Series("y", coords_perturbed[:, 1]),
            pl.Series("z", coords_perturbed[:, 2]),
        ])

        aligner = AtomAligner()
        result, _ = aligner.align(
            ligand_perturbed,
            sample_ligand,
            selection="resn ATP and name C1,C2,N1"
        )

        # Should align well (RMSD close to noise level)
        assert result.rmsd < 0.2
        assert result.n_atoms == 3


class TestMultipleAlignment:
    """Test aligning multiple structures."""

    def test_align_multiple_structures(self, sample_protein):
        """Test aligning multiple structures to a reference."""
        # Create 3 variants with different rotations
        structures = {"ref": sample_protein}

        for i in range(1, 3):
            theta = np.pi / (4 * i)
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            coords = sample_protein.select(["x", "y", "z"]).to_numpy()
            coords_rot = coords @ R.T

            structures[f"var{i}"] = sample_protein.with_columns([
                pl.Series("x", coords_rot[:, 0]),
                pl.Series("y", coords_rot[:, 1]),
                pl.Series("z", coords_rot[:, 2]),
            ])

        aligner = AtomAligner()
        results = aligner.align_multiple(
            structures,
            reference_id="ref",
            selection="name CA"
        )

        assert len(results) == 3
        assert "ref" in results
        assert "var1" in results
        assert "var2" in results

        # Reference should have RMSD = 0
        assert results["ref"][0].rmsd < 1e-6

        # Variants should align well
        assert results["var1"][0].rmsd < 0.01
        assert results["var2"][0].rmsd < 0.01

    def test_align_multiple_with_transform(self, sample_protein):
        """Test align_multiple with apply_to_all."""
        structures = {
            "ref": sample_protein,
            "mob": sample_protein.clone()
        }

        aligner = AtomAligner()
        results = aligner.align_multiple(
            structures,
            reference_id="ref",
            selection="name CA",
            apply_to_all=True
        )

        for sid, (result, transformed) in results.items():
            assert transformed is not None
            assert len(transformed) == len(sample_protein)


class TestPairwiseRMSD:
    """Test RMSD calculation."""

    def test_rmsd_identical(self, sample_protein):
        """RMSD of identical structures should be 0."""
        aligner = AtomAligner()
        rmsd = aligner.compute_pairwise_rmsd(
            sample_protein,
            sample_protein.clone(),
            selection="name CA",
            align=True
        )
        assert rmsd < 1e-6

    def test_rmsd_with_vs_without_alignment(self, sample_protein, rotated_protein):
        """Aligned RMSD should be lower than positional RMSD."""
        aligner = AtomAligner()

        rmsd_aligned = aligner.compute_pairwise_rmsd(
            rotated_protein,
            sample_protein,
            selection="name CA",
            align=True
        )

        rmsd_positional = aligner.compute_pairwise_rmsd(
            rotated_protein,
            sample_protein,
            selection="name CA",
            align=False
        )

        # Aligned RMSD should be much smaller
        assert rmsd_aligned < rmsd_positional

    def test_rmsd_positional_zero(self, sample_protein):
        """Positional RMSD of identical coords should be 0."""
        aligner = AtomAligner()
        rmsd = aligner.compute_pairwise_rmsd(
            sample_protein,
            sample_protein.clone(),
            selection="name CA",
            align=False
        )
        assert rmsd < 1e-6


# ===================================================================
# Integration Tests
# ===================================================================

class TestRealWorldUseCases:
    """Test realistic use cases."""

    def test_ligand_docking_overlay(self, sample_ligand):
        """Simulate overlaying two ligand poses."""
        # Create two different conformations
        coords = sample_ligand.select(["x", "y", "z"]).to_numpy()

        # Rotate second pose
        theta = np.pi / 6
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        coords_pose2 = coords @ R.T + np.array([0.5, 0.5, 0.5])

        pose2 = sample_ligand.with_columns([
            pl.Series("x", coords_pose2[:, 0]),
            pl.Series("y", coords_pose2[:, 1]),
            pl.Series("z", coords_pose2[:, 2]),
        ])

        # Align on pharmacophore atoms (C1, C2, N1)
        aligner = AtomAligner()
        result, aligned_pose2 = aligner.align(
            pose2,
            sample_ligand,
            selection="resn ATP and name C1,C2,N1",
            apply_to_mobile=pose2
        )

        assert result.rmsd < 0.01
        assert aligned_pose2 is not None

    def test_binding_site_comparison(self, sample_protein):
        """Simulate comparing binding sites from two structures."""
        # Create two structures with same binding site
        struct1 = sample_protein
        struct2 = sample_protein.clone()

        # Add noise to struct2
        coords = struct2.select(["x", "y", "z"]).to_numpy()
        coords_noisy = coords + np.random.RandomState(42).randn(*coords.shape) * 0.05

        struct2 = struct2.with_columns([
            pl.Series("x", coords_noisy[:, 0]),
            pl.Series("y", coords_noisy[:, 1]),
            pl.Series("z", coords_noisy[:, 2]),
        ])

        # Align on binding site residues
        aligner = AtomAligner()
        result, _ = aligner.align(
            struct2,
            struct1,
            selection="resi 1-2 and name CA,CB",
            apply_to_mobile=struct2
        )

        # Should align reasonably well
        assert result.rmsd < 0.1

    def test_transition_state_overlay(self):
        """Simulate overlaying QM/MM transition states."""
        # Create simple "transition state" geometries
        ts1 = pl.DataFrame({
            "atom_name": ["C1", "C2", "N1", "O1"],
            "residue_name": ["TS", "TS", "TS", "TS"],
            "chain": ["A", "A", "A", "A"],
            "residue_number": [1, 1, 1, 1],
            "x": [0.0, 1.5, 3.0, 1.5],
            "y": [0.0, 0.0, 0.0, 1.2],
            "z": [0.0, 0.0, 0.0, 0.0],
            "element": ["C", "C", "N", "O"],
        })

        # Slightly different geometry
        ts2 = ts1.clone()
        coords = ts2.select(["x", "y", "z"]).to_numpy()
        coords[1] = [1.4, 0.1, 0.0]  # Slight change in C2 position

        ts2 = ts2.with_columns([
            pl.Series("x", coords[:, 0]),
            pl.Series("y", coords[:, 1]),
            pl.Series("z", coords[:, 2]),
        ])

        # Align on reactive atoms (C1, C2, N1)
        aligner = AtomAligner()
        result, _ = aligner.align(
            ts2,
            ts1,
            selection="name C1,C2,N1"
        )

        assert result.n_atoms == 3
        assert result.rmsd < 0.2
