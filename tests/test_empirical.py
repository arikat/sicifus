"""Tests for the sicifus.empirical module (FoldX-style empirical scorer).

These tests are fast and need only numpy / scipy / gemmi — no OpenMM or
PDBFixer.  They cover the energy-term maths and the score_mutation contract.

Run:    pytest tests/test_empirical.py -v
"""

import math
import textwrap

import polars as pl
import pytest

from sicifus.empirical import EmpiricalScorer, EmpiricalEnergy, _fibonacci_sphere
from sicifus.data import empirical_params as P


def _gemmi_available() -> bool:
    try:
        import gemmi  # noqa: F401
        return True
    except ImportError:
        return False


requires_gemmi = pytest.mark.skipif(
    not _gemmi_available(), reason="gemmi not installed"
)

# A small two-residue peptide with plausible geometry.
PEPTIDE_PDB = textwrap.dedent("""\
    ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
    ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
    ATOM      3  C   ALA A   1      13.149   6.232  -5.180  1.00  0.00           C
    ATOM      4  O   ALA A   1      13.656   7.339  -5.319  1.00  0.00           O
    ATOM      5  CB  ALA A   1      11.006   7.142  -4.272  1.00  0.00           C
    ATOM      6  N   GLY A   2      13.880   5.130  -5.043  1.00  0.00           N
    ATOM      7  CA  GLY A   2      15.334   5.123  -5.043  1.00  0.00           C
    ATOM      8  C   GLY A   2      15.890   5.221  -3.626  1.00  0.00           C
    ATOM      9  O   GLY A   2      15.137   5.401  -2.662  1.00  0.00           O
    END
""")


def _pdb(atoms):
    """Build a PDB string from (serial, name, resn, chain, resi, x, y, z, elem)."""
    lines = []
    for (i, name, resn, ch, resi, x, y, z, elem) in atoms:
        lines.append(
            f"ATOM  {i:>5} {name:<4} {resn:<3} {ch}{resi:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}"
        )
    lines.append("END")
    return "\n".join(lines)


@pytest.fixture
def scorer(tmp_path):
    return EmpiricalScorer(work_dir=str(tmp_path / "emp"))


# ===================================================================
# Parameter table sanity
# ===================================================================

class TestParams:
    def test_charges(self):
        assert P.atom_charge("LYS", "NZ") == 1.0
        assert P.atom_charge("ASP", "OD1") == -0.5
        assert P.atom_charge("ALA", "O") == -0.5   # backbone fallback
        assert P.atom_charge("ALA", "CB") == 0.0   # neutral

    def test_asp_charged_variants(self):
        assert P.atom_asp("LYS", "NZ", "N") == P.ASP["N+"]
        assert P.atom_asp("ASP", "OD1", "O") == P.ASP["O-"]
        assert P.atom_asp("ALA", "CB", "C") == P.ASP["C"]

    def test_hbond_roles(self):
        assert P.atom_hbond_role("SER", "OG") == "B"
        assert P.atom_hbond_role("ALA", "N") == "D"
        assert P.atom_hbond_role("ALA", "O") == "A"
        assert P.atom_hbond_role("ALA", "CB") is None

    def test_entropy_tables_cover_all_residues(self):
        for resn in P.SIDECHAIN_ENTROPY:
            assert resn in P.BACKBONE_ENTROPY
        # Larger side chains lose more entropy than alanine.
        assert P.SIDECHAIN_ENTROPY["ARG"] > P.SIDECHAIN_ENTROPY["ALA"]


# ===================================================================
# Geometry helpers
# ===================================================================

class TestSphere:
    def test_points_are_unit_vectors(self):
        pts = _fibonacci_sphere(92)
        norms = (pts ** 2).sum(axis=1) ** 0.5
        assert pts.shape == (92, 3)
        assert all(abs(n - 1.0) < 1e-6 for n in norms)


# ===================================================================
# SASA
# ===================================================================

@requires_gemmi
class TestSASA:
    def test_isolated_atom_full_sphere(self, scorer):
        import numpy as np
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.70])           # carbon
        sasa = scorer._sasa(coords, radii)
        expected = 4 * math.pi * (1.70 + scorer.sasa_probe) ** 2
        assert abs(sasa[0] - expected) / expected < 0.02

    def test_buried_atom_less_than_isolated(self, scorer):
        import numpy as np
        # Central atom surrounded by close neighbours → reduced SASA.
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0], [0.0, 0.0, -2.0],
        ])
        radii = np.full(len(coords), 1.70)
        sasa = scorer._sasa(coords, radii)
        isolated = 4 * math.pi * (1.70 + scorer.sasa_probe) ** 2
        assert sasa[0] < isolated * 0.5


# ===================================================================
# Energy terms
# ===================================================================

@requires_gemmi
class TestTerms:
    def test_score_returns_all_terms(self, scorer):
        e = scorer.score(PEPTIDE_PDB)
        assert isinstance(e, EmpiricalEnergy)
        for field in ("vdw", "clash", "solvH", "solvP", "hbond",
                      "elec", "mc_entropy", "sc_entropy", "total"):
            assert isinstance(getattr(e, field), float)

    def test_total_is_sum_of_terms(self, scorer):
        e = scorer.score(PEPTIDE_PDB)
        parts = (e.vdw + e.clash + e.solvH + e.solvP + e.hbond
                 + e.elec + e.mc_entropy + e.sc_entropy)
        assert abs(e.total - parts) < 1e-2

    def test_clash_penalty_positive_on_overlap(self, scorer):
        # Two carbons 2.0 Å apart in different residues: > bond cutoff (1.9)
        # but well inside the summed vdW radii (3.4) → clash.
        pdb = _pdb([
            (1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            (2, "CA", "GLY", "A", 2, 2.0, 0.0, 0.0, "C"),
        ])
        e = scorer.score(pdb)
        assert e.clash > 0.0

    def test_no_clash_when_far(self, scorer):
        pdb = _pdb([
            (1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            (2, "CA", "GLY", "A", 2, 6.0, 0.0, 0.0, "C"),
        ])
        e = scorer.score(pdb)
        assert e.clash == 0.0

    def test_bonded_distance_excluded_from_clash(self, scorer):
        # 1.33 Å apart (peptide-bond length) must NOT be scored as a clash.
        pdb = _pdb([
            (1, "C", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            (2, "N", "GLY", "A", 2, 1.33, 0.0, 0.0, "N"),
        ])
        e = scorer.score(pdb)
        assert e.clash == 0.0

    def test_entropy_from_residue_identity(self, scorer):
        # sc_entropy is the sum of per-residue side-chain entropies.
        ala = _pdb([(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")])
        lys = _pdb([(1, "CA", "LYS", "A", 1, 0.0, 0.0, 0.0, "C")])
        e_ala = scorer.score(ala)
        e_lys = scorer.score(lys)
        # Reported sc_entropy is weight × Σ per-residue entropy.
        w = scorer.weights["sc_entropy"]
        assert e_ala.sc_entropy == pytest.approx(w * P.SIDECHAIN_ENTROPY["ALA"], abs=1e-3)
        assert e_lys.sc_entropy == pytest.approx(w * P.SIDECHAIN_ENTROPY["LYS"], abs=1e-3)
        assert e_lys.sc_entropy > e_ala.sc_entropy

    def test_custom_weights_scale_terms(self, tmp_path):
        base = EmpiricalScorer(work_dir=str(tmp_path / "a"))
        zeroed = EmpiricalScorer(
            work_dir=str(tmp_path / "b"),
            weights={"sc_entropy": 0.0, "mc_entropy": 0.0},
        )
        e0 = base.score(PEPTIDE_PDB)
        ez = zeroed.score(PEPTIDE_PDB)
        assert e0.mc_entropy > 0 and ez.mc_entropy == 0.0


# ===================================================================
# score_mutation contract
# ===================================================================

@requires_gemmi
class TestScoreMutation:
    def test_identical_structures_zero_ddg(self, scorer):
        result = scorer.score_mutation(PEPTIDE_PDB, PEPTIDE_PDB, ["A1S"])
        assert list(result.ddg.values())[0] == 0.0

    def test_energy_terms_schema(self, scorer):
        result = scorer.score_mutation(PEPTIDE_PDB, PEPTIDE_PDB, ["A1S"])
        df = result.energy_terms
        assert df.columns == ["term", "wt_energy", "mutant_energy", "delta"]
        assert "total" in df["term"].to_list()

    def test_label_from_mutations(self, scorer):
        result = scorer.score_mutation(PEPTIDE_PDB, PEPTIDE_PDB, ["A1S", "G2A"])
        assert "A1S+G2A" in result.ddg

    def test_mutant_pdb_preserved(self, scorer):
        result = scorer.score_mutation(PEPTIDE_PDB, PEPTIDE_PDB, ["A1S"])
        assert "ATOM" in list(result.mutant_pdbs.values())[0]
