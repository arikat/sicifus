"""Alanine dipeptide validation: AMBER ff14SB + GBn2 conformational energetics.

Alanine dipeptide (Ace-Ala-Nme) is the standard benchmark for force-field
validation.  These tests verify that our OpenMM pipeline reproduces the
known conformational energy landscape under AMBER ff14SB with GBn2 implicit
solvent.

Key reference behaviour (AMBER ff + GB):
  - αR (right-handed helix) is competitive with or lower than C7eq because
    GBn2 screens the intramolecular C7eq hydrogen bond.
  - C5 (extended) is slightly higher.
  - C7ax / αL are the highest-energy standard basins.
  - All energies should be negative (bound system).
  - The spread across basins is ~1–4 kcal/mol, NOT the 10+ kcal/mol
    seen in gas-phase calculations.

Marked ``slow`` because they require OpenMM + PDBFixer.

Run:  pytest tests/test_alanine_dipeptide.py -v
"""

import io
import math

import numpy as np
import pytest

slow = pytest.mark.slow


def _openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
        from pdbfixer import PDBFixer  # noqa: F401
        return True
    except ImportError:
        return False


requires_openmm = pytest.mark.skipif(
    not _openmm_available(), reason="OpenMM / PDBFixer not installed"
)

ALA_DIPEPTIDE_PDB = """\
ATOM      1 HH31 ACE A   1       2.000   1.000   0.000  1.00  0.00           H
ATOM      2  CH3 ACE A   1       2.000   2.090   0.000  1.00  0.00           C
ATOM      3 HH32 ACE A   1       1.486   2.454   0.890  1.00  0.00           H
ATOM      4 HH33 ACE A   1       1.486   2.454  -0.890  1.00  0.00           H
ATOM      5  C   ACE A   1       3.427   2.641   0.000  1.00  0.00           C
ATOM      6  O   ACE A   1       4.391   1.877   0.000  1.00  0.00           O
ATOM      7  N   ALA A   2       3.555   3.970   0.000  1.00  0.00           N
ATOM      8  H   ALA A   2       2.733   4.556   0.000  1.00  0.00           H
ATOM      9  CA  ALA A   2       4.853   4.614   0.000  1.00  0.00           C
ATOM     10  HA  ALA A   2       5.408   4.316   0.890  1.00  0.00           H
ATOM     11  CB  ALA A   2       5.661   4.221  -1.232  1.00  0.00           C
ATOM     12  HB1 ALA A   2       5.123   4.521  -2.131  1.00  0.00           H
ATOM     13  HB2 ALA A   2       6.630   4.719  -1.206  1.00  0.00           H
ATOM     14  HB3 ALA A   2       5.809   3.141  -1.241  1.00  0.00           H
ATOM     15  C   ALA A   2       4.713   6.129   0.000  1.00  0.00           C
ATOM     16  O   ALA A   2       3.601   6.653   0.000  1.00  0.00           O
ATOM     17  N   NME A   3       5.846   6.835   0.000  1.00  0.00           N
ATOM     18  H   NME A   3       6.737   6.359   0.000  1.00  0.00           H
ATOM     19  CH3 NME A   3       5.846   8.284   0.000  1.00  0.00           C
ATOM     20 HH31 NME A   3       4.819   8.648   0.000  1.00  0.00           H
ATOM     21 HH32 NME A   3       6.360   8.648   0.890  1.00  0.00           H
ATOM     22 HH33 NME A   3       6.360   8.648  -0.890  1.00  0.00           H
TER
END
"""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(axis, angle_deg):
    angle = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    norm1, norm2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    n1 /= norm1
    n2 /= norm2
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(-np.dot(m, n2), np.dot(n1, n2)))


def _set_dihedral(pos, atoms_to_rotate, pivot, axis_end, target,
                  p1, p2, p3, p4):
    pos = pos.copy()
    delta = target - _dihedral(pos[p1], pos[p2], pos[p3], pos[p4])
    R = _rotation_matrix(pos[axis_end] - pos[pivot], delta)
    origin = pos[pivot]
    for idx in atoms_to_rotate:
        pos[idx] = R @ (pos[idx] - origin) + origin
    return pos


def _positions_to_numpy(positions):
    return np.array([[p.x, p.y, p.z] for p in positions]) * 10.0


def _numpy_to_positions(arr_ang):
    from openmm import Vec3
    from openmm import unit as u
    arr_nm = arr_ang / 10.0
    return [Vec3(float(r[0]), float(r[1]), float(r[2]))
            for r in arr_nm] * u.nanometers


# ---------------------------------------------------------------------------
# Fixture: build the alanine dipeptide system once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ala_system():
    """Return (topology, base_positions_angstrom, atom_index, ff)."""
    from openmm.app import ForceField, Modeller, NoCutoff, HBonds
    from pdbfixer import PDBFixer

    fixer = PDBFixer(pdbfile=io.StringIO(ALA_DIPEPTIDE_PDB))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(ff, pH=7.0)

    topology = modeller.getTopology()
    positions = _positions_to_numpy(modeller.getPositions())

    atom_idx = {}
    for atom in topology.atoms():
        atom_idx[(atom.residue.name, atom.name)] = atom.index

    return topology, positions, atom_idx, ff


def _minimize_at(ala_system, target_phi, target_psi, max_iter=10000):
    """Set phi/psi, minimise, return (energy_kcal, final_phi, final_psi)."""
    from openmm.app import NoCutoff, HBonds
    from openmm import (unit, LangevinMiddleIntegrator, Context,
                        Platform, LocalEnergyMinimizer)

    topology, base_pos, ai, ff = ala_system

    i_C_ace = ai[("ACE", "C")]
    i_N_ala = ai[("ALA", "N")]
    i_CA = ai[("ALA", "CA")]
    i_C_ala = ai[("ALA", "C")]
    i_O_ala = ai[("ALA", "O")]
    i_N_nme = ai[("NME", "N")]

    phi_ds = [v for (r, n), v in ai.items()
              if (r == "ALA" and n not in ("N", "H")) or r == "NME"]
    psi_ds = [i_C_ala, i_O_ala] + [v for (r, _), v in ai.items() if r == "NME"]

    pos = _set_dihedral(base_pos, phi_ds, i_N_ala, i_CA,
                        target_phi, i_C_ace, i_N_ala, i_CA, i_C_ala)
    pos = _set_dihedral(pos, psi_ds, i_CA, i_C_ala,
                        target_psi, i_N_ala, i_CA, i_C_ala, i_N_nme)

    system = ff.createSystem(topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds)
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    ctx = Context(system, integrator, Platform.getPlatformByName("CPU"))
    ctx.setPositions(_numpy_to_positions(pos))
    LocalEnergyMinimizer.minimize(ctx, maxIterations=max_iter)

    state = ctx.getState(getEnergy=True, getPositions=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    fpos = _positions_to_numpy(state.getPositions())
    phi = _dihedral(fpos[i_C_ace], fpos[i_N_ala], fpos[i_CA], fpos[i_C_ala])
    psi = _dihedral(fpos[i_N_ala], fpos[i_CA], fpos[i_C_ala], fpos[i_N_nme])
    return e, phi, psi


# ===================================================================
# Validation tests
# ===================================================================


@slow
@requires_openmm
class TestAlanineDipeptideConformations:
    """Validate AMBER ff14SB + GBn2 conformational energetics on ala-dipeptide."""

    def test_all_energies_negative(self, ala_system):
        """Every minimised conformer should have negative potential energy."""
        for phi, psi in [(-150, 160), (-60, -45), (73, -53), (60, 45)]:
            e, _, _ = _minimize_at(ala_system, phi, psi)
            assert e < 0, f"Energy should be negative, got {e:.2f} at φ={phi},ψ={psi}"

    def test_alpha_r_is_global_minimum(self, ala_system):
        """Under GBn2, αR should be the global minimum (or within 1 kcal/mol).

        GBn2 screens the intramolecular C7eq H-bond, stabilising the more
        exposed helical conformation.
        """
        e_ar, _, _ = _minimize_at(ala_system, -60, -45)
        e_c5, _, _ = _minimize_at(ala_system, -150, 160)
        e_c7ax, _, _ = _minimize_at(ala_system, 73, -53)
        assert e_ar <= e_c5 + 0.5, (
            f"αR ({e_ar:.2f}) should be ≤ C5 ({e_c5:.2f}) + 0.5 kcal/mol"
        )
        assert e_ar < e_c7ax, (
            f"αR ({e_ar:.2f}) should be lower than C7ax ({e_c7ax:.2f})"
        )

    def test_c5_extended_slightly_higher(self, ala_system):
        """C5 (extended) should be 0–3 kcal/mol above the global minimum."""
        e_ar, _, _ = _minimize_at(ala_system, -60, -45)
        e_c5, _, _ = _minimize_at(ala_system, -150, 160)
        de = e_c5 - e_ar
        assert 0 <= de < 3.0, (
            f"C5 should be 0–3 kcal/mol above αR, got ΔE={de:.2f}"
        )

    def test_c7ax_higher_than_alpha_r(self, ala_system):
        """C7ax/αL basins should be 0.5–5 kcal/mol above the global minimum."""
        e_ar, _, _ = _minimize_at(ala_system, -60, -45)
        e_c7ax, _, _ = _minimize_at(ala_system, 73, -53)
        de = e_c7ax - e_ar
        assert 0.5 < de < 5.0, (
            f"C7ax should be 0.5–5 kcal/mol above αR, got ΔE={de:.2f}"
        )

    def test_energy_spread_within_implicit_solvent_range(self, ala_system):
        """Total spread across all basins should be < 5 kcal/mol.

        In gas phase the spread is ~12 kcal/mol; implicit solvent compresses
        it dramatically.  If we see > 5 kcal/mol, something is wrong with
        the solvation model.
        """
        energies = []
        for phi, psi in [(-150, 160), (-60, -45), (73, -53), (60, 45)]:
            e, _, _ = _minimize_at(ala_system, phi, psi)
            energies.append(e)
        spread = max(energies) - min(energies)
        assert spread < 5.0, (
            f"Energy spread should be < 5 kcal/mol in implicit solvent, "
            f"got {spread:.2f}"
        )

    def test_c7eq_converges_to_alpha_r_basin(self, ala_system):
        """C7eq starting point should converge near the αR basin under GBn2.

        The C7eq minimum (φ≈-83°, ψ≈73°) is shallow under GBn2.  Minimisation
        should drive it toward the deeper αR basin (φ≈-60 to -80°, ψ≈-15 to -45°).
        """
        _, phi, psi = _minimize_at(ala_system, -83, 73)
        assert -100 < phi < -40, f"Final φ should be in helical range, got {phi:.1f}"
        assert -60 < psi < 30, f"Final ψ should be near αR, got {psi:.1f}"

    def test_extended_stays_extended(self, ala_system):
        """C5 (extended) starting point should remain in the extended basin."""
        _, phi, psi = _minimize_at(ala_system, -150, 160)
        assert phi < -100, f"Extended should keep φ < -100°, got {phi:.1f}"
        assert psi > 100, f"Extended should keep ψ > 100°, got {psi:.1f}"


@slow
@requires_openmm
class TestAlanineDipeptideEnergyDecomposition:
    """Verify that energy decomposition works on the simplest system."""

    def test_decomposition_has_all_terms(self, ala_system):
        """The energy decomposition should include all standard AMBER terms."""
        from sicifus.mutate import MutationEngine
        engine = MutationEngine(work_dir="/tmp/sicifus_test_aladip")
        result = engine.calculate_stability(ALA_DIPEPTIDE_PDB, max_iterations=5000)

        expected_terms = {
            "HarmonicBondForce", "HarmonicAngleForce",
            "PeriodicTorsionForce", "NonbondedForce",
            "CustomGBForce", "total",
        }
        for term in expected_terms:
            assert term in result.energy_terms, f"Missing term: {term}"

    def test_total_energy_negative(self, ala_system):
        """Minimised alanine dipeptide should have negative total energy."""
        from sicifus.mutate import MutationEngine
        engine = MutationEngine(work_dir="/tmp/sicifus_test_aladip")
        result = engine.calculate_stability(ALA_DIPEPTIDE_PDB, max_iterations=5000)
        assert result.total_energy < 0, (
            f"Total energy should be negative, got {result.total_energy}"
        )

    def test_terms_sum_to_total(self, ala_system):
        """Sum of individual force terms should match the reported total."""
        from sicifus.mutate import MutationEngine
        engine = MutationEngine(work_dir="/tmp/sicifus_test_aladip")
        result = engine.calculate_stability(ALA_DIPEPTIDE_PDB, max_iterations=5000)
        partial_sum = sum(
            v for k, v in result.energy_terms.items() if k != "total"
        )
        assert abs(partial_sum - result.total_energy) < 1.0, (
            f"Term sum ({partial_sum:.2f}) should match total "
            f"({result.total_energy:.2f})"
        )

    def test_solvation_is_favorable(self, ala_system):
        """CustomGBForce (GBn2 solvation) should contribute negative energy."""
        from sicifus.mutate import MutationEngine
        engine = MutationEngine(work_dir="/tmp/sicifus_test_aladip")
        result = engine.calculate_stability(ALA_DIPEPTIDE_PDB, max_iterations=5000)
        gb = result.energy_terms.get("CustomGBForce", 0.0)
        assert gb < 0, f"GBn2 solvation should be favorable (negative), got {gb}"

    def test_bond_energy_is_small(self, ala_system):
        """After minimisation, bond stretch energy should be very small."""
        from sicifus.mutate import MutationEngine
        engine = MutationEngine(work_dir="/tmp/sicifus_test_aladip")
        result = engine.calculate_stability(ALA_DIPEPTIDE_PDB, max_iterations=5000)
        bonds = result.energy_terms.get("HarmonicBondForce", 0.0)
        assert abs(bonds) < 5.0, (
            f"Minimised bond energy should be near zero, got {bonds:.2f} kcal/mol"
        )


@slow
@requires_openmm
class TestAlanineDipeptideReproducibility:
    """Verify that the minimisation is deterministic for this simple system."""

    def test_energy_reproducible(self, ala_system):
        """Two independent minimisations from the same start should agree.

        Alanine dipeptide has no ambiguous hydrogen placement (all atoms
        specified), so the noise floor should be near zero.
        """
        e1, _, _ = _minimize_at(ala_system, -60, -45)
        e2, _, _ = _minimize_at(ala_system, -60, -45)
        assert abs(e1 - e2) < 0.01, (
            f"Same start should give same energy, diff={abs(e1 - e2):.4f}"
        )

    def test_phi_psi_reproducible(self, ala_system):
        """Final phi/psi should be nearly identical across runs.

        The αR basin is shallow, so small numerical differences in the
        minimiser can shift φ/ψ by up to ~1° while the energy stays
        within 0.01 kcal/mol.  1° tolerance is appropriate.
        """
        _, phi1, psi1 = _minimize_at(ala_system, -60, -45)
        _, phi2, psi2 = _minimize_at(ala_system, -60, -45)
        assert abs(phi1 - phi2) < 1.0, f"φ not reproducible: {phi1:.2f} vs {phi2:.2f}"
        assert abs(psi1 - psi2) < 1.0, f"ψ not reproducible: {psi1:.2f} vs {psi2:.2f}"
