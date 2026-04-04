import io
import re
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
STANDARD_RESIDUES = set(THREE_TO_ONE.keys())
ALL_AMINO_ACIDS = sorted(THREE_TO_ONE.keys())


@dataclass
class Mutation:
    """Describes a single point mutation.

    Args:
        position: Residue number in the structure.
        wt_residue: Wild-type residue (1-letter or 3-letter code).
        mut_residue: Mutant residue (1-letter or 3-letter code).
        chain: Chain identifier (default ``"A"``).
    """
    position: int
    wt_residue: str
    mut_residue: str
    chain: str = "A"

    def __post_init__(self):
        self.wt_residue = self._normalise(self.wt_residue)
        self.mut_residue = self._normalise(self.mut_residue)

    @staticmethod
    def _normalise(code: str) -> str:
        code = code.strip().upper()
        if len(code) == 1 and code in ONE_TO_THREE:
            return ONE_TO_THREE[code]
        if len(code) == 3 and code in STANDARD_RESIDUES:
            return code
        raise ValueError(f"Unknown residue code: {code!r}")

    @property
    def label(self) -> str:
        """Short label like ``G13L``."""
        wt1 = THREE_TO_ONE[self.wt_residue]
        mt1 = THREE_TO_ONE[self.mut_residue]
        return f"{wt1}{self.position}{mt1}"

    @classmethod
    def from_str(cls, notation: str, chain: str = "A") -> "Mutation":
        """Parse a mutation string like ``'G13L'`` (Gly at position 13 to Leu).

        Args:
            notation: String in the format ``WtPositionMut``
                      (e.g. ``'G13L'``, ``'F42W'``, ``'A100V'``).
            chain: Chain identifier (default ``'A'``).
        """
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", notation.strip().upper())
        if not m:
            raise ValueError(
                f"Invalid mutation string: {notation!r}. "
                "Expected format: WtPositionMut (e.g. G13L)"
            )
        wt_one, pos, mut_one = m.groups()
        return cls(
            position=int(pos),
            wt_residue=ONE_TO_THREE[wt_one],
            mut_residue=ONE_TO_THREE[mut_one],
            chain=chain.upper(),
        )

    def __repr__(self):
        if self.chain != "A":
            return f"Mutation({self.label}, chain={self.chain})"
        return f"Mutation({self.label})"


@dataclass
class _RepairCache:
    """Pre-processed wild-type structure for the repair-once pattern.

    Holds a fully protonated, minimised WT so that batch mutations
    skip redundant hydrogen placement and WT minimisations.
    """
    pdb_string: str
    topology: object
    positions: object
    system: object
    energy_kj: float
    energy_kcal: float


@dataclass
class RepairResult:
    topology: object
    positions: object
    energy_before: float
    energy_after: float
    pdb_string: str


@dataclass
class StabilityResult:
    total_energy: float
    energy_terms: Dict[str, float]
    pdb_string: str


@dataclass
class MutationResult:
    wt_energy: float
    mutant_energies: Dict[str, float]
    ddg: Dict[str, float]
    mutant_pdbs: Dict[str, str]
    energy_terms: pl.DataFrame

    # Statistical fields (all optional for backward compatibility)
    all_run_energies: Optional[Dict[str, List[float]]] = None
    mean_energy: Optional[Dict[str, float]] = None
    sd_energy: Optional[Dict[str, float]] = None
    min_energy: Optional[Dict[str, float]] = None
    max_energy: Optional[Dict[str, float]] = None
    ddg_mean: Optional[Dict[str, float]] = None
    ddg_sd: Optional[Dict[str, float]] = None
    ddg_ci_95: Optional[Dict[str, Tuple[float, float]]] = None
    convergence_metric: Optional[Dict[str, float]] = None


@dataclass
class BindingResult:
    binding_energy: float
    complex_energy: float
    chain_a_energy: float
    chain_b_energy: float
    interface_residues: pl.DataFrame


@dataclass
class InterfaceMutationResult:
    """Result from mutating residues at a protein-protein interface.

    Includes both stability changes (ΔΔG per chain) and binding affinity
    changes (ΔΔG_binding).
    """
    wt_binding_energy: float
    mutant_binding_energy: float
    ddg_binding: float
    wt_complex_energy: float
    mutant_complex_energy: float
    wt_chain_a_energy: float
    wt_chain_b_energy: float
    mutant_chain_a_energy: float
    mutant_chain_b_energy: float
    ddg_stability_a: float
    ddg_stability_b: float
    interface_residues: pl.DataFrame
    mutations_by_chain: Dict[str, List[Mutation]]
    mutant_pdb: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topology_to_pdb(topology, positions) -> str:
    from openmm.app import PDBFile
    buf = io.StringIO()
    PDBFile.writeFile(topology, positions, buf)
    return buf.getvalue()


def _pdb_string_to_fixer(pdb_string: str):
    from pdbfixer import PDBFixer
    fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
    return fixer


def _df_to_pdb_string(df: pl.DataFrame) -> str:
    """Convert a Polars atom DataFrame to a PDB-format string."""
    lines = []
    for i, row in enumerate(df.iter_rows(named=True)):
        atom_name = str(row.get("atom_name", "X")).strip()
        res_name = str(row.get("residue_name", "UNK")).strip()[:3]
        chain_id = str(row.get("chain", "A")).strip()[:1]
        res_seq = int(row.get("residue_number", 1))
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        elem = str(row.get("element", atom_name[0])).strip().upper()

        if len(atom_name) >= 4:
            aname_fmt = f"{atom_name[:4]}"
        elif len(elem) == 2:
            aname_fmt = f"{atom_name:<4}"
        else:
            aname_fmt = f" {atom_name:<3}"

        line = (
            f"ATOM  {i+1:>5} {aname_fmt:<4} {res_name:<3} {chain_id:>1}{res_seq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2}"
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


def _load_pdb(source) -> str:
    """Accept a file path, a PDB string, or a Polars DataFrame and return PDB text."""
    if isinstance(source, pl.DataFrame):
        return _df_to_pdb_string(source)
    if isinstance(source, str) and ("ATOM" in source or "HETATM" in source):
        return source
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists():
            return p.read_text()
    raise ValueError(
        "source must be a file path, a PDB string, or a Polars DataFrame of atoms"
    )


def _extract_chains(topology, positions, chain_ids: List[str]):
    """Return a new (topology, positions) containing only the requested chains."""
    from openmm.app import Topology
    from openmm import Vec3
    import openmm.unit as unit

    new_top = Topology()
    new_positions = []
    chain_map = {}
    residue_map = {}

    for chain in topology.chains():
        if chain.id in chain_ids:
            new_chain = new_top.addChain(chain.id)
            chain_map[chain.index] = new_chain

            for residue in chain.residues():
                new_res = new_top.addResidue(residue.name, new_chain, residue.id)
                residue_map[residue.index] = new_res

                for atom in residue.atoms():
                    new_top.addAtom(atom.name, atom.element, new_res)
                    new_positions.append(positions[atom.index])

    for bond in topology.bonds():
        a1, a2 = bond
        r1_idx = a1.residue.index
        r2_idx = a2.residue.index
        if r1_idx in residue_map and r2_idx in residue_map:
            new_a1_candidates = list(residue_map[r1_idx].atoms())
            new_a2_candidates = list(residue_map[r2_idx].atoms())
            a1_match = [a for a in new_a1_candidates if a.name == a1.name]
            a2_match = [a for a in new_a2_candidates if a.name == a2.name]
            if a1_match and a2_match:
                new_top.addBond(a1_match[0], a2_match[0])

    new_positions = new_positions * unit.nanometers if not hasattr(new_positions[0], 'unit') else new_positions
    return new_top, new_positions


def _find_interface_residues(topology, positions, chain_ids_a, chain_ids_b,
                             cutoff_nm: float = 0.5) -> pl.DataFrame:
    """Identify residues at the interface between two chain groups."""
    from openmm import Vec3

    atoms_a = []
    atoms_b = []
    for chain in topology.chains():
        for residue in chain.residues():
            for atom in residue.atoms():
                pos = positions[atom.index]
                entry = {
                    "chain": chain.id,
                    "residue_number": int(residue.id),
                    "residue_name": residue.name,
                    "x": pos[0].value_in_unit(pos[0].unit) if hasattr(pos[0], 'value_in_unit') else float(pos[0]),
                    "y": pos[1].value_in_unit(pos[1].unit) if hasattr(pos[1], 'value_in_unit') else float(pos[1]),
                    "z": pos[2].value_in_unit(pos[2].unit) if hasattr(pos[2], 'value_in_unit') else float(pos[2]),
                }
                if chain.id in chain_ids_a:
                    atoms_a.append(entry)
                elif chain.id in chain_ids_b:
                    atoms_b.append(entry)

    if not atoms_a or not atoms_b:
        return pl.DataFrame(schema={"chain": pl.Utf8, "residue_number": pl.Int64,
                                     "residue_name": pl.Utf8, "side": pl.Utf8})

    coords_a = np.array([[a["x"], a["y"], a["z"]] for a in atoms_a])
    coords_b = np.array([[a["x"], a["y"], a["z"]] for a in atoms_b])

    from scipy.spatial.distance import cdist
    dists = cdist(coords_a, coords_b)

    interface_rows = []
    seen = set()
    for i in range(len(atoms_a)):
        if np.min(dists[i]) < cutoff_nm:
            key = (atoms_a[i]["chain"], atoms_a[i]["residue_number"])
            if key not in seen:
                seen.add(key)
                interface_rows.append({**atoms_a[i], "side": "A"})
    for j in range(len(atoms_b)):
        if np.min(dists[:, j]) < cutoff_nm:
            key = (atoms_b[j]["chain"], atoms_b[j]["residue_number"])
            if key not in seen:
                seen.add(key)
                interface_rows.append({**atoms_b[j], "side": "B"})

    if not interface_rows:
        return pl.DataFrame(schema={"chain": pl.Utf8, "residue_number": pl.Int64,
                                     "residue_name": pl.Utf8, "side": pl.Utf8})

    idf = pl.DataFrame(interface_rows)
    return idf.select(["chain", "residue_number", "residue_name", "side"]).unique()


def _compute_energy_statistics(
    all_energies: List[float],
    wt_energy: float,
    n_runs: int
) -> Dict[str, any]:
    """Compute statistical summary from multiple minimization runs.

    Returns dict with: mean, sd, min, max, ddg_mean, ddg_sd, ci_95, cv
    """
    import scipy.stats as stats

    energies = np.array(all_energies)
    mean_e = float(np.mean(energies))
    sd_e = float(np.std(energies, ddof=1)) if n_runs > 1 else 0.0

    ddg_mean = mean_e - wt_energy

    # 95% CI using t-distribution
    if n_runs > 1:
        ci = stats.t.interval(0.95, n_runs - 1, loc=ddg_mean,
                              scale=stats.sem(energies))
    else:
        ci = (ddg_mean, ddg_mean)

    return {
        "mean": round(mean_e, 4),
        "sd": round(sd_e, 4),
        "min": round(float(np.min(energies)), 4),
        "max": round(float(np.max(energies)), 4),
        "ddg_mean": round(ddg_mean, 4),
        "ddg_sd": round(sd_e, 4),
        "ci_95": (round(ci[0], 4), round(ci[1], 4)),
        "cv": round(sd_e / abs(mean_e), 4) if mean_e != 0 else 0.0,
    }


def _detect_disulfide_bonds(topology, positions, distance_cutoff: float = 2.5) -> pl.DataFrame:
    """Detect disulfide bonds based on S-S distance.

    Args:
        topology: OpenMM topology
        positions: Atomic positions
        distance_cutoff: Maximum S-S distance for disulfide bond (Å, default 2.5)

    Returns:
        DataFrame with columns: [chain1, residue1, resname1, chain2, residue2, resname2, distance]
    """
    # Find all cysteine sulfur atoms
    cys_sulfurs = []

    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == "CYS":
                for atom in residue.atoms():
                    if atom.name == "SG":  # Sulfur atom in cysteine
                        pos = positions[atom.index]
                        x = pos[0].value_in_unit(pos[0].unit) if hasattr(pos[0], 'value_in_unit') else float(pos[0])
                        y = pos[1].value_in_unit(pos[1].unit) if hasattr(pos[1], 'value_in_unit') else float(pos[1])
                        z = pos[2].value_in_unit(pos[2].unit) if hasattr(pos[2], 'value_in_unit') else float(pos[2])

                        cys_sulfurs.append({
                            "chain": chain.id,
                            "residue_number": int(residue.id),
                            "residue_name": residue.name,
                            "coords": np.array([x, y, z])
                        })

    if len(cys_sulfurs) < 2:
        return pl.DataFrame(schema={
            "chain1": pl.Utf8, "residue1": pl.Int64, "resname1": pl.Utf8,
            "chain2": pl.Utf8, "residue2": pl.Int64, "resname2": pl.Utf8,
            "distance": pl.Float64
        })

    # Find pairs within distance cutoff
    disulfide_bonds = []
    cutoff_nm = distance_cutoff / 10.0  # Convert Å to nm

    for i in range(len(cys_sulfurs)):
        for j in range(i + 1, len(cys_sulfurs)):
            cys1 = cys_sulfurs[i]
            cys2 = cys_sulfurs[j]

            dist = np.linalg.norm(cys1["coords"] - cys2["coords"])

            if dist < cutoff_nm:
                disulfide_bonds.append({
                    "chain1": cys1["chain"],
                    "residue1": cys1["residue_number"],
                    "resname1": cys1["residue_name"],
                    "chain2": cys2["chain"],
                    "residue2": cys2["residue_number"],
                    "resname2": cys2["residue_name"],
                    "distance": round(dist * 10.0, 3),  # Convert back to Å
                })

    if not disulfide_bonds:
        return pl.DataFrame(schema={
            "chain1": pl.Utf8, "residue1": pl.Int64, "resname1": pl.Utf8,
            "chain2": pl.Utf8, "residue2": pl.Int64, "resname2": pl.Utf8,
            "distance": pl.Float64
        })

    return pl.DataFrame(disulfide_bonds)


# ---------------------------------------------------------------------------
# MutationEngine
# ---------------------------------------------------------------------------

class MutationEngine:
    """
    Industry-standard protein mutation and stability engine using OpenMM + PDBFixer.

    Provides structure repair, in silico mutagenesis, stability scoring,
    binding energy calculation, alanine scanning, and positional scanning
    without requiring the commercial protein design tools.
    """

    HARTREE_TO_KCAL = 627.509
    KJ_TO_KCAL = 1.0 / 4.184

    def __init__(
        self,
        forcefield: str = "amber14-all.xml",
        water_model: str = "implicit",
        platform: str = "CPU",
        work_dir: str = "./mutate_work",
    ):
        """
        Args:
            forcefield: OpenMM force field XML (default AMBER14).
            water_model: 'implicit' for GBn2 implicit solvent (fast, industry-standard)
                         or an explicit water XML like 'amber14/tip3pfb.xml'.
            platform: OpenMM platform ('CPU', 'CUDA', 'OpenCL').
            work_dir: Directory for temporary files.
        """
        self.forcefield_name = forcefield
        self.water_model = water_model
        self.platform_name = platform
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        self._ff = None
        self._platform = None

    def _get_forcefield(self):
        if self._ff is None:
            from openmm.app import ForceField
            if self.water_model == "implicit":
                self._ff = ForceField(self.forcefield_name, "implicit/gbn2.xml")
            else:
                self._ff = ForceField(self.forcefield_name, self.water_model)
        return self._ff

    def _get_platform(self):
        if self._platform is None:
            from openmm import Platform
            self._platform = Platform.getPlatformByName(self.platform_name)
        return self._platform

    # ------------------------------------------------------------------
    # Core: create an OpenMM system and minimise
    # ------------------------------------------------------------------

    def _build_system(self, topology, positions, constrain_backbone: bool = False,
                       skip_hydrogens: bool = False):
        """Create an OpenMM System from topology/positions.

        Returns (system, topology, positions) — topology/positions may have
        been modified (hydrogens added via Modeller).

        Args:
            skip_hydrogens: If True, skip ``addHydrogens`` — use when the
                            structure is already fully protonated (e.g. from a
                            repair cache).
        """
        from openmm.app import Modeller, ForceField
        from openmm import app as app_mod
        import openmm
        import openmm.unit as unit

        ff = self._get_forcefield()

        modeller = Modeller(topology, positions)
        if not skip_hydrogens:
            modeller.addHydrogens(ff, pH=7.0)
        topology = modeller.getTopology()
        positions = modeller.getPositions()

        if self.water_model == "implicit":
            system = ff.createSystem(
                topology,
                nonbondedMethod=app_mod.NoCutoff,
                constraints=app_mod.HBonds,
            )
        else:
            system = ff.createSystem(
                topology,
                nonbondedMethod=app_mod.PME,
                nonbondedCutoff=1.0 * unit.nanometers,
                constraints=app_mod.HBonds,
            )

        if constrain_backbone:
            from openmm import CustomExternalForce
            import openmm.unit as u
            restraint = CustomExternalForce(
                "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
            )
            restraint.addGlobalParameter("k", 1000.0 * u.kilojoules_per_mole / u.nanometers**2)
            restraint.addPerParticleParameter("x0")
            restraint.addPerParticleParameter("y0")
            restraint.addPerParticleParameter("z0")

            bb_names = {"CA", "C", "N", "O"}
            for atom in topology.atoms():
                if atom.name in bb_names:
                    pos_i = positions[atom.index]
                    x0 = pos_i[0].value_in_unit(u.nanometers) if hasattr(pos_i[0], 'value_in_unit') else float(pos_i[0])
                    y0 = pos_i[1].value_in_unit(u.nanometers) if hasattr(pos_i[1], 'value_in_unit') else float(pos_i[1])
                    z0 = pos_i[2].value_in_unit(u.nanometers) if hasattr(pos_i[2], 'value_in_unit') else float(pos_i[2])
                    restraint.addParticle(atom.index, [x0, y0, z0])
            system.addForce(restraint)

        return system, topology, positions

    def _minimise(self, system, topology, positions, max_iterations: int = 500,
                   tolerance: float = 1.0):
        """Energy-minimise and return (positions, energy_kj).

        Energy is returned in kJ/mol as reported by OpenMM.

        Args:
            tolerance: Convergence tolerance in kJ/mol/nm (default 1.0).
        """
        from openmm import LangevinMiddleIntegrator, Context
        import openmm.unit as unit

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        platform = self._get_platform()
        context = Context(system, integrator, platform)
        context.setPositions(positions)

        from openmm import LocalEnergyMinimizer
        LocalEnergyMinimizer.minimize(
            context,
            tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=max_iterations,
        )

        state = context.getState(getPositions=True, getEnergy=True)
        energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return state.getPositions(), energy_kj

    def _decompose_energy(self, system, topology, positions) -> Dict[str, float]:
        """Evaluate each force-group independently and return per-term energies (kcal/mol)."""
        from openmm import LangevinMiddleIntegrator, Context
        import openmm.unit as unit

        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i)

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        platform = self._get_platform()
        context = Context(system, integrator, platform)
        context.setPositions(positions)

        terms: Dict[str, float] = {}
        for i, force in enumerate(system.getForces()):
            state = context.getState(getEnergy=True, groups={i})
            e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            name = type(force).__name__
            terms[name] = round(e_kj * self.KJ_TO_KCAL, 4)

        state_all = context.getState(getEnergy=True)
        total_kj = state_all.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        terms["total"] = round(total_kj * self.KJ_TO_KCAL, 4)
        return terms

    # ------------------------------------------------------------------
    # Prepare (repair-once cache)
    # ------------------------------------------------------------------

    def _prepare_structure(self, source, max_iterations: int = 2000,
                           tolerance: float = 1.0) -> _RepairCache:
        """Protonate and thoroughly minimise a structure once.

        This is the "repair once, mutate from repaired" pattern:
        PDBFixer fills missing atoms/hydrogens, then the structure is
        minimised to convergence.  The resulting cache can be reused
        for many mutations without repeated WT hydrogen placement.
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        system, top, pos = self._build_system(fixer.topology, fixer.positions)
        pos_min, e_kj = self._minimise(system, top, pos,
                                        max_iterations=max_iterations,
                                        tolerance=tolerance)

        pdb_out = _topology_to_pdb(top, pos_min)

        return _RepairCache(
            pdb_string=pdb_out,
            topology=top,
            positions=pos_min,
            system=system,
            energy_kj=e_kj,
            energy_kcal=round(e_kj * self.KJ_TO_KCAL, 4),
        )

    def prepare(self, source, max_iterations: int = 2000,
                tolerance: float = 1.0) -> _RepairCache:
        """Prepare a wild-type structure for efficient batch mutations.

        Protonates, fills missing atoms, and minimises the structure
        thoroughly.  Pass the returned cache to :meth:`mutate` or
        :meth:`mutate_batch` to avoid redundant WT processing.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            max_iterations: Minimisation steps (default 2000).
            tolerance: Convergence tolerance in kJ/mol/nm (default 1.0).

        Returns:
            A ``_RepairCache`` that can be passed to ``mutate()`` and
            ``mutate_batch()`` for deterministic WT energy.
        """
        return self._prepare_structure(source, max_iterations, tolerance)

    # ------------------------------------------------------------------
    # Repair (RepairPDB equivalent)
    # ------------------------------------------------------------------

    def repair(
        self,
        source,
        pH: float = 7.0,
        max_iterations: int = 2000,
    ) -> RepairResult:
        """Repair a structure: fix missing atoms/residues, add hydrogens, minimise.

        Repairs protein structure by fixing clashes and adding missing atoms.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame of atoms.
            pH: pH for protonation (default 7.0).
            max_iterations: Maximum minimisation steps.

        Returns:
            RepairResult with repaired PDB and energy change.
        """
        from pdbfixer import PDBFixer
        import openmm.unit as unit

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH)

        topology = fixer.topology
        positions = fixer.positions

        system_pre, top_pre, pos_pre = self._build_system(topology, positions)
        _, e_before = self._minimise(system_pre, top_pre, pos_pre, max_iterations=0)

        system, top, pos = self._build_system(topology, positions)
        pos_min, e_after = self._minimise(system, top, pos, max_iterations)

        pdb_out = _topology_to_pdb(top, pos_min)

        return RepairResult(
            topology=top,
            positions=pos_min,
            energy_before=round(e_before * self.KJ_TO_KCAL, 4),
            energy_after=round(e_after * self.KJ_TO_KCAL, 4),
            pdb_string=pdb_out,
        )

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def calculate_stability(
        self,
        source,
        max_iterations: int = 2000,
    ) -> StabilityResult:
        """Calculate total potential energy with per-term decomposition.

        Calculates protein stability using energy minimization.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            max_iterations: Minimisation steps before scoring.

        Returns:
            StabilityResult with total energy and per-force-term breakdown (kcal/mol).
        """
        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        system, top, pos = self._build_system(fixer.topology, fixer.positions)
        pos_min, _ = self._minimise(system, top, pos, max_iterations)

        terms = self._decompose_energy(system, top, pos_min)
        pdb_out = _topology_to_pdb(top, pos_min)

        return StabilityResult(
            total_energy=terms.get("total", 0.0),
            energy_terms=terms,
            pdb_string=pdb_out,
        )

    # ------------------------------------------------------------------
    # Mutate (BuildModel equivalent)
    # ------------------------------------------------------------------

    def mutate(
        self,
        source,
        mutations: List[Union[Mutation, str]],
        chain: str = "A",
        n_runs: int = 3,
        max_iterations: int = 2000,
        constrain_backbone: bool = True,
        keep_statistics: bool = True,
        use_mean: bool = False,
        _repair_cache: Optional[_RepairCache] = None,
    ) -> MutationResult:
        """Apply one or more point mutations, minimise, and compute ddG.

        Mutations can be ``Mutation`` objects or short strings like ``'G13L'``.
        Multiple mutations in the same call are applied simultaneously.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
                    Ignored when ``_repair_cache`` is provided.
            mutations: List of Mutation objects or strings (e.g. ``'G13L'``).
            chain: Default chain ID applied when parsing mutation strings
                   (default ``'A'``).
            n_runs: Number of independent minimisation runs for the mutant
                    (default 3).
            max_iterations: Minimisation steps per run (default 2000).
            constrain_backbone: If True, restrain backbone atoms during mutant
                                minimisation, allowing only sidechain flexibility.
            keep_statistics: If True, collect and return statistical summary
                            (mean, SD, CI) from all runs (default True).
            use_mean: If True, use mean energy for ddG calculation (industry-standard).
                     If False, use best (minimum) energy (default False).
            _repair_cache: Pre-processed WT from :meth:`prepare`.  When
                           provided the WT energy comes from the cache,
                           eliminating redundant hydrogen placement and
                           minimisation.

        Returns:
            MutationResult with wild-type energy, mutant energy, ddG,
            mutant PDB strings, and a full energy-term DataFrame. If
            keep_statistics=True, also includes mean, SD, CI, and convergence
            metrics.
        """
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        parsed: List[Mutation] = []
        for m in mutations:
            if isinstance(m, str):
                parsed.append(Mutation.from_str(m, chain=chain))
            else:
                parsed.append(m)

        # --- Wild-type: use cache or build from scratch ---
        if _repair_cache is not None:
            pdb_text = _repair_cache.pdb_string
            e_wt = _repair_cache.energy_kcal
            pos_wt_min = _repair_cache.positions
            top_wt = _repair_cache.topology
            sys_wt = _repair_cache.system
        else:
            pdb_text = _load_pdb(source)
            fixer_wt = _pdb_string_to_fixer(pdb_text)
            fixer_wt.findMissingResidues()
            fixer_wt.findMissingAtoms()
            fixer_wt.addMissingAtoms()
            fixer_wt.addMissingHydrogens(7.0)

            sys_wt, top_wt, pos_wt = self._build_system(
                fixer_wt.topology, fixer_wt.positions)
            pos_wt_min, e_wt_kj = self._minimise(
                sys_wt, top_wt, pos_wt, max_iterations)
            e_wt = round(e_wt_kj * self.KJ_TO_KCAL, 4)
            pdb_text = _topology_to_pdb(top_wt, pos_wt_min)

        # --- Build mutant from the (repaired) WT PDB ---
        fixer_mut = _pdb_string_to_fixer(pdb_text)
        fixer_mut.findMissingResidues()

        pdbfixer_mutations = []
        for m in parsed:
            pdbfixer_mutations.append(f"{m.wt_residue}-{m.position}-{m.mut_residue}")

        chains_used = {m.chain for m in parsed}
        for chain_id in chains_used:
            fixer_mut.applyMutations(pdbfixer_mutations, chain_id)

        fixer_mut.findMissingAtoms()
        fixer_mut.addMissingAtoms()
        fixer_mut.addMissingHydrogens(7.0)

        mut_label = "+".join(m.label for m in parsed)

        best_e_mut = None
        best_pos_mut = None
        best_top_mut = None
        best_sys_mut = None
        all_run_energies_list = []

        for run in range(n_runs):
            sys_m, top_m, pos_m = self._build_system(
                fixer_mut.topology, fixer_mut.positions,
                constrain_backbone=constrain_backbone,
                skip_hydrogens=(_repair_cache is not None),
            )
            pos_m_min, e_m_kj = self._minimise(
                sys_m, top_m, pos_m, max_iterations)
            e_m = e_m_kj * self.KJ_TO_KCAL

            # Store all run energies
            all_run_energies_list.append(e_m)

            if best_e_mut is None or e_m < best_e_mut:
                best_e_mut = e_m
                best_pos_mut = pos_m_min
                best_top_mut = top_m
                best_sys_mut = sys_m

        # Compute primary ddG (based on use_mean flag)
        if use_mean and keep_statistics:
            primary_energy = float(np.mean(all_run_energies_list))
            ddg = primary_energy - e_wt
        else:
            primary_energy = best_e_mut
            ddg = best_e_mut - e_wt

        # Decompose using the exact system/topology that produced the
        # minimised positions (no redundant _build_system call).
        mut_terms = self._decompose_energy(best_sys_mut, best_top_mut, best_pos_mut)
        wt_terms = self._decompose_energy(sys_wt, top_wt, pos_wt_min)

        term_rows = []
        for key in sorted(set(wt_terms.keys()) | set(mut_terms.keys())):
            wt_val = wt_terms.get(key, 0.0)
            mt_val = mut_terms.get(key, 0.0)
            term_rows.append({
                "term": key,
                "wt_energy": wt_val,
                "mutant_energy": mt_val,
                "delta": round(mt_val - wt_val, 4),
            })
        terms_df = pl.DataFrame(term_rows)

        pdb_mut = _topology_to_pdb(best_top_mut, best_pos_mut)

        # Compute statistics if requested
        stats_dict = None
        if keep_statistics and n_runs > 1:
            stats_dict = _compute_energy_statistics(all_run_energies_list, e_wt, n_runs)

            # Warn if convergence is poor
            if stats_dict["cv"] > 0.1:
                print(f"Warning: {mut_label} has high energy variability (CV={stats_dict['cv']:.2f}). "
                      f"Consider increasing n_runs for more reliable results.")

        return MutationResult(
            wt_energy=e_wt,
            mutant_energies={mut_label: round(primary_energy, 4)},
            ddg={mut_label: round(ddg, 4)},
            mutant_pdbs={mut_label: pdb_mut},
            energy_terms=terms_df,
            # Statistical fields
            all_run_energies={mut_label: all_run_energies_list} if keep_statistics else None,
            mean_energy={mut_label: stats_dict["mean"]} if stats_dict else None,
            sd_energy={mut_label: stats_dict["sd"]} if stats_dict else None,
            min_energy={mut_label: stats_dict["min"]} if stats_dict else None,
            max_energy={mut_label: stats_dict["max"]} if stats_dict else None,
            ddg_mean={mut_label: stats_dict["ddg_mean"]} if stats_dict else None,
            ddg_sd={mut_label: stats_dict["ddg_sd"]} if stats_dict else None,
            ddg_ci_95={mut_label: stats_dict["ci_95"]} if stats_dict else None,
            convergence_metric={mut_label: stats_dict["cv"]} if stats_dict else None,
        )

    # ------------------------------------------------------------------
    # Binding energy (AnalyseComplex equivalent)
    # ------------------------------------------------------------------

    def calculate_binding_energy(
        self,
        source,
        chains_a: List[str],
        chains_b: List[str],
        max_iterations: int = 2000,
    ) -> BindingResult:
        """Calculate binding energy between two groups of chains.

        Calculates binding energy for protein-protein complexes.

        E_binding = E_complex - (E_chains_a + E_chains_b)

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            chains_a: Chain IDs for the first group (e.g. ['A']).
            chains_b: Chain IDs for the second group (e.g. ['B']).
            max_iterations: Minimisation steps.

        Returns:
            BindingResult with binding energy, component energies,
            and interface residues.
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # --- Complex energy ---
        sys_c, top_c, pos_c = self._build_system(fixer.topology, fixer.positions)
        pos_c_min, e_complex_kj = self._minimise(sys_c, top_c, pos_c, max_iterations)
        e_complex = e_complex_kj * self.KJ_TO_KCAL

        # Use minimised complex positions to extract chain subsets
        top_a, pos_a = _extract_chains(top_c, pos_c_min, chains_a)
        top_b, pos_b = _extract_chains(top_c, pos_c_min, chains_b)

        sys_a, top_a2, pos_a2 = self._build_system(top_a, pos_a)
        _, e_a_kj = self._minimise(sys_a, top_a2, pos_a2, max_iterations)
        e_a = e_a_kj * self.KJ_TO_KCAL

        sys_b, top_b2, pos_b2 = self._build_system(top_b, pos_b)
        _, e_b_kj = self._minimise(sys_b, top_b2, pos_b2, max_iterations)
        e_b = e_b_kj * self.KJ_TO_KCAL

        e_binding = e_complex - (e_a + e_b)

        interface_df = _find_interface_residues(
            top_c, pos_c_min, chains_a, chains_b, cutoff_nm=0.5
        )

        return BindingResult(
            binding_energy=round(e_binding, 4),
            complex_energy=round(e_complex, 4),
            chain_a_energy=round(e_a, 4),
            chain_b_energy=round(e_b, 4),
            interface_residues=interface_df,
        )

    # ------------------------------------------------------------------
    # Interface mutagenesis (mutation-to-binding pipeline)
    # ------------------------------------------------------------------

    def mutate_interface(
        self,
        source,
        mutations: Dict[str, List[Union[Mutation, str]]],
        chains_a: List[str],
        chains_b: List[str],
        max_iterations: int = 2000,
        n_runs: int = 3,
        constrain_backbone: bool = True,
    ) -> InterfaceMutationResult:
        """Apply mutations to protein-protein interface and compute ΔΔG_binding.

        This is a pipeline that combines mutate() and calculate_binding_energy()
        to automatically compute how mutations affect binding affinity.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame (complex).
            mutations: Dict mapping chain ID to list of mutations.
                      E.g., {"A": ["F13A", "W14L"], "B": ["Y25F"]}
            chains_a: Chain IDs for the first binding partner (e.g. ['A']).
            chains_b: Chain IDs for the second binding partner (e.g. ['B']).
            max_iterations: Minimisation steps.
            n_runs: Number of independent minimisation runs.
            constrain_backbone: Restrain backbone during mutant minimisation.

        Returns:
            InterfaceMutationResult with ΔΔG_binding, ΔΔG_stability per chain,
            and component energies.
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)

        # --- Step 1: Calculate WT binding energy ---
        print("Calculating wild-type binding energy...")
        wt_binding = self.calculate_binding_energy(
            pdb_text, chains_a, chains_b, max_iterations=max_iterations
        )

        # --- Step 2: Apply all mutations to create mutant complex ---
        print("Applying mutations to complex...")
        fixer_mut = _pdb_string_to_fixer(pdb_text)
        fixer_mut.findMissingResidues()

        # Parse mutations
        mutations_by_chain = {}
        for chain_id, mut_list in mutations.items():
            parsed = []
            for m in mut_list:
                if isinstance(m, str):
                    parsed.append(Mutation.from_str(m, chain=chain_id))
                else:
                    parsed.append(m)
            mutations_by_chain[chain_id] = parsed

        # Apply mutations per chain using PDBFixer
        for chain_id, mut_objs in mutations_by_chain.items():
            pdbfixer_mutations = []
            for m in mut_objs:
                pdbfixer_mutations.append(f"{m.wt_residue}-{m.position}-{m.mut_residue}")
            fixer_mut.applyMutations(pdbfixer_mutations, chain_id)

        fixer_mut.findMissingAtoms()
        fixer_mut.addMissingAtoms()
        fixer_mut.addMissingHydrogens(7.0)

        # Minimize mutant complex (best of n_runs)
        best_e_complex = None
        best_pos_complex = None
        best_top_complex = None

        for run in range(n_runs):
            sys_c, top_c, pos_c = self._build_system(
                fixer_mut.topology, fixer_mut.positions,
                constrain_backbone=constrain_backbone
            )
            pos_c_min, e_c_kj = self._minimise(sys_c, top_c, pos_c, max_iterations)
            e_c = e_c_kj * self.KJ_TO_KCAL

            if best_e_complex is None or e_c < best_e_complex:
                best_e_complex = e_c
                best_pos_complex = pos_c_min
                best_top_complex = top_c

        # --- Step 3: Extract and minimize mutant chains separately ---
        print("Calculating mutant component energies...")
        top_a_mut, pos_a_mut = _extract_chains(best_top_complex, best_pos_complex, chains_a)
        top_b_mut, pos_b_mut = _extract_chains(best_top_complex, best_pos_complex, chains_b)

        sys_a_mut, top_a_mut2, pos_a_mut2 = self._build_system(top_a_mut, pos_a_mut)
        _, e_a_mut_kj = self._minimise(sys_a_mut, top_a_mut2, pos_a_mut2, max_iterations)
        e_a_mut = e_a_mut_kj * self.KJ_TO_KCAL

        sys_b_mut, top_b_mut2, pos_b_mut2 = self._build_system(top_b_mut, pos_b_mut)
        _, e_b_mut_kj = self._minimise(sys_b_mut, top_b_mut2, pos_b_mut2, max_iterations)
        e_b_mut = e_b_mut_kj * self.KJ_TO_KCAL

        # --- Step 4: Calculate mutant binding energy ---
        e_binding_mut = best_e_complex - (e_a_mut + e_b_mut)

        # --- Step 5: Compute ΔΔG values ---
        ddg_binding = e_binding_mut - wt_binding.binding_energy
        ddg_stability_a = e_a_mut - wt_binding.chain_a_energy
        ddg_stability_b = e_b_mut - wt_binding.chain_b_energy

        # --- Step 6: Get mutant PDB ---
        mutant_pdb = _topology_to_pdb(best_top_complex, best_pos_complex)

        print(f"ΔΔG_binding: {ddg_binding:+.2f} kcal/mol")
        print(f"ΔΔG_stability (chain A): {ddg_stability_a:+.2f} kcal/mol")
        print(f"ΔΔG_stability (chain B): {ddg_stability_b:+.2f} kcal/mol")

        return InterfaceMutationResult(
            wt_binding_energy=wt_binding.binding_energy,
            mutant_binding_energy=round(e_binding_mut, 4),
            ddg_binding=round(ddg_binding, 4),
            wt_complex_energy=wt_binding.complex_energy,
            mutant_complex_energy=round(best_e_complex, 4),
            wt_chain_a_energy=wt_binding.chain_a_energy,
            wt_chain_b_energy=wt_binding.chain_b_energy,
            mutant_chain_a_energy=round(e_a_mut, 4),
            mutant_chain_b_energy=round(e_b_mut, 4),
            ddg_stability_a=round(ddg_stability_a, 4),
            ddg_stability_b=round(ddg_stability_b, 4),
            interface_residues=wt_binding.interface_residues,
            mutations_by_chain=mutations_by_chain,
            mutant_pdb=mutant_pdb,
        )

    # ------------------------------------------------------------------
    # Alanine scan (AlaScan equivalent)
    # ------------------------------------------------------------------

    def alanine_scan(
        self,
        source,
        chain: str,
        positions: Optional[List[int]] = None,
        max_iterations: int = 2000,
        constrain_backbone: bool = True,
    ) -> pl.DataFrame:
        """Perform alanine scanning on a chain.

        Performs systematic alanine scanning mutagenesis.  Each non-Ala/Gly position is mutated
        to alanine and the ddG is reported.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            chain: Chain ID to scan (e.g. 'A').
            positions: Specific residue numbers to scan.  If None, scans all
                       non-Ala/Gly standard residues.
            max_iterations: Minimisation steps per mutant.
            constrain_backbone: Freeze backbone atoms during minimisation.

        Returns:
            Polars DataFrame with columns:
            [chain, position, wt_residue, ddg_kcal_mol].
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)

        scan_positions = self._get_scannable_positions(
            fixer.topology, chain, positions, skip_residues={"ALA", "GLY"}
        )

        if not scan_positions:
            print(f"No scannable positions found on chain {chain}.")
            return pl.DataFrame(schema={
                "chain": pl.Utf8, "position": pl.Int64,
                "wt_residue": pl.Utf8, "ddg_kcal_mol": pl.Float64,
            })

        print(f"Alanine scan: {len(scan_positions)} positions on chain {chain}")

        rows = []
        for pos_num, wt_res in scan_positions:
            mut = Mutation(chain=chain, position=pos_num,
                          wt_residue=wt_res, mut_residue="ALA")
            try:
                result = self.mutate(
                    pdb_text, [mut],
                    max_iterations=max_iterations,
                    constrain_backbone=constrain_backbone,
                )
                ddg_val = list(result.ddg.values())[0]
            except Exception as e:
                print(f"  {mut.label}: FAILED ({e})")
                ddg_val = float("nan")

            rows.append({
                "chain": chain,
                "position": pos_num,
                "wt_residue": wt_res,
                "ddg_kcal_mol": round(ddg_val, 4),
            })
            print(f"  {mut.label}: ddG = {ddg_val:+.2f} kcal/mol")

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Position scan / PSSM
    # ------------------------------------------------------------------

    def position_scan(
        self,
        source,
        chain: str,
        positions: List[int],
        max_iterations: int = 2000,
        constrain_backbone: bool = True,
    ) -> pl.DataFrame:
        """Scan all 20 amino acids at specified positions.

        Generates position-specific scoring matrix by scanning all amino acids.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            chain: Chain ID.
            positions: List of residue numbers to scan.
            max_iterations: Minimisation steps per mutant.
            constrain_backbone: Freeze backbone atoms during minimisation.

        Returns:
            Polars DataFrame with columns:
            [chain, position, wt_residue, mut_residue, ddg_kcal_mol].
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)

        resmap = {}
        for chain_obj in fixer.topology.chains():
            if chain_obj.id == chain:
                for res in chain_obj.residues():
                    resmap[int(res.id)] = res.name

        rows = []
        total = len(positions) * 20
        done = 0
        for pos_num in positions:
            wt_res = resmap.get(pos_num)
            if wt_res is None or wt_res not in STANDARD_RESIDUES:
                continue

            for mut_res in ALL_AMINO_ACIDS:
                done += 1
                if mut_res == wt_res:
                    rows.append({
                        "chain": chain, "position": pos_num,
                        "wt_residue": wt_res, "mut_residue": mut_res,
                        "ddg_kcal_mol": 0.0,
                    })
                    continue

                mut = Mutation(chain=chain, position=pos_num,
                              wt_residue=wt_res, mut_residue=mut_res)
                try:
                    result = self.mutate(
                        pdb_text, [mut],
                        max_iterations=max_iterations,
                        constrain_backbone=constrain_backbone,
                    )
                    ddg_val = list(result.ddg.values())[0]
                except Exception as e:
                    ddg_val = float("nan")

                rows.append({
                    "chain": chain, "position": pos_num,
                    "wt_residue": wt_res, "mut_residue": mut_res,
                    "ddg_kcal_mol": round(ddg_val, 4),
                })

            print(f"  Position {pos_num} ({wt_res}) complete [{done}/{total}]")

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Per-residue energy (SequenceDetail equivalent)
    # ------------------------------------------------------------------

    def per_residue_energy(
        self,
        source,
        max_iterations: int = 2000,
    ) -> pl.DataFrame:
        """Approximate per-residue energy contribution.

        Computes per-residue energy decomposition.

        Uses an alanine-subtraction approach: for each residue, the energy
        difference between the full structure and the Ala-mutant estimates
        that residue's energetic contribution (positive = stabilising,
        negative = destabilising).

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            max_iterations: Minimisation steps.

        Returns:
            Polars DataFrame with columns:
            [chain, residue_number, residue_name, energy_contribution_kcal_mol].
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)

        all_positions = []
        for chain_obj in fixer.topology.chains():
            for res in chain_obj.residues():
                if res.name in STANDARD_RESIDUES and res.name not in ("ALA", "GLY"):
                    all_positions.append((chain_obj.id, int(res.id), res.name))

        if not all_positions:
            return pl.DataFrame(schema={
                "chain": pl.Utf8, "residue_number": pl.Int64,
                "residue_name": pl.Utf8, "energy_contribution_kcal_mol": pl.Float64,
            })

        print(f"Per-residue energy: {len(all_positions)} residues via Ala-subtraction")

        # Baseline WT energy
        stab = self.calculate_stability(pdb_text, max_iterations=max_iterations)
        e_wt = stab.total_energy

        rows = []
        for chain_id, pos_num, res_name in all_positions:
            mut = Mutation(chain=chain_id, position=pos_num,
                          wt_residue=res_name, mut_residue="ALA")
            try:
                result = self.mutate(
                    pdb_text, [mut],
                    max_iterations=max_iterations,
                    constrain_backbone=True,
                )
                ddg_val = list(result.ddg.values())[0]
                contribution = -ddg_val
            except Exception:
                contribution = float("nan")

            rows.append({
                "chain": chain_id,
                "residue_number": pos_num,
                "residue_name": res_name,
                "energy_contribution_kcal_mol": round(contribution, 4),
            })

        # Ala and Gly get 0.0 (self-reference)
        for chain_obj in fixer.topology.chains():
            for res in chain_obj.residues():
                if res.name in ("ALA", "GLY"):
                    rows.append({
                        "chain": chain_obj.id,
                        "residue_number": int(res.id),
                        "residue_name": res.name,
                        "energy_contribution_kcal_mol": 0.0,
                    })

        df = pl.DataFrame(rows).sort(["chain", "residue_number"])
        return df

    # ------------------------------------------------------------------
    # CSV-based batch mutations
    # ------------------------------------------------------------------

    @staticmethod
    def load_mutations(csv_path: str) -> pl.DataFrame:
        """Load a mutation list from a CSV file.

        The CSV must contain a ``mutation`` column with strings like ``G13L``.
        Optional columns:

        - ``chain`` — chain identifier (defaults to ``'A'`` if absent).
        - Any other columns (e.g. ``score``, ``source``, ``notes``) are
          preserved as metadata and carried through to the results.

        Args:
            csv_path: Path to a CSV file.

        Returns:
            Polars DataFrame with at least ``[mutation, chain]`` plus any
            extra columns from the CSV.
        """
        df = pl.read_csv(csv_path)

        if "mutation" not in df.columns:
            raise ValueError(
                f"CSV must contain a 'mutation' column. "
                f"Found columns: {df.columns}"
            )

        if "chain" not in df.columns:
            df = df.with_columns(pl.lit("A").alias("chain"))

        return df

    def mutate_batch(
        self,
        source,
        mutations_df: pl.DataFrame,
        max_iterations: int = 2000,
        n_runs: int = 3,
        constrain_backbone: bool = True,
        _repair_cache: Optional[_RepairCache] = None,
    ) -> pl.DataFrame:
        """Run every mutation in a DataFrame and return results.

        Each row is treated as an independent single-point mutation.
        Any extra columns in the input DataFrame are preserved in the output.

        The wild-type structure is prepared *once* and reused for every
        mutation, giving deterministic WT energies and eliminating
        hydrogen-placement noise.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            mutations_df: DataFrame with ``mutation`` and ``chain`` columns
                          (as returned by :meth:`load_mutations`).
            max_iterations: Minimisation steps per mutation (default 2000).
            n_runs: Independent minimisation runs per mutation (default 3).
            constrain_backbone: Restrain backbone during minimisation.
            _repair_cache: Optional pre-processed WT from :meth:`prepare`.
                           If not provided, one is created automatically.

        Returns:
            Polars DataFrame with the input columns plus
            ``[wt_energy, mutant_energy, ddg_kcal_mol]``.
        """
        if "mutation" not in mutations_df.columns:
            raise ValueError("DataFrame must contain a 'mutation' column.")
        if "chain" not in mutations_df.columns:
            mutations_df = mutations_df.with_columns(pl.lit("A").alias("chain"))

        if _repair_cache is None:
            print("Preparing wild-type structure (repair-once)...")
            _repair_cache = self._prepare_structure(
                source, max_iterations=max_iterations)

        result_rows = []
        total = mutations_df.height
        for i, row in enumerate(mutations_df.iter_rows(named=True)):
            mut_str = row["mutation"]
            chain_id = row["chain"]

            try:
                mut = Mutation.from_str(mut_str, chain=chain_id)
                result = self.mutate(
                    source, [mut],
                    chain=chain_id,
                    n_runs=n_runs,
                    max_iterations=max_iterations,
                    constrain_backbone=constrain_backbone,
                    _repair_cache=_repair_cache,
                )
                ddg_val = list(result.ddg.values())[0]
                wt_e = result.wt_energy
                mut_e = list(result.mutant_energies.values())[0]
            except Exception as e:
                print(f"  [{i+1}/{total}] {mut_str} chain {chain_id}: FAILED ({e})")
                ddg_val = float("nan")
                wt_e = float("nan")
                mut_e = float("nan")

            out = {**row, "wt_energy": wt_e, "mutant_energy": mut_e, "ddg_kcal_mol": ddg_val}
            result_rows.append(out)
            print(f"  [{i+1}/{total}] {mut_str} chain {chain_id}: ddG = {ddg_val:+.2f} kcal/mol")

        return pl.DataFrame(result_rows)

    # ------------------------------------------------------------------
    # Disulfide Bond Analysis
    # ------------------------------------------------------------------

    def detect_disulfides(
        self,
        source,
        distance_cutoff: float = 2.5,
    ) -> pl.DataFrame:
        """Detect disulfide bonds in a structure.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            distance_cutoff: Maximum S-S distance for disulfide bond (Å, default 2.5).

        Returns:
            DataFrame with columns:
            [chain1, residue1, resname1, chain2, residue2, resname2, distance].
        """
        from pdbfixer import PDBFixer

        pdb_text = _load_pdb(source)
        fixer = _pdb_string_to_fixer(pdb_text)

        # Get topology and positions
        topology = fixer.topology
        positions = fixer.positions

        return _detect_disulfide_bonds(topology, positions, distance_cutoff)

    def analyze_mutation_disulfide_impact(
        self,
        source,
        mutations: List[Union[Mutation, str]],
        chain: str = "A",
        distance_cutoff: float = 2.5,
    ) -> Dict[str, any]:
        """Analyze how mutations affect disulfide bonds.

        Args:
            source: PDB file path, PDB string, or Polars DataFrame.
            mutations: List of Mutation objects or strings (e.g. ``'C42A'``).
            chain: Default chain ID (default ``'A'``).
            distance_cutoff: Maximum S-S distance for disulfide bond (Å).

        Returns:
            Dict with:
            - wt_disulfides: DataFrame of WT disulfide bonds
            - mutant_disulfides: DataFrame of mutant disulfide bonds
            - broken_bonds: List of broken disulfide bonds
            - new_bonds: List of new disulfide bonds formed
            - affected_cysteines: List of mutated cysteine positions
        """
        from pdbfixer import PDBFixer

        # Parse mutations
        parsed = []
        for m in mutations:
            if isinstance(m, str):
                parsed.append(Mutation.from_str(m, chain=chain))
            else:
                parsed.append(m)

        # Detect WT disulfides
        wt_disulfides = self.detect_disulfides(source, distance_cutoff)

        # Check if any cysteines are being mutated
        affected_cysteines = []
        for m in parsed:
            if m.wt_residue == "CYS":
                affected_cysteines.append((m.chain, m.position))

        # Build mutant structure (simplified - just apply mutations)
        pdb_text = _load_pdb(source)
        fixer_mut = _pdb_string_to_fixer(pdb_text)
        fixer_mut.findMissingResidues()

        pdbfixer_mutations = []
        for m in parsed:
            pdbfixer_mutations.append(f"{m.wt_residue}-{m.position}-{m.mut_residue}")

        chains_used = {m.chain for m in parsed}
        for chain_id in chains_used:
            fixer_mut.applyMutations(pdbfixer_mutations, chain_id)

        fixer_mut.findMissingAtoms()
        fixer_mut.addMissingAtoms()

        # Detect mutant disulfides
        mutant_disulfides = _detect_disulfide_bonds(
            fixer_mut.topology, fixer_mut.positions, distance_cutoff
        )

        # Identify broken and new bonds
        wt_bonds = set()
        if wt_disulfides.height > 0:
            for row in wt_disulfides.iter_rows(named=True):
                bond = tuple(sorted([
                    (row["chain1"], row["residue1"]),
                    (row["chain2"], row["residue2"])
                ]))
                wt_bonds.add(bond)

        mutant_bonds = set()
        if mutant_disulfides.height > 0:
            for row in mutant_disulfides.iter_rows(named=True):
                bond = tuple(sorted([
                    (row["chain1"], row["residue1"]),
                    (row["chain2"], row["residue2"])
                ]))
                mutant_bonds.add(bond)

        broken_bonds = list(wt_bonds - mutant_bonds)
        new_bonds = list(mutant_bonds - wt_bonds)

        return {
            "wt_disulfides": wt_disulfides,
            "mutant_disulfides": mutant_disulfides,
            "broken_bonds": broken_bonds,
            "new_bonds": new_bonds,
            "affected_cysteines": affected_cysteines,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_scannable_positions(
        self, topology, chain_id: str,
        positions: Optional[List[int]],
        skip_residues: Optional[set] = None,
    ) -> List[Tuple[int, str]]:
        """Return (residue_number, residue_name) for positions eligible for scanning."""
        skip = skip_residues or set()
        result = []
        for chain_obj in topology.chains():
            if chain_obj.id != chain_id:
                continue
            for res in chain_obj.residues():
                pos_num = int(res.id)
                if res.name not in STANDARD_RESIDUES:
                    continue
                if res.name in skip:
                    continue
                if positions is not None and pos_num not in positions:
                    continue
                result.append((pos_num, res.name))
        return result
