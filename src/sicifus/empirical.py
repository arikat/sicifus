"""Empirical (FoldX-style) ΔΔG scorer.

This is an *empirical* free-energy scorer: it evaluates a built structure as a
weighted sum of physically-motivated terms (van der Waals, clash, solvation,
hydrogen bonding, electrostatics, and conformational entropy) **without any
molecular-dynamics minimisation**.  Because the terms are folding-referenced
(à la FoldX), the difference ``G_mutant − G_wild-type`` is well-posed — unlike
raw MM potential energies, whose absolute baselines do not cancel when two
chemically different residues are compared (the bug that made the OpenMM path
predict −36 kcal/mol for a +1 kcal/mol mutation).

Accuracy caveat: the parameters in :mod:`sicifus.data.empirical_params` are
published approximations that have **not** been calibrated against an
experimental ΔΔG set, and v1 does **not** repack side chains — it scores the
mutant as PDBFixer builds it.  Treat outputs as a physically-grounded baseline.
A rotamer-repacking step can be slotted into :meth:`EmpiricalScorer.repack`
later without changing the public API.

References: see :mod:`sicifus.data.empirical_params`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import polars as pl

from .data import empirical_params as P
from .data import rotamers as R
from .mutate import Mutation, MutationResult, _load_pdb

COULOMB_K = 332.0  # kcal·Å/(mol·e²)


@dataclass
class EmpiricalEnergy:
    """Per-term empirical free energy of a single structure (kcal/mol)."""
    vdw: float
    clash: float
    solvH: float
    solvP: float
    hbond: float
    elec: float
    mc_entropy: float
    sc_entropy: float
    total: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _AtomArrays:
    """Flat NumPy view of a structure for fast term evaluation."""
    coords: np.ndarray      # (N, 3) float
    radii: np.ndarray       # (N,) vdW radius
    wells: np.ndarray       # (N,) vdW well depth
    charges: np.ndarray     # (N,) partial charge
    asp: np.ndarray         # (N,) atomic solvation parameter
    res_index: np.ndarray   # (N,) integer residue id (for excluding intra-residue pairs)
    hbond_role: list        # (N,) 'D'/'A'/'B'/None
    res_names: list         # (N,) residue name per atom
    atom_names: list        # (N,) atom name per atom (for rotamer repacking)
    # One entry per residue, in first-seen order:
    residue_names: list
    residue_keys: list      # (R,) (chain_id, seq_id) per residue — for site lookup


class EmpiricalScorer:
    """Empirical FoldX-style ΔΔG scorer (no MD)."""

    def __init__(
        self,
        work_dir: str = "./empirical_work",
        weights: Optional[dict] = None,
        sasa_probe: float = 1.4,
        sasa_points: int = 92,
    ):
        """
        Args:
            work_dir: Directory for any temporary files (created if absent).
            weights: Optional override of term weights (see
                     ``empirical_params.WEIGHTS``).
            sasa_probe: Solvent-probe radius (Å) for SASA (default 1.4).
            sasa_points: Sphere sampling points per atom for Shrake–Rupley.
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.weights = dict(P.WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.sasa_probe = sasa_probe
        self._sphere = _fibonacci_sphere(sasa_points)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, source) -> _AtomArrays:
        """Parse a PDB path/string/DataFrame into flat per-atom arrays."""
        import gemmi

        pdb_text = _load_pdb(source)
        structure = gemmi.read_pdb_string(pdb_text)
        model = structure[0]

        coords, radii, wells, charges, asp = [], [], [], [], []
        res_index, hbond_role, res_names = [], [], []
        atom_names = []
        residue_names, residue_keys = [], []
        ridx = -1

        for chain in model:
            for residue in chain:
                ridx += 1
                resn = residue.name
                residue_names.append(resn)
                residue_keys.append((chain.name, residue.seqid.num))
                for atom in residue:
                    aname = atom.name
                    elem = P.element_of(aname, atom.element.name if atom.element else None)
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    radii.append(P.VDW_RADII.get(elem, P.DEFAULT_VDW_RADIUS))
                    wells.append(P.VDW_WELL.get(elem, P.DEFAULT_VDW_WELL))
                    charges.append(P.atom_charge(resn, aname))
                    asp.append(P.atom_asp(resn, aname, elem))
                    res_index.append(ridx)
                    hbond_role.append(P.atom_hbond_role(resn, aname))
                    res_names.append(resn)
                    atom_names.append(aname)

        if not coords:
            raise ValueError("No atoms parsed from structure.")

        return _AtomArrays(
            coords=np.asarray(coords, dtype=float),
            radii=np.asarray(radii, dtype=float),
            wells=np.asarray(wells, dtype=float),
            charges=np.asarray(charges, dtype=float),
            asp=np.asarray(asp, dtype=float),
            res_index=np.asarray(res_index, dtype=int),
            hbond_role=hbond_role,
            res_names=res_names,
            atom_names=atom_names,
            residue_names=residue_names,
            residue_keys=residue_keys,
        )

    # ------------------------------------------------------------------
    # Side-chain repacking seam (identity in v1)
    # ------------------------------------------------------------------

    def repack(
        self, atoms: _AtomArrays, sites: Optional[set] = None
    ) -> _AtomArrays:
        """Optimise side-chain conformations of the ``sites`` residues.

        For each residue index in ``sites`` the side chain is rebuilt at every
        rotamer in the coarse staggered library (:mod:`sicifus.data.rotamers`)
        plus its *native* (as-built) conformation; the conformation with the
        lowest packing energy against the rest of the structure (van der Waals,
        clash, electrostatics, hydrogen bonding) is kept.  Because the native
        conformation is always a candidate, repacking never worsens a residue
        that was already well placed — it only relieves the arbitrary,
        occasionally clashing rotamer PDBFixer assigns to a rebuilt mutant side
        chain.

        ``sites`` is a set of residue indices (as in :attr:`_AtomArrays.res_index`).
        ``None`` (the default, used by :meth:`score`) is a no-op so whole-
        structure scoring is unchanged.  Only side-chain ``coords`` are
        modified; atom ordering and every other array are preserved.
        """
        if not sites:
            return atoms

        from scipy.spatial import cKDTree

        coords = atoms.coords.copy()
        # Precompute residue → atom-index lists once.
        res_atoms: dict = {}
        for k, r in enumerate(atoms.res_index):
            res_atoms.setdefault(int(r), []).append(k)

        for r in sorted(sites):
            resn = atoms.residue_names[r]
            library = R.rotamers_for(resn)
            if not library:
                continue  # ALA/GLY/PRO or unknown — nothing to repack
            idxs = res_atoms[r]
            name_to_idx = {atoms.atom_names[k]: k for k in idxs}
            chi_defs = R.CHI_DEFS[resn]
            # Skip if the χ-defining atoms aren't all present (truncated residue).
            if any(any(n not in name_to_idx for n in quad) for quad in chi_defs):
                continue

            bonds = _intra_bonds(coords, idxs)
            moving = [
                _moving_atoms(bonds, name_to_idx[q[2]], name_to_idx[q[1]])
                for q in chi_defs
            ]

            # Side-chain atoms (everything that isn't a fixed backbone atom).
            sc = [k for k in idxs if atoms.atom_names[k] not in R.BACKBONE_ATOMS]
            # Environment: atoms of all other residues within an 8 Å pocket.
            env = [k for k in range(len(coords)) if atoms.res_index[k] != r]
            if not env or not sc:
                continue
            env_tree = cKDTree(coords[env])
            env = np.asarray(env)

            best_coords = {k: coords[k].copy() for k in sc}
            best_e = self._packing_energy(
                atoms, coords, sc, env, env_tree)

            for chis in library:
                trial = coords.copy()
                ok = True
                for c_idx, target in enumerate(chis):
                    a, b, c, d = (name_to_idx[n] for n in chi_defs[c_idx])
                    if not _rotate_to_chi(trial, a, b, c, d, target, moving[c_idx]):
                        ok = False
                        break
                if not ok:
                    continue
                e = self._packing_energy(atoms, trial, sc, env, env_tree)
                if e < best_e:
                    best_e = e
                    best_coords = {k: trial[k].copy() for k in sc}

            for k, xyz in best_coords.items():
                coords[k] = xyz

        return replace(atoms, coords=coords)

    def _packing_energy(self, a, coords, sc, env, env_tree) -> float:
        """Steric packing strain of a side chain vs its environment (kcal/mol).

        Sums only the **steric** terms — clash penalty plus van der Waals
        (attractive + the repulsive wall implicit in the clash term) — between
        the candidate side-chain atoms ``sc`` and nearby environment atoms.
        Electrostatics and hydrogen bonding are deliberately *excluded*: this
        objective relieves the arbitrary clashing rotamer PDBFixer builds, it
        does not hunt for favourable salt bridges/H-bonds (which the coarse,
        angle-free elec/hbond terms over-reward, over-stabilising the mutant and
        flipping the chosen rotamer run-to-run).  Those interactions are still
        evaluated in the final :meth:`_score_atoms`.
        """
        e = 0.0
        for i in sc:
            near = env_tree.query_ball_point(coords[i], 6.0)
            if not near:
                continue
            js = env[near]
            d = np.linalg.norm(coords[js] - coords[i], axis=1)
            keep = d > 1.2
            d = np.maximum(d[keep], 0.1)
            if not len(d):
                continue
            rsum = a.radii[i] + a.radii[np.asarray(js)[keep]]
            eps = np.sqrt(a.wells[i] * a.wells[np.asarray(js)[keep]])
            e += float(np.sum(-2.0 * eps * (rsum / d) ** 6 * (d >= rsum)))
            overlap = rsum - d - P.CLASH_TOLERANCE
            m = overlap > 0
            e += float(np.sum(P.CLASH_SCALE * overlap[m] ** 2))
        return e

    # ------------------------------------------------------------------
    # SASA (Shrake–Rupley)
    # ------------------------------------------------------------------

    def _sasa(self, coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Per-atom solvent-accessible surface area (Å²)."""
        from scipy.spatial import cKDTree

        r = radii + self.sasa_probe
        tree = cKDTree(coords)
        n = len(coords)
        sasa = np.zeros(n)
        sphere = self._sphere

        for i in range(n):
            ri = r[i]
            # Neighbours whose expanded spheres can occlude atom i's surface.
            idx = tree.query_ball_point(coords[i], ri + r.max())
            idx = [j for j in idx if j != i]
            pts = sphere * ri + coords[i]            # (P, 3) surface points
            if idx:
                nbr = coords[idx]                    # (M, 3)
                nbr_r2 = r[idx] ** 2                 # (M,)
                # (P, M) squared distances point→neighbour
                d2 = np.sum((pts[:, None, :] - nbr[None, :, :]) ** 2, axis=2)
                buried = np.any(d2 < nbr_r2[None, :], axis=1)
                accessible = np.count_nonzero(~buried)
            else:
                accessible = len(sphere)
            sasa[i] = 4.0 * math.pi * ri * ri * accessible / len(sphere)
        return sasa

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, source) -> EmpiricalEnergy:
        """Evaluate the empirical free energy of one structure."""
        atoms = self.repack(self._parse(source))
        return self._score_atoms(atoms)

    def _score_atoms(
        self, a: _AtomArrays, sel_res: Optional[set] = None
    ) -> EmpiricalEnergy:
        """Score a structure.

        If ``sel_res`` is given (a set of residue indices), only energy
        contributions touching those residues are summed: a non-bonded pair
        counts when *either* atom belongs to a selected residue, and the
        solvation/entropy sums run over selected residues only.  Pairs and
        residues entirely outside the selection are chemically identical
        between a wild-type and its mutant, so dropping them removes their
        non-cancelling numerical noise from ΔΔG.  SASA is still computed on the
        full structure so burial by excluded atoms is accounted for.
        """
        from scipy.spatial import cKDTree

        coords = a.coords
        tree = cKDTree(coords)
        # Non-bonded pair list within an interaction cutoff.
        pairs = tree.query_pairs(r=8.0, output_type="ndarray")

        sel_mask = None
        if sel_res is not None:
            sel_mask = np.array(
                [r in sel_res for r in range(len(a.residue_names))], dtype=bool
            )

        vdw = clash = hbond = elec = 0.0
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            # Exclude atom pairs within the same residue (bonded/geminal noise).
            inter = a.res_index[i] != a.res_index[j]
            i, j = i[inter], j[inter]
            if sel_mask is not None:
                # Keep a pair only if it touches the selected shell.
                touch = sel_mask[a.res_index[i]] | sel_mask[a.res_index[j]]
                i, j = i[touch], j[touch]

            d = np.linalg.norm(coords[i] - coords[j], axis=1)
            # Exclude covalently bonded cross-residue pairs (notably the
            # peptide C–N bond at ~1.33 Å); without connectivity we treat any
            # sub-covalent separation as bonded so it isn't scored as a clash.
            nonbonded = d > 1.9
            i, j, d = i[nonbonded], j[nonbonded], d[nonbonded]
            d = np.maximum(d, 0.1)
            rsum = a.radii[i] + a.radii[j]

            # --- van der Waals attraction (favourable well) ---
            eps = np.sqrt(a.wells[i] * a.wells[j])
            ratio6 = (rsum / d) ** 6
            vdw = float(np.sum(-2.0 * eps * ratio6 * (d >= rsum)))  # attractive side only

            # --- clash penalty (overlap beyond tolerance) ---
            overlap = rsum - d - P.CLASH_TOLERANCE
            clash_mask = overlap > 0
            clash = float(np.sum(P.CLASH_SCALE * overlap[clash_mask] ** 2))

            # --- electrostatics (distance-dependent dielectric ε = 4d) ---
            qi, qj = a.charges[i], a.charges[j]
            qmask = (qi != 0) & (qj != 0)
            if np.any(qmask):
                elec = float(np.sum(
                    COULOMB_K * qi[qmask] * qj[qmask] / (4.0 * d[qmask] ** 2)
                ))

            # --- hydrogen bonds (donor/acceptor, distance-gated) ---
            in_range = (d >= P.HBOND_DIST_MIN) & (d <= P.HBOND_DIST_MAX)
            for k in np.nonzero(in_range)[0]:
                ri, rj = a.hbond_role[i[k]], a.hbond_role[j[k]]
                if ri is None or rj is None:
                    continue
                donor_acceptor = (
                    (ri in ("D", "B") and rj in ("A", "B")) or
                    (rj in ("D", "B") and ri in ("A", "B"))
                )
                if donor_acceptor:
                    hbond += P.HBOND_ENERGY

        # --- solvation (Eisenberg: ΔG = Σ asp_i · SASA_i) ---
        # SASA on the full structure (correct burial), then restrict the sum to
        # selected residues' atoms when a shell is given.
        sasa = self._sasa(coords, a.radii)
        solv = a.asp * sasa
        if sel_mask is not None:
            atom_sel = sel_mask[a.res_index]
            solv = solv * atom_sel
        solvH = float(np.sum(solv[a.asp > 0]))
        solvP = float(np.sum(solv[a.asp < 0]))

        # --- conformational entropy (per residue identity) ---
        if sel_res is None:
            entropy_res = list(enumerate(a.residue_names))
        else:
            entropy_res = [(k, a.residue_names[k]) for k in sel_res]
        mc_entropy = float(sum(P.BACKBONE_ENTROPY.get(r, 0.0) for _, r in entropy_res))
        sc_entropy = float(sum(P.SIDECHAIN_ENTROPY.get(r, 0.0) for _, r in entropy_res))

        w = self.weights
        total = (
            w["vdw"] * vdw
            + w["clash"] * clash
            + w["solvH"] * solvH
            + w["solvP"] * solvP
            + w["hbond"] * hbond
            + w["elec"] * elec
            + w["mc_entropy"] * mc_entropy
            + w["sc_entropy"] * sc_entropy
        )

        return EmpiricalEnergy(
            vdw=round(w["vdw"] * vdw, 4),
            clash=round(w["clash"] * clash, 4),
            solvH=round(w["solvH"] * solvH, 4),
            solvP=round(w["solvP"] * solvP, 4),
            hbond=round(w["hbond"] * hbond, 4),
            elec=round(w["elec"] * elec, 4),
            mc_entropy=round(w["mc_entropy"] * mc_entropy, 4),
            sc_entropy=round(w["sc_entropy"] * sc_entropy, 4),
            total=round(total, 4),
        )

    # ------------------------------------------------------------------
    # Local shell selection
    # ------------------------------------------------------------------

    def _shell_residues(self, a: _AtomArrays, site_keys: set, radius: float):
        """Residue indices/keys within ``radius`` Å of the site residues.

        Returns ``(indices, keys)`` where a residue is included if any of its
        atoms lies within ``radius`` of any atom of a site residue.  The site
        residues themselves are always included.
        """
        from scipy.spatial import cKDTree

        site_atoms = np.array(
            [a.residue_keys[r] in site_keys for r in a.res_index], dtype=bool
        )
        if not site_atoms.any():
            raise ValueError(
                f"Mutation site(s) {sorted(site_keys)} not found in structure "
                f"(have e.g. {a.residue_keys[:5]} ...)."
            )
        tree = cKDTree(a.coords)
        near = set()
        for c in a.coords[site_atoms]:
            near.update(tree.query_ball_point(c, radius))
        indices = {int(a.res_index[k]) for k in near}
        indices.update(r for r, key in enumerate(a.residue_keys) if key in site_keys)
        keys = {a.residue_keys[r] for r in indices}
        return indices, keys

    # ------------------------------------------------------------------
    # Mutation ΔΔG
    # ------------------------------------------------------------------

    def score_mutation(
        self,
        wt_source,
        mutant_source,
        mutations: List[Union[Mutation, str]],
        chain: str = "A",
        radius: Optional[float] = 8.0,
    ) -> MutationResult:
        """ΔΔG between a wild-type and a pre-built mutant structure.

        Both structures must already carry the relevant atoms (e.g. built and
        protonated by :class:`~sicifus.mutate.MutationEngine`).  ΔΔG is simply
        ``G_mutant − G_wild-type`` — valid here because the empirical terms are
        folding-referenced.

        Args:
            wt_source: Wild-type PDB path/string/DataFrame.
            mutant_source: Mutant PDB path/string/DataFrame.
            mutations: Mutations applied (for labelling), ``Mutation`` objects
                       or strings like ``'H18K'``.
            chain: Default chain for parsing mutation strings.
            radius: Shell radius (Å) around the mutation site over which energy
                    terms are summed.  Far-from-site atoms are chemically
                    identical between WT and mutant, so excluding them removes
                    their non-cancelling numerical noise from ΔΔG (whole-protein
                    baselines are ~10³ kcal/mol, so even rebuild jitter swamps a
                    ~1 kcal/mol signal).  Pass ``None`` to score the whole
                    structure (legacy behaviour, noisier).

        Returns:
            A :class:`~sicifus.mutate.MutationResult` with ``ddg``, energies,
            and an ``energy_terms`` DataFrame ``[term, wt_energy, mutant_energy,
            delta]`` — the same shape the OpenMM path and the visualization
            helpers expect.
        """
        parsed = [
            m if isinstance(m, Mutation) else Mutation.from_str(m, chain=chain)
            for m in mutations
        ]
        label = "+".join(m.label for m in parsed)

        wt_parsed = self._parse(wt_source)
        mut_parsed = self._parse(mutant_source)
        site_keys = {(m.chain, m.position) for m in parsed}
        wt_sites = {r for r, k in enumerate(wt_parsed.residue_keys) if k in site_keys}
        mut_sites = {r for r, k in enumerate(mut_parsed.residue_keys) if k in site_keys}
        # Repack the mutated residue (and its native counterpart) so ΔΔG is not
        # hostage to the arbitrary rotamer PDBFixer built it in.
        wt_atoms = self.repack(wt_parsed, sites=wt_sites)
        mut_atoms = self.repack(mut_parsed, sites=mut_sites)

        wt_sel = mut_sel = None
        if radius is not None:
            _, wt_keys = self._shell_residues(wt_atoms, site_keys, radius)
            _, mut_keys = self._shell_residues(mut_atoms, site_keys, radius)
            # Same residue set (by chain/seq) on both sides for clean cancellation.
            shell_keys = wt_keys | mut_keys
            wt_sel = {r for r, k in enumerate(wt_atoms.residue_keys) if k in shell_keys}
            mut_sel = {r for r, k in enumerate(mut_atoms.residue_keys) if k in shell_keys}

        e_wt = self._score_atoms(wt_atoms, wt_sel)
        e_mut = self._score_atoms(mut_atoms, mut_sel)
        ddg = round(e_mut.total - e_wt.total, 4)

        wt_d, mut_d = e_wt.as_dict(), e_mut.as_dict()
        term_rows = [
            {
                "term": term,
                "wt_energy": wt_d[term],
                "mutant_energy": mut_d[term],
                "delta": round(mut_d[term] - wt_d[term], 4),
            }
            for term in ("vdw", "clash", "solvH", "solvP", "hbond",
                         "elec", "mc_entropy", "sc_entropy", "total")
        ]

        mutant_pdb = _load_pdb(mutant_source)
        return MutationResult(
            wt_energy=e_wt.total,
            mutant_energies={label: e_mut.total},
            ddg={label: ddg},
            mutant_pdbs={label: mutant_pdb},
            energy_terms=pl.DataFrame(term_rows),
        )

    # ------------------------------------------------------------------
    # Binding ΔΔG (interface mutations)
    # ------------------------------------------------------------------

    def score_binding_mutation(
        self,
        wt_complex,
        mutant_complex,
        mutations: List[Union[Mutation, str]],
        mutated_chains,
        chain: str = "A",
        radius: Optional[float] = 8.0,
    ) -> MutationResult:
        """ΔΔG of *binding* for a mutation in a protein–protein complex.

        Uses the identity

            ΔΔG_bind = ΔΔG_fold(complex) − ΔΔG_fold(isolated mutated partner)

        which is exact for a rigid two-body decomposition: the unmutated partner
        is identical on both sides and cancels, leaving the difference between
        how the mutation destabilises the bound state and the free state.  Both
        terms are ordinary folding ΔΔGs computed by :meth:`score_mutation`, so
        the locality + rotamer-repack machinery (and its noise cancellation)
        applies unchanged.  A positive result means the mutation weakens binding.

        Args:
            wt_complex: Wild-type complex PDB (all chains) path/string/DataFrame.
            mutant_complex: Pre-built mutant complex (all chains).
            mutations: Mutations applied (``Mutation`` objects or strings).
            mutated_chains: Chain id(s) of the partner that carries the mutation
                (str or iterable of str); the complementary chains are the other
                binding partner and cancel.
            chain: Default chain for parsing mutation strings.
            radius: Shell radius (Å); see :meth:`score_mutation`.

        Returns:
            :class:`~sicifus.mutate.MutationResult` whose ``ddg`` is the binding
            ΔΔG and whose ``energy_terms`` holds the per-term
            ``complex − isolated`` contributions.
        """
        keep = {mutated_chains} if isinstance(mutated_chains, str) else set(mutated_chains)

        res_complex = self.score_mutation(
            wt_complex, mutant_complex, mutations, chain=chain, radius=radius)
        wt_iso = _filter_chains(_load_pdb(wt_complex), keep)
        mut_iso = _filter_chains(_load_pdb(mutant_complex), keep)
        res_iso = self.score_mutation(
            wt_iso, mut_iso, mutations, chain=chain, radius=radius)

        label = next(iter(res_complex.ddg))
        ddg_bind = round(res_complex.ddg[label] - res_iso.ddg[label], 4)

        ct = res_complex.energy_terms
        it = res_iso.energy_terms
        iso_delta = dict(zip(it["term"].to_list(), it["delta"].to_list()))
        term_rows = [
            {
                "term": row["term"],
                "complex_delta": row["delta"],
                "isolated_delta": iso_delta.get(row["term"], 0.0),
                "delta": round(row["delta"] - iso_delta.get(row["term"], 0.0), 4),
            }
            for row in ct.iter_rows(named=True)
        ]
        return MutationResult(
            wt_energy=res_complex.wt_energy,
            mutant_energies={label: res_complex.mutant_energies[label]},
            ddg={label: ddg_bind},
            mutant_pdbs={label: _load_pdb(mutant_complex)},
            energy_terms=pl.DataFrame(term_rows),
        )


def _filter_chains(pdb_text: str, keep: set) -> str:
    """Return a PDB string with only ``keep`` chains' atom records retained.

    Keeps coordinate records (ATOM/HETATM/TER/ANISOU) whose chain id (column 22)
    is in ``keep``; passes through structural records (MODEL/ENDMDL/END).  Used
    to isolate one binding partner from a complex for the binding-ΔΔG cycle.
    """
    out = []
    for line in pdb_text.splitlines():
        rec = line[:6]
        if rec.startswith(("ATOM", "HETATM", "TER", "ANISOU")):
            if len(line) > 21 and line[21] in keep:
                out.append(line)
        elif rec.startswith(("MODEL", "ENDMDL", "END")):
            out.append(line)
    return "\n".join(out) + "\n"


def _intra_bonds(coords: np.ndarray, idxs: list) -> dict:
    """Bond graph among one residue's atoms, keyed by global atom index.

    A bond is inferred from interatomic distance: ≤1.35 Å when a hydrogen is
    involved, ≤1.9 Å otherwise.  Good enough for BFS across χ axes within a
    single residue (no need for explicit connectivity records).
    """
    graph = {k: set() for k in idxs}
    for ai, ka in enumerate(idxs):
        for kb in idxs[ai + 1:]:
            # ≤1.9 Å covers every real intra-residue bond (incl. C–S) and never
            # links non-bonded atoms within one residue.
            if float(np.linalg.norm(coords[ka] - coords[kb])) <= 1.9:
                graph[ka].add(kb)
                graph[kb].add(ka)
    return graph


def _moving_atoms(bonds: dict, c: int, b: int) -> list:
    """Atoms that rotate about the ``b–c`` bond: reachable from ``c`` not via ``b``.

    Every atom reachable from ``c`` without traversing ``b`` (the far side of the
    χ axis) moves — atom ``c``'s substituents and all distal atoms, including
    their hydrogens — while the backbone (past ``b``) stays fixed.  ``c`` itself
    is on the axis and maps to itself under the rotation, so keeping it is a
    no-op.
    """
    seen = {b}
    stack = [c]
    out = []
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)
        stack.extend(bonds.get(cur, ()))
    return out


def _rotate_to_chi(coords, a, b, c, d, target_deg, moving) -> bool:
    """Rotate ``moving`` atoms so dihedral (a,b,c,d) equals ``target_deg``.

    Rotation is about the ``b–c`` bond axis through point ``c``.  Returns False
    if the axis is degenerate (zero-length bond), True otherwise.  ``coords`` is
    modified in place.
    """
    pa, pb, pc, pd = coords[a], coords[b], coords[c], coords[d]
    axis = pc - pb
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return False
    axis = axis / norm
    cur = _dihedral(pa, pb, pc, pd)
    delta = math.radians(target_deg) - cur
    rot = _rotation_matrix(axis, delta)
    pivot = pc.copy()
    for m in moving:
        coords[m] = pivot + rot @ (coords[m] - pivot)
    return True


def _dihedral(p0, p1, p2, p3) -> float:
    """Signed dihedral angle (radians) of four points."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1) + 1e-12
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.atan2(y, x)


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``theta`` radians about a unit ``axis``."""
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = axis
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` roughly-uniform unit-sphere points (golden-spiral)."""
    pts = np.empty((n, 3))
    offset = 2.0 / n
    increment = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(n):
        y = k * offset - 1.0 + offset / 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = k * increment
        pts[k] = (math.cos(phi) * r, y, math.sin(phi) * r)
    return pts
