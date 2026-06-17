# Changelog

All notable changes to Sicifus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-06-17

### Changed
- **Empirical scorer is now the default mutation backend.** `Sicifus.mutate_structure`,
  `alanine_scan`, `position_scan`, and `calculate_stability` default to
  `method="empirical"` (fast, no MD). The OpenMM path is retained as an opt-in
  physics reference (`method="openmm"`) but is no longer the default. Note:
  structure *building* (add atoms/hydrogens, apply mutation) still uses PDBFixer,
  so the `[energy]` extra remains required for any mutation analysis — the change
  is that no MD minimisation is run by default.

### Added
- **Locality-restricted empirical ΔΔG**: `EmpiricalScorer.score_mutation` now
  sums energy terms only over a residue shell around the mutation site
  (`radius`, default 8 Å). Whole-protein baselines (~10³ kcal/mol) made tiny
  rebuild jitter swamp a ~1 kcal/mol signal; restricting to the shell removes
  that non-cancelling noise. `radius=None` restores legacy whole-structure
  scoring.
- **Rotamer repacking** (`EmpiricalScorer.repack`, `sicifus.data.rotamers`):
  the mutated side chain is rebuilt over a coarse staggered χ library plus its
  native conformation; the lowest steric-energy (clash + vdW) rotamer is kept.
  Native is always a candidate, so repacking never worsens a well-placed
  residue — it only relieves the arbitrary, occasionally clashing rotamer
  PDBFixer builds. Removes the run-to-run outliers (e.g. Barnase H18K no longer
  swings to +6 kcal/mol).
- **Build averaging** for the empirical path: `Sicifus.mutate(method="empirical",
  n_builds=N)` averages ΔΔG over N independent builds and reports `delta_std`.
- **Binding ΔΔG** (`EmpiricalScorer.score_binding_mutation`): interface
  mutations via the cycle `ΔΔG_bind = ΔΔG_fold(complex) −
  ΔΔG_fold(isolated mutated partner)`, reusing the locality + repack machinery.
- **Weight calibration** (`examples/calibrate_empirical.py`): non-negative
  least-squares fit of term weights to experimental ΔΔG with leave-one-out
  cross-validation. Reads simple CSVs (ProTherm-style stability) or SKEMPI v2
  exports (binding; ΔΔG from Kd, parsed from the `Mutation(s)_PDB` column);
  auto-downloads/cleans PDBs; subsampling flags. Calibrated weights load
  automatically from `data/empirical_weights.json` (stability) with a separate
  `empirical_weights_binding.json` for the binding regime.
- **Empirical scan/stability parity**: empirical `alanine_scan` / `position_scan`
  (loops over the empirical scorer) and an empirical `calculate_stability`, plus
  `Sicifus.binding_ddg` — the no-MD counterpart to `mutate_interface`.

### Fixed
- **Mutation/structure numbering guard** (`MutationEngine.build_pdb_pair`):
  validates each mutation's wild-type residue against the structure and raises a
  clear error (expected vs found, with `show_residues` hint) instead of
  PDBFixer's cryptic failure. Catches sequence-vs-structure numbering mismatches
  (e.g. SKEMPI's renumbered `Mutation(s)_cleaned` column).
- **OpenMM `mutate` energy decomposition now ties out to ddG**: `energy_terms`
  gains an `unfolded_reference` row and the `total` row equals the reported ddG
  (folded force terms + unfolded reference = total = ddG), so the per-term
  decomposition is consistent with the thermodynamic-cycle ddG.

## [0.5.0] - 2026-06-15

### Fixed
- **Critical ΔΔG accuracy fix (thermodynamic cycle)**: the OpenMM mutation path
  previously computed `ddG = E_mutant − E_wild-type` by subtracting the absolute
  potential energies of two chemically different molecules (different atom/term
  inventories), which is not a meaningful quantity. This produced large errors
  (e.g. Barnase **H18K**: predicted ≈ −36 kcal/mol vs experimental +1.19).
  - `MutationEngine.mutate` now uses a thermodynamic cycle with an unfolded
    reference state:
    `ddG = (E_folded^mut − E_folded^wt) − Σ[E_unfolded(mut_res) − E_unfolded(wt_res)]`
  - Unfolded reference = capped, isolated residue (ACE-X-NME), built once per
    residue type and cached (`_unfolded_cache`).
  - The same correction is applied to per-chain stability ΔΔG in
    `mutate_interface` (binding ΔΔG is self-referencing and unchanged).

### Added
- **Empirical (FoldX-style) ΔΔG scorer** (`sicifus.empirical.EmpiricalScorer`):
  fast, no-MD scorer that evaluates a built structure as a weighted sum of
  folding-referenced terms (van der Waals + clash, polar/apolar solvation via
  Shrake–Rupley SASA, hydrogen bonding, electrostatics, backbone/side-chain
  entropy). Because the terms are folding-referenced, `G_mut − G_wt` is
  well-posed, sidestepping the absolute-MM-energy problem.
  - `score(source) -> EmpiricalEnergy` (per-term breakdown).
  - `score_mutation(wt, mutant, mutations) -> MutationResult` — same shape as the
    OpenMM path, compatible with existing `visualization` helpers.
  - Side-chain **rotamer repacking is a documented seam** (`repack()`, identity
    in v1) for a later upgrade.
  - Parameters in `sicifus.data.empirical_params` are published approximations
    and are **not yet calibrated** against an experimental set (follow-up).
- `Sicifus.mutate_structure(..., method="openmm" | "empirical")` to select the
  backend; both return `MutationResult`.
- `MutationEngine.build_pdb_pair(source, mutations)` — build WT + mutant PDBs
  (atoms + hydrogens, no minimisation); used by the empirical path.
- New fast test suite `tests/test_empirical.py` (18 tests, no OpenMM).

### Changed
- **Mutation performance**: `n_runs` default changed `3 → 1` in `mutate`,
  `mutate_batch`, and `mutate_interface`. `LocalEnergyMinimizer` is deterministic,
  so repeated runs from the same start coordinates were identical — they
  multiplied cost (≈3×) and produced meaningless "statistics".
- `mutate()` gained a `decompose` flag; scans (`alanine_scan`, `position_scan`,
  `per_residue_energy`) and `mutate_batch` skip per-term decomposition (only ΔΔG
  needed), saving two OpenMM `Context` builds per mutation.
- WT energy decomposition is computed once and cached on `_RepairCache`
  (`wt_terms`) instead of recomputed for every mutation in a batch/scan.
- Net effect: ≈3× faster single `mutate`; ≈4–6× faster scans.

## [0.4.6] - 2026-04-17

### Fixed
- **Critical bug fix**: Corrected PDBFixer residue numbering preservation
  - Initialize `fixer.missingResidues = {}` to prevent adding N/C-terminal residues
  - Use `PDBFile.writeFile(..., keepIds=True)` to preserve residue IDs when saving
  - Without `keepIds=True`, OpenMM renumbers residues sequentially (1, 2, 3, ...) regardless of original PDB numbering
  - Ensures user mutation positions exactly match PDB file residue numbers
  - Fixes issue where Barnase (starts at residue 3) would have mutations applied to wrong positions
  - Example: position 18 HIS was becoming position 16, causing "residue 18 is LEU" error

## [0.4.5] - 2026-04-17

### Added
- **Atom-based alignment** with PyMOL-like selection syntax (`AtomAligner`, `SelectionParser`)
  - Flexible Kabsch alignment on arbitrary atom selections
  - Support for ligand overlay, binding site comparison, transition state analysis
  - Selection syntax: `"chain A and resi 50-60 and name CA,CB"`
  - Multiple structure alignment
  - Pairwise RMSD calculation (aligned and positional)
  - New module: `sicifus.atom_align`
  - 34 comprehensive tests
  - Demo: `examples/atom_alignment_demo.py`
  - Documentation: `ATOM_ALIGNMENT_GUIDE.md`

- **Experimental validation** against Serrano et al. (1993) dataset
  - 17 Barnase mutations with experimental ΔΔG values
  - R² correlation calculation in tests and demos
  - Dataset: `examples/serrano1993_experimental_data.csv`
  - Reference paper included: `serrano1993.pdf`
  - Updated demos to use validated mutations (H18K, I55V, K62R, etc.)

- **Variant library generation** for high-throughput screening
  - Demo: `examples/barnase_variant_library_demo.py`
  - Generate all single-point mutations (1000+ variants)
  - Batch ΔΔG prediction and ranking
  - Hotspot identification
  - Visualizations: ΔΔG distribution, position preferences, mutation tolerances
  - Export top stabilizing/destabilizing variants

- **Documentation**
  - `ATOM_ALIGNMENT_GUIDE.md` - Comprehensive alignment guide (800+ lines)
  - `ATOM_ALIGNMENT_SUMMARY.md` - Implementation summary
  - `MUTATION_APPROACHES_COMPARISON.md` - Comparison with threading scripts
  - `VARIANT_GENERATION_SUMMARY.md` - Variant library workflows
  - `SERRANO1993_INTEGRATION.md` - Experimental data integration details

### Fixed
- **Critical bug**: `AttributeError: 'PDBFixer' object has no attribute 'missingResidues'`
  - PDBFixer's `findMissingAtoms()` requires `findMissingResidues()` to be called first
  - Fixed by calling `findMissingResidues()` then clearing the list to preserve PDB numbering
  - Affects all methods: `mutate()`, `repair()`, `calculate_stability()`, etc.
  - Ensures user mutation positions match exact PDB residue numbers

### Changed
- Updated mutation demos to use Serrano 1993 experimental values
  - `mutation_analysis_demo.py`: Changed from Y24A to H18K
  - `mutation_analysis_demo.ipynb`: Updated with new experimental data
  - `test_setup.py`: Changed validation mutation to H18K
  - All demos now cite Serrano et al. (1993) Table 3

- Enhanced `examples/README_DEMOS.md`
  - Added experimental validation data section
  - Updated expected performance benchmarks
  - Mutation classification guidance

### Performance
- Atom alignment: ~1 ms for 1000 atoms (O(N) Kabsch algorithm)
- Selection parsing: <1 ms for typical selections
- Variant library screening: ~30 sec/mutation (fast mode)

### API
New exports in `sicifus`:
```python
from sicifus import AtomAligner, SelectionParser, AlignmentResult, write_pdb
```

No breaking changes - fully backward compatible.

## [0.4.2] - 2026-04-XX

### Previous Release
- (Add previous version notes here if available)

---

## Version History Summary

- **0.4.6**: PDBFixer residue numbering fix (prevents terminal residue addition)
- **0.4.5**: Atom alignment, experimental validation, variant libraries, critical bug fix
- **0.4.2**: (Previous release)
