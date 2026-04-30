# Changelog

All notable changes to Sicifus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
