# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sicifus** is a high-performance Python library for large-scale structural biology analysis. It ingests thousands of macromolecular structures (CIF/PDB files), compiles them into a partitioned Parquet database, and enables out-of-core processing for datasets larger than available RAM.

Key capabilities:
- **Ingestion**: Parse CIF/PDB files into structured, queryable Parquet databases
- **Alignment**: Sequence-independent structural alignment using 3Di structural alphabet
- **Phylogenetics**: RMSD-based phylogenetic trees and clustering
- **Ligand Analysis**: Binding pocket analysis, protein-ligand interactions
- **Mutation & Stability**: In silico mutagenesis, ddG prediction, powered by OpenMM
- **Metadata**: Join external metadata with structural data

## Development Commands

### Testing

Run all tests:
```bash
pytest
```

Run only fast tests (skip OpenMM minimization tests):
```bash
pytest -m "not slow"
```

Run specific test file:
```bash
pytest tests/test_mutate.py -v
```

Tests marked with `@pytest.mark.slow` require OpenMM/PDBFixer and run energy minimization.

### Building & Publishing

Build the package:
```bash
python -m build
```

Check distribution files:
```bash
twine check dist/*
```

See `PUBLISHING.md` for full publishing workflow to PyPI/TestPyPI.

### Documentation

Build and serve documentation locally:
```bash
mkdocs serve
```

Documentation uses MkDocs with Material theme. Configuration in `mkdocs.yml`, content in `docs/`.

## Architecture

### Database Structure

Sicifus stores structural data in a partitioned Parquet database with separate directories:
- `backbone/` - Backbone atoms (CA trace)
- `heavy_atoms/` - Protein heavy atoms (sidechains included)
- `hydrogens/` - Hydrogen atoms (if protonated)
- `ligands/` - Ligand molecules
- `metadata/` - External metadata tables

Each directory contains partitioned `.parquet` files loaded lazily via Polars `scan_parquet()`.

### Core Modules

**`api.py`** - Main user-facing API
- `Sicifus` class: Primary interface for all operations
- Lazy-loads Parquet data as `pl.LazyFrame` objects
- Orchestrates calls to specialized modules

**`io.py`** - Data ingestion
- `CIFLoader`: Parses CIF/PDB files using gemmi library
- Batch processing: groups files into partitions
- Optional protonation: Uses PDBFixer to add hydrogens before parsing
- `_parse_structure()`: Separates backbone, heavy atoms, hydrogens, ligands

**`align.py`** - Structural alignment
- `StructuralAligner`: Kabsch algorithm for superposition
- 3Di encoding: Converts CA traces into 20-state structural alphabet (theta/tau discretization)
- Numba-accelerated angle calculations (`@jit(nopython=True)`)
- K-mer indexing for fast prefiltering

**`analysis.py`** - Phylogenetics and analysis
- `AnalysisToolkit`: RMSD matrices, clustering, trees
  - Vectorized Kabsch for same-length structures
  - Parallel computation with joblib
  - Optional 3Di k-mer prefiltering for large datasets
- `LigandAnalyzer`: Binding pocket residue counts, interactions

**`mutate.py`** - Mutation and stability prediction
- `MutationEngine`: In silico mutagenesis with OpenMM force fields
- Energy minimization: AMBER14 force field, implicit solvent
- `Mutation` class: Parse notation like "G13L" (Gly13→Leu)
- `_RepairCache`: Caches protonated/minimized wild-type for batch mutations
- Result dataclasses: `StabilityResult`, `MutationResult`, `BindingResult`

**`energy.py`** - Energy scoring
- `XTBScorer`: External XTB binary interface (semi-empirical quantum chemistry)
- Work directory management for temporary files

**`kmer_index.py`** - Fast structural search
- K-mer indexing of 3Di sequences for prefiltering alignment candidates

### Key Design Patterns

**Lazy evaluation**: All database queries return `pl.LazyFrame` until `.collect()` is called. This enables query optimization and out-of-core processing.

**Batch processing**: Ingestion processes files in batches (default 100) to create balanced partitions. Mutations can be batched to reuse protonated wild-type structures.

**Parallel computation**: RMSD matrix calculation and alignment use `joblib.Parallel` with `n_jobs=-1` for multi-core processing.

**Structural alphabet**: 3Di encoding enables fast structural comparison without full alignment. Uses 4×5 binning of (theta, tau) angles for each residue.

**Repair-once pattern**: `MutationEngine` minimizes wild-type structure once, then reuses topology/system for batch mutations.

## Dependencies

Core dependencies (always required):
- `polars>=0.20.0` - DataFrame operations
- `gemmi>=0.6.0` - CIF/PDB parsing
- `scipy`, `numpy<2.0`, `matplotlib` - Numerical/scientific
- `networkx` - Graph analysis
- `biopython` - Phylogenetic tree construction
- `numba>=0.57.0` - JIT compilation for alignment
- `joblib` - Parallel processing

Optional dependencies:
- `rdkit` - 2D ligand visualization (install with `pip install "sicifus[viz]"`)
- `pdbfixer`, `openmm>=8.0` - Energy calculations, mutation (install with `pip install "sicifus[energy]"`)

## Common Workflows

**Adding ingestion features**: Modify `CIFLoader._parse_structure()` in `io.py`. Consider what data goes into which partition (backbone vs heavy_atoms vs ligands).

**Adding alignment methods**: Extend `StructuralAligner` in `align.py`. Use Numba `@jit` for performance-critical loops.

**Adding analysis tools**: Add methods to `AnalysisToolkit` or `LigandAnalyzer` in `analysis.py`. Ensure they work with lazy frames where possible.

**Adding mutation features**: Extend `MutationEngine` in `mutate.py`. Remember to update `_RepairCache` if preprocessing logic changes.

**Adding visualization**: Add plotting functions to `visualization.py` following matplotlib patterns. Always support `output_file` parameter. Use existing colormap conventions (`tab20` for ≤20 items).

**Testing slow operations**: Mark with `@pytest.mark.slow` so they can be skipped with `pytest -m "not slow"`. Tests requiring OpenMM should check availability with `pytest.mark.skipif`.

## Mutation Analysis & Visualization

### Statistical Analysis (Industry-parity)

The mutation engine provides comprehensive statistical analysis:

**Key parameters:**
- `keep_statistics=True` - Collect mean, SD, CI from all runs (default)
- `use_mean=False` - Use best energy (default) or mean energy for ddG
- `n_runs=3` - Number of independent minimization runs

**MutationResult statistical fields:**
- `mean_energy`, `sd_energy` - Energy statistics across runs
- `ddg_mean`, `ddg_sd` - Mean-based ddG ± SD
- `ddg_ci_95` - 95% confidence interval (t-distribution)
- `convergence_metric` - CV (SD/mean), warns if >0.1
- `all_run_energies` - Raw energies from all runs

**Example workflow:**
```python
result = engine.mutate("protein.pdb", ["F13A"], n_runs=5, keep_statistics=True)
print(f"ddG: {result.ddg_mean['F13A']:.2f} ± {result.ddg_sd['F13A']:.2f}")
print(f"95% CI: {result.ddg_ci_95['F13A']}")
```

### Visualization Module

Five plotting functions in `visualization.py`:

1. **`plot_ddg()`** - Sorted bar chart with stability colors and error bars
2. **`plot_energy_terms()`** - Energy decomposition (grouped or stacked bars)
3. **`plot_position_scan_heatmap()`** - PSSM-like 20×N heatmap
4. **`plot_alanine_scan()`** - Sorted bars highlighting hotspots
5. **`plot_ddg_distribution()`** - Histogram with statistics overlay

All functions:
- Follow matplotlib patterns: `fig, ax = plt.subplots()` → plot → `tight_layout()` → `savefig()`
- Support `output_file` parameter (save if provided, else `plt.show()`)
- Return processed DataFrames for further analysis
- Use existing colormap conventions (`tab20`, `RdBu_r`)

**API integration:**
```python
db = Sicifus("my_db")
results_df = db.mutate_batch("1CRN", mutations_df, n_runs=5)
db.plot_mutation_results(results_df, output_file="ddg.png", plot_type="ddg")
db.plot_position_scan(scan_df, output_file="heatmap.png")
```

### Advanced industry-standard Features

Three additional features for protein engineering and analysis:

**1. Mutation-to-Binding Pipeline**

Single-command workflow for interface mutagenesis:
```python
result = engine.mutate_interface(
    "complex.pdb",
    mutations={"A": ["F13A"], "B": ["Y25F"]},
    chains_a=["A"], chains_b=["B"]
)
# Returns ΔΔG_binding + ΔΔG_stability per chain
```

Key outputs:
- `ddg_binding` - Change in binding affinity
- `ddg_stability_a`, `ddg_stability_b` - Stability changes per chain
- `mutant_pdb` - Mutant complex structure

**2. Disulfide Bond Analysis**

Detect and analyze S-S bridges:
```python
# Detect all disulfides
disulfides = engine.detect_disulfides("protein.pdb", distance_cutoff=2.5)

# Analyze mutation impact
impact = engine.analyze_mutation_disulfide_impact("protein.pdb", ["C42A"])
# Returns: broken_bonds, new_bonds, affected_cysteines
```

**3. Residue Interaction Networks**

Graph-based analysis of residue interactions:
```python
# Build network (nodes=residues, edges=contacts)
G = toolkit.compute_residue_interaction_network(df, distance_cutoff=5.0)

# Identify hub residues
hubs = toolkit.analyze_network_centrality(G, top_n=10)

# Visualize
toolkit.plot_interaction_network(G, output_file="network.png")
```

Uses NetworkX for:
- Centrality metrics (degree, betweenness, closeness)
- Hub identification for mutation guidance
- Allosteric pathway analysis
