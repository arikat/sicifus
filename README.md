<div align="center">
  <h1>Sicifus</h1>
  
  <p>
    <strong>Large-scale Structural Biology Analysis</strong>
  </p>
  <p>
    <a href="https://avenkat.github.io/sicifus"><strong>Documentation</strong></a> ·
    <a href="#installation"><strong>Installation</strong></a> ·
    <a href="#quick-start"><strong>Quick Start</strong></a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
    <img src="https://img.shields.io/badge/polars-fast-orange.svg" alt="Powered by Polars">
  </p>
</div>


![image](docs/assets/logo.png)

---

**Sicifus** is a high-performance Python library for ingesting, aligning, and analyzing thousands of macromolecular structures (CIF/PDB). By compiling structural data into a partitioned **Parquet** database, it enables out-of-core processing and rapid querying of datasets far larger than available RAM.

## Features

-   **Massive Ingestion**: Parse thousands of `.cif` or `.pdb` files into a structured, queryable database.
-   **Fast Alignment**: Sequence-independent structural alignment using a 3Di-inspired structural alphabet.
-   **Phylogenetics**: Generate RMSD-based phylogenetic trees and structural similarity networks.
-   **Ligand Analysis**: Analyze binding pockets, protein-ligand interactions, and atom-level contacts.
-   **Mutation & Stability**: In silico mutagenesis, ddG prediction, alanine scanning, and binding energy — powered by OpenMM.
-   **Metadata Integration**: Seamlessly join external metadata with structural data.

## Installation

```bash
pip install sicifus
```

For 2D ligand visualization features (requires RDKit):

```bash
pip install "sicifus[viz]"
```

## Quick Start

### 1. Ingest Data

Compile your raw structure files into the database format once.

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")
db.ingest("./data/cif_files", batch_size=100)
```

### 2. Structural Alignment

Align an entire dataset to a reference structure.

```python
# Align all structures to "1atp"
results = db.align_all(reference_id="1atp")
print(results.sort("rmsd").head(5))
```

### 3. Phylogenetics & Clustering

Generate a structural tree and identify clusters based on RMSD.

```python
# Generate tree and export to Newick
db.generate_tree(output_file="tree.png", newick_file="tree.nwk")

# Annotate clusters with a 2.0 Å RMSD threshold
db.annotate_clusters(distance_threshold=2.0)
print(db.cluster_summary())
```

### 4. Ligand Analysis

Analyze binding pockets and interactions.

```python
# Get residue counts in the binding pocket for all structures
pockets = db.get_binding_pockets("LIG", distance_cutoff=8.0)

# Find structures with Tryptophan in the binding pocket
import polars as pl
trp_binders = pockets.filter(pl.col("TRP") > 0)
```

### 5. Mutation & Stability

Predict the effect of point mutations on protein stability using open-source tools.

```python
from sicifus import MutationEngine

engine = MutationEngine()

# Calculate total stability
result = engine.calculate_stability("my_protein.pdb")
print(f"Total energy: {result.total_energy:.1f} kcal/mol")

# Mutate Phe-13 to Ala and get ddG
result = engine.mutate("my_protein.pdb", ["F13A"])
print(f"ddG: {result.ddg['F13A']:+.2f} kcal/mol")

# Batch mutations from a CSV
mutations_df = engine.load_mutations("mutations.csv")
results = engine.mutate_batch("my_protein.pdb", mutations_df)
print(results)
```

## Documentation

Full documentation is available at [arikat.github.io/sicifus](https://arikat.github.io/sicifus).

## License

MIT
