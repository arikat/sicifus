# Ingestion Tutorial

Sicifus is designed to handle massive datasets by converting raw structure files (CIF/PDB) into a partitioned Parquet database. This process is called "ingestion".

## Basic Ingestion

The `ingest` method scans a directory for structure files and processes them in batches.

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")

# Ingest CIF files (default)
db.ingest("./data/cif_files")
```

## Supported Formats

Sicifus supports:
-   `.cif` (mmCIF)
-   `.pdb` (PDB format)
-   `.cif.gz` / `.pdb.gz` (Gzipped files are automatically handled)

To ingest PDB files:

```python
db.ingest("./data/pdb_files", file_extension="pdb")
```

## Batch Size

The `batch_size` parameter controls how many structures are written to a single Parquet partition file.
-   **Larger batches**: Fewer files, slightly faster ingestion, more RAM usage.
-   **Smaller batches**: More files, better for low-memory environments.

Default is 100.

```python
db.ingest("./data/huge_dataset", batch_size=500)
```

## Protonation (Adding Hydrogens)

If you plan to use **Energy Scoring (xTB)**, it is highly recommended to add hydrogens during ingestion. This ensures consistent protonation states across your entire dataset.

Sicifus uses:
-   **PDBFixer (OpenMM)** to robustly add hydrogens to the **protein** at pH 7.4.
-   **Meeko (Preferred)** or **RDKit** to add hydrogens to **ligands**.
    -   Existing hydrogens are stripped first to ensure complete and consistent protonation.
    -   Meeko is configured to preserve all explicit hydrogens (All Atom model) for QM calculations.

**Prerequisites:**
```bash
conda install -c conda-forge openmm pdbfixer rdkit meeko
```

**Usage:**
```python
# Ingest with protonation enabled
db.ingest("./data/cif_files", protonate=True)
```

**Note:**
-   This will slow down ingestion significantly as each structure is processed by PDBFixer and RDKit.
-   The resulting database will contain all hydrogen atoms in the `hydrogens/` dataset.
-   If you skip this step, Sicifus will attempt to add hydrogens on-the-fly during energy scoring, which is less robust and slower.

## Managing Raw vs. Protonated Data

If you need to maintain both the original experimental coordinates (for geometric analysis) and the fixed/protonated coordinates (for energy scoring), it is best practice to use separate database directories. This prevents mixing coordinate frames and ensures clean data separation.

```python
# 1. Create a database for raw geometric analysis (original coordinates)
db_raw = Sicifus(db_path="./db_raw")
db_raw.ingest("./data/cif_files", protonate=False)

# 2. Create a separate database for energy scoring (fixed/protonated coordinates)
db_energy = Sicifus(db_path="./db_energy")
db_energy.ingest("./data/cif_files", protonate=True)
```

-   **`db_raw`**: Use for RMSD alignment, contact analysis, and pi-stacking (where original crystal coordinates are preferred).
-   **`db_energy`**: Use for `score_ligand_energy` and `calculate_relative_energy` (where explicit hydrogens and complete residues are required).

## What Happens During Ingestion?

1.  **Protonation (Optional)**: If `protonate=True`:
    -   PDBFixer adds missing atoms and hydrogens to the protein.
    -   RDKit adds hydrogens to ligands.
2.  **Parsing**: Gemmi parses the structure file (or the protonated temporary file).
3.  **Extraction**:
    -   **Backbone**: CA atoms are extracted for fast alignment.
    -   **Heavy Atoms**: All non-hydrogen protein atoms.
    -   **Hydrogens**: All protein hydrogen atoms (stored separately for efficiency).
    -   **Ligands**: Non-polymer residues (protonated if requested).
4.  **Partitioning**: Data is written to `backbone/`, `heavy_atoms/`, `hydrogens/`, and `ligands/` subdirectories in Parquet format.

## Re-ingesting

If you add new files to your folder, you can run `ingest` again. However, currently, it will re-process the entire folder. Incremental updates are planned for future versions.

If you previously ingested data before the "All Atom" feature was added (v0.2.0+), simply re-run `ingest` to generate the `all_atom` dataset required for pi-stacking and contact analysis.
