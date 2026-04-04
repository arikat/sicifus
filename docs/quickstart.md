# Quick Start

## 1. Ingesting Data

First, compile your raw structure files into the Sicifus database format. This only needs to be done once.

```python
from sicifus import Sicifus

# Initialize the tool pointing to a database location (will be created if not exists)
db = Sicifus(db_path="./my_structural_db")

# Ingest a folder of CIF files (default)
db.ingest(input_folder="./data/cif_files", batch_size=100)

# Or ingest PDB files
db.ingest(input_folder="./data/pdb_files", file_extension="pdb")
```

## 2. Loading and Querying

Once ingested, you can load the database instantly. Sicifus uses lazy loading, so it won't consume memory until you compute a result.

```python
# Load the database
db = Sicifus(db_path="./my_structural_db")

# Access the backbone data (LazyFrame)
print(db.backbone.schema)

# Get a specific structure as a DataFrame
structure = db.get_structure("1abc")
print(structure)
```

## 3. Massive Structural Alignment

Align an entire dataset to a reference structure to find structural homologs.

```python
# Align all structures in the DB to "1abc"
# Returns a DataFrame with RMSD and alignment statistics
results = db.align_all(reference_id="1abc")

# Sort by RMSD to find the closest matches
top_matches = results.sort("rmsd").head(5)
print(top_matches)
```

## 4. Coordinate Transformation

You can retrieve the coordinates of any structure *after* it has been superposed onto a reference.

```python
# Get the coordinates of '2xyz' aligned to '1abc'
aligned_df = db.get_aligned_structure(structure_id="2xyz", reference_id="1abc")

# aligned_df now contains transformed x, y, z coordinates
```
