# Structural Alignment Tutorial

Sicifus uses a specialized structural alphabet encoder (inspired by Foldseek's 3Di) to perform sequence-independent structural alignments. This allows for extremely fast comparisons of protein structures, even when sequence identity is low.

## Aligning All Structures

The `align_all` method aligns all structures in the database to a reference structure.

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")

# Align all structures to "1abc"
results = db.align_all(reference_id="1abc")

# Sort by RMSD to find the closest matches
top_matches = results.sort("rmsd").head(5)
print(top_matches)
```

## Getting Aligned Coordinates

Once you have identified interesting structures, you can retrieve their coordinates transformed into the reference frame.

```python
# Get the coordinates of '2xyz' aligned to '1abc'
aligned_df = db.get_aligned_structure(structure_id="2xyz", reference_id="1abc")

# aligned_df now contains transformed x, y, z coordinates
```

## How It Works

1.  **Encoding**: Each structure is converted into a sequence of structural alphabet characters based on local torsion angles.
2.  **Alignment**: A fast sequence alignment algorithm (Needleman-Wunsch) is used to align the structural sequences.
3.  **Superposition**: The optimal superposition (rotation + translation) is computed using the Kabsch algorithm on the aligned residues.
4.  **RMSD Calculation**: The Root Mean Square Deviation is calculated between the superposed structures.

This approach is significantly faster than traditional iterative 3D alignment methods while maintaining high accuracy for structural similarity detection.
