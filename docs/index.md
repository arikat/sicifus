# Sicifus

**Sicifus** (pronounced Sisyphus) is a high-performance Python library designed for massive-scale structural biology analysis. It leverages the speed of **Polars** and **Gemmi** to ingest, align, and analyze thousands of macromolecular structures (CIF/PDB files) efficiently.

Unlike traditional tools that rely on slow, file-based loops, Sicifus compiles structural data into a highly optimized, partitioned **Parquet** dataset. This allows for out-of-core processing, enabling you to query and analyze datasets far larger than your available RAM.

## Key Features

-   **Massive Ingestion**: Rapidly parse folders containing thousands of structure files (`.cif`, `.pdb`, `.pdb.gz`) into a structured, queryable database.
-   **Partitioned Storage**: Data is stored as a partitioned Parquet dataset, separating backbone coordinates from ligand information for optimized query performance.
-   **Fast Structural Alignment**: Implements a structural alphabet encoder (inspired by Foldseek's 3Di) to perform sequence-independent structural alignments orders of magnitude faster than traditional methods.
-   **RMSD & Superposition**: Includes a vectorized Kabsch algorithm for precise coordinate superposition and RMSD calculation.
-   **Fast Structural Clustering**: Greedy centroid-based clustering with 3Di k-mer prefiltering — cluster thousands of structures in seconds without computing a full distance matrix.
-   **Structural Phylogenetics**: Generate RMSD-based phylogenetic trees with k-mer prefiltered alignment (20-100x faster) and structural similarity networks.
-   **Ligand Analysis**: Specialized tools to identify and visualize ligand-binding residues across massive datasets.
-   **Mutation & Stability**: Physics-based in silico mutagenesis, stability scoring, and binding energy calculation using OpenMM — using open-source OpenMM.

## Architecture

Sicifus organizes data into three main partitioned datasets:

1.  **Backbone**: C-alpha atoms only. Fast alignment, RMSD, and tree generation.
2.  **All Atom**: All protein atoms including sidechains. Used for pi-stacking and detailed contact analysis.
3.  **Ligands**: Non-polymer, non-water residues. Used for binding site analysis.

The **Structural Aligner** converts 3D coordinates into 1D "structural sequences" based on local torsion angles. This allows the use of fast sequence alignment algorithms (Needleman-Wunsch) to find the optimal superposition, rather than slow iterative 3D optimization.
