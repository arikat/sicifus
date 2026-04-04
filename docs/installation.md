# Installation

Sicifus requires Python 3.9+.

## Recommended Installation (Conda)

We strongly recommend using a fresh Conda environment to avoid version conflicts (especially with NumPy/Numba).

```bash
# 1. Create a fresh environment
conda create -n sicifus_env python=3.11
conda activate sicifus_env

# 2. Install core numerical stack (prevents binary conflicts)
# Note: We pin numpy<2.0 because Numba is not yet compatible with NumPy 2.0+
# Note: We install llvmlite/numba via conda to avoid build errors on some systems
conda install -c conda-forge "numpy<2.0" scipy matplotlib networkx polars gemmi openmm pdbfixer xtb llvmlite numba

# 3. Install Sicifus
pip install ".[all]"
```

## Fast Installation (uv)

If you prefer `uv` for lightning-fast package management:

```bash
# 1. Create venv
uv venv sicifus_env
source sicifus_env/bin/activate

# 2. Install dependencies
# Note: xtb still needs to be installed via conda or available in PATH
uv pip install "numpy<2.0"
uv pip install ".[all]"
```

## Standard Installation

```bash
git clone https://github.com/avenkat/sicifus.git
cd sicifus
pip install .
```

## Dependencies

-   `polars`: Fast DataFrames.
-   `gemmi`: Macromolecular structure parsing.
-   `numpy` & `scipy`: Numerical computing and alignment algorithms.
-   `networkx`: Graph/Network analysis.
-   `matplotlib`: Visualization.
-   `rdkit` (optional): 2D ligand depiction. Install with `pip install sicifus[viz]`.

## Installing with RDKit
For 2D ligand visualization features, you need RDKit.

```bash
pip install ".[viz]"
```

## Installing with Energy & Mutation Tools
The `[energy]` extra installs `openmm` and `pdbfixer`, which power both the **xTB energy scoring** pipeline and the **Mutation & Stability Engine** (in silico mutagenesis, ddG, alanine scanning, etc.).

```bash
# Python dependencies (covers both xTB scoring and mutation engine)
pip install ".[energy]"

# xTB binary (only needed for ligand energy scoring, not for mutations)
conda install -c conda-forge xtb
```

The mutation engine works entirely through OpenMM and PDBFixer — no external binaries required.

## Full Installation
To install everything:

```bash
pip install ".[all]"
conda install -c conda-forge xtb
```
