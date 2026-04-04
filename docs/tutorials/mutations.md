# Mutation & Stability Engine

Sicifus includes a physics-based mutation and stability engine powered by **OpenMM** and **PDBFixer**. It provides structure repair, in silico mutagenesis, stability scoring, binding energy calculation, and scanning — using open-source tools.

## Prerequisites

The mutation engine requires `openmm` and `pdbfixer`, which are already part of the `[energy]` optional dependency group:

```bash
pip install "sicifus[energy]"
```

Or via conda (recommended):

```bash
conda install -c conda-forge openmm pdbfixer
```

## How It Works

Under the hood, the engine uses:

- **PDBFixer** to repair structures (missing atoms, residues, hydrogens) and apply mutations (residue swaps + sidechain rebuild).
- **OpenMM** with the **AMBER ff14SB** force field and **GBn2 implicit solvent** to minimise structures and calculate potential energies.
- **Harmonic backbone restraints** during mutation minimisation, allowing sidechain flexibility while keeping the backbone fixed.
- **Force group decomposition** to report energy by term (bonds, angles, torsions, nonbonded, solvation).

All energies are reported in **kcal/mol**.

## Standalone vs. Database Usage

The `MutationEngine` can be used in two ways:

**1. Through the Sicifus database** — operates on structures already ingested into your Parquet database:

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")
db.load()

result = db.calculate_stability("1crn")
```

**2. Standalone** — accepts any PDB file path, PDB string, or Polars DataFrame directly:

```python
from sicifus import MutationEngine

engine = MutationEngine()

result = engine.calculate_stability("1crn.pdb")
# or
result = engine.calculate_stability(pdb_string)
# or
result = engine.calculate_stability(atoms_dataframe)
```

The rest of this tutorial uses the standalone `MutationEngine` for clarity. Every method shown here has a corresponding wrapper on the `Sicifus` class that takes a `structure_id` instead.

---

## Specifying Mutations

Mutations use a simple **WtPositionMut** format — just the wild-type residue, position number, and mutant residue:

```
G13L     → Gly at position 13 → Leu
F42W     → Phe at position 42 → Trp
A100V    → Ala at position 100 → Val
```

Chain defaults to `"A"` unless specified separately. You can specify mutations as strings or as `Mutation` objects:

```python
from sicifus import Mutation

# From a string (chain defaults to A)
m = Mutation.from_str("G13L")

# From a string with explicit chain
m = Mutation.from_str("G13L", chain="B")

# Direct construction (1-letter or 3-letter codes)
m = Mutation(position=13, wt_residue="G", mut_residue="L")
m = Mutation(position=13, wt_residue="GLY", mut_residue="LEU", chain="B")
```

---

## Batch Mutations from CSV

For real workflows you typically have a list of mutations — from experimental data, literature, or a design tool. Sicifus ingests these from a CSV file.

### CSV Format

The CSV must have a **`mutation`** column. An optional **`chain`** column provides chain IDs (defaults to `"A"` if absent). Any additional columns are preserved as metadata and carried through to results.

```csv
mutation,chain,source,score
G13L,A,experiment,1.5
F42W,A,design,-0.3
T7V,B,literature,0.8
```

### Loading and Running

```python
from sicifus import MutationEngine

engine = MutationEngine()

# Load the CSV
mutations_df = engine.load_mutations("my_mutations.csv")
print(mutations_df)
# shape: (3, 4)
# ┌──────────┬───────┬────────────┬───────┐
# │ mutation ┆ chain ┆ source     ┆ score │
# ╞══════════╪═══════╪════════════╪═══════╡
# │ G13L     ┆ A     ┆ experiment ┆ 1.5   │
# │ F42W     ┆ A     ┆ design     ┆ -0.3  │
# │ T7V      ┆ B     ┆ literature ┆ 0.8   │
# └──────────┴───────┴────────────┴───────┘

# Run all mutations against a structure
results = engine.mutate_batch("my_protein.pdb", mutations_df, max_iterations=200)
print(results)
# shape: (3, 7)
# ┌──────────┬───────┬────────────┬───────┬───────────┬───────────────┬──────────────┐
# │ mutation ┆ chain ┆ source     ┆ score ┆ wt_energy ┆ mutant_energy ┆ ddg_kcal_mol │
# ╞══════════╪═══════╪════════════╪═══════╪═══════════╪═══════════════╪══════════════╡
# │ G13L     ┆ A     ┆ experiment ┆ 1.5   ┆ -1842.3   ┆ -1838.1       ┆ 4.2          │
# │ F42W     ┆ A     ┆ design     ┆ -0.3  ┆ -1842.3   ┆ -1843.1       ┆ -0.8         │
# │ T7V      ┆ B     ┆ literature ┆ 0.8   ┆ -1842.3   ┆ -1840.5       ┆ 1.8          │
# └──────────┴───────┴────────────┴───────┴───────────┴───────────────┴──────────────┘
```

The same workflow works through the Sicifus database:

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")
db.load()

mutations_df = db.load_mutations("my_mutations.csv")
results = db.mutate_batch("1crn", mutations_df)
```

### Minimal CSV (no chain column)

If all mutations are on chain A, you can omit the chain column entirely:

```csv
mutation
G13L
F42W
T7V
```

---

## Repairing a Structure

`repair()` finds and adds missing atoms/residues, protonates at pH 7.0, and energy-minimises the structure.

```python
engine = MutationEngine()
result = engine.repair("my_protein.pdb", max_iterations=500)

print(f"Energy before: {result.energy_before:.1f} kcal/mol")
print(f"Energy after:  {result.energy_after:.1f} kcal/mol")

# The repaired PDB is available as a string
with open("repaired.pdb", "w") as f:
    f.write(result.pdb_string)
```

### RepairResult fields

| Field | Type | Description |
|---|---|---|
| `energy_before` | float | Energy before minimisation (kcal/mol) |
| `energy_after` | float | Energy after minimisation (kcal/mol) |
| `pdb_string` | str | Repaired + minimised PDB as a string |
| `topology` | object | OpenMM Topology (for advanced use) |
| `positions` | object | OpenMM Positions (for advanced use) |

---

## Calculating Stability

`calculate_stability()` minimises the structure and decomposes the total potential energy into individual force-field terms.

```python
result = engine.calculate_stability("my_protein.pdb", max_iterations=500)

print(f"Total energy: {result.total_energy:.1f} kcal/mol")
print()
for term, value in result.energy_terms.items():
    print(f"  {term}: {value:.1f}")
```

Example output:

```
Total energy: -1842.3 kcal/mol

  HarmonicBondForce: 142.8
  HarmonicAngleForce: 387.2
  PeriodicTorsionForce: 512.1
  NonbondedForce: -1654.3
  CustomGBForce: -1230.1
  total: -1842.3
```

### StabilityResult fields

| Field | Type | Description |
|---|---|---|
| `total_energy` | float | Total potential energy (kcal/mol) |
| `energy_terms` | dict | Per-force-term breakdown (kcal/mol) |
| `pdb_string` | str | Minimised PDB string |

### Energy terms explained

| Term | What it captures |
|---|---|
| `HarmonicBondForce` | Covalent bond stretching |
| `HarmonicAngleForce` | Bond angle bending |
| `PeriodicTorsionForce` | Dihedral/torsion angles |
| `NonbondedForce` | Van der Waals + electrostatics (combined) |
| `CustomGBForce` | Generalised Born implicit solvation (polar) |
| `total` | Sum of all terms |

!!! note
    The individual terms sum to the `total`. You can verify this:
    ```python
    terms_sum = sum(v for k, v in result.energy_terms.items() if k != "total")
    assert abs(terms_sum - result.total_energy) < 1.0
    ```

---

## Single Mutations

`mutate()` applies one or more point mutations, rebuilds the sidechain, minimises the mutant, and reports the change in stability (ddG).

```python
# Using a string
result = engine.mutate("my_protein.pdb", ["F13A"])

# Using a Mutation object
mut = Mutation(position=13, wt_residue="F", mut_residue="A")
result = engine.mutate("my_protein.pdb", [mut])

# With explicit chain
result = engine.mutate("my_protein.pdb", ["F13A"], chain="B")
```

### Reading the result

```python
result = engine.mutate("my_protein.pdb", ["F13A"], max_iterations=200)

# ddG: positive = destabilising, negative = stabilising
for label, ddg in result.ddg.items():
    print(f"{label}: ddG = {ddg:+.2f} kcal/mol")

# Full energy breakdown as a Polars DataFrame
print(result.energy_terms)
# shape: (6, 4)
# ┌─────────────────────┬───────────┬───────────────┬────────┐
# │ term                ┆ wt_energy ┆ mutant_energy ┆ delta  │
# ╞═════════════════════╪═══════════╪═══════════════╪════════╡
# │ HarmonicBondForce   ┆ 142.8     ┆ 138.2         ┆ -4.6   │
# │ NonbondedForce      ┆ -1654.3   ┆ -1640.1       ┆ 14.2   │
# │ ...                 ┆ ...       ┆ ...           ┆ ...    │
# └─────────────────────┴───────────┴───────────────┴────────┘

# The mutant PDB string
mutant_pdb = list(result.mutant_pdbs.values())[0]
```

### MutationResult fields

| Field | Type | Description |
|---|---|---|
| `wt_energy` | float | Wild-type energy (kcal/mol) |
| `mutant_energies` | dict | Mutant energy keyed by mutation label |
| `ddg` | dict | ddG = mutant − wild-type (kcal/mol) |
| `mutant_pdbs` | dict | Mutant PDB strings keyed by mutation label |
| `energy_terms` | DataFrame | Per-term wt vs mutant comparison |

### Interpreting ddG

| ddG (kcal/mol) | Interpretation |
|---|---|
| > +2 | Destabilising mutation |
| +0.5 to +2 | Mildly destabilising |
| −0.5 to +0.5 | Neutral |
| < −0.5 | Stabilising mutation |

### Multiple minimisation runs

Since minimisation can get trapped in local minima, you can run multiple independent minimisations and keep the best energy:

```python
result = engine.mutate("protein.pdb", ["F13A"], n_runs=3)
```

---

## Binding Energy

`calculate_binding_energy()` calculates the binding energy between two groups of chains using a thermodynamic cycle:

$$E_{\text{binding}} = E_{\text{complex}} - (E_{\text{chains\_A}} + E_{\text{chains\_B}})$$

```python
result = engine.calculate_binding_energy(
    "complex.pdb",
    chains_a=["A"],      # receptor
    chains_b=["B"],      # ligand chain
    max_iterations=500,
)

print(f"Binding energy: {result.binding_energy:.1f} kcal/mol")
print(f"Interface residues: {result.interface_residues.height}")
print(result.interface_residues)
```

### BindingResult fields

| Field | Type | Description |
|---|---|---|
| `binding_energy` | float | E_complex − (E_A + E_B) in kcal/mol |
| `complex_energy` | float | Energy of the full complex |
| `chain_a_energy` | float | Energy of chain group A in isolation |
| `chain_b_energy` | float | Energy of chain group B in isolation |
| `interface_residues` | DataFrame | Residues at the interface (within 5 Å) |

---

## Alanine Scanning

`alanine_scan()` systematically mutates each position to alanine to identify energetic hotspot residues.

```python
# Scan specific positions
df = engine.alanine_scan(
    "protein.pdb",
    chain="A",
    positions=[7, 13, 25, 42],
    max_iterations=100,
)
print(df)
# shape: (4, 4)
# ┌───────┬──────────┬────────────┬──────────────┐
# │ chain ┆ position ┆ wt_residue ┆ ddg_kcal_mol │
# ╞═══════╪══════════╪════════════╪══════════════╡
# │ A     ┆ 7        ┆ ILE        ┆ 3.21         │
# │ A     ┆ 13       ┆ PHE        ┆ 8.45         │  ← hotspot
# │ A     ┆ 25       ┆ SER        ┆ 0.12         │
# │ A     ┆ 42       ┆ LEU        ┆ 2.87         │
# └───────┴──────────┴────────────┴──────────────┘

# Scan all eligible residues (skips Ala and Gly automatically)
df = engine.alanine_scan("protein.pdb", chain="A")
```

Large positive ddG values indicate hotspot residues whose sidechains are critical for stability.

---

## Position Scan (PSSM)

`position_scan()` tests all 20 amino acids at specified positions, producing a position-specific scoring matrix.

```python
df = engine.position_scan(
    "protein.pdb",
    chain="A",
    positions=[13],
    max_iterations=100,
)
print(df)
# shape: (20, 5)
# ┌───────┬──────────┬────────────┬─────────────┬──────────────┐
# │ chain ┆ position ┆ wt_residue ┆ mut_residue ┆ ddg_kcal_mol │
# ╞═══════╪══════════╪════════════╪═════════════╪══════════════╡
# │ A     ┆ 13       ┆ PHE        ┆ ALA         ┆ 8.45         │
# │ A     ┆ 13       ┆ PHE        ┆ ARG         ┆ 5.12         │
# │ A     ┆ 13       ┆ PHE        ┆ ...         ┆ ...          │
# │ A     ┆ 13       ┆ PHE        ┆ PHE         ┆ 0.00         │  ← self
# │ A     ┆ 13       ┆ PHE        ┆ TRP         ┆ -0.32        │
# │ ...   ┆ ...      ┆ ...        ┆ ...         ┆ ...          │
# └───────┴──────────┴────────────┴─────────────┴──────────────┘
```

!!! warning
    Position scan is computationally expensive — it runs 19 mutations per position. Plan accordingly.

---

## Per-Residue Energy

`per_residue_energy()` approximates each residue's energetic contribution using an alanine-subtraction method.

```python
df = engine.per_residue_energy("protein.pdb")
print(df)
# shape: (46, 4)
# ┌───────┬────────────────┬──────────────┬─────────────────────────────────┐
# │ chain ┆ residue_number ┆ residue_name ┆ energy_contribution_kcal_mol    │
# ╞═══════╪════════════════╪══════════════╪═════════════════════════════════╡
# │ A     ┆ 1              ┆ THR          ┆ -0.84                           │
# │ A     ┆ 2              ┆ THR          ┆ -1.23                           │
# │ ...   ┆ ...            ┆ ...          ┆ ...                             │
# └───────┴────────────────┴──────────────┴─────────────────────────────────┘
```

!!! warning
    This method runs one mutation per residue, so it is very slow on large proteins.

---

## Using via the Sicifus Database

All methods above have wrappers on the `Sicifus` class that accept a `structure_id` instead of a PDB path:

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")
db.load()

# Repair
result = db.repair_structure("1crn")

# Stability
result = db.calculate_stability("1crn")

# Single mutation
result = db.mutate_structure("1crn", ["F13A"])

# Batch mutations from CSV
mutations_df = db.load_mutations("mutations.csv")
results = db.mutate_batch("1crn", mutations_df)

# Binding energy
result = db.calculate_binding_energy("my_complex", chains_a=["A"], chains_b=["B"])

# Alanine scan
df = db.alanine_scan("1crn", chain="A", positions=[7, 13])

# Position scan
df = db.position_scan("1crn", chain="A", positions=[13])

# Per-residue energy
df = db.per_residue_energy("1crn")
```

---

## Configuration

The `MutationEngine` can be customised at construction:

```python
engine = MutationEngine(
    forcefield="amber14-all.xml",   # OpenMM force field XML
    water_model="implicit",         # "implicit" for GBn2 (fast, default)
    platform="CPU",                 # "CPU", "CUDA", or "OpenCL"
    work_dir="./mutate_work",       # Directory for temporary files
)
```

| Parameter | Default | Description |
|---|---|---|
| `forcefield` | `"amber14-all.xml"` | AMBER ff14SB protein force field |
| `water_model` | `"implicit"` | `"implicit"` uses GBn2 implicit solvent (fast). Pass an explicit water XML like `"amber14/tip3pfb.xml"` for explicit solvent. |
| `platform` | `"CPU"` | OpenMM compute platform. Use `"CUDA"` for GPU acceleration. |
| `work_dir` | `"./mutate_work"` | Directory for temporary files |
