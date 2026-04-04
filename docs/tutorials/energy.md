# Energy Scoring Tutorial

Sicifus integrates with **GFN2-xTB** (via the `xtb` package) to perform semi-empirical quantum mechanical scoring of ligand binding poses. This goes beyond geometric analysis by calculating the actual electronic stability of a ligand within its protein pocket.

## Prerequisites

You must have `xtb` installed and available in your system PATH. For robust protonation, **OpenMM**, **PDBFixer**, and **Meeko** are highly recommended.

```bash
conda install -c conda-forge xtb openmm pdbfixer meeko
```

## Protonation (Crucial Step)

Semi-empirical methods like xTB require explicit hydrogens on all atoms. Standard PDB/CIF files often lack hydrogens or contain only polar hydrogens.

**Recommendation:** Add hydrogens during **ingestion**. This ensures all your structures are physically complete and consistent before you start any analysis.

```python
# Ingest with protonation enabled (uses PDBFixer + Meeko)
db.ingest("./data/cif_files", protonate=True)
```

Sicifus will:
1.  Use **PDBFixer** to protonate the protein at pH 7.4.
2.  Use **Meeko** (or RDKit) to protonate ligands.
    *   **Crucially**, it strips any existing hydrogens from the input file first to ensure a complete and consistent protonation state (All Atom model), preventing issues with "missing non-polar hydrogens" common in crystal structures.

If you skip this, Sicifus will attempt to add hydrogens on-the-fly using Meeko/RDKit/OpenBabel, but this is less robust for protein pockets than the ingestion-time PDBFixer method.

### Best Practice: Separate Databases

To avoid mixing original crystal coordinates (for geometric analysis) with fixed/protonated coordinates (for energy scoring), use separate database directories:

```python
# Raw data for geometry (RMSD, contacts)
db_raw = Sicifus(db_path="./db_raw")
db_raw.ingest("./data", protonate=False)

# Protonated data for energy scoring
db_energy = Sicifus(db_path="./db_energy")
db_energy.ingest("./data", protonate=True)
```

## How It Works

1.  **Extraction**: Sicifus extracts the ligand and all protein atoms within a specified cutoff (e.g., 6 Å).
2.  **Protonation Check**: If hydrogens are missing (and weren't added during ingestion), it attempts to add them on-the-fly.
3.  **Constraint Generation**: A constraint file is created to "freeze" the protein atoms in place. This ensures the pocket shape is maintained while allowing the ligand to relax.
4.  **Optimization**: `xtb` runs a geometry optimization on the ligand using the GFN2-xTB method.
5.  **Interaction Energy**: Sicifus calculates the **Interaction Energy** ($E_{\text{int}}$) to allow comparison across different pockets:
    $$E_{\text{int}} = E_{\text{complex}} - (E_{\text{protein}} + E_{\text{ligand}})$$
    *   $E_{\text{complex}}$: Optimized energy of Pocket + Ligand.
    *   $E_{\text{protein}}$: Single-point energy of the Pocket alone.
    *   $E_{\text{ligand}}$: Single-point energy of the Ligand alone.

## Scoring a Single Structure

Use `score_ligand_energy` to score a specific ligand in a specific structure.

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")

# Score ligand "GLC" in structure "1abc"
# Freezes protein atoms within 6.0 Å
result = db.score_ligand_energy("GLC", "1abc", distance_cutoff=6.0)

print(result)
# {
#   'interaction_energy': -25.4,   # kcal/mol — binding strength (key metric)
#   'energy': -123.45678,          # Total complex energy (Hartree)
#   'gap': 2.13,                   # HOMO-LUMO gap (eV)
#   'e_complex': -123.45678,       # E(protein + ligand) optimized (Hartree)
#   'e_protein': -100.12345,       # E(protein pocket) single-point (Hartree)
#   'e_ligand': -23.29388,         # E(ligand alone) single-point (Hartree)
# }
```

### Output Fields

| Field | Unit | Description |
|---|---|---|
| `interaction_energy` | kcal/mol | $E_{\text{complex}} - (E_{\text{protein}} + E_{\text{ligand}})$. **Primary metric.** More negative = stronger binding. |
| `gap` | eV | HOMO-LUMO gap. Larger = more stable electronic state. See [Interpreting the Gap](#interpreting-the-homo-lumo-gap). |
| `energy` | Hartree | Total optimized complex energy. Only meaningful for comparing identical systems. |
| `e_complex` | Hartree | Same as `energy` — the optimized complex. |
| `e_protein` | Hartree | Single-point energy of the frozen protein pocket. |
| `e_ligand` | Hartree | Single-point energy of the isolated ligand. |

## Comparing Binding Pockets

The real power comes from comparing the *same ligand* across *different structures* (e.g., homologs or different conformations). Because we calculate interaction energy, we can compare pockets with different amino acid compositions.

```python
import polars as pl

# 1. Get all structures containing the ligand
ligand = "LIG"
structure_ids = db.ligands.filter(pl.col("residue_name") == ligand)\
                          .select("structure_id").unique().collect().to_series().to_list()

print(f"Scoring {len(structure_ids)} structures for {ligand}...")

results = []
for sid in structure_ids:
    score = db.score_ligand_energy(ligand, sid, distance_cutoff=6.0, debug=True)
    
    if "interaction_energy" in score:
        results.append({
            "structure_id": sid,
            **score  # Capture all xTB output fields
        })

# 2. Convert to DataFrame and sort by strongest binding
df_scores = pl.DataFrame(results).sort("interaction_energy")

print("Top 5 strongest binding pockets:")
print(df_scores.head(5))

# 3. Filter out likely failed calculations (huge positive energy = diverged optimization)
df_reliable = df_scores.filter(pl.col("interaction_energy").abs() < 50)
print(f"\nReliable results ({df_reliable.height}/{df_scores.height}):")
print(df_reliable)
```

## Comparing Conformers (Relative Energy)

If you are comparing different conformers of the **exact same system** (e.g., docking poses of the same ligand in the same protein), you can use the **Relative Energy** (kcal/mol). This is derived from the total energy (Hartree).

```python
# Assume df_scores has an "energy" column (Hartree)
# Calculate relative energy (kcal/mol) relative to the global minimum
df_relative = db.calculate_relative_energy(df_scores)

print(df_relative.select(["structure_id", "energy", "relative_energy_kcal"]))
# structure_id  | energy (Eh) | relative_energy_kcal
# --------------------------------------------------
# pose_1        | -123.456    | 0.00   (Best)
# pose_2        | -123.454    | 1.25   (+1.25 kcal/mol)
# pose_3        | -123.450    | 3.76   (+3.76 kcal/mol)
```

If you have multiple ligands and want to find the best pose *per ligand*, use `group_by`:

```python
# Calculate relative energy per ligand group
df_relative = db.calculate_relative_energy(df_scores, group_by="ligand_name")
```

## Interpreting Results

### Interaction Energy (kcal/mol)

This is the **primary metric** for binding affinity:

| Range | Interpretation |
|---|---|
| < −10 | Strong binding (typical for drug-like molecules) |
| −10 to −1 | Moderate binding |
| −1 to 0 | Weak binding |
| > 0 | Unfavorable (repulsive) — ligand doesn't "want" to be there |
| > 100 | Almost certainly a failed calculation — check the gap and charge |

### Interpreting the HOMO-LUMO Gap

The HOMO-LUMO gap relates to electronic stability, but its usefulness **depends on system size**.

**For small molecules (< 50 atoms):**

| Gap (eV) | Interpretation |
|---|---|
| > 1.0 | Healthy electronic structure |
| 0.1 – 1.0 | Borderline |
| < 0.1 | Likely problematic — check charge |

**For protein-ligand pockets (100–300+ atoms):**

Gaps of **0.01–0.05 eV are normal** with GFN2-xTB. This is a known behavior of tight-binding methods on large systems — the density of electronic states increases with system size, making the frontier orbital gap systematically smaller. This does **not** mean the calculation failed.

The interaction energy (a *difference* of energies) benefits from error cancellation and remains robust even when the absolute gap is tiny. You can verify this by noting that interaction energies are consistent regardless of charge assignment.

**When to worry:**
- Gap is exactly 0.0 eV — SCF likely did not converge at all
- Gap is tiny **and** interaction energy is hugely positive (> 50 kcal/mol) — both indicate failure

**Tip:** For large systems, filter on interaction energy magnitude rather than gap:

```python
# Flag likely failed calculations by unreasonable interaction energy
df_reliable = df_scores.filter(pl.col("interaction_energy").abs() < 50)
```

### Total Energy (Hartree)

Useful **only** for comparing *identical* systems (e.g., conformers of the exact same protein-ligand complex). Do **not** use this to compare different pockets — use `interaction_energy` instead.

## Advanced Options

You can customize the calculation parameters:

```python
db.score_ligand_energy(
    "LIG", "1abc",
    distance_cutoff=8.0,  # Include more protein context
    charge=-1,            # Specify system charge if known
    solvent="water"       # Use implicit solvation (ALPB)
)
```

## Debugging

Use `debug=True` to save input/output structures for visual inspection:

```python
score = db.score_ligand_energy("LIG", "my_structure", debug=True)
# Saves to xtb_work/debug_structures/:
#   my_structure_input.xyz   — structure sent to xTB
#   my_structure_input.pdb   — same, converted for viewers
#   my_structure_optimized.xyz — xTB-optimized geometry
```

Open these in PyMOL, ChimeraX, or similar to verify:
- Hydrogens are present on both protein and ligand
- No "protons in space" (isolated H atoms far from any heavy atom)
- Complete residues (no fragmented amino acids)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `xtb not found` | Binary not in PATH | `conda install -c conda-forge xtb` |
| All gaps < 0.1 eV (large systems) | **Normal** for 100+ atom systems with GFN2-xTB | Trust interaction energy instead; see [gap section](#interpreting-the-homo-lumo-gap) |
| Gap = 0.0 eV exactly | SCF did not converge | Try different `charge=`; reduce `distance_cutoff` |
| Huge positive energies (>50 kcal/mol) | Failed optimization / bad geometry | Check debug structures; try `protonate=True` during ingestion |
| Ligand has 0 hydrogens | Protonation failed | Re-ingest with `protonate=True`; check `[Ligand H]` log messages |
| xTB crashes with stack trace | Fragmented residues or missing H | Ensure protonation; Sicifus now extracts complete residues |
| Very slow | Large pocket | Reduce `distance_cutoff` (e.g., 5.0 Å) |
