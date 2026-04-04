# Ligand Analysis Tutorial

Sicifus provides tools to analyze ligand binding sites across massive datasets. You can identify binding residues, detect pi-stacking interactions, and visualize atom-level contacts.

## 1. Binding Residue Distribution

Identify which residues are commonly found near a specific ligand (using CA atoms for speed).

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")

# Analyze binding residues for Glucose (GLC)
db.analyze_ligand_binding("GLC")
```

## 2. Pi-Stacking Interactions

Detect aromatic stacking interactions (sandwich, parallel displaced, T-shaped) between protein aromatic residues (PHE, TYR, TRP, HIS) and ligand aromatic rings.

**Note**: Requires `all_atom` data. If you ingested before v0.2.0, re-run `ingest`.

```python
# Detect pi-stacking with ATP
pi_df = db.analyze_pi_stacking("ATP")
# Found 47 interactions: 8 sandwich, 31 parallel displaced, 8 T-shaped
print(pi_df)

# Save the plot
db.analyze_pi_stacking("ATP", output_file="pi_stacking.png")
```

## 3. Atom-Level Contacts

Identify specific atomic contacts (e.g., hydrogen bonds) between ligand and protein atoms.

**Note**: Requires `all_atom` data.

```python
# Analyze contacts for Glucose (GLC) with a 3.3 Å cutoff (H-bonds)
contacts = db.analyze_ligand_contacts("GLC", distance_cutoff=3.3)
# Found 156 contacts (42 potential H-bonds: N/O/S pairs)
print(contacts)

# Widen the cutoff for more general contacts
contacts_wide = db.analyze_ligand_contacts("GLC", distance_cutoff=4.5)

# Save both the contacts chart and the 2D ligand depiction
db.analyze_ligand_contacts("GLC", output_file="contacts.png",
                            ligand_2d_file="glc_2d.png")

# If you only pass output_file, the 2D depiction auto-saves as {stem}_ligand2d.png
db.analyze_ligand_contacts("GLC", output_file="contacts.png")
# -> also saves contacts_ligand2d.png
```

## 4. Binding Pocket Composition

Filter structures based on the amino acid composition of their binding pockets.

```python
import polars as pl

# Get a DataFrame of residue counts in the binding pocket (e.g. within 8 Å)
pockets = db.get_binding_pockets("GLC", distance_cutoff=8.0)
print(pockets)
# shape: (150, 21)
# ┌──────────────┬─────┬─────┬─────┬─────┐
# │ structure_id ┆ ALA ┆ ARG ┆ ... ┆ TRP │
# ╞══════════════╪═════╪═════╪═════╪═════╡
# │ 1abc         ┆ 2   ┆ 0   ┆ ... ┆ 1   │
# │ 2xyz         ┆ 1   ┆ 1   ┆ ... ┆ 0   │
# └──────────────┴─────┴─────┴─────┴─────┘

# Filter: Find all structures that have at least one Tryptophan in the pocket
trp_binders = pockets.filter(pl.col("TRP") > 0)
print(f"Found {trp_binders.height} structures with TRP in the GLC pocket.")

# Visualize the aggregate composition across all structures
db.analyze_binding_pocket("GLC", distance_cutoff=8.0)
```

## 2D Visualization (RDKit)

The `analyze_ligand_contacts` function generates a 2D depiction of the ligand with atoms color-coded by contact count (red = many contacts, blue = few).

This requires **RDKit**. Install with:
`pip install sicifus[viz]` or `pip install rdkit`.

If RDKit is not available, the contacts bar chart still works, but the 2D image is skipped.
