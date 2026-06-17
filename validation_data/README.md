# validation_data/

Scratch directory for **downloaded** datasets and structures used by the
calibration and validation scripts. Everything here except this README is
git-ignored — the files are large, regenerable, and (for SKEMPI) under a
license that does not permit redistribution. Re-create them locally as needed.

## SKEMPI 2.0 (binding ΔΔG)

Used by `examples/calibrate_empirical.py --format skempi`.

```bash
curl -L -o validation_data/skempi_v2.csv \
  "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
```

Parser uses the `Mutation(s)_PDB` column (deposited/author numbering) — **not**
`Mutation(s)_cleaned`, which SKEMPI renumbers to a canonical sequence that does
not match PDB residue numbers.

**Cite:** Jankauskaitė et al. (2019), *SKEMPI 2.0: an updated benchmark of
changes in protein–protein binding energy, kinetics and thermodynamics upon
mutation*, Bioinformatics 35(3):462–469. https://doi.org/10.1093/bioinformatics/bty635

## PDB structures

`examples/1BNI.pdb` (barnase, the embedded stability set) and any
`<id>_clean.pdb` complexes are fetched/cleaned on demand from RCSB
(<https://files.rcsb.org/download/<ID>.pdb>) and HETATM-stripped. To recreate
the barnase demo structure:

```bash
curl -L "https://files.rcsb.org/download/1BNI.pdb" \
  | grep -E '^(ATOM|TER|MODEL|ENDMDL|END)' > examples/1BNI.pdb
```
