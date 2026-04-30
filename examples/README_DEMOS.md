# Sicifus Demo Scripts

This directory contains interactive examples demonstrating Sicifus mutation analysis capabilities.

## Experimental Validation Data

### Serrano et al. (1993) - Barnase Mutations

The file `serrano1993_experimental_data.csv` contains **17 single-point mutations** of Barnase (1BNI) with experimental ΔΔG values.

**Reference:** Serrano, L., Kellis, J.T., Cann, P., Matouschek, A., & Fersht, A.R. (1993). "Step-wise Mutation of Barnase to Binase." *J. Mol. Biol.* 233, 305-312, Table 3.

**Experimental Conditions:**
- Method: Urea denaturation monitored by fluorescence
- Temperature: 25°C
- pH: 6.3
- Protein: Wild-type Barnase from *Bacillus amyloliquefaciens*

**Mutation Coverage:**
- **Stabilizing** (ΔΔG < -0.2): Q15I, T16R, G65S, K66A, T79V, K108R
- **Neutral** (|ΔΔG| ≤ 0.2): E29Q, D44E, K19R
- **Destabilizing** (ΔΔG > +0.2): H18K, Q31S, I55V, K62R, S85A, I88L, L89V, Q104A

**Expected Computational Performance:**
- R² (correlation): 0.3 - 0.6 typical for force field methods
- MAE (mean absolute error): 0.8 - 1.5 kcal/mol
- RMSE: 1.0 - 2.0 kcal/mol

You can use this dataset to validate your Sicifus installation and benchmark prediction accuracy.

## Quick Start

### Test Your Setup
```bash
python test_setup.py
```
This runs a single mutation (H18K on Barnase) and compares to experimental data to verify your OpenMM installation is working correctly.

**Expected output:**
- ΔΔG: ~0.5-2.0 kcal/mol (experimental: +1.19 kcal/mol from Serrano 1993)
- Error: <1.5 kcal/mol ✅

**If you see:**
- ΔΔG: ±50 kcal/mol ❌ → Something is wrong with structure or force field
- ValueError about HETATM ❌ → PDB file wasn't cleaned properly

## Demo Scripts

### 1. mutation_analysis_demo.py
**Test System:** Barnase (1BNI)

**Features:**
- Single mutations with statistics (multiple runs, error bars, convergence)
- Batch mutations with experimental validation using Serrano et al. (1993) data
- 8 mutations with experimental ΔΔG values from Table 3
- Performance metrics (R², RMSE, MAE) calculated automatically
- Position scanning (all 20 amino acids at selected positions)
- Alanine scanning (hotspot identification)

**Experimental Validation:**
- Uses mutations from `serrano1993_experimental_data.csv`
- Calculates R² correlation against experimental ΔΔG
- Generates scatter plots comparing predicted vs experimental
- Outputs detailed error analysis

**Outputs:**
- `barnase_validation.png` - Scatter plot (predicted vs experimental with R²)
- `barnase_comparison.png` - Bar chart comparison
- `barnase_validation_results.csv` - Full results table with errors

**Runtime:** ~5-10 minutes for batch validation

```bash
python mutation_analysis_demo.py
```

### 2. experimental_validation_demo.py
**Test Systems:** Barnase (1BNI), T4 Lysozyme (2LZM), Chymotrypsin Inhibitor (2CI2)

**Features:**
- Larger validation dataset (13 mutations from 3 proteins)
- Mock vs real prediction modes
- Publication-quality scatter plots and error analysis
- Framework for extending to ProTherm/SKEMPI databases

**Outputs:**
- `validation_plot.png` - Predicted vs experimental scatter
- `error_distribution.png` - Error histogram and analysis
- `validation_data/prediction_results.csv` - Results table

**Runtime:**
- Mock mode: ~10 seconds
- Real predictions: 10-30 minutes

```bash
python experimental_validation_demo.py
```

### 3. interface_analysis_demo.py
**Test System:** Barnase-Barstar Complex (1BRS)

**Features:**
- Interface mutagenesis (ΔΔG_binding calculation)
- Disulfide bond detection and mutation impact
- Residue interaction networks
- Network centrality analysis (hub identification)

**Outputs:**
- `1BRS_R59A_mutant.pdb` - Mutant structure
- `barnase_barstar_network.png` - Interaction network visualization
- `interface_demo_db/` - Sicifus database

**Runtime:** ~5-10 minutes

```bash
python interface_analysis_demo.py
```

## Jupyter Notebooks

Each `.py` script has a corresponding `.ipynb` notebook in this directory:
- `mutation_analysis_demo.ipynb`
- `experimental_validation_demo.ipynb`
- `interface_analysis_demo.ipynb`

The notebooks provide:
- Step-by-step execution
- Inline plots and visualizations
- Educational explanations
- Interactive experimentation

## Requirements

### Basic Requirements
```bash
pip install sicifus
```

### For Mutation Analysis (Required for all demos)
```bash
pip install sicifus[energy]
```
This installs OpenMM and PDBFixer needed for energy calculations.

### For Visualization (Optional)
```bash
pip install sicifus[viz]
```
Adds RDKit for 2D ligand rendering.

## Common Issues

### 1. "No template found for residue (HOH)"
**Problem:** PDB file contains water molecules (HETATM records)

**Solution:** The demo scripts automatically clean PDB files, but if you're using your own structures:
```python
with open("raw.pdb", "r") as f_in, open("clean.pdb", "w") as f_out:
    for line in f_in:
        if line.startswith("ATOM"):
            f_out.write(line)
        elif line.startswith(("MODEL", "ENDMDL", "END", "TER")):
            f_out.write(line)
```

### 2. Nonsensical ΔΔG values (±50 kcal/mol)
**Problem:** Structure quality issues or force field problems

**Solution:**
1. Run `test_setup.py` to verify your setup
2. Use well-characterized structures (Barnase, Ubiquitin)
3. Increase minimization iterations: `max_iterations=5000`
4. Check if structure needs repair first

### 3. Low correlation with experimental data (R² < 0.3)
**This may be expected!** Computational prediction is challenging:
- State-of-the-art methods: R² ~ 0.4-0.7, MAE ~ 0.8-1.2 kcal/mol
- Force field limitations exist
- Experimental uncertainty: ~0.5-1.0 kcal/mol

## Extending the Demos

### Add Your Own Mutations
```python
mutations_df = pl.DataFrame({
    "mutation": ["F13A", "W14L"],  # Your mutations
    "chain": ["A", "A"],
    "experimental_ddg": [1.5, 2.0],  # If you have experimental data
})

results = engine.mutate_batch("your_protein.pdb", mutations_df)
```

### Use Your Own Structures
1. Download PDB file
2. Clean it (remove HETATM records)
3. Run mutation analysis
4. Compare to your experimental data

### Validate Against Larger Datasets
See `experimental_validation_demo.py` for framework to:
- Load ProTherm or SKEMPI datasets
- Run batch predictions
- Calculate comprehensive metrics
- Generate publication plots

## Citation

If you use Sicifus in your research, please cite:
[Citation information here]

## Support

- Documentation: https://arikat.github.io/sicifus
- Issues: https://github.com/arikat/sicifus/issues
- Examples: This directory

## Performance Notes

### Speed Tips
- Use `n_runs=1` for quick testing
- Use `constrain_backbone=True` for faster minimization
- Start with `max_iterations=1000`, increase if needed
- Use `keep_statistics=False` in batch mode if you don't need error bars

### Accuracy Tips
- Use `n_runs=5` or more for statistical rigor
- Increase `max_iterations` to 2000-5000 for better convergence
- Enable `keep_statistics=True` to assess prediction uncertainty
- Always validate against experimental data when available
