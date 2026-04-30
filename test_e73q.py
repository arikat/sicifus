#!/usr/bin/env python
"""Quick test of E73Q mutation after fixing findMissingResidues issue."""

import sys
import os
import urllib.request

# Import just what we need, avoiding full package import
sys.path.insert(0, 'src')

# Import mutate module directly
import importlib.util
spec = importlib.util.spec_from_file_location("mutate", "src/sicifus/mutate.py")
mutate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mutate)

MutationEngine = mutate.MutationEngine

os.chdir('examples')

# Download 1BNI if not present
if not os.path.exists('1BNI.pdb'):
    print('Downloading 1BNI...')
    url = 'https://files.rcsb.org/download/1BNI.pdb'
    urllib.request.urlretrieve(url, '1BNI_raw.pdb')

    # Clean it
    with open('1BNI_raw.pdb', 'r') as f_in, open('1BNI.pdb', 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)
            elif line.startswith(('MODEL', 'ENDMDL', 'END', 'TER')):
                f_out.write(line)
    print('Downloaded and cleaned 1BNI.pdb')
else:
    print('1BNI.pdb already exists')

engine = MutationEngine()

# First, show what residues are available
print('\n=== Checking residues available ===')
residues = engine.show_residues('1BNI.pdb', chain='A')
print(residues.filter(residues['position'].is_in([71, 72, 73, 74, 75])))

# Now try the mutation
print('\n=== Attempting E73Q mutation ===')
try:
    result = engine.mutate('1BNI.pdb', ['E73Q'], n_runs=1, max_iterations=100)
    print(f'✅ SUCCESS! Mutation worked.')
    print(f'   WT energy: {result.wt_energy_kcal} kcal/mol')
    print(f'   Mutant energy: {result.mut_energy_kcal} kcal/mol')
    print(f'   ΔΔG: {result.ddg_kcal} kcal/mol')
except Exception as e:
    print(f'❌ FAILED: {e}')
    import traceback
    traceback.print_exc()
