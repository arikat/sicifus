#!/usr/bin/env python3
"""
Quick test to verify Sicifus mutation analysis setup.

This script:
1. Downloads and cleans Barnase (1BNI)
2. Runs a single mutation with known experimental value
3. Compares to literature data to verify setup is working

Expected result:
- H18K experimental ΔΔG: +1.19 kcal/mol (Serrano et al. 1993, Table 3)
- Typical prediction error: 0.5-1.5 kcal/mol
- If you get ±50 kcal/mol, something is wrong!

Usage:
    python test_setup.py
"""

import urllib.request
from pathlib import Path


def download_and_clean_barnase():
    """Download and clean Barnase structure."""
    if not Path("1BNI.pdb").exists():
        print("Downloading Barnase structure (1BNI) from RCSB...")
        urllib.request.urlretrieve("https://files.rcsb.org/download/1BNI.pdb", "1BNI_raw.pdb")

        print("Cleaning structure (removing waters and heteroatoms)...")
        with open("1BNI_raw.pdb", "r") as f_in, open("1BNI.pdb", "w") as f_out:
            for line in f_in:
                if line.startswith("ATOM"):
                    f_out.write(line)
                elif line.startswith(("MODEL", "ENDMDL", "END", "TER")):
                    f_out.write(line)

        # Verify cleaning
        with open("1BNI.pdb", "r") as f:
            lines = f.readlines()
            hetatm_count = sum(1 for l in lines if l.startswith("HETATM"))
            if hetatm_count > 0:
                print(f"⚠️  WARNING: Still has {hetatm_count} HETATM records!")
            else:
                print("✅ Cleaned 1BNI.pdb (no waters/heteroatoms)")
    else:
        print("Using existing 1BNI.pdb")


def test_mutation():
    """Run test mutation and compare to experimental value."""
    print("\n" + "=" * 60)
    print("Testing mutation analysis setup")
    print("=" * 60)

    try:
        from sicifus import MutationEngine
    except ImportError:
        print("\n❌ ERROR: sicifus not installed")
        print("   Install with: pip install sicifus[energy]")
        return False

    print("\nRunning H18K mutation (this will take ~30 seconds)...")
    engine = MutationEngine()

    try:
        result = engine.mutate(
            "1BNI.pdb",
            ["H18K"],
            n_runs=1,
            max_iterations=1000
        )

        wt_energy = result.wt_energy
        mut_energy = result.mutant_energies["H18K"]
        ddg = result.ddg["H18K"]

        print(f"\nResults:")
        print(f"  WT energy:    {wt_energy:.2f} kcal/mol")
        print(f"  Mutant energy: {mut_energy:.2f} kcal/mol")
        print(f"  ΔΔG:          {ddg:+.2f} kcal/mol")

        # Compare to experimental (Serrano et al. 1993, Table 3)
        experimental_ddg = 1.19
        error = ddg - experimental_ddg

        print(f"\nComparison to Literature:")
        print(f"  Experimental:  +{experimental_ddg:.2f} kcal/mol (Serrano et al. 1993)")
        print(f"  Predicted:     {ddg:+.2f} kcal/mol")
        print(f"  Error:         {error:+.2f} kcal/mol")
        print(f"  Abs. error:    {abs(error):.2f} kcal/mol")

        # Sanity checks
        print("\n" + "=" * 60)
        if abs(ddg) > 10.0:
            print("❌ FAIL: ΔΔG is unreasonably large (>10 kcal/mol)")
            print("   Something is wrong with the structure or force field")
            return False
        elif abs(error) > 3.0:
            print("⚠️  WARNING: Large error vs experiment (>3 kcal/mol)")
            print("   This may indicate a setup issue, but could also be")
            print("   due to force field limitations")
            return True
        else:
            print("✅ SUCCESS: Results look reasonable!")
            print("   Error is within expected range for OpenMM force field")
            return True

    except Exception as e:
        print(f"\n❌ ERROR: Mutation failed with: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Sicifus Mutation Analysis Setup Test")
    print("=" * 60)

    # Download and clean
    download_and_clean_barnase()

    # Test mutation
    success = test_mutation()

    print("\n" + "=" * 60)
    if success:
        print("✅ Setup test PASSED - ready to use Sicifus!")
        print("\nNext steps:")
        print("  - Explore mutation_analysis_demo.ipynb")
        print("  - Check experimental_validation_demo.ipynb")
    else:
        print("❌ Setup test FAILED - check errors above")
        print("\nTroubleshooting:")
        print("  1. Ensure OpenMM is installed: pip install sicifus[energy]")
        print("  2. Check that 1BNI.pdb has no HETATM records")
        print("  3. Try with a different structure")
    print("=" * 60)
