#!/usr/bin/env python3
"""
Show what residues are actually present in a PDB structure.

This helps identify the correct residue numbers for mutations, especially
when structures have missing residues or non-sequential numbering.

Usage:
    python show_residues.py 1BNI.pdb
    python show_residues.py 1BNI.pdb --chain A
"""

import sys
import argparse
from pathlib import Path


def show_residues(pdb_file, chain=None, show_all=False):
    """Show residues in a PDB file."""

    if not Path(pdb_file).exists():
        print(f"Error: File '{pdb_file}' not found")
        return

    residues = {}

    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue

            chain_id = line[21:22].strip()
            if chain and chain_id != chain:
                continue

            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()

            if chain_id not in residues:
                residues[chain_id] = {}

            if res_num not in residues[chain_id]:
                residues[chain_id][res_num] = res_name

    # Display
    for chain_id in sorted(residues.keys()):
        if chain and chain_id != chain:
            continue

        print(f"\nChain {chain_id}:")
        print(f"{'Num':<6} {'Res':<4} {'1-letter':<10} {'For mutation string'}")
        print("-" * 50)

        res_list = sorted(residues[chain_id].items())

        # Check for gaps
        start_num = res_list[0][0]
        end_num = res_list[-1][0]
        missing = []
        for i in range(start_num, end_num + 1):
            if i not in residues[chain_id]:
                missing.append(i)

        if missing:
            print(f"⚠️  WARNING: Missing residues: {missing}")
            print(f"   Structure starts at residue {start_num}, not 1!")
            print()

        # Show residues
        count = 0
        for res_num, res_name in res_list:
            if not show_all and count >= 20 and count < len(res_list) - 10:
                if count == 20:
                    print(f"  ... ({len(res_list) - 30} more residues) ...")
                count += 1
                continue

            one_letter = THREE_TO_ONE.get(res_name, '?')
            mutation_str = f"{one_letter}{res_num}A"
            print(f"{res_num:<6} {res_name:<4} {one_letter:<10} {mutation_str}")
            count += 1

        print(f"\nTotal: {len(res_list)} residues (numbers {start_num}-{end_num})")

        if missing:
            print(f"\n💡 TIP: Use the residue numbers shown above for mutations.")
            print(f"   Example: To mutate GLU at position {res_list[10][0]}, use 'E{res_list[10][0]}Q'")


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show residues present in a PDB file"
    )
    parser.add_argument("pdb_file", help="PDB file to analyze")
    parser.add_argument("--chain", help="Show only this chain")
    parser.add_argument("--all", action="store_true",
                       help="Show all residues (not just first/last 20)")

    args = parser.parse_args()

    show_residues(args.pdb_file, chain=args.chain, show_all=args.all)
