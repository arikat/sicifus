"""
Flexible atom-based alignment using Kabsch algorithm with PyMOL-like selection syntax.

This module provides tools for aligning structures based on arbitrary atom selections,
useful for:
- Aligning ligands for docking analysis
- Transition state overlay in computational chemistry
- Binding site comparison
- Custom structural superposition

Supports PyMOL-like selection syntax:
    "chain A and resi 50-60"
    "resn ATP and name C1,C2,N1"
    "chain A,B and name CA"
    "name CA,CB,C and resi 10-20"
"""

import re
import numpy as np
import polars as pl
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from .align import _superimpose_numba


@dataclass
class AlignmentResult:
    """
    Result of atom-based alignment.

    Attributes:
        rmsd: Root mean squared deviation after alignment (Å)
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3D translation vector
        mobile_transformed: Transformed coordinates of mobile selection
        n_atoms: Number of atoms used for alignment
        selection: Selection string used
    """
    rmsd: float
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    mobile_transformed: np.ndarray
    n_atoms: int
    selection: str


class SelectionParser:
    """
    Parser for PyMOL-like atom selection syntax.

    Supported syntax:
        - chain A, B, C              (chain identifier)
        - resi 10-20                 (residue number range)
        - resi 10,15,20              (specific residues)
        - resn LIG,ATP               (residue name)
        - name CA,CB,C               (atom name)
        - element C,N,O              (element symbol)

    Combine with 'and':
        - chain A and resi 50-60
        - resn ATP and name C1,C2,N1
        - chain A,B and name CA
    """

    def __init__(self):
        pass

    def parse(self, selection: str, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse selection string and filter dataframe.

        Args:
            selection: Selection string (PyMOL-like syntax)
            df: DataFrame with atom data

        Returns:
            Filtered DataFrame

        Examples:
            >>> parser = SelectionParser()
            >>> selected = parser.parse("chain A and resi 50-60", df)
            >>> selected = parser.parse("resn ATP and name C1,C2,N1", df)
        """
        if not selection or selection.strip() == "all":
            return df

        # Split by 'and' to get individual conditions
        conditions = [c.strip() for c in selection.lower().split(" and ")]

        # Start with full dataframe
        filtered = df

        for condition in conditions:
            filtered = self._apply_condition(condition, filtered)

        if len(filtered) == 0:
            raise ValueError(f"Selection '{selection}' matched 0 atoms")

        return filtered

    def _apply_condition(self, condition: str, df: pl.DataFrame) -> pl.DataFrame:
        """Apply a single selection condition."""
        # Parse: "keyword value1,value2,..."
        parts = condition.split(None, 1)  # Split on first whitespace
        if len(parts) != 2:
            raise ValueError(f"Invalid selection syntax: '{condition}'")

        keyword, values = parts

        if keyword == "chain":
            return self._select_chains(values, df)
        elif keyword == "resi" or keyword == "resid":
            return self._select_residues(values, df)
        elif keyword == "resn" or keyword == "resname":
            return self._select_resnames(values, df)
        elif keyword == "name":
            return self._select_atom_names(values, df)
        elif keyword == "element" or keyword == "elem":
            return self._select_elements(values, df)
        else:
            raise ValueError(f"Unknown selection keyword: '{keyword}'. "
                           f"Supported: chain, resi, resn, name, element")

    def _select_chains(self, values: str, df: pl.DataFrame) -> pl.DataFrame:
        """Select by chain identifier: chain A,B,C"""
        chains = [v.strip().upper() for v in values.split(",")]
        return df.filter(pl.col("chain").is_in(chains))

    def _select_residues(self, values: str, df: pl.DataFrame) -> pl.DataFrame:
        """Select by residue number: resi 10-20 or resi 10,15,20"""
        residues = set()

        for val in values.split(","):
            val = val.strip()
            if "-" in val:
                # Range: 10-20
                start, end = val.split("-")
                residues.update(range(int(start), int(end) + 1))
            else:
                # Single value: 10
                residues.add(int(val))

        return df.filter(pl.col("residue_number").is_in(list(residues)))

    def _select_resnames(self, values: str, df: pl.DataFrame) -> pl.DataFrame:
        """Select by residue name: resn LIG,ATP"""
        resnames = [v.strip().upper() for v in values.split(",")]
        return df.filter(pl.col("residue_name").is_in(resnames))

    def _select_atom_names(self, values: str, df: pl.DataFrame) -> pl.DataFrame:
        """Select by atom name: name CA,CB,C"""
        names = [v.strip().upper() for v in values.split(",")]
        return df.filter(pl.col("atom_name").is_in(names))

    def _select_elements(self, values: str, df: pl.DataFrame) -> pl.DataFrame:
        """Select by element: element C,N,O"""
        elements = [v.strip().upper() for v in values.split(",")]
        return df.filter(pl.col("element").is_in(elements))


class AtomAligner:
    """
    Flexible atom-based structural alignment using Kabsch algorithm.

    This class provides methods for aligning structures based on arbitrary
    atom selections, going beyond traditional CA-only alignment.

    Use cases:
        - Align ligands for docking comparison
        - Overlay transition states in QM/MM
        - Compare binding sites
        - Custom structural superposition
    """

    def __init__(self):
        self.parser = SelectionParser()

    def align(
        self,
        mobile: pl.DataFrame,
        target: pl.DataFrame,
        selection: str = "name CA",
        apply_to_mobile: Optional[pl.DataFrame] = None,
    ) -> Tuple[AlignmentResult, Optional[pl.DataFrame]]:
        """
        Align mobile structure onto target using specified atom selection.

        Args:
            mobile: Mobile structure DataFrame (will be transformed)
            target: Target/reference structure DataFrame (fixed)
            selection: Atom selection string (PyMOL-like syntax)
            apply_to_mobile: Optional DataFrame to apply transformation to
                            (e.g., all atoms when aligning on subset)

        Returns:
            Tuple of (AlignmentResult, transformed_mobile_df or None)

        Examples:
            # Align on CA atoms
            >>> aligner = AtomAligner()
            >>> result, _ = aligner.align(mobile, target, "name CA")
            >>> print(f"RMSD: {result.rmsd:.2f} Å")

            # Align on ligand atoms and apply to entire structure
            >>> result, transformed = aligner.align(
            ...     mobile, target,
            ...     selection="resn ATP and name C1,C2,N1",
            ...     apply_to_mobile=mobile
            ... )

            # Align on binding site
            >>> result, _ = aligner.align(
            ...     mobile, target,
            ...     "chain A and resi 50-60 and name CA,CB"
            ... )
        """
        # Select atoms for alignment
        mobile_selected = self.parser.parse(selection, mobile)
        target_selected = self.parser.parse(selection, target)

        # Verify same number of atoms
        n_mobile = len(mobile_selected)
        n_target = len(target_selected)

        if n_mobile != n_target:
            raise ValueError(
                f"Selection must match same number of atoms. "
                f"Mobile: {n_mobile}, Target: {n_target}\n"
                f"Tip: Ensure both structures have identical atom topology for selection."
            )

        if n_mobile < 3:
            raise ValueError(
                f"Need at least 3 atoms for alignment, got {n_mobile}. "
                f"Selection: '{selection}'"
            )

        # Extract coordinates
        coords_mobile = mobile_selected.select(["x", "y", "z"]).to_numpy()
        coords_target = target_selected.select(["x", "y", "z"]).to_numpy()

        # Perform Kabsch alignment
        rmsd, R, t, coords_mobile_transformed = _superimpose_numba(
            coords_mobile, coords_target
        )

        result = AlignmentResult(
            rmsd=rmsd,
            rotation_matrix=R,
            translation_vector=t,
            mobile_transformed=coords_mobile_transformed,
            n_atoms=n_mobile,
            selection=selection,
        )

        # Apply transformation to entire mobile structure if requested
        transformed_df = None
        if apply_to_mobile is not None:
            transformed_df = self._apply_transformation(apply_to_mobile, R, t)

        return result, transformed_df

    def _apply_transformation(
        self,
        df: pl.DataFrame,
        R: np.ndarray,
        t: np.ndarray
    ) -> pl.DataFrame:
        """
        Apply rotation R and translation t to all atoms in DataFrame.

        Args:
            df: DataFrame with x, y, z columns
            R: 3x3 rotation matrix
            t: 3D translation vector

        Returns:
            DataFrame with transformed coordinates
        """
        # Extract coordinates
        coords = df.select(["x", "y", "z"]).to_numpy()

        # Apply transformation: coords_new = (R @ coords.T).T + t
        # Use contiguous R.T for performance
        Rt = np.ascontiguousarray(R.T)
        coords_transformed = np.dot(coords, Rt) + t

        # Update DataFrame with new coordinates
        df_transformed = df.with_columns([
            pl.Series("x", coords_transformed[:, 0]),
            pl.Series("y", coords_transformed[:, 1]),
            pl.Series("z", coords_transformed[:, 2]),
        ])

        return df_transformed

    def align_multiple(
        self,
        structures: Dict[str, pl.DataFrame],
        reference_id: str,
        selection: str = "name CA",
        apply_to_all: bool = False,
    ) -> Dict[str, Tuple[AlignmentResult, pl.DataFrame]]:
        """
        Align multiple structures onto a reference structure.

        Args:
            structures: Dictionary of structure_id -> DataFrame
            reference_id: ID of reference structure (fixed)
            selection: Atom selection for alignment
            apply_to_all: If True, apply transformation to all atoms

        Returns:
            Dictionary of structure_id -> (AlignmentResult, transformed_df)

        Example:
            >>> structures = {"1ABC": df1, "2DEF": df2, "3GHI": df3}
            >>> results = aligner.align_multiple(
            ...     structures,
            ...     reference_id="1ABC",
            ...     selection="chain A and name CA"
            ... )
            >>> for sid, (result, transformed) in results.items():
            ...     print(f"{sid}: RMSD = {result.rmsd:.2f} Å")
        """
        if reference_id not in structures:
            raise ValueError(f"Reference '{reference_id}' not in structures")

        reference = structures[reference_id]
        results = {}

        for struct_id, mobile in structures.items():
            if struct_id == reference_id:
                # Reference aligns to itself with RMSD=0
                coords_ref = self.parser.parse(selection, reference).select(["x", "y", "z"]).to_numpy()
                result = AlignmentResult(
                    rmsd=0.0,
                    rotation_matrix=np.eye(3),
                    translation_vector=np.zeros(3),
                    mobile_transformed=coords_ref,
                    n_atoms=len(coords_ref),
                    selection=selection,
                )
                results[struct_id] = (result, reference.clone() if apply_to_all else reference)
            else:
                apply_df = mobile if apply_to_all else None
                result, transformed = self.align(mobile, reference, selection, apply_df)
                results[struct_id] = (result, transformed if transformed is not None else mobile)

        return results

    def compute_pairwise_rmsd(
        self,
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        selection: str = "name CA",
        align: bool = True,
    ) -> float:
        """
        Compute RMSD between two structures.

        Args:
            df1: First structure
            df2: Second structure
            selection: Atom selection
            align: If True, align before computing RMSD (optimal superposition)
                  If False, compute RMSD without alignment (positional RMSD)

        Returns:
            RMSD in Ångströms

        Example:
            >>> # Aligned RMSD (after optimal superposition)
            >>> rmsd_aligned = aligner.compute_pairwise_rmsd(df1, df2, "name CA", align=True)

            >>> # Positional RMSD (no alignment)
            >>> rmsd_positional = aligner.compute_pairwise_rmsd(df1, df2, "name CA", align=False)
        """
        sel1 = self.parser.parse(selection, df1)
        sel2 = self.parser.parse(selection, df2)

        if len(sel1) != len(sel2):
            raise ValueError(
                f"Selection must match same number of atoms. "
                f"Structure 1: {len(sel1)}, Structure 2: {len(sel2)}"
            )

        coords1 = sel1.select(["x", "y", "z"]).to_numpy()
        coords2 = sel2.select(["x", "y", "z"]).to_numpy()

        if align:
            # Perform alignment and return RMSD
            rmsd, _, _, _ = _superimpose_numba(coords1, coords2)
            return rmsd
        else:
            # Direct RMSD without alignment
            diff = coords1 - coords2
            rmsd = np.sqrt(np.sum(diff**2) / len(coords1))
            return rmsd


def write_pdb(df: pl.DataFrame, filename: str):
    """
    Write DataFrame to PDB file.

    Args:
        df: DataFrame with atom data
        filename: Output PDB file path
    """
    with open(filename, 'w') as f:
        for i, row in enumerate(df.iter_rows(named=True), start=1):
            atom_name = row['atom_name']
            # Format atom name (right-align if <= 3 chars, left if 4)
            if len(atom_name) <= 3:
                atom_name_str = f" {atom_name:<3s}"
            else:
                atom_name_str = f"{atom_name:4s}"

            line = (
                f"ATOM  {i:5d} {atom_name_str} "
                f"{row['residue_name']:3s} {row['chain']:1s}"
                f"{row['residue_number']:4d}    "
                f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}"
                f"  1.00  0.00          "
                f"{row['element']:>2s}\n"
            )
            f.write(line)
        f.write("END\n")
