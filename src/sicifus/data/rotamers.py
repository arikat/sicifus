"""Side-chain χ-angle definitions for the empirical scorer's rotamer repack.

This is a *coarse, backbone-independent* rotamer scheme.  For each rotatable
side chain we list the heavy-atom quadruples that define each χ dihedral; the
rotamer library itself is the canonical staggered set ``{-60°, 180°, +60°}``
(``g⁻ / t / g⁺``) enumerated over every χ.  Combined with the residue's *native*
(as-built) conformation — which the repacker always keeps as a candidate — this
relieves the arbitrary, occasionally clashing rotamer that PDBFixer places when
it rebuilds a mutated side chain, without a full Dunbrack search.

The set of atoms that move with each χ is *not* tabulated here: the repacker
derives it from the residue's own bond graph (BFS across the χ axis), which
places hydrogens correctly without per-residue moving-set tables.

A full backbone-dependent Dunbrack library can replace ``ROTAMER_CHI`` later
without touching the repack algorithm.

References:
- Lovell et al. (2000) Proteins 40:389 — the penultimate rotamer library
  (staggered χ wells this coarse set approximates).
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

# χ dihedral definitions (heavy-atom names), one quadruple per χ, in order.
CHI_DEFS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "VAL": [("N", "CA", "CB", "CG1")],
}

# Canonical staggered χ wells (degrees).
_WELLS = (-60.0, 180.0, 60.0)


def rotamers_for(resn: str) -> List[Tuple[float, ...]]:
    """Library χ-tuples (degrees) for a residue, or ``[]`` if not rotatable.

    The native conformation is added separately by the repacker; this returns
    only the enumerated staggered library.
    """
    defs = CHI_DEFS.get(resn)
    if not defs:
        return []
    return [tuple(combo) for combo in product(_WELLS, repeat=len(defs))]


# Backbone atom names that never move during a repack.
BACKBONE_ATOMS = frozenset({
    "N", "CA", "C", "O", "OXT",
    "H", "H1", "H2", "H3", "HA", "HA2", "HA3", "HN", "HT1", "HT2", "HT3",
})
