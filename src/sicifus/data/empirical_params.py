"""Parameter tables for the empirical (FoldX-style) ΔΔG scorer.

IMPORTANT — these parameters are *approximations* assembled from published
sources (Bondi van der Waals radii; Eisenberg–McLachlan atomic solvation
parameters; Pickett–Sternberg / Abagyan side-chain entropies; FoldX-style term
weights).  They are **not calibrated** against an experimental ΔΔG training set.
The scorer that consumes them (:mod:`sicifus.empirical`) is therefore a
physically-motivated baseline, not a drop-in FoldX replacement.  Calibrating the
weights/parameters against ProTherm/SKEMPI is a documented follow-up.

References:
- Bondi, A. (1964) J. Phys. Chem. 68:441 — van der Waals radii.
- Eisenberg & McLachlan (1986) Nature 319:199 — atomic solvation parameters.
- Guerois, Nielsen & Serrano (2002) JMB 320:369 — FoldX energy terms/weights.
- Pickett & Sternberg (1993) JMB 231:825 — side-chain conformational entropy.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Term weights (FoldX-style linear combination).
#   ΔG = Wvdw·vdw + solvH + solvP + hbond + elec + Wmc·T·Smc + Wsc·T·Ssc + clash
# vdW carries the canonical 0.33 weight (vapour→water reference vs solvent→
# protein); the entropy/solvation/hbond/electrostatic weights are 1.0 by
# construction and absorbed into their parameter magnitudes.
# ---------------------------------------------------------------------------
# Physical-prior weights (uncalibrated). ``CALIBRATED_WEIGHTS`` (loaded from
# ``empirical_weights.json``) overrides these at runtime when available.
PHYSICAL_WEIGHTS = {
    "vdw": 0.33,
    "clash": 1.0,
    "solvH": 1.0,
    "solvP": 1.0,
    "hbond": 1.0,
    "elec": 1.0,
    "mc_entropy": 1.0,
    "sc_entropy": 1.0,
}


def _load_calibrated_weights() -> dict | None:
    """Calibrated term weights from ``empirical_weights.json``, or None."""
    import json
    from pathlib import Path

    path = Path(__file__).with_name("empirical_weights.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        w = data.get("weights")
        return dict(w) if w else None
    except (ValueError, OSError):
        return None


CALIBRATED_WEIGHTS = _load_calibrated_weights()

# Default weights used by EmpiricalScorer: calibrated if present, else physical.
WEIGHTS = dict(CALIBRATED_WEIGHTS) if CALIBRATED_WEIGHTS else dict(PHYSICAL_WEIGHTS)

# Bondi van der Waals radii (Angstrom), by element symbol (upper-case).
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
}
DEFAULT_VDW_RADIUS = 1.70

# Eisenberg–McLachlan atomic solvation parameters (kcal/mol/Å²).
# Sign convention: burying an apolar atom (positive ASP) is favourable;
# burying a polar/charged atom (negative ASP) is unfavourable.  Applied to the
# *buried* surface area (max SASA − actual SASA).
ASP = {
    "C": 0.016,    # apolar carbon
    "N": -0.006,   # neutral polar nitrogen
    "O": -0.006,   # neutral polar oxygen
    "S": 0.021,    # sulfur (mildly apolar)
    "N+": -0.050,  # positively charged nitrogen (Lys NZ, Arg guanidinium)
    "O-": -0.024,  # negatively charged oxygen (Asp/Glu carboxylate)
}

# ---------------------------------------------------------------------------
# Partial charges (e) for notable atoms, keyed by (residue_name, atom_name).
# Backbone atoms handled generically in BACKBONE_CHARGES.  Anything not listed
# is treated as neutral (0.0).  These are coarse, integer-group charges meant
# for a screened-Coulomb term, not a full force-field charge set.
# ---------------------------------------------------------------------------
BACKBONE_CHARGES = {
    "N": -0.40, "H": 0.25, "CA": 0.10, "C": 0.50, "O": -0.50,
}
SIDECHAIN_CHARGES = {
    # Acidic — carboxylate (-1 split over two oxygens)
    ("ASP", "OD1"): -0.50, ("ASP", "OD2"): -0.50,
    ("GLU", "OE1"): -0.50, ("GLU", "OE2"): -0.50,
    # Basic — Lys ammonium (+1)
    ("LYS", "NZ"): 1.00,
    # Basic — Arg guanidinium (+1 split)
    ("ARG", "NE"): 0.33, ("ARG", "NH1"): 0.33, ("ARG", "NH2"): 0.34,
    # Histidine (treated near-neutral at pH 7)
    ("HIS", "ND1"): 0.10, ("HIS", "NE2"): 0.10,
}

# Atoms carrying a formal charge — used to pick the charged ASP variant.
POSITIVE_N = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
NEGATIVE_O = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}

# ---------------------------------------------------------------------------
# Hydrogen-bond roles.  "D" = donor heavy atom, "A" = acceptor, "B" = both.
# Backbone N is a donor, backbone O an acceptor (BACKBONE_HBOND).  Side-chain
# polar atoms listed by (residue_name, atom_name).
# ---------------------------------------------------------------------------
BACKBONE_HBOND = {"N": "D", "O": "A"}
SIDECHAIN_HBOND = {
    ("SER", "OG"): "B", ("THR", "OG1"): "B", ("TYR", "OH"): "B",
    ("ASN", "OD1"): "A", ("ASN", "ND2"): "D",
    ("GLN", "OE1"): "A", ("GLN", "NE2"): "D",
    ("ASP", "OD1"): "A", ("ASP", "OD2"): "A",
    ("GLU", "OE1"): "A", ("GLU", "OE2"): "A",
    ("LYS", "NZ"): "D", ("ARG", "NE"): "D",
    ("ARG", "NH1"): "D", ("ARG", "NH2"): "D",
    ("HIS", "ND1"): "B", ("HIS", "NE2"): "B",
    ("TRP", "NE1"): "D", ("CYS", "SG"): "A",
}

# Per-bond hydrogen-bond energy (kcal/mol, favourable = negative) and geometry.
HBOND_ENERGY = -1.5
HBOND_DIST_MIN = 2.5   # Å, donor–acceptor heavy-atom distance
HBOND_DIST_MAX = 3.5   # Å

# ---------------------------------------------------------------------------
# Side-chain conformational entropy cost of folding, T·ΔS at 298 K (kcal/mol).
# Larger = more entropy lost on burial/fixing (destabilising when folded).
# Pickett & Sternberg (1993) scale, rounded.
# ---------------------------------------------------------------------------
SIDECHAIN_ENTROPY = {
    "ALA": 0.0, "GLY": 0.0, "PRO": 0.0,
    "SER": 0.55, "CYS": 0.69, "THR": 0.65, "VAL": 0.51,
    "ASP": 0.69, "ASN": 0.82, "ILE": 0.69, "LEU": 0.69,
    "HIS": 0.99, "TRP": 0.99, "TYR": 1.02, "PHE": 0.76,
    "GLU": 1.36, "GLN": 1.46, "MET": 1.43, "LYS": 1.92, "ARG": 2.13,
}

# Backbone conformational entropy (relative, kcal/mol).  Glycine has the most
# accessible φ/ψ space (largest entropy cost to fix); proline the least.
# Coarse φ/ψ-propensity-derived scale.
BACKBONE_ENTROPY = {
    "GLY": 1.00, "PRO": 0.0,
    "ALA": 0.40, "SER": 0.45, "CYS": 0.45, "THR": 0.45, "VAL": 0.40,
    "ASP": 0.45, "ASN": 0.50, "ILE": 0.40, "LEU": 0.42,
    "HIS": 0.45, "TRP": 0.42, "TYR": 0.42, "PHE": 0.42,
    "GLU": 0.45, "GLN": 0.45, "MET": 0.42, "LYS": 0.45, "ARG": 0.45,
}

# van der Waals well depth (kcal/mol) for the attractive LJ term, by element.
VDW_WELL = {
    "H": 0.02, "C": 0.11, "N": 0.17, "O": 0.21, "S": 0.25,
}
DEFAULT_VDW_WELL = 0.11

# Clash penalty: quadratic in the overlap (sum-of-radii − distance) beyond a
# small tolerance, scaled by this constant (kcal/mol/Å²).
CLASH_TOLERANCE = 0.4   # Å of allowed overlap before penalising
CLASH_SCALE = 10.0      # kcal/mol/Å²


def element_of(atom_name: str, element: str | None) -> str:
    """Best-effort element symbol (upper-case) from a stored element or name."""
    if element:
        return element.strip().upper()
    # Fall back to the first alphabetic character of the atom name.
    for ch in atom_name:
        if ch.isalpha():
            return ch.upper()
    return "C"


def atom_charge(residue_name: str, atom_name: str) -> float:
    """Partial charge (e) for an atom; backbone first, then side chain."""
    key = (residue_name, atom_name)
    if key in SIDECHAIN_CHARGES:
        return SIDECHAIN_CHARGES[key]
    return BACKBONE_CHARGES.get(atom_name, 0.0)


def atom_asp(residue_name: str, atom_name: str, element: str) -> float:
    """Atomic solvation parameter (kcal/mol/Å²) accounting for formal charge."""
    key = (residue_name, atom_name)
    if key in POSITIVE_N:
        return ASP["N+"]
    if key in NEGATIVE_O:
        return ASP["O-"]
    return ASP.get(element, 0.0)


def atom_hbond_role(residue_name: str, atom_name: str) -> str | None:
    """'D', 'A', 'B', or None for an atom's hydrogen-bond role."""
    key = (residue_name, atom_name)
    if key in SIDECHAIN_HBOND:
        return SIDECHAIN_HBOND[key]
    return BACKBONE_HBOND.get(atom_name)
