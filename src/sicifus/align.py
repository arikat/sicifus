import numpy as np
import polars as pl
from typing import Tuple, List, Optional
from numba import jit


# ---------------------------------------------------------------------------
# 20-state 3Di-like structural alphabet
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _encode_3di_numba(coords):
    """Encode a CA trace into a 20-state structural alphabet.

    For each residue *i* (needs positions i-2 … i+1), computes:

    * **theta** – virtual bond angle CA(i-1)–CA(i)–CA(i+1)
    * **tau**   – pseudo-dihedral CA(i-2)–CA(i-1)–CA(i)–CA(i+1)

    (theta, tau) is discretised into a 4×5 = 20-bin grid (states 0–19).

    Theta bins: <75° | 75–100° (helices ~88°) | 100–125° | >=125° (extended)
    Tau bins:   <-120° | -120 to -40° | -40 to 30° | 30–110° (helix ~50°) | >=110°

    Boundary residues (first 2, last 1) are assigned state 0.

    Parameters
    ----------
    coords : ndarray, shape (N, 3), float64
        CA atom positions.

    Returns
    -------
    ndarray, shape (N,), int8
        Per-residue state labels in [0, 19].
    """
    n = coords.shape[0]
    states = np.zeros(n, dtype=np.int8)

    if n < 4:
        return states

    theta_edges = np.array([75.0, 100.0, 125.0])
    tau_edges = np.array([-120.0, -40.0, 30.0, 110.0])

    for i in range(2, n - 1):
        # --- virtual bond angle at CA(i) ---
        v1x = coords[i - 1, 0] - coords[i, 0]
        v1y = coords[i - 1, 1] - coords[i, 1]
        v1z = coords[i - 1, 2] - coords[i, 2]

        v2x = coords[i + 1, 0] - coords[i, 0]
        v2y = coords[i + 1, 1] - coords[i, 1]
        v2z = coords[i + 1, 2] - coords[i, 2]

        norm1 = np.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        norm2 = np.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
        if norm1 < 1e-8 or norm2 < 1e-8:
            continue

        cos_theta = (v1x * v2x + v1y * v2y + v1z * v2z) / (norm1 * norm2)
        if cos_theta > 1.0:
            cos_theta = 1.0
        elif cos_theta < -1.0:
            cos_theta = -1.0
        theta = np.arccos(cos_theta) * 180.0 / np.pi

        # --- pseudo-dihedral CA(i-2)–CA(i-1)–CA(i)–CA(i+1) ---
        b1x = coords[i - 1, 0] - coords[i - 2, 0]
        b1y = coords[i - 1, 1] - coords[i - 2, 1]
        b1z = coords[i - 1, 2] - coords[i - 2, 2]

        b2x = coords[i, 0] - coords[i - 1, 0]
        b2y = coords[i, 1] - coords[i - 1, 1]
        b2z = coords[i, 2] - coords[i - 1, 2]

        b3x = coords[i + 1, 0] - coords[i, 0]
        b3y = coords[i + 1, 1] - coords[i, 1]
        b3z = coords[i + 1, 2] - coords[i, 2]

        # cross-products n1 = b1 × b2,  n2 = b2 × b3
        n1x = b1y * b2z - b1z * b2y
        n1y = b1z * b2x - b1x * b2z
        n1z = b1x * b2y - b1y * b2x

        n2x = b2y * b3z - b2z * b3y
        n2y = b2z * b3x - b2x * b3z
        n2z = b2x * b3y - b2y * b3x

        norm_n1 = np.sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
        norm_n2 = np.sqrt(n2x * n2x + n2y * n2y + n2z * n2z)
        if norm_n1 < 1e-8 or norm_n2 < 1e-8:
            continue

        cos_tau = (n1x * n2x + n1y * n2y + n1z * n2z) / (norm_n1 * norm_n2)
        if cos_tau > 1.0:
            cos_tau = 1.0
        elif cos_tau < -1.0:
            cos_tau = -1.0

        sign_val = n1x * b3x + n1y * b3y + n1z * b3z
        tau = np.arccos(cos_tau) * 180.0 / np.pi
        if sign_val < 0.0:
            tau = -tau

        # --- bin theta (4 bins) ---
        theta_bin = 0
        for t in range(3):
            if theta >= theta_edges[t]:
                theta_bin = t + 1

        # --- bin tau (5 bins) ---
        tau_bin = 0
        for t in range(4):
            if tau >= tau_edges[t]:
                tau_bin = t + 1

        states[i] = theta_bin * 5 + tau_bin

    return states


# ---------------------------------------------------------------------------
# Kabsch (unchanged)
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _superimpose_numba(coords_mobile: np.ndarray, coords_target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled Kabsch algorithm.
    """
    n = coords_mobile.shape[0]
    centroid_mobile = np.sum(coords_mobile, axis=0) / n
    centroid_target = np.sum(coords_target, axis=0) / n
    
    # Center coordinates
    p = coords_mobile - centroid_mobile
    q = coords_target - centroid_target
    
    # Computation of the covariance matrix
    H = np.dot(p.T, q)
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # Translation: t = centroid_target - R @ centroid_mobile
    t = centroid_target - np.dot(R, centroid_mobile)
    
    # Apply transformation — use contiguous copy of R.T to avoid performance warning
    Rt = np.ascontiguousarray(R.T)
    coords_mobile_transformed = np.dot(coords_mobile, Rt) + t
    
    # RMSD
    diff = coords_mobile_transformed - coords_target
    rmsd = np.sqrt(np.sum(diff**2) / n)
    
    return rmsd, R, t, coords_mobile_transformed

@jit(nopython=True, cache=True)
def _align_sequences_numba(seq1_arr: np.ndarray, seq2_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled Needleman-Wunsch alignment.
    Inputs are integer arrays (converted from string).
    """
    n = len(seq1_arr)
    m = len(seq2_arr)
    score_matrix = np.zeros((n + 1, m + 1), dtype=np.int32)
    
    # Simple scoring
    match = 1
    mismatch = -1
    gap = -1
    
    for i in range(1, n + 1):
        score_matrix[i, 0] = i * gap
    for j in range(1, m + 1):
        score_matrix[0, j] = j * gap
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1_arr[i-1] == seq2_arr[j-1] else mismatch
            
            # Calculate scores
            diag = score_matrix[i-1, j-1] + s
            up = score_matrix[i-1, j] + gap
            left = score_matrix[i, j-1] + gap
            
            # Max
            if diag >= up and diag >= left:
                score_matrix[i, j] = diag
            elif up >= left:
                score_matrix[i, j] = up
            else:
                score_matrix[i, j] = left
                
    # Traceback
    # Pre-allocate max possible size
    max_len = n + m
    align1 = np.empty(max_len, dtype=np.int32)
    align2 = np.empty(max_len, dtype=np.int32)
    
    idx = 0
    i, j = n, m
    while i > 0 and j > 0:
        s = match if seq1_arr[i-1] == seq2_arr[j-1] else mismatch
        current_score = score_matrix[i, j]
        
        if current_score == score_matrix[i-1, j-1] + s:
            align1[idx] = i-1
            align2[idx] = j-1
            i -= 1
            j -= 1
            idx += 1
        elif current_score == score_matrix[i-1, j] + gap:
            i -= 1
        else:
            j -= 1
            
    # Trim and reverse
    return align1[:idx][::-1], align2[:idx][::-1]

class StructuralAligner:
    """
    Handles structural alignment and RMSD calculation.
    """
    
    def __init__(self):
        pass

    def get_ca_coords(self, df: pl.DataFrame) -> np.ndarray:
        """Extracts CA coordinates from a dataframe as a (N, 3) numpy array."""
        return df.select(["x", "y", "z"]).to_numpy()

    def superimpose(self, coords_mobile: np.ndarray, coords_target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Superimposes coords_mobile onto coords_target using Kabsch algorithm.
        Assumes 1-to-1 correspondence (same length).
        """
        if coords_mobile.shape != coords_target.shape:
            raise ValueError("Coordinates must have same shape for superposition.")
        return _superimpose_numba(coords_mobile, coords_target)

    def encode_3di(self, coords: np.ndarray) -> np.ndarray:
        """Encode CA coords into a 20-state structural alphabet (int8 array).

        This is the richer alphabet used for k-mer prefiltering.  For the
        older 4-letter alphabet used by Needleman-Wunsch alignment, see
        :meth:`encode_structure`.
        """
        return _encode_3di_numba(np.ascontiguousarray(coords, dtype=np.float64))

    def encode_structure(self, coords: np.ndarray) -> str:
        """
        Encodes a CA trace into a structural string using local geometry (simplified).
        """
        if len(coords) < 4:
            return "X" * len(coords)
            
        # Calculate vectors between CAs
        v = coords[1:] - coords[:-1] # (N-1, 3)
        
        # Normalize
        norms = np.linalg.norm(v, axis=1)
        # Avoid division by zero
        norms[norms == 0] = 1e-8
        v_norm = v / norms[:, None]
        
        # Calculate angles (cos theta)
        # dot product of consecutive vectors
        cos_angles = np.sum(v_norm[:-1] * v_norm[1:], axis=1) # (N-2,)
        
        # Vectorized quantization using np.digitize
        # Bins for: D (< -0.5), C (-0.5..0), B (0..0.5), A (> 0.5)
        bins = np.array([-0.5, 0.0, 0.5])
        inds = np.digitize(cos_angles, bins)
        
        # Map indices to characters: 0->D, 1->C, 2->B, 3->A
        mapping = np.array(['D', 'C', 'B', 'A'])
        chars = mapping[inds]
        
        return "X" + "".join(chars) + "X"

    def align_sequences(self, seq1: str, seq2: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs global alignment (Needleman-Wunsch) on two structural strings.
        Returns indices of aligned residues (0-based).
        """
        s1_arr = np.array([ord(c) for c in seq1], dtype=np.int32)
        s2_arr = np.array([ord(c) for c in seq2], dtype=np.int32)
        
        return _align_sequences_numba(s1_arr, s2_arr)

    def align_and_superimpose(self, df_mobile: pl.DataFrame, df_target: pl.DataFrame) -> Tuple[float, int]:
        """
        Aligns two structures using structural alphabet and calculates RMSD.
        Returns (rmsd, num_aligned_residues).
        """
        coords_mobile = self.get_ca_coords(df_mobile)
        coords_target = self.get_ca_coords(df_target)
        
        seq_mobile = self.encode_structure(coords_mobile)
        seq_target = self.encode_structure(coords_target)
        
        return self._align_and_superimpose_from_data(coords_mobile, seq_mobile, coords_target, seq_target)

    def _align_and_superimpose_from_data(self, coords_mobile: np.ndarray, seq_mobile: str, 
                                         coords_target: np.ndarray, seq_target: str) -> Tuple[float, int]:
        """
        Internal method to align using pre-computed coords and sequences.
        """
        idx_mobile, idx_target = self.align_sequences(seq_mobile, seq_target)
        
        if len(idx_mobile) < 3:
            return float('inf'), 0
            
        aligned_mobile = coords_mobile[idx_mobile]
        aligned_target = coords_target[idx_target]
        
        rmsd, _, _, _ = self.superimpose(aligned_mobile, aligned_target)
        return rmsd, len(idx_mobile)

    def align_and_transform(self, df_mobile: pl.DataFrame, df_target: pl.DataFrame) -> Tuple[pl.DataFrame, float]:
        """
        Aligns mobile to target and returns the transformed mobile dataframe.
        """
        coords_mobile = self.get_ca_coords(df_mobile)
        coords_target = self.get_ca_coords(df_target)
        
        seq_mobile = self.encode_structure(coords_mobile)
        seq_target = self.encode_structure(coords_target)
        
        idx_mobile, idx_target = self.align_sequences(seq_mobile, seq_target)
        
        if len(idx_mobile) < 3:
            return df_mobile, float('inf')
            
        aligned_mobile = coords_mobile[idx_mobile]
        aligned_target = coords_target[idx_target]
        
        rmsd, R, t, _ = self.superimpose(aligned_mobile, aligned_target)
        
        # Apply transformation to ALL atoms in df_mobile
        # We need to extract all coords, transform, and put back
        all_coords = df_mobile.select(["x", "y", "z"]).to_numpy()
        transformed_all = np.dot(all_coords, R.T) + t
        
        # Update dataframe
        df_transformed = df_mobile.with_columns([
            pl.Series("x", transformed_all[:, 0]),
            pl.Series("y", transformed_all[:, 1]),
            pl.Series("z", transformed_all[:, 2])
        ])
        
        return df_transformed, rmsd
