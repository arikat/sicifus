"""Tests for the 3Di structural alphabet, k-mer index, and fast clustering.

Unit tests (no network / large data) run instantly and verify:
  - 3Di encoding produces correct state ranges and distinguishes geometries
  - K-mer index building and querying
  - prefilter_pairs identifies similar pairs
  - cluster_fast produces valid clusters
  - compute_rmsd_matrix with prefilter=True produces a valid matrix
"""

import math
import numpy as np
import polars as pl
import pytest

from sicifus.align import StructuralAligner, _encode_3di_numba
from sicifus.kmer_index import (
    ALPHABET_SIZE,
    _extract_kmer_hashes,
    build_kmer_index,
    prefilter_pairs,
)
from sicifus.analysis import AnalysisToolkit


# ---------------------------------------------------------------------------
# Synthetic geometry helpers
# ---------------------------------------------------------------------------


def _make_helix(n_residues: int = 30) -> np.ndarray:
    """Right-handed alpha-helix CA trace."""
    R, d, theta = 2.3, 1.5, np.radians(100.0)
    coords = np.zeros((n_residues, 3))
    for i in range(n_residues):
        coords[i] = [R * np.cos(i * theta), R * np.sin(i * theta), i * d]
    return coords


def _make_extended(n_residues: int = 30) -> np.ndarray:
    """Extended CA trace with beta-strand-like pleated geometry.

    Uses a zigzag in y with alternating z-displacement to avoid
    degenerate (fully planar) coordinates that collapse dihedrals to ±180°.
    """
    coords = np.zeros((n_residues, 3))
    for i in range(n_residues):
        coords[i] = [i * 3.5, (i % 2) * 0.8, (i % 3) * 0.3]
    return coords


def _make_random_coil(n_residues: int = 30, seed: int = 42) -> np.ndarray:
    """Random walk CA trace with ~3.8 Å steps."""
    rng = np.random.RandomState(seed)
    coords = np.zeros((n_residues, 3))
    for i in range(1, n_residues):
        step = rng.randn(3)
        step = step / np.linalg.norm(step) * 3.8
        coords[i] = coords[i - 1] + step
    return coords


def _coords_to_df(coords: np.ndarray, sid: str = "test") -> pl.DataFrame:
    """Wrap CA coords into a minimal DataFrame matching StructuralAligner expectations."""
    return pl.DataFrame({
        "structure_id": [sid] * len(coords),
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "z": coords[:, 2].tolist(),
    })


# ===================================================================
# 3Di structural alphabet
# ===================================================================


class TestEncode3Di:

    def test_returns_correct_shape(self):
        coords = _make_helix(20)
        states = _encode_3di_numba(coords)
        assert states.shape == (20,)

    def test_values_in_range(self):
        for factory in [_make_helix, _make_extended, _make_random_coil]:
            states = _encode_3di_numba(factory(40))
            assert states.min() >= 0
            assert states.max() <= 19

    def test_boundary_residues_are_zero(self):
        states = _encode_3di_numba(_make_helix(20))
        assert states[0] == 0
        assert states[1] == 0
        assert states[-1] == 0

    def test_helix_vs_extended_different(self):
        s_helix = _encode_3di_numba(_make_helix(30))
        s_ext = _encode_3di_numba(_make_extended(30))
        interior_helix = s_helix[2:-1]
        interior_ext = s_ext[2:-1]
        assert not np.array_equal(interior_helix, interior_ext)

    def test_identical_coords_identical_states(self):
        coords = _make_helix(25)
        s1 = _encode_3di_numba(coords)
        s2 = _encode_3di_numba(coords.copy())
        np.testing.assert_array_equal(s1, s2)

    def test_short_structure(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=np.float64)
        states = _encode_3di_numba(coords)
        assert len(states) == 3
        assert np.all(states == 0)

    def test_aligner_method(self):
        aligner = StructuralAligner()
        coords = _make_helix(20)
        states = aligner.encode_3di(coords)
        assert states.dtype == np.int8
        assert len(states) == 20

    def test_helix_interior_consistent(self):
        """Interior residues of a perfect helix should get the same state."""
        states = _encode_3di_numba(_make_helix(40))
        interior = states[3:-2]
        assert len(set(interior.tolist())) <= 2

    # ----- Numerical validation of theta/tau -----

    def test_helix_theta_in_bin_1(self):
        """Alpha-helix virtual bond angle should be ~90° (bin 1: 75-100°).

        For our synthetic helix (R=2.3, d=1.5, step=100°), the CA-CA-CA
        angle works out to ~90.4°.  With edges [75, 100, 125], this sits
        comfortably in bin 1 with ~10° margin from both boundaries.
        """
        coords = _make_helix(20)
        states = _encode_3di_numba(coords)
        interior = states[3:-2]
        for s in interior:
            theta_bin = int(s) // 5
            assert theta_bin == 1, f"Expected theta_bin=1 (75-100°), got {theta_bin} for state {s}"

    def test_helix_tau_in_bin_3(self):
        """Alpha-helix pseudo-dihedral should be ~50° (bin 3: 30-110°).

        Combined with theta_bin=1, state should be 1*5 + 3 = 8.
        With tau edges [-120, -40, 30, 110], the helix tau (~50°) has
        ~20° margin from the nearest boundary (30°).
        """
        coords = _make_helix(20)
        states = _encode_3di_numba(coords)
        interior = states[3:-2]
        for s in interior:
            tau_bin = int(s) % 5
            assert tau_bin == 3, f"Expected tau_bin=3 (30-110°), got {tau_bin} for state {s}"

    def test_helix_state_is_8(self):
        """Interior helix residues should map to state 8 = theta_bin(1)*5 + tau_bin(3)."""
        states = _encode_3di_numba(_make_helix(40))
        interior = states[3:-2]
        assert np.all(interior == 8), f"Expected all state 8, got unique states {set(interior.tolist())}"

    def test_extended_state_differs_from_helix(self):
        """Extended strand theta_bin should be >= 2 (theta > 100°)."""
        states = _encode_3di_numba(_make_extended(30))
        interior = states[3:-2]
        for s in interior:
            theta_bin = int(s) // 5
            assert theta_bin >= 2, f"Extended should have theta > 100°, got theta_bin={theta_bin}"

    # ----- Rotation / translation invariance -----

    def test_rotation_invariance(self):
        """3Di encoding must be unchanged under rigid rotation."""
        coords = _make_helix(30)
        angle = np.radians(47.0)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        rotated = coords @ R.T
        s1 = _encode_3di_numba(coords)
        s2 = _encode_3di_numba(rotated)
        np.testing.assert_array_equal(s1, s2)

    def test_translation_invariance(self):
        """3Di encoding must be unchanged under translation."""
        coords = _make_helix(30)
        shifted = coords + np.array([100.0, -50.0, 200.0])
        s1 = _encode_3di_numba(coords)
        s2 = _encode_3di_numba(shifted)
        np.testing.assert_array_equal(s1, s2)

    def test_rotation_and_translation_combined(self):
        """Arbitrary rigid transform should not change 3Di states."""
        coords = _make_random_coil(40, seed=7)
        angle = np.radians(123.0)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)],
        ])
        transformed = (coords @ Rx.T) + np.array([-30.0, 42.0, 7.7])
        s1 = _encode_3di_numba(coords)
        s2 = _encode_3di_numba(transformed)
        np.testing.assert_array_equal(s1, s2)

    # ----- Degenerate / edge cases -----

    def test_collinear_atoms_no_crash(self):
        """Collinear CAs produce zero cross-product norms; should not crash."""
        coords = np.zeros((10, 3))
        for i in range(10):
            coords[i] = [i * 3.8, 0.0, 0.0]
        states = _encode_3di_numba(coords)
        assert states.shape == (10,)
        assert np.all(states >= 0)

    def test_four_residues_minimum(self):
        """Exactly 4 residues: only position 2 is encodable."""
        coords = np.array([
            [0, 0, 0], [3.8, 0, 0], [7.0, 2.0, 0], [10.0, 1.0, 1.5]
        ], dtype=np.float64)
        states = _encode_3di_numba(coords)
        assert states.shape == (4,)
        assert states[0] == 0  # boundary
        assert states[1] == 0  # boundary
        assert states[3] == 0  # boundary
        assert states[2] >= 0  # the only interior position


# ===================================================================
# K-mer index
# ===================================================================


class TestKmerHashes:

    def test_hash_length(self):
        seq = np.arange(10, dtype=np.int8)
        hashes = _extract_kmer_hashes(seq, 6, 20)
        assert len(hashes) == 5  # 10 - 6 + 1

    def test_short_seq_empty(self):
        seq = np.array([1, 2, 3], dtype=np.int8)
        hashes = _extract_kmer_hashes(seq, 6, 20)
        assert len(hashes) == 0

    def test_hashes_are_unique_for_different_kmers(self):
        seq = np.arange(12, dtype=np.int8)
        hashes = _extract_kmer_hashes(seq, 6, 20)
        assert len(set(hashes.tolist())) == len(hashes)

    def test_identical_kmer_same_hash(self):
        seq = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=np.int8)
        hashes = _extract_kmer_hashes(seq, 6, 20)
        assert hashes[0] == hashes[6]


class TestBuildIndex:

    def test_index_contains_entries(self):
        seqs = [np.arange(10, dtype=np.int8), np.arange(10, dtype=np.int8)]
        idx = build_kmer_index(seqs, k=4, alphabet_size=20)
        assert len(idx) > 0

    def test_identical_sequences_share_all_kmers(self):
        seq = np.arange(10, dtype=np.int8)
        seqs = [seq.copy(), seq.copy()]
        idx = build_kmer_index(seqs, k=4, alphabet_size=20)
        for posting_list in idx.values():
            assert 0 in posting_list
            assert 1 in posting_list

    def test_disjoint_sequences_share_nothing(self):
        s1 = np.zeros(10, dtype=np.int8)
        s2 = np.full(10, 19, dtype=np.int8)
        seqs = [s1, s2]
        idx = build_kmer_index(seqs, k=4, alphabet_size=20)
        for posting_list in idx.values():
            assert not (0 in posting_list and 1 in posting_list)


class TestPrefilterPairs:

    def test_identical_pair_is_candidate(self):
        seq = _encode_3di_numba(_make_helix(30))
        pairs = prefilter_pairs([seq.copy(), seq.copy()], k=6, min_score=0.1)
        assert (0, 1) in pairs

    def test_dissimilar_pair_filtered(self):
        s1 = _encode_3di_numba(_make_helix(30))
        s2 = _encode_3di_numba(_make_extended(30))
        pairs = prefilter_pairs([s1, s2], k=6, min_score=0.5)
        assert (0, 1) not in pairs

    def test_symmetry(self):
        s1 = _encode_3di_numba(_make_helix(30))
        s2 = _encode_3di_numba(_make_helix(30))
        s2[10:15] = 0  # perturb
        pairs = prefilter_pairs([s1, s2], k=6, min_score=0.1)
        for i, j in pairs:
            assert i < j

    def test_self_pair_excluded(self):
        seq = _encode_3di_numba(_make_helix(30))
        pairs = prefilter_pairs([seq], k=6, min_score=0.01)
        assert len(pairs) == 0


# ===================================================================
# Fast clustering
# ===================================================================


class TestClusterFast:

    @pytest.fixture()
    def toolkit(self):
        return AnalysisToolkit()

    def test_returns_dataframe(self, toolkit):
        structures = {
            "h1": _coords_to_df(_make_helix(30), "h1"),
            "h2": _coords_to_df(_make_helix(30), "h2"),
            "e1": _coords_to_df(_make_extended(30), "e1"),
        }
        df = toolkit.cluster_fast(structures, distance_threshold=5.0)
        assert isinstance(df, pl.DataFrame)
        assert "structure_id" in df.columns
        assert "cluster" in df.columns
        assert "centroid_id" in df.columns
        assert "rmsd_to_centroid" in df.columns
        assert df.height == 3

    def test_identical_structures_same_cluster(self, toolkit):
        structures = {
            "a": _coords_to_df(_make_helix(30), "a"),
            "b": _coords_to_df(_make_helix(30), "b"),
            "c": _coords_to_df(_make_helix(30), "c"),
        }
        df = toolkit.cluster_fast(structures, distance_threshold=5.0)
        clusters = df["cluster"].unique().to_list()
        assert len(clusters) == 1

    def test_different_structures_separate_clusters(self, toolkit):
        structures = {
            "helix": _coords_to_df(_make_helix(30), "helix"),
            "extended": _coords_to_df(_make_extended(30), "extended"),
        }
        df = toolkit.cluster_fast(structures, distance_threshold=0.5)
        assert df["cluster"].n_unique() == 2

    def test_single_structure(self, toolkit):
        structures = {"only": _coords_to_df(_make_helix(20), "only")}
        df = toolkit.cluster_fast(structures, distance_threshold=5.0)
        assert df.height == 1
        assert df["cluster"][0] == 1

    def test_centroid_rmsd_is_zero(self, toolkit):
        """Every centroid should report rmsd_to_centroid = 0."""
        structures = {
            "h1": _coords_to_df(_make_helix(30), "h1"),
            "e1": _coords_to_df(_make_extended(30), "e1"),
        }
        df = toolkit.cluster_fast(structures, distance_threshold=0.5)
        centroids = df.filter(pl.col("structure_id") == pl.col("centroid_id"))
        for val in centroids["rmsd_to_centroid"].to_list():
            assert val == 0.0, f"Centroid rmsd_to_centroid should be 0, got {val}"

    def test_variable_length_clustering(self, toolkit):
        """Structures of different lengths should cluster correctly."""
        structures = {
            "short_h": _coords_to_df(_make_helix(20), "short_h"),
            "long_h": _coords_to_df(_make_helix(50), "long_h"),
            "ext": _coords_to_df(_make_extended(35), "ext"),
        }
        df = toolkit.cluster_fast(structures, distance_threshold=5.0, coverage_threshold=0.3)
        assert df.height == 3
        assert "cluster" in df.columns
        ext_cluster = df.filter(pl.col("structure_id") == "ext")["cluster"][0]
        short_cluster = df.filter(pl.col("structure_id") == "short_h")["cluster"][0]
        assert ext_cluster != short_cluster, "Extended and helix should be in different clusters"

    def test_perturbed_extended_clusters_with_original(self, toolkit):
        """An extended chain with small noise should cluster with the clean version.

        Uses the extended chain (theta ≈ 154°) rather than the helix
        (theta ≈ 90.4°) because the helix sits right at the 90° bin
        boundary where even tiny noise flips states.  The extended
        chain is deep inside bin 3 (130-180°), so noise is safe.
        """
        clean = _make_extended(30)
        rng = np.random.RandomState(99)
        noisy = clean + rng.normal(0, 0.2, clean.shape)
        structures = {
            "clean": _coords_to_df(clean, "clean"),
            "noisy": _coords_to_df(noisy, "noisy"),
        }
        df = toolkit.cluster_fast(structures, distance_threshold=3.0)
        assert df["cluster"].n_unique() == 1, (
            "Extended chain with 0.2 Å noise should cluster with the clean version"
        )

    def test_bin_boundary_resilience_helix(self, toolkit):
        """With shifted bin edges, 0.2 Å coordinate noise on a helix
        preserves k-mer overlap and the prefilter finds the pair.

        The helix theta (~90.4°) sits in the center of the 75–100° bin
        with ~10° margin, so moderate noise (0.2 Å on ~2.7 Å CA-CA
        distances) preserves the same 3Di states.
        """
        clean = _make_helix(30)
        rng = np.random.RandomState(99)
        noisy = clean + rng.normal(0, 0.2, clean.shape)
        structures = {
            "clean": _coords_to_df(clean, "clean"),
            "noisy": _coords_to_df(noisy, "noisy"),
        }
        matrix, labels = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        i_c, i_n = labels.index("clean"), labels.index("noisy")
        assert matrix[i_c, i_n] < 1.0, (
            f"Prefilter should find this pair (RMSD={matrix[i_c, i_n]:.2f} Å); "
            "shifted bin edges keep the helix away from boundaries"
        )


# ===================================================================
# Prefiltered RMSD matrix
# ===================================================================


class TestPrefilteredMatrix:

    @pytest.fixture()
    def toolkit(self):
        return AnalysisToolkit()

    def test_matrix_shape(self, toolkit):
        structures = {
            "a": _coords_to_df(_make_helix(25), "a"),
            "b": _coords_to_df(_make_helix(25), "b"),
            "c": _coords_to_df(_make_extended(25), "c"),
        }
        matrix, labels = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        assert matrix.shape == (3, 3)
        assert len(labels) == 3

    def test_diagonal_is_zero(self, toolkit):
        structures = {
            "x": _coords_to_df(_make_helix(20), "x"),
            "y": _coords_to_df(_make_extended(20), "y"),
        }
        matrix, _ = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        np.testing.assert_array_equal(np.diag(matrix), [0.0, 0.0])

    def test_symmetric(self, toolkit):
        structures = {
            "a": _coords_to_df(_make_helix(20), "a"),
            "b": _coords_to_df(_make_random_coil(20), "b"),
        }
        matrix, _ = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_identical_structures_zero_rmsd(self, toolkit):
        coords = _make_helix(25)
        structures = {
            "a": _coords_to_df(coords, "a"),
            "b": _coords_to_df(coords.copy(), "b"),
        }
        matrix, _ = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        assert matrix[0, 1] < 0.01

    def test_prefilter_off_still_works(self, toolkit):
        coords = _make_helix(20)
        structures = {
            "a": _coords_to_df(coords, "a"),
            "b": _coords_to_df(coords.copy(), "b"),
        }
        matrix, _ = toolkit.compute_rmsd_matrix(structures, prefilter=False)
        assert matrix[0, 1] < 0.01

    def test_prefilter_agrees_with_exact_for_similar_pairs(self, toolkit):
        """For pairs the prefilter keeps, RMSD should match the exact computation."""
        h1 = _make_helix(25)
        rng = np.random.RandomState(77)
        h2 = h1 + rng.normal(0, 0.2, h1.shape)
        structures = {
            "a": _coords_to_df(h1, "a"),
            "b": _coords_to_df(h2, "b"),
        }
        matrix_pf, ids_pf = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        matrix_ex, ids_ex = toolkit.compute_rmsd_matrix(structures, prefilter=False)
        i, j = ids_pf.index("a"), ids_pf.index("b")
        if matrix_pf[i, j] < 90.0:
            np.testing.assert_almost_equal(
                matrix_pf[i, j], matrix_ex[i, j], decimal=2,
                err_msg="Prefiltered RMSD should match exact for similar pairs",
            )

    def test_variable_length_matrix(self, toolkit):
        """Matrix computation with mixed-length structures."""
        structures = {
            "short": _coords_to_df(_make_helix(15), "short"),
            "medium": _coords_to_df(_make_helix(25), "medium"),
            "long": _coords_to_df(_make_helix(40), "long"),
        }
        matrix, labels = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        assert matrix.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(matrix), [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_dissimilar_pairs_get_high_default(self, toolkit):
        """Pairs the prefilter skips should get the default 99.9 RMSD."""
        structures = {
            "h": _coords_to_df(_make_helix(30), "h"),
            "e": _coords_to_df(_make_extended(30), "e"),
        }
        matrix, labels = toolkit.compute_rmsd_matrix(structures, prefilter=True)
        i_h, i_e = labels.index("h"), labels.index("e")
        assert matrix[i_h, i_e] > 5.0, "Helix vs extended should have high RMSD"
