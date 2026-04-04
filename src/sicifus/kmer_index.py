"""K-mer inverted index for fast structural prefiltering.

Encodes 3Di structural-alphabet sequences into k-mer hashes, builds an
inverted index, and identifies candidate pairs that share enough k-mers
to warrant a full structural alignment.
"""

import numpy as np
from numba import jit
from typing import List, Set, Tuple, Dict


ALPHABET_SIZE = 20
DEFAULT_K = 6
DEFAULT_MIN_SCORE = 0.1


@jit(nopython=True, cache=True)
def _extract_kmer_hashes(seq, k, alphabet_size):
    """Extract all overlapping k-mer hashes from a 3Di int8 sequence."""
    n = len(seq)
    if n < k:
        return np.empty(0, dtype=np.int64)
    n_kmers = n - k + 1
    hashes = np.empty(n_kmers, dtype=np.int64)
    for i in range(n_kmers):
        h = np.int64(0)
        for j in range(k):
            h = h * np.int64(alphabet_size) + np.int64(seq[i + j])
        hashes[i] = h
    return hashes


def build_kmer_index(
    sequences: List[np.ndarray],
    k: int = DEFAULT_K,
    alphabet_size: int = ALPHABET_SIZE,
) -> Dict[int, List[int]]:
    """Build an inverted index mapping k-mer hash -> list of structure indices.

    Each structure contributes each *unique* k-mer at most once so that long
    repetitive regions do not inflate scores.

    Parameters
    ----------
    sequences : list of int8 ndarrays
        Per-structure 3Di sequences (from ``encode_3di``).
    k : int
        K-mer length (default 6).
    alphabet_size : int
        Alphabet cardinality (default 20).

    Returns
    -------
    dict
        ``{kmer_hash: [struct_idx, ...], ...}``
    """
    index: Dict[int, List[int]] = {}
    for idx, seq in enumerate(sequences):
        hashes = _extract_kmer_hashes(seq, k, alphabet_size)
        seen: set = set()
        for h_raw in hashes:
            h = int(h_raw)
            if h not in seen:
                seen.add(h)
                if h not in index:
                    index[h] = []
                index[h].append(idx)
    return index


def prefilter_pairs(
    all_sequences: List[np.ndarray],
    k: int = DEFAULT_K,
    alphabet_size: int = ALPHABET_SIZE,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Set[Tuple[int, int]]:
    """Identify (i, j) candidate pairs that share enough k-mers.

    For each structure *i*, its unique k-mer hashes are looked up in the
    inverted index.  Structures that share at least ``min_score`` fraction
    of the query's unique k-mers are considered candidates.

    Returns
    -------
    set of (int, int)
        Pairs ``(i, j)`` with ``i < j`` that passed the prefilter.
    """
    n = len(all_sequences)
    index = build_kmer_index(all_sequences, k, alphabet_size)

    pairs: Set[Tuple[int, int]] = set()
    for i in range(n):
        hashes = _extract_kmer_hashes(all_sequences[i], k, alphabet_size)
        unique_hashes = set(int(h) for h in hashes)
        n_query = len(unique_hashes)
        if n_query == 0:
            continue

        threshold = max(int(min_score * n_query), 1)

        scores = np.zeros(n, dtype=np.int32)
        for h in unique_hashes:
            if h in index:
                for idx in index[h]:
                    scores[idx] += 1

        candidates = np.where(scores >= threshold)[0]
        for j in candidates:
            if j != i:
                pair = (min(i, j), max(i, j))
                pairs.add(pair)

    return pairs
