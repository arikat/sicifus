import numpy as np
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
from typing import List, Dict, Tuple, Optional, Union
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from joblib import Parallel, delayed
from .align import StructuralAligner

class AnalysisToolkit:
    """
    Tools for analyzing structural dataframes.
    """
    
    def __init__(self):
        self.aligner = StructuralAligner()

    def compute_rmsd_matrix(self, structures: Dict[str, pl.DataFrame], n_jobs: int = -1,
                            pruning_threshold: Optional[float] = None,
                            prefilter: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Computes all-vs-all RMSD matrix for a dictionary of structures.
        Returns (matrix, labels).
        
        Args:
            structures: Dictionary of structure_id -> DataFrame
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            pruning_threshold: If set (0.0-1.0), skip alignment if sequence length ratio < threshold.
                               Skipped pairs get a high RMSD value (e.g., 99.9).
            prefilter: If True (default), use 3Di k-mer prefiltering to skip
                       dissimilar pairs. Much faster for large N.
        """
        ids = list(structures.keys())
        n = len(ids)
        
        print(f"Pre-processing {n} structures...")
        
        coords_list = []
        for sid in ids:
            coords_list.append(self.aligner.get_ca_coords(structures[sid]))
        
        lengths = [len(c) for c in coords_list]
        num_pairs = n * (n - 1) // 2
        print(f"Computing RMSD matrix for {n} structures ({num_pairs} pairs)...")
        
        if prefilter:
            print(f"  Using 3Di k-mer prefilter.")
            matrix = self._rmsd_matrix_prefiltered(coords_list, ids, n, n_jobs, pruning_threshold)
        else:
            unique_lengths = sorted(set(lengths))
            all_same_length = len(unique_lengths) == 1
            if all_same_length:
                print(f"  All structures have {lengths[0]} residues — using vectorized Kabsch.")
                matrix = self._rmsd_matrix_vectorized(coords_list, n)
            else:
                print(f"  Variable lengths detected — using threaded alignment path.")
                matrix = self._rmsd_matrix_variable_length(coords_list, ids, n, n_jobs, pruning_threshold)
        
        return matrix, ids

    def _rmsd_matrix_vectorized(self, coords_list: List[np.ndarray], n: int) -> np.ndarray:
        """
        Fully vectorized RMSD matrix for same-length structures.
        Uses batched numpy einsum + SVD — no Python loops over pairs.
        """
        L = len(coords_list[0])
        num_pairs = n * (n - 1) // 2
        
        # Stack all structures: (n, L, 3)
        coords_array = np.array(coords_list)
        
        # Center all structures at their centroids
        centroids = coords_array.mean(axis=1, keepdims=True)  # (n, 1, 3)
        centered = coords_array - centroids  # (n, L, 3)
        
        # Precompute squared norms per structure: sum of all squared coords
        sq_norms = np.sum(centered ** 2, axis=(1, 2))  # (n,)
        
        # Compute ALL pairwise covariance matrices in one shot
        H_all = np.einsum('ila,jlb->ijab', centered, centered)  # (n, n, 3, 3)
        
        # Extract upper triangle pairs only
        i_idx, j_idx = np.triu_indices(n, k=1)
        H_pairs = H_all[i_idx, j_idx]  # (num_pairs, 3, 3)
        del H_all
        
        # Batched SVD on all pairs at once
        U, S, Vt = np.linalg.svd(H_pairs)  # S: (num_pairs, 3)
        
        # Handle reflections + compute RMSD
        R = np.einsum('...ji,...kj->...ik', Vt, U)  # (num_pairs, 3, 3)
        dets = np.linalg.det(R)  # (num_pairs,)
        S[dets < 0, -1] *= -1
        
        sum_S = np.sum(S, axis=1)  # (num_pairs,)
        rmsd_sq = (sq_norms[i_idx] + sq_norms[j_idx] - 2.0 * sum_S) / L
        rmsd = np.sqrt(np.maximum(rmsd_sq, 0.0))
        
        # Fill symmetric matrix
        matrix = np.zeros((n, n))
        matrix[i_idx, j_idx] = rmsd
        matrix[j_idx, i_idx] = rmsd
        
        return matrix

    def _rmsd_matrix_variable_length(self, coords_list: List[np.ndarray], ids: List[str], 
                                      n: int, n_jobs: int, pruning_threshold: Optional[float]) -> np.ndarray:
        """
        RMSD matrix for variable-length structures using alignment + threading.
        """
        matrix = np.zeros((n, n))
        
        # Pre-compute structural sequences
        seq_list = []
        lengths = []
        for coords in coords_list:
            lengths.append(len(coords))
            seq_str = self.aligner.encode_structure(coords)
            seq_list.append(np.array([ord(c) for c in seq_str], dtype=np.int32))
        
        from .align import _align_sequences_numba, _superimpose_numba
        
        # Warm up Numba JIT (first call compiles, would skew timing)
        if len(coords_list) >= 2:
            _superimpose_numba(coords_list[0][:3], coords_list[1][:3])
            _align_sequences_numba(seq_list[0][:3], seq_list[1][:3])

        def compute_row(i):
            row_results = np.zeros(n)
            c1, s1, l1 = coords_list[i], seq_list[i], lengths[i]
            
            for j in range(i + 1, n):
                c2, s2, l2 = coords_list[j], seq_list[j], lengths[j]
                
                if pruning_threshold is not None:
                    if l1 > 0 and l2 > 0:
                        if min(l1, l2) / max(l1, l2) < pruning_threshold:
                            row_results[j] = 99.9
                            continue
                    else:
                        row_results[j] = 99.9
                        continue
                
                if l1 == l2:
                    rmsd, _, _, _ = _superimpose_numba(c1, c2)
                else:
                    idx1, idx2 = _align_sequences_numba(s1, s2)
                    if len(idx1) < 3:
                        rmsd = 99.9
                    else:
                        rmsd, _, _, _ = _superimpose_numba(c1[idx1], c2[idx2])
                
                row_results[j] = rmsd
            return i, row_results

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_row)(i) for i in range(n)
        )
        
        for i, row_data in results:
            for j in range(i + 1, n):
                val = row_data[j]
                if val != 0.0:
                    matrix[i, j] = val
                    matrix[j, i] = val
        
        return matrix

    def _rmsd_matrix_prefiltered(self, coords_list: List[np.ndarray], ids: List[str],
                                    n: int, n_jobs: int,
                                    pruning_threshold: Optional[float]) -> np.ndarray:
        """RMSD matrix using 3Di k-mer prefilter to skip dissimilar pairs."""
        from .align import _align_sequences_numba, _superimpose_numba, _encode_3di_numba
        from .kmer_index import build_kmer_index, prefilter_pairs

        sequences_3di = [
            _encode_3di_numba(np.ascontiguousarray(c, dtype=np.float64))
            for c in coords_list
        ]
        candidate_pairs = prefilter_pairs(sequences_3di, k=6, alphabet_size=20, min_score=0.1)

        total_pairs = n * (n - 1) // 2
        pct = 100.0 * len(candidate_pairs) / max(total_pairs, 1)
        print(f"  Prefilter kept {len(candidate_pairs)}/{total_pairs} pairs ({pct:.1f}%)")

        seq_list = []
        lengths = []
        for coords in coords_list:
            lengths.append(len(coords))
            seq_str = self.aligner.encode_structure(coords)
            seq_list.append(np.array([ord(c) for c in seq_str], dtype=np.int32))

        if n >= 2:
            _superimpose_numba(coords_list[0][:3], coords_list[1][:3])
            _align_sequences_numba(seq_list[0][:3], seq_list[1][:3])

        matrix = np.full((n, n), 99.9)
        np.fill_diagonal(matrix, 0.0)

        pairs_list = list(candidate_pairs)

        def compute_pair(pair):
            i, j = pair
            c1, s1, l1 = coords_list[i], seq_list[i], lengths[i]
            c2, s2, l2 = coords_list[j], seq_list[j], lengths[j]

            if pruning_threshold is not None:
                if l1 > 0 and l2 > 0:
                    if min(l1, l2) / max(l1, l2) < pruning_threshold:
                        return i, j, 99.9
                else:
                    return i, j, 99.9

            if l1 == l2:
                rmsd, _, _, _ = _superimpose_numba(c1, c2)
            else:
                idx1, idx2 = _align_sequences_numba(s1, s2)
                if len(idx1) < 3:
                    return i, j, 99.9
                rmsd, _, _, _ = _superimpose_numba(c1[idx1], c2[idx2])

            return i, j, rmsd

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_pair)(p) for p in pairs_list
        )

        for i, j, rmsd in results:
            matrix[i, j] = rmsd
            matrix[j, i] = rmsd

        return matrix

    # ------------------------------------------------------------------
    # Fast greedy clustering (no full distance matrix)
    # ------------------------------------------------------------------

    def cluster_fast(self, structures: Dict[str, pl.DataFrame],
                     distance_threshold: float = 2.0,
                     coverage_threshold: float = 0.8) -> pl.DataFrame:
        """Greedy centroid-based structural clustering (linclust-inspired).

        Uses the 3Di k-mer index to quickly identify candidate centroids for
        each structure, then only computes RMSD to those candidates.  No full
        N×N distance matrix is needed.

        Args:
            structures: Dictionary of structure_id -> DataFrame.
            distance_threshold: Maximum RMSD to assign a structure to an
                                existing cluster centroid (Å).
            coverage_threshold: Minimum length-ratio between a structure and a
                                centroid (0-1) for them to be compared.

        Returns:
            Polars DataFrame with columns
            ``[structure_id, cluster, centroid_id, rmsd_to_centroid]``.
        """
        from .align import _encode_3di_numba, _superimpose_numba, _align_sequences_numba
        from .kmer_index import build_kmer_index, _extract_kmer_hashes

        ids = list(structures.keys())
        n = len(ids)

        print(f"Fast clustering {n} structures (threshold={distance_threshold} Å)...")

        coords_list = [self.aligner.get_ca_coords(structures[sid]) for sid in ids]
        lengths = [len(c) for c in coords_list]

        sequences_3di = [
            _encode_3di_numba(np.ascontiguousarray(c, dtype=np.float64))
            for c in coords_list
        ]
        seq_list = [
            np.array([ord(c) for c in self.aligner.encode_structure(c)], dtype=np.int32)
            for c in coords_list
        ]

        index = build_kmer_index(sequences_3di, k=6, alphabet_size=20)

        if n >= 2:
            _superimpose_numba(coords_list[0][:3], coords_list[1][:3])

        order = sorted(range(n), key=lambda i: lengths[i], reverse=True)

        centroid_indices: List[int] = []
        centroid_set: set = set()
        centroid_to_cluster: Dict[int, int] = {}
        cluster_of: Dict[int, int] = {}
        rmsd_of: Dict[int, float] = {}

        for idx in order:
            if not centroid_indices:
                centroid_indices.append(idx)
                centroid_set.add(idx)
                centroid_to_cluster[idx] = 0
                cluster_of[idx] = 0
                rmsd_of[idx] = 0.0
                continue

            hashes = _extract_kmer_hashes(sequences_3di[idx], 6, 20)
            unique_hashes = set(int(h) for h in hashes)
            n_query = len(unique_hashes)

            candidate_centroids: List[int] = []
            if n_query > 0:
                threshold = max(int(0.05 * n_query), 1)
                scores: Dict[int, int] = {}
                for h in unique_hashes:
                    if h in index:
                        for j in index[h]:
                            if j in centroid_set:
                                scores[j] = scores.get(j, 0) + 1
                candidate_centroids = [
                    c for c, s in scores.items() if s >= threshold
                ]

            best_cluster = -1
            best_rmsd = float("inf")

            for c_idx in candidate_centroids:
                l1, l2 = lengths[idx], lengths[c_idx]
                if l1 > 0 and l2 > 0:
                    if min(l1, l2) / max(l1, l2) < coverage_threshold:
                        continue
                if l1 == l2:
                    rmsd, _, _, _ = _superimpose_numba(coords_list[idx], coords_list[c_idx])
                else:
                    i1, i2 = _align_sequences_numba(seq_list[idx], seq_list[c_idx])
                    if len(i1) < 3:
                        continue
                    rmsd, _, _, _ = _superimpose_numba(
                        coords_list[idx][i1], coords_list[c_idx][i2]
                    )
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_cluster = centroid_to_cluster[c_idx]

            if best_cluster >= 0 and best_rmsd <= distance_threshold:
                cluster_of[idx] = best_cluster
                rmsd_of[idx] = best_rmsd
            else:
                new_cid = len(centroid_indices)
                centroid_indices.append(idx)
                centroid_set.add(idx)
                centroid_to_cluster[idx] = new_cid
                cluster_of[idx] = new_cid
                rmsd_of[idx] = 0.0

        cluster_to_centroid = {v: ids[k] for k, v in centroid_to_cluster.items()}

        rows = []
        for idx in range(n):
            cid = cluster_of[idx]
            rows.append({
                "structure_id": ids[idx],
                "cluster": cid + 1,
                "centroid_id": cluster_to_centroid[cid],
                "rmsd_to_centroid": round(rmsd_of[idx], 4),
            })

        df = pl.DataFrame(rows)
        n_clusters = df["cluster"].n_unique()
        print(f"  {n_clusters} clusters found")
        return df

    def build_tree(self, rmsd_matrix: np.ndarray, labels: List[str], method: str = "average"):
        """
        Builds a hierarchical clustering tree from RMSD matrix using Scipy.
        Returns the linkage matrix.
        """
        # Convert to condensed distance matrix
        condensed_dist = squareform(rmsd_matrix)
        Z = linkage(condensed_dist, method=method)
        return Z

    def build_phylo_tree(self, rmsd_matrix: np.ndarray, labels: List[str], root_id: Optional[str] = None):
        """
        Builds a phylogenetic tree from RMSD matrix.
        Uses Scipy's fast C-based linkage, then converts to Biopython Tree for Newick export.
        Tree is unrooted by default. If root_id is provided, roots at that structure.
        """
        condensed_dist = squareform(rmsd_matrix)
        Z = linkage(condensed_dist, method="average")
        
        tree = self._linkage_to_biopython_tree(Z, labels)
        
        if root_id:
            if root_id not in labels:
                raise ValueError(f"Root ID {root_id} not found in labels.")
            tree.root_with_outgroup({"name": root_id})
            tree.rooted = True
            
        return tree

    def _linkage_to_biopython_tree(self, Z: np.ndarray, labels: List[str]):
        """
        Converts a Scipy linkage matrix to a Biopython Tree object.
        """
        from Bio.Phylo.BaseTree import Tree, Clade
        
        root_node = to_tree(Z)
        
        def _build_clade(node, parent_height=None):
            """
            Recursively build Biopython Clade from Scipy ClusterNode.
            node.dist = the height (cumulative RMSD) at which this node's cluster formed.
            Leaves have dist=0. Branch length = parent_height - this_height.
            """
            height = node.dist
            
            if parent_height is not None:
                branch_length = max(parent_height - height, 0.0)
            else:
                branch_length = 0.0  # Root has no branch length
            
            if node.is_leaf():
                return Clade(name=labels[node.id], branch_length=branch_length)
            else:
                left = _build_clade(node.get_left(), parent_height=height)
                right = _build_clade(node.get_right(), parent_height=height)
                clade = Clade(branch_length=branch_length)
                clade.clades = [left, right]
                return clade
        
        root_clade = _build_clade(root_node)
        
        return Tree(root=root_clade, rooted=False)

    def cluster_from_tree(self, tree, distance_threshold: float) -> pl.DataFrame:
        """
        Derives clusters directly from the phylogenetic tree by cutting branches
        whose length (RMSD) exceeds the threshold. Each resulting subtree's 
        leaves form a cluster.
        
        This uses the actual tree topology and RMSD branch lengths — the clusters
        are defined by the tree itself, not by an external algorithm.
        
        Args:
            tree: Biopython Tree object with RMSD branch lengths.
            distance_threshold: Cut any branch longer than this RMSD value.
                               e.g. 1.0 means groups separated by > 1 Å RMSD are different clusters.
            
        Returns:
            Polars DataFrame with columns: structure_id, cluster
        """
        assignments = {}
        cluster_counter = [0]
        
        def _assign(clade, current_cluster):
            """Walk the tree; cut branches that exceed the threshold."""
            if clade.is_terminal():
                assignments[clade.name] = current_cluster
            else:
                for child in clade.clades:
                    bl = child.branch_length if child.branch_length is not None else 0.0
                    if bl > distance_threshold:
                        # This branch is too long — new cluster for this subtree
                        cluster_counter[0] += 1
                        _assign(child, cluster_counter[0])
                    else:
                        # Same cluster
                        _assign(child, current_cluster)
        
        cluster_counter[0] = 1
        _assign(tree.root, 1)
        
        return pl.DataFrame({
            "structure_id": list(assignments.keys()),
            "cluster": list(assignments.values())
        })

    def plot_tree(self, tree_obj: Union[np.ndarray, Phylo.BaseTree.Tree], labels: Optional[List[str]] = None, output_file: Optional[str] = None):
        """
        Plots the tree. Handles both Scipy linkage matrix and Biopython Tree.
        """
        plt.figure(figsize=(10, 7))
        
        if isinstance(tree_obj, np.ndarray):
            # Scipy Linkage Matrix
            dendrogram(tree_obj, labels=labels, leaf_rotation=90)
            plt.title("Structural Phylogenetic Tree (RMSD-based)")
            plt.xlabel("Structure ID")
            plt.ylabel("RMSD")
        elif isinstance(tree_obj, Phylo.BaseTree.Tree):
            # Biopython Tree
            plt.clf()
            Phylo.draw(tree_obj, do_show=False)
            plt.title("Structural Phylogenetic Tree (Rooted)" if tree_obj.rooted else "Structural Phylogenetic Tree")
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

    def plot_circular_tree(self, Z: np.ndarray, labels: List[str], 
                           cluster_df: Optional[pl.DataFrame] = None,
                           output_file: Optional[str] = None,
                           show_labels: Optional[bool] = None,
                           figsize: Tuple[int, int] = (14, 14),
                           linewidth: float = 0.4):
        """
        Plots an unrooted circular/radial dendrogram from a linkage matrix.
        
        Args:
            Z: Scipy linkage matrix.
            labels: Structure IDs (leaf labels).
            cluster_df: Optional cluster assignments (from cluster_structures) to color branches.
            output_file: If provided, saves to file instead of showing.
            show_labels: Whether to show leaf labels. Defaults to True if <=100 leaves, else False.
            figsize: Figure size.
            linewidth: Line width for branches.
        """
        n_leaves = len(labels)
        
        if show_labels is None:
            show_labels = n_leaves <= 100
        
        # Get dendrogram coordinate data (without plotting)
        dn = dendrogram(Z, no_plot=True, count_sort=True, labels=labels)
        plt.close()  # close the blank figure dendrogram might have created
        
        icoord = np.array(dn['icoord'])  # x-coords of each U-shape: (n_merges, 4)
        dcoord = np.array(dn['dcoord'])  # y-coords (heights): (n_merges, 4)
        leaf_label_order = dn['ivl']     # ordered leaf labels
        
        # Build a leaf_label -> cluster color mapping
        cluster_colors = None
        if cluster_df is not None:
            n_clusters = cluster_df["cluster"].n_unique()
            cmap = plt.cm.get_cmap("tab20" if n_clusters <= 20 else "hsv", n_clusters)
            
            cluster_color_map = {}
            for row in cluster_df.iter_rows(named=True):
                cluster_color_map[row["structure_id"]] = cmap((row["cluster"] - 1) / max(n_clusters - 1, 1))
            cluster_colors = cluster_color_map
        
        # Map leaf x-positions to angles (0 to 2π)
        # Dendrogram leaf x-positions are at 5, 15, 25, ... (spacing=10)
        x_min = 5.0
        x_max = 5.0 + (n_leaves - 1) * 10.0
        x_range = x_max - x_min if x_max > x_min else 1.0
        
        max_height = np.max(dcoord) if np.max(dcoord) > 0 else 1.0
        
        def x_to_angle(x):
            return (x - x_min) / x_range * 2.0 * np.pi
        
        def y_to_radius(y):
            # Height 0 (leaves) → outer ring; max height (root) → center
            return 1.0 - (y / max_height) * 0.85
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        # Draw each U-shape link
        for xs, ys in zip(icoord, dcoord):
            # xs: [left_x, left_x, right_x, right_x]
            # ys: [bottom_left_y, top_y, top_y, bottom_right_y]
            
            a = [x_to_angle(x) for x in xs]
            r = [y_to_radius(y) for y in ys]
            
            # Determine branch color: if cluster_colors, use the color of the left child leaf
            color = '#555555'
            
            # Left vertical: (a[0], r[0]) → (a[1], r[1])
            ax.plot([a[0], a[1]], [r[0], r[1]], color=color, linewidth=linewidth, solid_capstyle='round')
            
            # Right vertical: (a[2], r[2]) → (a[3], r[3])
            ax.plot([a[2], a[3]], [r[2], r[3]], color=color, linewidth=linewidth, solid_capstyle='round')
            
            # Top arc: from a[1] to a[2] at radius r[1] (= r[2])
            n_arc = max(int(abs(a[2] - a[1]) / (2 * np.pi) * 100), 2)
            arc_angles = np.linspace(a[1], a[2], n_arc)
            arc_radii = np.full_like(arc_angles, r[1])
            ax.plot(arc_angles, arc_radii, color=color, linewidth=linewidth, solid_capstyle='round')
        
        # Draw colored dots and labels at leaf positions
        leaf_angles = [x_to_angle(5.0 + i * 10.0) for i in range(n_leaves)]
        leaf_radius = y_to_radius(0)
        
        for i, (angle, label) in enumerate(zip(leaf_angles, leaf_label_order)):
            dot_color = cluster_colors.get(label, '#888888') if cluster_colors else '#333333'
            ax.plot(angle, leaf_radius, 'o', color=dot_color, markersize=2.5, zorder=5)
            
            if show_labels:
                # Rotate label to read outward
                angle_deg = np.degrees(angle)
                ha = 'left' if angle < np.pi else 'right'
                rotation = angle_deg if angle < np.pi else angle_deg - 180
                ax.text(angle, leaf_radius + 0.04, label, fontsize=5, rotation=rotation,
                        ha=ha, va='center', rotation_mode='anchor',
                        color=dot_color if cluster_colors else '#333333')
        
        # Draw cluster legend if clusters provided
        if cluster_df is not None:
            n_clusters = cluster_df["cluster"].n_unique()
            cmap = plt.cm.get_cmap("tab20" if n_clusters <= 20 else "hsv", n_clusters)
            cluster_ids = sorted(cluster_df["cluster"].unique().to_list())
            
            legend_handles = []
            for cid in cluster_ids:
                count = cluster_df.filter(pl.col("cluster") == cid).height
                color = cmap((cid - 1) / max(n_clusters - 1, 1))
                patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                   markersize=8, label=f"Cluster {cid} ({count})")
                legend_handles.append(patch)
            
            ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.3, 1.05),
                      fontsize=8, title="Clusters", title_fontsize=9)
        
        # Clean up polar plot
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def build_similarity_network(self, rmsd_matrix: np.ndarray, labels: List[str], threshold: float) -> nx.Graph:
        """
        Builds a network where edges exist if RMSD < threshold.
        """
        G = nx.Graph()
        n = len(labels)
        
        for i in range(n):
            G.add_node(labels[i])
            
        for i in range(n):
            for j in range(i + 1, n):
                if rmsd_matrix[i, j] < threshold:
                    G.add_edge(labels[i], labels[j], weight=rmsd_matrix[i, j])
                    
        return G

    def calculate_relative_energy(self, df: pl.DataFrame, group_by: Optional[str] = None) -> pl.DataFrame:
        """
        Converts total energies (Hartree) to relative energies (kcal/mol).
        
        If group_by is provided (e.g. "ligand_name"), relative energies are calculated
        separately for each group. Otherwise, the global minimum is used.
        
        Args:
            df: DataFrame with an "energy" column (Hartree).
            group_by: Column to group by before finding minimum (optional).
            
        Returns:
            DataFrame with new "relative_energy_kcal" column.
        """
        if "energy" not in df.columns:
            raise ValueError("DataFrame must have an 'energy' column (Hartree).")
            
        HARTREE_TO_KCAL = 627.509
        
        if group_by:
            return df.with_columns(
                ((pl.col("energy") - pl.col("energy").min().over(group_by)) * HARTREE_TO_KCAL)
                .alias("relative_energy_kcal")
            )
        else:
            min_e = df["energy"].min()
            return df.with_columns(
                ((pl.col("energy") - min_e) * HARTREE_TO_KCAL).alias("relative_energy_kcal")
            )

    def compute_residue_interaction_network(
        self,
        structure_df: pl.DataFrame,
        distance_cutoff: float = 5.0,
        interaction_types: Optional[List[str]] = None,
    ) -> nx.Graph:
        """Compute residue interaction network based on spatial proximity.

        Creates a graph where nodes are residues and edges represent
        interactions (contacts within distance_cutoff).

        Args:
            structure_df: DataFrame with columns [chain, residue_number, residue_name, x, y, z]
            distance_cutoff: Maximum distance (Å) for residue contact (default 5.0)
            interaction_types: Optional filter for specific residues (e.g., ['PHE', 'TYR', 'TRP'])

        Returns:
            NetworkX graph with residue nodes and interaction edges.
            Node attributes: chain, residue_number, residue_name
            Edge attributes: distance, atom_contacts
        """
        # Group atoms by residue
        residues = (
            structure_df
            .group_by(["chain", "residue_number", "residue_name"])
            .agg([
                pl.col("x").mean().alias("center_x"),
                pl.col("y").mean().alias("center_y"),
                pl.col("z").mean().alias("center_z"),
            ])
        )

        # Filter by interaction types if specified
        if interaction_types:
            residues = residues.filter(pl.col("residue_name").is_in(interaction_types))

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        for row in residues.iter_rows(named=True):
            node_id = (row["chain"], row["residue_number"])
            G.add_node(
                node_id,
                chain=row["chain"],
                residue_number=row["residue_number"],
                residue_name=row["residue_name"],
                pos=(row["center_x"], row["center_y"], row["center_z"])
            )

        # Compute pairwise distances
        residue_list = list(residues.iter_rows(named=True))
        cutoff_nm = distance_cutoff / 10.0  # Å to nm

        for i in range(len(residue_list)):
            for j in range(i + 1, len(residue_list)):
                res1 = residue_list[i]
                res2 = residue_list[j]

                # Skip same residue
                if res1["chain"] == res2["chain"] and res1["residue_number"] == res2["residue_number"]:
                    continue

                # Calculate center-of-mass distance
                center1 = np.array([res1["center_x"], res1["center_y"], res1["center_z"]])
                center2 = np.array([res2["center_x"], res2["center_y"], res2["center_z"]])
                dist = np.linalg.norm(center1 - center2)

                if dist < cutoff_nm:
                    node1 = (res1["chain"], res1["residue_number"])
                    node2 = (res2["chain"], res2["residue_number"])

                    # Count atom-level contacts
                    atoms1 = structure_df.filter(
                        (pl.col("chain") == res1["chain"]) &
                        (pl.col("residue_number") == res1["residue_number"])
                    )
                    atoms2 = structure_df.filter(
                        (pl.col("chain") == res2["chain"]) &
                        (pl.col("residue_number") == res2["residue_number"])
                    )

                    coords1 = atoms1.select(["x", "y", "z"]).to_numpy()
                    coords2 = atoms2.select(["x", "y", "z"]).to_numpy()

                    from scipy.spatial.distance import cdist
                    atom_distances = cdist(coords1, coords2)
                    num_contacts = int(np.sum(atom_distances < cutoff_nm))

                    G.add_edge(
                        node1,
                        node2,
                        distance=round(dist * 10.0, 3),  # Convert back to Å
                        atom_contacts=num_contacts
                    )

        return G

    def analyze_network_centrality(
        self,
        G: nx.Graph,
        top_n: int = 10,
    ) -> pl.DataFrame:
        """Analyze network centrality metrics to identify key residues.

        Args:
            G: NetworkX graph from compute_residue_interaction_network()
            top_n: Number of top residues to return (default 10)

        Returns:
            DataFrame with columns: [chain, residue_number, residue_name,
            degree_centrality, betweenness_centrality, closeness_centrality]
        """
        # Compute centrality metrics
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)

        # Build DataFrame
        rows = []
        for node in G.nodes():
            chain, res_num = node
            rows.append({
                "chain": chain,
                "residue_number": res_num,
                "residue_name": G.nodes[node]["residue_name"],
                "degree_centrality": round(degree_cent[node], 4),
                "betweenness_centrality": round(betweenness_cent[node], 4),
                "closeness_centrality": round(closeness_cent[node], 4),
            })

        df = pl.DataFrame(rows)

        # Sort by betweenness (often most informative for hubs)
        df = df.sort("betweenness_centrality", descending=True).head(top_n)

        return df

    def plot_interaction_network(
        self,
        G: nx.Graph,
        output_file: Optional[str] = None,
        node_color_by: str = "chain",
        figsize: Tuple[int, int] = (12, 12),
    ):
        """Visualize residue interaction network.

        Args:
            G: NetworkX graph from compute_residue_interaction_network()
            output_file: Path to save figure (if None, shows interactively)
            node_color_by: Color nodes by "chain" or "residue_name" (default "chain")
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

        # Node colors
        if node_color_by == "chain":
            unique_chains = sorted(set(G.nodes[n]["chain"] for n in G.nodes()))
            cmap = plt.cm.get_cmap("tab10", len(unique_chains))
            chain_to_color = {ch: cmap(i) for i, ch in enumerate(unique_chains)}
            node_colors = [chain_to_color[G.nodes[n]["chain"]] for n in G.nodes()]
        else:
            unique_residues = sorted(set(G.nodes[n]["residue_name"] for n in G.nodes()))
            cmap = plt.cm.get_cmap("tab20", len(unique_residues))
            res_to_color = {res: cmap(i) for i, res in enumerate(unique_residues)}
            node_colors = [res_to_color[G.nodes[n]["residue_name"]] for n in G.nodes()]

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300,
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, ax=ax)

        # Labels (residue number only for readability)
        labels = {n: str(n[1]) for n in G.nodes()}  # Just residue number
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title("Residue Interaction Network", fontsize=14, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class LigandAnalyzer:
    """
    Tools for analyzing ligand binding sites, pi-stacking interactions,
    and atom-level protein-ligand contacts.
    """

    # ── Aromatic ring definitions for standard amino acids ────────────
    # Each entry maps residue name -> list of rings, each ring is a tuple
    # of atom names that form the aromatic ring.
    PROTEIN_AROMATIC_RINGS: Dict[str, List[Tuple[str, ...]]] = {
        "PHE": [("CG", "CD1", "CE1", "CZ", "CE2", "CD2")],
        "TYR": [("CG", "CD1", "CE1", "CZ", "CE2", "CD2")],
        "TRP": [
            ("CG", "CD1", "NE1", "CE2", "CD2"),           # 5-membered indole ring
            ("CD2", "CE2", "CE3", "CZ3", "CH2", "CZ2"),   # 6-membered indole ring
        ],
        "HIS": [("CG", "ND1", "CE1", "NE2", "CD2")],
    }
    
    def __init__(self):
        pass

    # ── Binding residue detection (existing, uses backbone CA) ────────

    def find_binding_residues(self, backbone_df: pl.DataFrame, ligand_df: pl.DataFrame, 
                              ligand_name: str, distance_cutoff: float = 5.0) -> pl.DataFrame:
        """
        Identifies residues within a cutoff distance of a specific ligand.
        Uses CA atoms from backbone for fast residue-level proximity.
        """
        target_ligand = ligand_df.filter(pl.col("residue_name") == ligand_name)
        if target_ligand.height == 0:
            return pl.DataFrame()
        
        from scipy.spatial.distance import cdist
        backbone_coords = backbone_df.select(["x", "y", "z"]).to_numpy()
        ligand_coords = target_ligand.select(["x", "y", "z"]).to_numpy()
        
        dists = cdist(backbone_coords, ligand_coords)
        min_dists = np.min(dists, axis=1)
        mask = min_dists < distance_cutoff
        binding_residues = backbone_df.filter(mask)
        return binding_residues.unique(subset=["chain", "residue_number", "residue_name"])

    def get_pocket_residues(self, all_atom_df: pl.DataFrame, ligand_df: pl.DataFrame,
                            ligand_name: str, distance_cutoff: float = 8.0) -> List[str]:
        """
        Identifies all unique residues within the specified distance cutoff
        of any atom in the ligand. Uses all-atom coordinates for accuracy.
        
        Returns a list of residue names (e.g. ["ALA", "HIS", ...]) found in the pocket.
        """
        from scipy.spatial.distance import cdist
        
        target_ligand = ligand_df.filter(pl.col("residue_name") == ligand_name)
        if target_ligand.height == 0 or all_atom_df.height == 0:
            return []
        
        prot_coords = all_atom_df.select(["x", "y", "z"]).to_numpy()
        lig_coords = target_ligand.select(["x", "y", "z"]).to_numpy()
        
        # Calculate distances between all protein atoms and all ligand atoms
        dists = cdist(prot_coords, lig_coords)
        
        # Find protein atoms that are close to ANY ligand atom
        min_dists = np.min(dists, axis=1)
        mask = min_dists < distance_cutoff
        
        # Filter protein atoms
        pocket_atoms = all_atom_df.filter(mask)
        
        if pocket_atoms.height == 0:
            return []
            
        # Get unique residues (chain + number + name)
        unique_residues = pocket_atoms.unique(subset=["chain", "residue_number", "residue_name"])
        
        # Return just the residue names
        return unique_residues.get_column("residue_name").to_list()

    def plot_binding_pocket_composition(self, residue_counts: Dict[str, int],
                                        title: str = "Binding Pocket Composition",
                                        output_file: Optional[str] = None):
        """
        Plots a histogram of residue types found in the binding pocket.
        Ensures all 20 standard amino acids are represented on the X-axis.
        """
        # Standard 20 amino acids
        standard_aa = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        ]
        
        # Separate standard vs non-standard counts
        standard_counts = {aa: residue_counts.get(aa, 0) for aa in standard_aa}
        non_standard_counts = {k: v for k, v in residue_counts.items() if k not in standard_aa}
        
        # Combine for plotting: standard first (alphabetical), then non-standard (sorted by count)
        plot_labels = standard_aa + sorted(non_standard_counts.keys(), key=lambda k: non_standard_counts[k], reverse=True)
        plot_values = [standard_counts.get(l, non_standard_counts.get(l, 0)) for l in plot_labels]
        
        # Filter out non-standard with 0 counts (shouldn't happen based on logic but good safety)
        # Keep all standard even if 0
        final_labels = []
        final_values = []
        for l, v in zip(plot_labels, plot_values):
            if l in standard_aa or v > 0:
                final_labels.append(l)
                final_values.append(v)
        
        fig, ax = plt.subplots(figsize=(max(10, len(final_labels) * 0.4), 6))
        bars = ax.bar(final_labels, final_values, edgecolor="black", alpha=0.8, color="#4CAF50")
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Residue Type", fontsize=12)
        ax.set_ylabel("Frequency (Count)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_binding_histogram(self, residues_list: List[str], title: str = "Ligand Binding Residue Distribution",
                               output_file: Optional[str] = None):
        """Plots a histogram of binding residue types."""
        from collections import Counter
        counts = Counter(residues_list)
        
        if not counts:
            print("No binding residues to plot.")
            return
        
        labels, values = zip(*counts.most_common())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values, edgecolor="black", alpha=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Residue Type", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ── Pi-stacking detection ─────────────────────────────────────────

    @staticmethod
    def _ring_geometry(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the centroid and unit normal vector for a ring 
        defined by a set of 3D coordinates (N, 3).
        """
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[2]  # eigenvector corresponding to smallest singular value
        normal = normal / np.linalg.norm(normal)
        return centroid, normal

    def _get_protein_rings(self, all_atom_df: pl.DataFrame) -> List[Dict]:
        """
        Extracts aromatic ring definitions from protein all-atom data.
        Returns list of dicts with keys: centroid, normal, residue_name,
        residue_number, chain, ring_type.
        """
        rings = []
        # Group by residue
        aromatic_res = all_atom_df.filter(
            pl.col("residue_name").is_in(list(self.PROTEIN_AROMATIC_RINGS.keys()))
        )
        if aromatic_res.height == 0:
            return rings

        grouped = aromatic_res.group_by(["chain", "residue_number", "residue_name"])

        for (chain, resnum, resname), group_df in grouped:
            ring_defs = self.PROTEIN_AROMATIC_RINGS.get(resname, [])
            for ring_atoms in ring_defs:
                ring_df = group_df.filter(pl.col("atom_name").is_in(ring_atoms))
                if ring_df.height < len(ring_atoms) - 1:
                    continue  # missing atoms, skip
                coords = ring_df.select(["x", "y", "z"]).to_numpy()
                if coords.shape[0] < 3:
                    continue
                centroid, normal = self._ring_geometry(coords)
                rings.append({
                    "centroid": centroid,
                    "normal": normal,
                    "residue_name": resname,
                    "residue_number": resnum,
                    "chain": chain,
                    "ring_size": len(ring_atoms),
                    "source": "protein",
                })
        return rings

    def _detect_ligand_rings(self, ligand_atoms: pl.DataFrame) -> List[Dict]:
        """
        Detects aromatic rings in a ligand using distance-based connectivity
        and planarity analysis. No external chemistry toolkit needed.
        """
        import networkx as nx_graph
        from scipy.spatial.distance import cdist as cdist_fn
        
        rings = []
        if ligand_atoms.height < 5:
            return rings

        coords = ligand_atoms.select(["x", "y", "z"]).to_numpy()
        atom_names = ligand_atoms.get_column("atom_name").to_list()
        elements = ligand_atoms.get_column("element").to_list()

        # Build connectivity graph from interatomic distances (covalent bonds ~1.0–1.8 Å)
        dists = cdist_fn(coords, coords)
        G = nx_graph.Graph()
        for i in range(len(coords)):
            G.add_node(i)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if 0.8 < dists[i, j] < 1.85:
                    G.add_edge(i, j)

        # Find all cycles of size 5 or 6
        try:
            cycles = nx_graph.cycle_basis(G)
        except Exception:
            return rings

        for cycle in cycles:
            if len(cycle) not in (5, 6):
                continue
            ring_coords = coords[cycle]
            ring_elements = [elements[idx] for idx in cycle]
            # Aromatic rings are made of C, N, O, S
            if not all(e in ("C", "N", "O", "S") for e in ring_elements):
                continue
            # Check planarity: smallest singular value should be near zero
            centroid = ring_coords.mean(axis=0)
            centered = ring_coords - centroid
            _, s, Vt = np.linalg.svd(centered)
            planarity = s[2] / (s[0] + 1e-10)
            if planarity > 0.15:
                continue  # not planar enough
            normal = Vt[2]
            normal = normal / np.linalg.norm(normal)
            rings.append({
                "centroid": centroid,
                "normal": normal,
                "atom_indices": cycle,
                "atom_names": [atom_names[idx] for idx in cycle],
                "elements": ring_elements,
                "ring_size": len(cycle),
                "source": "ligand",
            })
        return rings

    @staticmethod
    def _classify_pi_interaction(centroid1: np.ndarray, normal1: np.ndarray,
                                  centroid2: np.ndarray, normal2: np.ndarray) -> Optional[str]:
        """
        Classifies the pi-stacking geometry between two aromatic rings.
        Returns one of: "sandwich", "parallel_displaced", "t_shaped", or None.
        
        Criteria (standard computational chemistry definitions):
          - Sandwich:            distance < 4.0 Å, angle < 30°, offset < 1.5 Å
          - Parallel displaced:  distance < 5.5 Å, angle < 30°
          - T-shaped:            distance < 5.5 Å, angle 60–90°
        """
        vec = centroid2 - centroid1
        d = np.linalg.norm(vec)
        
        if d > 7.0 or d < 2.0:
            return None

        cos_angle = abs(np.dot(normal1, normal2))
        cos_angle = np.clip(cos_angle, 0.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Perpendicular offset: projection of centroid-centroid vector onto ring normal
        projection = abs(np.dot(vec, normal1))
        offset = np.sqrt(max(d**2 - projection**2, 0.0))

        if angle < 30:  # roughly parallel normals
            if d < 4.0 and offset < 1.5:
                return "sandwich"
            elif d < 5.5:
                return "parallel_displaced"
        elif angle > 60:  # roughly perpendicular normals
            if d < 5.5:
                return "t_shaped"
        return None

    def detect_pi_stacking(self, all_atom_df: pl.DataFrame, ligand_df: pl.DataFrame,
                           ligand_name: str) -> pl.DataFrame:
        """
        Detects pi-stacking interactions between protein aromatic residues
        and aromatic rings in the specified ligand.
        
        Returns a DataFrame with columns:
          protein_chain, protein_residue, protein_resname, 
          ligand_ring_atoms, interaction_type, distance, angle
        """
        target_ligand = ligand_df.filter(pl.col("residue_name") == ligand_name)
        if target_ligand.height == 0:
            return pl.DataFrame()

        protein_rings = self._get_protein_rings(all_atom_df)
        ligand_rings = self._detect_ligand_rings(target_ligand)

        if not protein_rings or not ligand_rings:
            return pl.DataFrame()

        interactions = []
        for pr in protein_rings:
            for lr in ligand_rings:
                interaction = self._classify_pi_interaction(
                    pr["centroid"], pr["normal"],
                    lr["centroid"], lr["normal"],
                )
                if interaction is not None:
                    vec = lr["centroid"] - pr["centroid"]
                    d = float(np.linalg.norm(vec))
                    cos_a = abs(np.dot(pr["normal"], lr["normal"]))
                    angle = float(np.degrees(np.arccos(np.clip(cos_a, 0.0, 1.0))))
                    interactions.append({
                        "protein_chain": pr["chain"],
                        "protein_residue": pr["residue_number"],
                        "protein_resname": pr["residue_name"],
                        "ligand_ring_atoms": ",".join(lr["atom_names"]),
                        "interaction_type": interaction,
                        "distance": round(d, 2),
                        "angle": round(angle, 1),
                    })

        if not interactions:
            return pl.DataFrame()
        return pl.DataFrame(interactions)

    def plot_pi_stacking(self, interactions_list: List[Dict],
                         title: str = "Pi-Stacking Interactions",
                         output_file: Optional[str] = None):
        """
        Plots a grouped bar chart of pi-stacking interaction types
        broken down by interaction type and protein residue type.
        """
        from collections import Counter
        
        if not interactions_list:
            print("No pi-stacking interactions to plot.")
            return

        type_counts = Counter()
        residue_type_counts: Dict[str, Counter] = {
            "sandwich": Counter(), "parallel_displaced": Counter(), "t_shaped": Counter()
        }
        
        for ix in interactions_list:
            itype = ix["interaction_type"]
            resname = ix["protein_resname"]
            type_counts[itype] += 1
            if itype in residue_type_counts:
                residue_type_counts[itype][resname] += 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: overall counts by interaction type
        type_labels = ["sandwich", "parallel_displaced", "t_shaped"]
        type_colors = {"sandwich": "#2196F3", "parallel_displaced": "#FF9800", "t_shaped": "#4CAF50"}
        counts = [type_counts.get(t, 0) for t in type_labels]
        display_labels = ["Sandwich", "Parallel\nDisplaced", "T-Shaped"]
        bars = axes[0].bar(display_labels, counts, 
                           color=[type_colors[t] for t in type_labels], edgecolor="black", alpha=0.85)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("By Interaction Type", fontsize=12)
        for bar, c in zip(bars, counts):
            if c > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                           str(c), ha="center", fontsize=10)

        # Right: breakdown by residue type
        all_resnames = sorted(set(r for c in residue_type_counts.values() for r in c))
        x = np.arange(len(all_resnames))
        width = 0.25
        for i, (itype, label) in enumerate(zip(type_labels, display_labels)):
            vals = [residue_type_counts[itype].get(r, 0) for r in all_resnames]
            axes[1].bar(x + i * width, vals, width, label=label.replace("\n", " "),
                       color=type_colors[itype], edgecolor="black", alpha=0.85)
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(all_resnames)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title("By Residue Type", fontsize=12)
        axes[1].legend(fontsize=9)
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ── Ligand atom-level contacts (hydrogen bonding etc.) ────────────

    def find_ligand_atom_contacts(self, all_atom_df: pl.DataFrame, ligand_df: pl.DataFrame,
                                   ligand_name: str, distance_cutoff: float = 3.3) -> pl.DataFrame:
        """
        Identifies atom-level contacts between protein atoms and individual
        ligand atoms. Default cutoff of 3.3 Å targets hydrogen-bond-like
        interactions (user can adjust).
        
        Returns a DataFrame with columns:
          ligand_atom, ligand_element, protein_chain, protein_residue,
          protein_resname, protein_atom, protein_element, distance
        """
        from scipy.spatial.distance import cdist as cdist_fn
        
        target_ligand = ligand_df.filter(pl.col("residue_name") == ligand_name)
        if target_ligand.height == 0 or all_atom_df.height == 0:
            return pl.DataFrame()
        
        prot_coords = all_atom_df.select(["x", "y", "z"]).to_numpy()
        lig_coords = target_ligand.select(["x", "y", "z"]).to_numpy()
        
        dists = cdist_fn(prot_coords, lig_coords)  # (P, L)
        
        contacts = []
        # For each ligand atom, find protein atoms within cutoff
        for lig_idx in range(lig_coords.shape[0]):
            close_mask = dists[:, lig_idx] < distance_cutoff
            close_indices = np.where(close_mask)[0]
            
            lig_row = target_ligand.row(lig_idx, named=True)
            
            for prot_idx in close_indices:
                prot_row = all_atom_df.row(int(prot_idx), named=True)
                contacts.append({
                    "ligand_atom": lig_row["atom_name"],
                    "ligand_element": lig_row["element"],
                    "protein_chain": prot_row["chain"],
                    "protein_residue": prot_row["residue_number"],
                    "protein_resname": prot_row["residue_name"],
                    "protein_atom": prot_row["atom_name"],
                    "protein_element": prot_row["element"],
                    "distance": round(float(dists[prot_idx, lig_idx]), 2),
                })
        
        if not contacts:
            return pl.DataFrame()
        return pl.DataFrame(contacts)

    def plot_ligand_contacts(self, contacts_df: pl.DataFrame,
                             title: str = "Ligand Atom Contacts",
                             output_file: Optional[str] = None):
        """
        Plots a bar chart showing which ligand atoms form the most contacts,
        colored by the protein residue type they interact with.
        
        Uses canonical atom labels (e.g. C1, O2) if available, otherwise
        falls back to PDB atom names.
        """
        if contacts_df.height == 0:
            print("No contacts to plot.")
            return

        # Use canonical_atom column if available (consistent across predictors)
        label_col = "canonical_atom" if "canonical_atom" in contacts_df.columns else "ligand_atom"

        # Aggregate: for each ligand atom, count contacts by protein residue type
        agg = (
            contacts_df
            .group_by([label_col, "protein_resname"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        # Get all unique ligand atoms, sorted by total contacts
        atom_totals = (
            agg.group_by(label_col)
            .agg(pl.col("count").sum().alias("total"))
            .sort("total", descending=True)
        )
        lig_atoms = atom_totals.get_column(label_col).to_list()
        all_resnames = sorted(agg.get_column("protein_resname").unique().to_list())

        fig, ax = plt.subplots(figsize=(max(10, len(lig_atoms) * 0.8), 6))
        
        cmap = plt.cm.get_cmap("tab20", len(all_resnames))
        x = np.arange(len(lig_atoms))
        bottom = np.zeros(len(lig_atoms))
        
        for i, resname in enumerate(all_resnames):
            vals = []
            for atom in lig_atoms:
                row = agg.filter(
                    (pl.col(label_col) == atom) & (pl.col("protein_resname") == resname)
                )
                vals.append(row.get_column("count").sum() if row.height > 0 else 0)
            vals = np.array(vals, dtype=float)
            color = cmap(i / max(len(all_resnames) - 1, 1))
            ax.bar(x, vals, bottom=bottom, label=resname, color=color, edgecolor="black", alpha=0.85)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(lig_atoms, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Number of Contacts", fontsize=11)
        ax.set_xlabel("Ligand Atom (canonical)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=7, ncol=3, title="Protein Residue", title_fontsize=8,
                  bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ── 2D Ligand Depiction (RDKit) ──────────────────────────────────

    @staticmethod
    def _rdkit_available() -> bool:
        try:
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError:
            return False

    def build_ligand_mol(self, ligand_atoms: pl.DataFrame, charge: Optional[int] = None,
                         infer_bond_orders: bool = True):
        """
        Builds an RDKit molecule from ligand atom 3D coordinates.
        
        Uses canonical SMILES ordering to assign consistent atom labels
        that are independent of the input file's atom naming convention.
        This ensures the same atom always gets the same label regardless
        of which structure predictor generated the file.
        
        Args:
            ligand_atoms: DataFrame with x, y, z, element, atom_name columns.
            charge: Total formal charge of the ligand. Helps RDKit infer
                    correct bond orders / protonation state.  For example,
                    citrate at pH 7 is typically -3. If None, RDKit guesses.
            infer_bond_orders: If True, attempt to determine double/aromatic
                    bonds from 3D geometry.  If False, only connectivity
                    (single bonds) is determined — safer when the protonation
                    state is unknown.
        
        Returns:
            (mol, pdb_atom_names, canonical_labels, canonical_smiles)
            or (None, None, None, None) if RDKit is not installed.
            
            canonical_labels: list of str like ["C1", "C2", "O1", "O2", ...]
                numbered per-element in canonical SMILES traversal order.
        """
        if not self._rdkit_available():
            return None, None, None, None
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D
        
        coords = ligand_atoms.select(["x", "y", "z"]).to_numpy()
        elements = ligand_atoms.get_column("element").to_list()
        atom_names = ligand_atoms.get_column("atom_name").to_list()
        
        if len(coords) == 0:
            return None, None, None, None
        
        # Build editable molecule with 3D conformer
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(coords))
        
        for i, (elem, coord) in enumerate(zip(elements, coords)):
            atom = Chem.Atom(elem)
            idx = mol.AddAtom(atom)
            conf.SetAtomPosition(idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
        
        mol.AddConformer(conf, assignId=True)
        
        # Step 1: Determine connectivity (which atoms are bonded)
        # Step 2 (optional): Determine bond orders (single / double / aromatic)
        connectivity_ok = False
        try:
            from rdkit.Chem import rdDetermineBonds
            rdDetermineBonds.DetermineConnectivity(mol)
            connectivity_ok = True
            
            if infer_bond_orders:
                try:
                    if charge is not None:
                        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
                    else:
                        rdDetermineBonds.DetermineBondOrders(mol)
                except Exception:
                    pass
        except (ImportError, Exception):
            pass
        
        if not connectivity_ok:
            from scipy.spatial.distance import cdist as cdist_fn
            dists = cdist_fn(coords, coords)
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    if 0.8 < dists[i, j] < 1.85:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
        
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        
        mol_final = mol.GetMol()
        
        # ── Canonical atom labelling ─────────────────────────────────
        # CanonicalRankAtoms gives each atom a unique rank based on the
        # canonical SMILES traversal — identical molecular graphs always
        # produce the same ranking, regardless of input atom order/names.
        canonical_smiles = None
        canonical_labels = list(atom_names)  # fallback to PDB names
        try:
            canonical_smiles = Chem.MolToSmiles(mol_final)
            ranks = Chem.CanonicalRankAtoms(mol_final)
            # Build per-element numbering in canonical order:
            #   rank 0 gets assigned first, then rank 1, etc.
            #   e.g. C1, C2, C3, O1, O2, N1, ...
            n = mol_final.GetNumAtoms()
            # Sort atom indices by their canonical rank
            sorted_indices = sorted(range(n), key=lambda i: ranks[i])
            element_counters: Dict[str, int] = {}
            label_by_idx: Dict[int, str] = {}
            for idx in sorted_indices:
                elem = mol_final.GetAtomWithIdx(idx).GetSymbol()
                element_counters[elem] = element_counters.get(elem, 0) + 1
                label_by_idx[idx] = f"{elem}{element_counters[elem]}"
            canonical_labels = [label_by_idx.get(i, atom_names[i]) for i in range(n)]
        except Exception:
            pass
        
        return mol_final, atom_names, canonical_labels, canonical_smiles

    def plot_ligand_2d(self, ligand_atoms: pl.DataFrame,
                       contacts_df: Optional[pl.DataFrame] = None,
                       title: str = "Ligand 2D Structure",
                       output_file: Optional[str] = None,
                       size: Tuple[int, int] = (700, 500),
                       charge: Optional[int] = None,
                       infer_bond_orders: bool = True,
                       prebuilt_mol=None,
                       prebuilt_canonical_labels: Optional[List[str]] = None):
        """
        Generates a 2D depiction of the ligand using RDKit.  Atoms are
        labelled with **canonical SMILES-derived names** (e.g. C1, O2, N1)
        so they match the contacts bar chart and are consistent across
        different structure predictors.
        
        If contacts_df is provided, atoms are color-coded by the number of
        protein contacts (red = many, blue = few, gray = none).
        
        Args:
            ligand_atoms: DataFrame of ligand atoms (from one structure).
            contacts_df: Optional contacts DataFrame to color-code atoms.
                         Must contain a "canonical_atom" column.
            title: Plot title.
            output_file: Save image to file. If None, displays inline.
            size: Image dimensions (width, height) in pixels.
            charge: Total formal charge of the ligand (e.g. -3 for citrate).
            infer_bond_orders: If True, attempt to determine double/aromatic
                    bonds. If False, show only connectivity (all single bonds).
            prebuilt_mol: Optional pre-built RDKit Mol from build_ligand_mol().
                    Avoids a redundant rebuild and guarantees the same
                    canonical labels used by the bar chart.
            prebuilt_canonical_labels: Optional canonical labels matching
                    prebuilt_mol atom order.  Must be provided together
                    with prebuilt_mol for consistency.
        
        Returns:
            PNG data as bytes (or None if RDKit unavailable).
        """
        if not self._rdkit_available():
            print("RDKit is not installed. Install with: pip install rdkit")
            print("  2D ligand depiction requires RDKit.")
            return None
        
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        
        # Reuse pre-built molecule + labels when available (single source
        # of truth shared with the bar chart) to guarantee consistency.
        if prebuilt_mol is not None and prebuilt_canonical_labels is not None:
            mol = prebuilt_mol
            canonical_labels = prebuilt_canonical_labels
        else:
            mol, _pdb_names, canonical_labels, canonical_smiles = self.build_ligand_mol(
                ligand_atoms, charge=charge, infer_bond_orders=infer_bond_orders
            )
            if mol is None:
                print("Could not build ligand molecule.")
                return None
            if canonical_smiles:
                print(f"  Canonical SMILES: {canonical_smiles}")
        
        # Compute 2D coordinates for clean layout
        mol_2d = Chem.RWMol(mol)
        AllChem.Compute2DCoords(mol_2d)
        mol_2d = mol_2d.GetMol()
        
        # Label each atom with its canonical label (e.g. "C1", "O2")
        for idx, label in enumerate(canonical_labels):
            if idx < mol_2d.GetNumAtoms():
                mol_2d.GetAtomWithIdx(idx).SetProp("atomNote", label)
        
        # Build highlight colors if contact data is available
        highlight_atoms = {}  # idx -> color tuple (r, g, b, a)
        highlight_radii = {}
        
        if contacts_df is not None and contacts_df.height > 0:
            # Use canonical_atom column if present, otherwise fall back to ligand_atom
            label_col = "canonical_atom" if "canonical_atom" in contacts_df.columns else "ligand_atom"
            contact_counts = (
                contacts_df.group_by(label_col)
                .agg(pl.len().alias("count"))
            )
            count_map = dict(zip(
                contact_counts.get_column(label_col).to_list(),
                contact_counts.get_column("count").to_list(),
            ))
            
            max_count = max(count_map.values()) if count_map else 1
            
            for idx, label in enumerate(canonical_labels):
                if idx >= mol_2d.GetNumAtoms():
                    break
                c = count_map.get(label, 0)
                if c > 0:
                    # Gradient: blue (few) → red (many)
                    frac = c / max_count
                    r = frac
                    b = 1.0 - frac
                    g = 0.2
                    highlight_atoms[idx] = (r, g, b, 0.4)
                    highlight_radii[idx] = 0.35 + 0.15 * frac
        
        # Draw
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        draw_opts = drawer.drawOptions()
        draw_opts.annotationFontScale = 0.3
        draw_opts.bondLineWidth = 2.0
        # Increase font size if the attribute exists (varies by RDKit version)
        for attr in ("baseFontSize", "minFontSize"):
            if hasattr(draw_opts, attr):
                try:
                    setattr(draw_opts, attr, 8)
                except Exception:
                    pass
        
        if highlight_atoms:
            atom_indices = list(highlight_atoms.keys())
            atom_colors = highlight_atoms
            atom_radii = highlight_radii
            drawer.DrawMolecule(
                mol_2d,
                highlightAtoms=atom_indices,
                highlightAtomColors=atom_colors,
                highlightAtomRadii=atom_radii,
                highlightBonds=[],
            )
        else:
            drawer.DrawMolecule(mol_2d)
        
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        
        if output_file:
            with open(output_file, "wb") as f:
                f.write(png_data)
            print(f"  2D ligand structure saved to {output_file}")
        else:
            # Display inline (works in Jupyter notebooks)
            try:
                from IPython.display import display, Image as IPImage
                display(IPImage(data=png_data))
            except ImportError:
                # Not in a notebook — save to temp and inform user
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(png_data)
                tmp.close()
                print(f"  2D ligand structure saved to {tmp.name}")
        
        return png_data
