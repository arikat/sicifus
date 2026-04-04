import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
import time

from .io import CIFLoader
from .align import StructuralAligner
from .analysis import AnalysisToolkit, LigandAnalyzer
from .energy import XTBScorer
from .mutate import MutationEngine, Mutation, RepairResult, StabilityResult, MutationResult, BindingResult, InterfaceMutationResult, _RepairCache
from .visualization import (
    plot_ddg, plot_energy_terms, plot_position_scan_heatmap,
    plot_alanine_scan, plot_ddg_distribution
)

class Sicifus:
    """
    Main API for Sicifus.
    """
    
    def __init__(self, db_path: str = "sicifus_db", xtb_work_dir: str = "./xtb_work"):
        self.db_path = Path(db_path)
        self.backbone_path = self.db_path / "backbone"
        self.heavy_atoms_path = self.db_path / "heavy_atoms"
        self.hydrogens_path = self.db_path / "hydrogens"
        self.ligands_path = self.db_path / "ligands"
        self.metadata_path = self.db_path / "metadata"
        
        # Legacy path support
        self.legacy_all_atom_path = self.db_path / "all_atom"
        
        self.loader = CIFLoader()
        self.aligner = StructuralAligner()
        self.toolkit = AnalysisToolkit()
        self.ligand_analyzer = LigandAnalyzer()
        self.xtb_scorer = XTBScorer(work_dir=xtb_work_dir)
        self.mutation_engine = MutationEngine(work_dir=str(self.db_path / "mutate_work"))
        
        self._backbone_lf: Optional[pl.LazyFrame] = None
        self._heavy_atoms_lf: Optional[pl.LazyFrame] = None
        self._hydrogens_lf: Optional[pl.LazyFrame] = None
        self._ligands_lf: Optional[pl.LazyFrame] = None
        self._metadata_lfs: Dict[str, pl.LazyFrame] = {}
        
        # Cached tree/cluster state
        self._linkage: Optional[np.ndarray] = None
        self._tree_labels: Optional[List[str]] = None
        self._rmsd_matrix: Optional[np.ndarray] = None
        self._clusters: Optional[pl.DataFrame] = None

    def ingest(self, input_folder: str, batch_size: int = 100, file_extension: str = "cif", 
               protonate: bool = False):
        """
        Ingests structure files from a folder into the database.
        
        Args:
            input_folder: Folder containing structure files.
            batch_size: Number of files per parquet partition.
            file_extension: File extension to look for (e.g., "cif", "pdb").
            protonate: If True, uses PDBFixer (OpenMM) to add hydrogens to the structure 
                       before parsing. This ensures consistent protonation for energy calculations.
        """
        print(f"Ingesting {file_extension} files from {input_folder} to {self.db_path}...")
        if protonate:
            print("  Protonation enabled (PDBFixer). This may take longer.")
            
        self.loader.ingest_folder(input_folder, str(self.db_path), batch_size, file_extension, protonate=protonate)
        self.load()

    def load(self):
        """Loads the database (lazy)."""
        if self.backbone_path.exists():
            self._backbone_lf = pl.scan_parquet(str(self.backbone_path / "*.parquet"))
            
        # Load heavy atoms (preferred) or legacy all_atom
        if self.heavy_atoms_path.exists():
            self._heavy_atoms_lf = pl.scan_parquet(str(self.heavy_atoms_path / "*.parquet"))
        elif self.legacy_all_atom_path.exists():
            self._heavy_atoms_lf = pl.scan_parquet(str(self.legacy_all_atom_path / "*.parquet"))
            
        if self.hydrogens_path.exists():
            self._hydrogens_lf = pl.scan_parquet(str(self.hydrogens_path / "*.parquet"))
            
        if self.ligands_path.exists():
            self._ligands_lf = pl.scan_parquet(str(self.ligands_path / "*.parquet"))
        if self.metadata_path.exists():
            for pq in self.metadata_path.glob("*.parquet"):
                name = pq.stem
                self._metadata_lfs[name] = pl.scan_parquet(str(pq))

    @property
    def backbone(self) -> pl.LazyFrame:
        if self._backbone_lf is None:
            self.load()
        if self._backbone_lf is None:
            raise ValueError("No backbone data found. Run ingest() first.")
        return self._backbone_lf

    @property
    def all_atom(self) -> pl.LazyFrame:
        """
        Returns protein heavy atoms (sidechains included). 
        Hydrogens are excluded by default for performance, unless using legacy data.
        """
        if self._heavy_atoms_lf is None:
            self.load()
        if self._heavy_atoms_lf is None:
            raise ValueError(
                "No heavy atom data found. Re-ingest your structures."
            )
        return self._heavy_atoms_lf
        
    @property
    def hydrogens(self) -> pl.LazyFrame:
        """Returns protein hydrogens (if available)."""
        if self._hydrogens_lf is None:
            self.load()
        # It's okay if this is None (e.g. legacy data or no protonation)
        return self._hydrogens_lf

    @property
    def ligands(self) -> pl.LazyFrame:
        if self._ligands_lf is None:
            self.load()
        if self._ligands_lf is None:
            raise ValueError("No ligand data found. Run ingest() first.")
        return self._ligands_lf

    def get_structure(self, structure_id: str) -> pl.DataFrame:
        """Retrieves a specific structure as a DataFrame."""
        return self.backbone.filter(pl.col("structure_id") == structure_id).collect()

    def get_all_atoms(self, structure_id: str) -> pl.DataFrame:
        """Retrieves ALL protein atoms (including sidechains) for a structure."""
        return self.all_atom.filter(pl.col("structure_id") == structure_id).collect()

    def get_ligands(self, structure_id: str) -> pl.DataFrame:
        """Retrieves ligands for a specific structure."""
        return self.ligands.filter(pl.col("structure_id") == structure_id).collect()

    # ── Metadata ─────────────────────────────────────────────────────────

    def load_metadata(self, path: str, name: Optional[str] = None, 
                      id_column: str = "id") -> pl.DataFrame:
        """
        Loads external metadata (CSV) and stores it in the database as parquet.
        The metadata is joined to structures via structure_id.
        
        Supports:
          - A single CSV file with an id column matching structure IDs.
          - A directory of CSVs — all are concatenated.
        
        Args:
            path: Path to a CSV file or a directory of CSVs.
            name: Name for this metadata source (used for storage and lookup).
                  Defaults to the filename stem (e.g. "3ca3.summarize" → "3ca3_summarize").
            id_column: Name of the column in the CSV that contains structure IDs.
                       Defaults to "id".
        
        Returns:
            The loaded metadata as a Polars DataFrame.
        """
        p = Path(path).expanduser()
        
        if p.is_file():
            df = pl.read_csv(str(p))
            if name is None:
                name = p.stem.replace(".", "_").replace("-", "_")
        elif p.is_dir():
            csvs = list(p.rglob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No CSV files found in {p}")
            dfs = [pl.read_csv(str(f)) for f in csvs]
            df = pl.concat(dfs, how="diagonal")
            if name is None:
                name = p.name.replace(".", "_").replace("-", "_")
        else:
            raise FileNotFoundError(f"Path not found: {p}")
        
        # Rename the id column to structure_id for consistency
        if id_column in df.columns and id_column != "structure_id":
            df = df.rename({id_column: "structure_id"})
        elif "structure_id" not in df.columns:
            raise ValueError(
                f"Column '{id_column}' not found in CSV. "
                f"Available columns: {df.columns}. "
                f"Set id_column= to the column containing structure IDs."
            )
        
        # Store as parquet
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        out_path = self.metadata_path / f"{name}.parquet"
        df.write_parquet(str(out_path))
        
        # Cache the lazy frame
        self._metadata_lfs[name] = pl.scan_parquet(str(out_path))
        
        n_rows = df.height
        n_cols = len(df.columns) - 1  # minus structure_id
        matched = 0
        if self._backbone_lf is not None:
            all_ids = self.backbone.select("structure_id").unique().collect().to_series()
            matched = df.filter(pl.col("structure_id").is_in(all_ids)).height
        
        print(f"Loaded metadata '{name}': {n_rows} rows, {n_cols} columns")
        if matched > 0:
            print(f"  {matched}/{n_rows} rows match structures in the database")
        
        return df

    @property
    def meta(self) -> pl.LazyFrame:
        """
        Returns all loaded metadata joined into a single LazyFrame on structure_id.
        If multiple metadata sources are loaded, they are joined together.
        """
        if not self._metadata_lfs:
            self.load()
        if not self._metadata_lfs:
            raise ValueError("No metadata loaded. Use load_metadata() first.")
        
        lfs = list(self._metadata_lfs.values())
        if len(lfs) == 1:
            return lfs[0]
        
        # Left-join all metadata sources on structure_id
        combined = lfs[0]
        for lf in lfs[1:]:
            combined = combined.join(lf, on="structure_id", how="full", coalesce=True)
        return combined

    def meta_columns(self) -> List[str]:
        """Lists all available metadata columns (across all loaded sources)."""
        cols = set()
        for lf in self._metadata_lfs.values():
            cols.update(c for c in lf.columns if c != "structure_id")
        return sorted(cols)

    def hist(self, column: str, bins: int = 30, title: Optional[str] = None,
             output_file: Optional[str] = None, **kwargs):
        """
        Plots a histogram of any metadata column.
        
        If cluster annotations exist, you can pass color_by="cluster" to 
        color the histogram by cluster assignment.
        
        Args:
            column: Column name from the metadata (e.g. "radius_of_gyration").
            bins: Number of histogram bins.
            title: Plot title. Defaults to the column name.
            output_file: Save to file instead of showing.
            **kwargs: Extra kwargs passed to matplotlib hist().
        
        Examples:
            db.hist("radius_of_gyration")
            db.hist("protein_length", bins=50)
        """
        # Collect the column from metadata
        df = self.meta.select(["structure_id", column]).collect().drop_nulls(column)
        
        if df.height == 0:
            print(f"No data found for column '{column}'. Available columns:")
            print(f"  {self.meta_columns()}")
            return
        
        values = df.get_column(column).to_numpy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color_by = kwargs.pop("color_by", None)
        
        if color_by == "cluster" and self._clusters is not None:
            # Join with cluster assignments
            joined = df.join(self._clusters, on="structure_id", how="left")
            cluster_col = joined.get_column("cluster")
            unique_clusters = sorted(cluster_col.drop_nulls().unique().to_list())
            
            n_clust = len(unique_clusters)
            cmap = plt.cm.get_cmap("tab20" if n_clust <= 20 else "hsv", n_clust)
            
            for i, cid in enumerate(unique_clusters):
                cluster_vals = joined.filter(pl.col("cluster") == cid).get_column(column).to_numpy()
                color = cmap(i / max(n_clust - 1, 1))
                ax.hist(cluster_vals, bins=bins, alpha=0.6, color=color, 
                        label=f"Cluster {cid}", **kwargs)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.hist(values, bins=bins, edgecolor='black', alpha=0.8, **kwargs)
        
        ax.set_xlabel(column, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title or column, fontsize=13)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def scatter(self, x: str, y: str, title: Optional[str] = None,
                output_file: Optional[str] = None, **kwargs):
        """
        Scatter plot of two metadata columns.
        
        Args:
            x: Column name for x-axis.
            y: Column name for y-axis.
            title: Plot title.
            output_file: Save to file instead of showing.
            **kwargs: Extra kwargs passed to matplotlib scatter().
        
        Examples:
            db.scatter("protein_length", "radius_of_gyration")
        """
        df = self.meta.select(["structure_id", x, y]).collect().drop_nulls([x, y])
        
        if df.height == 0:
            print(f"No data found. Available columns: {self.meta_columns()}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color_by = kwargs.pop("color_by", None)
        
        if color_by == "cluster" and self._clusters is not None:
            joined = df.join(self._clusters, on="structure_id", how="left")
            cluster_col = joined.get_column("cluster").to_numpy().astype(float)
            sc = ax.scatter(joined.get_column(x).to_numpy(), 
                           joined.get_column(y).to_numpy(),
                           c=cluster_col, cmap="tab20", s=10, alpha=0.7, **kwargs)
            plt.colorbar(sc, label="Cluster")
        else:
            ax.scatter(df.get_column(x).to_numpy(), 
                      df.get_column(y).to_numpy(), s=10, alpha=0.7, **kwargs)
        
        ax.set_xlabel(x, fontsize=11)
        ax.set_ylabel(y, fontsize=11)
        ax.set_title(title or f"{y} vs {x}", fontsize=13)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def align_all(self, reference_id: str, target_ids: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Aligns all (or specified) structures to a reference structure.
        Returns a DataFrame with RMSD and alignment stats.
        """
        ref_df = self.get_structure(reference_id)
        if ref_df.height == 0:
            raise ValueError(f"Reference structure {reference_id} not found.")
            
        if target_ids is None:
            # Get all unique IDs (this might be expensive for massive DB, better to use metadata)
            target_ids = self.backbone.select("structure_id").unique().collect().to_series().to_list()
            
        # Remove reference from targets
        if reference_id in target_ids:
            target_ids.remove(reference_id)
            
        results = []
        print(f"Aligning {len(target_ids)} structures to {reference_id}...")
        
        for tid in target_ids:
            target_df = self.get_structure(tid)
            if target_df.height > 0:
                try:
                    rmsd, n_aligned = self.aligner.align_and_superimpose(target_df, ref_df)
                    results.append({
                        "structure_id": tid,
                        "reference_id": reference_id,
                        "rmsd": rmsd,
                        "aligned_residues": n_aligned
                    })
                except Exception as e:
                    print(f"Failed to align {tid}: {e}")
                    
        return pl.DataFrame(results)

    def get_aligned_structure(self, structure_id: str, reference_id: str) -> pl.DataFrame:
        """
        Returns the structure transformed to align with the reference.
        """
        mobile_df = self.get_structure(structure_id)
        ref_df = self.get_structure(reference_id)
        
        if mobile_df.height == 0 or ref_df.height == 0:
            raise ValueError("Structure not found.")
            
        transformed_df, rmsd = self.aligner.align_and_transform(mobile_df, ref_df)
        print(f"Aligned {structure_id} to {reference_id} with RMSD: {rmsd:.2f}")
        return transformed_df

    def generate_tree(self, structure_ids: Optional[List[str]] = None, output_file: Optional[str] = None, 
                      root_id: Optional[str] = None, newick_file: Optional[str] = None,
                      pruning_threshold: Optional[float] = None,
                      layout: str = "circular"):
        """
        Generates a structural phylogenetic tree. Unrooted by default.
        Branch lengths are RMSD values.
        
        This is the expensive step (O(N^2) alignments). After this, use tree_stats() 
        to inspect branch lengths, then annotate_clusters() to assign clusters cheaply.
        
        Args:
            structure_ids: List of structure IDs. If None, uses all structures (warning: O(N^2)).
            output_file: Save the tree plot to this file (e.g. "tree.png").
            root_id: Root the tree at this structure ID. If None, tree is unrooted.
            newick_file: Export to Newick format for iTOL or similar tools.
            pruning_threshold: Skip alignment for structurally dissimilar pairs (0.0-1.0).
            layout: Tree layout for the plot: "circular" (default, unrooted radial) or "rectangular".
        
        Returns:
            Biopython Tree object.
        """
        t_total = time.perf_counter()
        
        if structure_ids is None:
            t0 = time.perf_counter()
            structure_ids = self.backbone.select("structure_id").unique().collect().to_series().to_list()
            
        if len(structure_ids) > 100:
            print(f"Warning: Generating tree for {len(structure_ids)} structures. This involves O(N^2) alignments.")
            
        # Load ALL backbone data in ONE scan, then group by structure_id
        all_data = self.backbone.filter(
            pl.col("structure_id").is_in(structure_ids)
        ).collect()
        
        structures = {}
        for sid, group in all_data.group_by("structure_id"):
            structures[sid[0] if isinstance(sid, tuple) else sid] = group
        
        matrix, labels = self.toolkit.compute_rmsd_matrix(structures, pruning_threshold=pruning_threshold)
        
        # Build the linkage matrix (fast, C-based) — needed for circular plot
        Z = self.toolkit.build_tree(matrix, labels)
        
        # Cache for later use
        self._linkage = Z
        self._tree_labels = labels
        self._rmsd_matrix = matrix
        
        # Build the Biopython tree (fast — needed for Newick and clustering)
        tree_obj = self.toolkit.build_phylo_tree(matrix, labels, root_id)
        self._tree = tree_obj
        
        # Write Newick
        if newick_file:
            from Bio import Phylo
            Phylo.write(tree_obj, newick_file, "newick")
        
        # Plot the tree
        if output_file:
            if layout == "circular":
                self.toolkit.plot_circular_tree(
                    Z, labels, 
                    cluster_df=self._clusters,
                    output_file=output_file
                )
            else:
                self.toolkit.plot_tree(tree_obj, output_file=output_file)
        
        return tree_obj

    def cluster(self, structure_ids: Optional[List[str]] = None,
                distance_threshold: float = 2.0,
                coverage_threshold: float = 0.8,
                output_file: Optional[str] = None) -> pl.DataFrame:
        """Fast greedy structural clustering (no full tree required).

        Uses a 3Di k-mer prefilter to rapidly identify candidate centroids,
        then only computes RMSD for those candidates.  Much faster than
        building a full phylogenetic tree for large datasets.

        Args:
            structure_ids: Structures to cluster. If None, uses all.
            distance_threshold: Max RMSD (Å) for assigning to a centroid.
            coverage_threshold: Min length-ratio for comparing two structures.
            output_file: Save a summary bar-chart of cluster sizes.

        Returns:
            Polars DataFrame with columns
            ``[structure_id, cluster, centroid_id, rmsd_to_centroid]``.
        """
        t0 = time.perf_counter()

        if structure_ids is None:
            structure_ids = (
                self.backbone.select("structure_id")
                .unique().collect().to_series().to_list()
            )

        all_data = self.backbone.filter(
            pl.col("structure_id").is_in(structure_ids)
        ).collect()

        structures = {}
        for sid, group in all_data.group_by("structure_id"):
            structures[sid[0] if isinstance(sid, tuple) else sid] = group

        df = self.toolkit.cluster_fast(
            structures,
            distance_threshold=distance_threshold,
            coverage_threshold=coverage_threshold,
        )

        self._clusters = df.select(["structure_id", "cluster"])

        elapsed = time.perf_counter() - t0
        print(f"Clustering completed in {elapsed:.1f}s")

        if output_file:
            import matplotlib.pyplot as plt
            sizes = (
                df.group_by("cluster")
                .agg(pl.col("structure_id").count().alias("size"))
                .sort("size", descending=True)
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(sizes.height), sizes["size"].to_list(), edgecolor="black", alpha=0.8)
            ax.set_xlabel("Cluster (sorted by size)")
            ax.set_ylabel("Members")
            ax.set_title(f"Fast Clustering — {df['cluster'].n_unique()} clusters "
                         f"(threshold={distance_threshold} Å)")
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

        return df

    def annotate_clusters(self, distance_threshold: float, output_file: Optional[str] = None,
                          layout: str = "circular") -> pl.DataFrame:
        """
        Annotates the tree with cluster labels by cutting branches whose RMSD 
        exceeds distance_threshold. Each resulting subtree becomes a cluster.
        
        This is cheap and instant — run it multiple times with different thresholds 
        after generate_tree() to explore coarse vs fine clustering.
        
        Use tree_stats() first to see the branch length distribution and pick 
        a meaningful threshold.
        
        Args:
            distance_threshold: Cut branches longer than this RMSD value.
                               e.g. 1.0 = subtrees separated by > 1 Å RMSD become different clusters.
            output_file: Optionally re-plot the tree with cluster colors.
            layout: "circular" (default) or "rectangular".
            
        Returns:
            Polars DataFrame with columns: structure_id, cluster
        """
        if not hasattr(self, '_tree') or self._tree is None:
            raise ValueError("No tree available. Run generate_tree() first.")
        
        self._clusters = self.toolkit.cluster_from_tree(self._tree, distance_threshold)
        n_clust = self._clusters["cluster"].n_unique()
        n_structs = self._clusters.height
        
        # Report singleton vs multi-member breakdown
        cluster_sizes = self._clusters.group_by("cluster").agg(pl.col("structure_id").count().alias("size"))
        n_singletons = cluster_sizes.filter(pl.col("size") == 1).height
        n_multi = n_clust - n_singletons
        n_in_multi = n_structs - n_singletons
        
        print(f"Annotated {n_structs} structures into {n_clust} clusters (threshold={distance_threshold})")
        print(f"  {n_multi} clusters with 2+ members ({n_in_multi} structures)")
        print(f"  {n_singletons} singletons (distinct outliers)")
        
        # Re-plot with cluster colors if requested
        if output_file:
            if layout == "circular":
                self.toolkit.plot_circular_tree(
                    self._linkage, self._tree_labels,
                    cluster_df=self._clusters,
                    output_file=output_file
                )
            else:
                self.toolkit.plot_tree(self._tree, output_file=output_file)
        
        return self._clusters

    # ── Tree inspection ───────────────────────────────────────────────────

    def tree_branch_lengths(self) -> np.ndarray:
        """
        Returns all branch lengths from the tree. Use this to understand the
        distribution and pick a meaningful distance_threshold for clustering.
        
        Example:
            bls = db.tree_branch_lengths()
            print(f"min={bls.min():.2f}, median={np.median(bls):.2f}, max={bls.max():.2f}")
            # Then pick a threshold that makes biological sense
        """
        if not hasattr(self, '_tree') or self._tree is None:
            raise ValueError("No tree available. Run generate_tree() first.")
        
        branch_lengths = []
        for clade in self._tree.find_clades():
            if clade.branch_length is not None and clade.branch_length > 0:
                branch_lengths.append(clade.branch_length)
        return np.array(branch_lengths)

    def tree_stats(self):
        """
        Prints summary statistics of the tree's branch lengths (RMSD).
        Helps you pick a good distance_threshold for clustering.
        """
        bls = self.tree_branch_lengths()
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pvals = np.percentile(bls, percentiles)
        
        print(f"Tree branch length statistics ({len(bls)} branches):")
        print(f"  min:    {bls.min():.4f}")
        for p, v in zip(percentiles, pvals):
            print(f"  {p:3d}th:  {v:.4f}")
        print(f"  max:    {bls.max():.4f}")
        print(f"  mean:   {bls.mean():.4f}")
        print()
        print("Tip: set distance_threshold to a percentile value.")
        print("  Higher threshold → fewer, larger clusters (coarser)")
        print("  Lower threshold  → more, smaller clusters (finer)")

    # ── Cluster querying API ──────────────────────────────────────────────

    @property
    def clusters(self) -> pl.DataFrame:
        """
        Returns the cluster assignments DataFrame (structure_id, cluster).
        Available after calling annotate_clusters().
        """
        if self._clusters is None:
            raise ValueError(
                "No cluster assignments available. "
                "Run annotate_clusters(distance_threshold=...) first."
            )
        return self._clusters

    def get_cluster(self, cluster_id: int) -> List[str]:
        """
        Returns all structure IDs belonging to a specific cluster.
        
        Args:
            cluster_id: The cluster number to query.
            
        Returns:
            List of structure IDs in that cluster.
        """
        return (
            self.clusters
            .filter(pl.col("cluster") == cluster_id)
            .get_column("structure_id")
            .to_list()
        )

    def get_cluster_for(self, structure_id: str) -> int:
        """
        Returns the cluster ID that a specific structure belongs to.
        
        Args:
            structure_id: The structure to look up.
            
        Returns:
            The cluster number.
        """
        result = (
            self.clusters
            .filter(pl.col("structure_id") == structure_id)
            .get_column("cluster")
            .to_list()
        )
        if not result:
            raise ValueError(f"Structure {structure_id} not found in cluster assignments.")
        return result[0]

    def get_cluster_siblings(self, structure_id: str) -> List[str]:
        """
        Returns all structure IDs in the same cluster as the given structure.
        
        Args:
            structure_id: The reference structure.
            
        Returns:
            List of structure IDs in the same cluster (including the reference).
        """
        cluster_id = self.get_cluster_for(structure_id)
        return self.get_cluster(cluster_id)

    def cluster_summary(self) -> pl.DataFrame:
        """
        Returns a summary of all clusters: cluster ID, count, and member IDs.
        """
        return (
            self.clusters
            .group_by("cluster")
            .agg([
                pl.col("structure_id").count().alias("count"),
                pl.col("structure_id").alias("members")
            ])
            .sort("cluster")
        )

    def get_clustered_ids(self, min_size: int = 2) -> List[str]:
        """
        Returns structure IDs that belong to clusters with at least min_size members.
        Filters out singletons (or small clusters) — the outliers on long branches.
        
        Use this to rebuild a pruned tree with only related structures:
            related = db.get_clustered_ids(min_size=2)
            db.generate_tree(structure_ids=related, ...)
        
        Args:
            min_size: Minimum cluster size to include (default 2 = drop singletons).
            
        Returns:
            List of structure IDs.
        """
        cluster_sizes = (
            self.clusters
            .group_by("cluster")
            .agg(pl.col("structure_id").count().alias("size"))
        )
        big_clusters = cluster_sizes.filter(pl.col("size") >= min_size).get_column("cluster").to_list()
        
        return (
            self.clusters
            .filter(pl.col("cluster").is_in(big_clusters))
            .get_column("structure_id")
            .to_list()
        )

    def analyze_ligand_binding(self, ligand_name: str, structure_ids: Optional[List[str]] = None,
                               output_file: Optional[str] = None):
        """
        Analyzes binding residues for a specific ligand across structures.
        Plots a histogram of binding residue types.
        """
        if structure_ids is None:
            structure_ids = self.ligands.filter(pl.col("residue_name") == ligand_name)\
                                        .select("structure_id").unique().collect().to_series().to_list()
                                        
        print(f"Analyzing {ligand_name} binding in {len(structure_ids)} structures...")
        
        all_binding_residues = []
        for sid in structure_ids:
            backbone = self.get_structure(sid)
            ligands = self.get_ligands(sid)
            binding_df = self.ligand_analyzer.find_binding_residues(backbone, ligands, ligand_name)
            if binding_df.height > 0:
                all_binding_residues.extend(binding_df["residue_name"].to_list())
                
        if all_binding_residues:
            self.ligand_analyzer.plot_binding_histogram(
                all_binding_residues, 
                title=f"{ligand_name} — Binding Residue Distribution ({len(structure_ids)} structures)",
                output_file=output_file,
            )
        else:
            print("No binding residues found.")

    def analyze_pi_stacking(self, ligand_name: str, structure_ids: Optional[List[str]] = None,
                            output_file: Optional[str] = None,
                            charge: Optional[int] = None,
                            infer_bond_orders: bool = True) -> pl.DataFrame:
        """
        Detects pi-stacking interactions (sandwich, parallel displaced, T-shaped)
        between protein aromatic residues and aromatic rings in the specified ligand.
        
        Requires all_atom data (re-ingest if you only have backbone/CA).
        
        Produces a grouped bar chart of interaction types and residue breakdown.
        Returns a DataFrame of all detected interactions.
        
        The ``ligand_ring_atoms`` column uses the same canonical atom labels
        as ``analyze_ligand_contacts`` and the 2D depiction, so atom names
        are consistent across all analyses.
        
        Args:
            ligand_name: Three-letter ligand code (e.g. "GLC", "ATP").
            structure_ids: Optional list of structures to analyze. Defaults to all
                           structures containing the ligand.
            output_file: Save the plot instead of displaying.
            charge: Total formal charge of the ligand (passed to build_ligand_mol).
            infer_bond_orders: Whether to infer double/aromatic bonds (passed to
                    build_ligand_mol).
        
        Example:
            pi_df = db.analyze_pi_stacking("ATP")
        """
        if structure_ids is None:
            structure_ids = self.ligands.filter(pl.col("residue_name") == ligand_name)\
                                        .select("structure_id").unique().collect().to_series().to_list()
        
        print(f"Detecting pi-stacking with {ligand_name} in {len(structure_ids)} structures...")
        
        all_interactions = []
        unique_smiles = set()
        
        for sid in structure_ids:
            try:
                all_atoms = self.get_all_atoms(sid)
            except ValueError:
                print("  No all_atom data available. Re-ingest structures to enable pi-stacking analysis.")
                return pl.DataFrame()
            ligands = self.get_ligands(sid)
            
            pi_df = self.ligand_analyzer.detect_pi_stacking(all_atoms, ligands, ligand_name)
            
            if pi_df.height > 0:
                # ── Per-structure Canonical Mapping ──────────────────────────
                # Remap ring atoms to canonical labels for this specific structure
                lig_for_struct = ligands.filter(pl.col("residue_name") == ligand_name)
                
                _, pdb_names_i, canonical_labels_i, smi_i = \
                    self.ligand_analyzer.build_ligand_mol(
                        lig_for_struct,
                        charge=charge,
                        infer_bond_orders=infer_bond_orders,
                    )
                
                if smi_i:
                    unique_smiles.add(smi_i)
                
                if pdb_names_i and canonical_labels_i:
                    pdb_to_canonical = dict(zip(pdb_names_i, canonical_labels_i))
                    
                    def _remap_ring_atoms(pdb_csv: str) -> str:
                        return ",".join(
                            pdb_to_canonical.get(a.strip(), a.strip())
                            for a in pdb_csv.split(",")
                        )
                    
                    remapped = [
                        _remap_ring_atoms(v)
                        for v in pi_df.get_column("ligand_ring_atoms").to_list()
                    ]
                    pi_df = pi_df.with_columns(
                        pl.Series("ligand_ring_atoms", remapped)
                    )

                pi_df = pi_df.with_columns(pl.lit(sid).alias("structure_id"))
                all_interactions.append(pi_df)
        
        if len(unique_smiles) > 1:
            print(f"Warning: Found {len(unique_smiles)} different molecular graphs (SMILES) for {ligand_name}.")
            print("  Canonical labeling might be inconsistent if structures have different connectivity/protonation.")
            
        if not all_interactions:
            print("No pi-stacking interactions detected.")
            return pl.DataFrame()
        
        result = pl.concat(all_interactions)
        
        n_sandwich = result.filter(pl.col("interaction_type") == "sandwich").height
        n_parallel = result.filter(pl.col("interaction_type") == "parallel_displaced").height
        n_tshaped = result.filter(pl.col("interaction_type") == "t_shaped").height
        print(f"  Found {result.height} interactions: "
              f"{n_sandwich} sandwich, {n_parallel} parallel displaced, {n_tshaped} T-shaped")
        
        # Plot
        self.ligand_analyzer.plot_pi_stacking(
            result.to_dicts(),
            title=f"{ligand_name} — Pi-Stacking Interactions ({len(structure_ids)} structures)",
            output_file=output_file,
        )
        return result

    def analyze_ligand_contacts(self, ligand_name: str, distance_cutoff: float = 3.3,
                                structure_ids: Optional[List[str]] = None,
                                output_file: Optional[str] = None,
                                ligand_2d_file: Optional[str] = None,
                                charge: Optional[int] = None,
                                infer_bond_orders: bool = True) -> pl.DataFrame:
        """
        Identifies atom-level protein-ligand contacts within a distance cutoff.
        Default 3.3 Å is appropriate for hydrogen bonding.
        
        Requires all_atom data (re-ingest if you only have backbone/CA).
        
        Produces TWO visualizations:
          1) Stacked bar chart: which ligand atoms form the most contacts.
          2) 2D ligand depiction (RDKit): atoms color-coded by contact count
             (red = many, blue = few) so you can cross-reference the chart.
        
        Returns a DataFrame of all contacts.
        
        Args:
            ligand_name: Three-letter ligand code.
            distance_cutoff: Max distance in Å (default 3.3 for H-bonds).
            structure_ids: Optional list of structures.
            output_file: Save the contacts bar chart to file.
            ligand_2d_file: Save the 2D ligand depiction to file.
                            If not set, auto-generates filename from output_file
                            or displays inline.
            charge: Total formal charge of the ligand (e.g. -3 for citrate).
                    Helps RDKit infer correct protonation / bond orders in
                    the 2D depiction.
            infer_bond_orders: If True, RDKit will try to determine double
                    and aromatic bonds from 3D geometry. Set to False if the
                    2D depiction shows incorrect double bonds — it will display
                    connectivity only (all single bonds), which is still useful.
        
        Example:
            contacts = db.analyze_ligand_contacts("GLC", distance_cutoff=3.3)
            # If the 2D shows wrong double bonds, pass the charge or disable inference:
            contacts = db.analyze_ligand_contacts("CIT", charge=-3)
            contacts = db.analyze_ligand_contacts("LIG", infer_bond_orders=False)
        """
        if structure_ids is None:
            structure_ids = self.ligands.filter(pl.col("residue_name") == ligand_name)\
                                        .select("structure_id").unique().collect().to_series().to_list()
        
        print(f"Analyzing {ligand_name} atom contacts (<{distance_cutoff}Å) in {len(structure_ids)} structures...")
        
        all_contacts = []
        representative_ligand_atoms = None
        unique_smiles = set()
        
        # Pre-built mol/labels for the 2D depiction (from the first structure)
        _mol = None
        _canonical_labels = None
        
        for sid in structure_ids:
            try:
                all_atoms = self.get_all_atoms(sid)
            except ValueError:
                print("  No all_atom data available. Re-ingest structures to enable contact analysis.")
                return pl.DataFrame()
            ligands = self.get_ligands(sid)
            
            contacts_df = self.ligand_analyzer.find_ligand_atom_contacts(
                all_atoms, ligands, ligand_name, distance_cutoff=distance_cutoff
            )
            
            if contacts_df.height > 0:
                # ── Per-structure Canonical Mapping ──────────────────────────
                # Compute canonical labels for THIS specific structure instance
                # so that even if PDB atom names differ (e.g. O1 vs O-1),
                # chemically equivalent atoms get the same label (e.g. O4).
                lig_for_struct = ligands.filter(pl.col("residue_name") == ligand_name)
                
                # Build mol for this structure
                mol_i, pdb_names_i, canonical_labels_i, smi_i = \
                    self.ligand_analyzer.build_ligand_mol(
                        lig_for_struct,
                        charge=charge,
                        infer_bond_orders=infer_bond_orders,
                    )
                
                if smi_i:
                    unique_smiles.add(smi_i)
                
                # Save the first valid mol for the 2D plot
                if _mol is None and mol_i is not None:
                    _mol = mol_i
                    _canonical_labels = canonical_labels_i
                    representative_ligand_atoms = lig_for_struct

                # Map PDB names -> Canonical labels for this structure
                if pdb_names_i and canonical_labels_i:
                    pdb_to_canonical = dict(zip(pdb_names_i, canonical_labels_i))
                    
                    # Apply mapping immediately to this structure's contacts
                    mapping_df = pl.DataFrame({
                        "ligand_atom": list(pdb_to_canonical.keys()),
                        "canonical_atom": list(pdb_to_canonical.values()),
                    })
                    contacts_df = contacts_df.join(mapping_df, on="ligand_atom", how="left")
                    contacts_df = contacts_df.with_columns(
                        pl.col("canonical_atom").fill_null(pl.col("ligand_atom"))
                    )
                else:
                    # Fallback if RDKit failed or no atoms
                    contacts_df = contacts_df.with_columns(
                        pl.col("ligand_atom").alias("canonical_atom")
                    )

                contacts_df = contacts_df.with_columns(pl.lit(sid).alias("structure_id"))
                all_contacts.append(contacts_df)
        
        if len(unique_smiles) > 1:
            print(f"Warning: Found {len(unique_smiles)} different molecular graphs (SMILES) for {ligand_name}.")
            print("  Canonical labeling might be inconsistent if structures have different connectivity/protonation.")

        if not all_contacts:
            print("No contacts found at this cutoff.")
            return pl.DataFrame()
        
        result = pl.concat(all_contacts)
        
        # Summary
        n_hbond_candidates = result.filter(
            pl.col("ligand_element").is_in(["N", "O", "S"]) & 
            pl.col("protein_element").is_in(["N", "O", "S"])
        ).height
        print(f"  Found {result.height} contacts ({n_hbond_candidates} potential H-bonds: N/O/S pairs)")
        
        # Plot 1: stacked bar chart — uses canonical_atom column
        self.ligand_analyzer.plot_ligand_contacts(
            result,
            title=f"{ligand_name} — Atom Contacts <{distance_cutoff}Å ({len(structure_ids)} structures)",
            output_file=output_file,
        )
        
        # Plot 2: 2D ligand depiction — reuses the representative mol + canonical
        # labels so atom numbering is identical to the bar chart.
        if representative_ligand_atoms is not None:
            lig2d_out = ligand_2d_file
            if lig2d_out is None and output_file is not None:
                from pathlib import Path as _P
                stem = _P(output_file).stem
                lig2d_out = str(_P(output_file).parent / f"{stem}_ligand2d.png")
            
            self.ligand_analyzer.plot_ligand_2d(
                representative_ligand_atoms,
                contacts_df=result,
                title=f"{ligand_name} — 2D Structure (colored by contacts)",
                output_file=lig2d_out,
                charge=charge,
                infer_bond_orders=infer_bond_orders,
                prebuilt_mol=_mol,
                prebuilt_canonical_labels=_canonical_labels,
            )
        return result

    def get_binding_pockets(self, ligand_name: str, distance_cutoff: float = 8.0,
                            structure_ids: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Returns a DataFrame where each row is a structure and columns are residue counts 
        in the binding pocket (e.g. ALA, TRP, etc.).
        
        This is useful for filtering structures based on pocket composition.
        
        Args:
            ligand_name: Three-letter ligand code.
            distance_cutoff: Radius in Angstroms (default 8.0).
            structure_ids: Optional list of structures.
            
        Returns:
            DataFrame with structure_id and counts for each residue type found.
            Missing residues are filled with 0.
        """
        if structure_ids is None:
            structure_ids = self.ligands.filter(pl.col("residue_name") == ligand_name)\
                                        .select("structure_id").unique().collect().to_series().to_list()
        
        print(f"Extracting {ligand_name} binding pockets (<{distance_cutoff}Å) from {len(structure_ids)} structures...")
        
        pocket_data = []
        all_residue_types = set()
        
        for sid in structure_ids:
            try:
                all_atoms = self.get_all_atoms(sid)
            except ValueError:
                # Skip structures without all-atom data
                continue

            ligands = self.get_ligands(sid)
            
            # Get list of residues in pocket (e.g. ["ALA", "HIS", "ALA"])
            pocket_residues = self.ligand_analyzer.get_pocket_residues(
                all_atoms, ligands, ligand_name, distance_cutoff=distance_cutoff
            )
            
            if pocket_residues:
                from collections import Counter
                counts = Counter(pocket_residues)
                row = {"structure_id": sid}
                row.update(counts)
                pocket_data.append(row)
                all_residue_types.update(counts.keys())
        
        if not pocket_data:
            return pl.DataFrame()
            
        # Create DataFrame and fill missing columns with 0
        df = pl.from_dicts(pocket_data)
        
        # Ensure all columns are numeric (except structure_id) and fill nulls with 0
        fill_cols = [c for c in df.columns if c != "structure_id"]
        df = df.with_columns([
            pl.col(c).fill_null(0).cast(pl.Int32) for c in fill_cols
        ])
        
        return df

    def analyze_binding_pocket(self, ligand_name: str, distance_cutoff: float = 8.0,
                               structure_ids: Optional[List[str]] = None,
                               output_file: Optional[str] = None) -> Dict[str, int]:
        """
        Analyzes the amino acid composition of the binding pocket (residues within 
        a defined radius of the ligand) across all structures.
        
        Produces a histogram of residue counts (X-axis = 20 amino acids).
        
        Args:
            ligand_name: Three-letter ligand code.
            distance_cutoff: Radius in Angstroms (default 8.0).
            structure_ids: Optional list of structures.
            output_file: Save the histogram to file.
            
        Returns:
            Dictionary of residue counts (e.g. {'ALA': 10, 'HIS': 5}).
        """
        # Get per-structure pocket data
        df = self.get_binding_pockets(ligand_name, distance_cutoff, structure_ids)
        
        if df.height == 0:
            print("No pocket residues found.")
            return {}
            
        # Aggregate counts across all structures
        # Sum each residue column
        residue_cols = [c for c in df.columns if c != "structure_id"]
        sums = df.select([pl.col(c).sum() for c in residue_cols]).row(0, named=True)
        
        self.ligand_analyzer.plot_binding_pocket_composition(
            sums,
            title=f"{ligand_name} — Binding Pocket Composition (<{distance_cutoff}Å, {df.height} structures)",
            output_file=output_file
        )
        return sums

    def calculate_relative_energy(self, df: pl.DataFrame, group_by: Optional[str] = None) -> pl.DataFrame:
        """
        Converts total energies (Hartree) to relative energies (kcal/mol).
        Useful for comparing conformers of the *same* system (same atoms).
        
        Args:
            df: DataFrame with an "energy" column (Hartree).
            group_by: Column to group by before finding minimum (optional).
                      e.g. "ligand_name" to find best conformer per ligand.
            
        Returns:
            DataFrame with new "relative_energy_kcal" column.
        """
        return self.toolkit.calculate_relative_energy(df, group_by)

    def score_ligand_energy(self, ligand_name: str, structure_id: str, 
                            distance_cutoff: float = 6.0, charge: int = 0,
                            uhf: int = 0, solvent: str = "water",
                            debug: bool = False) -> Dict[str, float]:
        """
        Calculates the semi-empirical QM energy (GFN2-xTB) of the ligand in its binding pocket.
        
        This extracts the ligand and surrounding protein residues, freezes the protein atoms,
        and optimizes the ligand geometry to find its local energy minimum.
        
        It also calculates the **Interaction Energy**:
            E_int = E_complex - (E_protein + E_ligand)
        
        This allows for comparison across different binding pockets (different protein sequences),
        as it normalizes for the size/composition of the pocket.
        
        Args:
            ligand_name: Three-letter ligand code.
            structure_id: ID of the structure to analyze.
            distance_cutoff: Radius around ligand to include in the calculation (default 6.0 Å).
            charge: Total system charge (default 0).
            uhf: Number of unpaired electrons (default 0).
            solvent: Implicit solvent model (default "water").
            debug: If True, saves input/output structures to debug_structures/ folder.
            
        Returns:
            Dictionary with:
              - "energy": Total energy of complex (Hartree)
              - "gap": HOMO-LUMO gap (eV)
              - "interaction_energy": Interaction energy (kcal/mol)
              - "e_complex", "e_protein", "e_ligand": Component energies (Hartree)
            Returns empty dict if calculation fails or xtb is missing.
        """
        try:
            # Load heavy atoms (default)
            all_atoms = self.get_all_atoms(structure_id)
            
            # Load hydrogens if available and merge
            if self.hydrogens is not None:
                hydrogens = self.hydrogens.filter(pl.col("structure_id") == structure_id).collect()
                if hydrogens.height > 0:
                    all_atoms = pl.concat([all_atoms, hydrogens])
            
            ligands = self.get_ligands(structure_id)
        except ValueError:
            print(f"Error: Could not retrieve atoms for {structure_id}. Ensure all_atom data exists.")
            return {}
            
        # Get pocket atoms for charge estimation
        # We need to re-extract them or pass them through.
        # Since extract_pocket_xyz doesn't return the DF, we quickly re-filter here
        # or update extract_pocket_xyz to return it.
        # For simplicity, let's re-filter (cheap).
        from scipy.spatial.distance import cdist
        target_ligand = ligands.filter(pl.col("residue_name") == ligand_name)
        if target_ligand.height > 0:
            lig_coords = target_ligand.select(["x", "y", "z"]).to_numpy()
            prot_coords = all_atoms.select(["x", "y", "z"]).to_numpy()
            dists = cdist(prot_coords, lig_coords)
            min_dists = np.min(dists, axis=1)
            close_mask = min_dists < distance_cutoff
            close_atoms = all_atoms.filter(close_mask)
            # Expand to complete residues for accurate charge estimation
            residue_id_cols = ["chain", "residue_number", "residue_name"]
            available_cols = [c for c in residue_id_cols if c in all_atoms.columns]
            if available_cols and close_atoms.height > 0:
                touched = close_atoms.select(available_cols).unique()
                pocket_atoms = all_atoms.join(touched, on=available_cols, how="semi")
            else:
                pocket_atoms = close_atoms
        else:
            pocket_atoms = None

        xyz_complex, xyz_protein, xyz_ligand, n_prot = self.xtb_scorer.extract_pocket_xyz(
            all_atoms, ligands, ligand_name, distance_cutoff
        )
        
        if xyz_complex is None:
            return {}

        print(f"Running xTB scoring for {structure_id} (pocket size: {distance_cutoff}Å)...")
        results = self.xtb_scorer.run_scoring(
            xyz_complex, xyz_protein, xyz_ligand,
            n_protein_atoms=n_prot, charge=charge, uhf=uhf, solvent=solvent,
            pocket_atoms=pocket_atoms,
            save_structures=debug,
            structure_id=structure_id
        )
        
        if "interaction_energy" in results:
            print(f"  Interaction Energy: {results['interaction_energy']:.2f} kcal/mol")
        elif "energy" in results:
            print(f"  Total Energy: {results['energy']:.6f} Eh")
        
        return results

    # ── Mutation & Stability (industry-standard) ─────────────────────────────────

    def _structure_to_pdb(self, structure_id: str) -> str:
        """Reconstruct a PDB string for a structure from the Parquet database."""
        from .mutate import _df_to_pdb_string
        all_atoms = self.get_all_atoms(structure_id)
        if self.hydrogens is not None:
            h = self.hydrogens.filter(pl.col("structure_id") == structure_id).collect()
            if h.height > 0:
                all_atoms = pl.concat([all_atoms, h])
        return _df_to_pdb_string(all_atoms)

    def repair_structure(self, structure_id: str, **kwargs) -> RepairResult:
        """Repair a structure: fix missing atoms, add hydrogens, minimise.

        Repairs protein structure by fixing clashes and adding missing atoms.  Requires ``pdbfixer`` and ``openmm``
        (install with ``pip install sicifus[energy]``).

        Args:
            structure_id: ID of the structure in the database.
            **kwargs: Forwarded to :meth:`MutationEngine.repair`.

        Returns:
            RepairResult with repaired PDB and energy change.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.repair(pdb_text, **kwargs)

    def calculate_stability(self, structure_id: str, **kwargs) -> StabilityResult:
        """Calculate total potential energy with per-term decomposition.

        Calculates protein stability using energy minimization.

        Args:
            structure_id: ID of the structure in the database.
            **kwargs: Forwarded to :meth:`MutationEngine.calculate_stability`.

        Returns:
            StabilityResult with total energy (kcal/mol) and per-force-term breakdown.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.calculate_stability(pdb_text, **kwargs)

    def mutate_structure(
        self, structure_id: str, mutations: List[Union[Mutation, str]], **kwargs
    ) -> MutationResult:
        """Apply point mutations, minimise, and compute ddG.

        Args:
            structure_id: ID of the structure in the database.
            mutations: List of Mutation objects or strings
                       (e.g. ``'G13L'`` for Gly at position 13 to Leu).
            **kwargs: Forwarded to :meth:`MutationEngine.mutate`.

        Returns:
            MutationResult with wild-type energy, mutant energy, ddG, and
            mutant PDB strings.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.mutate(pdb_text, mutations, **kwargs)

    def load_mutations(self, csv_path: str) -> pl.DataFrame:
        """Load a mutation list from a CSV file.

        The CSV must have a ``mutation`` column (e.g. ``G13L``).  An optional
        ``chain`` column provides chain IDs; if absent, defaults to ``'A'``.
        Extra columns are preserved as metadata.

        Args:
            csv_path: Path to a CSV file.

        Returns:
            Polars DataFrame ready for :meth:`mutate_batch`.
        """
        return MutationEngine.load_mutations(csv_path)

    def mutate_batch(
        self, structure_id: str, mutations_df: pl.DataFrame, **kwargs
    ) -> pl.DataFrame:
        """Run every mutation in a DataFrame against a structure.

        Each row is an independent single-point mutation.  Extra columns
        from the input are carried through to the result.

        Args:
            structure_id: ID of the structure in the database.
            mutations_df: DataFrame with ``mutation`` and ``chain`` columns
                          (as returned by :meth:`load_mutations`).
            **kwargs: Forwarded to :meth:`MutationEngine.mutate_batch`.

        Returns:
            DataFrame with input columns plus
            ``[wt_energy, mutant_energy, ddg_kcal_mol]``.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.mutate_batch(pdb_text, mutations_df, **kwargs)

    def calculate_binding_energy(
        self, structure_id: str, chains_a: List[str], chains_b: List[str], **kwargs
    ) -> BindingResult:
        """Calculate binding energy between two groups of chains.

        Calculates binding energy for protein-protein complexes.

        Args:
            structure_id: ID of the structure in the database.
            chains_a: Chain IDs for the first group (e.g. ``['A']``).
            chains_b: Chain IDs for the second group (e.g. ``['B']``).
            **kwargs: Forwarded to :meth:`MutationEngine.calculate_binding_energy`.

        Returns:
            BindingResult with binding energy and interface residues.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.calculate_binding_energy(
            pdb_text, chains_a, chains_b, **kwargs
        )

    def alanine_scan(
        self, structure_id: str, chain: str, positions: Optional[List[int]] = None,
        **kwargs
    ) -> pl.DataFrame:
        """Alanine scan: mutate each position to Ala and report ddG.

        Performs systematic alanine scanning mutagenesis.

        Args:
            structure_id: ID of the structure in the database.
            chain: Chain ID to scan.
            positions: Specific residue numbers. If None, scans all eligible residues.
            **kwargs: Forwarded to :meth:`MutationEngine.alanine_scan`.

        Returns:
            DataFrame with columns [chain, position, wt_residue, ddg_kcal_mol].
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.alanine_scan(pdb_text, chain, positions, **kwargs)

    def position_scan(
        self, structure_id: str, chain: str, positions: List[int], **kwargs
    ) -> pl.DataFrame:
        """Scan all 20 amino acids at specified positions.

        Generates position-specific scoring matrix.

        Args:
            structure_id: ID of the structure in the database.
            chain: Chain ID.
            positions: List of residue numbers to scan.
            **kwargs: Forwarded to :meth:`MutationEngine.position_scan`.

        Returns:
            DataFrame with columns
            [chain, position, wt_residue, mut_residue, ddg_kcal_mol].
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.position_scan(pdb_text, chain, positions, **kwargs)

    def per_residue_energy(self, structure_id: str, **kwargs) -> pl.DataFrame:
        """Approximate per-residue energy contribution via Ala-subtraction.

        Computes per-residue energy decomposition.

        Args:
            structure_id: ID of the structure in the database.
            **kwargs: Forwarded to :meth:`MutationEngine.per_residue_energy`.

        Returns:
            DataFrame with columns
            [chain, residue_number, residue_name, energy_contribution_kcal_mol].
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.per_residue_energy(pdb_text, **kwargs)

    # ------------------------------------------------------------------
    # Mutation Visualization
    # ------------------------------------------------------------------

    def plot_mutation_results(
        self,
        results_df: pl.DataFrame,
        output_file: Optional[str] = None,
        plot_type: str = "ddg",
        **kwargs
    ) -> pl.DataFrame:
        """Visualize mutation analysis results.

        Args:
            results_df: DataFrame from mutate_batch(), alanine_scan(), or position_scan()
            output_file: Path to save figure (if None, shows interactively)
            plot_type: One of "ddg", "distribution"
            **kwargs: Passed to visualization function

        Returns:
            Processed DataFrame used for plotting
        """
        if plot_type == "ddg":
            return plot_ddg(results_df, output_file, **kwargs)
        elif plot_type == "distribution":
            return plot_ddg_distribution(results_df, output_file, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Choose 'ddg' or 'distribution'")

    def plot_position_scan(
        self,
        scan_df: pl.DataFrame,
        output_file: Optional[str] = None,
        **kwargs
    ) -> pl.DataFrame:
        """Visualize position scan as heatmap.

        Args:
            scan_df: DataFrame from position_scan()
            output_file: Path to save figure (if None, shows interactively)
            **kwargs: Passed to plot_position_scan_heatmap()

        Returns:
            Pivoted DataFrame (rows=amino acids, cols=positions)
        """
        return plot_position_scan_heatmap(scan_df, output_file, **kwargs)

    def plot_alanine_scan_results(
        self,
        scan_df: pl.DataFrame,
        output_file: Optional[str] = None,
        **kwargs
    ) -> pl.DataFrame:
        """Visualize alanine scan results.

        Args:
            scan_df: DataFrame from alanine_scan()
            output_file: Path to save figure (if None, shows interactively)
            **kwargs: Passed to plot_alanine_scan()

        Returns:
            Sorted DataFrame used for plotting
        """
        return plot_alanine_scan(scan_df, output_file, **kwargs)

    def plot_energy_breakdown(
        self,
        energy_terms_df: pl.DataFrame,
        output_file: Optional[str] = None,
        **kwargs
    ) -> pl.DataFrame:
        """Visualize energy term breakdown for WT vs mutant.

        Args:
            energy_terms_df: DataFrame from MutationResult.energy_terms
            output_file: Path to save figure (if None, shows interactively)
            **kwargs: Passed to plot_energy_terms()

        Returns:
            Processed DataFrame with term contributions
        """
        return plot_energy_terms(energy_terms_df, output_file, **kwargs)

    # ------------------------------------------------------------------
    # Interface Mutagenesis (NEW)
    # ------------------------------------------------------------------

    def mutate_interface(
        self,
        structure_id: str,
        mutations: Dict[str, List[Union[Mutation, str]]],
        chains_a: List[str],
        chains_b: List[str],
        **kwargs
    ) -> InterfaceMutationResult:
        """Apply mutations to protein-protein interface and compute ΔΔG_binding.

        Args:
            structure_id: ID of the structure (complex) in the database.
            mutations: Dict mapping chain ID to list of mutations.
                      E.g., {"A": ["F13A", "W14L"], "B": ["Y25F"]}
            chains_a: Chain IDs for the first binding partner (e.g. ['A']).
            chains_b: Chain IDs for the second binding partner (e.g. ['B']).
            **kwargs: Forwarded to MutationEngine.mutate_interface().

        Returns:
            InterfaceMutationResult with ΔΔG_binding and component energies.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.mutate_interface(
            pdb_text, mutations, chains_a, chains_b, **kwargs
        )

    # ------------------------------------------------------------------
    # Disulfide Bond Analysis (NEW)
    # ------------------------------------------------------------------

    def detect_disulfides(self, structure_id: str, **kwargs) -> pl.DataFrame:
        """Detect disulfide bonds in a structure.

        Args:
            structure_id: ID of the structure in the database.
            **kwargs: Forwarded to MutationEngine.detect_disulfides().

        Returns:
            DataFrame with columns:
            [chain1, residue1, resname1, chain2, residue2, resname2, distance].
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.detect_disulfides(pdb_text, **kwargs)

    def analyze_mutation_disulfide_impact(
        self,
        structure_id: str,
        mutations: List[Union[Mutation, str]],
        **kwargs
    ) -> Dict[str, any]:
        """Analyze how mutations affect disulfide bonds.

        Args:
            structure_id: ID of the structure in the database.
            mutations: List of Mutation objects or strings (e.g. ``'C42A'``).
            **kwargs: Forwarded to MutationEngine.analyze_mutation_disulfide_impact().

        Returns:
            Dict with wt_disulfides, mutant_disulfides, broken_bonds, new_bonds.
        """
        pdb_text = self._structure_to_pdb(structure_id)
        return self.mutation_engine.analyze_mutation_disulfide_impact(
            pdb_text, mutations, **kwargs
        )

    # ------------------------------------------------------------------
    # Residue Interaction Networks (NEW)
    # ------------------------------------------------------------------

    def compute_interaction_network(
        self,
        structure_id: str,
        distance_cutoff: float = 5.0,
        interaction_types: Optional[List[str]] = None,
    ):
        """Compute residue interaction network for a structure.

        Args:
            structure_id: ID of the structure in the database.
            distance_cutoff: Maximum distance (Å) for residue contact (default 5.0).
            interaction_types: Optional filter for specific residues.

        Returns:
            NetworkX graph with residue nodes and interaction edges.
        """
        structure_df = self.get_structure(structure_id).collect()
        return self.toolkit.compute_residue_interaction_network(
            structure_df, distance_cutoff, interaction_types
        )

    def analyze_network_centrality(self, G, top_n: int = 10) -> pl.DataFrame:
        """Analyze network centrality metrics to identify key residues.

        Args:
            G: NetworkX graph from compute_interaction_network()
            top_n: Number of top residues to return (default 10)

        Returns:
            DataFrame with centrality metrics.
        """
        return self.toolkit.analyze_network_centrality(G, top_n)

    def plot_interaction_network(self, G, output_file: Optional[str] = None, **kwargs):
        """Visualize residue interaction network.

        Args:
            G: NetworkX graph from compute_interaction_network()
            output_file: Path to save figure (if None, shows interactively)
            **kwargs: Passed to toolkit.plot_interaction_network()
        """
        return self.toolkit.plot_interaction_network(G, output_file, **kwargs)
