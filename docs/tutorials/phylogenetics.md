# Structural Phylogenetics & Clustering

Sicifus provides two complementary approaches for analyzing structural relationships:

1. **Fast clustering** (`cluster()`) — greedy centroid-based clustering with k-mer prefiltering. Scales to thousands of structures in seconds. No full distance matrix needed.
2. **Phylogenetic tree** (`generate_tree()`) — full all-vs-all RMSD matrix and neighbor-joining tree. Slower but gives branch lengths and full topology.

Both approaches use a **3Di-like 20-state structural alphabet** and **k-mer prefiltering** (inspired by Foldseek) to avoid expensive alignments between dissimilar structures.

---

## Fast Clustering (Recommended for Large Datasets)

The `cluster()` method uses a greedy centroid-based algorithm (similar to MMseqs2 linclust):

1. Encode each structure into a 20-state structural alphabet
2. Build a k-mer inverted index
3. For each structure (longest first), use k-mer overlap to find candidate centroids
4. Compute RMSD only to candidate centroids
5. Assign to the nearest centroid within threshold, or become a new centroid

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")
db.load()

# Cluster all structures (default threshold: 2.0 Å RMSD)
df = db.cluster()
print(df)
# shape: (5000, 4)
# ┌──────────────┬─────────┬─────────────┬──────────────────┐
# │ structure_id  ┆ cluster ┆ centroid_id ┆ rmsd_to_centroid │
# ╞══════════════╪═════════╪═════════════╪══════════════════╡
# │ 1abc          ┆ 1       ┆ 1xyz        ┆ 0.82             │
# │ 2def          ┆ 1       ┆ 1xyz        ┆ 1.15             │
# │ 3ghi          ┆ 2       ┆ 3ghi        ┆ 0.00             │
# │ ...           ┆ ...     ┆ ...         ┆ ...              │
# └──────────────┴─────────┴─────────────┴──────────────────┘

# Tighter clustering (smaller threshold = more, smaller clusters)
df = db.cluster(distance_threshold=1.0)

# Coarser clustering
df = db.cluster(distance_threshold=5.0)

# Save a cluster size plot
df = db.cluster(distance_threshold=2.0, output_file="clusters.png")
```

### Cluster Parameters

| Parameter | Default | Description |
|---|---|---|
| `distance_threshold` | 2.0 | Max RMSD (Å) for assigning to a centroid |
| `coverage_threshold` | 0.8 | Min length-ratio for comparing two structures |
| `structure_ids` | None | Specific structures to cluster (default: all) |

### Performance

| Dataset Size | Approximate Time |
|---|---|
| 1,000 structures | ~10 sec |
| 5,000 structures | ~30-60 sec |
| 10,000 structures | ~2-4 min |

---

## Generating a Phylogenetic Tree

The `generate_tree` method computes an all-vs-all RMSD matrix and constructs a tree. The k-mer prefilter is enabled by default, which dramatically reduces the number of pairwise alignments needed.

```python
# Generate a circular (unrooted) tree — the default layout
db.generate_tree(output_file="structural_tree.png")

# Generate a rectangular dendrogram instead
db.generate_tree(output_file="tree_rect.png", layout="rectangular")

# Generate a tree rooted at a specific structure
db.generate_tree(output_file="rooted_tree.png", root_id="1abc")

# Export to Newick format for iTOL or other visualization tools
db.generate_tree(newick_file="tree.nwk")
```

### K-mer Prefiltering

The prefilter is enabled by default. It encodes each structure into a 20-state structural alphabet (3Di-like), builds a k-mer inverted index, and only runs the expensive Needleman-Wunsch alignment + Kabsch superposition on pairs that share enough k-mers.

Typically only 1-5% of pairs need full alignment, giving a **20-100x speedup**.

You can disable prefiltering if you want the exact all-vs-all matrix:

```python
# With prefilter (default, fast)
db.generate_tree(output_file="tree.png")

# Without prefilter (exact, slower)
db.generate_tree(output_file="tree_exact.png", prefilter=False)
```

!!! note
    Pairs that fail the prefilter are assigned a high RMSD (99.9 Å) in the distance
    matrix. This means they will appear on long branches in the tree, which is the
    correct behavior — dissimilar structures should be far apart.

### Length-ratio Pruning

For datasets with very different structure sizes, you can additionally skip alignment for pairs whose length ratio is below a threshold:

```python
# Skip alignment if length ratio < 0.8
db.generate_tree(pruning_threshold=0.8)
```

### Tree Performance

| Dataset Size | Without Prefilter | With Prefilter (default) |
|---|---|---|
| 1,000 structures | ~7 min | ~20 sec |
| 5,000 structures | ~1.7 hr | ~2-5 min |

---

## Clustering from the Tree

Clustering from the tree is a separate step from tree generation. First build the tree (the expensive part), then inspect branch lengths and annotate clusters cheaply — try as many thresholds as you want without recomputing the tree.

Clusters are derived directly from the tree's branch lengths (RMSD). You provide a `distance_threshold` — any branch longer than that is cut, and each resulting subtree becomes a cluster.

```python
# Step 1: Generate the tree
db.generate_tree(output_file="tree.png", newick_file="tree.nwk")

# Step 2: Inspect branch lengths to pick a good threshold
db.tree_stats()
# Tree branch length statistics (1987 branches):
#   min:    0.0012
#   25th:   0.2100
#   50th:   0.5321
#   75th:   1.2450
#   90th:   2.1800
#   max:    6.3100

# Step 3: Annotate clusters (instant — just walks the tree)
db.annotate_clusters(distance_threshold=2.0)

# Re-plot with cluster colors
db.annotate_clusters(distance_threshold=2.0, output_file="tree_clustered.png")

# Try different thresholds cheaply
db.annotate_clusters(distance_threshold=0.5)   # finer — more clusters
db.annotate_clusters(distance_threshold=3.0)   # coarser — fewer clusters
```

---

## When to Use Which

| Use Case | Method | Why |
|---|---|---|
| Quick structural grouping | `cluster()` | No full matrix, scales well |
| Browsing cluster assignments | `cluster()` | Immediate results with centroids |
| Publication-quality phylogeny | `generate_tree()` | Full topology + branch lengths |
| Newick export for iTOL | `generate_tree()` | Standard tree format |
| Exploring thresholds interactively | `generate_tree()` + `annotate_clusters()` | Re-annotate without recomputing |
| > 5,000 structures | `cluster()` first, then `generate_tree()` on a subset | Cluster fast, tree on interesting groups |

---

## Accessing Clusters

Both `cluster()` and `annotate_clusters()` populate the same cluster state, so the querying API works with either:

```python
# Access the cluster DataFrame (structure_id, cluster)
print(db.clusters)
print(db.cluster_summary())

# Get all structures in cluster 5
cluster_5 = db.get_cluster(5)
print(f"Cluster 5 has {len(cluster_5)} structures: {cluster_5[:5]}...")

# What cluster is "1abc" in?
cid = db.get_cluster_for("1abc")
print(f"1abc is in cluster {cid}")

# Get all structures in the same cluster as "1abc"
siblings = db.get_cluster_siblings("1abc")
print(f"{len(siblings)} structures share a cluster with 1abc")

# Combine with Polars — e.g. pull backbone data for an entire cluster
cluster_ids = db.get_cluster(3)
cluster_data = db.backbone.filter(pl.col("structure_id").is_in(cluster_ids)).collect()
```

---

## How the 3Di-like Alphabet Works

Each CA residue is assigned one of 20 structural states based on the local backbone geometry:

- **theta** — the virtual bond angle CA(i-1)–CA(i)–CA(i+1) (captures helix vs sheet vs extended)
- **tau** — the pseudo-dihedral CA(i-2)–CA(i-1)–CA(i)–CA(i+1) (captures handedness and torsion)

These are discretized into a 4x5 = 20-state grid. Two structures with similar local geometry will produce similar state sequences, and therefore share many k-mers — which is what the prefilter detects.

This is conceptually similar to Foldseek's 3Di alphabet, but implemented entirely in-house with Numba JIT — no external binaries required.
