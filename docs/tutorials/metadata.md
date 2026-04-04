# Metadata Tutorial

Sicifus allows you to integrate external metadata (e.g., experimental conditions, sequence properties, etc.) with your structural data.

## Loading Metadata

Load metadata from a CSV file. The file must contain an `id` column that matches the structure file names (minus the extension).

```python
from sicifus import Sicifus

db = Sicifus(db_path="./my_db")

# Load a metadata CSV — id column is matched to structure_id
db.load_metadata("path/to/summaries.csv")
# Loaded metadata 'summaries': 994 rows, 18 columns
#   994/994 rows match structures in the database
```

## Exploring Metadata

Once loaded, metadata is available via `db.meta` (LazyFrame) or `db.meta_columns()`.

```python
# See what columns are available
print(db.meta_columns())
# ['clash_score', 'radius_of_gyration', 'protein_length', ...]

# Quick histogram of any column
db.hist("radius_of_gyration")
db.hist("protein_length", bins=50)

# Scatter plot
db.scatter("protein_length", "radius_of_gyration")
```

## Combining with Clustering

You can color-code plots by cluster assignment to see if structural clusters correlate with metadata properties.

```python
# Color by cluster (after annotate_clusters)
db.hist("radius_of_gyration", color_by="cluster")
db.scatter("protein_length", "radius_of_gyration", color_by="cluster")
```

## Custom Queries with Polars

You can access the raw metadata as a Polars LazyFrame for custom queries.

```python
import polars as pl

# Access raw metadata
meta_df = db.meta.collect()
print(meta_df.describe())

# Join metadata with backbone data
combined = db.backbone.join(db.meta, on="structure_id").collect()
```
