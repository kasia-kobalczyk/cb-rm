from datasets import load_from_disk, concatenate_datasets

base = "./datasets/synthetic_cbm_data_new"
shards = [f"shard_{i}" for i in range(4)]

def merge_and_save(name):
    parts = [load_from_disk(f"{base}/{shard}/{name}") for shard in shards]
    full = concatenate_datasets(parts)
    full.save_to_disk(f"{base}/{name}")

# Merge each dataset
for name in [
    "embeddings",
    "concept_labels_complete",
    "concept_labels_incomplete",
    "concept_labels_gated",
    "preference_labels_complete",
    "preference_labels_incomplete",
    "preference_labels_gated",
]:
    merge_and_save(name)

# Merging splits.csv
import pandas as pd
dfs = [pd.read_csv(f"{base}/{shard}/splits.csv") for shard in shards]
pd.concat(dfs, ignore_index=True).to_csv(f"{base}/splits.csv", index=False)
