import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import itertools

# Step 1: Load and shuffle ArmoRM dataset as in your embedding script
ds = load_dataset("RLHFlow/ArmoRM-Multi-Objective-Data-v0.1")["train"]
ds = ds.shuffle(seed=0)
df = ds.to_pandas()
df["embedding_idx"] = list(range(len(df)))  # aligns with saved embeddings

# Step 2: Group by prompt (we assume "messages[0].content" holds the prompt)
def extract_prompt(messages):
    if messages[0]["role"] != 'user':
        print('malll')
    return messages[0]["content"]

df["prompt"] = df["messages"].apply(extract_prompt)
# df = df.dropna(subset=["prompt"])  # drop rows with invalid prompt

# Step 3: Select label columns (reward objectives)
reward_cols = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence", "helpsteer-complexity", "helpsteer-verbosity",
    "ultrafeedback-overall_score", "ultrafeedback-instruction_following", "ultrafeedback-truthfulness", "ultrafeedback-honesty", "ultrafeedback-helpfulness",
    "beavertails-is_safe", "prometheus-score",
    "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation", "code-instruction-following", "code-readability"
]

# Step 4: Build pairs for each group of matching prompts
pair_meta = []
raw_diffs = []
for prompt, group in tqdm(df.groupby("prompt"), desc="Building pairs"):
    examples = list(group[["embedding_idx"] + reward_cols].itertuples(index=False, name=None))
    if len(examples) < 2:
        continue
    for (i1, *scores1), (i2, *scores2) in itertools.combinations(examples, 2):
        s1 = np.array(scores1)
        s2 = np.array(scores2)
        diff = s1 - s2
        raw_diffs.append(diff)
        pair_meta.append((i1, i2, s1, s2))
        

raw_diffs = np.stack(raw_diffs)
min_val = np.nanmin(raw_diffs)
max_val = np.nanmax(raw_diffs)
denom = max_val - min_val

# Final pass: normalize and assemble pairs
pairs = []
for (i1, i2, s1, s2), diff in zip(pair_meta, raw_diffs):
    mask = (~np.isnan(s1)) & (~np.isnan(s2))
    rel = -1.0 * np.ones_like(diff)
    rel[mask] = 1.0 - (diff[mask] - min_val) / denom

    pref = 1 if rel[mask].mean() > 0.5 else 0


    pairs.append({
        "idx_a": i1,
        "idx_b": i2,
        "preference_label": pref,
        "relative_concept_labels": rel,
    })

# Step 5: Save as HuggingFace dataset
from datasets import Dataset
pairs_df = pd.DataFrame(pairs)
hf_dataset = Dataset.from_pandas(pairs_df)
hf_dataset.save_to_disk("/mnt/pdata/sonia111/ArmoRM/embedding_all_prompt/armo_pairs_pref_concepts")

