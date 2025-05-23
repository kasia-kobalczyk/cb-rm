import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.datasets import make_spd_matrix
from tqdm import tqdm

# Configuration
N_PROMPTS = 50000 #500000
RESPONSES_PER_PROMPT = 2
P = 4096
K = 10
SEED = 42
np.random.seed(SEED)

SAVE_DIR = "./datasets/synthetic_cbm_data_small/"
os.makedirs(SAVE_DIR, exist_ok=True)
concept_names = [f"concept_{i}" for i in range(K)]

def synthetic_concept_mlp(x, W1, W2, b1, b2):
    h = np.tanh(x @ W1 + b1)           # Nonlinearity
    out = h @ W2 + b2
    return 1 / (1 + np.exp(-out))     # Soft concept vector

# Projections and weights
W_proj_prompt = np.random.randn(P, P)
W_proj_response = np.random.randn(P, P)
W_concepts = np.random.randn(P, K)
W_gating = np.random.randn(P, K)  # For gated weighting

HIDDEN_DIM = 256  # You can change this

W1 = np.random.randn(P, HIDDEN_DIM) * 0.1
W2 = np.random.randn(HIDDEN_DIM, K) * 0.1
b1 = np.zeros(HIDDEN_DIM)
b2 = np.zeros(K)


# Step 1: Create synthetic prompt & response embeddings
records = []
meta = []  # store per-response metadata
idx_counter = 0

for prompt_id in tqdm(range(N_PROMPTS), desc="Generating embeddings"):
    z = np.random.multivariate_normal(np.zeros(P), make_spd_matrix(P))
    prompt_emb = z @ W_proj_prompt

    for _ in range(RESPONSES_PER_PROMPT):
        noise = np.random.normal(scale=2.0, size=P)
        resp_emb = (z + noise) @ W_proj_response

        records.append({
            "idx": idx_counter,
            "prompt": "",
            "response": "",
            "prompt_embedding": prompt_emb.tolist(),
            "prompt_response_embedding": resp_emb.tolist(),
            "success": True,
        })

        meta.append({
            "prompt_id": prompt_id,
            "idx": idx_counter,
            "prompt_embedding": prompt_emb,
            "response_embedding": resp_emb,
        })

        idx_counter += 1

Dataset.from_list(records).save_to_disk(os.path.join(SAVE_DIR, "embeddings"))

# Step 2: Build labels for each mechanism
pair_rows = list()
mechanisms = ["complete", "incomplete", "gated"]
label_outputs = {m: {"concept_labels": [], "preference_labels": []} for m in mechanisms}

for prompt_id in tqdm(range(N_PROMPTS), desc="Generating labels"):
    group = [m for m in meta if m["prompt_id"] == prompt_id]
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            idx_a, idx_b = a["idx"], b["idx"]
            xa, xb = a["response_embedding"], b["response_embedding"]
            pa = a["prompt_embedding"]

            # Core concept generation
            # c_a = (np.array(xa) @ W_concepts >= 0).astype(float)
            # c_b = (np.array(xb) @ W_concepts >= 0).astype(float)
            c_a = synthetic_concept_mlp(np.array(xa), W1, W2, b1, b2)
            c_b = synthetic_concept_mlp(np.array(xb), W1, W2, b1, b2)
            rel = 1 - ((c_a - c_b) +1)/2

            # Save base pair info
            pair_rows.append({"idx_a": idx_a, "idx_b": idx_b, "split": "train"})

            # ----------- Complete
            label_outputs["complete"]["concept_labels"].append({
                "idx_a": idx_a, "idx_b": idx_b,
                "relative_concept_labels": rel.tolist(),
                "concept_names": concept_names,
            })
            reward_a = np.sum(c_a)
            reward_b = np.sum(c_b)
            label_outputs["complete"]["preference_labels"].append({
                "idx_a": idx_a, "idx_b": idx_b,
                "preference_label": float(reward_b > reward_a),
            })

            # ----------- Incomplete (masking some concepts)
            mask = np.random.rand(K) > 0.3
            c_a_masked = c_a * mask
            c_b_masked = c_b * mask
            visible_indices = np.where(mask)[0]
            rel_visible = 1 - (c_a[mask] - c_b[mask] + 1) / 2
            
            label_outputs["incomplete"]["concept_labels"].append({
                "idx_a": idx_a, "idx_b": idx_b,
                "relative_concept_labels": rel_visible.tolist(),
                "concept_names": [concept_names[i] for i in visible_indices],
            })
            reward_a_masked = np.sum(c_a)
            reward_b_masked = np.sum(c_b)
            label_outputs["incomplete"]["preference_labels"].append({
                "idx_a": idx_a, "idx_b": idx_b,
                "preference_label": float(reward_b_masked > reward_a_masked),
            })

            # ----------- Gated (per-prompt weights)
            pa_norm = pa / np.linalg.norm(pa)
            w_gate = np.tanh((pa_norm @ W_gating) / 2.0)
            reward_a_gated = np.sum(w_gate * c_a)
            reward_b_gated = np.sum(w_gate * c_b)

            label_outputs["gated"]["concept_labels"].append({
                "idx_a": idx_a, "idx_b": idx_b,
                "relative_concept_labels": rel.tolist(),
                "concept_names": concept_names,
            })
            label_outputs["gated"]["preference_labels"].append({
                "idx_a": idx_a, "idx_b": idx_b,
                "preference_label": float(reward_b_gated > reward_a_gated),
            })

# Step 3: Save all outputs

for mech in mechanisms:
    Dataset.from_pandas(pd.DataFrame(label_outputs[mech]["concept_labels"])).save_to_disk(
        os.path.join(SAVE_DIR, f"concept_labels_{mech}")
    )
    Dataset.from_pandas(pd.DataFrame(label_outputs[mech]["preference_labels"])).save_to_disk(
        os.path.join(SAVE_DIR, f"preference_labels_{mech}")
    )

# After you’ve created all the pair_rows:
splits_df = pd.DataFrame(pair_rows)[["idx_a", "idx_b"]].copy()
splits_df["split"] = np.random.choice(
    ["train", "val", "test"],
    size=len(splits_df),
    p=[0.7, 0.1, 0.2]
)

splits_df.to_csv(os.path.join(SAVE_DIR, "splits.csv"), index=False)