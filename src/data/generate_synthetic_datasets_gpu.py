import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import random
from collections import defaultdict
import itertools

# Configuration
N_PROMPTS = 500000
RESPONSES_PER_PROMPT = 2
P = 4096
K = 10
BATCH_SIZE = 100
SEED = 42
torch.manual_seed(SEED)

SAVE_DIR = "./datasets/synthetic_cbm_data/"
os.makedirs(SAVE_DIR, exist_ok=True)
concept_names = [f"concept_{i}" for i in range(K)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define synthetic MLP
def synthetic_concept_mlp(x, W1, W2, b1, b2, temp=1.5):
    h = torch.tanh(x @ W1 + b1)
    out = h @ W2 + b2
    return torch.sigmoid(out/temp)

# Initialize weights
W_proj_prompt = torch.randn(P, P, device=device)
W_proj_response = torch.randn(P, P, device=device)
W_concepts = torch.randn(P, K, device=device)
W_gating = torch.randn(P, K, device=device)

HIDDEN_DIM = 256
W1 = torch.randn(P, HIDDEN_DIM, device=device) * 0.1
W2 = torch.randn(HIDDEN_DIM, K, device=device) * 0.5
b1 = torch.zeros(HIDDEN_DIM, device=device)
b2 = torch.zeros(K, device=device)

# Create reusable SPD covariance matrices
COV_POOL = [
    torch.distributions.MultivariateNormal(
        torch.zeros(P, device=device),
        covariance_matrix=(
            (A := torch.randn(P, P, device=device)) @ A.T / torch.norm(A @ A.T) + torch.eye(P, device=device) * 1e-3
        )
    )
    for _ in range(100)
]

records = []
meta = []
idx_counter = 0

num_batches = N_PROMPTS // BATCH_SIZE
if N_PROMPTS % BATCH_SIZE != 0:
    num_batches += 1

# Step 1: Generate embeddings
for batch_id in tqdm(range(num_batches), desc="Generating embeddings"):
    current_batch_size = min(BATCH_SIZE, N_PROMPTS - (batch_id * BATCH_SIZE))

    z_batch = random.choice(COV_POOL).sample((current_batch_size,))
    prompt_emb_batch = z_batch @ W_proj_prompt.T

    for i in range(current_batch_size):
        z = z_batch[i]
        prompt_emb = prompt_emb_batch[i]

        for _ in range(RESPONSES_PER_PROMPT):
            noise = torch.randn(P, device=device) * 1.0
            resp_emb = (z + noise) @ W_proj_response

            records.append({
                "idx": idx_counter,
                "prompt": "",
                "response": "",
                "prompt_embedding": prompt_emb.cpu().tolist(),
                "prompt_response_embedding": resp_emb.cpu().tolist(),
                "success": True,
            })

            meta.append({
                "prompt_id": batch_id * BATCH_SIZE + i,
                "idx": idx_counter,
                "prompt_embedding": prompt_emb.clone().cpu(),
                "response_embedding": resp_emb.clone().cpu(),
            })

            idx_counter += 1

Dataset.from_list(records).save_to_disk(os.path.join(SAVE_DIR, "embeddings"))

# Step 2: Generate labels
pair_rows = []
mechanisms = ["complete", "incomplete", "gated"]
label_outputs = {m: {"concept_labels": [], "preference_labels": []} for m in mechanisms}

# Organize meta by prompt_id

grouped_meta = defaultdict(list)
for m in meta:
    grouped_meta[m["prompt_id"]].append(m)

for prompt_id in tqdm(range(N_PROMPTS), desc="Generating labels"):
    group = grouped_meta[prompt_id]

    if len(group) < 2:
        continue

    # Extract all embeddings for this group
    idx_list = [m["idx"] for m in group]
    prompt_embeddings = torch.stack([m["prompt_embedding"] for m in group]).to(device)
    response_embeddings = torch.stack([m["response_embedding"] for m in group]).to(device)

    # Build all pair indices
    pairs = list(itertools.combinations(range(len(group)), 2))
    idx_a_all, idx_b_all = zip(*pairs)

    idx_a_all = torch.tensor(idx_a_all, device=device)
    idx_b_all = torch.tensor(idx_b_all, device=device)

    xa = response_embeddings[idx_a_all]
    xb = response_embeddings[idx_b_all]
    pa = prompt_embeddings[idx_a_all]  # one per pair

    # Run MLP in batch
    c_a = synthetic_concept_mlp(xa, W1, W2, b1, b2)
    c_b = synthetic_concept_mlp(xb, W1, W2, b1, b2)

    rel = 1 - ((c_a - c_b) + 1) / 2

    for n in range(len(pairs)):
        idx_a = idx_list[idx_a_all[n].item()]
        idx_b = idx_list[idx_b_all[n].item()]
        pair_rows.append({"idx_a": idx_a, "idx_b": idx_b, "split": "train"})

        # Complete
        label_outputs["complete"]["concept_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "relative_concept_labels": rel[n].cpu().tolist(),
            "concept_names": concept_names,
        })
        label_outputs["complete"]["preference_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "preference_label": float(torch.sum(c_b[n]) > torch.sum(c_a[n])),
        })

        # Incomplete
        mask = torch.rand(K, device=device) > 0.3
        rel_visible = 1 - ((c_a[n][mask] - c_b[n][mask]) + 1) / 2
        label_outputs["incomplete"]["concept_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "relative_concept_labels": rel_visible.cpu().tolist(),
            "concept_names": [concept_names[i] for i in mask.nonzero(as_tuple=True)[0].tolist()],
        })
        reward_a_masked = torch.sum(c_a[n][mask])
        reward_b_masked = torch.sum(c_b[n][mask])
        label_outputs["incomplete"]["preference_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "preference_label": float(reward_b_masked > reward_a_masked),
        })

        # Gated
        pa_norm = pa[n] / pa[n].norm()
        w_gate = torch.tanh((pa_norm @ W_gating) / 2.0)
        reward_a_gated = torch.sum(w_gate * c_a[n])
        reward_b_gated = torch.sum(w_gate * c_b[n])
        label_outputs["gated"]["concept_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "relative_concept_labels": rel[n].cpu().tolist(),
            "concept_names": concept_names,
        })
        label_outputs["gated"]["preference_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "preference_label": float(reward_b_gated > reward_a_gated),
        })

# Save labels and splits
for mech in mechanisms:
    Dataset.from_pandas(pd.DataFrame(label_outputs[mech]["concept_labels"])).save_to_disk(
        os.path.join(SAVE_DIR, f"concept_labels_{mech}")
    )
    Dataset.from_pandas(pd.DataFrame(label_outputs[mech]["preference_labels"])).save_to_disk(
        os.path.join(SAVE_DIR, f"preference_labels_{mech}")
    )

splits_df = pd.DataFrame(pair_rows)[["idx_a", "idx_b"]].copy()
splits_df["split"] = np.random.choice(["train", "val", "test"], size=len(splits_df), p=[0.7, 0.1, 0.2])
splits_df.to_csv(os.path.join(SAVE_DIR, "splits.csv"), index=False)

