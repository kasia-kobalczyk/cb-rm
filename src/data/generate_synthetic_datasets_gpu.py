# Run with 
# Use physical GPU 0
# python src/data/generate_synthetic_datasets_gpu.py --start_idx 0       --end_idx 125000 --gpu 0

# Use physical GPU 1
# python src/data/generate_synthetic_datasets_gpu.py --start_idx 125000  --end_idx 250000 --gpu 1

# Use physical GPU 2
# python src/data/generate_synthetic_datasets_gpu.py --start_idx 250000  --end_idx 375000 --gpu 2

# Use physical GPU 3
# python src/data/generate_synthetic_datasets_gpu.py --start_idx 375000  --end_idx 500000 --gpu 3

import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import random
from collections import defaultdict
import itertools
import argparse

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, required=True)
parser.add_argument("--end_idx", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--save_dir", type=str, default="./datasets/synthetic_cbm_data")
args = parser.parse_args()

# ----------------------
# Setup
# ----------------------
torch.cuda.set_device(args.gpu)
device = torch.device("cuda")

START_IDX = args.start_idx
END_IDX = args.end_idx
RESPONSES_PER_PROMPT = 2
N_PROMPTS = (END_IDX - START_IDX) // RESPONSES_PER_PROMPT
P = 4096
K = 10
BATCH_SIZE = 100
SEED = 42 + args.gpu  
HIDDEN_DIM = 256
K_incomplete = 4 


random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

incomplete_mask = torch.zeros(K, dtype=torch.bool, device=device)
incomplete_indices = torch.randperm(K, device=device)[:K_incomplete]
incomplete_mask[incomplete_indices] = True

SAVE_PATH = os.path.join(args.save_dir, f"shard_{args.gpu}")
os.makedirs(SAVE_PATH, exist_ok=True)
concept_names = [f"concept_{i}" for i in range(K)]

# ----------------------
# Define MLP & Weights
# ----------------------
def synthetic_concept_mlp(x, W1, W2, b1, b2, temp=1.5):
    h = torch.tanh(x @ W1 + b1)
    out = h @ W2 + b2
    return torch.sigmoid(out / temp)

W_proj_prompt = torch.randn(P, P, device=device)
W_proj_response = torch.randn(P, P, device=device)
W_concepts = torch.randn(P, K, device=device)
W_gating = torch.randn(P, K, device=device)
W1 = torch.randn(P, HIDDEN_DIM, device=device) * 0.1
W2 = torch.randn(HIDDEN_DIM, K, device=device) * 0.5
b1 = torch.zeros(HIDDEN_DIM, device=device)
b2 = torch.zeros(K, device=device)

# ----------------------
# Covariance Pool
# ----------------------
COV_POOL = [
    torch.distributions.MultivariateNormal(
        torch.zeros(P, device=device),
        covariance_matrix=((A := torch.randn(P, P, device=device)) @ A.T / torch.norm(A @ A.T) + torch.eye(P, device=device) * 1e-3)
    )
    for _ in range(100)
]

# ----------------------
# Generate Embeddings
# ----------------------
records = []
meta = []
idx_counter = START_IDX

num_batches = (N_PROMPTS + BATCH_SIZE - 1) // BATCH_SIZE

for batch_id in tqdm(range(num_batches), desc="Generating embeddings"):
    current_batch_size = min(BATCH_SIZE, N_PROMPTS - batch_id * BATCH_SIZE)
    z_batch = random.choice(COV_POOL).sample((current_batch_size,))
    prompt_emb_batch = z_batch @ W_proj_prompt.T

    for i in range(current_batch_size):
        z = z_batch[i]
        prompt_emb = prompt_emb_batch[i]

        for _ in range(RESPONSES_PER_PROMPT):
            noise = torch.randn(P, device=device)
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
                "prompt_id": batch_id * BATCH_SIZE + i + START_IDX,
                "idx": idx_counter,
                "prompt_embedding": prompt_emb.clone().cpu(),
                "response_embedding": resp_emb.clone().cpu(),
            })

            idx_counter += 1

Dataset.from_list(records).save_to_disk(os.path.join(SAVE_PATH, "embeddings"))

# ----------------------
# Generate Labels
# ----------------------
pair_rows = []
mechanisms = ["complete", "incomplete", "gated"]
label_outputs = {m: {"concept_labels": [], "preference_labels": []} for m in mechanisms}

# Group by prompt
grouped_meta = defaultdict(list)
for m in meta:
    grouped_meta[m["prompt_id"]].append(m)

for prompt_id in tqdm(range(START_IDX, END_IDX), desc="Generating labels"):
    group = grouped_meta[prompt_id]
    if len(group) < 2:
        continue

    idx_list = [m["idx"] for m in group]
    prompt_embeddings = torch.stack([m["prompt_embedding"] for m in group]).to(device)
    response_embeddings = torch.stack([m["response_embedding"] for m in group]).to(device)

    pairs = list(itertools.combinations(range(len(group)), 2))
    idx_a_all, idx_b_all = zip(*pairs)
    idx_a_all = torch.tensor(idx_a_all, device=device)
    idx_b_all = torch.tensor(idx_b_all, device=device)

    xa = response_embeddings[idx_a_all]
    xb = response_embeddings[idx_b_all]
    pa = prompt_embeddings[idx_a_all]

    c_a = synthetic_concept_mlp(xa, W1, W2, b1, b2)
    c_b = synthetic_concept_mlp(xb, W1, W2, b1, b2)
    rel = 1 - ((c_a - c_b) + 1) / 2

    for n in range(len(pairs)):
        idx_a = idx_list[idx_a_all[n].item()]
        idx_b = idx_list[idx_b_all[n].item()]
        pair_rows.append({"idx_a": idx_a, "idx_b": idx_b, "split": "train"})

        label_outputs["complete"]["concept_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "relative_concept_labels": rel[n].cpu().tolist(),
            "concept_names": concept_names,
        })
        label_outputs["complete"]["preference_labels"].append({
            "idx_a": idx_a, "idx_b": idx_b,
            "preference_label": float(torch.sum(c_b[n]) > torch.sum(c_a[n])),
        })

        mask = incomplete_mask
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

# ----------------------
# Save Outputs
# ----------------------
for mech in mechanisms:
    Dataset.from_pandas(pd.DataFrame(label_outputs[mech]["concept_labels"]))\
        .save_to_disk(os.path.join(SAVE_PATH, f"concept_labels_{mech}"))

    Dataset.from_pandas(pd.DataFrame(label_outputs[mech]["preference_labels"]))\
        .save_to_disk(os.path.join(SAVE_PATH, f"preference_labels_{mech}"))

splits_df = pd.DataFrame(pair_rows)[["idx_a", "idx_b"]].copy()
splits_df["split"] = np.random.choice(["train", "val", "test"], size=len(splits_df), p=[0.7, 0.1, 0.2])
splits_df.to_csv(os.path.join(SAVE_PATH, "splits.csv"), index=False)

print(f"Shard {args.gpu} complete. Data saved to {SAVE_PATH}")
