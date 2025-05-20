import os
import torch
import numpy as np
import datasets
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
from datasets import Dataset
import pandas as pd

# CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Multi-objective reward attributes
attributes = [
    "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence", "helpsteer-complexity", "helpsteer-verbosity",
    "ultrafeedback-overall_score", "ultrafeedback-instruction_following", "ultrafeedback-truthfulness", "ultrafeedback-honesty", "ultrafeedback-helpfulness",
    "beavertails-is_safe", "prometheus-score",
    "argilla-overall_quality", "argilla-judge_lm",
    "code-complexity", "code-style", "code-explanation", "code-instruction-following", "code-readability"
]

token_patterns = {
    "llama3": [128009, 128006, 78191, 128007, 271],
    "gemma2": [107, 108, 106, 2516, 108],
}

def find_token_for_gating(lst, model_family):
    token_pattern = token_patterns[model_family]
    for j in range(len(lst) - len(token_pattern), -1, -1):
        if lst[j: j + len(token_pattern)] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

# ------------------------ ARGUMENTS ------------------------
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
parser.add_argument("--dataset_path", type=str, default="RLHFlow/ArmoRM-Multi-Objective-Data-v0.1")
parser.add_argument("--n_shards", type=int, default=1)
parser.add_argument("--shard_idx", type=int, default=1)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--model_family", type=str, default="llama3")
args = parser.parse_args()

# ------------------------ SAVE PATH ------------------------
HOME = "/mnt/pdata/sonia111/ArmoRM"
model_name = args.model_path.split("/")[-1]
dataset_name = args.dataset_path.split("/")[-1]
# save_path = os.path.join(HOME, "hf_dataset", model_name, dataset_name)
save_path = f"/mnt/pdata/knk25/active_pref_learning/datasets/armo-rm/embeddings/FsfairX-LLaMA3-RM-v0.1_full"
os.makedirs(save_path, exist_ok=True)

# ------------------------ LOAD DATASET ------------------------
ds = datasets.load_dataset(args.dataset_path)["train"]
ds = ds.shuffle(seed=0)
if args.n_shards > 1:
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

# ------------------------ LOAD MODEL ------------------------
device = f"cuda:{args.device}"
model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# ------------------------ INITIALIZE STORAGE ------------------------
embeddings = []
prompt_embeddings = []
labels = []
prompts = []
responses = []
success_flags = []
embedding_idx = []

# ------------------------ PROCESS EXAMPLES ------------------------
for idx, example in enumerate(tqdm(ds, desc="Processing dataset")):
    try:
        conv = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        tokens = tokenizer(conv, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            output = model(**tokens).last_hidden_state[0]
            gate_idx = find_token_for_gating(input_ids[0].tolist(), args.model_family)
            prompt_emb = output[gate_idx].cpu()
            full_emb = output[-1].cpu()
        success = True
    except Exception:
        prompt_emb = torch.zeros_like(output[-1]).cpu()
        full_emb = output[-1].cpu()
        success = False

    # Extract reward attributes
    label = [np.nan if example[attr] is None else example[attr] for attr in attributes]

    # Append everything
    prompt_embeddings.append(prompt_emb)
    embeddings.append(full_emb)
    labels.append(label)
    prompts.append(example["messages"][0]["content"])
    responses.append(example["messages"][-1]["content"])
    success_flags.append(success)
    embedding_idx.append(idx)

# ------------------------ SAVE TO HUGGINGFACE DATASET ------------------------
df = pd.DataFrame({
    "idx": embedding_idx,
    "prompt": prompts,
    "response": responses,
    "success": success_flags,
    "labels": labels,
    "embedding": torch.stack(embeddings).to(torch.float32).numpy().tolist(), # Should be prompt_response_embedding
    "prompt_embedding": torch.stack(prompt_embeddings).to(torch.float32).numpy().tolist(),
})

hf_dataset = Dataset.from_pandas(df)
hf_dataset.save_to_disk(save_path)

print(f"✅ Saved Hugging Face dataset to: {save_path}")
