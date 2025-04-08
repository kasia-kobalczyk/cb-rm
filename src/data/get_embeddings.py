import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser



def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern = TOKEN_PATTERNS[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

TOKEN_PATTERNS = {    
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


def get_embeddings(examples, model, tokenizer, args):    
    llama3_template = ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        + "{response}<|eot_id|>") # TO DO: add the template for other model families

    prompts = examples['prompt']
    responses = examples['response']

    # Apply the template function to format the prompt and response
    formatted_texts = [
        llama3_template.format(prompt=p, response=r)
        for p, r in zip(prompts, responses)
    ]
    
    # Tokenize the formatted text
    batched_tokens = [
         tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
         for text in formatted_texts
    ]              
    
    # Generate embeddings
    with torch.no_grad():
        last_hidden_states = [
             model(**tokens).last_hidden_state[-1]
             for tokens in batched_tokens
        ]
        prompt_embeddings = []
        prompt_response_embeddings = []
        success_ls = []
        for i in range(len(batched_tokens)):
            try:
                gating_token_position = find_token_for_gating(
                    list(batched_tokens[i]['input_ids'].flatten()), args.model_family
                )
                prompt_embedding = last_hidden_states[i][gating_token_position].cpu()
                prompt_response_embedding = last_hidden_states[i][-1].cpu()
                success = torch.ones(1, dtype=torch.bool)

            except ValueError:
                prompt_embedding = torch.zeros_like(last_hidden_states[i][-1]).cpu()
                prompt_response_embedding = last_hidden_states[i][-1].cpu()
                success = torch.zeros(1, dtype=torch.bool)
            prompt_embeddings.append(prompt_embedding)
            prompt_response_embeddings.append(prompt_response_embedding)
            success_ls.append(success)
        
        prompt_embedding = torch.stack(prompt_embeddings)
        prompt_response_embedding = torch.stack(prompt_response_embeddings)
        success = torch.cat(success_ls)

    return {
        'prompt_embedding': prompt_embedding,
        'prompt_response_embedding': prompt_response_embedding,
        'success': success,
    }


def embed_dataset(dataset, model, tokenizer, args):
    print('Embedding dataset on device', model.device)
    return dataset.map(
        lambda x: get_embeddings(x, model, tokenizer, args), batched=True, batch_size=8
    )




if __name__ == "__main__":
    from multiprocessing import Pool, set_start_method
    set_start_method('spawn', force=True)
    
    parser = ArgumentParser(description="Get embeddings for a dataset")
    parser.add_argument(
        "--dataset_path", type=str, help="Dataset name", default='../../datasets/ultrafeedback/'
    )
    parser.add_argument(
        "--model_name", type=str, help="Model name", default="meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument(
        "--model_family", type=str, help="Model family", default="llama3"
    )
    parser.add_argument(
        "--devices_idx", type=int, default=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max length for the tokenizer"
    )
    args = parser.parse_args()
    
    
    devices = [f'cuda:{i}' for i in args.devices_idx]


    print("Loading the dataset ...")
    ds = datasets.load_from_disk(os.path.join(args.dataset_path + 'examples/'))
    n_devices = len(devices)
    n = len(ds)
    chunked_datasets = [
        ds.select(list(range(i, n, n_devices)))
        for i in range(n_devices)
    ]
    
    print("Loading the tokenizers and the models ...")
    def initialize_model(args, device):
        print('Initializing model on', device)
        model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        return model
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    models = [initialize_model(args, device) for device in devices]

    print("Generating the embeddings ...")
    with Pool(n_devices) as pool:
        embedded_datasets = pool.starmap(
            embed_dataset, 
            [(ds, model, tokenizer, args) for ds, model in zip(chunked_datasets, models)]
        )
    
    # Concatenate the results from all devices
    embedded_dataset = datasets.concatenate_datasets(embedded_datasets)

    # Save the embeddings
    save_path = os.path.join(args.dataset_path, f'embeddings/{args.model_name}/')

    embedded_dataset.save_to_disk(save_path)
    print(f"Embeddings saved to {save_path}")
