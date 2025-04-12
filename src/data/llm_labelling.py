import os, re
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
import gc
from safetensors.torch import save_file
import openai
import argparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np

MAX_RETRY = 5 # in case when you prompt an llm the api returns an error (it may sometime happen if you do multiple LLM calls in parallel)
MAX_TOKENS = 1024

concepts = [
    "helpfulness", "correctness", "coherence", "complexity", 
    "verbosity", "instruction_following", "truthfulness", 
    "honesty", "safety", "readability"
]

def call_model_safe(messages, model_name, kwargs):
    try:
        return call_model(messages, model_name, kwargs)
    except Exception as e:
        print(f"Error: {e}")
        return ''

def build_rating_chat(system_prompt, user_prompt, response_a, response_b, concepts):
    concept_str = ", ".join(concepts)
    instruction = (
        f"Rate which of the two assistant responses is better for each of the following concepts: {concept_str}.\n"
        f"Provide your confidence score on the scale fron 0 to 1 per each concept, where 0 means the first response is clearly better and 1 that the second response is clearly better.\n"
        f"Respond strictly in JSON format without any explanation.\n"
        f"Example:\n"
        f"{{\n"
        + "\n".join([f'  "{c}": <score>,' for c in concepts]) + "\n"
        f"}}\n\n"
        f"Now rate the two responses."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response_a},
        {"role": "assistant", "content": response_b},
        {"role": "user", "content": instruction}
    ]
    return messages

def parse_scores(text, concepts):
    """
    Extract and parse JSON object from possibly messy text.
    Removes ```json blocks and extra text around JSON.
    """
    # Remove markdown ticks if present
    scores = dict(zip(concepts, [-1.0] * len(concepts))) # default to -1.0 if not found
    text = text.strip()
    if text == '':
        return scores
    
    if text.startswith("```json"):
        text = text.lstrip("```json").rstrip("```").strip()
    elif text.startswith("```"):
        text = text.lstrip("```").rstrip("```").strip()
    

    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            json_obj =  json.loads(json_str)
            for c in concepts:
                scores[c.lower()] = json_obj.get(c, -1.0) 
        except json.JSONDecodeError as e:
            pass
    if not scores:
        for line in text.splitlines():
            if ":" in line:
                key = key.strip().lower().strip('"').replace(" ", "_").replace("-", "_")
                val = val.strip().replace(",", "")
                try:
                    val = float(val)
                    scores[key] = val
                except ValueError:
                    continue  # skip non-numeric
    
    for k, v in scores.items():
        if isinstance(v, str):
            try:
                scores[k] = float(v)
            except ValueError:
                scores[k] = -1.0
        elif not isinstance(v, (int, float)):
            scores[k] = -1.0
        
    return scores


def call_model_safe(messages, model_name, kwargs):
    try:
        return call_model(messages, model_name, kwargs)
    except Exception as e:
        print(f"Error: {e}")
        return ''

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def call_model(messages, model_name, kwargs={}):
    response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            **kwargs,
        )
    output = response.choices[0].message.content
    return  output

def label_chunk(chunk, chunk_id, model_name, system_prompt, save_path, kwargs):
    print(f"Processing chunk {chunk_id} with {len(chunk)} examples...")
    for example in tqdm(chunk, total=len(chunk), desc=f"{chunk_id}"):
        prompt = example["prompt"]
        response_a = example["response_a"]
        response_b = example["response_b"]
        # Build the chat messages
        messages = build_rating_chat(system_prompt, prompt, response_a, response_b, concepts)
        # Call the model
        try:
            output = call_model_safe(messages, model_name, kwargs)
            scores = parse_scores(output, concepts)
            example = {
                "idx_a": example["idx_a"],
                "idx_b": example["idx_b"],
                "scores": scores
            }
            # Write to jsonl
            with open(save_path, "a") as f:
                f.write(json.dumps(example) + "\n")

        except Exception as e:
            output = f"Error: {e}"

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./datasets/ultrafeedback/", type=str, help="Path to the pairs dataset")
    parser.add_argument("--model_name", type=str, default="gpt4o-eus2-202407")#"meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--port", type=str, default='8000')
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to use for the model")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel jobs to run")
    args = parser.parse_args()

    # === Config ===
    pairs_path = os.path.join(args.dataset_path, "pairs")
    save_path = os.path.join(args.dataset_path, "concept_labels", args.model_name)
    os.makedirs(save_path, exist_ok=False)

    if args.system_prompt is None:
        args.system_prompt = "You are an AI assistant that helps scoring a system reponse." 
    
    if args.model_name.startswith("gpt"):
        if args.model_name == "gpt4o-eus2-202407":
            azure_endpoint = "https://vdslabten-oai-eus2.openai.azure.com/"
            openai_api_key = os.environ['GPT4o_API_KEY']

        else:
            raise ValueError(f"Unknown gpt model name: {args.model_name}")

        if openai_api_key is None:
            raise ValueError("Please set the environment variable GPT4o_API_KEY")

        client = openai.AzureOpenAI(
            api_key=openai_api_key,
            api_version="2024-07-01-preview",
            azure_endpoint=azure_endpoint,
        )

    else:
        openai_api_base = f"http://localhost:{args.port}/v1"
        openai_api_key = "EMPTY"
        client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    kwargs = {}
    # === Load dataset ===
    ds = load_from_disk(pairs_path)
    ds = ds.remove_columns(['scores_a', 'scores_b'])

    # Split into chunks
    n = len(ds)
    num_chunks = args.num_workers
    chunked_ds = [
        ds.select(list(range(i, n, num_chunks)))
        for i in range(num_chunks)
    ]

    save_paths = [
        os.path.join(save_path, f"scores-{i}-of-{num_chunks}.jsonl")
        for i in range(num_chunks)
    ]

    print("Starting to process chunks...")
    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
        futures = {
            executor.submit(
                label_chunk,
                chunk, chunk_id, args.model_name, args.system_prompt, save_path, kwargs
            ): chunk_id
            for chunk_id, (chunk, save_path) in enumerate(zip(chunked_ds, save_paths))
        }
