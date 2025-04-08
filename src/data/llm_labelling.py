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


def call_model(
        messages,
        model_name,
        kwargs={},
    ):

    success = False
    it = 0

    while not success and it < MAX_RETRY:
        it += 1
        try: 
            response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    **kwargs,
                )
            output = response.choices[0].message.content

        except Exception as e:
            print(e)
            output = ''
            print("not getting the full list of concepts bc max_tokens reached")
        try:
            output is not None
            success = True
        except:
            print("---------- NOT PASS -------------")
            pass

    if not success:
        raise RuntimeError("Failed after 5 attempts.")
    return  output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./datasets/ultrafeedback/", type=str, help="Path to the pairs dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--port", type=str, default='8000')
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to use for the model")
    args = parser.parse_args()

    # === Config ===
    pairs_path = os.path.join(args.dataset_path, "pairs")
    save_path = os.path.join(args.dataset_path, "labels", args.model_name)
    os.makedirs(save_path, exist_ok=True)
    
    openai_api_base = f"http://localhost:{args.port}/v1"
    openai_api_key = "EMPTY"
    client = openai.OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # === Load dataset ===
    ds = load_from_disk(pairs_path)

    # === Main loop ===
    for example in tqdm(ds, desc="Generating labels"):
        if args.system_prompt is None:
            args.system_prompt = "You are an AI assistant that helps scoring a system reponse." 
        prompt = example["prompt"]
        response_a = example["response_a"]
        response_b = example["response_b"]
        messages = build_rating_chat(args.system_prompt, prompt, response_a, response_b, concepts)
        output = call_model_safe(messages, args.model_name, kwargs={})
        scores = parse_scores(output, concepts)
        example = {
            'idx_a': example['idx_a'],
            'idx_b': example['idx_b'],
            'concept_scores': scores,
        }

        # Write to jsonl
        with open(f"{save_path}/raw_concept_scores.jsonl", "a") as f:
            f.write(json.dumps(example) + "\n")

