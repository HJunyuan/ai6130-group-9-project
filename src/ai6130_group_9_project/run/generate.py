"""Run every prompt through one model with Hugging Face transformers and save one JSON line per answer (role B).

Input: a JSONL file with one object per line containing at least `prompt_id` and `prompt`.
Runs on an NVIDIA GPU (CUDA), Apple Silicon (MPS) or CPU, picked automatically. Final results come from a GPU run.
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SGT = timezone(timedelta(hours=8), "SGT")


def pick_device_map() -> str | None:
    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return "mps"
    return None  # CPU


def load_model(hf_id: str, revision: str | None):
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=revision, device_map=pick_device_map(), torch_dtype="auto"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # We always pass max_new_tokens to generate(); clear any default max_length to avoid a warning.
    model.generation_config.max_length = None
    return tokenizer, model


def eos_token_ids(tokenizer, model) -> list[int]:
    ids = model.generation_config.eos_token_id
    ids = list(ids) if isinstance(ids, (list, tuple)) else ([ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


def generate_batch(prompts: list[str], tokenizer, model, max_new_tokens: int) -> list[dict]:
    """Greedy-decode a batch of prompts with the model's chat template."""
    rendered = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    # The chat template already contains BOS/special tokens.
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    eos_ids = eos_token_ids(tokenizer, model)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

    answers = []
    for row in output_ids[:, prompt_len:].tolist():
        cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
        gen_ids = row if cut is None else row[:cut]
        answers.append(
            {
                "response_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
                "finish_reason": "length" if cut is None else "stop",
                "n_output_tokens": len(gen_ids),
            }
        )
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model ID, as listed in config.yaml")
    parser.add_argument("--input", required=True, type=Path, help="prompts JSONL, e.g. data/prompts_v1.jsonl")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, help="output JSONL (default: results/raw/<model>__<run_id>.jsonl)")
    parser.add_argument("--limit", type=int, help="only run the first N prompts after shuffling (pilots)")
    parser.add_argument("--batch-size", type=int, default=8, help="lower this if you run out of memory")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    models = {m["hf_id"]: m for m in config["models"]}
    if args.model not in models:
        parser.error(f"{args.model} is not in {args.config}; known models: {', '.join(models)}")
    seed = config["seed"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    set_seed(seed)

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer, model = load_model(args.model, models[args.model]["revision"])
    # Record the exact commit the weights came from, even when config.yaml does not pin one yet.
    revision = HfApi().model_info(args.model, revision=models[args.model]["revision"]).sha
    print(f"Loaded {args.model} (revision {revision}) on {model.device}")

    # Run ID = start time in Singapore time, e.g. 20261016-170512. Each run gets its own file, so reruns never overwrite.
    run_id = datetime.now(SGT).strftime("%Y%m%d-%H%M%S")
    output = args.output or Path("results/raw") / f"{args.model.replace('/', '__')}__{run_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="batches"):
            batch = rows[start : start + args.batch_size]
            for row, answer in zip(batch, generate_batch([r["prompt"] for r in batch], tokenizer, model, max_new_tokens)):
                record = {
                    "run_id": run_id,
                    "prompt_id": row["prompt_id"],
                    "model": args.model,
                    "model_revision": revision,
                    "seed": seed,
                    **answer,
                    "timestamp": datetime.now(SGT).isoformat(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Wrote {len(rows)} answers to {output}")
    print(f"Run ID {run_id}: add a row to the run log in docs/REPORT.md (Appendix A)")
