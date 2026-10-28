"""Run every prompt through an open-weight Hugging Face model and extract logit telemetry (Role B).

Input: a JSONL file with one object per line containing at least `prompt_id` and `prompt`.
The script loads models locally (or on Colab GPUs) in bfloat16 for batched greedy generation.
It extracts step-0 refusal/compliance logprobs and answer-step option probabilities for analysis,
saving results atomically to a JSONL file with built-in resume capability.
"""

import os
import re
import gc
import json
import yaml
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from huggingface_hub import model_info

SGT = timezone(timedelta(hours=8))

# Canonical starter strings for continuous RQ1 refusal vs. compliance scoring at step 0
REFUSAL_STARTERS = ["I", " I", "Sorry", " Sorry", "Apologies", " Apologies", "Cannot", " Cannot", "Unable", "As", " As"]
COMPLIANCE_STARTERS = ["To", " To", "The", " The", "Let", " Let", "Here", " Here", "Sure", " Sure", "Answer", "A", "B", "C", "D", "E"]
MC_OPTIONS = ["A", "B", "C", "D", "E"]


def select_device_and_dtype(preferred_dtype="bfloat16"):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if (preferred_dtype == "bfloat16" and torch.cuda.is_bf16_supported()) else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def load_config(config_path="config.yaml"):
    if not Path(config_path).exists():
        return {"seed": 42, "generation": {"do_sample": False, "max_new_tokens": 512, "dtype": "bfloat16"}, "models": {}}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_single_token_lookup(tokenizer, candidate_strings):
    """Maps candidate surface strings to their single-token IDs if they encode to 1 token."""
    token_ids = set()
    for s in candidate_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(ids[0])
    return sorted(list(token_ids))


def build_mc_option_token_map(tokenizer):
    """Maps each option letter A-E to its single-token variants (e.g. 'A' and ' A')."""
    opt_map = {}
    for letter in MC_OPTIONS:
        ids = set()
        for variant in [letter, f" {letter}"]:
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        opt_map[letter] = sorted(list(ids))
    return opt_map


def find_answer_token_step(tokenizer, gen_token_ids):
    """
    Finds the generation step index t* immediately following the last 'Answer:' in the output.
    Returns None if 'Answer:' was not generated.
    """
    full_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
    if "Answer:" not in full_text:
        return None

    # Search backwards from the end of the generated sequence to locate the step right after 'Answer:'
    for step_idx in range(len(gen_token_ids) - 1, -1, -1):
        prefix_text = tokenizer.decode(gen_token_ids[:step_idx], skip_special_tokens=True)
        if "Answer:" in prefix_text:
            # Check if step_idx - 1 did NOT have the final 'Answer:' or if prefix_text ends right after 'Answer:'
            stripped = prefix_text.rstrip()
            if stripped.endswith("Answer:"):
                return step_idx
    return None


def extract_logits_telemetry(
    step_scores,
    batch_idx,
    gen_token_ids,
    tokenizer,
    refusal_token_ids,
    compliance_token_ids,
    mc_option_map,
    top_k=10,
):
    """Extracts step-0 refusal/compliance logprobs and answer-step option probabilities."""
    if not step_scores or len(step_scores) == 0:
        return None

    # 1. Step 0 (First generated token) logprobs
    step0_logits = step_scores[0][batch_idx].float()
    step0_logprobs = F.log_softmax(step0_logits, dim=-1)

    top_vals, top_idxs = torch.topk(step0_logprobs, k=min(top_k, step0_logprobs.shape[-1]))
    first_step_top = [
        {
            "token": tokenizer.decode([idx.item()]),
            "token_id": int(idx.item()),
            "logprob": round(float(val.item()), 5),
            "prob": round(float(torch.exp(val).item()), 5),
        }
        for val, idx in zip(top_vals, top_idxs)
    ]

    refusal_prob = float(torch.exp(torch.logsumexp(step0_logprobs[refusal_token_ids], dim=0)).item()) if refusal_token_ids else 0.0
    compliance_prob = float(torch.exp(torch.logsumexp(step0_logprobs[compliance_token_ids], dim=0)).item()) if compliance_token_ids else 0.0

    # 2. Answer step (immediately after 'Answer:') logprobs
    ans_step = find_answer_token_step(tokenizer, gen_token_ids)
    ans_option_probs = None
    ans_top_tokens = None

    if ans_step is not None and ans_step < len(step_scores):
        ans_logits = step_scores[ans_step][batch_idx].float()
        ans_logprobs = F.log_softmax(ans_logits, dim=-1)

        # Top-5 raw tokens at the answer position (useful for GSM8K numbers and debugging)
        a_vals, a_idxs = torch.topk(ans_logprobs, k=5)
        ans_top_tokens = [
            {
                "token": tokenizer.decode([idx.item()]).strip(),
                "token_id": int(idx.item()),
                "logprob": round(float(val.item()), 5),
                "prob": round(float(torch.exp(val).item()), 5),
            }
            for val, idx in zip(a_vals, a_idxs)
        ]

        # Option letter probabilities (A-E)
        raw_opt_probs = {}
        for letter, ids in mc_option_map.items():
            if ids:
                raw_opt_probs[letter] = float(torch.exp(torch.logsumexp(ans_logprobs[ids], dim=0)).item())
            else:
                raw_opt_probs[letter] = 0.0

        total_opt_mass = sum(raw_opt_probs.values())
        ans_option_probs = {
            "raw_probs": {k: round(v, 5) for k, v in raw_opt_probs.items()},
            "normalized_probs": {
                k: round(v / total_opt_mass, 5) if total_opt_mass > 1e-12 else 0.0
                for k, v in raw_opt_probs.items()
            },
        }

    return {
        "first_step_refusal_prob": round(refusal_prob, 5),
        "first_step_compliance_prob": round(compliance_prob, 5),
        "first_step_top_tokens": first_step_top,
        "answer_token_step": ans_step,
        "answer_step_option_probs": ans_option_probs,
        "answer_step_top_tokens": ans_top_tokens,
    }


def resolve_commit_sha(model_id: str, revision: str = "main", token: str | None = None) -> str:
    """Resolves the exact 40-character commit SHA via local cache or Hub API."""
    # Check local Hugging Face cache refs file
    safe_repo_dir = f"models--{model_id.replace('/', '--')}"
    ref_path = Path.home() / ".cache" / "huggingface" / "hub" / safe_repo_dir / "refs" / revision
    if ref_path.exists():
        cached_sha = ref_path.read_text(encoding="utf-8").strip()
        if len(cached_sha) == 40:
            return cached_sha

    # Query Hugging Face Hub API
    try:
        info = model_info(model_id, revision=revision, token=token)
        if getattr(info, "sha", None):
            return str(info.sha)
    except Exception as e:
        print(f"[WARN] Failed to fetch commit SHA from Hub API for {model_id}: {e}")

    # Fallback
    return revision


def main():
    parser = argparse.ArgumentParser(description="Run prompts through an open-weight Hugging Face model.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID (e.g., Qwen/Qwen2.5-32B-Instruct)")
    parser.add_argument("--input", default="data/prompts_v1.jsonl", help="Input JSONL file of prompts")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N prompts (for laptop pilots)")
    parser.add_argument("--batch-size", type=int, default=None, help="Prompts per batch (overrides config default)")
    parser.add_argument("--output", default=None, help="Custom output JSONL path (automatically resumes if file exists)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    gen_cfg = cfg.get("generation", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    extract_logits = bool(gen_cfg.get("extract_logits", True))

    model_cfg = cfg.get("models", {}).get(args.model, {})
    revision = model_cfg.get("revision", "main")
    template_kwargs = model_cfg.get("template_kwargs", {})
    batch_size = args.batch_size or int(model_cfg.get("default_batch_size", 8))

    set_seed(seed)
    device, dtype = select_device_and_dtype(gen_cfg.get("dtype", "bfloat16"))

    # Generate Run ID in Singapore Time (SGT)
    run_id = datetime.now(SGT).strftime("%Y%m%d-%H%M%S")
    safe_model_name = args.model.replace("/", "--")

    if args.output:
        out_path = Path(args.output)
        # If resuming an existing file, preserve its run_id from the filename if formatted as <model>__<run_id>.jsonl
        if "__" in out_path.stem:
            run_id = out_path.stem.split("__")[-1]
    else:
        out_path = Path("results/raw") / f"{safe_model_name}__{run_id}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts and shuffle deterministically with fixed seed (per ROADMAP.md)
    with open(args.input, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    if args.limit is not None:
        prompts = prompts[: args.limit]

    rng = random.Random(seed)
    rng.shuffle(prompts)

    # Check for completed prompt_ids if resuming
    completed_ids = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_ids.add(rec["prompt_id"])
                    except Exception:
                        pass
        if completed_ids:
            print(f"[RESUME] Found {len(completed_ids)} completed prompts in {out_path}. Skipping them.")

    remaining_prompts = [p for p in prompts if p["prompt_id"] not in completed_ids]
    if not remaining_prompts:
        print("[DONE] All prompts in this run are already completed.")
        return

    print(f"[INIT] Model: {args.model} | Device: {device} | Dtype: {dtype} | Batch Size: {batch_size} | Run ID: {run_id}")
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=revision,
        token=hf_token,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "revision": revision,
        "token": hf_token,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()

    exact_commit_sha = getattr(model.config, "_commit_hash", None)
    if not exact_commit_sha or exact_commit_sha == revision:
        exact_commit_sha = resolve_commit_sha(args.model, revision=revision, token=hf_token)

    refusal_ids = build_single_token_lookup(tokenizer, REFUSAL_STARTERS)
    compliance_ids = build_single_token_lookup(tokenizer, COMPLIANCE_STARTERS)
    mc_option_map = build_mc_option_token_map(tokenizer)

    with open(out_path, "a", encoding="utf-8") as out_f:
        for start_idx in tqdm(range(0, len(remaining_prompts), batch_size), desc=f"Generating ({safe_model_name})"):
            batch = remaining_prompts[start_idx : start_idx + batch_size]

            formatted_inputs = []
            for item in batch:
                messages = [{"role": "user", "content": item["prompt"]}]
                chat_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                formatted_inputs.append(chat_str)

            enc = tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(model.device)

            input_seq_len = enc["input_ids"].shape[1]

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]
            terminators = [t for t in terminators if t is not None]

            with torch.no_grad():
                outputs = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=terminators,
                    return_dict_in_generate=True,
                    output_scores=extract_logits,
                )

            sequences = outputs.sequences
            step_scores = outputs.scores if extract_logits else None

            for b_idx, item in enumerate(batch):
                gen_ids = sequences[b_idx, input_seq_len:].tolist()

                # Strip trailing pad/eos tokens to compute exact n_output_tokens and finish_reason
                clean_gen_ids = []
                hit_eos = False
                eos_ids = tokenizer.eos_token_id
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                elif eos_ids is None:
                    eos_ids = []

                for tid in gen_ids:
                    if tid in eos_ids:
                        hit_eos = True
                        break
                    if tid == tokenizer.pad_token_id:
                        continue
                    clean_gen_ids.append(tid)

                response_text = tokenizer.decode(clean_gen_ids, skip_special_tokens=True).strip()
                n_output_tokens = len(clean_gen_ids)
                finish_reason = "stop" if (hit_eos or n_output_tokens < max_new_tokens) else "length"

                logits_meta = None
                if extract_logits and step_scores is not None:
                    logits_meta = extract_logits_telemetry(
                        step_scores=step_scores,
                        batch_idx=b_idx,
                        gen_token_ids=clean_gen_ids,
                        tokenizer=tokenizer,
                        refusal_token_ids=refusal_ids,
                        compliance_token_ids=compliance_ids,
                        mc_option_map=mc_option_map,
                    )

                record = {
                    "run_id": run_id,
                    "prompt_id": item["prompt_id"],
                    "model": args.model,
                    "model_revision": exact_commit_sha,
                    "seed": seed,
                    "response_text": response_text,
                    "finish_reason": finish_reason,
                    "n_output_tokens": n_output_tokens,
                    "timestamp": datetime.now(SGT).isoformat(),
                    "logits_meta": logits_meta,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Atomic flush after every batch so Colab disconnects never lose progress
            out_f.flush()
            os.fsync(out_f.fileno())

            del outputs, sequences, step_scores, enc
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n[COMPLETE] Saved outputs to {out_path}")
    print(f"[REPORT LOG] Date: {datetime.now(SGT).strftime('%d %b')} | Run ID: {run_id} | Model: {args.model} | Revision: {exact_commit_sha[:10]} | Items: {len(prompts)}")


if __name__ == "__main__":
    main()