"""Run every prompt through one model via its provider's API and save one JSON line per answer (role B).

Input: a JSONL file with one object per line containing at least `prompt_id` and `prompt`.
The model is picked by its API model ID as listed in config.yaml; the matching API key must be set (see .env.example).
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from ai6130_group_9_project.providers import generate

SGT = timezone(timedelta(hours=8), "SGT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="API model ID, as listed in config.yaml")
    parser.add_argument("--input", required=True, type=Path, help="prompts JSONL, e.g. data/prompts_v1.jsonl")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, help="output JSONL (default: results/raw/<model>__<run_id>.jsonl)")
    parser.add_argument("--limit", type=int, help="only run the first N prompts after shuffling (pilots)")
    parser.add_argument("--workers", type=int, default=4, help="requests in flight at once; lower it if rate-limited")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    models = {m["id"]: m for m in config["models"]}
    if args.model not in models:
        parser.error(f"{args.model} is not in {args.config}; known models: {', '.join(models)}")
    model = models[args.model]
    seed = config["seed"]
    generation = config["generation"]

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error(f"{args.input} has no prompts")
    random.Random(seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    # Run ID = start time in Singapore time, e.g. 20261016-170512. Each run gets its own file, so reruns never overwrite.
    run_id = datetime.now(SGT).strftime("%Y%m%d-%H%M%S")
    output = args.output or Path("results/raw") / f"{args.model}__{run_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    def to_record(row: dict, result: dict) -> dict:
        return {
            "run_id": run_id,
            "prompt_id": row["prompt_id"],
            "model": args.model,
            "provider": model["provider"],
            "seed": seed,
            "temperature": generation["temperature"],
            **result,
            "timestamp": datetime.now(SGT).isoformat(),
        }

    def answer(row: dict) -> dict:
        try:
            result = generate(model, row["prompt"], generation, seed)
        except Exception as e:  # noqa: BLE001 - one failed prompt (after the SDK's retries) must not stop the run
            result = {"response_text": "", "finish_reason": "error", "error": f"{type(e).__name__}: {e}"}
        return to_record(row, result)

    # The first prompt runs on its own and without error capture, so a wrong key or model ID fails immediately
    # instead of producing a file full of errors.
    first = to_record(rows[0], generate(model, rows[0]["prompt"], generation, seed))
    print(f"Connected to {args.model} ({model['provider']}); API reports model version {first['model_version']}")

    n_errors = 0
    with output.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        f.write(json.dumps(first, ensure_ascii=False) + "\n")
        # map() keeps input order, so the file is in the same (shuffled) order as the prompts.
        for record in tqdm(pool.map(answer, rows[1:]), total=len(rows) - 1, desc="prompts"):
            n_errors += record["finish_reason"] == "error"
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Wrote {len(rows)} answers to {output}")
    if n_errors:
        print(f"WARNING: {n_errors} prompts failed after retries (finish_reason 'error'); see the `error` field")
    print(f"Run ID {run_id}: add a row to the run log in docs/REPORT.md (Appendix A)")
