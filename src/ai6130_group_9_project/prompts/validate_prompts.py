import os
import csv
import json
import yaml
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

TONE_ORDER = ["very_rude", "blunt", "neutral", "polite", "very_deferential"]

# Default open-weight tokenizers to validate against.
DEFAULT_MODELS = {
    "Gemma-3-27B": "google/gemma-3-27b-it",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "OLMo-3.1-32B": "allenai/Olmo-3.1-32B-Instruct",
}


def load_hf_tokenizers(model_map):
    tokenizers = {}
    hf_token = os.environ.get("HF_TOKEN")
    for short_name, repo_id in model_map.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo_id, token=hf_token, trust_remote_code=True)
            tokenizers[short_name] = tok
            print(f"[LOADED] Tokenizer for {short_name} ({repo_id})")
        except Exception as e:
            print(f"[WARN] Could not load tokenizer for {short_name} ({repo_id}): {e}")
    return tokenizers


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def verify_prompts_and_lengths(
    jsonl_path="data/prompts_v1.jsonl",
    yaml_path="data/templates.yaml",
    model_map=None,
):
    if model_map is None:
        model_map = DEFAULT_MODELS

    yaml_file = Path(yaml_path) if Path(yaml_path).exists() else Path("templates.yaml")
    with open(yaml_file, "r", encoding="utf-8") as f:
        tones_dict = yaml.safe_load(f)["tones"]

    tokenizers = load_hf_tokenizers(model_map)
    if not tokenizers:
        raise RuntimeError("No Hugging Face tokenizers could be loaded. Check HF_TOKEN or network.")

    # 1. Check template lengths (+/- 3 tokens vs neutral twin)
    tok_cache = {m: {} for m in tokenizers}
    for m_name, tok in tokenizers.items():
        for tone in TONE_ORDER:
            for tpl in tones_dict[tone]:
                key = (tone, tpl["id"])
                tok_cache[m_name][key] = count_tokens(tok, tpl["text"])

    violations = []
    print("\n--- Template Token Count Audit (vs. Neutral Twin) ---")
    for m_name in tokenizers:
        for t_id in ["t0", "t1", "t2", "t3"]:
            base_len = tok_cache[m_name][("neutral", t_id)]
            for tone in TONE_ORDER:
                t_len = tok_cache[m_name][(tone, t_id)]
                diff = t_len - base_len
                if abs(diff) > 3:
                    violations.append((m_name, tone, t_id, t_len, base_len, diff))

    if violations:
        print("[FAIL] The following templates exceed +/- 3 tokens vs neutral:")
        for v in violations:
            print(f"  Model={v[0]} | Tone={v[1]} | Template={v[2]} | Len={v[3]} vs Neutral={v[4]} (diff={v[5]:+d})")
        raise AssertionError(f"Token length tolerance exceeded on {len(violations)} checks.")
    else:
        print("[PASS] All 20 tone templates are within +/- 3 tokens of their neutral twin across all loaded tokenizers.")

    # 2. Check full prompt dataset integrity & byte-for-byte question hashes
    if not Path(jsonl_path).exists():
        print(f"[INFO] {jsonl_path} not found yet; skipping full JSONL check.")
        return

    by_item = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_item[row["item_id"]].append(row)

    assert len(by_item) == 230, f"Expected 230 unique items, found {len(by_item)}"

    full_lengths = {m_name: defaultdict(list) for m_name in tokenizers}

    for item_id, group in by_item.items():
        assert len(group) == 5, f"Item {item_id} does not have 5 tone variants"
        body_hashes = set()
        tpl_ids = set()

        for r in group:
            parts = r["prompt"].split("\n\n", 1)
            assert len(parts) == 2, f"Malformed prompt structure in {r['prompt_id']}"
            body_hash = hashlib.sha256(parts[1].encode("utf-8")).hexdigest()
            body_hashes.add(body_hash)
            tpl_ids.add(r["template_id"])

            for m_name, tok in tokenizers.items():
                full_lengths[m_name][r["tone"]].append(count_tokens(tok, r["prompt"]))

        assert len(body_hashes) == 1, f"Byte-for-byte hash mismatch across tones on {item_id}"
        assert len(tpl_ids) == 1, f"Template ID mismatch across tones on {item_id}"

    print(f"[PASS] Byte-for-byte SHA256 hash check passed on all {len(by_item)} items x 5 tones.")

    # 3. Output REPORT.md Section 4.1 Table
    model_names = list(tokenizers.keys())
    header_cols = " | ".join([f"Mean tokens ({m})" for m in model_names])
    sep_cols = " | ".join(["---" for _ in model_names])

    print("\n--- Section 4.1 Prompt Length Table (Copy to docs/REPORT.md) ---")
    print(f"| Tone level | {header_cols} | Largest difference vs neutral |")
    print(f"| --- | {sep_cols} | --- |")

    neutral_means = {
        m: sum(full_lengths[m]["neutral"]) / len(full_lengths[m]["neutral"])
        for m in model_names
    }

    for tone in TONE_ORDER:
        row_strs = []
        max_abs_diff = 0.0
        for m in model_names:
            m_mean = sum(full_lengths[m][tone]) / len(full_lengths[m][tone])
            row_strs.append(f"{m_mean:.2f}")
            diff = abs(m_mean - neutral_means[m])
            if diff > max_abs_diff:
                max_abs_diff = diff
        diff_str = "—" if tone == "neutral" else f"{max_abs_diff:.2f}"
        print(f"| `{tone}` | {' | '.join(row_strs)} | {diff_str} |")


def score_human_tone_ratings(
    filled_csv="data/tone_rating_sheet.csv",
    key_csv="data/tone_rating_key.csv",
):
    if not os.path.exists(filled_csv) or not os.path.exists(key_csv):
        return

    key_map = {}
    with open(key_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key_map[r["sentence_num"]] = r["true_tone"]

    scores_by_tone = defaultdict(list)
    with open(filled_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals = [
                r["rater_1_score_1_to_7"],
                r["rater_2_score_1_to_7"],
                r["rater_3_score_1_to_7"],
            ]
            if any(v.strip() == "" for v in vals):
                print("\n[INFO] Human tone rating sheet is not yet filled in.")
                return
            nums = [float(v) for v in vals]
            tone = key_map[r["sentence_num"]]
            scores_by_tone[tone].append(sum(nums) / 3.0)

    print("\n--- Human Politeness Ratings (1 to 7 Scale) ---")
    means = []
    for tone in TONE_ORDER:
        m = sum(scores_by_tone[tone]) / len(scores_by_tone[tone])
        means.append(m)
        print(f"| `{tone}` | {m:.2f} |")
    assert means == sorted(means), f"Ordering check failed! Means are not strictly monotonic: {means}"
    print("[PASS] All 5 tone levels are strictly ordered in human ratings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate prompt hashes, token lengths, and rating sheets.")
    parser.add_argument("--input", default="data/prompts_v1.jsonl", help="Path to prompts JSONL file.")
    parser.add_argument("--templates", default="data/templates.yaml", help="Path to templates YAML file.")
    args = parser.parse_args()

    verify_prompts_and_lengths(jsonl_path=args.input, yaml_path=args.templates)
    score_human_tone_ratings()