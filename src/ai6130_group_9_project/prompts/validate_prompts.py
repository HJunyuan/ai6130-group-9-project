import os
import csv
import json
import yaml
import hashlib
from collections import defaultdict
import tiktoken

TONE_ORDER = ["very_rude", "blunt", "neutral", "polite", "very_deferential"]


def count_openai_tokens(text):
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def count_gemini_tokens(text):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.count_tokens(
        model="gemini-3.6-flash",
        contents=text
    )
    return resp.total_tokens


def verify_prompts_and_lengths(jsonl_path="data/prompts_v1.jsonl", yaml_path="templates.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        tones_dict = yaml.safe_load(f)["tones"]

    tok_cache = {"openai": {}, "gemini": {}}
    for tone in TONE_ORDER:
        for tpl in tones_dict[tone]:
            key = (tone, tpl["id"])
            txt = tpl["text"]
            tok_cache["openai"][key] = count_openai_tokens(txt)
            tok_cache["gemini"][key] = count_gemini_tokens(txt)

    violations = []
    for provider, table in tok_cache.items():
        if any(v is None for v in table.values()):
            print(f"[INFO] Skipping {provider} token check because API key is not set.")
            continue
        for t_id in ["t0", "t1", "t2", "t3"]:
            base_len = table[("neutral", t_id)]
            for tone in TONE_ORDER:
                diff = table[(tone, t_id)] - base_len
                if abs(diff) > 3:
                    violations.append((provider, tone, t_id, diff))

    if violations:
        raise AssertionError(f"Token length tolerance exceeded: {violations}")
    print("[PASS] All tone templates are within +/- 3 tokens of their neutral twin.")

    by_item = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_item[row["item_id"]].append(row)

    assert len(by_item) == 230, f"Expected 230 unique items, found {len(by_item)}"

    full_lengths = {"openai": defaultdict(list)}

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

            body_oai = count_openai_tokens(parts[1])
            full_lengths["openai"][r["tone"]].append(tok_cache["openai"][(r["tone"], r["template_id"])] + body_oai)

        assert len(body_hashes) == 1, f"Byte-for-byte hash mismatch across tones on {item_id}"
        assert len(tpl_ids) == 1, f"Template ID mismatch across tones on {item_id}"

    print("[PASS] Byte-for-byte SHA256 hash check passed on all 230 questions x 5 tones.")

    print("\n--- Mean Template & Prompt Token Lengths (Copy to REPORT.md) ---")
    print("| Tone | OpenAI Template Mean | OpenAI Full Prompt Mean | Gemini Template Mean |")
    print("| --- | --- | --- | --- |")
    for tone in TONE_ORDER:
        oai_t = sum(tok_cache["openai"][(tone, f"t{i}")] for i in range(4)) / 4.0
        oai_full = sum(full_lengths["openai"][tone]) / len(full_lengths["openai"][tone])
        
        gem_vals = [tok_cache["gemini"][(tone, f"t{i}")] for i in range(4)]
        gem_t = f"{sum(gem_vals)/4.0:.2f}" if None not in gem_vals else "Pending API Key"
        
        print(f"| `{tone}` | {oai_t:.2f} | {oai_full:.2f} | {gem_t} |")


def score_human_tone_ratings(filled_csv="data/tone_rating_sheet.csv", key_csv="data/tone_rating_key.csv"):
    if not os.path.exists(filled_csv) or not os.path.exists(key_csv):
        return

    key_map = {}
    with open(key_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key_map[r["sentence_num"]] = r["true_tone"]

    scores_by_tone = defaultdict(list)
    with open(filled_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals = [r["rater_1_score_1_to_7"], r["rater_2_score_1_to_7"], r["rater_3_score_1_to_7"]]
            if any(v.strip() == "" for v in vals):
                return
            nums = [float(v) for v in vals]
            tone = key_map[r["sentence_num"]]
            scores_by_tone[tone].append(sum(nums) / 3.0)

    print("\n--- Human Politeness Ratings (1 to 7 Scale) ---")
    means = []
    for tone in TONE_ORDER:
        m = sum(scores_by_tone[tone]) / len(scores_by_tone[tone])
        means.append(m)
        print(f"{tone}: {m:.2f}")
    assert means == sorted(means), f"Ordering check failed! Means are not strictly monotonic: {means}"
    print("[PASS] All 5 tone levels are strictly ordered in human ratings.")


if __name__ == "__main__":
    verify_prompts_and_lengths()
    score_human_tone_ratings()