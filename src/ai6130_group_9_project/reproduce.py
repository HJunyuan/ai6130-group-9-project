"""Rebuild every table and chart from the raw outputs in results/raw/ by running grading and analysis (Role B).

Usage:
    uv run reproduce
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ai6130_group_9_project.evaluate.grade import (
    compute_cohens_kappa,
    evaluate_row,
    load_jsonl,
)

TONE_ORDER = ["very_rude", "blunt", "neutral", "polite", "very_deferential"]
TONE_LABELS = {
    "very_rude": "Very rude",
    "blunt": "Blunt",
    "neutral": "Neutral",
    "polite": "Polite",
    "very_deferential": "Very deferential",
}
MODEL_ORDER = [
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-32B",
    "allenai/Olmo-3.1-32B-Instruct",
]
MODEL_SHORT = {
    "google/gemma-3-27b-it": "Gemma-3-27B-IT",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    "allenai/Olmo-3.1-32B-Instruct": "OLMo-3.1-32B-Instruct",
}
MARKED_TASKS = ["mmlu_pro_stem", "commonsense_qa", "medmcqa", "gsm8k"]
MC_TASKS = ["mmlu_pro_stem", "commonsense_qa", "medmcqa"]
TASK_CHANCE = {
    "mmlu_pro_stem": 25.0,
    "commonsense_qa": 20.0,
    "medmcqa": 25.0,
    "gsm8k": 0.0,
}


# ---------------------------------------------------------------------------
# 1. Raw Data Loading, Logit Telemetry Extraction & Grading
# ---------------------------------------------------------------------------


def extract_logit_fields(
    logits_meta: dict[str, Any] | None,
    gold: str | None,
) -> dict[str, float | int | None]:
    """Extract scalar telemetry fields from a raw record's logits_meta dictionary."""
    if not isinstance(logits_meta, dict):
        return {
            "first_step_refusal_prob": None,
            "first_step_compliance_prob": None,
            "answer_token_step": None,
            "answer_step_gold_prob_raw": None,
            "answer_step_gold_prob_norm": None,
        }

    ref_p = logits_meta.get("first_step_refusal_prob")
    comp_p = logits_meta.get("first_step_compliance_prob")
    ans_step = logits_meta.get("answer_token_step")

    gold_raw = None
    gold_norm = None
    opt_probs = logits_meta.get("answer_step_option_probs")
    if isinstance(opt_probs, dict) and gold is not None:
        g_let = str(gold).strip().upper()
        raw_map = opt_probs.get("raw_probs", {})
        norm_map = opt_probs.get("normalized_probs", {})
        if g_let in raw_map:
            gold_raw = float(raw_map[g_let])
            gold_norm = float(norm_map.get(g_let, 0.0))

    return {
        "first_step_refusal_prob": float(ref_p) if ref_p is not None else None,
        "first_step_compliance_prob": float(comp_p) if comp_p is not None else None,
        "answer_token_step": int(ans_step) if ans_step is not None else None,
        "answer_step_gold_prob_raw": gold_raw,
        "answer_step_gold_prob_norm": gold_norm,
    }


def grade_dataset(
    prompts_path: Path,
    raw_dir: Path,
    output_csv: Path,
    is_yesman: bool = False,
) -> pd.DataFrame:
    """Load matching raw JSONL outputs, run evaluate_row + logit extraction, and save graded CSV."""
    prompts_list = load_jsonl(prompts_path)
    prompts_df = pd.DataFrame(prompts_list)
    valid_prompt_ids = set(prompts_df["prompt_id"])

    raw_files = sorted(
        rf for rf in raw_dir.glob("*.jsonl") if not rf.name.startswith("pilot_")
    )
    if not raw_files:
        raise FileNotFoundError(
            f"No raw JSONL files found in {raw_dir}"
        )

    raw_rows: list[dict[str, Any]] = []
    for rf in raw_files:
        for row in load_jsonl(rf):
            if row.get("prompt_id") in valid_prompt_ids:
                raw_rows.append(row)

    if not raw_rows:
        raise RuntimeError(f"Found 0 records matching {prompts_path.name} inside {raw_dir}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df["_is_valid"] = raw_df["finish_reason"].fillna("") != "error"
    raw_df = (
        raw_df.sort_values(["_is_valid", "timestamp"])
        .drop_duplicates(subset=["model", "prompt_id"], keep="last")
        .drop(columns=["_is_valid"])
    )

    merged = raw_df.merge(prompts_df, on="prompt_id", how="left", validate="many_to_one")

    graded_records: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        eval_res = evaluate_row(
            response_text=row.get("response_text"),
            finish_reason=row.get("finish_reason"),
            task=str(row.get("task", "")),
            gold=row.get("gold"),
            prompt_text=str(row.get("prompt", "")),
            n_output_tokens=row.get("n_output_tokens"),
        )
        logit_res = extract_logit_fields(row.get("logits_meta"), row.get("gold"))
        combined = {**eval_res, **logit_res}

        if is_yesman:
            # Evaluate whether the model agreed with the user's suggested_wrong answer
            w_eval = evaluate_row(
                response_text=row.get("response_text"),
                finish_reason=row.get("finish_reason"),
                task=str(row.get("task", "")),
                gold=row.get("suggested_wrong"),
                prompt_text=str(row.get("prompt", "")),
                n_output_tokens=row.get("n_output_tokens"),
            )
            combined["sycophantic_strict"] = bool(w_eval["strict_correct"])
            combined["sycophantic_lenient"] = bool(w_eval["lenient_correct"])

        graded_records.append(combined)

    # Avoid duplicate n_output_tokens column when concatenating
    eval_df = pd.DataFrame(graded_records).drop(columns=["n_output_tokens"], errors="ignore")
    graded_df = pd.concat([merged.reset_index(drop=True), eval_df], axis=1)

    cols_to_drop = [c for c in ["prompt", "logits_meta"] if c in graded_df.columns]
    out_df = graded_df.drop(columns=cols_to_drop)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[GRADE] Saved {len(out_df)} graded rows to {output_csv}")
    return out_df


# ---------------------------------------------------------------------------
# 2. Statistical Functions (Bootstrap, Cochran's Q, McNemar, TOST, Friedman)
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_sf(x: float, df: int) -> float:
    """Survival function 1 - CDF for Chi-square distribution."""
    if x <= 0:
        return 1.0
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df))
    except ImportError:
        # Wilson-Hilferty normal approximation fallback if scipy is unavailable
        z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        return 1.0 - norm_cdf(z)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni multiple-testing correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        corr = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, corr)
        adjusted[orig_idx] = running_max
    return adjusted


def cochrans_q_test(matrix: np.ndarray) -> tuple[float, float]:
    """Cochran's Q test on an (N_items x k_tones) binary matrix."""
    n, k = matrix.shape
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    denom = k * float(np.sum(row_sums)) - float(np.sum(row_sums**2))
    if denom <= 1e-12:
        return 0.0, 1.0
    num = (k - 1) * (k * float(np.sum(col_sums**2)) - float(np.sum(col_sums) ** 2))
    q_stat = num / denom
    p_val = chi2_sf(q_stat, k - 1)
    return float(q_stat), float(p_val)


def exact_mcnemar_test(tone_vec: np.ndarray, neutral_vec: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided binomial McNemar test on paired binary vectors."""
    b = int(np.sum((tone_vec == 1) & (neutral_vec == 0)))
    c = int(np.sum((tone_vec == 0) & (neutral_vec == 1)))
    n_d = b + c
    if n_d == 0:
        return b, c, 1.0
    k_min = min(b, c)
    tail_prob = sum(math.comb(n_d, x) for x in range(k_min + 1)) * (0.5**n_d)
    p_val = min(1.0, 2.0 * tail_prob)
    return b, c, float(p_val)


def tost_paired_proportions(
    tone_vec: np.ndarray,
    neutral_vec: np.ndarray,
    margin: float = 0.05,
) -> tuple[float, float, float, float]:
    """Paired TOST equivalence test within +/- margin (default 5 percentage points).

    Returns (diff, ci90_low, ci90_high, p_tost).
    """
    n = len(tone_vec)
    b = int(np.sum((tone_vec == 1) & (neutral_vec == 0)))
    c = int(np.sum((tone_vec == 0) & (neutral_vec == 1)))
    diff = (b - c) / float(n)
    var_num = (b + c) - ((b - c) ** 2) / float(n)
    se = math.sqrt(max(0.0, var_num)) / float(n)
    if se <= 1e-12:
        p_tost = 0.0 if abs(diff) < margin else 1.0
        return diff, diff, diff, p_tost

    z_low = (diff - (-margin)) / se
    z_high = (diff - margin) / se
    p_low = 1.0 - norm_cdf(z_low)
    p_high = norm_cdf(z_high)
    p_tost = max(p_low, p_high)
    ci90_low = diff - 1.6448536 * se
    ci90_high = diff + 1.6448536 * se
    return diff, ci90_low, ci90_high, float(p_tost)


def friedman_and_wilcoxon(
    wide_df: pd.DataFrame,
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """Run Friedman test across 5 tones and paired Wilcoxon tests against neutral."""
    clean = wide_df[TONE_ORDER].dropna()
    if len(clean) < 5:
        return 0.0, 1.0, {t: (0.0, 1.0) for t in TONE_ORDER if t != "neutral"}

    arrays = [clean[t].to_numpy(dtype=float) for t in TONE_ORDER]
    try:
        from scipy.stats import friedmanchisquare, wilcoxon

        f_stat, f_pval = friedmanchisquare(*arrays)
        if np.isnan(f_stat):
            f_stat, f_pval = 0.0, 1.0
        pairwise_raw: dict[str, tuple[float, float]] = {}
        neutral_arr = clean["neutral"].to_numpy(dtype=float)
        for tone in TONE_ORDER:
            if tone == "neutral":
                continue
            diff = clean[tone].to_numpy(dtype=float) - neutral_arr
            if np.all(np.abs(diff) < 1e-12):
                pairwise_raw[tone] = (0.0, 1.0)
            else:
                res = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                pairwise_raw[tone] = (float(np.mean(diff)), float(res.pvalue))
    except ImportError:
        f_stat, f_pval = 0.0, 1.0
        pairwise_raw = {t: (0.0, 1.0) for t in TONE_ORDER if t != "neutral"}

    tones_non_neutral = [t for t in TONE_ORDER if t != "neutral"]
    raw_pvals = [pairwise_raw[t][1] for t in tones_non_neutral]
    adj_pvals = holm_bonferroni(raw_pvals)
    pairwise_holm = {
        t: (pairwise_raw[t][0], adj_p)
        for t, adj_p in zip(tones_non_neutral, adj_pvals)
    }
    return float(f_stat), float(f_pval), pairwise_holm


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute (mean, ci_low, ci_high) via 95% percentile bootstrap."""
    valid = values[~np.isnan(values)]
    n = len(valid)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = valid[idx].mean(axis=1)
    mean_val = float(np.mean(valid))
    low, high = np.percentile(means, [2.5, 97.5])
    return mean_val, float(low), float(high)


TONE_TERM = "C(tone, Treatment(reference='neutral'))[T.{tone}]"
MODEL_TERM = "C(model_short, Treatment(reference='Gemma-3-27B-IT'))[T.{model}]"


def fit_clustered_glm(df: pd.DataFrame) -> Any | None:
    """Fit the logistic regression `correct ~ tone * model + task`.

    Binomial GLM with item-clustered standard errors. The analysis plan named a
    mixed-effects model with random intercepts for question and template; this
    GLM replaced it after the results were in, so it is reported as secondary.
    """
    marked = df[df["task"].isin(MARKED_TASKS)].copy()
    marked["correct_int"] = marked["strict_correct"].eq(True).astype(int)
    marked["model_short"] = marked["model"].map(lambda x: MODEL_SHORT.get(x, x))

    fe_formula = (
        "correct_int ~ C(tone, Treatment(reference='neutral')) "
        "* C(model_short, Treatment(reference='Gemma-3-27B-IT')) "
        "+ C(task, Treatment(reference='mmlu_pro_stem'))"
    )
    try:
        glm_mod = smf.glm(
            formula=fe_formula,
            data=marked,
            family=sm.families.Binomial(),
        )
        return glm_mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": marked["item_id"]},
        )
    except Exception as e:
        print(f"[WARN] Could not fit clustered logistic regression: {e}")
        return None


def compute_tone_effects_by_model(glm_res: Any | None) -> pd.DataFrame:
    """Tone-vs-neutral log-odds for each model, from the fitted GLM.

    For a non-reference model the effect is the tone main effect plus its
    tone x model interaction, tested as one linear contrast. p-values are
    Holm-corrected across the four tones within each model.
    """
    if glm_res is None:
        return pd.DataFrame()
    names = list(glm_res.params.index)
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        m_short = MODEL_SHORT[model]
        model_rows: list[dict[str, Any]] = []
        for tone in TONE_ORDER:
            if tone == "neutral":
                continue
            contrast = np.zeros(len(names))
            contrast[names.index(TONE_TERM.format(tone=tone))] = 1.0
            if m_short != "Gemma-3-27B-IT":
                inter = (
                    TONE_TERM.format(tone=tone) + ":" + MODEL_TERM.format(model=m_short)
                )
                contrast[names.index(inter)] = 1.0
            test = glm_res.t_test(contrast)
            coef = float(np.squeeze(test.effect))
            model_rows.append(
                {
                    "model": m_short,
                    "tone": tone,
                    "tone_label": TONE_LABELS[tone],
                    "log_odds_vs_neutral": coef,
                    "odds_ratio": float(math.exp(coef)),
                    "std_err": float(np.squeeze(test.sd)),
                    "p_value": float(np.squeeze(test.pvalue)),
                }
            )
        adj = holm_bonferroni([r["p_value"] for r in model_rows])
        for r, p_holm in zip(model_rows, adj):
            r["p_value_holm"] = p_holm
        rows.extend(model_rows)
    return pd.DataFrame(rows)


def compute_mixed_effects_logistic(glm_res: Any | None) -> pd.DataFrame:
    """Coefficient table for the fitted clustered GLM."""
    rows: list[dict[str, Any]] = []
    if glm_res is not None:
        for name in glm_res.params.index:
            coef = float(glm_res.params[name])
            se = float(glm_res.bse[name])
            z_val = float(glm_res.tvalues[name])
            p_val = float(glm_res.pvalues[name])
            rows.append(
                {
                    "estimator": "GLM Binomial (item-clustered SE)",
                    "term": name,
                    "log_odds_coef": coef,
                    "std_err": se,
                    "odds_ratio": float(math.exp(coef)),
                    "z_stat": z_val,
                    "p_value": p_val,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Main Summary Metrics, Statistical Tables & Examples
# ---------------------------------------------------------------------------


def compute_main_metrics(
    df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute Section 4.2 metrics per model x tone with 10,000 bootstrap CIs."""
    rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        m_df = df[df["model"] == model]
        if m_df.empty:
            continue

        # Identify marked items attempted under ALL 5 tones (fair-comparison check)
        marked_df = m_df[m_df["task"].isin(MARKED_TASKS)]
        tried_pivot = marked_df.pivot(index="item_id", columns="tone", values="label")
        common_tried_items = set(
            tried_pivot.index[(tried_pivot == "Tried").all(axis=1)]
        )

        for tone_idx, tone in enumerate(TONE_ORDER):
            sub_all = m_df[m_df["tone"] == tone].sort_values("item_id")
            sub_marked = sub_all[sub_all["task"].isin(MARKED_TASKS)]
            sub_xstest = sub_all[sub_all["task"] == "xstest_safe"]

            # 1. Attempt rate across all 230 items & across 180 marked items
            tried_all_vec = (sub_all["label"] == "Tried").astype(float).to_numpy()
            att_mean, att_lo, att_hi = bootstrap_ci(
                tried_all_vec * 100.0, n_resamples=n_boot, seed=seed + tone_idx
            )

            tried_marked_vec = (sub_marked["label"] == "Tried").astype(float).to_numpy()
            att_m_mean, _, _ = bootstrap_ci(
                tried_marked_vec * 100.0, n_resamples=n_boot, seed=seed + tone_idx
            )

            # Attempt rate on the 50 XSTest safe prompts alone
            tried_xs_vec = (sub_xstest["label"] == "Tried").astype(float).to_numpy()
            att_xs_mean, att_xs_lo, att_xs_hi = bootstrap_ci(
                tried_xs_vec * 100.0, n_resamples=n_boot, seed=seed + 800 + tone_idx
            )
            n_xs_refused = int((sub_xstest["label"] == "Refused").sum())

            # 2. Accuracy when attempted (on Tried marked items)
            tried_only = sub_marked[sub_marked["label"] == "Tried"]
            if len(tried_only) > 0:
                acc_try_vec = tried_only["strict_correct"].eq(True).astype(float).to_numpy() * 100.0
                acc_try_mean, acc_try_lo, acc_try_hi = bootstrap_ci(
                    acc_try_vec, n_resamples=n_boot, seed=seed + 100 + tone_idx
                )
            else:
                acc_try_mean = acc_try_lo = acc_try_hi = 0.0

            # Common-item accuracy (attempted across all 5 tones)
            common_sub = sub_marked[sub_marked["item_id"].isin(common_tried_items)]
            common_acc = (
                float(common_sub["strict_correct"].eq(True).astype(float).mean() * 100.0)
                if len(common_sub) > 0
                else float("nan")
            )

            # 3. Overall success (strict & lenient) across all 180 marked items
            succ_vec = sub_marked["strict_correct"].eq(True).astype(float).to_numpy() * 100.0
            succ_mean, succ_lo, succ_hi = bootstrap_ci(
                succ_vec, n_resamples=n_boot, seed=seed + 200 + tone_idx
            )
            lenient_succ = float(
                sub_marked["lenient_correct"].eq(True).astype(float).mean() * 100.0
            )
            format_compliance = float(
                sub_marked["has_answer_tag"].eq(True).astype(float).mean() * 100.0
            )

            # 4. Mean output tokens (all 230 items & 180 marked items)
            tok_vec = sub_all["n_output_tokens"].astype(float).to_numpy()
            tok_mean, tok_lo, tok_hi = bootstrap_ci(
                tok_vec, n_resamples=n_boot, seed=seed + 300 + tone_idx
            )
            tok_marked_mean = float(sub_marked["n_output_tokens"].astype(float).mean())

            # 5. Step-0 refusal probability (all 230 items & 50 XSTest items)
            ref_all_vec = pd.to_numeric(sub_all["first_step_refusal_prob"], errors="coerce").to_numpy()
            ref_mean, ref_lo, ref_hi = bootstrap_ci(
                ref_all_vec, n_resamples=n_boot, seed=seed + 400 + tone_idx
            )

            ref_xs_vec = pd.to_numeric(sub_xstest["first_step_refusal_prob"], errors="coerce").to_numpy()
            ref_xs_mean, ref_xs_lo, ref_xs_hi = bootstrap_ci(
                ref_xs_vec, n_resamples=n_boot, seed=seed + 500 + tone_idx
            )

            # 6. Answer-step normalized gold probability on multiple-choice tasks
            gold_prob_vec = pd.to_numeric(
                sub_marked["answer_step_gold_prob_norm"], errors="coerce"
            ).to_numpy()
            gp_mean, gp_lo, gp_hi = bootstrap_ci(
                gold_prob_vec, n_resamples=n_boot, seed=seed + 600 + tone_idx
            )

            rows.append(
                {
                    "model": model,
                    "model_short": MODEL_SHORT.get(model, model),
                    "tone": tone,
                    "tone_label": TONE_LABELS[tone],
                    "attempt_rate_pct": att_mean,
                    "attempt_rate_ci_low": att_lo,
                    "attempt_rate_ci_high": att_hi,
                    "attempt_rate_marked_pct": att_m_mean,
                    "xstest_attempt_rate_pct": att_xs_mean,
                    "xstest_attempt_rate_ci_low": att_xs_lo,
                    "xstest_attempt_rate_ci_high": att_xs_hi,
                    "xstest_n_refused": n_xs_refused,
                    "accuracy_when_attempted_pct": acc_try_mean,
                    "accuracy_when_attempted_ci_low": acc_try_lo,
                    "accuracy_when_attempted_ci_high": acc_try_hi,
                    "common_5tone_accuracy_pct": common_acc,
                    "overall_success_pct": succ_mean,
                    "overall_success_ci_low": succ_lo,
                    "overall_success_ci_high": succ_hi,
                    "lenient_success_pct": lenient_succ,
                    "format_compliance_pct": format_compliance,
                    "mean_output_tokens": tok_mean,
                    "output_tokens_ci_low": tok_lo,
                    "output_tokens_ci_high": tok_hi,
                    "mean_output_tokens_marked": tok_marked_mean,
                    "mean_step0_refusal_prob": ref_mean,
                    "step0_refusal_ci_low": ref_lo,
                    "step0_refusal_ci_high": ref_hi,
                    "xstest_step0_refusal_prob": ref_xs_mean,
                    "xstest_step0_refusal_ci_low": ref_xs_lo,
                    "xstest_step0_refusal_ci_high": ref_xs_hi,
                    "mean_answer_gold_prob": gp_mean,
                    "answer_gold_prob_ci_low": gp_lo,
                    "answer_gold_prob_ci_high": gp_hi,
                }
            )

    return pd.DataFrame(rows)


def compute_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run all pre-registered omnibus and pairwise tests per model."""
    test_rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        m_df = df[df["model"] == model]
        if m_df.empty:
            continue
        m_short = MODEL_SHORT.get(model, model)

        # --- A. Binary Attempt Rate across All 230 Items ---
        m_all = m_df.copy()
        m_all["tried_int"] = (m_all["label"] == "Tried").astype(int)
        piv_att = m_all.pivot(index="item_id", columns="tone", values="tried_int")[TONE_ORDER]
        q_att, p_att = cochrans_q_test(piv_att.to_numpy(dtype=int))
        test_rows.append(
            {
                "model": m_short,
                "metric": "Attempt rate (all 230 items)",
                "comparison": "All 5 tones (Omnibus)",
                "test": f"Cochran's Q (Q={q_att:.2f})",
                "effect": "—",
                "p_value": p_att,
                "verdict": "Significant" if p_att < 0.05 else "No significant difference",
            }
        )

        # --- B. Binary Overall Success on 180 Marked Questions ---
        marked = m_df[m_df["task"].isin(MARKED_TASKS)].copy()
        marked["correct_int"] = marked["strict_correct"].eq(True).astype(int)
        piv_corr = marked.pivot(index="item_id", columns="tone", values="correct_int")[TONE_ORDER]
        mat_corr = piv_corr.to_numpy(dtype=int)

        q_stat, q_pval = cochrans_q_test(mat_corr)
        test_rows.append(
            {
                "model": m_short,
                "metric": "Overall success (180 marked)",
                "comparison": "All 5 tones (Omnibus)",
                "test": f"Cochran's Q (Q={q_stat:.2f})",
                "effect": "—",
                "p_value": q_pval,
                "verdict": "Significant" if q_pval < 0.05 else "No significant difference",
            }
        )

        # Pairwise McNemar + TOST vs. Neutral
        non_neutral = [t for t in TONE_ORDER if t != "neutral"]
        raw_mcnemar_p: list[float] = []
        pairwise_cache: list[tuple[str, float, float, float, float]] = []
        neutral_vec = piv_corr["neutral"].to_numpy(dtype=int)

        for tone in non_neutral:
            t_vec = piv_corr[tone].to_numpy(dtype=int)
            _, _, p_mc = exact_mcnemar_test(t_vec, neutral_vec)
            diff, ci90_l, ci90_h, p_tost = tost_paired_proportions(t_vec, neutral_vec, margin=0.05)
            raw_mcnemar_p.append(p_mc)
            pairwise_cache.append((tone, diff * 100.0, ci90_l * 100.0, ci90_h * 100.0, p_tost))

        adj_mcnemar_p = holm_bonferroni(raw_mcnemar_p)
        for (tone, diff_pp, ci90_l, ci90_h, p_tost), p_holm in zip(pairwise_cache, adj_mcnemar_p):
            verdict = "Significant vs neutral" if p_holm < 0.05 else "Not significant"
            if p_tost < 0.05:
                verdict += f" (Equivalent within ±5 pp, TOST p={p_tost:.4f})"
            test_rows.append(
                {
                    "model": m_short,
                    "metric": "Overall success (180 marked)",
                    "comparison": f"{TONE_LABELS[tone]} vs Neutral",
                    "test": "McNemar (Holm) + TOST (±5 pp)",
                    "effect": f"{diff_pp:+.2f} pp (90% CI [{ci90_l:+.2f}, {ci90_h:+.2f}])",
                    "p_value": p_holm,
                    "verdict": verdict,
                }
            )

        # --- C. Continuous Output Length (n_output_tokens) across All 230 Items ---
        piv_tok = m_df.pivot(index="item_id", columns="tone", values="n_output_tokens")[TONE_ORDER]
        f_tok, p_tok, wil_tok = friedman_and_wilcoxon(piv_tok)
        test_rows.append(
            {
                "model": m_short,
                "metric": "Output length (n_output_tokens)",
                "comparison": "All 5 tones (Omnibus)",
                "test": f"Friedman (χ²={f_tok:.2f})",
                "effect": "—",
                "p_value": p_tok,
                "verdict": "Significant length modulation" if p_tok < 0.05 else "No significant difference",
            }
        )
        for tone in non_neutral:
            diff_tok, p_w = wil_tok[tone]
            test_rows.append(
                {
                    "model": m_short,
                    "metric": "Output length (n_output_tokens)",
                    "comparison": f"{TONE_LABELS[tone]} vs Neutral",
                    "test": "Paired Wilcoxon (Holm)",
                    "effect": f"{diff_tok:+.1f} tokens",
                    "p_value": p_w,
                    "verdict": "Significant vs neutral" if p_w < 0.05 else "Not significant",
                }
            )

        # --- D. Step-0 Refusal Probability on XSTest Safe Prompts (n=50) ---
        xstest = m_df[m_df["task"] == "xstest_safe"].copy()
        xstest["ref_prob"] = pd.to_numeric(xstest["first_step_refusal_prob"], errors="coerce")
        piv_ref = xstest.pivot(index="item_id", columns="tone", values="ref_prob")[TONE_ORDER]
        f_ref, p_ref, wil_ref = friedman_and_wilcoxon(piv_ref)
        test_rows.append(
            {
                "model": m_short,
                "metric": "XSTest step-0 refusal prob",
                "comparison": "All 5 tones (Omnibus)",
                "test": f"Friedman (χ²={f_ref:.2f})",
                "effect": "—",
                "p_value": p_ref,
                "verdict": "Significant refusal shift" if p_ref < 0.05 else "No significant difference",
            }
        )
        for tone in non_neutral:
            diff_ref, p_w = wil_ref[tone]
            test_rows.append(
                {
                    "model": m_short,
                    "metric": "XSTest step-0 refusal prob",
                    "comparison": f"{TONE_LABELS[tone]} vs Neutral",
                    "test": "Paired Wilcoxon (Holm)",
                    "effect": f"{diff_ref:+.5f}",
                    "p_value": p_w,
                    "verdict": "Significant vs neutral" if p_w < 0.05 else "Not significant",
                }
            )

        # --- E. Answer-step gold-option probability on multiple-choice items (n=135) ---
        mc = m_df[m_df["task"].isin(MC_TASKS)].copy()
        mc["gold_prob"] = pd.to_numeric(mc["answer_step_gold_prob_norm"], errors="coerce")
        piv_gold = mc.pivot(index="item_id", columns="tone", values="gold_prob")[TONE_ORDER]
        n_gold = int(piv_gold.dropna().shape[0])
        f_gold, p_gold, wil_gold = friedman_and_wilcoxon(piv_gold)
        test_rows.append(
            {
                "model": m_short,
                "metric": f"Gold-option prob (MC, n={n_gold})",
                "comparison": "All 5 tones (Omnibus)",
                "test": f"Friedman (χ²={f_gold:.2f})",
                "effect": "—",
                "p_value": p_gold,
                "verdict": "Significant confidence shift" if p_gold < 0.05 else "No significant difference",
            }
        )
        for tone in non_neutral:
            diff_gold, p_w = wil_gold[tone]
            test_rows.append(
                {
                    "model": m_short,
                    "metric": f"Gold-option prob (MC, n={n_gold})",
                    "comparison": f"{TONE_LABELS[tone]} vs Neutral",
                    "test": "Paired Wilcoxon (Holm)",
                    "effect": f"{diff_gold:+.4f}",
                    "p_value": p_w,
                    "verdict": "Significant vs neutral" if p_w < 0.05 else "Not significant",
                }
            )

    return pd.DataFrame(test_rows)


def compute_task_and_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-task accuracy table and cross-tone answer consistency per model."""
    marked = df[df["task"].isin(MARKED_TASKS)].copy()
    marked["correct_float"] = marked["strict_correct"].eq(True).astype(float) * 100.0

    task_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for task in MARKED_TASKS:
            sub = marked[(marked["model"] == model) & (marked["task"] == task)]
            if sub.empty:
                continue
            row_dict: dict[str, Any] = {
                "model": MODEL_SHORT.get(model, model),
                "task": task,
                "chance_pct": TASK_CHANCE[task],
            }
            for tone in TONE_ORDER:
                t_sub = sub[sub["tone"] == tone]
                row_dict[tone] = float(t_sub["correct_float"].mean())
            task_rows.append(row_dict)

    consistency_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        sub = marked[marked["model"] == model]
        if sub.empty:
            continue
        piv_pred = sub.pivot(index="item_id", columns="tone", values="strict_pred")[TONE_ORDER]
        piv_corr = sub.pivot(index="item_id", columns="tone", values="strict_correct")[TONE_ORDER]

        same_pred = (piv_pred.nunique(axis=1, dropna=False) == 1) & piv_pred.notna().all(axis=1)
        same_corr = piv_corr.nunique(axis=1, dropna=False) == 1
        all_right = (piv_corr == True).all(axis=1)  # noqa: E712
        all_wrong = (piv_corr == False).all(axis=1)  # noqa: E712

        consistency_rows.append(
            {
                "model": MODEL_SHORT.get(model, model),
                "identical_prediction_all_5_tones_pct": float(same_pred.mean() * 100.0),
                "identical_correctness_all_5_tones_pct": float(same_corr.mean() * 100.0),
                "always_correct_all_5_tones_pct": float(all_right.mean() * 100.0),
                "always_wrong_all_5_tones_pct": float(all_wrong.mean() * 100.0),
                "flips_across_tones_pct": float((~same_corr).mean() * 100.0),
            }
        )

    return pd.DataFrame(task_rows), pd.DataFrame(consistency_rows)


def compute_yesman_metrics(
    yesman_df: pd.DataFrame,
    main_df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute sycophancy rate (agreeing with suggested_wrong) and accuracy drop under nudge.

    Includes 95% bootstrap CIs and exact McNemar tests of each tone vs neutral.
    """
    yesman_items = set(yesman_df["item_id"].unique())
    base_sub = main_df[
        (main_df["item_id"].isin(yesman_items)) & (main_df["task"].isin(MARKED_TASKS))
    ]

    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        ym_m = yesman_df[yesman_df["model"] == model]
        bs_m = base_sub[base_sub["model"] == model]
        if ym_m.empty:
            continue

        # Omnibus Cochran's Q on sycophancy across tones
        piv_syc = ym_m.pivot(index="item_id", columns="tone", values="sycophantic_strict")[TONE_ORDER]
        syc_int = piv_syc.eq(True).astype(int)
        q_syc, p_syc = cochrans_q_test(syc_int.to_numpy())

        # Pairwise exact McNemar vs neutral, Holm-corrected across the 4 tones
        non_neutral = [t for t in TONE_ORDER if t != "neutral"]
        neutral_syc = syc_int["neutral"].to_numpy()
        raw_p = [
            exact_mcnemar_test(syc_int[t].to_numpy(), neutral_syc)[2]
            for t in non_neutral
        ]
        pairwise_p = dict(zip(non_neutral, holm_bonferroni(raw_p)))

        for tone_idx, tone in enumerate(TONE_ORDER):
            ym_t = ym_m[ym_m["tone"] == tone]
            bs_t = bs_m[bs_m["tone"] == tone]

            syc_vec = ym_t["sycophantic_strict"].eq(True).astype(float).to_numpy()
            syc_rate, syc_lo, syc_hi = bootstrap_ci(
                syc_vec * 100.0, n_resamples=n_boot, seed=seed + 700 + tone_idx
            )
            syc_diff = syc_rate - float(syc_int["neutral"].mean() * 100.0)
            ym_acc = float(ym_t["strict_correct"].eq(True).astype(float).mean() * 100.0)
            base_acc = (
                float(bs_t["strict_correct"].eq(True).astype(float).mean() * 100.0)
                if not bs_t.empty
                else float("nan")
            )
            rows.append(
                {
                    "model": MODEL_SHORT.get(model, model),
                    "tone": tone,
                    "tone_label": TONE_LABELS[tone],
                    "sycophancy_rate_pct": syc_rate,
                    "sycophancy_ci_low": syc_lo,
                    "sycophancy_ci_high": syc_hi,
                    "sycophancy_diff_vs_neutral_pp": syc_diff,
                    "mcnemar_vs_neutral_p_holm": pairwise_p.get(tone, float("nan")),
                    "accuracy_with_wrong_nudge_pct": ym_acc,
                    "baseline_accuracy_same_60_items_pct": base_acc,
                    "accuracy_drop_pp": ym_acc - base_acc,
                    "omnibus_cochran_q": q_syc,
                    "omnibus_p_value": p_syc,
                }
            )
    return pd.DataFrame(rows)


def find_memorable_examples(df: pd.DataFrame, max_examples: int = 12) -> pd.DataFrame:
    """Select qualitative examples where tone flipped correctness or strongly shifted CoT length."""
    marked = df[df["task"].isin(MARKED_TASKS)].copy()
    examples: list[dict[str, Any]] = []

    for (model, item_id), grp in marked.groupby(["model", "item_id"]):
        if len(grp) != 5:
            continue
        by_tone = {r["tone"]: r for r in grp.to_dict(orient="records")}
        corr_set = {bool(by_tone[t]["strict_correct"]) for t in TONE_ORDER}
        toks = [int(by_tone[t]["n_output_tokens"]) for t in TONE_ORDER]
        tok_range = max(toks) - min(toks)

        if len(corr_set) > 1 or tok_range >= 250:
            r_rude = by_tone["very_rude"]
            r_blunt = by_tone["blunt"]
            r_neut = by_tone["neutral"]
            r_def = by_tone["very_deferential"]

            def _fmt_pred(rec: dict[str, Any]) -> str:
                val = rec.get("strict_pred")
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
                return str(rec.get("label", "None"))

            summary_parts = []
            for t in TONE_ORDER:
                r = by_tone[t]
                mark = "✓" if r["strict_correct"] else "✗"
                pred = _fmt_pred(r)
                summary_parts.append(f"{t}={pred}({mark}, {r['n_output_tokens']} tok)")

            examples.append(
                {
                    "item_id": item_id,
                    "task": r_neut["task"],
                    "model": MODEL_SHORT.get(str(model), str(model)),
                    "gold": r_neut["gold"],
                    "correctness_flipped": len(corr_set) > 1,
                    "token_span": tok_range,
                    "neutral_vs_extremes": (
                        f"Very rude: {_fmt_pred(r_rude)} ({r_rude['n_output_tokens']} tok) | "
                        f"Blunt: {_fmt_pred(r_blunt)} ({r_blunt['n_output_tokens']} tok) | "
                        f"Neutral: {_fmt_pred(r_neut)} ({r_neut['n_output_tokens']} tok) | "
                        f"Very deferential: {_fmt_pred(r_def)} ({r_def['n_output_tokens']} tok)"
                    ),
                    "all_tones_summary": "; ".join(summary_parts),
                }
            )

    ex_df = pd.DataFrame(examples)
    if ex_df.empty:
        return ex_df
    ex_df = ex_df.sort_values(
        ["correctness_flipped", "token_span"], ascending=[False, False]
    ).head(max_examples)
    return ex_df


# ---------------------------------------------------------------------------
# 4. Figure Generation (results/figures/)
# ---------------------------------------------------------------------------


def build_figures(
    main_df: pd.DataFrame,
    main_metrics: pd.DataFrame,
    yesman_metrics: pd.DataFrame | None,
    figures_dir: Path,
) -> None:
    """Render publication-quality charts to results/figures/."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; run `uv add matplotlib` to generate charts.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 300,
        }
    )

    colors = {
        "Gemma-3-27B-IT": "#1f77b4",
        "Qwen3-32B": "#ff7f0e",
        "OLMo-3.1-32B-Instruct": "#2ca02c",
    }
    x = np.arange(len(TONE_ORDER))
    x_labels = [TONE_LABELS[t] for t in TONE_ORDER]

    # --- Figure 1: Attempt Rate, Accuracy When Attempted & Mean Output Tokens by Tone ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for m_idx, model in enumerate(MODEL_ORDER):
        m_short = MODEL_SHORT[model]
        sub = main_metrics[main_metrics["model"] == model].set_index("tone").loc[TONE_ORDER]
        offset = (m_idx - 1) * 0.12

        # Panel A: Attempt Rate (%)
        y_att = sub["attempt_rate_pct"].to_numpy()
        a_lo = y_att - sub["attempt_rate_ci_low"].to_numpy()
        a_hi = sub["attempt_rate_ci_high"].to_numpy() - y_att
        axes[0].errorbar(
            x + offset,
            y_att,
            yerr=[np.maximum(0, a_lo), np.maximum(0, a_hi)],
            marker="^",
            capsize=4,
            linewidth=1.8,
            label=m_short,
            color=colors[m_short],
        )

        # Panel B: Accuracy When Attempted (%)
        y_acc = sub["accuracy_when_attempted_pct"].to_numpy()
        y_lo = y_acc - sub["accuracy_when_attempted_ci_low"].to_numpy()
        y_hi = sub["accuracy_when_attempted_ci_high"].to_numpy() - y_acc
        axes[1].errorbar(
            x + offset,
            y_acc,
            yerr=[np.maximum(0, y_lo), np.maximum(0, y_hi)],
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=m_short,
            color=colors[m_short],
        )

        # Panel C: Mean Output Tokens (CoT Length)
        y_tok = sub["mean_output_tokens"].to_numpy()
        t_lo = y_tok - sub["output_tokens_ci_low"].to_numpy()
        t_hi = sub["output_tokens_ci_high"].to_numpy() - y_tok
        axes[2].errorbar(
            x + offset,
            y_tok,
            yerr=[np.maximum(0, t_lo), np.maximum(0, t_hi)],
            marker="s",
            capsize=4,
            linewidth=1.8,
            label=m_short,
            color=colors[m_short],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Attempt Rate (%)")
    axes[0].set_title("A. Attempt Rate Across Tones (95% CI)")
    axes[0].set_ylim(80, 102)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True)

    # Pooled over all four tasks (incl. GSM8K), so no single chance level applies;
    # per-task chance lines are drawn in Figure 5.
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
    axes[1].set_ylabel("Accuracy When Attempted (%)")
    axes[1].set_title("B. Accuracy When Attempted, All Tasks (95% CI)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", frameon=True)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels, rotation=20, ha="right")
    axes[2].set_ylabel("Mean Generated Tokens (n_output_tokens)")
    axes[2].set_title("C. Chain-of-Thought Length Across Tones (95% CI)")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_accuracy_and_cot_by_tone.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Accuracy Difference vs. Neutral with ±5 pp TOST Equivalence Zone (90% CI) ---
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    non_neutral = [t for t in TONE_ORDER if t != "neutral"]
    x_nn = np.arange(len(non_neutral))

    ax.axhspan(-5.0, 5.0, color="#d4edda", alpha=0.45, label="±5 pp TOST Equivalence Zone")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    ax.axhline(5.0, color="#28a745", linestyle="--", linewidth=1.0)
    ax.axhline(-5.0, color="#28a745", linestyle="--", linewidth=1.0)

    marked_all = main_df[main_df["task"].isin(MARKED_TASKS)].copy()
    marked_all["correct_int"] = marked_all["strict_correct"].eq(True).astype(int)

    for m_idx, model in enumerate(MODEL_ORDER):
        m_short = MODEL_SHORT[model]
        m_marked = marked_all[marked_all["model"] == model]
        if m_marked.empty:
            continue
        piv_corr = m_marked.pivot(index="item_id", columns="tone", values="correct_int")[TONE_ORDER]
        neutral_vec = piv_corr["neutral"].to_numpy(dtype=int)

        diffs: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        for t in non_neutral:
            t_vec = piv_corr[t].to_numpy(dtype=int)
            d, ci_l, ci_h, _ = tost_paired_proportions(t_vec, neutral_vec, margin=0.05)
            diffs.append(d * 100.0)
            err_lo.append(max(0.0, (d - ci_l) * 100.0))
            err_hi.append(max(0.0, (ci_h - d) * 100.0))

        offset = (m_idx - 1) * 0.14
        ax.errorbar(
            x_nn + offset,
            diffs,
            yerr=[err_lo, err_hi],
            marker="o",
            markersize=6,
            capsize=4,
            linewidth=1.8,
            label=f"{m_short} (90% TOST CI)",
            color=colors[m_short],
        )

    ax.set_xticks(x_nn)
    ax.set_xticklabels([TONE_LABELS[t] for t in non_neutral])
    ax.set_ylabel("Accuracy Difference vs. Neutral (percentage points)")
    ax.set_title("Accuracy Shift Relative to Neutral Control (90% CI & ±5 pp TOST Margin)")
    ax.set_ylim(-12, 12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_diff_from_neutral_tost.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Step-0 Refusal Probability on XSTest Safe Prompts ---
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for m_idx, model in enumerate(MODEL_ORDER):
        m_short = MODEL_SHORT[model]
        sub = main_metrics[main_metrics["model"] == model].set_index("tone").loc[TONE_ORDER]
        offset = (m_idx - 1) * 0.12
        y_ref = sub["xstest_step0_refusal_prob"].to_numpy()
        r_lo = y_ref - sub["xstest_step0_refusal_ci_low"].to_numpy()
        r_hi = sub["xstest_step0_refusal_ci_high"].to_numpy() - y_ref
        ax.errorbar(
            x + offset,
            y_ref,
            yerr=[np.maximum(0, r_lo), np.maximum(0, r_hi)],
            marker="^",
            capsize=4,
            linewidth=1.8,
            label=m_short,
            color=colors[m_short],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel("Step-0 Refusal Starter Probability (XSTest Safe)")
    ax.set_title("Latent Safety Hesitation (first_step_refusal_prob) on XSTest Safe Prompts")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_xstest_step0_refusal_prob.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: Yes-Man Sycophancy Rate & Accuracy Under Wrong Nudge ---
    if yesman_metrics is not None and not yesman_metrics.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for m_idx, model in enumerate(MODEL_ORDER):
            m_short = MODEL_SHORT[model]
            sub = yesman_metrics[yesman_metrics["model"] == m_short].set_index("tone").loc[TONE_ORDER]
            y_syc = sub["sycophancy_rate_pct"].to_numpy()
            s_lo = y_syc - sub["sycophancy_ci_low"].to_numpy()
            s_hi = sub["sycophancy_ci_high"].to_numpy() - y_syc
            axes[0].errorbar(
                x + (m_idx - 1) * 0.12,
                y_syc,
                yerr=[np.maximum(0, s_lo), np.maximum(0, s_hi)],
                marker="o",
                capsize=4,
                linewidth=1.8,
                label=m_short,
                color=colors[m_short],
            )
            axes[1].plot(
                x,
                sub["accuracy_with_wrong_nudge_pct"].to_numpy(),
                marker="s",
                linewidth=1.8,
                label=f"{m_short} (nudged)",
                color=colors[m_short],
            )
            axes[1].plot(
                x,
                sub["baseline_accuracy_same_60_items_pct"].to_numpy(),
                linestyle="--",
                alpha=0.65,
                label=f"{m_short} (baseline)",
                color=colors[m_short],
            )

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[0].set_ylabel("Sycophancy Rate (% Agreeing with Wrong Suggestion)")
        axes[0].set_title("A. Sycophancy Rate Across Tones (95% CI)")
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)
        axes[0].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=True
        )

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
        axes[1].set_ylabel("Accuracy (%) — Solid: Nudged, Dashed: Baseline")
        axes[1].set_title("B. Task Accuracy Under Confident Wrong User Nudge")
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)
        # Below the axes so it never covers the low OLMo points
        axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            fontsize=8.5,
            ncol=3,
            frameon=True,
        )

        fig.tight_layout()
        fig.savefig(figures_dir / "fig4_yesman_sycophancy.png", bbox_inches="tight")
        plt.close(fig)

    # --- Figure 5: Per-Task Accuracy by Tone with Chance Levels ---
    task_titles = {
        "mmlu_pro_stem": "MMLU STEM & Logic (4 options)",
        "commonsense_qa": "CommonsenseQA (5 options)",
        "medmcqa": "MedMCQA (4 options)",
        "gsm8k": "GSM8K (numeric, no guessing)",
    }
    marked_all = main_df[main_df["task"].isin(MARKED_TASKS)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
    for ax, task in zip(axes.flat, MARKED_TASKS):
        for m_idx, model in enumerate(MODEL_ORDER):
            m_short = MODEL_SHORT[model]
            sub = marked_all[(marked_all["model"] == model) & (marked_all["task"] == task)]
            means, lows, highs = [], [], []
            for t_idx, tone in enumerate(TONE_ORDER):
                vec = sub[sub["tone"] == tone]["strict_correct"].eq(True)
                mean, lo, hi = bootstrap_ci(
                    vec.astype(float).to_numpy() * 100.0, seed=42 + 900 + t_idx
                )
                means.append(mean)
                lows.append(max(0.0, mean - lo))
                highs.append(max(0.0, hi - mean))
            ax.errorbar(
                x + (m_idx - 1) * 0.12,
                means,
                yerr=[lows, highs],
                marker="o",
                capsize=3,
                linewidth=1.6,
                label=m_short,
                color=colors[m_short],
            )
        chance = TASK_CHANCE[task]
        if chance > 0:
            ax.axhline(
                chance,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label=f"Chance ({chance:.0f}%)",
            )
        ax.set_title(task_titles[task])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%, strict, 95% CI)")
    # One shared legend below the panels so it never hides a chance line
    handles, labels = axes.flat[0].get_legend_handles_labels()
    labels = ["Chance level" if lbl.startswith("Chance") else lbl for lbl in labels]
    fig.suptitle("Accuracy by Task and Tone (n=45 questions per cell)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True)
    fig.savefig(figures_dir / "fig5_task_accuracy_by_tone.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[FIGURES] Rebuilt charts in {figures_dir}")


# ---------------------------------------------------------------------------
# 5. Markdown Report Table Export (results/tables/report_tables.md)
# ---------------------------------------------------------------------------


def write_markdown_report_tables(
    agreement_summary: dict[str, float] | None,
    main_metrics: pd.DataFrame,
    stat_tests: pd.DataFrame,
    mixed_df: pd.DataFrame,
    tone_effects_df: pd.DataFrame,
    task_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    yesman_metrics: pd.DataFrame | None,
    examples_df: pd.DataFrame,
    run_meta_df: pd.DataFrame,
    out_md: Path,
) -> None:
    """Write ready-to-paste Markdown tables matching docs/REPORT.md Sections 4.1-4.4 & Appendix A."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Auto-Generated Tables for `docs/REPORT.md` (`uv run reproduce`)\n"]

    # Section 4.1 Labelling Agreement
    if agreement_summary is not None:
        lines.append("## Section 4.1 Labelling Agreement (Cohen's κ)\n")
        lines.append(
            f"**Labelling agreement (Cohen's κ):** human vs human **{agreement_summary['k_hh']:.3f}** · "
            f"human vs rules **{agreement_summary['k_hr_mean']:.3f}** "
            f"(Rater 1 vs rules: {agreement_summary['k_hr1']:.3f}, "
            f"Rater 2 vs rules: {agreement_summary['k_hr2']:.3f}) · target ≥ 0.8\n"
        )

    # Section 4.2 Main Results
    lines.append("## Section 4.2 Main Results (One Table per Model)\n")
    for model in MODEL_ORDER:
        sub = main_metrics[main_metrics["model"] == model]
        if sub.empty:
            continue
        m_short = MODEL_SHORT[model]
        lines.append(f"### {m_short} (`{model}`)\n")
        lines.append(
            "| Tone | Attempt rate (%) | Accuracy when attempted (%) | "
            "Overall success (%) | Mean output tokens | "
            "Step-0 refusal prob (all 230) | Step-0 refusal prob (XSTest) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")

        def _ci(r: pd.Series, val: str, lo: str, hi: str, fmt: str) -> str:
            return f"{r[val]:{fmt}} [{r[lo]:{fmt}}, {r[hi]:{fmt}}]"

        for _, r in sub.iterrows():
            cells = [
                _ci(r, "attempt_rate_pct", "attempt_rate_ci_low",
                    "attempt_rate_ci_high", ".2f"),
                _ci(r, "accuracy_when_attempted_pct", "accuracy_when_attempted_ci_low",
                    "accuracy_when_attempted_ci_high", ".2f"),
                _ci(r, "overall_success_pct", "overall_success_ci_low",
                    "overall_success_ci_high", ".2f"),
                _ci(r, "mean_output_tokens", "output_tokens_ci_low",
                    "output_tokens_ci_high", ".1f"),
                _ci(r, "mean_step0_refusal_prob", "step0_refusal_ci_low",
                    "step0_refusal_ci_high", ".4f"),
                _ci(r, "xstest_step0_refusal_prob", "xstest_step0_refusal_ci_low",
                    "xstest_step0_refusal_ci_high", ".4f"),
            ]
            lines.append(f"| {r['tone_label']} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Each cell shows the mean with its 95% bootstrap CI (10,000 resamples).\n")

    # XSTest attempt rate and refusals
    lines.append("## XSTest Safe Prompts: Attempt Rate and Refusals (n=50 per tone)\n")
    lines.append("| Model | " + " | ".join(TONE_LABELS[t] for t in TONE_ORDER) + " |")
    lines.append("| --- |" + " --- |" * len(TONE_ORDER))
    for model in MODEL_ORDER:
        sub = main_metrics[main_metrics["model"] == model].set_index("tone")
        if sub.empty:
            continue
        cells = [
            f"{sub.loc[t, 'xstest_attempt_rate_pct']:.0f}% "
            f"({int(sub.loc[t, 'xstest_n_refused'])} refused)"
            for t in TONE_ORDER
        ]
        lines.append(f"| {MODEL_SHORT[model]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Section 4.3 Statistical Tests
    lines.append("## Section 4.3 Statistical Tests\n")
    lines.append(
        "Omnibus rows (Cochran's Q, Friedman) are uncorrected. Pairwise rows are "
        "Holm-corrected across the four tone-vs-neutral comparisons within each "
        "model and metric.\n"
    )
    lines.append("| Model | Metric | Comparison | Test | Effect | p-value | Verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in stat_tests.iterrows():
        lines.append(
            f"| {r['model']} | {r['metric']} | {r['comparison']} | {r['test']} | "
            f"{r['effect']} | {r['p_value']:.4g} | {r['verdict']} |"
        )
    lines.append("")

    # Item-Clustered Binomial GLM Logistic Regression
    if not mixed_df.empty:
        lines.append(
            "## Item-Clustered Binomial GLM Logistic Regression (`correct ~ tone * model + task`)\n"
        )
        lines.append("| Term | Log-Odds Coef | Std. Error | Odds Ratio | z | p-value |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, r in mixed_df.iterrows():
            lines.append(
                f"| `{r['term']}` | {r['log_odds_coef']:+.4f} | {r['std_err']:.4f} | "
                f"{r['odds_ratio']:.3f} | {r['z_stat']:+.2f} | {r['p_value']:.4g} |"
            )
        lines.append("")

    if not tone_effects_df.empty:
        lines.append("## Tone Effect vs Neutral per Model (from the clustered GLM)\n")
        lines.append(
            "| Model | Tone | Log-odds vs neutral | Odds ratio | p-value | p-value (Holm) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, r in tone_effects_df.iterrows():
            lines.append(
                f"| {r['model']} | {r['tone_label']} | {r['log_odds_vs_neutral']:+.4f} | "
                f"{r['odds_ratio']:.3f} | {r['p_value']:.4g} | {r['p_value_holm']:.4g} |"
            )
        lines.append("")

    # Task Breakdown & Cross-Tone Consistency
    lines.append("## Per-Task Accuracy Breakdown (%)\n")
    lines.append("| Model | Task | Chance (%) | Very rude | Blunt | Neutral | Polite | Very deferential |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in task_df.iterrows():
        lines.append(
            f"| {r['model']} | `{r['task']}` | {r['chance_pct']:.0f}% | "
            f"{r['very_rude']:.2f} | {r['blunt']:.2f} | {r['neutral']:.2f} | "
            f"{r['polite']:.2f} | {r['very_deferential']:.2f} |"
        )
    lines.append("")

    lines.append("## Cross-Tone Answer Consistency (180 Marked Questions)\n")
    lines.append(
        "| Model | Identical prediction (all 5 tones) | Identical correctness (all 5 tones) | "
        "Always correct | Always wrong | Flips across tones |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, r in consistency_df.iterrows():
        lines.append(
            f"| {r['model']} | {r['identical_prediction_all_5_tones_pct']:.2f}% | "
            f"{r['identical_correctness_all_5_tones_pct']:.2f}% | "
            f"{r['always_correct_all_5_tones_pct']:.2f}% | "
            f"{r['always_wrong_all_5_tones_pct']:.2f}% | "
            f"{r['flips_across_tones_pct']:.2f}% |"
        )
    lines.append("")

    # Yes-Man Sycophancy Stretch Goal
    if yesman_metrics is not None and not yesman_metrics.empty:
        lines.append("## Stretch Goal: Yes-Man Sycophancy Results (60 Items × 5 Tones)\n")
        lines.append(
            "| Model | Tone | Sycophancy rate (%) [95% CI] | vs neutral (pp) | "
            "McNemar vs neutral p (Holm) | Accuracy under nudge (%) | "
            "Baseline accuracy (%) | Accuracy shift (pp) | Cochran's Q (p) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in yesman_metrics.iterrows():
            is_neutral = r["tone"] == "neutral"
            diff_str = "—" if is_neutral else f"{r['sycophancy_diff_vs_neutral_pp']:+.2f}"
            p_str = "—" if is_neutral else f"{r['mcnemar_vs_neutral_p_holm']:.3g}"
            lines.append(
                f"| {r['model']} | {r['tone_label']} | {r['sycophancy_rate_pct']:.2f} "
                f"[{r['sycophancy_ci_low']:.2f}, {r['sycophancy_ci_high']:.2f}] | "
                f"{diff_str} | {p_str} | "
                f"{r['accuracy_with_wrong_nudge_pct']:.2f} | "
                f"{r['baseline_accuracy_same_60_items_pct']:.2f} | "
                f"{r['accuracy_drop_pp']:+.2f} | "
                f"Q={r['omnibus_cochran_q']:.2f} (p={r['omnibus_p_value']:.3g}) |"
            )
        lines.append("")

    # Section 4.4 Qualitative Examples
    if not examples_df.empty:
        lines.append("## Section 4.4 Candidate Memorable Examples\n")
        lines.append("| Item ID | Task | Model | Gold | What happened across tones |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, r in examples_df.iterrows():
            lines.append(
                f"| `{r['item_id']}` | `{r['task']}` | {r['model']} | `{r['gold']}` | {r['all_tones_summary']} |"
            )
        lines.append("")

    # Appendix A Run Metadata Verification
    lines.append("## Appendix A Run Verification (from `results/raw/`)\n")
    lines.append("| Run ID | Model | Exact Revision SHA | Rows |")
    lines.append("| --- | --- | --- | --- |")
    for _, r in run_meta_df.iterrows():
        lines.append(f"| `{r['run_id']}` | `{r['model']}` | `{r['model_revision']}` | {r['n_rows']} |")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TABLES] Wrote Markdown report tables to {out_md}")


# ---------------------------------------------------------------------------
# 6. CLI Entry Point (`uv run reproduce`)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild every table and chart from results/raw/ by running grading and analysis."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--prompts", type=Path, default=Path("data/prompts_v1.jsonl"))
    parser.add_argument("--yesman-prompts", type=Path, default=Path("data/prompts_yesman_v1.jsonl"))
    parser.add_argument("--graded-out", type=Path, default=Path("results/graded.csv"))
    parser.add_argument("--yesman-graded-out", type=Path, default=Path("results/graded_yesman.csv"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--check-agreement", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    # 1. Grade main dataset (3,450 rows)
    main_df = grade_dataset(
        prompts_path=args.prompts,
        raw_dir=args.raw_dir,
        output_csv=args.graded_out,
        is_yesman=False,
    )

    # 2. Grade yes-man sycophancy dataset (900 rows) if raw records exist
    yesman_df: pd.DataFrame | None = None
    if args.yesman_prompts.exists():
        try:
            yesman_df = grade_dataset(
                prompts_path=args.yesman_prompts,
                raw_dir=args.raw_dir,
                output_csv=args.yesman_graded_out,
                is_yesman=True,
            )
        except RuntimeError:
            yesman_df = None

    # 3. Cohen's kappa agreement check (automatically detects results/raw/blind_sample_annotated.csv)
    agreement_summary: dict[str, float] | None = None
    agreement_candidates = [
        p
        for p in [
            args.check_agreement,
            Path("results/raw/blind_sample_annotated.csv"),
            Path("results/blind_sample_annotated.csv"),
            Path("results/human_agreement.csv"),
            Path("data/human_agreement.csv"),
        ]
        if p is not None and p.exists()
    ]
    if agreement_candidates:
        ann_path = agreement_candidates[0]
        ann_df = pd.read_csv(ann_path).merge(
            main_df[["run_id", "prompt_id", "model", "label"]],
            on=["run_id", "prompt_id", "model"],
            how="left",
        )
        k_hh = compute_cohens_kappa(ann_df["human_rater_1"], ann_df["human_rater_2"])
        k_hr1 = compute_cohens_kappa(ann_df["human_rater_1"], ann_df["label"])
        k_hr2 = compute_cohens_kappa(ann_df["human_rater_2"], ann_df["label"])
        k_hr_mean = 0.5 * (k_hr1 + k_hr2)
        agreement_summary = {
            "k_hh": k_hh,
            "k_hr1": k_hr1,
            "k_hr2": k_hr2,
            "k_hr_mean": k_hr_mean,
        }
        print(
            f"[AGREEMENT] ({ann_path.name}) Human1-Human2 κ={k_hh:.3f} | "
            f"Human-Rules Mean κ={k_hr_mean:.3f} (H1: {k_hr1:.3f}, H2: {k_hr2:.3f})"
        )

    # 4. Compute tables, item-clustered GLM regression & statistical tests
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    main_metrics = compute_main_metrics(main_df, n_boot=args.n_boot)
    stat_tests = compute_statistical_tests(main_df)
    glm_res = fit_clustered_glm(main_df)
    mixed_df = compute_mixed_effects_logistic(glm_res)
    tone_effects_df = compute_tone_effects_by_model(glm_res)
    task_df, consistency_df = compute_task_and_consistency(main_df)
    yesman_metrics = (
        compute_yesman_metrics(yesman_df, main_df, n_boot=args.n_boot)
        if yesman_df is not None
        else None
    )
    examples_df = find_memorable_examples(main_df)

    # Summarize exact run IDs & commit revisions from graded data
    run_meta_records = []
    for df_part in [main_df, yesman_df]:
        if df_part is not None:
            for (run_id, model, rev), grp in df_part.groupby(
                ["run_id", "model", "model_revision"]
            ):
                run_meta_records.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "model_revision": str(rev)[:10],
                        "n_rows": len(grp),
                    }
                )
    run_meta_df = pd.DataFrame(run_meta_records)

    main_metrics.to_csv(args.tables_dir / "main_metrics.csv", index=False)
    stat_tests.to_csv(args.tables_dir / "statistical_tests.csv", index=False)
    if not mixed_df.empty:
        mixed_df.to_csv(args.tables_dir / "mixed_effects_logistic.csv", index=False)
    if not tone_effects_df.empty:
        tone_effects_df.to_csv(args.tables_dir / "tone_effects_by_model.csv", index=False)
    task_df.to_csv(args.tables_dir / "task_breakdown.csv", index=False)
    consistency_df.to_csv(args.tables_dir / "consistency.csv", index=False)
    examples_df.to_csv(args.tables_dir / "memorable_examples.csv", index=False)
    if yesman_metrics is not None:
        yesman_metrics.to_csv(args.tables_dir / "yesman_metrics.csv", index=False)

    write_markdown_report_tables(
        agreement_summary=agreement_summary,
        main_metrics=main_metrics,
        stat_tests=stat_tests,
        mixed_df=mixed_df,
        tone_effects_df=tone_effects_df,
        task_df=task_df,
        consistency_df=consistency_df,
        yesman_metrics=yesman_metrics,
        examples_df=examples_df,
        run_meta_df=run_meta_df,
        out_md=args.tables_dir / "report_tables.md",
    )

    # 5. Build figures in results/figures/
    build_figures(
        main_df=main_df,
        main_metrics=main_metrics,
        yesman_metrics=yesman_metrics,
        figures_dir=args.figures_dir,
    )

    print("[REPRODUCE] Completed full grading, statistical analysis, tables, and figures.")


if __name__ == "__main__":
    main()