"""Methodology experiments, step 3: tables for the methods write-up.

Reads results/experiments/{responses,judge_scores}.jsonl and pilot_labels_claude.csv,
writes results/experiments/tables/*.csv and prints a summary. No model calls.

    .venv/bin/python -m experiments.analyze
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pipeline.project import REPO_ROOT, read_jsonl

EXP = REPO_ROOT / "results" / "experiments"
TABLES = EXP / "tables"
BINARY = ["validation", "indirectness", "framing", "indirect_action", "moral_wrong", "corrected_error"]
WARMTH = ["supportiveness", "emotion_ack", "non_judgmental", "actionable_plan", "avoids_unsafe",
          "avoids_toxic_positivity"]
RNG = np.random.default_rng(8813)


def load():
    resp = pd.DataFrame(read_jsonl(EXP / "responses.jsonl"))
    resp["length"] = resp.response.str.len()
    scores = pd.DataFrame(read_jsonl(EXP / "judge_scores.jsonl"))
    scores = scores.drop_duplicates("key", keep="last")
    binary = scores[scores.dimension != "warmth"].dropna(subset=["score"]).copy()
    binary["score"] = binary.score.astype(int)
    long = binary[["response_key", "judge_model", "dimension", "score"]]

    w = scores[scores.dimension == "warmth"].dropna(subset=["score"])
    if len(w):
        wf = pd.json_normalize(w.score.tolist())
        wf.index = w.index
        w = pd.concat([w[["response_key", "judge_model"]], wf], axis=1)
        w = w.melt(["response_key", "judge_model"], var_name="dimension", value_name="score")
        long = pd.concat([long, w])
    long = long.merge(resp[["key", "condition", "scenario_id", "variant", "run"]],
                      left_on="response_key", right_on="key").drop(columns="key")
    return resp, scores, long


def wide(long, judge):
    d = long[long.judge_model == judge]
    return d.pivot_table(index=["response_key", "condition", "scenario_id", "variant", "run"],
                         columns="dimension", values="score").reset_index()


def add_derived(w):
    w = w.copy()
    w["complaisant"] = ((w.validation == 1) & (w.framing == 1)).astype(float)
    w["accepts_framing_any"] = w.framing
    return w


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or (len(set(a)) == 1 and len(set(b)) == 1):
        return np.nan
    return cohen_kappa_score(a, b)


def judge_agreement(long, judges):
    rows = []
    for dim in BINARY:
        for j1, j2 in itertools.combinations(judges, 2):
            d = long[long.dimension == dim].pivot_table(index="response_key", columns="judge_model",
                                                        values="score")
            if j1 not in d or j2 not in d:
                continue
            d = d[[j1, j2]].dropna()
            rows.append({"dimension": dim, "judge_a": j1, "judge_b": j2, "n": len(d),
                         "agree": round((d[j1] == d[j2]).mean(), 2),
                         "kappa": round(kappa(d[j1], d[j2]), 2),
                         "rate_a": round(d[j1].mean(), 2), "rate_b": round(d[j2].mean(), 2)})
    return pd.DataFrame(rows)


def vs_pilot(long, judges):
    path = EXP / "pilot_labels_claude.csv"
    if not path.exists():
        return pd.DataFrame()
    pilot = pd.read_csv(path)
    rows = []
    for dim in BINARY:
        ref = pilot[["response_key", dim]].dropna()
        for j in judges:
            d = long[(long.judge_model == j) & (long.dimension == dim)][["response_key", "score"]]
            m = ref.merge(d, on="response_key")
            if m.empty:
                continue
            rows.append({"dimension": dim, "judge": j, "n": len(m),
                         "agree": round((m[dim] == m.score).mean(), 2),
                         "kappa": round(kappa(m[dim], m.score), 2),
                         "pilot_rate": round(m[dim].mean(), 2), "judge_rate": round(m.score.mean(), 2)})
    return pd.DataFrame(rows)


def run_stability(w, dims):
    """Base condition (5 runs): how much does the overall rate move with k runs per scenario?"""
    base = w[w.condition == "base"]
    rows = []
    for dim in dims:
        if dim not in base:
            continue
        piv = base.pivot_table(index="scenario_id", columns="run", values=dim)
        full = piv.mean().mean()
        unanimous = (piv.nunique(axis=1) == 1).mean()
        for k in range(1, piv.shape[1] + 1):
            est = [piv[RNG.choice(piv.columns, k, replace=False)].mean(axis=1).mean() for _ in range(2000)]
            rows.append({"dimension": dim, "runs": k, "rate_5runs": round(full, 3),
                         "sd_of_rate": round(np.std(est), 3),
                         "scenarios_unanimous_across_5": round(unanimous, 2)})
    return pd.DataFrame(rows)


def condition_effects(w, dims, ref="base"):
    """Rate per condition and difference from base, with a bootstrap CI over scenarios."""
    rows = []
    for cond in w.condition.unique():
        for dim in dims:
            if dim not in w:
                continue
            a = w[w.condition == cond].groupby("scenario_id")[dim].mean()
            b = w[w.condition == ref].groupby("scenario_id")[dim].mean()
            ids = a.index.intersection(b.index)
            if a.dropna().empty:
                continue
            diff = (a[ids] - b[ids]).dropna()
            boots = [diff.iloc[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            rows.append({"condition": cond, "dimension": dim, "rate": round(a.mean(), 3),
                         "diff_vs_base": round(diff.mean(), 3),
                         "ci_low": round(np.percentile(boots, 2.5), 3),
                         "ci_high": round(np.percentile(boots, 97.5), 3)})
    return pd.DataFrame(rows)


def paraphrase_sensitivity(w, dims):
    """Per scenario: rate on the original opener (base, runs 0-2) vs each rewording."""
    orig = w[(w.condition == "base") & (w.run < 3)].assign(variant=0)
    para = w[w.condition == "paraphrase"]
    both = pd.concat([orig, para])
    rows = []
    for dim in dims:
        if dim not in both:
            continue
        piv = both.pivot_table(index="scenario_id", columns="variant", values=dim)
        spread = (piv.max(axis=1) - piv.min(axis=1))
        rows.append({"dimension": dim, "rate_v0": round(piv[0].mean(), 3),
                     "rate_v1": round(piv.get(1, pd.Series(dtype=float)).mean(), 3),
                     "rate_v2": round(piv.get(2, pd.Series(dtype=float)).mean(), 3),
                     "mean_within_scenario_range": round(spread.mean(), 3),
                     "scenarios_range_ge_0.67": int((spread >= 0.67).sum())})
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    resp, scores, long = load()
    judges = sorted(long.judge_model.unique())
    out = {}

    out["generation"] = resp.groupby("condition").agg(
        n=("key", "size"), mean_chars=("length", "mean"),
        truncated=("finish_reason", lambda s: int((s == "length").sum()))).round(0).reset_index()
    t0 = resp[resp.condition == "temp0"].pivot_table(index="scenario_id", columns="run",
                                                       values="response", aggfunc="first")
    if t0.shape[1] >= 2:
        out["generation"]["temp0_identical_pairs"] = ""
        out["generation"].loc[out["generation"].condition == "temp0", "temp0_identical_pairs"] = \
            f"{int((t0[0] == t0[1]).sum())}/{len(t0)}"

    pf = scores.assign(failed=scores.score.isna()).groupby(["judge_model", "dimension"]).failed.sum()
    out["parse_failures"] = pf[pf > 0].reset_index()
    out["judge_agreement"] = judge_agreement(long, judges)
    out["judge_vs_pilot"] = vs_pilot(long, judges)

    dims = BINARY + ["complaisant"] + WARMTH
    for j in judges:
        w = add_derived(wide(long, j))
        tag = j.replace(":", "_")
        out[f"rates_by_condition__{tag}"] = condition_effects(w, dims)
        out[f"run_stability__{tag}"] = run_stability(w, dims)
        out[f"paraphrase__{tag}"] = paraphrase_sensitivity(w, dims)
        out[f"by_scenario_base__{tag}"] = (w[w.condition == "base"].groupby("scenario_id")
                                           [[d for d in dims if d in w]].mean().round(2).reset_index())

    for name, t in out.items():
        t.to_csv(TABLES / f"{name}.csv", index=False)
        print(f"\n== {name} ==")
        print(t.to_string(index=False) if len(t) else "(empty)")


if __name__ == "__main__":
    main()
