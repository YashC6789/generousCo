"""Step 5: does the local judge separate clear cases?

Two checks:
1. ELEPHANT's 10-example samples (OEQ and AITA-YTA). We judge the human responses
   and compare with the `*_human` labels shipped with each sample file. Those labels
   are ELEPHANT's gpt-4o judge outputs on the human responses (one is the string
   "ERROR", which their scorer writes on a failed call), so this measures agreement
   with gpt-4o, not with human annotators. Labels other than 0/1 are skipped.
2. Our hand-written contrast replies (data/judge_sanity_pairs.yaml): one clearly
   flagged and one clearly not flagged reply per question and dimension.
"""

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

from .judge import Judge
from .prompts import ELEPHANT_DIMENSIONS

ELEPHANT_SAMPLES = {
    "OEQ": ("prompt", "human"),
    "AITA-YTA": ("prompt", "top_comment"),
}


def run_sanity(project, client=None):
    judge = Judge(project.systems_config()["judge"], client=client)
    project.sanity_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (q_col, r_col) in ELEPHANT_SAMPLES.items():
        df = pd.read_csv(project.elephant_dir / "sample_datasets" / f"{name}_sample.csv")
        for i, row in df.iterrows():
            for dim in ELEPHANT_DIMENSIONS:
                expected = _label(row[f"{dim}_human"])
                if expected is None:
                    continue
                score, raw = judge.score(dim, row[q_col], row[r_col])
                rows.append(
                    {"source": f"ELEPHANT {name} (gpt-4o labels)", "item": i, "dimension": dim,
                     "expected": expected, "judge": score, "raw": raw}
                )

    pairs = yaml.safe_load((project.root / "data" / "judge_sanity_pairs.yaml").read_text())
    for dim, items in pairs.items():
        for i, item in enumerate(items):
            for reply in item["replies"]:
                score, raw = judge.score(dim, item["question"], reply["text"], item.get("factual_error"))
                rows.append(
                    {"source": "contrast pairs", "item": i, "dimension": dim,
                     "expected": reply["expected"], "judge": score, "raw": raw}
                )

    df = pd.DataFrame(rows)
    df.to_csv(project.sanity_dir / f"sanity_{judge.model.replace(':', '_')}.csv", index=False)
    summary = summarize(df)
    summary.to_csv(project.sanity_dir / f"summary_{judge.model.replace(':', '_')}.csv", index=False)
    print(f"Judge: {judge.model}\n")
    print(summary.to_string(index=False))

    contrast = df[df.source == "contrast pairs"]
    ok = bool((contrast.judge == contrast.expected).all())
    print()
    if ok:
        print("CHECK OK: the judge separates every hand-written contrast pair.")
    else:
        bad = contrast[contrast.judge != contrast.expected]
        print(f"CHECK FAILED: {len(bad)} contrast replies scored against expectation:")
        print(bad[["dimension", "item", "expected", "judge", "raw"]].to_string(index=False))
    return ok


def summarize(df):
    out = []
    for (source, dim), g in df.groupby(["source", "dimension"], sort=False):
        parsed = g.dropna(subset=["judge"])
        y_true, y_pred = parsed.expected.astype(int), parsed.judge.astype(int)
        out.append(
            {
                "source": source,
                "dimension": dim,
                "n": len(g),
                "parse_fail": len(g) - len(parsed),
                "accuracy": round((y_true == y_pred).mean(), 2) if len(parsed) else None,
                "kappa": _kappa(y_true, y_pred),
                "expected_rate": round(y_true.mean(), 2),
                "judge_rate": round(y_pred.mean(), 2) if len(parsed) else None,
            }
        )
    return pd.DataFrame(out)


def _label(value):
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return None
    return v if v in (0, 1) else None


def _kappa(a, b):
    if len(a) < 2 or (len(set(a)) == 1 and len(set(b)) == 1):
        return None  # undefined when both raters give a single constant label
    return round(cohen_kappa_score(a, b), 2)
