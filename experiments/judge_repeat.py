"""Judge repeatability: re-score the base condition with the same judge and settings.

Writes results/experiments/judge_repeat.jsonl; compare with judge_scores.jsonl.

    .venv/bin/python -m experiments.judge_repeat JUDGE_MODEL
"""

import sys

from experiments import judge as J
from pipeline.project import REPO_ROOT

if __name__ == "__main__":
    J.OUT = REPO_ROOT / "results" / "experiments" / "judge_repeat.jsonl"
    J.judge(sys.argv[1], ["base"], dims=J.BINARY + ["corrected_error"])
