"""Run the pipeline's judge sanity check (step 5) for each candidate judge model.

    .venv/bin/python -m experiments.sanity_judges MODEL [MODEL ...]
"""

import sys

from pipeline.project import Project
from pipeline.sanity import run_sanity


class JudgeOverride(Project):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def systems_config(self):
        cfg = super().systems_config()
        cfg["judge"]["model"] = self.model
        return cfg


if __name__ == "__main__":
    for model in sys.argv[1:]:
        run_sanity(JudgeOverride(model))
        print("\n" + "=" * 70 + "\n")
