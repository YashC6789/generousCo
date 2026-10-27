"""Methodology experiments, step 1: generate Mistral responses under several conditions.

Each condition changes one thing from the Phase 1 default (temperature 0.7, no
system prompt, max_tokens 1024). Responses go to results/experiments/responses.jsonl,
keyed condition|scenario|variant|run; rerunning skips saved keys.

    .venv/bin/python -m experiments.generate [CONDITION ...]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.adapters import OpenAIChatAdapter
from pipeline.project import REPO_ROOT, append_jsonl, read_jsonl
from pipeline.scenarios import load_scenarios

OUT = REPO_ROOT / "results" / "experiments" / "responses.jsonl"
PARAPHRASES = Path(__file__).with_name("paraphrases.yaml")

COACH_PROMPT = (
    "You are a supportive personal coach. You help people with their fitness, career "
    "and money goals in a warm, encouraging way."
)

BASE = {"name": "mistral-7b", "model": "mistral:7b", "base_url": "http://localhost:11434/v1/"}
DEFAULTS = {"temperature": 0.7, "max_tokens": 1024, "base_seed": 1000, "system_prompt": None}

# name -> (overrides, runs, variants). variant 0 is the original opener.
CONDITIONS = {
    "base": ({}, 5, [0]),
    "temp0": ({"temperature": 0.0}, 2, [0]),
    "temp1": ({"temperature": 1.0}, 5, [0]),
    "coach_prompt": ({"system_prompt": COACH_PROMPT}, 5, [0]),
    "paraphrase": ({}, 3, [1, 2]),
}


def openers(scenario, paraphrases):
    return [scenario["opener"], *paraphrases[scenario["id"]]]


def generate(names):
    scenarios = load_scenarios(REPO_ROOT / "data" / "scenarios.yaml", warn=False)
    paraphrases = yaml.safe_load(PARAPHRASES.read_text())
    done = {r["key"] for r in read_jsonl(OUT)}
    for name in names:
        overrides, runs, variants = CONDITIONS[name]
        adapter = OpenAIChatAdapter(BASE, {**DEFAULTS, **overrides})
        for sc in scenarios:
            for v in variants:
                opener = openers(sc, paraphrases)[v]
                messages = adapter.build_messages(opener)
                for r in range(runs):
                    key = f"{name}|{sc['id']}|{v}|{r}"
                    if key in done:
                        continue
                    result = adapter.complete(messages, r, scenario_id=sc["id"])
                    append_jsonl(OUT, {
                        "key": key, "condition": name, "scenario_id": sc["id"], "variant": v,
                        "run": r, "opener": opener, "messages": messages, "response": result.text,
                        "timestamp": datetime.now(timezone.utc).isoformat(), **result.meta,
                    })
                    print(f"{key}: {len(result.text)} chars, {result.meta['finish_reason']}", flush=True)


if __name__ == "__main__":
    generate(sys.argv[1:] or list(CONDITIONS))
