"""Step 4: score every saved response with a local judge.

ELEPHANT's validation, indirectness and framing prompts run on every response,
and our corrected-error check runs on the factual-error rows. As in ELEPHANT,
the score is the first 0 or 1 in the judge's reply. A reply with no 0/1 is saved
as a parse failure (score null) and retried on the next `judge` run.
"""

import os
import re
import sys
from datetime import datetime, timezone

from .prompts import (
    CORRECTED_ERROR_SYSTEM_MESSAGE,
    ELEPHANT_DIMENSIONS,
    ELEPHANT_SYSTEM_MESSAGE,
    corrected_error_prompt,
    elephant_prompt,
)
from .project import append_jsonl, read_jsonl
from .scenarios import load_scenarios

ALL_DIMENSIONS = ELEPHANT_DIMENSIONS + ("corrected_error",)


def parse_score(text):
    m = re.search(r"[01]", text or "")
    return int(m.group(0)) if m else None


class Judge:
    def __init__(self, cfg, client=None):
        self.model = cfg["model"]
        self.base_url = cfg["base_url"]
        self.settings = {
            "temperature": cfg.get("temperature", 0),
            "max_tokens": cfg.get("max_tokens", 2),
            "seed": cfg.get("seed", 0),
        }
        if client is None:
            from openai import OpenAI

            key_env = cfg.get("api_key_env")
            client = OpenAI(base_url=self.base_url, api_key=os.environ.get(key_env) if key_env else "ollama")
        self.client = client

    def score(self, dimension, question, response, factual_error=None):
        if dimension == "corrected_error":
            system = CORRECTED_ERROR_SYSTEM_MESSAGE
            prompt = corrected_error_prompt(question, response, factual_error)
        else:
            system = ELEPHANT_SYSTEM_MESSAGE
            prompt = elephant_prompt(dimension, question, response)
        return self.ask(system, prompt)

    def ask(self, system, prompt):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **self.settings,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return parse_score(raw), raw


def dimensions_for(scenario):
    return ALL_DIMENSIONS if scenario.get("factual_error") else ELEPHANT_DIMENSIONS


def latest_scores(records):
    """Latest record per (response, dimension, judge model). Later lines win."""
    latest = {}
    for r in records:
        latest[r["key"]] = r
    return list(latest.values())


def judge_all(project, client=None):
    cfg = project.systems_config()["judge"]
    judge = Judge(cfg, client=client)
    scenarios = {s["id"]: s for s in load_scenarios(project.scenarios_file)}
    new = failed = skipped = 0

    for private in (False, True):
        responses = read_jsonl(project.responses_file(private))
        out = project.judge_file(private)
        done = {r["key"] for r in latest_scores(read_jsonl(out)) if r["score"] is not None}
        for resp in responses:
            sc = scenarios.get(resp["scenario_id"])
            if sc is None:
                print(f"skipping {resp['key']}: scenario not in scenarios.yaml", file=sys.stderr)
                continue
            for dim in dimensions_for(sc):
                key = f"{resp['key']}|{dim}|{judge.model}"
                if key in done:
                    skipped += 1
                    continue
                score, raw = judge.score(dim, sc["opener"], resp["response"], sc.get("factual_error"))
                append_jsonl(
                    out,
                    {
                        "key": key,
                        "response_key": resp["key"],
                        "dimension": dim,
                        "score": score,
                        "raw": raw,
                        "judge_model": judge.model,
                        "judge_base_url": judge.base_url,
                        **{f"judge_{k}": v for k, v in judge.settings.items()},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                new += 1
                if score is None:
                    failed += 1
                    print(f"PARSE FAIL {key}: {raw!r}", file=sys.stderr)

    print(f"Judged {new} new ({skipped} already scored). Parse failures this run: {failed}.")
    return check(project, judge.model)


def check(project, judge_model):
    """Every response has a parsed score on every dimension it needs."""
    scenarios = {s["id"]: s for s in load_scenarios(project.scenarios_file, warn=False)}
    scores = {r["key"]: r for r in latest_scores(project.all_judge_scores())}
    missing, unparsed = [], []
    for resp in project.all_responses():
        sc = scenarios.get(resp["scenario_id"])
        if sc is None:
            continue
        for dim in dimensions_for(sc):
            rec = scores.get(f"{resp['key']}|{dim}|{judge_model}")
            if rec is None:
                missing.append((resp["key"], dim))
            elif rec["score"] is None:
                unparsed.append((resp["key"], dim))
    if missing or unparsed:
        print(f"CHECK FAILED: {len(missing)} missing, {len(unparsed)} unparsed judge scores.")
        print("  Rerun `judge` to retry parse failures. If they persist, the judge model is not")
        print("  answering with 0/1 in max_tokens; try a different judge model or raise max_tokens.")
        return False
    print(f"CHECK OK: every response has a parsed score on every dimension ({judge_model}).")
    return True
