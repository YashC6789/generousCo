# Phase 1: ELEPHANT baseline on Generous Co

Runs 12 coaching openers through a system, scores the replies with ELEPHANT's
social-sycophancy prompts on a local judge, has two annotators score the same replies
blind, and tabulates everything by system, core scenario and family.

Only the system adapter changes when Generous API access arrives: add one entry to
`config/systems.yaml`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install ollama            # or the installer from ollama.com
ollama serve                   # leave running in another terminal
ollama pull llama3.1:8b        # stand-in
ollama pull mistral:7b         # stand-in
ollama pull qwen2.5:7b         # judge: a different family from every stand-in
```

All commands are `.venv/bin/python -m pipeline <step>`.

## Steps

| Step | Command | Check |
|---|---|---|
| 1. Scenario file | `scenarios` | 12 rows load; lists the factual-error rows |
| 2-3. Adapter + runner | `run` (`--systems NAME ...`, `--runs N`) | rerunning prints `0 new`; every response is saved with model, version, settings, seed and timestamp |
| 4. Judge | `judge` | `CHECK OK`: every response has a parsed 0/1 on every dimension; parse failures are retried on the next run |
| 5. Judge sanity | `sanity` | the judge separates every hand-written contrast pair; also reports agreement with ELEPHANT's sample labels |
| 6. Annotation | `annotate export BATCH`, then `annotate kappa` | Cohen's kappa A vs B, then judge vs each annotator, against the thresholds in `config/systems.yaml` |
| 7. Analysis | `analyze` | rebuilds every table in `results/tables/` from saved files, with no model calls |
| 8. Human-coach reference | paste replies into `data/pasted/human-coach.csv`, then `run --systems human-coach` | set `analysis.reference_system: human-coach` to get difference-from-human columns |

## What is where

```
data/scenarios.yaml           the 12 openers (PLACEHOLDERS until replaced)
data/judge_sanity_pairs.yaml  hand-written contrast replies for step 5
data/pasted/                  pasted-in responses (consumer apps, human coach); copy TEMPLATE.csv
config/systems.yaml           systems under test, judge, agreement thresholds
config/annotation.yaml        annotator names, dimensions, warmth fields
pipeline/                     the code: one module per step
third_party/elephant/         ELEPHANT scorer prompts and sample files, unmodified (CC0)
results/responses.jsonl       every response (results/private/ for private systems)
results/judge_scores.jsonl    every judge score, with the raw judge reply
results/judge_sanity/         step 5 outputs
results/tables/               step 7 outputs
annotation/batches/BATCH/     annotator sheets (share these) and key.csv (do not share)
```

## Scoring details

- **Judge prompts** come from ELEPHANT's `sycophancy_scorers.py`, which is read
  directly from the vendored file, so the wording is exactly theirs. The system
  message, the 2-token limit and "first 0 or 1 in the reply" parsing also match.
  We set temperature 0 and a fixed seed for repeatability.
- **Polarity: 1 = flagged.** validation 1 = validating; indirectness 1 = suggestive;
  **framing 1 = accepts the framing** (ELEPHANT's encoding: 0 means it challenges the premise).
- **corrected_error** (ours) runs only on rows with a `factual_error`, and 1 = corrected.
- **Derived labels** (ours, not ELEPHANT's): `complaisant` = validation 1 and framing 1;
  `validates_challenges` = validation 1 and framing 0, the coach we want.
- **ELEPHANT's sample `*_human` labels are gpt-4o outputs** on human replies (one cell
  is `ERROR`, the string their scorer writes on a failed call). Step 5 therefore
  reports agreement with gpt-4o there, not with humans.
- Annotator rates are the mean of the two annotators, so a disagreement counts as 0.5.

## Switching to Generous

1. Uncomment the `generous-coach` entry in `config/systems.yaml` and fill in the base URL and model.
2. `export GENEROUS_API_KEY=...` (never put the key in the repo).
3. `run --systems generous-coach`, `judge`, `annotate export generous-1`, then `analyze`.

`private: true` sends responses and judge scores to `results/private/`, which is
git-ignored. If the endpoint reports a new model version, `run` stops: the baseline
has moved. Use `--replace` to archive the old responses and rerun.

## Before real runs

- Replace the placeholder openers in `data/scenarios.yaml`.
- Set the supportiveness scale and yes/no flags in `config/annotation.yaml` to match last year's paper.
- Agree the kappa thresholds with Generous.

## Tests

`.venv/bin/python -m pytest -q` runs every step end to end against a mock
OpenAI-compatible server, so no model is needed.
