# Vendored from ELEPHANT

- Repo: https://github.com/myracheng/elephant (CC0-1.0, see `LICENSE`)
- Commit: `61f7044fc0e11defc971dfaa9bc84d7340fa88eb`
- Paper: Cheng et al., *ELEPHANT: Measuring and understanding social sycophancy in LLMs*, arXiv 2505.13995

Files are unmodified copies:

- `sycophancy_scorers.py`: the judge prompts. `pipeline/prompts.py` extracts
  `create_prompt` from this file at runtime, so our judge uses the exact prompt text.
  The rest of the file (the gpt-4o client) is never executed.
- `sample_datasets/OEQ_sample.csv`, `sample_datasets/AITA-YTA_sample.csv`: the
  10-example samples with human responses and their `*_human` labels, used for the
  judge sanity check (step 5).
