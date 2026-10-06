"""Project paths, config loading and JSONL storage."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Project:
    root: Path = REPO_ROOT

    @property
    def scenarios_file(self):
        return self.root / "data" / "scenarios.yaml"

    @property
    def systems_file(self):
        return self.root / "config" / "systems.yaml"

    @property
    def annotation_config_file(self):
        return self.root / "config" / "annotation.yaml"

    @property
    def results_dir(self):
        return self.root / "results"

    def responses_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "responses.jsonl"

    def judge_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "judge_scores.jsonl"

    @property
    def sanity_dir(self):
        return self.results_dir / "judge_sanity"

    @property
    def tables_dir(self):
        return self.results_dir / "tables"

    @property
    def annotation_dir(self):
        return self.root / "annotation"

    @property
    def elephant_dir(self):
        return self.root / "third_party" / "elephant"

    def systems_config(self):
        return yaml.safe_load(self.systems_file.read_text())

    def annotation_config(self):
        return yaml.safe_load(self.annotation_config_file.read_text())

    def all_responses(self):
        return read_jsonl(self.responses_file()) + read_jsonl(self.responses_file(private=True))

    def all_judge_scores(self):
        return read_jsonl(self.judge_file()) + read_jsonl(self.judge_file(private=True))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
