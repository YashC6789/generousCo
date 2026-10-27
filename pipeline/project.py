"""Project paths, config loading and JSONL storage."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
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
    def ladders_file(self):
        return self.root / "data" / "ladders.yaml"

    def pressure_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "pressure.jsonl"

    def stance_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "stance_scores.jsonl"

    @property
    def mitigations_config_file(self):
        return self.root / "config" / "mitigations.yaml"

    def mitigation_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "mitigations.jsonl"

    def mitigation_scores_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "mitigation_scores.jsonl"

    def harness_dir(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "harness"

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


def archive_system(paths, system):
    """Move one system's records out of each live JSONL file into a timestamped archive file.

    Every record's key starts with "<system>|", so that is what is matched.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path in map(Path, paths):
        if not path.exists():
            continue
        rows = read_jsonl(path)
        archive = path.with_name(f"{path.stem}.archived-{system}-{stamp}.jsonl")
        shutil.copy(path, archive)
        path.write_text("")
        for r in rows:
            if r["key"].split("|")[0] != system:
                append_jsonl(path, r)
        print(f"[{system}] archived old {path.stem.replace('_', ' ')} to {archive.name}")


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
