"""Project paths, config loading and JSONL storage."""

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path):
    """Read KEY=value lines from a git-ignored .env into the environment.

    Variables already set in the shell win, so `export X=...` still overrides the file.
    Returns the names loaded (never the values).
    """
    path = Path(path)
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


@dataclass
class Project:
    """Paths for one scenario set.

    The default set (None or "v1") is data/scenarios.yaml + data/ladders.yaml with results in
    results/. Another set NAME reads data/sets/NAME/ and writes results/sets/NAME/ and
    annotation/sets/NAME/, so sets never overwrite each other's data.
    """
    root: Path = REPO_ROOT
    scenario_set: str = None

    @property
    def is_default_set(self):
        return self.scenario_set in (None, "", "v1")

    @property
    def set_dir(self):
        return self.root / "data" if self.is_default_set else self.root / "data" / "sets" / self.scenario_set

    @property
    def scenarios_file(self):
        return self.set_dir / "scenarios.yaml"

    @property
    def systems_file(self):
        return self.root / "config" / "systems.yaml"

    @property
    def annotation_config_file(self):
        return self.root / "config" / "annotation.yaml"

    @property
    def results_dir(self):
        base = self.root / "results"
        return base if self.is_default_set else base / "sets" / self.scenario_set

    def responses_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "responses.jsonl"

    def judge_file(self, private=False):
        base = self.results_dir / "private" if private else self.results_dir
        return base / "judge_scores.jsonl"

    @property
    def ladders_file(self):
        return self.set_dir / "ladders.yaml"

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
        base = self.root / "annotation"
        return base if self.is_default_set else base / "sets" / self.scenario_set

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
