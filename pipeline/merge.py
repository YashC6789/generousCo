"""Merge a results folder from another machine (e.g. a GPU job) into this one.

`merge PATH` reads PATH/ (a copy of another checkout's results/) and adds what is new:

- Live JSONL files (responses, judge scores, pressure, stance, mitigations, ...): records
  not already here are appended. Every record's key starts with "<system>|", and a
  system's replies must come from one machine: if a system has a reply on both sides
  under the same key but with different text, none of that system's records are
  merged from PATH, and it is reported. (Rerun it on one side, or archive one side.)
- Archived JSONL files, judge sanity outputs, harness snapshots, API tests and job
  logs are copied if this side does not have them; nothing is overwritten.
- tables/ and report.html are skipped: rebuild them with `analyze`, `pressure analyze`,
  `mitigate analyze` and `report`.

Rerunning the same merge adds nothing.
"""

import json
import shutil
from pathlib import Path

from .project import read_jsonl

# Files whose records are model replies; the others hold scores of those replies.
GENERATED = {"responses.jsonl", "pressure.jsonl", "mitigations.jsonl"}
SKIP = {"tables", "report.html"}


def merge(project, source):
    source = Path(source).resolve()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a folder (pass a copy of another machine's results/)")
    if source == project.results_dir.resolve():
        raise SystemExit("that is this checkout's own results folder")

    totals = {"records": 0, "files": 0}
    conflicts = {}
    for sub in ("", "private"):
        src_dir = source / sub
        if not src_dir.is_dir():
            continue
        dst_dir = project.results_dir / sub
        live = sorted(p for p in src_dir.glob("*.jsonl") if ".archived-" not in p.name)
        bad = _conflicting_systems(src_dir, dst_dir)
        conflicts.update({f"{sub + '/' if sub else ''}{s}": why for s, why in bad.items()})
        for src in live:
            totals["records"] += _merge_jsonl(src, dst_dir / src.name, skip_systems=set(bad))
    totals["files"] = _copy_new_files(source, project.results_dir)

    print(f"Merged {totals['records']} new records and {totals['files']} new files from {source}.")
    for system, why in sorted(conflicts.items()):
        print(f"  NOT MERGED {system}: {why}")
    if totals["records"]:
        print("Rebuild the tables and report: analyze, pressure analyze, mitigate analyze, report.")
    return totals, conflicts


def _system(record):
    return str(record.get("key", "")).split("|", 1)[0]


def _conflicting_systems(src_dir, dst_dir):
    """Systems with a reply under the same key on both sides but different text."""
    bad = {}
    for name in sorted(GENERATED):
        here = {r["key"]: r for r in read_jsonl(dst_dir / name)}
        for r in read_jsonl(src_dir / name):
            mine = here.get(r["key"])
            if mine is not None and mine.get("response") != r.get("response"):
                bad.setdefault(_system(r), f"{name} has a different reply for {r['key']} on each side")
    return bad


def _merge_jsonl(src, dst, skip_systems):
    have = {json.dumps(r, sort_keys=True) for r in read_jsonl(dst)}
    new = []
    for r in read_jsonl(src):
        line = json.dumps(r, sort_keys=True)
        if _system(r) in skip_systems or line in have:
            continue
        have.add(line)
        new.append(r)
    if new:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def _copy_new_files(source, dest):
    copied = 0
    for src in sorted(source.rglob("*")):
        rel = src.relative_to(source)
        if not src.is_file() or rel.parts[0] in SKIP or (rel.parts[0] == "private" and len(rel.parts) > 1
                                                         and rel.parts[1] in SKIP):
            continue
        if src.suffix == ".jsonl" and ".archived-" not in src.name:
            continue  # live files are merged record by record
        dst = dest / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied
