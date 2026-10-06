"""Step 3: run every scenario through every system, a few times each.

Each response is appended to results/responses.jsonl (results/private/ for
systems marked `private: true`) with the model, settings, seed and timestamp.
Rerunning skips anything already saved.
"""

import shutil
import sys
from datetime import datetime, timezone

from .adapters import PastedCSVAdapter, make_adapter
from .project import append_jsonl, read_jsonl
from .scenarios import load_scenarios


def response_key(system, scenario_id, run):
    return f"{system}|{scenario_id}|{run}"


def select_systems(cfg, names=None):
    systems = cfg["systems"]
    if names:
        unknown = set(names) - {s["name"] for s in systems}
        if unknown:
            raise SystemExit(f"unknown system(s): {sorted(unknown)}")
        systems = [s for s in systems if s["name"] in names]
    return systems


def run(project, system_names=None, runs=None, replace=False, clients=None):
    """Run scenarios through systems. `clients` maps system name to a fake client (tests)."""
    cfg = project.systems_config()
    defaults = cfg.get("defaults", {})
    n_runs = runs or defaults.get("runs", 3)
    scenarios = load_scenarios(project.scenarios_file)
    clients = clients or {}
    totals = {"new": 0, "skipped": 0, "missing": 0, "errors": 0}

    for sys_cfg in select_systems(cfg, system_names):
        name = sys_cfg["name"]
        adapter = make_adapter(sys_cfg, defaults, project.root, client=clients.get(name))
        if isinstance(adapter, PastedCSVAdapter) and not adapter.available:
            print(f"[{name}] skipped: {adapter.path.relative_to(project.root)} does not exist yet")
            continue

        out = project.responses_file(private=sys_cfg.get("private", False))
        existing = [r for r in read_jsonl(out) if r["system"] == name]

        try:
            version = adapter.model_version()
        except Exception as e:  # never block a run on version lookup
            version = f"unknown ({e})"
        old_versions = {r["model_version"] for r in existing} - {version}
        if old_versions and not isinstance(adapter, PastedCSVAdapter):
            if not replace:
                raise SystemExit(
                    f"[{name}] saved responses come from model version {sorted(old_versions)}, "
                    f"but the endpoint now reports {version!r}. The baseline has moved: rerun with "
                    f"--replace to archive the old responses and start this system again."
                )
            existing = _archive_system(out, name)

        done = {r["key"] for r in existing}
        print(f"[{name}] model version {version}")
        for sc in scenarios:
            messages = adapter.build_messages(sc["opener"])
            for r in range(n_runs):
                key = response_key(name, sc["id"], r)
                if key in done:
                    totals["skipped"] += 1
                    continue
                try:
                    result = adapter.complete(messages, r, scenario_id=sc["id"])
                except Exception as e:
                    if type(e).__name__ == "APIConnectionError":
                        raise SystemExit(
                            f"[{name}] cannot reach {sys_cfg.get('base_url')}. "
                            f"Is the server running (e.g. `ollama serve`) and the model pulled?"
                        ) from e
                    print(f"[{name}] {sc['id']} run {r}: ERROR {e}", file=sys.stderr)
                    totals["errors"] += 1
                    continue
                if result is None:
                    totals["missing"] += 1
                    continue
                append_jsonl(
                    out,
                    {
                        "key": key,
                        "system": name,
                        "scenario_id": sc["id"],
                        "run": r,
                        "messages": messages,
                        "response": result.text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **result.meta,
                    },
                )
                totals["new"] += 1
                print(f"[{name}] {sc['id']} run {r}: saved ({len(result.text)} chars)")

    print(
        f"Done: {totals['new']} new, {totals['skipped']} already saved (skipped), "
        f"{totals['missing']} missing from pasted sheets, {totals['errors']} errors."
    )
    return totals


def _archive_system(path, system):
    """Move one system's responses out of the live file into an archive file."""
    rows = read_jsonl(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(f"responses.archived-{system}-{stamp}.jsonl")
    shutil.copy(path, archive)
    keep = [r for r in rows if r["system"] != system]
    path.write_text("")
    for r in keep:
        append_jsonl(path, r)
    print(f"[{system}] archived old responses to {archive.name}")
    return []
