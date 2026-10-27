"""Phase 3 output: a script-based harness to rerun whenever a system's prompt or model changes.

`harness --systems NAME [--replace] [--fixes [FIX ...]]` runs every step for those
systems (Phase 1 replies and judge, Phase 2 ladders and stance judge, optionally the
Phase 3 fixes, warmth), then saves a dated snapshot of the headline numbers and a
report comparing it with the previous snapshot:

    results/harness/<system>/<UTC stamp>.csv   one row per condition and metric
    results/harness/<system>/<UTC stamp>.md    the report, with the change since last time

(results/private/harness/ for private systems.) Each snapshot records what it ran on:
model version, hashes of the system prompt, scenarios, ladders and fixes, and the judge
models, so the report says what changed between two runs. Pass `--replace` after a
prompt or model change: the old replies are archived, never deleted.
"""

import hashlib
from datetime import datetime, timezone

import pandas as pd

from .project import read_jsonl
from .runner import select_systems


def _hash(text):
    return hashlib.sha256((text or "").encode()).hexdigest()[:12]


def identity(project, sys_cfg):
    from .mitigate import warmth_config
    from .stance import judge_config

    cfg = project.systems_config()
    saved = [r for r in read_jsonl(project.responses_file(sys_cfg.get("private", False)))
             if r["system"] == sys_cfg["name"]]
    system_prompt = sys_cfg.get("system_prompt", cfg.get("defaults", {}).get("system_prompt"))
    return {
        "model_version": saved[-1]["model_version"] if saved else "unknown",
        "system_prompt": _hash(system_prompt),
        "scenarios": _hash(project.scenarios_file.read_text()),
        "ladders": _hash(project.ladders_file.read_text()),
        "fixes": _hash(project.mitigations_config_file.read_text()),
        "judge": cfg["judge"]["model"],
        "stance_judge": judge_config(project)["model"],
        "warmth_judge": warmth_config(project)["model"],
    }


def snapshot(project, sys_cfg, summary):
    name = sys_cfg["name"]
    rows = summary[summary.system == name]
    if rows.empty:
        print(f"[{name}] nothing scored yet; no snapshot written")
        return None
    ident = identity(project, sys_cfg)
    out_dir = project.harness_dir(sys_cfg.get("private", False)) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = sorted(out_dir.glob("*.csv"))
    previous = pd.read_csv(previous[-1], dtype={"value": float}) if previous else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    snap = rows[["condition", "metric", "label", "better", "n", "value"]].assign(**ident, stamp=stamp)
    snap.to_csv(out_dir / f"{stamp}.csv", index=False)

    lines = [f"# Harness report: {name}, {stamp}", "",
             "Run on: " + "; ".join(f"{k} `{v}`" for k, v in ident.items()), ""]
    if previous is None:
        lines += ["First snapshot for this system: nothing to compare with yet.", ""]
        table = snap.assign(previous=None, change=None)
    else:
        prev_ident = previous.iloc[0]
        changed = [k for k in ident if str(prev_ident.get(k)) != str(ident[k])]
        lines += [f"Previous snapshot: {prev_ident['stamp']}. Changed since then: "
                  f"{', '.join(changed) if changed else 'nothing it records'}.", ""]
        table = snap.merge(previous[["condition", "metric", "value"]].rename(columns={"value": "previous"}),
                           on=["condition", "metric"], how="left")
        table["change"] = (table.value - table.previous).round(3)
    if any("placeholder" in str(s) for s in (project.scenarios_file.read_text(), project.ladders_file.read_text())):
        lines += ["**Placeholder scenarios or ladders: these numbers are not reportable.**", ""]
    lines += ["| condition | metric | better | n | value | previous | change |", "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        cells = [r.condition, r.label, r.better or "", r.n, r.value, r.previous, r.change]
        lines.append("| " + " | ".join("" if pd.isna(c) else str(c) for c in cells) + " |")
    report = out_dir / f"{stamp}.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"[{name}] snapshot and report written to {report.relative_to(project.root)}")
    return report


def harness(project, system_names, replace=False, fix_names=None, runs=None):
    """Run every step for `system_names`. `fix_names`: None skips Phase 3 fixes, [] runs all of them."""
    from .compare import analyze_mitigations
    from .judge import judge_all
    from .mitigate import judge_mitigations, run_mitigations
    from .pressure import run_pressure
    from .runner import run
    from .stance import judge_stance

    systems = select_systems(project.systems_config(), system_names)
    run(project, system_names, runs, replace)
    ok = judge_all(project)
    run_pressure(project, system_names, runs, replace)
    ok = judge_stance(project) and ok
    if fix_names is not None:
        run_mitigations(project, system_names, fix_names or None, runs, replace)
    ok = judge_mitigations(project) and ok
    summary = analyze_mitigations(project)
    for sys_cfg in systems:
        snapshot(project, sys_cfg, summary)
    return ok
