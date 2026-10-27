"""Phase 3, steps 1-2: rerun the scenarios and ladders with each low-cost fix, then score them.

Fixes are defined in config/mitigations.yaml, cheapest first, and apply only to
systems we can configure (adapter `openai`). Under a fix every conversation is run
from scratch: turn 0 is the fix's own reply to the opener, then the same Phase 2
ladder, with the same seeds as the baseline. Turns go to results/mitigations.jsonl
(results/private/ for private systems), keyed system|fix|scenario|run|turn. The
baseline is the saved Phase 1 + Phase 2 conversation, so it is not rerun.

A saved turn is kept only while the fix's definition, the ladder, the system prompt
and the model version are unchanged; otherwise `--replace` archives the system's
Phase 3 turns.

Scoring (`mitigate judge`) writes results/mitigation_scores.jsonl, each score with a
hash of the reply it judged:
  stance_abandoned  every turn under a fix (baseline stance comes from Phase 2)
  ELEPHANT + corrected_error   turn 0 under a fix, Phase 1 judge (baseline: Phase 1)
  warmth            every turn, baseline included, on config/annotation.yaml's fields
"""

import json
import re
import sys
from datetime import datetime, timezone

import yaml

from .adapters import make_adapter
from .judge import Judge, dimensions_for, latest_scores
from .ladders import load_ladders
from .pressure import check_context, conversation_turns, priority_pairs
from .project import append_jsonl, archive_system, read_jsonl
from .prompts import STANCE_SYSTEM_MESSAGE, WARMTH_SYSTEM_MESSAGE, stance_prompt, warmth_prompt
from .runner import select_systems
from .scenarios import load_scenarios
from .stance import DIMENSION as STANCE, _sha, judge_config as stance_judge_config

BASELINE = "baseline"
PARTS = {"system_append", "goal_reminder", "critique", "final_marker"}
WARMTH = "warmth"


def load_mitigations(path, scenarios):
    """{name: fix} in cost order, checked."""
    items = (yaml.safe_load(path.read_text()) or {}).get("mitigations") or []
    fixes = {}
    for m in items:
        name = m.get("name")
        if not name or name == BASELINE or name in fixes or "|" in name:
            raise SystemExit(f"{path.name}: fix names must be unique, non-empty, without '|', and not {BASELINE!r}")
        unknown = set(m) - PARTS - {"name"}
        if unknown or not set(m) & (PARTS - {"final_marker"}):
            raise SystemExit(f"{path.name}: fix {name!r} needs at least one of system_append, goal_reminder, "
                             f"critique (unknown keys: {sorted(unknown)})")
        if "goal_reminder" in m:
            missing = [sc["id"] for sc in scenarios if not sc.get("goal")]
            if missing:
                raise SystemExit(f"{path.name}: fix {name!r} needs a `goal` for scenarios {missing}")
        if "critique" in m and ("{draft}" not in m["critique"] or not m.get("final_marker")):
            raise SystemExit(f"{path.name}: fix {name!r}: critique needs a {{draft}} slot and a final_marker")
        fixes[name] = m
    return fixes


def mitigation_key(system, fix, scenario_id, run, turn):
    return f"{system}|{fix}|{scenario_id}|{run}|{turn}"


def system_prompt_for(adapter, fix):
    parts = [adapter.system_prompt, fix.get("system_append")]
    return "\n\n".join(p.strip() for p in parts if p) or None


def turn_messages(adapter, fix, scenario, ladder, replies, turn):
    """Messages sent at `turn` (0 = opener): system prompt + fix, history, goal reminder, user message."""
    user = [scenario["opener"], *ladder]
    messages = []
    system = system_prompt_for(adapter, fix)
    if system:
        messages.append({"role": "system", "content": system})
    for t in range(turn):
        messages += [{"role": "user", "content": user[t]}, {"role": "assistant", "content": replies[t]}]
    if fix.get("goal_reminder"):
        messages.append({"role": "system", "content": fix["goal_reminder"].strip().format(goal=scenario["goal"])})
    messages.append({"role": "user", "content": user[turn]})
    return messages


def critique(adapter, fix, messages, draft, run, scenario_id):
    """Second pass: returns (reply to send, critique completion, whether the marker was found)."""
    transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system")
    prompt = fix["critique"].format(system_prompt=system_prompt_for(adapter, fix) or "(none)",
                                    transcript=transcript, draft=draft)
    result = adapter.complete([{"role": "user", "content": prompt}], run, scenario_id=scenario_id)
    marker = fix["final_marker"]
    if marker in result.text and result.text.rsplit(marker, 1)[1].strip():
        return result.text.rsplit(marker, 1)[1].strip(), result, True
    return draft, result, False


def run_mitigations(project, system_names=None, fix_names=None, runs=None, replace=False, clients=None,
                    scenario_ids=None, top=None):
    cfg = project.systems_config()
    defaults = cfg.get("defaults", {})
    n_runs = runs or defaults.get("runs", 3)
    scenarios = load_scenarios(project.scenarios_file)
    by_id = {sc["id"]: sc for sc in scenarios}
    loaded = load_ladders(project.ladders_file, scenarios)
    steps, ladders = loaded["steps"], loaded["ladders"]
    fixes = load_mitigations(project.mitigations_config_file, scenarios)
    chosen = fix_names or list(fixes)
    unknown = set(chosen) - set(fixes)
    if unknown:
        raise SystemExit(f"unknown fix(es): {sorted(unknown)}; config/mitigations.yaml has {list(fixes)}")
    chosen = [f for f in fixes if f in chosen]  # always cheapest first
    if scenario_ids:
        unknown = set(scenario_ids) - set(by_id)
        if unknown:
            raise SystemExit(f"unknown scenario(s): {sorted(unknown)}")

    systems = []
    for sys_cfg in select_systems(cfg, system_names):
        if sys_cfg.get("adapter") == "openai":
            systems.append(sys_cfg)
        else:
            print(f"[{sys_cfg['name']}] skipped: fixes apply only to systems we can configure")
    pairs = priority_pairs(project, [s["name"] for s in systems], top) if top else None
    clients = clients or {}
    totals = {"new": 0, "skipped": 0, "errors": 0, "critique_unparsed": 0}

    for sys_cfg in systems:
        name = sys_cfg["name"]
        wanted = [sc for sc in scenarios if (not scenario_ids or sc["id"] in scenario_ids)
                  and (pairs is None or (name, sc["id"]) in pairs)]
        if not wanted:
            continue
        adapter = make_adapter(sys_cfg, defaults, project.root, client=clients.get(name))
        out = project.mitigation_file(sys_cfg.get("private", False))
        saved = {r["key"]: r for r in read_jsonl(out) if r["system"] == name}
        try:
            version = adapter.model_version()
        except Exception as e:  # never block a run on version lookup
            version = f"unknown ({e})"
        stale = _stale_turns(saved, adapter, fixes, by_id, ladders, version)
        if stale:
            if not replace:
                raise SystemExit(
                    f"[{name}] {len(stale)} saved Phase 3 turn(s) no longer match the fix definitions, ladders, "
                    f"system prompt or model version (e.g. {stale[0]}). Rerun with --replace to archive "
                    f"this system's Phase 3 turns and start again."
                )
            archive_system([out], name)
            saved = {}

        print(f"[{name}] model version {version}; fixes in order: {', '.join(chosen)}")
        for fix_name in chosen:
            fix = fixes[fix_name]
            for sc in wanted:
                ladder = ladders[sc["id"]]["messages"]
                for r in range(n_runs):
                    replies = {}
                    for t in range(len(ladder) + 1):
                        key = mitigation_key(name, fix_name, sc["id"], r, t)
                        if key in saved:
                            replies[t] = saved[key]["response"]
                            totals["skipped"] += 1
                            continue
                        messages = turn_messages(adapter, fix, sc, ladder, replies, t)
                        try:
                            result = adapter.complete(messages, r, scenario_id=sc["id"])
                            check_context(adapter, result, key)
                            record = {}
                            if fix.get("critique"):
                                final, cres, parsed = critique(adapter, fix, messages, result.text, r, sc["id"])
                                check_context(adapter, cres, key)
                                record = {"draft": result.text, "critique": cres.text, "critique_parsed": parsed}
                                totals["critique_unparsed"] += not parsed
                            else:
                                final = result.text
                        except Exception as e:
                            if type(e).__name__ == "APIConnectionError":
                                raise SystemExit(
                                    f"[{name}] cannot reach {sys_cfg.get('base_url')}. "
                                    f"Is the server running (e.g. `ollama serve`) and the model pulled?"
                                ) from e
                            print(f"[{name}] {key}: ERROR {e}; the rest of this conversation is left for a rerun",
                                  file=sys.stderr)
                            totals["errors"] += 1
                            break
                        append_jsonl(out, {
                            "key": key, "system": name, "mitigation": fix_name, "fix": fix,
                            "scenario_id": sc["id"], "run": r, "turn": t,
                            "step": "opener" if t == 0 else steps[t - 1], "messages": messages,
                            "response": final, **record, "timestamp": datetime.now(timezone.utc).isoformat(),
                            **result.meta,
                        })
                        replies[t] = final
                        totals["new"] += 1
                        print(f"[{name}] {key}: saved ({len(final)} chars, "
                              f"{result.meta.get('prompt_tokens')} prompt tokens)", flush=True)

    print(f"Done: {totals['new']} new turns, {totals['skipped']} already saved (skipped), {totals['errors']} errors, "
          f"{totals['critique_unparsed']} self-critique replies without the final marker (draft sent instead).")
    return totals


def _stale_turns(saved, adapter, fixes, scenarios, ladders, version):
    convs = {}
    for rec in saved.values():
        convs.setdefault(rec["key"].rsplit("|", 1)[0], {})[rec["turn"]] = rec
    stale = []
    for turns in convs.values():
        first = turns[min(turns)]
        fix, sc = fixes.get(first["mitigation"]), scenarios.get(first["scenario_id"])
        ladder = (ladders.get(first["scenario_id"]) or {}).get("messages")
        replies = {t: rec["response"] for t, rec in turns.items()}
        for t, rec in sorted(turns.items()):
            fresh = (
                fix is not None and sc is not None and ladder is not None and t <= len(ladder)
                and rec["fix"] == fix and rec["model_version"] == version
                and all(u in replies for u in range(t))
                and rec["messages"] == turn_messages(adapter, fix, sc, ladder, replies, t)
            )
            if not fresh:
                stale.append(rec["key"])
    return sorted(stale)


def condition_turns(project, private=False):
    """Every scored unit of Phase 3: baseline turns (Phase 1 + 2) and turns under each current fix.

    Turns saved under an edited or removed fix are left out until rerun.
    """
    scenarios = load_scenarios(project.scenarios_file, warn=False)
    fixes = load_mitigations(project.mitigations_config_file, scenarios)
    rows = [{**t, "condition": BASELINE} for t in conversation_turns(project, private)]
    stale = 0
    for rec in read_jsonl(project.mitigation_file(private)):
        if fixes.get(rec["mitigation"]) != rec["fix"]:
            stale += 1
            continue
        rows.append({"key": rec["key"], "condition": rec["mitigation"], "system": rec["system"],
                     "scenario_id": rec["scenario_id"], "run": rec["run"], "turn": rec["turn"],
                     "user_message": rec["messages"][-1]["content"], "response": rec["response"]})
    if stale:
        print(f"WARNING: {stale} Phase 3 turn(s) were made under a fix that has since been edited or removed "
              f"and are left out. Rerun `mitigate run --replace`.", file=sys.stderr)
    return rows


# Scoring ---------------------------------------------------------------------

def warmth_config(project):
    cfg = project.systems_config()
    return cfg.get("warmth_judge") or stance_judge_config(project)


def parse_warmth(raw, fields):
    m = re.search(r"\{.*?\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0))
        scores = {f["name"]: int(d[f["name"]]) for f in fields}
    except Exception:
        return None
    return scores if all(scores[f["name"]] in f["values"] for f in fields) else None


def judge_mitigations(project, client=None):
    """Score every Phase 3 unit that lacks a score for its current reply. True if all are scored."""
    cfg = project.systems_config()
    judges = {"stance": Judge(stance_judge_config(project), client=client),
              "elephant": Judge(cfg["judge"], client=client),
              WARMTH: Judge(warmth_config(project), client=client)}
    fields = project.annotation_config()["warmth_fields"]
    scenarios = {s["id"]: s for s in load_scenarios(project.scenarios_file, warn=False)}
    ladders = load_ladders(project.ladders_file, list(scenarios.values()), warn=False)["ladders"]
    new = failed = skipped = 0
    for private in (False, True):
        out = project.mitigation_scores_file(private)
        done = {r["key"]: r["response_sha"] for r in latest_scores(read_jsonl(out)) if r["score"] is not None}
        for turn in condition_turns(project, private):
            sc = scenarios[turn["scenario_id"]]
            sha = _sha(turn["response"])
            for dim, judge in jobs_for(turn, sc, judges):
                key = f"{turn['key']}|{dim}|{judge.model}"
                if done.get(key) == sha:
                    skipped += 1
                    continue
                if dim == STANCE:
                    score, raw = judge.ask(STANCE_SYSTEM_MESSAGE, stance_prompt(
                        sc["opener"], ladders[sc["id"]]["stance"], turn["user_message"], turn["response"]))
                elif dim == WARMTH:
                    raw = judge.ask(WARMTH_SYSTEM_MESSAGE, warmth_prompt(fields, turn["user_message"],
                                                                         turn["response"]))[1]
                    score = parse_warmth(raw, fields)
                else:
                    score, raw = judge.score(dim, sc["opener"], turn["response"], sc.get("factual_error"))
                append_jsonl(out, {
                    "key": key, "turn_key": turn["key"], "condition": turn["condition"], "dimension": dim,
                    "score": score, "raw": raw, "response_sha": sha, "judge_model": judge.model,
                    "judge_base_url": judge.base_url, **{f"judge_{k}": v for k, v in judge.settings.items()},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                new += 1
                if score is None:
                    failed += 1
                    print(f"PARSE FAIL {key}: {raw!r}", file=sys.stderr)
    print(f"Scored {new} new ({skipped} already scored). Parse failures this run: {failed}.")
    missing = len(missing_scores(project))
    print("CHECK OK: every Phase 3 unit is scored." if not missing else
          f"CHECK FAILED: {missing} missing or unparsed scores. Rerun `mitigate judge`.")
    return not missing


def jobs_for(turn, scenario, judges):
    jobs = []
    if turn["condition"] != BASELINE:
        jobs.append((STANCE, judges["stance"]))
        if turn["turn"] == 0:
            jobs += [(dim, judges["elephant"]) for dim in dimensions_for(scenario)]
    jobs.append((WARMTH, judges[WARMTH]))
    return jobs


def mitigation_scores(project):
    """{(turn key, dimension): score} for current replies, from the configured judges."""
    cfg = project.systems_config()
    models = {STANCE: stance_judge_config(project)["model"], WARMTH: warmth_config(project)["model"]}
    out = {}
    for private in (False, True):
        latest = latest_scores(read_jsonl(project.mitigation_scores_file(private)))
        current = {t["key"]: _sha(t["response"]) for t in condition_turns(project, private)}
        for r in latest:
            model = models.get(r["dimension"], cfg["judge"]["model"])
            if r["judge_model"] == model and current.get(r["turn_key"]) == r["response_sha"]:
                out[(r["turn_key"], r["dimension"])] = r["score"]
    return out


def missing_scores(project):
    scenarios = {s["id"]: s for s in load_scenarios(project.scenarios_file, warn=False)}
    scores = mitigation_scores(project)
    dummy = {"stance": None, "elephant": None, WARMTH: None}
    return [(t["key"], dim) for private in (False, True) for t in condition_turns(project, private)
            for dim, _ in jobs_for(t, scenarios[t["scenario_id"]], dummy)
            if scores.get((t["key"], dim)) is None]
