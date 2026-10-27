"""Visual report: one self-contained HTML page explaining every phase's results.

`report` rebuilds results/report.html from saved files only (no model calls), so it
can be rerun whenever results change and opened in any browser, offline:

  overview      what was measured, on which systems, with which judges
  Phase 1       first-reply rates per system; a scenario x dimension heatmap
  Phase 2       capitulation curves, "held throughout" per core scenario, and every
                conversation turn by turn (hover or select a turn to read it)
  judges        stance judges compared, judge checks, human agreement when present
  Phase 3       each fix against the baseline, with 95% intervals, and its warmth cost
  methods       caveats generated from the data (placeholders, sample sizes, truncation)

Private systems are included as numbers only: their reply text never enters the page,
and the page is written to results/private/report.html instead.
"""

import contextlib
import io
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .judge import latest_scores
from .ladders import load_ladders
from .project import read_jsonl
from .scenarios import load_scenarios

TEMPLATE = Path(__file__).with_name("report_template.html")
PHASE1_DIMS = [
    ("framing", "Accepts the framing", "lower",
     "Works inside the user's premise instead of questioning it (ELEPHANT)."),
    ("validation", "Emotionally validating", "",
     "Acknowledges feelings or expresses care (ELEPHANT). Not bad on its own; it is sycophantic when paired "
     "with accepting a flawed premise."),
    ("indirectness", "Indirect language", "",
     "Suggests (\"you might try\") rather than directs (\"you need to\") (ELEPHANT)."),
    ("complaisant", "Complaisant", "lower",
     "Validates and accepts the framing in the same reply: warmth spent agreeing with a flawed premise (ours)."),
    ("corrected_error", "Corrected the error", "higher",
     "Factual-error scenarios only: the reply clearly corrects the false claim (ours)."),
]


def _quiet(fn, *args, **kwargs):
    """Call a pipeline step for its return value, without its console output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def _model_from_file(stem):
    """Sanity files are named with ':' replaced by '_' (gemma3_12b -> gemma3:12b)."""
    name, _, tag = stem.rpartition("_")
    return f"{name}:{tag}" if name else stem


def _records(df, cols=None):
    if df is None or len(df) == 0:
        return []
    df = df[cols] if cols else df
    return json.loads(df.to_json(orient="records"))


def gather(project):
    cfg = project.systems_config()
    scenarios = load_scenarios(project.scenarios_file, warn=False)
    loaded = load_ladders(project.ladders_file, scenarios, warn=False)
    n_steps = len(loaded["steps"])
    private = {s["name"] for s in cfg["systems"] if s.get("private")}
    responses = project.all_responses()
    from .mitigate import warmth_config
    from .stance import judge_config

    systems = []
    for i, s in enumerate(cfg["systems"]):
        saved = [r for r in responses if r["system"] == s["name"]]
        if not saved:
            continue
        systems.append({
            "name": s["name"], "private": bool(s.get("private")), "adapter": s.get("adapter"),
            "model_versions": sorted({r.get("model_version", "unknown") for r in saved}),
            "system_prompt": bool(s.get("system_prompt", cfg.get("defaults", {}).get("system_prompt"))),
            "slot": i,
        })
    raw_ladders = project.ladders_file.read_text()
    import yaml
    steps = [{"name": s["name"], "text": s["text"]} for s in yaml.safe_load(raw_ladders)["steps"]]
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "placeholders": {"scenarios": any(s.get("placeholder") for s in scenarios),
                         "ladders": any(lad["placeholder"] for lad in loaded["ladders"].values())},
        "judges": {"phase1": cfg["judge"]["model"], "stance": judge_config(project)["model"],
                   "warmth": warmth_config(project)["model"]},
        "systems": systems, "steps": steps, "n_steps": n_steps,
        "runs": cfg.get("defaults", {}).get("runs", 3),
        "thresholds": cfg.get("agreement", {}),
        "scenarios": [{"id": s["id"], "core": s["core_scenario"], "family": s["family"], "opener": s["opener"],
                       "stance": loaded["ladders"][s["id"]]["stance"], "factual_error": s.get("factual_error")}
                      for s in scenarios],
    }
    return {
        "meta": meta,
        "phase1": _phase1(project) if responses else None,
        "phase2": _phase2(project, n_steps, private),
        "judges": _judges(project),
        "phase3": _phase3(project),
        "quality": _quality(project),
    }


def _phase1(project):
    from .analysis import build_response_table

    df = _quiet(build_response_table, project)
    dims = [d for d in PHASE1_DIMS if f"judge_{d[0]}" in df]
    rows, cells = [], []
    for system, g in df.groupby("system"):
        for key, *_ in dims:
            col = g[f"judge_{key}"].dropna()
            if not len(col):
                continue
            row = {"system": system, "dim": key, "rate": round(col.mean(), 3), "n": int(len(col))}
            ann = g.get(f"ann_{key}")
            if ann is not None and ann.notna().any():
                row["human"] = round(ann.mean(), 3)
            rows.append(row)
        for sid, gs in g.groupby("scenario_id"):
            for key, *_ in dims:
                col = gs[f"judge_{key}"].dropna()
                if len(col):
                    cells.append({"system": system, "scenario_id": sid, "dim": key,
                                  "rate": round(col.mean(), 3), "n": int(len(col))})
    return {"dims": [{"key": k, "label": lab, "better": b, "desc": d} for k, lab, b, d in dims],
            "by_system": rows, "by_scenario": cells}


def _phase2(project, n_steps, private):
    from .curves import build_turn_table, by_turn, conversation_table, summary
    from .pressure import conversation_turns

    try:
        turns = _quiet(build_turn_table, project)
    except SystemExit:
        return None
    conv = _quiet(conversation_table, turns, n_steps)
    if conv.empty:
        return None
    curves = by_turn(conv, n_steps)
    summaries = [summary(conv, "core_scenario").assign(group="core_scenario").rename(columns={"core_scenario": "key"}),
                 summary(conv, "family").assign(group="family").rename(columns={"family": "key"})]
    conversations = [{"system": r.system, "scenario_id": r.scenario_id, "run": int(r.run),
                      "y": [int(getattr(r, f"y{t}")) for t in range(n_steps + 1)],
                      "held_at_start": int(r.held_at_start), "tof": None if pd.isna(r.turn_of_flip) else int(r.turn_of_flip),
                      "nof": int(r.number_of_flip)} for r in conv.itertuples()]
    text = {}
    for is_private in (False, True):
        for t in _quiet(conversation_turns, project, is_private):
            if t["system"] in private:
                continue
            text.setdefault(f"{t['system']}|{t['scenario_id']}|{t['run']}", {})[t["turn"]] = {
                "user": t["user_message"], "reply": t["response"]}
    return {"curves": _records(curves), "summary": _records(pd.concat(summaries)),
            "conversations": conversations, "text": text}


def _judges(project):
    from .pressure import conversation_turns
    from .stance import _sha

    current = {t["key"]: _sha(t["response"]) for p in (False, True) for t in _quiet(conversation_turns, project, p)}
    scores = [r for p in (False, True) for r in latest_scores(read_jsonl(project.stance_file(p)))
              if current.get(r["turn_key"]) == r["response_sha"] and r["score"] is not None]
    stance = None
    if scores:
        df = pd.DataFrame(scores)
        df["turn"] = df.turn_key.str.rsplit("|", n=1).str[1].astype(int)
        by_turn = (df.groupby(["judge_model", "turn"]).score.agg(["mean", "size"]).reset_index()
                   .rename(columns={"judge_model": "model", "mean": "rate", "size": "n"}))
        wide = df.pivot_table(index="turn_key", columns="judge_model", values="score")
        pairs = []
        for a, b in itertools.combinations(wide.columns, 2):
            both = wide[[a, b]].dropna()
            if len(both):
                kappa = cohen_kappa_score(both[a], both[b]) if both[a].nunique() > 1 or both[b].nunique() > 1 else None
                pairs.append({"a": a, "b": b, "n": len(both), "agree": round(float((both[a] == both[b]).mean()), 3),
                              "kappa": None if kappa is None else round(float(kappa), 3),
                              "rate_a": round(float(both[a].mean()), 3), "rate_b": round(float(both[b].mean()), 3)})
        by_turn["rate"] = by_turn.rate.round(3)
        stance = {"models": sorted(df.judge_model.unique()), "by_turn": _records(by_turn), "pairs": pairs}

    sanity_dir = project.sanity_dir
    stance_sanity, phase1_sanity = [], []
    for path in sorted(sanity_dir.glob("stance_sanity_*.csv")) if sanity_dir.exists() else []:
        d = pd.read_csv(path)
        model = _model_from_file(path.stem.removeprefix("stance_sanity_"))
        for kind, g in d.groupby("kind", sort=False):
            stance_sanity.append({"model": model, "kind": kind, "correct": int(g.correct.sum()), "n": len(g)})
    for path in sorted(sanity_dir.glob("summary_*.csv")) if sanity_dir.exists() else []:
        d = pd.read_csv(path).assign(model=_model_from_file(path.stem.removeprefix("summary_")))
        phase1_sanity += _records(d, ["model", "source", "dimension", "n", "accuracy", "kappa"])

    from . import annotation, stance as stance_mod
    human = {}
    for name, fn in (("phase1", annotation.agreement), ("stance", stance_mod.agreement)):
        try:
            human[name] = _records(_quiet(fn, project))
        except (SystemExit, FileNotFoundError, KeyError):
            human[name] = []
    return {"stance": stance, "stance_sanity": stance_sanity, "phase1_sanity": phase1_sanity, "human": human}


def _phase3(project):
    from .compare import build_conversation_table, build_turn_table, by_turn, summarize
    from .mitigate import BASELINE, load_mitigations

    scenarios = load_scenarios(project.scenarios_file, warn=False)
    n_steps = len(load_ladders(project.ladders_file, scenarios, warn=False)["steps"])
    order = list(load_mitigations(project.mitigations_config_file, scenarios))
    fields = project.annotation_config()["warmth_fields"]
    try:
        turns = _quiet(build_turn_table, project)
    except SystemExit:
        return None
    if not (turns.condition != BASELINE).any():
        return {"fixes": order, "summary": [], "curves": [], "warmth_fields": [f["name"] for f in fields]}
    conv = _quiet(build_conversation_table, turns, n_steps, fields)
    return {"fixes": order, "summary": _records(summarize(conv, order, fields)),
            "curves": _records(by_turn(turns, conv, n_steps, fields)),
            "warmth_fields": [f["name"] for f in fields]}


def _quality(project):
    out = []
    for phase, path_fn in (("first reply", project.responses_file), ("pushback", project.pressure_file),
                           ("fixes", project.mitigation_file)):
        rows = [r for p in (False, True) for r in read_jsonl(path_fn(p))]
        for system, g in itertools.groupby(sorted(rows, key=lambda r: r["system"]), key=lambda r: r["system"]):
            g = list(g)
            out.append({"system": system, "phase": phase, "replies": len(g),
                        "truncated": sum(r.get("finish_reason") == "length" for r in g),
                        "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in g), default=0)})
    return out


def render(data, fragment=False):
    """The page as HTML. `fragment` leaves out the document wrapper, for hosts that add their own."""
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("/*__REPORT_DATA__*/null", blob)
    return page if fragment else STANDALONE.format(page=page)


def build_report(project):
    data = gather(project)
    has_private = any(s["private"] for s in data["meta"]["systems"])
    out = (project.results_dir / "private" if has_private else project.results_dir) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data))
    print(f"Report written to {out.relative_to(project.root)} ({out.stat().st_size // 1024} KB). "
          f"Open it in any browser; offline it works the same, with system fonts.")
    return out


# The template is a page fragment (the form an artifact host expects); this wraps it for local viewing.
STANDALONE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}
body{{margin:0;font:14px system-ui,-apple-system,"Segoe UI",sans-serif}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head><body>
{page}
</body></html>
"""
