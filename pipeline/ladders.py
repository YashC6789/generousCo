"""Phase 2, step 1: load and check the pushback ladders (data/ladders.yaml)."""

import string
import sys

import yaml

from .scenarios import load_scenarios

MAX_STEPS = 5
SLOTS = {"view", "ask"}


class LadderError(ValueError):
    pass


def load_ladders(path, scenarios, warn=True):
    """Returns {"steps": [step names], "ladders": {scenario id: {"stance", "messages", "placeholder"}}}."""
    data = yaml.safe_load(path.read_text()) or {}
    steps = data.get("steps") or []
    if not 1 <= len(steps) <= MAX_STEPS:
        raise LadderError(f"{path.name}: expected 1 to {MAX_STEPS} steps, found {len(steps)}")
    for s in steps:
        used = {f for _, f, _, _ in string.Formatter().parse(s["text"]) if f}
        if used - SLOTS:
            raise LadderError(f"{path.name}: step {s['name']!r} uses unknown slot(s) {sorted(used - SLOTS)}")

    entries = {str(k): v for k, v in (data.get("scenarios") or {}).items()}
    ids = [sc["id"] for sc in scenarios]
    missing, extra = [i for i in ids if i not in entries], sorted(set(entries) - set(ids))
    if missing or extra:
        raise LadderError(f"{path.name}: scenarios must match scenarios.yaml (missing {missing}, unknown {extra})")

    ladders = {}
    for i in ids:
        e = entries[i]
        if "messages" in e:
            messages = [str(m).strip() for m in e["messages"]]
            if len(messages) != len(steps):
                raise LadderError(f"{path.name}: scenario {i} gives {len(messages)} messages for {len(steps)} steps")
            needed = ["stance"]
        else:
            needed = ["stance", *sorted(SLOTS)]
        empty = [k for k in needed if not str(e.get(k) or "").strip()]
        if empty:
            raise LadderError(f"{path.name}: scenario {i} is missing {empty}")
        if "messages" not in e:
            messages = [s["text"].format(view=e["view"].strip(), ask=e["ask"].strip()) for s in steps]
        ladders[i] = {"stance": e["stance"].strip(), "messages": messages, "placeholder": bool(e.get("placeholder"))}

    placeholders = [i for i, ladder in ladders.items() if ladder["placeholder"]]
    if warn and placeholders:
        print(f"WARNING: {len(placeholders)} placeholder ladder(s) in {path.name}: {', '.join(placeholders)}. "
              f"Replace them before real runs.", file=sys.stderr)
    return {"steps": [s["name"] for s in steps], "ladders": ladders}


def check(project, show=None):
    """`show`: None prints only the summary, [] every ladder, a list of ids just those."""
    scenarios = load_scenarios(project.scenarios_file)
    loaded = load_ladders(project.ladders_file, scenarios)
    print(f"OK: {len(loaded['ladders'])} ladders of {len(loaded['steps'])} steps: {', '.join(loaded['steps'])}")
    for sc in scenarios:
        if show is None or (show and sc["id"] not in show):
            continue
        ladder = loaded["ladders"][sc["id"]]
        print(f"\n{sc['id']} stance: {ladder['stance']}\n  0. {sc['opener']}")
        for t, msg in enumerate(ladder["messages"], 1):
            print(f"  {t}. {msg}")
    return loaded
