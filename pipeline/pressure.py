"""Phase 2, step 2: run the pushback ladders as multi-turn conversations.

Each conversation continues a saved Phase 1 response: turn 0 is the Phase 1 reply
to the opener (same system, scenario and run), and turns 1..K send the scripted
ladder one message at a time with the whole conversation so far. Every turn is
appended to results/pressure.jsonl (results/private/ for private systems), keyed
system|scenario|run|turn. Rerunning skips saved turns and resumes a conversation
where it stopped.

A saved turn is kept only while the messages it was sent still match what would be
sent now (same Phase 1 reply, same ladder text, same model version). Otherwise the
conversation is stale, and `--replace` archives the system's Phase 2 data.
"""

import sys
from datetime import datetime, timezone

from .adapters import make_adapter
from .ladders import load_ladders
from .project import append_jsonl, archive_system, read_jsonl
from .runner import response_key, select_systems
from .scenarios import load_scenarios


def turn_key(system, scenario_id, run, turn):
    return f"{system}|{scenario_id}|{run}|{turn}"


def conversation_messages(turn0, replies, ladder, turn):
    """Messages sent at `turn` (>= 1): the Phase 1 exchange, then each earlier pushback and its reply."""
    messages = [*turn0["messages"], {"role": "assistant", "content": turn0["response"]}]
    for t in range(1, turn):
        messages += [{"role": "user", "content": ladder[t - 1]}, {"role": "assistant", "content": replies[t]}]
    messages.append({"role": "user", "content": ladder[turn - 1]})
    return messages


def run_pressure(project, system_names=None, runs=None, replace=False, clients=None, scenario_ids=None, top=None):
    """Run the ladders. `top` keeps only the N (system, scenario) pairs Phase 1 found most agreeable."""
    cfg = project.systems_config()
    defaults = cfg.get("defaults", {})
    n_runs = runs or defaults.get("runs", 3)
    scenarios = load_scenarios(project.scenarios_file)
    loaded = load_ladders(project.ladders_file, scenarios)
    steps, ladders = loaded["steps"], loaded["ladders"]
    openers = {sc["id"]: sc["opener"] for sc in scenarios}
    if scenario_ids:
        unknown = set(scenario_ids) - {sc["id"] for sc in scenarios}
        if unknown:
            raise SystemExit(f"unknown scenario(s): {sorted(unknown)}")
        scenarios = [sc for sc in scenarios if sc["id"] in scenario_ids]

    systems = []
    for sys_cfg in select_systems(cfg, system_names):
        if sys_cfg.get("adapter") == "openai":
            systems.append(sys_cfg)
        else:
            print(f"[{sys_cfg['name']}] skipped: the ladders need a live chat endpoint, and a pasted "
                  f"system cannot be scripted turn by turn")
    pairs = priority_pairs(project, [s["name"] for s in systems], top) if top else None
    clients = clients or {}
    totals = {"new": 0, "skipped": 0, "no_phase1": 0, "errors": 0}

    for sys_cfg in systems:
        name = sys_cfg["name"]
        wanted = [sc for sc in scenarios if pairs is None or (name, sc["id"]) in pairs]
        if not wanted:
            continue
        adapter = make_adapter(sys_cfg, defaults, project.root, client=clients.get(name))
        private = sys_cfg.get("private", False)
        out = project.pressure_file(private)
        phase1 = {r["key"]: r for r in read_jsonl(project.responses_file(private)) if r["system"] == name}
        saved = {r["key"]: r for r in read_jsonl(out) if r["system"] == name}
        try:
            version = adapter.model_version()
        except Exception as e:  # never block a run on version lookup
            version = f"unknown ({e})"

        outdated = sorted(k for k, r in phase1.items() if r["scenario_id"] in openers and (
            r["model_version"] != version or r["messages"] != adapter.build_messages(openers[r["scenario_id"]])))
        if outdated:
            raise SystemExit(
                f"[{name}] {len(outdated)} Phase 1 response(s) were made with a different model version or "
                f"system prompt than the current one (e.g. {outdated[0]}). Rerun Phase 1 first: "
                f"`run --systems {name} --replace`."
            )
        stale = _stale_turns(saved, phase1, ladders, version)
        if stale:
            if not replace:
                raise SystemExit(
                    f"[{name}] {len(stale)} saved Phase 2 turn(s) no longer match the Phase 1 responses, the "
                    f"ladder text or the model version (e.g. {stale[0]}). Rerun with --replace to archive "
                    f"this system's Phase 2 data and start again."
                )
            archive_system([out, project.stance_file(private)], name)
            saved = {}

        print(f"[{name}] model version {version}")
        for sc in wanted:
            ladder = ladders[sc["id"]]["messages"]
            for r in range(n_runs):
                turn0 = phase1.get(response_key(name, sc["id"], r))
                if turn0 is None:
                    totals["no_phase1"] += 1
                    continue
                replies = {}
                for t in range(1, len(ladder) + 1):
                    key = turn_key(name, sc["id"], r, t)
                    if key in saved:
                        replies[t] = saved[key]["response"]
                        totals["skipped"] += 1
                        continue
                    messages = conversation_messages(turn0, replies, ladder, t)
                    try:
                        result = adapter.complete(messages, r, scenario_id=sc["id"])
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
                    check_context(adapter, result, key)
                    append_jsonl(out, {
                        "key": key, "system": name, "scenario_id": sc["id"], "run": r, "turn": t,
                        "step": steps[t - 1], "turn0_key": turn0["key"], "messages": messages,
                        "response": result.text, "timestamp": datetime.now(timezone.utc).isoformat(),
                        **result.meta,
                    })
                    replies[t] = result.text
                    totals["new"] += 1
                    print(f"[{name}] {key}: saved ({len(result.text)} chars, "
                          f"{result.meta.get('prompt_tokens')} prompt tokens)", flush=True)

    print(f"Done: {totals['new']} new turns, {totals['skipped']} already saved (skipped), "
          f"{totals['no_phase1']} conversations without a Phase 1 response (run Phase 1 first), "
          f"{totals['errors']} errors.")
    return totals


def check_context(adapter, result, key):
    """Stop before saving a turn the server may have cut: Ollama silently drops the start of a conversation
    that does not fit its context window (by default 4096 tokens, too small for 5 turns)."""
    limit = adapter.context_length() if hasattr(adapter, "context_length") else None
    used = result.meta.get("prompt_tokens")
    if limit and used is not None and used + adapter.max_tokens > limit:
        raise SystemExit(
            f"{key}: the conversation is {used} tokens and a reply may add {adapter.max_tokens}, but the "
            f"server's context window is {limit} tokens, so the start of the conversation could be dropped. "
            f"Restart Ollama with a larger window (e.g. `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`) and rerun."
        )


def _stale_turns(saved, phase1, ladders, version):
    stale = []
    by_conv = {}
    for rec in saved.values():
        by_conv.setdefault(rec["turn0_key"], {})[rec["turn"]] = rec
    for conv_key, turns in by_conv.items():
        turn0 = phase1.get(conv_key)
        ladder = (ladders.get(turns[min(turns)]["scenario_id"]) or {}).get("messages")
        replies = {t: rec["response"] for t, rec in turns.items()}
        for t, rec in sorted(turns.items()):
            fresh = (
                turn0 is not None and ladder is not None and t <= len(ladder)
                and all(u in replies for u in range(1, t))
                and rec["messages"] == conversation_messages(turn0, replies, ladder, t)
                and rec["model_version"] == version
            )
            if not fresh:
                stale.append(rec["key"])
    return sorted(stale)


def conversation_turns(project, private=False):
    """Every turn of every Phase 2 conversation, turn 0 (the Phase 1 reply) included.

    Conversations whose turn 1 no longer continues the saved Phase 1 reply are left out.
    """
    phase1 = {r["key"]: r for r in read_jsonl(project.responses_file(private))}
    convs = {}
    for rec in read_jsonl(project.pressure_file(private)):
        convs.setdefault(rec["turn0_key"], []).append(rec)
    turns, stale = [], []
    for conv_key, recs in convs.items():
        turn0 = phase1.get(conv_key)
        recs = sorted(recs, key=lambda r: r["turn"])
        if turn0 is None or recs[0]["turn"] != 1 or recs[0]["messages"][-2]["content"] != turn0["response"]:
            stale.append(conv_key)
            continue
        base = {"system": turn0["system"], "scenario_id": turn0["scenario_id"], "run": turn0["run"]}
        turns.append({**base, "key": turn_key(base["system"], base["scenario_id"], base["run"], 0), "turn": 0,
                      "user_message": turn0["messages"][-1]["content"], "previous_reply": None,
                      "response": turn0["response"]})
        for rec in recs:
            turns.append({**base, "key": rec["key"], "turn": rec["turn"],
                          "user_message": rec["messages"][-1]["content"],
                          "previous_reply": rec["messages"][-2]["content"], "response": rec["response"]})
    if stale:
        print(f"WARNING: {len(stale)} Phase 2 conversation(s) no longer continue their Phase 1 response and "
              f"are left out (e.g. {stale[0]}). Rerun `pressure run --replace`.", file=sys.stderr)
    return turns


def priority_pairs(project, systems, top):
    """The `top` (system, scenario) pairs with the highest Phase 1 judge rate of accepting the framing."""
    from .analysis import build_response_table

    df = build_response_table(project)
    df = df[df.system.isin(systems)]
    if "judge_framing" not in df or df.judge_framing.isna().all():
        raise SystemExit("--top ranks by Phase 1 judge scores: run `run` and `judge` for these systems first.")
    cols = [c for c in ("judge_framing", "judge_complaisant") if c in df]
    ranked = df.groupby(["system", "scenario_id"])[cols].mean().sort_values(cols, ascending=False).head(top)
    print("Running the ladders on the most agreeable Phase 1 pairs (mean judge framing, complaisant):")
    print(ranked.round(3).to_string())
    return set(ranked.index)
