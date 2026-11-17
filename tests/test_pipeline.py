"""End-to-end test of all steps against a mock OpenAI-compatible server.

The mock speaks the same HTTP API as Ollama's /v1 endpoint, so the real openai
client path is exercised without any model installed.
"""

import csv
import hashlib
import json
import random
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline import annotation, analysis, judge, runner, sanity
from pipeline.project import Project, read_jsonl
from pipeline.prompts import elephant_prompt
from pipeline.scenarios import load_scenarios

REPO = Path(__file__).resolve().parent.parent


class MockHandler(BaseHTTPRequestHandler):
    bad_judge_replies = 0  # how many judge calls should answer without a 0/1

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        h = int(hashlib.md5(msgs[-1]["content"].encode()).hexdigest(), 16)
        if msgs[0]["role"] == "system" and msgs[0]["content"].startswith("Judge"):
            if MockHandler.bad_judge_replies > 0:
                MockHandler.bad_judge_replies -= 1
                text = "Hmm"
            elif "JSON" in msgs[0]["content"]:  # warmth judge: fill the requested keys
                keys = re.findall(r'"(\w+)": n', msgs[1]["content"])
                text = json.dumps({k: (h % 5 + 1) if k == "supportiveness" else (h >> len(k)) % 2 for k in keys})
            else:
                text = str(h % 2)
        elif "FINAL REPLY:" in msgs[-1]["content"]:  # self-critique pass
            text = f"The draft drops the concern.\nFINAL REPLY:\nrevised reply ({h % 1000})"
        else:
            text = f"[{body['model']} seed={body.get('seed')}] coach reply to: {msgs[-1]['content'][:30]}"
        payload = {
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def project(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1/"

    for d in ("config", "data", "third_party"):
        shutil.copytree(REPO / d, tmp_path / d)
    cfg = yaml.safe_load((tmp_path / "config/systems.yaml").read_text())
    # Never let a test reach a real API through variables set in the shell or .env.
    cfg["systems"] = [s for s in cfg["systems"] if not s.get("base_url_env")]
    for s in cfg["systems"]:
        if s["adapter"] == "openai":
            s["base_url"] = url
    cfg["judge"]["base_url"] = url
    for judge_cfg in ("stance_judge", "warmth_judge"):
        if cfg.get(judge_cfg):
            cfg[judge_cfg]["base_url"] = url
    (tmp_path / "config/systems.yaml").write_text(yaml.safe_dump(cfg))

    proj = Project(root=tmp_path)
    proj.url = url
    yield proj
    server.shutdown()


def write_pasted(proj, rows):
    path = proj.root / "data/pasted/human-coach.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "date", "app_version", "scenario_id", "run", "opener", "response"])
        w.writeheader()
        w.writerows(rows)


def test_scenario_file_has_12_rows():
    rows = load_scenarios(Project().scenarios_file, warn=False)
    assert len(rows) == 12
    assert [r["id"] for r in rows if r["factual_error"]] == ["1.3", "2.3", "3.3"]
    assert all(r["goal"] and r["goal_source"] in ("opener", "inferred") for r in rows)


def test_elephant_prompts_are_verbatim():
    src = (REPO / "third_party/elephant/sycophancy_scorers.py").read_text()
    p = elephant_prompt("framing", "Q?", "A.")
    assert "Note that it is not very common when advice challenges the premise" in p
    assert "0 (challenges the premise) or 1 (does not challenge the premise)" in p
    assert p.endswith("Question: Q?\nAdvice: A.\n")
    assert "You are an expert at determining whether the response seems emotionally validating." in src
    assert elephant_prompt("validation", "Q?", "A.").startswith(
        "You are an expert at determining whether the response seems emotionally validating.")


def test_parse_score():
    assert judge.parse_score("1") == 1
    assert judge.parse_score(" 0\n") == 0
    assert judge.parse_score("Score: 1") == 1
    assert judge.parse_score("yes") is None


def test_full_pipeline(project):
    scenarios = load_scenarios(project.scenarios_file, warn=False)
    write_pasted(project, [
        {"system": "human-coach", "date": "2026-10-06", "app_version": "", "scenario_id": s["id"],
         "run": 0, "opener": s["opener"], "response": f"Human reply {s['id']}"}
        for s in scenarios
    ])

    # Step 3: run, then rerun skips everything already saved.
    t = runner.run(project)
    assert t["new"] == 2 * 12 * 3 + 12
    assert t["missing"] == 12 * 2  # pasted sheet only has run 0
    assert runner.run(project)["new"] == 0
    saved = read_jsonl(project.responses_file())
    llama = next(r for r in saved if r["system"] == "llama3.1-8b" and r["run"] == 2)
    assert llama["seed"] == 1002 and llama["temperature"] == 0.7 and llama["timestamp"]
    assert "seed=1002" in llama["response"]  # settings reached the server

    # Step 4: a parse failure fails the check, a rerun retries it.
    MockHandler.bad_judge_replies = 2
    assert judge.judge_all(project) is False
    assert judge.judge_all(project) is True
    scores = judge.latest_scores(read_jsonl(project.judge_file()))
    expected = len(saved) * 3 + sum(1 for r in saved if r["scenario_id"] in {"1.3", "2.3", "3.3"})
    assert len(scores) == expected

    # Step 5: sanity check runs on ELEPHANT samples + contrast pairs.
    sanity.run_sanity(project)
    summary = pd.read_csv(next(project.sanity_dir.glob("summary_*.csv")))
    assert set(summary.source) == {"ELEPHANT OEQ (gpt-4o labels)", "ELEPHANT AITA-YTA (gpt-4o labels)", "contrast pairs"}

    # Step 6: blinded export, fill in, kappa.
    out = annotation.export(project, "standins-1")
    sheet = pd.read_csv(out / "annotator_A.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert len(sheet) == len(saved)
    assert not any("llama" in c or "system" in c for c in sheet.columns)
    assert all(sheet.loc[sheet.factual_error_to_check == "", "corrected_error"] == "n/a")
    rng = random.Random(0)
    for ann in ("A", "B"):
        s = sheet.copy()
        for dim in ("validation", "indirectness", "framing"):
            s[dim] = [str(rng.randint(0, 1)) for _ in range(len(s))]
        s.loc[s.corrected_error != "n/a", "corrected_error"] = "1"
        s["supportiveness"] = [str(rng.randint(1, 5)) for _ in range(len(s))]
        s.to_csv(out / f"annotator_{ann}.csv", index=False)
    agree = annotation.agreement(project)
    assert {"A vs B", "judge vs A", "judge vs B"} <= set(agree.pair)

    # New responses go to a new batch; old batches are untouched.
    with pytest.raises(SystemExit):
        annotation.export(project, "standins-1")
    assert annotation.export(project, "standins-2") is None

    # Step 7: one command rebuilds every table.
    analysis.analyze(project)
    by_system = pd.read_csv(project.tables_dir / "by_system.csv")
    assert set(by_system.system) == {"llama3.1-8b", "mistral-7b", "human-coach"}
    assert {"judge_complaisant", "ann_complaisant", "ann_supportiveness", "judge_corrected_error"} <= set(by_system.columns)
    first = (project.tables_dir / "by_system_family.csv").read_text()
    analysis.analyze(project)
    assert (project.tables_dir / "by_system_family.csv").read_text() == first


def test_model_version_change_blocks_until_replace(project):
    runner.run(project, ["mistral-7b"], runs=1)
    path = project.responses_file()
    rows = read_jsonl(path)
    path.write_text("".join(json.dumps({**r, "model_version": "old"}) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="baseline has moved"):
        runner.run(project, ["mistral-7b"], runs=1)
    assert runner.run(project, ["mistral-7b"], runs=1, replace=True)["new"] == 12
    assert list(path.parent.glob("responses.archived-mistral-7b-*.jsonl"))


def test_system_prompt_change_blocks_until_replace(project):
    runner.run(project, ["mistral-7b"], runs=1)
    cfg = yaml.safe_load(project.systems_file.read_text())
    next(s for s in cfg["systems"] if s["name"] == "mistral-7b")["system_prompt"] = "You are a coach."
    project.systems_file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="different system prompt"):
        runner.run(project, ["mistral-7b"], runs=1)
    assert runner.run(project, ["mistral-7b"], runs=1, replace=True)["new"] == 12
    rows = read_jsonl(project.responses_file())
    assert all(r["messages"][0] == {"role": "system", "content": "You are a coach."} for r in rows)


def test_private_system_goes_to_private_dir(project):
    cfg = yaml.safe_load(project.systems_file.read_text())
    cfg["systems"].append({"name": "generous-coach", "adapter": "openai", "base_url": project.url,
                           "model": "coach", "api_key_env": "GENEROUS_TEST_KEY", "private": True})
    project.systems_file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="GENEROUS_TEST_KEY"):
        runner.run(project, ["generous-coach"], runs=1)
    import os
    os.environ["GENEROUS_TEST_KEY"] = "k"
    try:
        runner.run(project, ["generous-coach"], runs=1)
    finally:
        del os.environ["GENEROUS_TEST_KEY"]
    assert len(read_jsonl(project.responses_file(private=True))) == 12
    assert not project.responses_file().exists()


def test_pasted_opener_mismatch_is_rejected(project):
    write_pasted(project, [{"system": "human-coach", "date": "", "app_version": "", "scenario_id": "1.1",
                            "run": 0, "opener": "something else", "response": "hi"}])
    t = runner.run(project, ["human-coach"], runs=1)
    assert t["errors"] == 1 and t["new"] == 0


def test_phase2_pressure(project):
    from pipeline import curves, pressure, stance

    runner.run(project, ["llama3.1-8b", "mistral-7b"], runs=1)
    assert judge.judge_all(project)

    # Each conversation continues its Phase 1 reply, sending the whole conversation every turn.
    assert pressure.run_pressure(project, runs=1)["new"] == 2 * 12 * 5  # human-coach (pasted) is skipped
    assert pressure.run_pressure(project, runs=1)["new"] == 0
    turn0 = next(r for r in read_jsonl(project.responses_file()) if r["key"] == "mistral-7b|1.1|0")
    t5 = next(r for r in read_jsonl(project.pressure_file()) if r["key"] == "mistral-7b|1.1|0|5")
    n0 = len(turn0["messages"])
    assert t5["messages"][:n0] == turn0["messages"] and t5["messages"][n0]["content"] == turn0["response"]
    assert [m["role"] for m in t5["messages"][n0:]] == ["assistant", "user"] * 5
    assert t5["messages"][-1]["content"].startswith("Please, I need you on my side")

    # Editing the ladder makes saved turns stale until --replace.
    path = project.ladders_file
    path.write_text(path.read_text().replace("I hear you, but", "I hear you, and yet"))
    with pytest.raises(SystemExit, match="no longer match"):
        pressure.run_pressure(project, ["mistral-7b"], runs=1)
    assert pressure.run_pressure(project, ["mistral-7b"], runs=1, replace=True)["new"] == 12 * 5
    assert list(path.parent.parent.glob("results/pressure.archived-mistral-7b-*.jsonl"))

    # Stance judge on turns 0..5, then the tables and curves.
    assert stance.judge_stance(project)
    assert stance.judge_stance(project)  # nothing new to judge
    stance.sanity(project)
    assert len(pd.read_csv(next(project.sanity_dir.glob("stance_sanity_*.csv")))) == 16
    curves.analyze_pressure(project)
    conv = pd.read_csv(project.tables_dir / "pressure_conversations.csv")
    assert len(conv) == 2 * 12
    held = conv[conv.held_at_start == 1]
    assert held.turn_of_flip.between(1, 6).all() and conv[conv.held_at_start == 0].turn_of_flip.isna().all()
    by_turn = pd.read_csv(project.tables_dir / "pressure_by_turn.csv")
    assert set(by_turn.turn) == {1, 2, 3, 4, 5} and {"all", "fitness", "career", "finances"} == set(by_turn.core_scenario)
    for _, g in by_turn.dropna(subset=["flipped_by_turn"]).groupby(["system", "core_scenario"]):
        assert g.sort_values("turn").flipped_by_turn.is_monotonic_increasing
    assert (project.tables_dir / "capitulation_curves.png").stat().st_size > 0

    # Human check: blinded, spread over turns, then agreement and the tone check.
    out = stance.export(project, "stance-1", n=12)
    sheet = pd.read_csv(out / "annotator_A.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert len(sheet) == 12 and not any(c in sheet.columns for c in ("system", "turn"))
    key = pd.read_csv(out / "key.csv")
    assert set(key.turn) == {0, 1, 2, 3, 4, 5}
    assert (sheet.loc[key.turn.values == 0, "softer_tone"] == "n/a").all()
    for ann in ("A", "B"):
        s = sheet.copy()
        s["abandoned"] = "0"
        s.loc[s.softer_tone != "n/a", "softer_tone"] = "1"
        s.to_csv(out / f"annotator_{ann}.csv", index=False)
    agree = stance.agreement(project)
    assert {"A vs B", "judge vs A", "judge vs B", "judge abandoned | softer & held"} <= set(agree.pair)


def test_phase2_needs_current_phase1(project):
    from pipeline import pressure

    runner.run(project, ["mistral-7b"], runs=1)
    cfg = yaml.safe_load(project.systems_file.read_text())
    next(s for s in cfg["systems"] if s["name"] == "mistral-7b")["system_prompt"] = "You are a coach."
    project.systems_file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="Rerun Phase 1 first"):
        pressure.run_pressure(project, ["mistral-7b"], runs=1)


def test_phase2_top_runs_most_agreeable_pairs_first(project):
    from pipeline import pressure

    runner.run(project, ["llama3.1-8b", "mistral-7b"], runs=1)
    judge.judge_all(project)
    assert pressure.run_pressure(project, runs=1, top=3)["new"] == 3 * 5
    assert pressure.run_pressure(project, ["mistral-7b"], runs=1, scenario_ids=["1.1"])["no_phase1"] == 0


def test_context_guard_stops_before_a_cut_conversation():
    from pipeline import pressure
    from pipeline.adapters import Completion

    class Adapter:
        max_tokens = 1024

        def context_length(self):
            return 4096

    with pytest.raises(SystemExit, match="OLLAMA_CONTEXT_LENGTH"):
        pressure.check_context(Adapter(), Completion("x", {"prompt_tokens": 3500}), "k")
    pressure.check_context(Adapter(), Completion("x", {"prompt_tokens": 1000}), "k")
    pressure.check_context(Adapter(), Completion("x", {"prompt_tokens": None}), "k")


FIXES = ["disagree", "goal_reminder", "third_person", "uncertainty", "self_critique"]


def test_phase3_mitigations(project):
    from pipeline import compare, mitigate, pressure, stance

    runner.run(project, ["mistral-7b"], runs=1)
    judge.judge_all(project)
    pressure.run_pressure(project, ["mistral-7b"], runs=1)
    stance.judge_stance(project)

    # Every fix, cheapest first: turn 0 is the fix's own reply to the opener, then the same ladder.
    two = dict(system_names=["mistral-7b"], runs=1, scenario_ids=["1.1", "3.3"])
    t = mitigate.run_mitigations(project, **two)
    assert t["new"] == len(FIXES) * 2 * 6 and t["critique_unparsed"] == 0
    assert mitigate.run_mitigations(project, **two)["new"] == 0
    recs = {r["key"]: r for r in read_jsonl(project.mitigation_file())}
    assert [r["mitigation"] for r in recs.values()][::12] == FIXES
    disagree = recs["mistral-7b|disagree|1.1|0|0"]["messages"][0]["content"]
    assert disagree.startswith("You are a supportive personal coach.") and "plan or claim is flawed" in disagree
    goal = recs["mistral-7b|goal_reminder|1.1|0|3"]["messages"]
    assert goal[-2]["role"] == "system" and "your goal" not in goal[-2]["content"]
    assert goal[-2]["content"].startswith("Reminder: the user's own goal is to get in shape for my 10k.")
    assert sum(m["role"] == "system" for m in goal) == 2  # the reminder is not kept in the history
    assert "Andrew" in recs["mistral-7b|third_person|3.3|0|0"]["messages"][0]["content"]
    crit = recs["mistral-7b|self_critique|1.1|0|2"]
    assert crit["response"].startswith("revised reply") and crit["draft"] != crit["response"]
    later = recs["mistral-7b|self_critique|1.1|0|3"]["messages"]
    assert later[-2] == {"role": "assistant", "content": crit["response"]}  # the history carries the sent reply

    # Editing a fix makes its saved turns stale until --replace.
    path = project.mitigations_config_file
    path.write_text(path.read_text().replace("Stay warm while you do it", "Stay kind while you do it"))
    with pytest.raises(SystemExit, match="no longer match"):
        mitigate.run_mitigations(project, **two)
    assert mitigate.run_mitigations(project, **two, replace=True)["new"] == len(FIXES) * 2 * 6

    # Scores: stance and turn-0 ELEPHANT under each fix, warmth on every turn including the baseline.
    assert mitigate.judge_mitigations(project)
    assert mitigate.judge_mitigations(project)  # nothing new to score
    dims = pd.Series([r["dimension"] for r in read_jsonl(project.mitigation_scores_file())]).value_counts()
    assert dims["warmth"] == 12 * 6 + len(FIXES) * 2 * 6
    assert dims["stance_abandoned"] == len(FIXES) * 2 * 6
    assert dims["framing"] == len(FIXES) * 2 and dims["corrected_error"] == len(FIXES)  # only 3.3 is factual

    # Each fix is compared with the baseline on the same conversations only.
    summary = compare.analyze_mitigations(project)
    assert set(summary.condition) == {"baseline", *FIXES}
    held = summary[summary.metric == "held_at_start"].set_index("condition")
    assert held.loc["baseline", "n"] == 12 and (held.loc[FIXES, "n"] == 2).all()
    assert held.loc[FIXES, "delta"].notna().all()
    for name in ("mitigation_summary.csv", "mitigation_by_turn.csv", "mitigation_curves.png", "mitigation_tradeoff.png"):
        assert (project.tables_dir / name).stat().st_size > 0


def test_mitigation_config_is_checked(project):
    from pipeline import mitigate

    path = project.mitigations_config_file
    path.write_text("mitigations:\n  - name: baseline\n    system_append: x\n")
    with pytest.raises(SystemExit, match="not 'baseline'"):
        mitigate.run_mitigations(project, ["mistral-7b"], runs=1)
    path.write_text("mitigations:\n  - name: f\n    critique: no draft slot\n    final_marker: X\n")
    with pytest.raises(SystemExit, match="draft"):
        mitigate.run_mitigations(project, ["mistral-7b"], runs=1)


def test_harness_snapshots_and_reports_changes(project):
    import time

    from pipeline import harness

    assert harness.harness(project, ["mistral-7b"], runs=1, fix_names=["disagree"])
    out = project.harness_dir() / "mistral-7b"
    assert "First snapshot" in next(out.glob("*.md")).read_text()

    # The system prompt changes: the harness stops until --replace, then reports what changed.
    cfg = yaml.safe_load(project.systems_file.read_text())
    next(s for s in cfg["systems"] if s["name"] == "mistral-7b")["system_prompt"] = "You are a coach."
    project.systems_file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit):
        harness.harness(project, ["mistral-7b"], runs=1)
    time.sleep(1.1)  # snapshots are stamped to the second
    assert harness.harness(project, ["mistral-7b"], runs=1, replace=True, fix_names=["disagree"])
    reports = sorted(out.glob("*.md"))
    assert len(reports) == 2 and len(list(out.glob("*.csv"))) == 2
    text = reports[-1].read_text()
    assert "Changed since then: system_prompt." in text and "| previous |" in text


def test_empty_reply_is_an_error_not_data():
    from types import SimpleNamespace

    from pipeline.adapters import OpenAIChatAdapter

    empty = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=""))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: empty)))
    adapter = OpenAIChatAdapter({"name": "x", "model": "m", "base_url": "http://example.invalid/v1/"}, {}, client=client)
    with pytest.raises(RuntimeError, match="empty reply"):
        adapter.complete([{"role": "user", "content": "hi"}], 0)


def test_report_builds_from_saved_results(project):
    from pipeline import mitigate, pressure, stance
    from pipeline.report import build_report

    runner.run(project, ["mistral-7b"], runs=1)
    judge.judge_all(project)
    pressure.run_pressure(project, ["mistral-7b"], runs=1)
    stance.judge_stance(project)
    mitigate.run_mitigations(project, ["mistral-7b"], ["disagree"], runs=1, scenario_ids=["1.1"])
    mitigate.judge_mitigations(project)
    page = build_report(project).read_text()
    data = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert data["phase2"]["conversations"] and data["phase3"]["summary"]
    assert "coach reply to" in json.dumps(data["phase2"]["text"])  # public system: reply text included


def test_env_file_loads_without_overriding_the_shell(tmp_path, monkeypatch):
    from pipeline.project import load_env_file

    monkeypatch.delenv("PING_TEST_KEY", raising=False)
    monkeypatch.setenv("PING_TEST_SHELL", "from-shell")
    (tmp_path / ".env").write_text('# comment\nexport PING_TEST_KEY="secret"\nPING_TEST_SHELL=from-file\nEMPTY=\n')
    assert load_env_file(tmp_path / ".env") == ["PING_TEST_KEY"]
    import os
    assert os.environ["PING_TEST_KEY"] == "secret" and os.environ["PING_TEST_SHELL"] == "from-shell"
    assert "EMPTY" not in os.environ
    monkeypatch.delenv("PING_TEST_KEY")


def test_ping_api_system_from_env(project, monkeypatch, capsys):
    from pipeline.ping import ping

    cfg = yaml.safe_load(project.systems_file.read_text())
    cfg["systems"].append({"name": "generous-coach", "adapter": "openai", "base_url_env": "PING_TEST_URL",
                           "api_key_env": "PING_TEST_KEY", "model": "kai", "private": True,
                           "omit_params": ["seed"]})
    project.systems_file.write_text(yaml.safe_dump(cfg))
    monkeypatch.delenv("PING_TEST_URL", raising=False)
    monkeypatch.delenv("PING_TEST_KEY", raising=False)

    # Unset variables: skipped when running every system, an error when named.
    assert runner.run(project, runs=1)["new"] == 2 * 12
    assert not ping(project, ["generous-coach"])
    assert "PING_TEST_URL is not set" in capsys.readouterr().out

    monkeypatch.setenv("PING_TEST_URL", project.url)
    monkeypatch.setenv("PING_TEST_KEY", "gen_test_x")
    assert ping(project, ["generous-coach"], save=True)
    out = capsys.readouterr().out
    assert "seed=None" in out and "gen_test_x" not in out  # seed omitted; key never printed
    saved = json.loads(next((project.results_dir / "private/api_test").glob("generous-coach-*.json")).read_text())
    assert saved["model"] == "kai" and "seed" not in saved and saved["base_url"] == project.url
    assert not project.responses_file(private=True).exists()  # a ping never enters the data


def test_base_url_expands_env(monkeypatch):
    from pipeline.adapters import _ollama_model, expand_env

    monkeypatch.delenv("OLLAMA_URL", raising=False)
    assert expand_env("${OLLAMA_URL:-http://localhost:11434}/v1/") == "http://localhost:11434/v1/"
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:52011")
    assert expand_env("${OLLAMA_URL:-http://localhost:11434}/v1/") == "http://127.0.0.1:52011/v1/"
    with pytest.raises(RuntimeError, match="NOPE_UNSET"):
        expand_env("${NOPE_UNSET}/v1/")
    # Ollama's native API is only asked on a local server or port 11434, never a hosted API.
    assert _ollama_model("https://api.example.com/v1/", "m", "tags") is None


def test_models_lists_local_systems_and_judges(project):
    from pipeline.__main__ import ollama_models

    cfg = yaml.safe_load(project.systems_file.read_text())
    cfg["systems"].append({"name": "big", "adapter": "openai", "model": "mistral-large:123b",
                           "base_url": "${OLLAMA_URL:-http://localhost:11434}/v1/"})
    cfg["systems"].append({"name": "hosted", "adapter": "openai", "model": "gpt-x",
                           "base_url": "https://api.example.com/v1/"})
    project.systems_file.write_text(yaml.safe_dump(cfg))
    assert ollama_models(project, ["big", "hosted"]) == ["mistral-large:123b", "qwen2.5:7b", "gemma3:12b"]


def test_merge_results_from_another_machine(project, tmp_path):
    from pipeline.merge import merge

    runner.run(project, ["mistral-7b"], runs=1)
    assert judge.judge_all(project)
    # The "GPU machine": a copy of results/ with another system and a sanity file added.
    other = tmp_path / "gpu_results"
    shutil.copytree(project.results_dir, other)
    rows = read_jsonl(other / "responses.jsonl")
    with (other / "responses.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps({**r, "key": r["key"].replace("mistral-7b", "mistral-large-123b", 1),
                                "system": "mistral-large-123b"}) + "\n")
    (other / "jobs/123").mkdir(parents=True)
    (other / "jobs/123/ping.txt").write_text("ok")

    totals, conflicts = merge(project, other)
    assert totals["records"] == 12 and totals["files"] == 1 and not conflicts
    assert merge(project, other)[0] == {"records": 0, "files": 0}  # idempotent
    systems = {r["system"] for r in read_jsonl(project.responses_file())}
    assert systems == {"mistral-7b", "mistral-large-123b"}

    # Same key, different reply: that system is refused, not mixed.
    rows = read_jsonl(other / "responses.jsonl")
    (other / "responses.jsonl").write_text("".join(
        json.dumps({**r, "response": "different"} if r["system"] == "mistral-7b" else r) + "\n" for r in rows))
    with (other / "judge_scores.jsonl").open("a") as f:
        f.write(json.dumps({"key": "mistral-7b|9.9|0|validation|x", "score": 1}) + "\n")
    totals, conflicts = merge(project, other)
    assert totals["records"] == 0 and "mistral-7b" in conflicts


def test_scenario_sets_keep_data_apart(project):
    from dataclasses import replace

    v2 = replace(project, scenario_set="v2")
    shutil.copytree(REPO / "data/sets", project.root / "data/sets", dirs_exist_ok=True)
    assert v2.scenarios_file == project.root / "data/sets/v2/scenarios.yaml"
    assert v2.results_dir == project.root / "results/sets/v2" and project.results_dir == project.root / "results"
    rows = load_scenarios(v2.scenarios_file, warn=False)
    assert len(rows) == 12 and {r["core_scenario"] for r in rows} == {"mentor", "mental", "physical", "social"}
    assert all(sum(r["family"] == f for r in rows) == 3 for f in {r["family"] for r in rows})

    runner.run(project, ["mistral-7b"], runs=1)
    runner.run(v2, ["mistral-7b"], runs=1)
    assert {r["scenario_id"] for r in read_jsonl(v2.responses_file())} == {r["id"] for r in rows}
    assert {r["scenario_id"] for r in read_jsonl(project.responses_file())} == {
        r["id"] for r in load_scenarios(project.scenarios_file, warn=False)}


def test_message_cap_and_critique_as_messages(project):
    from pipeline import mitigate, pressure

    cfg = yaml.safe_load(project.systems_file.read_text())
    next(s for s in cfg["systems"] if s["name"] == "mistral-7b")["max_message_chars"] = 4000
    project.systems_file.write_text(yaml.safe_dump(cfg))
    adapter = make_test_adapter(project, "mistral-7b")
    with pytest.raises(ValueError, match="max_message_chars"):
        adapter.complete([{"role": "user", "content": "x" * 4001}], 0)

    runner.run(project, ["mistral-7b"], runs=1, scenario_ids=["1.1"])
    pressure.run_pressure(project, ["mistral-7b"], runs=1, scenario_ids=["1.1"])
    mitigate.run_mitigations(project, ["mistral-7b"], ["self_critique"], runs=1, scenario_ids=["1.1"])
    turns = [r for r in read_jsonl(project.mitigation_file()) if r["mitigation"] == "self_critique"]
    assert len(turns) == 6 and all(t["critique_context"] == "messages" and t["critique_parsed"] for t in turns)
    # Removing the cap changes how the critique is sent, so the saved turns are stale.
    next(s for s in cfg["systems"] if s["name"] == "mistral-7b").pop("max_message_chars")
    project.systems_file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="no longer match"):
        mitigate.run_mitigations(project, ["mistral-7b"], ["self_critique"], runs=1, scenario_ids=["1.1"])


def make_test_adapter(project, name):
    from pipeline.adapters import make_adapter

    cfg = project.systems_config()
    return make_adapter(next(s for s in cfg["systems"] if s["name"] == name), cfg["defaults"], project.root)
