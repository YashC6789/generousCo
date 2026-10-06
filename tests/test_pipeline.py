"""End-to-end test of all steps against a mock OpenAI-compatible server.

The mock speaks the same HTTP API as Ollama's /v1 endpoint, so the real openai
client path is exercised without any model installed.
"""

import csv
import hashlib
import json
import random
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
        if msgs[0]["role"] == "system" and msgs[0]["content"].startswith("Judge"):
            if MockHandler.bad_judge_replies > 0:
                MockHandler.bad_judge_replies -= 1
                text = "Hmm"
            else:
                text = str(int(hashlib.md5(msgs[1]["content"].encode()).hexdigest(), 16) % 2)
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
    for s in cfg["systems"]:
        if s["adapter"] == "openai":
            s["base_url"] = url
    cfg["judge"]["base_url"] = url
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
