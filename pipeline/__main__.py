"""Command line: python -m pipeline [--set NAME] <step>

--set NAME runs any step on the scenario set in data/sets/NAME/ and keeps its results in
results/sets/NAME/; without it, data/scenarios.yaml and results/ are used.

  ping --systems NAME [--message TEXT] [--save]
                             API test: one message, print the reply (not saved to results)
  scenarios                  step 1: check the scenario file
  run [--systems/--scenarios] steps 2-3: run scenarios through systems (skips saved)
  judge                      step 4: score every response with the local judge
  sanity                     step 5: judge sanity check
  annotate export BATCH      step 6: export a blinded annotation batch
  annotate kappa             step 6: agreement between annotators and judge
  analyze                    step 7: rebuild every table from saved files

Phase 2 (multi-turn pressure):
  pressure ladders [--show ID ...]  check data/ladders.yaml (and print ladders)
  pressure run [--systems/--scenarios/--runs/--top N/--replace]
                                    continue each Phase 1 reply with the ladder
  pressure judge                    stance judge on every turn
  pressure sanity                   stance judge on hand-written held / softer / abandoned replies
  pressure export BATCH [--n N]     export a blinded stance annotation batch
  pressure kappa                    agreement on the stance, and the tone check
  pressure analyze                  ToF / NoF tables and capitulation curves

Phase 3 (low-cost fixes, config/mitigations.yaml):
  mitigate run [--fixes ...] [--systems/--scenarios/--runs/--top N/--replace]
                                    rerun scenarios + ladders under each fix, cheapest first
  mitigate judge                    stance, turn-0 ELEPHANT and warmth scores (baseline warmth too)
  mitigate analyze                  each fix vs baseline, with 95% intervals; curves; warmth trade-off
  harness --systems NAME [--fixes [FIX ...]] [--replace] [--runs N]
                                    every step for a system, then a snapshot compared with the last one

  report                     one HTML page explaining every phase's results, from saved files

GPU machine (jobs/):
  models --systems NAME ...  the Ollama models a job needs (those systems + the judges)
  merge PATH                 add records from another machine's results/ copy, never overwriting
"""

import argparse
import sys

from .project import Project, load_env_file


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", dest="scenario_set", metavar="NAME",
                   help="scenario set: data/sets/NAME/, results in results/sets/NAME/ (default: data/scenarios.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("ping")
    pg.add_argument("--systems", nargs="+", required=True, help="system names from config/systems.yaml")
    pg.add_argument("--message", help="what the user says (default: the quit-my-job example)")
    pg.add_argument("--save", action="store_true",
                    help="also write the reply to results/api_test/ (results/private/ for private systems)")
    sub.add_parser("scenarios")
    r = sub.add_parser("run")
    r.add_argument("--systems", nargs="+", help="system names from config/systems.yaml (default: all)")
    r.add_argument("--runs", type=int, help="runs per scenario (default: config)")
    r.add_argument("--scenarios", nargs="+", help="scenario ids, e.g. 1.1 2.1 (default: all)")
    r.add_argument("--replace", action="store_true",
                   help="model version changed: archive this system's old responses and rerun")
    sub.add_parser("judge")
    sub.add_parser("sanity")
    a = sub.add_parser("annotate")
    asub = a.add_subparsers(dest="action", required=True)
    e = asub.add_parser("export")
    e.add_argument("batch", help="batch name, e.g. standins-1 or generous-1")
    asub.add_parser("kappa")
    sub.add_parser("analyze")
    pr = sub.add_parser("pressure")
    psub = pr.add_subparsers(dest="action", required=True)
    pl = psub.add_parser("ladders")
    pl.add_argument("--show", nargs="*", help="print the ladders (all, or these scenario ids)")
    prun = psub.add_parser("run")
    prun.add_argument("--systems", nargs="+", help="system names from config/systems.yaml (default: all)")
    prun.add_argument("--scenarios", nargs="+", help="scenario ids, e.g. 1.1 2.1 (default: all)")
    prun.add_argument("--runs", type=int, help="conversations per scenario, one per Phase 1 run (default: config)")
    prun.add_argument("--top", type=int, help="only the N (system, scenario) pairs most agreeable in Phase 1")
    prun.add_argument("--replace", action="store_true",
                      help="saved turns are stale: archive this system's Phase 2 data and rerun")
    psub.add_parser("judge")
    psub.add_parser("sanity")
    pe = psub.add_parser("export")
    pe.add_argument("batch", help="batch name, e.g. stance-1")
    pe.add_argument("--n", type=int, default=30, help="number of turns to export (default: 30)")
    psub.add_parser("kappa")
    psub.add_parser("analyze")
    m = sub.add_parser("mitigate")
    msub = m.add_subparsers(dest="action", required=True)
    mrun = msub.add_parser("run")
    mrun.add_argument("--fixes", nargs="+", help="fix names from config/mitigations.yaml (default: all, cheapest first)")
    mrun.add_argument("--systems", nargs="+", help="system names from config/systems.yaml (default: all)")
    mrun.add_argument("--scenarios", nargs="+", help="scenario ids, e.g. 1.1 2.1 (default: all)")
    mrun.add_argument("--runs", type=int, help="conversations per scenario (default: config)")
    mrun.add_argument("--top", type=int, help="only the N (system, scenario) pairs most agreeable in Phase 1")
    mrun.add_argument("--replace", action="store_true",
                      help="saved turns are stale: archive this system's Phase 3 turns and rerun")
    msub.add_parser("judge")
    msub.add_parser("analyze")
    h = sub.add_parser("harness")
    h.add_argument("--systems", nargs="+", required=True, help="system names from config/systems.yaml")
    h.add_argument("--fixes", nargs="*", help="also run Phase 3 fixes (no names: all of them)")
    h.add_argument("--runs", type=int, help="runs per scenario (default: config)")
    h.add_argument("--replace", action="store_true", help="prompt or model changed: archive old replies and rerun")
    sub.add_parser("report")
    mo = sub.add_parser("models")
    mo.add_argument("--systems", nargs="+", required=True, help="system names from config/systems.yaml")
    mg = sub.add_parser("merge")
    mg.add_argument("path", help="a copy of another machine's results/ folder")
    args = p.parse_args(argv)

    project = Project(scenario_set=args.scenario_set)
    if not project.scenarios_file.exists():
        raise SystemExit(f"no scenario set {args.scenario_set!r}: {project.scenarios_file} does not exist")
    load_env_file(project.root / ".env")  # API keys and private URLs; git-ignored
    if args.cmd == "ping":
        from .ping import ping
        return 0 if ping(project, args.systems, args.message, args.save) else 1
    if args.cmd == "scenarios":
        from .scenarios import check
        check(project)
    elif args.cmd == "run":
        from .runner import run
        run(project, args.systems, args.runs, args.replace, scenario_ids=args.scenarios)
    elif args.cmd == "judge":
        from .judge import judge_all
        return 0 if judge_all(project) else 1
    elif args.cmd == "sanity":
        from .sanity import run_sanity
        return 0 if run_sanity(project) else 1
    elif args.cmd == "annotate":
        from . import annotation
        if args.action == "export":
            annotation.export(project, args.batch)
        else:
            annotation.agreement(project)
    elif args.cmd == "analyze":
        from .analysis import analyze
        analyze(project)
    elif args.cmd == "pressure":
        return _pressure(project, args)
    elif args.cmd == "mitigate":
        return _mitigate(project, args)
    elif args.cmd == "report":
        from .report import build_report
        build_report(project)
    elif args.cmd == "models":
        print("\n".join(ollama_models(project, args.systems)))
    elif args.cmd == "merge":
        from .merge import merge
        merge(project, args.path)
    elif args.cmd == "harness":
        from .harness import harness
        return 0 if harness(project, args.systems, args.replace, args.fixes, args.runs) else 1
    return 0


def ollama_models(project, system_names):
    """Models served by the local Ollama: the named systems' and every judge's."""
    from .runner import select_systems

    cfg = project.systems_config()
    local = [s for s in select_systems(cfg, system_names) if s.get("adapter") == "openai"
             and ("OLLAMA_URL" in str(s.get("base_url", "")) or ":11434" in str(s.get("base_url", "")))]
    judges = [cfg[k] for k in ("judge", "stance_judge", "warmth_judge") if cfg.get(k)]
    return list(dict.fromkeys(m["model"] for m in local + judges))


def _mitigate(project, args):
    if args.action == "run":
        from .mitigate import run_mitigations
        run_mitigations(project, args.systems, args.fixes, args.runs, args.replace,
                        scenario_ids=args.scenarios, top=args.top)
    elif args.action == "judge":
        from .mitigate import judge_mitigations
        return 0 if judge_mitigations(project) else 1
    elif args.action == "analyze":
        from .compare import analyze_mitigations
        analyze_mitigations(project)
    return 0


def _pressure(project, args):
    if args.action == "ladders":
        from .ladders import check
        check(project, show=args.show)
    elif args.action == "run":
        from .pressure import run_pressure
        run_pressure(project, args.systems, args.runs, args.replace, scenario_ids=args.scenarios, top=args.top)
    elif args.action == "judge":
        from .stance import judge_stance
        return 0 if judge_stance(project) else 1
    elif args.action == "sanity":
        from .stance import sanity
        return 0 if sanity(project) else 1
    elif args.action == "export":
        from .stance import export
        export(project, args.batch, n=args.n)
    elif args.action == "kappa":
        from .stance import agreement
        agreement(project)
    elif args.action == "analyze":
        from .curves import analyze_pressure
        analyze_pressure(project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
