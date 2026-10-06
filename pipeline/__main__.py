"""Command line: python -m pipeline <step>

  scenarios                  step 1: check the scenario file
  run [--systems ...]        steps 2-3: run scenarios through systems (skips saved)
  judge                      step 4: score every response with the local judge
  sanity                     step 5: judge sanity check
  annotate export BATCH      step 6: export a blinded annotation batch
  annotate kappa             step 6: agreement between annotators and judge
  analyze                    step 7: rebuild every table from saved files
"""

import argparse
import sys

from .project import Project


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios")
    r = sub.add_parser("run")
    r.add_argument("--systems", nargs="+", help="system names from config/systems.yaml (default: all)")
    r.add_argument("--runs", type=int, help="runs per scenario (default: config)")
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
    args = p.parse_args(argv)

    project = Project()
    if args.cmd == "scenarios":
        from .scenarios import check
        check(project)
    elif args.cmd == "run":
        from .runner import run
        run(project, args.systems, args.runs, args.replace)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
