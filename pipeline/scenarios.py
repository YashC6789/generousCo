"""Step 1: load and check the scenario file."""

import sys

import yaml

REQUIRED = ["id", "core_scenario", "family", "opener", "correct_stance"]
EXPECTED_ROWS = 12


class ScenarioError(ValueError):
    pass


def load_scenarios(path, warn=True):
    rows = yaml.safe_load(path.read_text())
    if not isinstance(rows, list):
        raise ScenarioError(f"{path} must be a YAML list of scenarios")

    seen = set()
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED if not str(row.get(k) or "").strip()]
        if missing:
            raise ScenarioError(f"row {i} ({row.get('id')}) is missing {missing}")
        row["id"] = str(row["id"])
        if row["id"] in seen:
            raise ScenarioError(f"duplicate scenario id {row['id']}")
        seen.add(row["id"])
        row.setdefault("factual_error", None)

    if len(rows) != EXPECTED_ROWS:
        raise ScenarioError(f"expected {EXPECTED_ROWS} scenarios, found {len(rows)}")

    placeholders = [r["id"] for r in rows if r.get("placeholder")]
    if warn and placeholders:
        print(
            f"WARNING: {len(placeholders)} placeholder scenario(s) in {path.name}: "
            f"{', '.join(placeholders)}. Replace them before real runs.",
            file=sys.stderr,
        )
    return rows


def check(project):
    rows = load_scenarios(project.scenarios_file)
    factual = [r["id"] for r in rows if r["factual_error"]]
    print(f"OK: {len(rows)} scenarios loaded from {project.scenarios_file.name}")
    print(f"    core scenarios: {sorted({r['core_scenario'] for r in rows})}")
    print(f"    families:       {sorted({r['family'] for r in rows})}")
    print(f"    factual-error rows (extra corrected-error check): {factual}")
    return rows
