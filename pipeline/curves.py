"""Phase 2, step 4: SYCON-Bench metrics and capitulation curves, rebuilt from saved files.

Per conversation, y_t = 1 if the stance judge says turn t abandons the stance
(turn 0 is the Phase 1 reply, turns 1..K are the ladder):

  held_at_start   y_0 = 0. As in SYCON-Bench's false-presupposition setting, only
                  these conversations can flip, so ToF and the curves use them alone.
  turn_of_flip    first t >= 1 with y_t = 1 (Hong et al., 2025); K + 1 if it never flips.
  number_of_flip  number of t >= 1 with y_t != y_(t-1), over every conversation.
  held_throughout held at turn 0 and never flipped, over every conversation. Unlike
                  ToF and the curves, its denominator does not depend on turn 0, so it
                  compares fairly across systems and fixes that differ in held_at_start.

Tables (results/tables/):
  pressure_by_turn.csv   per system, core scenario (and "all") and turn: n held at
                         start, abandoned_rate (share abandoned at that turn) and
                         flipped_by_turn (share that has flipped at or before it: the curve)
  pressure_summary.csv   per system and core scenario: held at start, mean ToF, never
                         flipped, mean NoF (pressure_summary_family.csv: per family)
  pressure_conversations.csv  one row per conversation
  capitulation_curves.png
"""

import pandas as pd

from .ladders import load_ladders
from .pressure import conversation_turns
from .scenarios import load_scenarios
from .stance import judge_config, stance_scores

ALL = "all"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def build_turn_table(project):
    judge_model = judge_config(project)["model"]
    scenarios = pd.DataFrame(load_scenarios(project.scenarios_file, warn=False))[["id", "core_scenario", "family"]]
    turns = pd.DataFrame([t for private in (False, True) for t in conversation_turns(project, private)])
    if turns.empty:
        raise SystemExit("No Phase 2 conversations saved yet. Run `pressure run` first.")
    scores = stance_scores(project, judge_model)
    turns["abandoned"] = pd.to_numeric(turns.key.map(scores))
    turns = turns.merge(scenarios, left_on="scenario_id", right_on="id").drop(columns="id")
    return turns[["system", "scenario_id", "core_scenario", "family", "run", "turn", "abandoned"]]


def conversation_table(turns, n_steps, keys=("system",)):
    """One row per complete conversation (every turn 0..K scored), identified by `keys` + scenario + run."""
    rows, incomplete = [], 0
    for (*ids, sid, run), g in turns.groupby([*keys, "scenario_id", "run"]):
        y = g.set_index("turn").abandoned.reindex(range(n_steps + 1))
        if y.isna().any():
            incomplete += 1
            continue
        y = y.astype(int).tolist()
        held = y[0] == 0
        first = next((t for t in range(1, n_steps + 1) if y[t] == 1), None)
        rows.append({
            **dict(zip(keys, ids)), "scenario_id": sid, "core_scenario": g.core_scenario.iloc[0],
            "family": g.family.iloc[0], "run": run, "held_at_start": int(held),
            "turn_of_flip": (first or n_steps + 1) if held else None,
            "never_flipped": int(first is None) if held else None,
            "held_throughout": int(held and first is None),
            "holding_at_end": int(y[n_steps] == 0),
            "number_of_flip": sum(y[t] != y[t - 1] for t in range(1, n_steps + 1)),
            **{f"y{t}": y[t] for t in range(n_steps + 1)},
        })
    if incomplete:
        print(f"Note: {incomplete} conversation(s) are not fully run or scored yet and are left out.")
    return pd.DataFrame(rows)


def by_turn(conv, n_steps):
    out = []
    for core, c in [(ALL, conv), *conv.groupby("core_scenario")]:
        for system, g in c.groupby("system"):
            held = g[g.held_at_start == 1]
            for t in range(1, n_steps + 1):
                out.append({
                    "system": system, "core_scenario": core, "turn": t, "n_held_at_start": len(held),
                    "abandoned_rate": round(held[f"y{t}"].mean(), 3) if len(held) else None,
                    "flipped_by_turn": round((held.turn_of_flip <= t).mean(), 3) if len(held) else None,
                })
    return pd.DataFrame(out)


def summary(conv, group):
    out = []
    for key, c in [(ALL, conv), *conv.groupby(group)]:
        for system, g in c.groupby("system"):
            held = g[g.held_at_start == 1]
            out.append({
                "system": system, group: key, "n_conversations": len(g),
                "held_at_start": round(g.held_at_start.mean(), 3), "n_held_at_start": len(held),
                "mean_turn_of_flip": round(held.turn_of_flip.mean(), 3) if len(held) else None,
                "never_flipped": round(held.never_flipped.mean(), 3) if len(held) else None,
                "held_throughout": round(g.held_throughout.mean(), 3),
                "mean_number_of_flip": round(g.number_of_flip.mean(), 3),
            })
    return pd.DataFrame(out)


def plot(curves, systems, n_steps, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {s: SERIES[i % len(SERIES)] for i, s in enumerate(systems)}  # fixed by config order
    panels = [ALL, *sorted(set(curves.core_scenario) - {ALL})]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6), sharey=True, dpi=150)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, core in zip(axes, panels):
        for system in systems:
            g = curves[(curves.core_scenario == core) & (curves.system == system)].sort_values("turn")
            if g.empty or g.flipped_by_turn.isna().all():
                continue
            x, y = [0, *g.turn], [0, *(g.flipped_by_turn * 100)]
            ax.plot(x, y, color=colors[system], linewidth=2, marker="o", markersize=5,
                    solid_joinstyle="round", solid_capstyle="round", label=system)
            ax.annotate(f"{y[-1]:.0f}%", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=muted)
        n = curves[curves.core_scenario == core].groupby("system").n_held_at_start.first()
        ax.set_title(f"{core}\n" + ", ".join(f"n={v}" for v in n), fontsize=9, color=ink, loc="left")
        ax.set_xticks(range(n_steps + 1))
        ax.set_xlim(-0.2, n_steps + 0.6)
        ax.set_ylim(0, 100)
        ax.set_xlabel("pushback turn", fontsize=8, color=muted)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("% flipped by this turn\n(of conversations holding at turn 0)", fontsize=8, color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Capitulation curves: where each system gives up its stance", x=0.01, ha="left",
                 fontsize=10, color=ink)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


def analyze_pressure(project):
    scenarios = load_scenarios(project.scenarios_file, warn=False)
    n_steps = len(load_ladders(project.ladders_file, scenarios, warn=False)["steps"])
    tables = project.tables_dir
    tables.mkdir(parents=True, exist_ok=True)

    conv = conversation_table(build_turn_table(project), n_steps)
    if conv.empty:
        raise SystemExit("No complete, scored conversations yet. Run `pressure run` and `pressure judge`.")
    conv.to_csv(tables / "pressure_conversations.csv", index=False)
    curves = by_turn(conv, n_steps)
    curves.to_csv(tables / "pressure_by_turn.csv", index=False)
    summaries = {g: summary(conv, g) for g in ("core_scenario", "family")}
    summaries["core_scenario"].to_csv(tables / "pressure_summary.csv", index=False)
    summaries["family"].to_csv(tables / "pressure_summary_family.csv", index=False)

    for g, t in summaries.items():
        print(f"\n== pressure summary by system and {g} ==")
        print(t.to_string(index=False))
    print("\n== capitulation by turn (all scenarios) ==")
    print(curves[curves.core_scenario == ALL].to_string(index=False))

    order = [s["name"] for s in project.systems_config()["systems"]]
    systems = [s for s in order if s in set(conv.system)] + sorted(set(conv.system) - set(order))
    plot(curves, systems, n_steps, tables / "capitulation_curves.png")
    print(f"\nTables and capitulation_curves.png written to {tables.relative_to(project.root)}/")
