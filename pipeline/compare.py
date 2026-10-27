"""Phase 3, step 3: which fixes move the numbers, and what each costs in warmth.

Rebuilt from saved files, no model calls. Per system, each fix is compared with the
baseline on the same (scenario, run) conversations only, so a fix run on a subset
(e.g. `--top`) is never compared with the whole baseline. Metrics per conversation:

  turn 0 (Phase 1)   framing, validation, indirectness, complaisant, corrected_error
  pressure (Phase 2) held_throughout (the headline: every conversation counts),
                     held_at_start, flipped (by the last turn, of those holding at
                     turn 0), turn_of_flip, number_of_flip. flipped and turn_of_flip
                     only count conversations that held at turn 0, so a fix that
                     raises held_at_start changes their denominator.
  warmth             each warmth field, averaged over the conversation's turns

`delta` is fix minus baseline; ci_low/ci_high is a 95% bootstrap interval that
resamples scenarios (replies within a scenario are not independent). `better` says
which direction is an improvement; warmth falling is the cost of a fix.

Tables (results/tables/): mitigation_summary.csv (long), mitigation_by_turn.csv,
mitigation_conversations.csv, mitigation_curves.png, mitigation_tradeoff.png
"""

import numpy as np
import pandas as pd

from .curves import SERIES, conversation_table
from .judge import ALL_DIMENSIONS, latest_scores
from .ladders import load_ladders
from .mitigate import BASELINE, STANCE, WARMTH, condition_turns, load_mitigations, mitigation_scores
from .scenarios import load_scenarios
from .stance import judge_config, stance_scores

KEYS = ["condition", "system", "scenario_id", "run"]
INK, MUTED, GRID, SURFACE, BASE_COLOR = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8d8c88"


def metrics(fields):
    """(column, label, better) for every reported metric."""
    return [
        ("framing", "accepts the framing (turn 0)", "lower"),
        ("complaisant", "complaisant (turn 0)", "lower"),
        ("validation", "validating (turn 0)", ""),
        ("indirectness", "indirect language (turn 0)", ""),
        ("corrected_error", "corrected the error (turn 0, factual rows)", "higher"),
        ("held_throughout", "holds the stance through every turn", "higher"),
        ("held_at_start", "holds the stance at turn 0", "higher"),
        ("flipped", "flipped by the last turn (of those holding)", "lower"),
        ("turn_of_flip", "turn of flip (of those holding; K+1 = never)", "higher"),
        ("number_of_flip", "number of flips", "lower"),
        *[(f"warmth_{f['name']}", f"{f['name']} (mean over turns)", "higher") for f in fields],
    ]


def build_turn_table(project):
    cfg = project.systems_config()
    fields = project.annotation_config()["warmth_fields"]
    turns = pd.DataFrame([t for private in (False, True) for t in condition_turns(project, private)])
    if turns.empty:
        raise SystemExit("Nothing to compare yet. Run Phase 1 and 2, then `mitigate run` and `mitigate judge`.")
    scores = mitigation_scores(project)
    base_stance = stance_scores(project, judge_config(project)["model"])
    phase1 = {(r["response_key"], r["dimension"]): r["score"] for r in latest_scores(project.all_judge_scores())
              if r["judge_model"] == cfg["judge"]["model"]}

    rows = zip(turns.key, turns.condition, turns.turn)
    turns["abandoned"] = [base_stance.get(k) if c == BASELINE else scores.get((k, STANCE)) for k, c, _ in rows]
    for dim in ALL_DIMENSIONS:
        turns[dim] = [None if t != 0 else phase1.get((k.rsplit("|", 1)[0], dim)) if c == BASELINE
                      else scores.get((k, dim)) for k, c, t in zip(turns.key, turns.condition, turns.turn)]
    warmth = [scores.get((k, WARMTH)) or {} for k in turns.key]
    for f in fields:
        turns[f"warmth_{f['name']}"] = [w.get(f["name"]) for w in warmth]
    num = ["abandoned", *ALL_DIMENSIONS, *[f"warmth_{f['name']}" for f in fields]]
    turns[num] = turns[num].apply(pd.to_numeric)
    sc = pd.DataFrame(load_scenarios(project.scenarios_file, warn=False))[["id", "core_scenario", "family"]]
    return turns.merge(sc, left_on="scenario_id", right_on="id").drop(columns="id")


def build_conversation_table(turns, n_steps, fields):
    conv = conversation_table(turns, n_steps, keys=("condition", "system"))
    if conv.empty:
        return conv
    t0 = turns[turns.turn == 0][[*KEYS, *ALL_DIMENSIONS]]
    warmth_cols = [f"warmth_{f['name']}" for f in fields]
    w = turns.groupby(KEYS)[warmth_cols].mean().reset_index()
    conv = conv.merge(t0, on=KEYS, how="left").merge(w, on=KEYS, how="left")
    both = conv[["validation", "framing"]].notna().all(axis=1)
    conv["complaisant"] = ((conv.validation == 1) & (conv.framing == 1)).astype(float).where(both)
    conv["flipped"] = (conv.turn_of_flip <= n_steps).astype(float).where(conv.held_at_start == 1)
    return conv


def bootstrap_delta(fix, base, col, reps=2000, seed=8813):
    ids = sorted(set(fix.scenario_id))

    def per_scenario(df):
        g = df.groupby("scenario_id")[col]
        return (g.sum().reindex(ids, fill_value=0).to_numpy(float),
                g.count().reindex(ids, fill_value=0).to_numpy(float))

    (fs, fc), (bs, bc) = per_scenario(fix), per_scenario(base)
    idx = np.random.default_rng(seed).integers(0, len(ids), (reps, len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = fs[idx].sum(1) / fc[idx].sum(1) - bs[idx].sum(1) / bc[idx].sum(1)
    d = d[~np.isnan(d)]
    return (round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)) if len(d) else (None, None)


def summarize(conv, order, fields):
    out = []
    for system, cs in conv.groupby("system"):
        base_all = cs[cs.condition == BASELINE]
        for cond in [c for c in [BASELINE, *order] if c in set(cs.condition)]:
            c = cs[cs.condition == cond]
            fix = base_all if cond == BASELINE else _same_pairs(c, base_all)
            base = base_all if cond == BASELINE else _same_pairs(base_all, fix)
            for col, label, better in metrics(fields):
                value = fix[col].mean()
                row = {"system": system, "condition": cond, "metric": col, "label": label, "better": better,
                       "n": int(fix[col].notna().sum()), "value": None if pd.isna(value) else round(value, 3)}
                if cond != BASELINE and len(fix):
                    b = base[col].mean()
                    row["baseline"] = None if pd.isna(b) else round(b, 3)
                    row["delta"] = None if pd.isna(value) or pd.isna(b) else round(value - b, 3)
                    row["ci_low"], row["ci_high"] = bootstrap_delta(fix, base, col)
                out.append(row)
    return pd.DataFrame(out)


def _same_pairs(df, ref):
    pairs = set(zip(ref.scenario_id, ref.run))
    return df[[p in pairs for p in zip(df.scenario_id, df.run)]]


def by_turn(turns, conv, n_steps, fields):
    """Curve and warmth per turn. `comparison` names the fix a pair of rows belongs to: under it, the
    fix and the baseline cover the same conversations. comparison == baseline is the whole baseline."""
    warmth_cols = [f"warmth_{f['name']}" for f in fields]
    out = []

    def curve(c, system, comparison, condition):
        held = c[c.held_at_start == 1]
        t_rows = turns[(turns.condition == condition) & (turns.system == system)]
        w = _same_pairs(t_rows, c).groupby("turn")[warmth_cols].mean()
        for t in range(n_steps + 1):
            row = {"system": system, "comparison": comparison, "condition": condition, "turn": t,
                   "n_conversations": len(c), "n_held_at_start": len(held),
                   "abandoned_rate": round(held[f"y{t}"].mean(), 3) if len(held) and t else None,
                   "flipped_by_turn": round((held.turn_of_flip <= t).mean(), 3) if len(held) else None}
            if t in w.index:
                row.update(w.loc[t].round(3).to_dict())
            out.append(row)

    for system, cs in conv.groupby("system"):
        base = cs[cs.condition == BASELINE]
        curve(base, system, BASELINE, BASELINE)
        for fix, c in cs[cs.condition != BASELINE].groupby("condition"):
            c = _same_pairs(c, base)
            curve(c, system, fix, fix)
            curve(_same_pairs(base, c), system, fix, BASELINE)
    return pd.DataFrame(out)


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_curves(curves, systems, order, n_steps, path):
    """One panel per system and fix: the fix's capitulation curve against the baseline's."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixes = [f for f in order if f in set(curves.comparison)]
    if not fixes:
        return
    colors = {f: SERIES[i % len(SERIES)] for i, f in enumerate(order)}  # fixed by config order
    fig, axes = plt.subplots(len(systems), len(fixes), figsize=(2.9 * len(fixes), 2.9 * len(systems)),
                             sharey=True, dpi=150, squeeze=False)
    for i, system in enumerate(systems):
        for j, fix in enumerate(fixes):
            ax = axes[i][j]
            for cond, color in ((BASELINE, BASE_COLOR), (fix, colors[fix])):
                g = curves[(curves.system == system) & (curves.comparison == fix)
                           & (curves.condition == cond)].sort_values("turn")
                if g.empty or g.flipped_by_turn.isna().all():
                    continue
                y = g.flipped_by_turn * 100
                ax.plot(g.turn, y, color=color, linewidth=2, marker="o", markersize=4,
                        solid_joinstyle="round", solid_capstyle="round", label=cond)
                ax.annotate(f"{y.iloc[-1]:.0f}%", (g.turn.iloc[-1], y.iloc[-1]), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7, color=MUTED)
            n = curves[(curves.system == system) & (curves.comparison == fix)].n_held_at_start
            ax.set_title(f"{fix}" + (f", {system}" if len(systems) > 1 else "")
                         + (f"\nn={n.max()} holding at turn 0" if len(n) else ""), fontsize=9, color=INK, loc="left")
            ax.set_xticks(range(n_steps + 1))
            ax.set_xlim(-0.2, n_steps + 0.8)
            ax.set_ylim(0, 100)
            _style(ax)
        axes[i][0].set_ylabel("% flipped by this turn", fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("pushback turn", fontsize=8, color=MUTED)
    handles = [plt.Line2D([], [], color=BASE_COLOR, linewidth=2, label=BASELINE),
               plt.Line2D([], [], color=MUTED, linewidth=2, label="with the fix (panel title)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Capitulation under pressure: each fix against the baseline on the same conversations",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _err(row, value, scale):
    """Error-bar lengths around `value` from a summary row's interval; none if there is no interval."""
    if pd.isna(row.ci_low) or pd.isna(row.ci_high):
        return [[0], [0]]
    return [[value - row.ci_low * scale], [row.ci_high * scale - value]]


def _label_points(ax, points):
    """Name each point; points that nearly coincide get their labels stacked instead of overlapping."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    placed = []
    for cond, x, y in points:
        near = sum(abs(x - px) < 0.06 * (x1 - x0) and abs(y - py) < 0.06 * (y1 - y0) for px, py in placed)
        ax.annotate(cond, (x, y), xytext=(7, 3 + 11 * near), textcoords="offset points", fontsize=8, color=INK)
        placed.append((x, y))


def plot_tradeoff(summary, systems, field, path):
    """Per fix: change in holding the stance throughout (y) against change in one warmth field (x)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary[summary.condition != BASELINE]
    if s.empty or "delta" not in s:
        return
    col = f"warmth_{field['name']}"
    scale, unit = (1, "points") if max(field["values"]) > 1 else (100, "percentage points")
    fig, axes = plt.subplots(1, len(systems), figsize=(4.4 * len(systems), 4.0), dpi=150, squeeze=False)
    for ax, system in zip(axes[0], systems):
        d = s[s.system == system].set_index(["condition", "metric"])
        points = []
        for cond in dict.fromkeys(s[s.system == system].condition):
            if (cond, col) not in d.index or (cond, "held_throughout") not in d.index:
                continue
            x, y = d.loc[(cond, col)], d.loc[(cond, "held_throughout")]
            if pd.isna(x.delta) or pd.isna(y.delta):
                continue
            xs, ys = x.delta * scale, y.delta * 100
            ax.errorbar(xs, ys, xerr=_err(x, xs, scale), yerr=_err(y, ys, 100), fmt="o", color=SERIES[0],
                        ecolor=GRID, elinewidth=1.5, markersize=6, capsize=0)
            points.append((cond, xs, ys))
        _label_points(ax, points)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel(f"change in {field['name']} ({unit})\n< colder    warmer >", fontsize=8, color=MUTED)
        ax.set_ylabel("change in % holding the stance throughout\n< gives way more    holds more >", fontsize=8, color=MUTED)
        ax.set_title(system, fontsize=9, color=INK, loc="left")
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.suptitle("What each fix buys, and its warmth cost\n(change vs baseline, 95% intervals)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def analyze_mitigations(project):
    scenarios = load_scenarios(project.scenarios_file, warn=False)
    n_steps = len(load_ladders(project.ladders_file, scenarios, warn=False)["steps"])
    order = list(load_mitigations(project.mitigations_config_file, scenarios))
    fields = project.annotation_config()["warmth_fields"]
    tables = project.tables_dir
    tables.mkdir(parents=True, exist_ok=True)

    turns = build_turn_table(project)
    conv = build_conversation_table(turns, n_steps, fields)
    if conv.empty:
        raise SystemExit("No complete, scored conversations yet. Run `mitigate run` and `mitigate judge`.")
    conv.to_csv(tables / "mitigation_conversations.csv", index=False)
    summary = summarize(conv, order, fields)
    summary.to_csv(tables / "mitigation_summary.csv", index=False)
    curves = by_turn(turns, conv, n_steps, fields)
    curves.to_csv(tables / "mitigation_by_turn.csv", index=False)

    headline = ["framing", "complaisant", "held_throughout", "held_at_start", "flipped", "number_of_flip",
                *[f"warmth_{f['name']}" for f in fields[:2]]]
    shown = summary[summary.metric.isin(headline)].copy()
    shown["cell"] = [f"{v}" if pd.isna(d) else f"{v} ({d:+.3f})"
                     for v, d in zip(shown.value, shown.get("delta", pd.Series(index=shown.index, dtype=float)))]
    wide = shown.pivot_table(index=["system", "condition"], columns="metric", values="cell", aggfunc="first")
    print("== value (change vs baseline on the same conversations); full table with 95% intervals in "
          "mitigation_summary.csv ==")
    print(wide.reindex(columns=[h for h in headline if h in wide.columns]).to_string())

    sys_order = [s["name"] for s in project.systems_config()["systems"]]
    systems = [s for s in sys_order if s in set(conv.system)] + sorted(set(conv.system) - set(sys_order))
    plot_curves(curves, systems, order, n_steps, tables / "mitigation_curves.png")
    plot_tradeoff(summary, systems, fields[0], tables / "mitigation_tradeoff.png")
    print(f"\nTables and plots written to {tables.relative_to(project.root)}/")
    return summary
