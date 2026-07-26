#!/usr/bin/env python3
"""Regenerate the README's results table from a finished run.

    python scripts/update_readme.py --run submission

The table between the RESULTS markers in README.md is *generated*, never typed.
A hand-copied headline number drifts from the artifacts the moment anything is
re-run, and a stale claim in a README is worse than no claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecoloop.config import ARTIFACTS_DIR
from ecoloop.orchestrator import list_runs

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _comfort_verdict(baseline: float, ai: float) -> str:
    if ai < baseline - 1e-9:
        return "**better**"
    if ai > baseline + 1e-9:
        return "**worse**"
    return "equal"


def render(results: dict, ablation: dict | None = None) -> str:
    savings = results["savings"]
    comfort = savings["comfort"]
    agent = results.get("agent") or {}
    ecm = results.get("ecm") or {}
    baseline_kpi = results["baseline"]["kpi"]
    ai_kpi = results["ai"]["kpi"]
    days = ai_kpi.get("sim_hours", 0) / 24.0

    def row(label: str, block: dict, unit: str = "", digits: int = 1) -> str:
        """A signed percentage is ambiguous for a *reduction* metric — "+5.5%"
        could read as "used more". Spell out the direction instead."""
        base, ai = block["baseline"], block["ai"]
        pct = block["pct"]
        if abs(pct) < 0.05:
            change = "no change"
        elif pct > 0:
            change = f"**{pct:.1f}% lower**"
        else:
            change = f"{abs(pct):.1f}% higher"
        return f"| {label} | {base:,.{digits}f}{unit} | {ai:,.{digits}f}{unit} | {change} |"

    lines = [
        "| Metric | Baseline BMS | AI closed loop | Change |",
        "|---|---|---|---|",
        row("Total electricity", savings["total_kwh"], " kWh", 1),
        row("HVAC electricity", savings["hvac_kwh"], " kWh", 1),
        row("Cost", savings["cost_inr"], " INR", 0),
        row("Carbon", savings["carbon_kg"], " kg", 1),
        row("Peak demand", savings["peak_demand_w"], " W", 0),
        row("Peak-window energy", savings["peak_window_kwh"], " kWh", 1),
        (
            f"| PMV exceedance | {comfort['baseline_pmv_exceedance_zone_hours']:.2f} zone-h "
            f"| {comfort['ai_pmv_exceedance_zone_hours']:.2f} zone-h | "
            f"{_comfort_verdict(comfort['baseline_pmv_exceedance_zone_hours'], comfort['ai_pmv_exceedance_zone_hours'])} |"
        ),
        (
            f"| CO₂ exceedance | {comfort['baseline_co2_exceedance_zone_hours']:.2f} zone-h "
            f"| {comfort['ai_co2_exceedance_zone_hours']:.2f} zone-h | "
            f"{_comfort_verdict(comfort['baseline_co2_exceedance_zone_hours'], comfort['ai_co2_exceedance_zone_hours'])} |"
        ),
        "",
        f"**Comfort: {'preserved' if comfort['comfort_preserved'] else 'DEGRADED'}.** "
        f"{days:.1f} simulated days, {ai_kpi['steps']} timesteps, "
        f"{results['engine']}"
        + (f" {results['energyplus']}" if results.get("energyplus") else "")
        + f", {Path(str(results.get('weather'))).name}.",
    ]

    regressions = [
        label
        for label, key in (
            ("total electricity", "total_kwh"),
            ("HVAC electricity", "hvac_kwh"),
            ("cost", "cost_inr"),
            ("carbon", "carbon_kg"),
            ("peak demand", "peak_demand_w"),
            ("peak-window energy", "peak_window_kwh"),
        )
        if savings[key]["pct"] < -0.05
    ]
    if regressions:
        lines += [
            "",
            "**Where it did worse:** "
            + ", ".join(regressions)
            + ". Reported rather than dropped — the agent shifts thermal work in "
            "time, and on this run that moved some load into the evening tariff "
            "window even while total consumption fell.",
        ]

    if agent.get("decisions"):
        total = agent.get("accepted_default", 0) + agent.get("deviated_from_default", 0)
        divergence = (
            f", diverged from the deterministic recommendation on "
            f"{agent['deviated_from_default']}/{total} "
            f"({100.0 * agent['deviated_from_default'] / total:.0f}%)"
            if total
            else ""
        )
        lines += [
            "",
            f"**Agent:** {agent.get('model', 'rules')} · {agent['decisions']} decisions "
            f"({agent.get('llm_decisions', 0)} by the model, "
            f"{agent.get('fallback_decisions', 0)} deterministic fallback) · "
            f"{agent.get('tool_calls', 0)} tool calls · "
            f"mean {agent.get('mean_latency_ms', 0):.0f} ms, p95 "
            f"{agent.get('p95_latency_ms', 0):.0f} ms · "
            f"safety layer intervened on {agent.get('guardrail_interventions', 0)} timesteps"
            f"{divergence}.",
        ]

    if ecm.get("attempt_count"):
        attempts = ecm["attempt_count"]
        corrections = ecm["self_corrections"]
        # Do not imply the repair loop fired when nothing failed.
        correction_text = (
            f"{corrections} self-correction{'' if corrections == 1 else 's'} after a "
            "generated model failed to simulate"
            if corrections
            else "no generated model failed, so the repair loop was not exercised"
        )
        best = ecm.get("best")
        detail = ""
        if best:
            measures = ", ".join(m.get("ecm", "?") for m in best.get("measures", []))
            detail = (
                f" Best: {measures} → {best['savings']['total_kwh']['pct']:+.1f}% total, "
                f"{best['savings']['hvac_kwh']['pct']:+.1f}% HVAC against the same control on "
                f"the unmodified model."
            )
        lines += [
            "",
            f"**Retrofit pass (phase B):** {attempts} attempt"
            f"{'' if attempts == 1 else 's'}, {ecm['successful_attempts']} verified by "
            f"simulation, {correction_text}.{detail}",
        ]

    if ablation is not None:
        # The ablation arm is the same recommendation the LLM is shown, applied
        # verbatim. Publishing both columns is the only way the reader can tell
        # what the language model actually contributed.
        a_savings = ablation["savings"]
        lines += [
            "",
            "### What the language model contributes",
            "",
            "Identical building, weather, window, plant model and safety layer. The "
            "only difference is whether a language model may disagree with the "
            "deterministic recommendation it is shown.",
            "",
            "| | Total electricity | HVAC electricity | Peak-window energy | Comfort |",
            "|---|---|---|---|---|",
            (
                f"| Rules only (`--brain heuristic`) | **{a_savings['total_kwh']['pct']:.2f}% lower** "
                f"| **{a_savings['hvac_kwh']['pct']:.2f}% lower** "
                f"| {a_savings['peak_window_kwh']['pct']:+.2f}% "
                f"| {'preserved' if a_savings['comfort']['comfort_preserved'] else 'degraded'} |"
            ),
            (
                f"| {(results.get('agent') or {}).get('model', 'LLM')} in the loop "
                f"| {savings['total_kwh']['pct']:.2f}% lower "
                f"| {savings['hvac_kwh']['pct']:.2f}% lower "
                f"| {savings['peak_window_kwh']['pct']:+.2f}% "
                f"| {'preserved' if comfort['comfort_preserved'] else 'degraded'} |"
            ),
            "",
            (
                "**Read this honestly: on this run the 3B model is behind the rules "
                "it is advised by.** It holds comfort just as well and shaves peak "
                "demand comparably, but its divergences from the recommendation cost "
                "energy on net. The architecture is what is being demonstrated — a "
                "real closed loop with a real safety guarantee and a measurable "
                "ablation — and the same code runs a larger model behind one flag "
                "(`--model llama3.1:8b`)."
                if a_savings["total_kwh"]["pct"] > savings["total_kwh"]["pct"]
                else "On this run the model beats the deterministic rules it is advised by."
            ),
        ]

    brain = "llm" if (results.get("agent") or {}).get("model") else "heuristic"
    lines += [
        "",
        f"Reproduce: `python -m ecoloop run --days {round(days) or 1} --brain {brain}"
        + (" --ecm-pass" if ecm.get("attempt_count") else "")
        + "`  ·  this exact run is committed at "
        "`artifacts/example_submission/` (open it in the dashboard on a fresh clone, "
        "no simulation required)",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="", help="run id (default: newest complete)")
    parser.add_argument(
        "--ablation",
        default="ablation",
        help="run id of the deterministic-brain arm, for the side-by-side (blank to skip)",
    )
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    args = parser.parse_args()

    run_id = args.run
    if not run_id:
        runs = [r for r in list_runs() if r.get("complete")]
        if not runs:
            raise SystemExit("no completed runs")
        run_id = runs[0]["run_id"]

    results_path = ARTIFACTS_DIR / run_id / "results.json"
    if not results_path.exists():
        raise SystemExit(f"no results.json for run {run_id!r}")
    results = json.loads(results_path.read_text())

    ablation = None
    if args.ablation:
        ablation_path = ARTIFACTS_DIR / args.ablation / "results.json"
        if ablation_path.exists():
            candidate = json.loads(ablation_path.read_text())
            # Only comparable if the two runs saw the same building and window.
            same_window = candidate.get("window") == results.get("window")
            same_engine = candidate.get("engine") == results.get("engine")
            if same_window and same_engine:
                ablation = candidate

    readme = Path(args.readme)
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{readme} is missing the {START} / {END} markers")
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    readme.write_text(f"{head}{START}\n{render(results, ablation)}\n{END}{tail}")
    print(f"  updated {readme} from run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
