"""``python -m ecoloop`` — the command line.

    ecoloop doctor                  check EnergyPlus, weather, LLM reachability
    ecoloop run                     baseline + AI closed loop, then the savings
    ecoloop serve                   dashboard at http://127.0.0.1:8765
    ecoloop mcp                     MCP server on stdio
    ecoloop report <run_id>         re-print a finished run's savings table
    ecoloop tools                   the tool registry, as documentation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import ARTIFACTS_DIR, LLMConfig, RunConfig
from .energyplus_locate import find_energyplus
from .weather import resolve_epw

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    print(_c(f"Eco-Loop Building Agents {__version__}", BOLD))
    print()
    ok = True

    install = find_energyplus()
    if install:
        print(f"  {_c('OK', GREEN)}    EnergyPlus {install.version}")
        print(f"        {_c(str(install.root), DIM)}")
    else:
        ok = False
        print(f"  {_c('MISSING', YELLOW)}  EnergyPlus — the surrogate engine will be used instead")
        print(f"        {_c('fix: ./scripts/install_energyplus.sh', DIM)}")

    epw = resolve_epw(args.epw)
    if epw:
        print(f"  {_c('OK', GREEN)}    Weather {epw.name}")
    else:
        ok = False
        print(f"  {_c('MISSING', YELLOW)}  no .epw weather file — synthetic weather will be used")

    idf = Path(args.idf)
    if idf.exists():
        print(f"  {_c('OK', GREEN)}    Model {idf.name}")
    else:
        ok = False
        print(f"  {_c('MISSING', RED)}  model not found: {idf}")

    llm_cfg = LLMConfig.from_env()
    if args.model:
        llm_cfg.model = args.model
    from .agent.llm import LLMClient

    health = LLMClient(llm_cfg).health()
    if health.get("ok"):
        loaded = health.get("model_available")
        mark = _c("OK", GREEN) if loaded else _c("WARN", YELLOW)
        print(f"  {mark}    LLM {llm_cfg.provider} at {llm_cfg.base_url}")
        print(f"        model {llm_cfg.model}: {'available' if loaded else 'NOT PULLED'}")
        if not loaded:
            ok = False
            print(f"        {_c(f'fix: ollama pull {llm_cfg.model}', DIM)}")
            available = health.get("available") or []
            if available:
                print(f"        {_c('installed: ' + ', '.join(available[:6]), DIM)}")
    else:
        ok = False
        print(f"  {_c('MISSING', YELLOW)}  LLM unreachable at {llm_cfg.base_url}")
        print(f"        {_c(str(health.get('error'))[:100], DIM)}")
        print(f"        {_c('fix: ollama serve   (then: ollama pull ' + llm_cfg.model + ')', DIM)}")
        print(f"        {_c('the run will fall back to the deterministic heuristic brain', DIM)}")

    print()
    print(
        "  Ready to run the full closed loop."
        if ok
        else "  Runnable, with the fallbacks noted above."
    )
    print(f"  {_c('ecoloop run --days 3', BOLD)}")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    llm = LLMConfig.from_env()
    if args.model:
        llm.model = args.model
    if args.provider:
        llm.provider = args.provider
    if args.llm_url:
        llm.base_url = args.llm_url
    if args.llm_timeout:
        llm.timeout_s = args.llm_timeout

    cfg = RunConfig(
        run_id=args.run_id,
        engine=args.engine,
        idf=args.idf,
        epw=args.epw,
        start_month=args.start_month,
        start_day=args.start_day,
        timesteps_per_hour=args.timesteps_per_hour,
        decision_interval_min=args.decision_interval,
        agent_mode=args.agent_mode,
        pace_s=args.pace,
        brain=args.brain,
        llm=llm,
        ecm_pass=args.ecm_pass,
    )
    # --days is friendlier than an end date and cannot express an empty window.
    end_month, end_day = _advance(args.start_month, args.start_day, max(1, args.days) - 1)
    cfg.end_month, cfg.end_day = end_month, end_day
    return cfg


_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _advance(month: int, day: int, days: int) -> tuple[int, int]:
    for _ in range(days):
        day += 1
        if day > _DAYS_IN_MONTH[month - 1]:
            day = 1
            month = month % 12 + 1
    return month, day


def cmd_run(args: argparse.Namespace) -> int:
    from .bus import DECISION, EventBus, STATUS
    from .orchestrator import Orchestrator

    cfg = _config_from_args(args)
    bus = EventBus(cfg.out_dir / "events.jsonl")

    if not args.quiet:
        def on_event(event: dict[str, Any]) -> None:
            if event["kind"] == STATUS and event.get("phase") in ("run_start", "start", "done", "run_done"):
                phase = event.get("phase")
                if phase == "run_start":
                    print(_c(f"run {event.get('run_id')}", BOLD))
                    print(f"  engine   {event.get('engine')}  ({event.get('energyplus')})")
                    print(f"  brain    {event.get('brain')}  {event.get('model')}")
                    print(f"  weather  {Path(str(event.get('weather'))).name}")
                    print(f"  window   {event.get('window')}")
                    print()
                elif phase == "start":
                    print(f"  {_c('>', DIM)} simulating {event.get('label')} ...")
                elif phase == "done":
                    mark = _c("ok", GREEN) if event.get("ok") else _c("FAILED", RED)
                    print(
                        f"    {mark} {event.get('steps')} timesteps in "
                        f"{event.get('wall_seconds')}s"
                        + (f"  {_c(str(event.get('error')), RED)}" if event.get("error") else "")
                    )
            elif event["kind"] == DECISION and args.verbose:
                print(
                    f"      D{event.get('decision_id'):>3} {event.get('clock')} "
                    f"{event.get('source'):9s} {float(event.get('latency_ms') or 0):6.0f}ms "
                    f"{str(event.get('rationale'))[:70]}"
                )

        bus.subscribe(on_event)

    results = Orchestrator(cfg, bus).run()
    bus.close()
    print()
    print_report(results)
    print()
    print(f"  artifacts  {_c(str(cfg.out_dir), BOLD)}")
    print(f"  dashboard  ecoloop serve  {_c('->  http://127.0.0.1:8765', DIM)}")
    return 0 if results["baseline"]["ok"] and results["ai"]["ok"] else 1


def print_report(results: dict[str, Any]) -> None:
    savings = results["savings"]
    comfort = savings["comfort"]
    print(_c("  SAVINGS — AI closed loop vs rule-based baseline", BOLD))
    print(f"  {'metric':<24}{'baseline':>12}{'AI':>12}{'saved':>12}{'':>3}{'reduction':>10}")
    rows = (
        ("total electricity kWh", savings["total_kwh"]),
        ("HVAC electricity kWh", savings["hvac_kwh"]),
        ("cost INR", savings["cost_inr"]),
        ("carbon kgCO2", savings["carbon_kg"]),
    )
    for label, block in rows:
        base = block["baseline"]
        ai = block["ai"]
        pct = block["pct"]
        colour = GREEN if pct > 0 else RED
        print(
            f"  {label:<24}{base:>12,.2f}{ai:>12,.2f}{base - ai:>12,.2f}   "
            + _c(f"{pct:>9.2f}%", colour)
        )
    peak = savings["peak_demand_w"]
    print(
        f"  {'peak demand W':<24}{peak['baseline']:>12,.0f}{peak['ai']:>12,.0f}"
        f"{peak['baseline'] - peak['ai']:>12,.0f}   "
        + _c(f"{peak['pct']:>9.2f}%", GREEN if peak["pct"] > 0 else RED)
    )
    window = savings["peak_window_kwh"]
    print(
        f"  {'peak-window kWh':<24}{window['baseline']:>12,.2f}{window['ai']:>12,.2f}"
        f"{window['baseline'] - window['ai']:>12,.2f}   "
        + _c(f"{window['pct']:>9.2f}%", GREEN if window["pct"] > 0 else RED)
    )
    print()
    verdict = (
        _c("comfort preserved", GREEN)
        if comfort["comfort_preserved"]
        else _c("COMFORT DEGRADED", RED)
    )
    print(f"  {_c('COMFORT', BOLD)}   {verdict}")
    print(
        f"    PMV exceedance zone-hours   baseline {comfort['baseline_pmv_exceedance_zone_hours']:>6.2f}"
        f"   AI {comfort['ai_pmv_exceedance_zone_hours']:>6.2f}"
    )
    print(
        f"    CO2 exceedance zone-hours   baseline {comfort['baseline_co2_exceedance_zone_hours']:>6.2f}"
        f"   AI {comfort['ai_co2_exceedance_zone_hours']:>6.2f}"
    )
    print(
        f"    mean |PMV| when occupied    baseline {comfort['baseline_mean_abs_pmv']:>6.3f}"
        f"   AI {comfort['ai_mean_abs_pmv']:>6.3f}"
    )
    agent = results.get("agent") or {}
    if agent.get("decisions"):
        print()
        print(f"  {_c('AGENT', BOLD)}     {agent.get('model', agent.get('brain', '?'))}")
        external = agent.get("external_decisions", 0)
        print(
            f"    decisions {agent['decisions']}  "
            f"by model {agent.get('llm_decisions', 0)}  "
            f"fallback {agent.get('fallback_decisions', 0)}  "
            + (f"external {external}  " if external else "")
            + f"tool calls {agent.get('tool_calls', 0)}"
        )
        print(
            f"    latency mean {agent.get('mean_latency_ms', 0):.0f} ms  "
            f"p95 {agent.get('p95_latency_ms', 0):.0f} ms  "
            f"guardrail interventions {agent.get('guardrail_interventions', 0)}"
        )
        if agent.get("deviated_from_default") is not None:
            total = agent.get("accepted_default", 0) + agent.get("deviated_from_default", 0)
            if total:
                pct = 100.0 * agent["deviated_from_default"] / total
                print(
                    f"    diverged from the deterministic recommendation on "
                    f"{agent['deviated_from_default']}/{total} decisions ({pct:.0f}%)"
                )
    ecm = results.get("ecm")
    if ecm:
        print()
        print(f"  {_c('RETROFIT (phase B)', BOLD)}")
        print(
            f"    attempts {ecm['attempt_count']}  succeeded {ecm['successful_attempts']}  "
            f"self-corrections {ecm['self_corrections']}"
        )
        best = ecm.get("best")
        if best:
            print(
                f"    best: {', '.join(best['applied'])[:110]}\n"
                f"      -> {best['savings']['total_kwh']['pct']:.2f}% total, "
                f"{best['savings']['hvac_kwh']['pct']:.2f}% HVAC vs the same control on the base model"
            )


def cmd_report(args: argparse.Namespace) -> int:
    path = ARTIFACTS_DIR / args.run_id / "results.json"
    if not path.exists():
        print(f"no results for run {args.run_id!r} (looked in {path})", file=sys.stderr)
        return 1
    results = json.loads(path.read_text())
    print()
    print_report(results)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server.app import serve

    return serve(host=args.host, port=args.port, reload=False)


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp.server import resolve_run_dir, serve

    return serve(run_dir=resolve_run_dir(args.run_id))


def cmd_tools(args: argparse.Namespace) -> int:
    from .agent.context import FileContext
    from .agent.tools import build_registry

    registry = build_registry(FileContext(ARTIFACTS_DIR / "none"))
    rows = registry.docs_table()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{len(rows)} tools — shared by the in-process agent and the MCP server")
    print()
    for row in rows:
        flag = "write" if row["mutating"] else "read"
        print(f"  {_c(row['name'], BOLD)}  [{row['scope']}/{flag}]")
        print(f"    args: {', '.join(row['args']) or '(none)'}")
        print(f"    {row['description'][:150]}")
        print()
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from .orchestrator import list_runs

    runs = list_runs()
    if not runs:
        print("no runs yet — try: ecoloop run")
        return 0
    print(f"{'run_id':<28}{'engine':<12}{'total':>8}{'hvac':>8}  comfort")
    for run in runs:
        if not run.get("complete"):
            print(f"{run['run_id']:<28}{'(incomplete)':<12}")
            continue
        comfort = "preserved" if run.get("comfort_preserved") else "degraded"
        print(
            f"{run['run_id']:<28}{str(run.get('engine')):<12}"
            f"{run.get('total_saving_pct', 0):>7.2f}%{run.get('hvac_saving_pct', 0):>7.2f}%  {comfort}"
        )
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecoloop",
        description="Autonomous closed-loop building control: EnergyPlus + an open-source LLM.",
    )
    parser.add_argument("--version", action="version", version=f"ecoloop {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the environment")
    doctor.add_argument("--idf", default=str(RunConfig().idf))
    doctor.add_argument("--epw", default="")
    doctor.add_argument("--model", default="")
    doctor.set_defaults(func=cmd_doctor)

    run = sub.add_parser("run", help="run the closed-loop experiment")
    run.add_argument("--run-id", default="latest", help="artifacts/<run-id>/ (default: latest)")
    run.add_argument("--engine", default="auto", choices=["auto", "energyplus", "surrogate"])
    run.add_argument("--idf", default=str(RunConfig().idf))
    run.add_argument("--epw", default="")
    run.add_argument("--days", type=int, default=3, help="simulated days (default 3)")
    run.add_argument("--start-month", type=int, default=5)
    run.add_argument("--start-day", type=int, default=12)
    run.add_argument("--timesteps-per-hour", type=int, default=4)
    run.add_argument("--decision-interval", type=int, default=30, help="minutes between decisions")
    run.add_argument("--agent-mode", default="sync", choices=["sync", "async"])
    run.add_argument("--pace", type=float, default=0.0, help="seconds of wall clock per timestep")
    run.add_argument("--brain", default="llm", choices=["llm", "heuristic", "baseline"])
    run.add_argument("--provider", default="", choices=["", "ollama", "openai_compat", "mock"])
    run.add_argument("--model", default="", help="e.g. qwen2.5:3b, llama3.2:3b, mistral")
    run.add_argument("--llm-url", default="")
    run.add_argument("--llm-timeout", type=float, default=0.0)
    run.add_argument("--ecm-pass", action="store_true", help="also run phase B (retrofit measures)")
    run.add_argument("--verbose", action="store_true", help="print every decision")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="print a finished run's savings table")
    report.add_argument("run_id")
    report.set_defaults(func=cmd_report)

    runs = sub.add_parser("runs", help="list runs")
    runs.set_defaults(func=cmd_runs)

    serve_cmd = sub.add_parser("serve", help="live dashboard")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="MCP server on stdio")
    mcp.add_argument("--run-id", default="")
    mcp.set_defaults(func=cmd_mcp)

    tools = sub.add_parser("tools", help="show the tool registry")
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(func=cmd_tools)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
