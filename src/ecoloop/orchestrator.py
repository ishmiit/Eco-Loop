"""Run one complete experiment: baseline, then AI, then the savings table.

The experimental design is deliberately narrow, because the headline claim
("X% less electricity") is only worth anything if exactly one thing differs
between the two runs:

* identical IDF (the same generated file, byte for byte),
* identical weather file and run period,
* identical plant model converting thermal load to electricity,
* identical safety layer applied to both,
* **only the control policy differs.**

Everything the submission needs is written under ``artifacts/<run_id>/``:

    manifest.json          the full RunConfig — the run is reproducible from it
    results.json           KPIs, the savings table, agent statistics
    savings.csv            the headline comparison as a flat table
    telemetry_*.csv        per-timestep data for both runs (the data export)
    events.jsonl           every telemetry/decision/tool-call/log event
    decisions.jsonl        just the decisions, for reading
    idf/*.idf              the exact models used, including ECM variants
    eplus/*/               raw EnergyPlus output, including eplustbl.htm
    live/                  live mirror for the dashboard and MCP clients
    ecm_report.json        phase B attempts, self-corrections and results
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

from .agent.controller import LLMPolicy
from .agent.ecm_agent import build_evidence, run_ecm_pass
from .agent.policies import BaselinePolicy, HeuristicPolicy
from .bus import DECISION, ECM, KPI as KPI_EVENT, LOG, STATUS, EventBus
from .config import ARTIFACTS_DIR, RunConfig
from .energyplus_locate import find_energyplus
from .logs import LogDigest
from .metrics import compare
from .sim.base import EngineResult
from .sim.energyplus import EnergyPlusEngine
from .sim.surrogate import SurrogateEngine
from .telemetry import Snapshot
from .weather import resolve_epw


def select_engine(cfg: RunConfig) -> str:
    """``auto`` prefers EnergyPlus and falls back to the surrogate."""
    if cfg.engine == "surrogate":
        return "surrogate"
    install = find_energyplus()
    if cfg.engine == "energyplus":
        if install is None:
            raise RuntimeError(
                "engine=energyplus was requested but no EnergyPlus install was found. "
                "Run ./scripts/install_energyplus.sh or set ECOLOOP_ENERGYPLUS_DIR."
            )
        return "energyplus"
    return "energyplus" if install is not None else "surrogate"


def _make_engine(
    cfg: RunConfig,
    bus: EventBus,
    engine_name: str,
    digest: LogDigest,
    sink: list[Snapshot],
    idf_override: Path | None = None,
) -> Any:
    kwargs = dict(digest=digest, snapshot_sink=sink, idf_override=idf_override)
    if engine_name == "energyplus":
        return EnergyPlusEngine(cfg, bus, **kwargs)
    return SurrogateEngine(cfg, bus, **kwargs)


class Orchestrator:
    def __init__(self, cfg: RunConfig, bus: EventBus | None = None) -> None:
        self.cfg = cfg
        self.out_dir = cfg.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bus = bus or EventBus(self.out_dir / "events.jsonl")
        self.engine_name = select_engine(cfg)
        self.results: dict[str, Any] = {}

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        epw = resolve_epw(cfg.epw)
        if epw is not None:
            cfg.epw = str(epw)
        cfg.save(self.out_dir / "manifest.json")

        install = find_energyplus()
        self.bus.publish(
            STATUS,
            phase="run_start",
            run_id=cfg.run_id,
            engine=self.engine_name,
            energyplus=f"{install.version} ({install.root})" if install else "not installed",
            brain=cfg.brain,
            model=cfg.llm.model if cfg.brain == "llm" else cfg.brain,
            weather=cfg.epw,
            window=f"{cfg.start_month:02d}-{cfg.start_day:02d} to {cfg.end_month:02d}-{cfg.end_day:02d}",
        )
        started = time.time()

        # ---- run 1: the control group ------------------------------------
        baseline_digest = LogDigest()
        baseline_sink: list[Snapshot] = []
        baseline_engine = _make_engine(
            cfg, self.bus, self.engine_name, baseline_digest, baseline_sink
        )
        baseline = baseline_engine.run(BaselinePolicy(bus=self.bus), "baseline")
        self._write_telemetry("baseline", baseline)
        if not baseline.ok:
            self.bus.publish(LOG, level="severe", source="orchestrator",
                             message=f"baseline run failed: {baseline.error}")

        # ---- run 2: the agent --------------------------------------------
        ai_digest = LogDigest()
        ai_sink: list[Snapshot] = []
        ai_engine = _make_engine(cfg, self.bus, self.engine_name, ai_digest, ai_sink)
        policy, agent_summary_fn = self._make_brain(ai_digest, ai_sink)
        try:
            ai = ai_engine.run(policy, "ai")
        finally:
            close = getattr(policy, "close", None)
            if callable(close):
                close()
        self._write_telemetry("ai", ai)

        agent_summary = agent_summary_fn() if agent_summary_fn else {"brain": cfg.brain}
        # Fold the agent's own latency/decision counters into the AI KPI so the
        # savings table and the dashboard read from one object.
        ai.kpi.decisions = int(agent_summary.get("decisions", 0) or 0)
        ai.kpi.llm_decisions = int(agent_summary.get("llm_decisions", 0) or 0)
        ai.kpi.fallback_decisions = int(agent_summary.get("fallback_decisions", 0) or 0)
        ai.kpi.mean_decision_latency_ms = float(agent_summary.get("mean_latency_ms", 0.0) or 0.0)
        ai.kpi.p95_decision_latency_ms = float(agent_summary.get("p95_latency_ms", 0.0) or 0.0)

        savings = compare(baseline.kpi, ai.kpi)
        self.bus.publish(KPI_EVENT, savings=savings, baseline=baseline.kpi.to_dict(),
                         ai=ai.kpi.to_dict(), agent=agent_summary)

        # ---- phase B: retrofit measures ----------------------------------
        ecm_report: dict[str, Any] | None = None
        if cfg.ecm_pass and cfg.brain == "llm":
            ecm_report = self._run_ecm_pass(ai, baseline, savings, ai_digest)

        self.results = {
            "run_id": cfg.run_id,
            "engine": self.engine_name,
            "energyplus": f"{install.version}" if install else None,
            "weather": cfg.epw,
            "window": {
                "start": [cfg.start_month, cfg.start_day],
                "end": [cfg.end_month, cfg.end_day],
                "timesteps_per_hour": cfg.timesteps_per_hour,
            },
            "baseline": baseline.to_dict(),
            "ai": ai.to_dict(),
            "savings": savings,
            "agent": agent_summary,
            "guardrail": self._guardrail_stats(baseline, ai),
            "logs": {"baseline": baseline_digest.to_dict(8), "ai": ai_digest.to_dict(8)},
            "ecm": ecm_report,
            "wall_seconds": round(time.time() - started, 2),
            "artifacts": self._artifact_index(),
        }
        (self.out_dir / "results.json").write_text(json.dumps(self.results, indent=2, default=str))
        self._write_savings_csv(savings)
        self._write_decisions()

        self.bus.publish(
            STATUS, phase="run_done", run_id=cfg.run_id, ok=baseline.ok and ai.ok,
            total_saving_pct=savings["total_kwh"]["pct"],
            hvac_saving_pct=savings["hvac_kwh"]["pct"],
            comfort_preserved=savings["comfort"]["comfort_preserved"],
            wall_seconds=self.results["wall_seconds"],
        )
        return self.results

    # ---------------------------------------------------------------- brain
    def _make_brain(
        self, digest: LogDigest, sink: list[Snapshot]
    ) -> tuple[Any, Callable[[], dict[str, Any]] | None]:
        cfg = self.cfg
        if cfg.brain == "baseline":
            return BaselinePolicy(bus=self.bus), None
        if cfg.brain == "heuristic":
            policy = HeuristicPolicy(cfg.comfort, cfg.grid, bus=self.bus)
            return policy, lambda: {"brain": "heuristic", "decisions": len(sink)}
        policy = LLMPolicy(cfg, self.bus, digest, sink)
        policy.enable_live_bridge(self.out_dir)
        return policy, policy.summary

    # ------------------------------------------------------------ phase B
    def _run_ecm_pass(
        self,
        ai: EngineResult,
        baseline: EngineResult,
        savings: dict[str, Any],
        digest: LogDigest,
    ) -> dict[str, Any]:
        cfg = self.cfg
        evidence = build_evidence(ai, baseline, savings)
        (self.out_dir / "ecm_evidence.json").write_text(json.dumps(evidence, indent=2, default=str))

        def simulate(idf_path: Path, label: str) -> EngineResult:
            """Score a generated variant under the *same* deterministic policy as
            each other variant, so differences are the measures and not the
            agent's run-to-run variability (and so phase B costs no LLM calls)."""
            variant_digest = LogDigest()
            engine = _make_engine(
                cfg, self.bus, self.engine_name, variant_digest, [], idf_override=idf_path
            )
            return engine.run(HeuristicPolicy(cfg.comfort, cfg.grid), label)

        # Reference for ECM savings: the same deterministic policy on the
        # unmodified building. Comparing a patched model under the heuristic
        # against the LLM run would mix a control change into a fabric change.
        ref_digest = LogDigest()
        ref_engine = _make_engine(cfg, self.bus, self.engine_name, ref_digest, [])
        reference = ref_engine.run(HeuristicPolicy(cfg.comfort, cfg.grid), "ecm_reference")
        self.bus.publish(
            ECM, phase="reference", total_kwh=round(reference.kpi.total_kwh, 3), ok=reference.ok
        )
        return run_ecm_pass(
            cfg=cfg,
            bus=self.bus,
            evidence=evidence,
            digest=digest,
            simulate=simulate,
            reference=reference.kpi,
        )

    # ------------------------------------------------------------ artifacts
    def _write_telemetry(self, label: str, result: EngineResult) -> Path:
        path = self.out_dir / f"telemetry_{label}.csv"
        snaps = result.snapshots
        if not snaps:
            path.write_text("")
            return path
        zone_names = [z.name for z in snaps[0].zones]
        header = [
            "step", "clock", "weekday", "hour", "minute", "outdoor_temp_c", "outdoor_rh_pct",
            "solar_w_m2", "occupied", "control_source", "decision_id",
            "hvac_cooling_elec_w", "hvac_heating_elec_w", "fan_elec_w",
            "lights_elec_w", "equip_elec_w", "total_elec_w",
            "cum_kwh", "cum_hvac_kwh", "cum_cost_inr", "cum_carbon_kg", "peak_demand_w",
            "grid_carbon_g_kwh", "grid_tariff_inr_kwh", "grid_peak_window",
            "comfort_violation", "guardrail_actions",
        ]
        for name in zone_names:
            header += [
                f"{name}_temp_c", f"{name}_rh_pct", f"{name}_co2_ppm", f"{name}_occupants",
                f"{name}_pmv", f"{name}_pmv_limit", f"{name}_cooling_sp_c",
                f"{name}_heating_sp_c", f"{name}_cooling_w",
            ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for s in snaps:
                row = [
                    s.step, s.clock, s.weekday, s.hour, s.minute,
                    round(s.outdoor_temp_c, 2), round(s.outdoor_rh_pct, 1), round(s.solar_w_m2, 1),
                    int(s.occupied), s.control_source, s.decision_id,
                    round(s.hvac_cooling_elec_w, 1), round(s.hvac_heating_elec_w, 1),
                    round(s.fan_elec_w, 1), round(s.lights_elec_w, 1), round(s.equip_elec_w, 1),
                    round(s.total_elec_w, 1), round(s.cum_kwh, 4), round(s.cum_hvac_kwh, 4),
                    round(s.cum_cost_inr, 3), round(s.cum_carbon_kg, 4), round(s.peak_demand_w, 1),
                    round(s.grid.carbon_g_per_kwh, 1), s.grid.tariff_inr_per_kwh,
                    int(s.grid.peak_window), int(s.comfort_violation), len(s.guardrail_notes),
                ]
                for z in s.zones:
                    row += [
                        round(z.temp_c, 2), round(z.rh_pct, 1), round(z.co2_ppm, 1),
                        round(z.occupants, 2), round(z.pmv, 3), z.pmv_limit,
                        round(z.cooling_setpoint_c, 2), round(z.heating_setpoint_c, 2),
                        round(z.cooling_rate_w, 1),
                    ]
                writer.writerow(row)
        return path

    def _write_savings_csv(self, savings: dict[str, Any]) -> Path:
        path = self.out_dir / "savings.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "baseline", "ai_closed_loop", "delta", "pct_reduction"])
            for key in ("total_kwh", "hvac_kwh", "cost_inr", "carbon_kg"):
                block = savings[key]
                base = block["baseline"]
                ai = block["ai"]
                writer.writerow([key, base, ai, round(base - ai, 3), block["pct"]])
            peak = savings["peak_demand_w"]
            writer.writerow(
                ["peak_demand_w", peak["baseline"], peak["ai"],
                 round(peak["baseline"] - peak["ai"], 1), peak["pct"]]
            )
            window = savings["peak_window_kwh"]
            writer.writerow(
                ["peak_window_kwh", window["baseline"], window["ai"],
                 round(window["baseline"] - window["ai"], 3), window["pct"]]
            )
            comfort = savings["comfort"]
            for key in (
                "pmv_exceedance_zone_hours",
                "co2_exceedance_zone_hours",
                "temp_exceedance_zone_hours",
            ):
                writer.writerow(
                    [key, comfort[f"baseline_{key}"], comfort[f"ai_{key}"],
                     round(comfort[f"baseline_{key}"] - comfort[f"ai_{key}"], 3), ""]
                )
        return path

    def _write_decisions(self) -> Path:
        path = self.out_dir / "decisions.jsonl"
        events = self.bus.history([DECISION])
        with path.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, default=str) + "\n")
        return path

    def _guardrail_stats(self, baseline: EngineResult, ai: EngineResult) -> dict[str, Any]:
        def count(result: EngineResult) -> dict[str, Any]:
            adjusted = [s for s in result.snapshots if s.guardrail_notes]
            reasons: dict[str, int] = {}
            for snap in adjusted:
                for note in snap.guardrail_notes:
                    kind = note.split(":", 1)[1].strip().split(" ")[0] if ":" in note else note
                    reasons[kind] = reasons.get(kind, 0) + 1
            return {
                "timesteps": len(result.snapshots),
                "timesteps_adjusted": len(adjusted),
                "top_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:6]),
            }

        return {"baseline": count(baseline), "ai": count(ai)}

    def _artifact_index(self) -> dict[str, Any]:
        index: dict[str, Any] = {}
        for path in sorted(self.out_dir.rglob("*")):
            if path.is_file() and path.stat().st_size > 0:
                index[str(path.relative_to(self.out_dir))] = path.stat().st_size
        return index


def run_experiment(cfg: RunConfig, bus: EventBus | None = None) -> dict[str, Any]:
    return Orchestrator(cfg, bus).run()


def list_runs() -> list[dict[str, Any]]:
    """Every run in ``artifacts/``, newest first, for the dashboard's picker."""
    out: list[dict[str, Any]] = []
    if not ARTIFACTS_DIR.is_dir():
        return out
    for directory in sorted(ARTIFACTS_DIR.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        results = directory / "results.json"
        entry: dict[str, Any] = {"run_id": directory.name, "complete": results.exists()}
        try:
            entry["modified"] = directory.stat().st_mtime
        except OSError:
            entry["modified"] = 0.0
        if results.exists():
            try:
                data = json.loads(results.read_text())
                entry.update(
                    {
                        "engine": data.get("engine"),
                        "total_saving_pct": data.get("savings", {}).get("total_kwh", {}).get("pct"),
                        "hvac_saving_pct": data.get("savings", {}).get("hvac_kwh", {}).get("pct"),
                        "comfort_preserved": data.get("savings", {})
                        .get("comfort", {})
                        .get("comfort_preserved"),
                        "model": data.get("agent", {}).get("model"),
                    }
                )
            except (OSError, json.JSONDecodeError):
                entry["complete"] = False
        out.append(entry)
    return out
