"""Phase B — the agent modifies the building model, not just its set-points.

Real-time set-point control (phase A) is bounded by what the building *is*.
Phase B lets the agent change that: it reads the evidence from the closed-loop
run it just observed, writes modified ``.idf`` files, and each one is simulated
and scored against the baseline.

This is also where **self-correction** is exercised for real rather than
claimed. When a generated model fails, the failure is not swallowed: the
EnergyPlus error is deduplicated by ``LogDigest`` and handed back to the model,
which diagnoses it and calls ``propose_ecm`` again. The loop is:

    propose -> write .idf -> simulate -> (fail) -> read error -> repair -> retry

with a hard attempt budget so a model that cannot converge still terminates.
Every generated variant is kept in ``artifacts/<run>/idf/`` whether it
succeeded or not, because the failures are the evidence that the correction
loop did something.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..bus import ECM, LOG, TOOL_CALL, EventBus
from ..config import RunConfig
from ..logs import LogDigest
from ..metrics import KPI, compare
from ..sim.base import EngineResult
from ..sim.idf import ECM_DOCS, ECM_LIBRARY, IDF, apply_ecms
from . import prompts
from .llm import LLMClient
from .tools import SCOPE_ANALYSIS, ToolRegistry, build_registry

MAX_ATTEMPTS = 3


@dataclass
class ECMAttempt:
    index: int
    measures: list[dict[str, Any]]
    rationale: str
    idf_path: str = ""
    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    kpi: KPI | None = None
    savings: dict[str, Any] = field(default_factory=dict)
    log_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "measures": self.measures,
            "rationale": self.rationale,
            "idf": self.idf_path,
            "applied": self.applied,
            "rejected": self.rejected,
            "ok": self.ok,
            "error": self.error,
            "kpi": self.kpi.to_dict() if self.kpi else None,
            "savings": self.savings,
        }


class ECMContext:
    """A :class:`~ecoloop.agent.tools.RunContext` for the retrofit phase.

    Read-only tools serve the evidence from the completed run; ``propose_ecm``
    writes and simulates a variant through the injected ``simulate`` callable.
    """

    def __init__(
        self,
        cfg: RunConfig,
        evidence: dict[str, Any],
        digest: LogDigest,
        simulate: Callable[[Path, str], EngineResult],
        reference: KPI,
        bus: EventBus,
    ) -> None:
        self.cfg = cfg
        self.evidence = evidence
        self.digest = digest
        self.simulate = simulate
        self.reference = reference
        self.bus = bus
        self.attempts: list[ECMAttempt] = []
        self._base_idf_path = Path(cfg.idf)

    # -- reads --------------------------------------------------------------
    def building_state(self) -> dict[str, Any]:
        return self.evidence

    def history(self, minutes: int, metric: str) -> dict[str, Any]:
        return {"note": "the run has finished; see the run evidence in the prompt", "metric": metric}

    def targets(self) -> dict[str, Any]:
        return {
            "objective": "reduce annualised HVAC electricity without breaching comfort",
            "comfort": self.cfg.comfort.to_dict(),
        }

    def grid_forecast(self, hours: int) -> dict[str, Any]:
        return {"note": "not applicable in the retrofit phase"}

    def read_log(self, level: str, max_chars: int) -> dict[str, Any]:
        levels = ("fatal", "severe", "warning") if level in ("all", "") else (level,)
        return {"summary": self.digest.summary(), "digest": self.digest.for_llm(max_chars, levels)}

    def search_log(self, pattern: str, limit: int) -> dict[str, Any]:
        return {"pattern": pattern, "matches": self.digest.search(pattern, limit)}

    def list_idf_objects(self, object_class: str, limit: int) -> dict[str, Any]:
        idf = IDF.load(self._base_idf_path)
        if not object_class:
            return {"file": str(self._base_idf_path), "classes": idf.classes()}
        objects = idf.of_class(object_class)
        return {
            "object_class": object_class,
            "count": len(objects),
            "objects": [{"name": o.name, "fields": o.fields[:14]} for o in objects[:limit]],
        }

    def available_ecms(self) -> dict[str, Any]:
        return {
            "ecms": [
                {"ecm": name, "description": ECM_DOCS.get(name, "")} for name in sorted(ECM_LIBRARY)
            ]
        }

    def apply_setpoints(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "error": "set-point control is not available in the retrofit phase"}

    def hold(self, rationale: str) -> dict[str, Any]:
        return {"ok": True, "note": "no action taken"}

    # -- the write path -----------------------------------------------------
    def propose_ecm(self, measures: list[dict[str, Any]], rationale: str) -> dict[str, Any]:
        index = len(self.attempts) + 1
        attempt = ECMAttempt(index=index, measures=list(measures), rationale=rationale)
        self.attempts.append(attempt)

        if not measures:
            attempt.error = "no measures supplied"
            return {
                "ok": False,
                "error": attempt.error,
                "hint": "call list_available_ecms and pass 2-4 measures",
            }

        idf = IDF.load(self._base_idf_path)
        results = apply_ecms(idf, measures)
        attempt.applied = [f"{r.ecm}: {r.detail}" for r in results if r.ok]
        attempt.rejected = [f"{r.ecm}: {r.detail}" for r in results if not r.ok]

        if not any(r.ok for r in results):
            attempt.error = "every measure was rejected"
            return {
                "ok": False,
                "error": attempt.error,
                "rejected": attempt.rejected,
                "valid_ecms": sorted(ECM_LIBRARY),
                "hint": "use the exact ECM names from list_available_ecms",
            }

        target = self.cfg.out_dir / "idf" / f"ecm_attempt_{index}.idf"
        idf.save(target)
        attempt.idf_path = str(target)

        result = self.simulate(target, f"ecm_{index}")
        attempt.ok = result.ok
        attempt.error = result.error
        if not result.ok:
            digest = LogDigest()
            digest.add_file(Path(result.out_dir) / "eplusout.err")
            attempt.log_digest = digest.for_llm(900)
            self.bus.publish(
                ECM, attempt=index, ok=False, error=result.error, idf=str(target),
                applied=attempt.applied, rejected=attempt.rejected,
            )
            return {
                "ok": False,
                "error": f"the generated model failed to simulate: {result.error}",
                "applied": attempt.applied,
                "rejected": attempt.rejected,
                "energyplus_log": attempt.log_digest,
                "hint": "diagnose the error above, then call propose_ecm again with corrected measures",
            }

        attempt.kpi = result.kpi
        attempt.savings = compare(self.reference, result.kpi)
        self.bus.publish(
            ECM,
            attempt=index,
            ok=True,
            idf=str(target),
            applied=attempt.applied,
            rejected=attempt.rejected,
            total_kwh=round(result.kpi.total_kwh, 3),
            savings_pct=attempt.savings["total_kwh"]["pct"],
            hvac_savings_pct=attempt.savings["hvac_kwh"]["pct"],
        )
        return {
            "ok": True,
            "applied": attempt.applied,
            "rejected": attempt.rejected,
            "generated_model": str(target),
            "result": {
                "total_kwh": round(result.kpi.total_kwh, 2),
                "reference_total_kwh": round(self.reference.total_kwh, 2),
                "saving_vs_reference_pct": attempt.savings["total_kwh"]["pct"],
                "hvac_saving_pct": attempt.savings["hvac_kwh"]["pct"],
                "comfort_preserved": attempt.savings["comfort"]["comfort_preserved"],
            },
        }


def run_ecm_pass(
    cfg: RunConfig,
    bus: EventBus,
    evidence: dict[str, Any],
    digest: LogDigest,
    simulate: Callable[[Path, str], EngineResult],
    reference: KPI,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Drive the retrofit conversation. Returns a serialisable report."""
    context = ECMContext(cfg, evidence, digest, simulate, reference, bus)
    registry: ToolRegistry = build_registry(context)
    tools = registry.openai_schema(SCOPE_ANALYSIS)
    names = registry.names(SCOPE_ANALYSIS)
    client = LLMClient(cfg.llm)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.ECM_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.ecm_prompt(evidence)},
    ]
    started = time.time()
    transcript: list[dict[str, Any]] = []
    # Generous per-call budget: this phase runs after the simulation, so it is
    # not competing with a control deadline.
    deadline = time.monotonic() + max(60.0, cfg.llm.timeout_s * 6)

    for round_index in range(max_attempts * 3):
        if len([a for a in context.attempts if a.ok]) >= 1 and round_index >= 2:
            break
        if len(context.attempts) >= max_attempts:
            break
        reply = client.chat(messages, tools=tools, deadline=deadline, valid_names=names)
        transcript.append({"role": "assistant", "reply": reply.to_dict()})
        if not reply.ok:
            bus.publish(LOG, level="severe", source="ecm-agent", message=f"LLM error: {reply.error}")
            break
        if not reply.tool_calls:
            if round_index == 0:
                messages.append({"role": "assistant", "content": reply.content[:500]})
                messages.append(
                    {
                        "role": "user",
                        "content": "Call list_available_ecms, then propose_ecm with 2-4 measures.",
                    }
                )
                continue
            break

        messages.append(client.assistant_turn(reply))
        failed_attempt: ECMAttempt | None = None
        for call in reply.tool_calls:
            result = registry.call(call.name, call.arguments)
            bus.publish(
                TOOL_CALL, phase="ecm", tool=call.name, arguments=call.arguments,
                ok=result.ok, error=result.error, latency_ms=round(result.latency_ms, 1),
            )
            transcript.append({"role": "tool", "name": call.name, "result": result.payload})
            messages.append({"role": "tool", "name": call.name, "content": result.as_text(1500)})
            if call.name == "propose_ecm" and context.attempts and not context.attempts[-1].ok:
                failed_attempt = context.attempts[-1]

        # The self-correction turn: show the model exactly why it failed.
        if failed_attempt is not None and len(context.attempts) < max_attempts:
            messages.append(
                {
                    "role": "user",
                    "content": prompts.repair_prompt(
                        ecm_summary=json.dumps(failed_attempt.measures),
                        log_digest=failed_attempt.log_digest
                        or failed_attempt.error
                        or "\n".join(failed_attempt.rejected),
                    ),
                }
            )

    successes = [a for a in context.attempts if a.ok]
    best = max(successes, key=lambda a: a.savings["total_kwh"]["pct"], default=None)
    report = {
        "attempts": [a.to_dict() for a in context.attempts],
        "attempt_count": len(context.attempts),
        "successful_attempts": len(successes),
        "self_corrections": sum(1 for a in context.attempts if not a.ok),
        "best": best.to_dict() if best else None,
        "wall_seconds": round(time.time() - started, 2),
        "tool_calls": len(registry.calls),
        "generated_models": [a.idf_path for a in context.attempts if a.idf_path],
    }
    report_path = cfg.out_dir / "ecm_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({**report, "transcript": transcript}, indent=2, default=str))
    return report


def build_evidence(ai: EngineResult, baseline: EngineResult, savings: dict[str, Any]) -> dict[str, Any]:
    """Condense the run into the ~2 kB of evidence the retrofit agent reasons over."""
    snaps = ai.snapshots
    if not snaps:
        return {"note": "no telemetry"}
    occupied = [s for s in snaps if s.occupied]
    zone_names = [z.name for z in snaps[0].zones]
    zone_rows = []
    for i, name in enumerate(zone_names):
        temps = [s.zones[i].temp_c for s in occupied] or [0.0]
        pmvs = [s.zones[i].pmv for s in occupied] or [0.0]
        cools = [s.zones[i].cooling_rate_w for s in snaps]
        zone_rows.append(
            {
                "zone": name,
                "mean_temp_occupied_c": round(sum(temps) / len(temps), 2),
                "max_temp_c": round(max(s.zones[i].temp_c for s in snaps), 2),
                "mean_pmv_occupied": round(sum(pmvs) / len(pmvs), 2),
                "peak_cooling_w": round(max(cools)),
                "cooling_kwh": round(
                    sum(c * s.timestep_hours for c, s in zip(cools, snaps)) / 1000.0, 1
                ),
            }
        )
    solar = [s.solar_w_m2 for s in snaps]
    return {
        "climate": "Chennai, hot-humid; TMYx weather",
        "period_days": round(len(snaps) * snaps[0].timestep_hours / 24.0, 2),
        "envelope": {
            "roof": "uninsulated 150 mm RCC + 40 mm screed, solar absorptance 0.75, U ~ 3.0 W/m2K",
            "walls": "230 mm brick + plaster, U ~ 2.0 W/m2K",
            "glazing": "single clear, U 5.8 W/m2K, SHGC 0.82",
            "roof_area_m2": 240,
            "peak_solar_w_m2": round(max(solar)),
        },
        "energy_kwh": {
            "baseline_total": round(baseline.kpi.total_kwh, 1),
            "ai_controlled_total": round(ai.kpi.total_kwh, 1),
            "ai_hvac": round(ai.kpi.hvac_kwh, 1),
            "ai_cooling": round(ai.kpi.cooling_kwh, 1),
            "ai_fans": round(ai.kpi.fan_kwh, 1),
            "ai_lights_and_plug": round(ai.kpi.plug_light_kwh, 1),
            "control_saving_pct": savings["total_kwh"]["pct"],
        },
        "peak_demand_w": round(ai.kpi.peak_demand_w),
        "comfort": {
            "mean_abs_pmv_occupied": round(ai.kpi.mean_abs_pmv_occupied, 2),
            "worst_co2_ppm": round(ai.kpi.worst_co2_ppm),
            "pmv_exceedance_zone_hours": round(ai.kpi.pmv_exceedance_zone_hours, 2),
        },
        "zones": zone_rows,
        "observations": [
            "Cooling dominates; there is no heating load in this climate.",
            "Mean radiant temperature is pulled up by the uninsulated roof slab, which is why "
            "PMV reads warm even when air temperature is at set-point — this limits how far "
            "set-points can be relaxed by control alone.",
            "Outdoor air is hot and humid, so its latent load is a large share of cooling.",
        ],
    }
