"""Typed configuration for a closed-loop run.

Every tunable in the system lives here so that a run is fully described by one
serialisable object. ``RunConfig.to_dict()`` is written into every run's
``manifest.json``, which makes results reproducible.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

EngineName = Literal["energyplus", "surrogate", "auto"]
AgentMode = Literal["async", "sync"]
Brain = Literal["llm", "heuristic", "baseline"]


@dataclass
class ComfortTargets:
    """Hard comfort envelope. The guardrail layer enforces these regardless of
    what the LLM proposes, which is what lets us be aggressive on energy without
    ever trading away occupant comfort."""

    # Occupied hours: tight band.
    occupied_temp_min_c: float = 22.5
    occupied_temp_max_c: float = 26.5
    # Unoccupied hours: wide setback band.
    unoccupied_temp_min_c: float = 16.0
    unoccupied_temp_max_c: float = 30.0
    # Default |PMV| envelope (ISO 7730) for occupied hours. Per-zone limits in
    # ``sim.base.DEFAULT_ZONES`` override this, because the achievable PMV
    # depends on the activity level of the space: 0.7 for the seated office,
    # 1.5 for the 1.7-met production hall, 1.1 for packing.
    pmv_limit: float = 0.7
    # Indoor air quality ceiling (ppm). The classic ASHRAE 62.1 guideline is
    # 700 ppm above ambient; at 420 ppm outdoor that is 1120 ppm, so 1100 is a
    # slightly conservative ceiling.
    co2_limit_ppm: float = 1100.0
    rh_max_pct: float = 65.0

    # Set-point authority handed to the agent, while the space is OCCUPIED.
    # The upper cooling bound equals occupied_temp_max_c: the agent can never
    # ask for a set-point that would itself violate the comfort envelope.
    cooling_setpoint_min_c: float = 23.0
    cooling_setpoint_max_c: float = 26.5
    heating_setpoint_min_c: float = 19.0
    heating_setpoint_max_c: float = 22.0
    # Set-point authority while UNOCCUPIED — a wide setback band, which is where
    # most of the honest savings come from.
    unoccupied_cooling_min_c: float = 26.0
    unoccupied_cooling_max_c: float = 30.0
    unoccupied_heating_min_c: float = 15.0
    unoccupied_heating_max_c: float = 18.0

    min_deadband_c: float = 2.0
    # Maximum set-point movement per decision — prevents oscillation and
    # protects the plant from short-cycling.
    max_step_c: float = 1.5
    # Ventilation authority (fraction of design outdoor air).
    oa_fraction_min: float = 0.35
    oa_fraction_max: float = 1.0

    # Absolute floor on the cooling set-point. Lower than the recommended
    # authority band on purpose: going below 23 C while occupied wastes energy
    # but is not *unsafe*, so it is the optimiser's business, not the safety
    # layer's. The safety layer only stops physically absurd values.
    absolute_cooling_min_c: float = 21.0

    def cooling_bounds(self, occupied: bool) -> tuple[float, float]:
        """The *recommended authority* handed to the agent, advertised in the
        prompt and used by the heuristic policy."""
        if occupied:
            return self.cooling_setpoint_min_c, self.cooling_setpoint_max_c
        return self.unoccupied_cooling_min_c, self.unoccupied_cooling_max_c

    def heating_bounds(self, occupied: bool) -> tuple[float, float]:
        if occupied:
            return self.heating_setpoint_min_c, self.heating_setpoint_max_c
        return self.unoccupied_heating_min_c, self.unoccupied_heating_max_c

    def safety_cooling_bounds(self, occupied: bool) -> tuple[float, float]:
        """The *hard* envelope the guardrail enforces on every brain, including
        the rule-based baseline. Deliberately wider than the recommended
        authority: the guardrail must not quietly optimise the control group's
        energy use, or the measured savings would be understated.

        The upper bound is the comfort envelope itself, so no action reaching an
        actuator can put an occupied zone above its comfort ceiling.
        """
        return (
            self.absolute_cooling_min_c,
            self.occupied_temp_max_c if occupied else self.unoccupied_temp_max_c,
        )

    def safety_heating_bounds(self, occupied: bool) -> tuple[float, float]:
        return self.unoccupied_heating_min_c, self.heating_setpoint_max_c

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class GridTargets:
    """Grid-side objectives the agent reasons against."""

    # Facility peak electrical demand the agent should stay under (W). Set just
    # below the measured baseline peak (~18 kW) so the constraint actually
    # binds — a limit nothing ever approaches teaches the agent nothing.
    peak_demand_limit_w: float = 15000.0
    # Local peak-tariff window (hour of day, inclusive-exclusive).
    peak_window: tuple[int, int] = (18, 22)
    # Off-peak / mid / peak tariff (INR per kWh) — CEA-style commercial ToD slabs.
    tariff_offpeak: float = 6.5
    tariff_mid: float = 8.0
    tariff_peak: float = 11.5
    # Grid carbon intensity profile, gCO2/kWh by hour (India-average shape:
    # solar-rich midday trough, coal-heavy evening ramp).
    carbon_profile: tuple[float, ...] = (
        710, 705, 700, 700, 690, 660, 600, 520, 440, 380, 340, 320,
        315, 320, 350, 420, 520, 640, 760, 790, 780, 760, 740, 725,
    )

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["peak_window"] = list(self.peak_window)
        d["carbon_profile"] = list(self.carbon_profile)
        return d


@dataclass
class LLMConfig:
    provider: Literal["ollama", "openai_compat", "mock"] = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    api_key: str = ""
    temperature: float = 0.1
    # A decision is one tool call plus a sentence. A 700-token budget just
    # invites a small model to ramble before acting, and completion tokens
    # dominate decision latency — 300 measured ~40% faster with no loss.
    max_tokens: int = 300
    # Hard wall-clock budget for one decision (seconds). On timeout the run
    # holds the last known-good action and the heuristic policy takes the step.
    timeout_s: float = 120.0
    # Maximum tool-call round trips inside one decision.
    max_tool_rounds: int = 4
    # Identical-context cache: skip the model when the situation has not moved.
    enable_cache: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("api_key", None)
        return d

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.environ.get("ECOLOOP_LLM_PROVIDER", "ollama"),  # type: ignore[arg-type]
            base_url=os.environ.get("ECOLOOP_LLM_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("ECOLOOP_LLM_MODEL", "qwen2.5:3b"),
            api_key=os.environ.get("ECOLOOP_LLM_KEY", ""),
        )


@dataclass
class RunConfig:
    """Everything needed to reproduce one baseline+AI comparison."""

    run_id: str = "run"
    engine: EngineName = "auto"
    idf: str = str(MODELS_DIR / "baseline.idf")
    epw: str = ""  # resolved at run time (see weather.resolve_epw)
    # Simulation window (month/day). Kept short by default so the whole demo
    # completes in a couple of minutes.
    start_month: int = 5
    start_day: int = 12
    end_month: int = 5
    end_day: int = 14
    timesteps_per_hour: int = 4

    # Closed-loop cadence.
    decision_interval_min: int = 15
    agent_mode: AgentMode = "async"
    # Artificial pacing (seconds of wall clock per simulation timestep). Keeps
    # the dashboard legible during a live demo; 0 = run flat out.
    pace_s: float = 0.0

    brain: Brain = "llm"
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    comfort: ComfortTargets = field(default_factory=ComfortTargets)
    grid: GridTargets = field(default_factory=GridTargets)

    # Phase B: let the agent write and verify IDF-level ECM variants.
    ecm_pass: bool = False
    seed: int = 7

    @property
    def out_dir(self) -> Path:
        return ARTIFACTS_DIR / self.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "idf": self.idf,
            "epw": self.epw,
            "window": {
                "start": [self.start_month, self.start_day],
                "end": [self.end_month, self.end_day],
                "timesteps_per_hour": self.timesteps_per_hour,
            },
            "decision_interval_min": self.decision_interval_min,
            "agent_mode": self.agent_mode,
            "pace_s": self.pace_s,
            "brain": self.brain,
            "llm": self.llm.to_dict(),
            "comfort": self.comfort.to_dict(),
            "grid": self.grid.to_dict(),
            "ecm_pass": self.ecm_pass,
            "seed": self.seed,
            "version": __import__("ecoloop").__version__,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
