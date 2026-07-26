"""Telemetry and control data model — the contract between the simulation
engine, the agent and the dashboard.

Both engines (EnergyPlus and the surrogate) emit ``Snapshot`` objects and accept
``ControlAction`` objects. Nothing else crosses that boundary, which is what
lets the identical agent drive either engine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ZoneState:
    name: str
    temp_c: float = 0.0
    rh_pct: float = 0.0
    co2_ppm: float = 420.0
    occupants: float = 0.0
    pmv: float = 0.0
    cooling_setpoint_c: float = 24.0
    heating_setpoint_c: float = 21.0
    cooling_rate_w: float = 0.0
    heating_rate_w: float = 0.0
    lights_w: float = 0.0
    equip_w: float = 0.0
    area_m2: float = 0.0
    # Per-zone PMV envelope. A 1.7-met production hall physically cannot reach
    # |PMV| <= 0.7 at any sane set-point, so a single building-wide limit would
    # make the comfort metric meaningless. Category limits follow EN 16798-1
    # practice for the activity level of each space.
    pmv_limit: float = 0.7
    # Occupant parameters, carried so the guardrail can linearise PMV about the
    # current operating point instead of assuming a fixed sensitivity.
    met: float = 1.2
    clo: float = 0.5
    air_velocity_m_s: float = 0.15
    # Occupancy foresight from the known shift pattern (None = beyond horizon).
    # This is what makes optimum start possible.
    minutes_until_occupied: float | None = None
    minutes_until_vacant: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ZoneState":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})


@dataclass
class GridSignal:
    carbon_g_per_kwh: float = 700.0
    tariff_inr_per_kwh: float = 8.0
    peak_window: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Snapshot:
    """One simulation timestep of building state."""

    step: int = 0
    sim_seconds: float = 0.0
    month: int = 1
    day: int = 1
    hour: int = 0
    minute: int = 0
    weekday: int = 1          # ISO: 1 = Monday ... 7 = Sunday
    timestep_hours: float = 0.25

    outdoor_temp_c: float = 0.0
    outdoor_rh_pct: float = 0.0
    solar_w_m2: float = 0.0
    wind_speed_m_s: float = 0.0

    zones: list[ZoneState] = field(default_factory=list)
    grid: GridSignal = field(default_factory=GridSignal)

    # Electrical power decomposition (W).
    hvac_cooling_elec_w: float = 0.0
    hvac_heating_elec_w: float = 0.0
    fan_elec_w: float = 0.0
    lights_elec_w: float = 0.0
    equip_elec_w: float = 0.0

    # Running totals for the run so far.
    cum_kwh: float = 0.0
    cum_hvac_kwh: float = 0.0
    cum_cost_inr: float = 0.0
    cum_carbon_kg: float = 0.0
    peak_demand_w: float = 0.0

    # Comfort accounting.
    occupied: bool = False
    comfort_violation: bool = False
    pmv_worst: float = 0.0
    co2_worst_ppm: float = 420.0

    # Which brain produced the action currently in force, and its age.
    control_source: str = "baseline"
    control_age_s: float = 0.0
    decision_id: int = 0
    # What the safety layer changed on this timestep (empty when it did nothing).
    guardrail_notes: list[str] = field(default_factory=list)

    @property
    def total_elec_w(self) -> float:
        return (
            self.hvac_cooling_elec_w
            + self.hvac_heating_elec_w
            + self.fan_elec_w
            + self.lights_elec_w
            + self.equip_elec_w
        )

    @property
    def clock(self) -> str:
        return f"{self.month:02d}-{self.day:02d} {self.hour:02d}:{self.minute:02d}"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["total_elec_w"] = self.total_elec_w
        d["clock"] = self.clock
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Snapshot":
        d = dict(d)
        d.pop("total_elec_w", None)
        d.pop("clock", None)
        zones = [ZoneState.from_dict(z) for z in d.pop("zones", [])]
        grid = GridSignal(**d.pop("grid", {}))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(zones=zones, grid=grid, **{k: v for k, v in d.items() if k in known})

    # --- compact serialisation for LLM prompts (token discipline) -----------
    def compact(self) -> dict[str, Any]:
        """A ~120-token view of the building. Full snapshots are ~40x larger;
        sending them every decision is the fastest way to blow latency."""
        return {
            "clock": self.clock,
            "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                max(1, min(7, self.weekday)) - 1
            ],
            "outdoor_c": round(self.outdoor_temp_c, 1),
            "solar_w_m2": round(self.solar_w_m2),
            "occupied": self.occupied,
            "zones": [
                {
                    "z": z.name,
                    "t": round(z.temp_c, 1),
                    "rh": round(z.rh_pct),
                    "co2": round(z.co2_ppm),
                    "occ": round(z.occupants, 1),
                    "pmv": round(z.pmv, 2),
                    "pmv_limit": z.pmv_limit,
                    "csp": round(z.cooling_setpoint_c, 1),
                    "hsp": round(z.heating_setpoint_c, 1),
                    "mins_to_occupied": z.minutes_until_occupied,
                }
                for z in self.zones
            ],
            "power_w": {
                "cool": round(self.hvac_cooling_elec_w),
                "heat": round(self.hvac_heating_elec_w),
                "fan": round(self.fan_elec_w),
                "plug_light": round(self.lights_elec_w + self.equip_elec_w),
                "total": round(self.total_elec_w),
            },
            "grid": {
                "carbon_g_kwh": round(self.grid.carbon_g_per_kwh),
                "tariff": self.grid.tariff_inr_per_kwh,
                "peak_window": self.grid.peak_window,
            },
            "run_totals": {
                "kwh": round(self.cum_kwh, 2),
                "peak_w": round(self.peak_demand_w),
            },
        }


@dataclass
class ControlAction:
    """A supervisory command. ``zone_overrides`` wins over the global values."""

    cooling_setpoint_c: float = 24.0
    heating_setpoint_c: float = 21.0
    oa_fraction: float = 1.0
    zone_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    source: str = "baseline"          # llm | heuristic | baseline | guardrail
    rationale: str = ""
    decision_id: int = 0
    latency_ms: float = 0.0
    clamped: list[str] = field(default_factory=list)
    # Where this action differs from the deterministic recommendation shown to
    # the model — the measure of what the language model contributed.
    deviations: list[str] = field(default_factory=list)
    tool_calls: int = 0
    model: str = ""

    def setpoints_for(self, zone: str) -> tuple[Any, Any]:
        """The requested set-points for a zone, **uncoerced**.

        Deliberately does not cast to float: an LLM can put ``null`` or a word
        in a numeric field, and casting here would raise inside the guardrail —
        the one place that must never fail. Coercion and substitution belong to
        ``guardrails._finite``, which has a sane fallback for each field.
        """
        ov = self.zone_overrides.get(zone, {})
        return (
            ov.get("cooling_setpoint_c", self.cooling_setpoint_c),
            ov.get("heating_setpoint_c", self.heating_setpoint_c),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ControlAction":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def copy(self) -> "ControlAction":
        return ControlAction.from_dict(self.to_dict())
