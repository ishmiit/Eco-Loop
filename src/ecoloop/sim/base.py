"""The engine/controller contract.

An engine drives simulated time and, at every timestep, does exactly two
things: hand a :class:`~ecoloop.telemetry.Snapshot` to the policy, and apply the
:class:`~ecoloop.telemetry.ControlAction` the policy hands back.

The policy's ``decide`` **must not block** for longer than a timestep — that is
the whole reason the LLM runs on its own thread with a held last-known-good
action (see ``ecoloop.agent.controller``). An engine that respects this
contract cannot be stalled by a slow model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..metrics import KPI
from ..telemetry import ControlAction, Snapshot


@runtime_checkable
class ControlPolicy(Protocol):
    def decide(self, snap: Snapshot) -> ControlAction:
        """Return the action to apply *now*. Must return promptly."""
        ...

    def close(self) -> None:  # pragma: no cover - trivial
        ...


@dataclass
class EngineResult:
    label: str
    engine: str
    kpi: KPI
    snapshots: list[Snapshot] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    idf_used: str = ""
    epw_used: str = ""
    out_dir: str = ""
    wall_seconds: float = 0.0
    severe_count: int = 0
    warning_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "engine": self.engine,
            "ok": self.ok,
            "error": self.error,
            "kpi": self.kpi.to_dict(),
            "idf_used": self.idf_used,
            "epw_used": self.epw_used,
            "out_dir": self.out_dir,
            "wall_seconds": round(self.wall_seconds, 2),
            "severe_count": self.severe_count,
            "warning_count": self.warning_count,
            "steps": len(self.snapshots),
        }


class SimulationEngine(Protocol):
    name: str

    def run(self, policy: ControlPolicy, label: str) -> EngineResult:
        ...


# Zone metadata shared by both engines so telemetry is identical regardless of
# which one produced it. Mirrors models/baseline.idf.
@dataclass(frozen=True)
class ZoneSpec:
    name: str
    area_m2: float
    volume_m3: float
    cooling_sp_schedule: str
    heating_sp_schedule: str
    oa_schedule: str
    people_object: str
    design_oa_m3_s: float          # design ventilation at OAF = 1.0
    design_flow_m3_s: float        # design supply airflow, for the fan model
    lights_w_m2: float
    equip_w_m2: float
    met: float                     # metabolic rate for PMV
    air_velocity_m_s: float        # matches the AIR_VELO_* schedules in the IDF
    pmv_limit: float               # |PMV| envelope for this activity level
    priority: str                  # human-readable comfort priority


DEFAULT_ZONES: tuple[ZoneSpec, ...] = (
    ZoneSpec(
        name="OFFICE",
        area_m2=80.0,
        volume_m3=288.0,
        cooling_sp_schedule="CSP_OFFICE",
        heating_sp_schedule="HSP_OFFICE",
        oa_schedule="OAF_OFFICE",
        people_object="OFFICE_PEOPLE",
        design_oa_m3_s=6 * 0.0025 + 80 * 0.0003,   # 0.039 m3/s
        design_flow_m3_s=0.55,
        lights_w_m2=9.0,
        equip_w_m2=11.5,
        met=1.05,
        air_velocity_m_s=0.40,
        pmv_limit=0.8,
        priority="seated office work — tightest comfort band",
    ),
    ZoneSpec(
        name="PROD_HALL",
        area_m2=80.0,
        volume_m3=288.0,
        cooling_sp_schedule="CSP_PROD",
        heating_sp_schedule="HSP_PROD",
        oa_schedule="OAF_PROD",
        people_object="PROD_PEOPLE",
        design_oa_m3_s=12 * 0.0045 + 80 * 0.0010,  # 0.134 m3/s
        design_flow_m3_s=0.95,
        lights_w_m2=12.0,
        equip_w_m2=29.0,
        met=1.7,
        air_velocity_m_s=0.80,
        pmv_limit=1.5,
        priority="standing light work, high internal gain — ventilation critical (FSSAI hygiene)",
    ),
    ZoneSpec(
        name="PACK_STORE",
        area_m2=80.0,
        volume_m3=288.0,
        cooling_sp_schedule="CSP_STORE",
        heating_sp_schedule="HSP_STORE",
        oa_schedule="OAF_STORE",
        people_object="STORE_PEOPLE",
        design_oa_m3_s=4 * 0.0025 + 80 * 0.0004,   # 0.042 m3/s
        design_flow_m3_s=0.45,
        lights_w_m2=7.0,
        equip_w_m2=6.5,
        met=1.4,
        air_velocity_m_s=0.40,
        pmv_limit=1.1,
        priority="packing and dry storage — widest tolerable band",
    ),
)


def zone_by_name(name: str) -> ZoneSpec | None:
    upper = name.strip().upper()
    for z in DEFAULT_ZONES:
        if z.name == upper:
            return z
    return None


def ensure_out_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
