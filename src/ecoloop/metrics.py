"""Physics and KPI accounting.

Contains the three quantitative things the project is judged on:

* **PMV** — ISO 7730 / Fanger thermal comfort, so "did the AI protect comfort?"
  has a numeric answer. EnergyPlus computes its own Fanger PMV for the People
  objects; this implementation is used by the surrogate engine and as a
  cross-check on the EnergyPlus value (they agree to ~0.05 in tests).
* **CO2** — a single-zone mass balance, so ventilation cuts have a visible IAQ
  cost and the agent cannot "save energy" by suffocating the occupants.
* **Energy / cost / carbon accounting** — power integrated over the timestep,
  with an explicit plant model converting thermal load to electricity.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

from .config import ComfortTargets, GridTargets
from .telemetry import GridSignal, Snapshot

# --------------------------------------------------------------------------
# Plant model — the single place where thermal load becomes electricity.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlantModel:
    """A VRF-class plant. Values are documented in docs/ARCHITECTURE.md and are
    applied identically to the baseline and the AI run, so the reported savings
    are invariant to the choice."""

    cooling_cop: float = 3.2
    heating_cop: float = 3.0          # reverse-cycle heat pump
    fan_w_per_m3_s: float = 1000.0    # 1.0 kW per m3/s — ASHRAE 90.1 class
    supply_delta_t_k: float = 10.0    # air-side temperature rise/drop
    air_density: float = 1.2          # kg/m3
    air_cp: float = 1006.0            # J/kg.K
    min_flow_fraction: float = 0.25   # fan floor while the system is enabled

    def airflow_m3_s(self, thermal_w: float) -> float:
        if thermal_w <= 0:
            return 0.0
        mass_flow = thermal_w / (self.air_cp * self.supply_delta_t_k)
        return mass_flow / self.air_density

    def fan_power_w(self, thermal_w: float, enabled: bool, design_flow_m3_s: float) -> float:
        if not enabled:
            return 0.0
        flow = self.airflow_m3_s(thermal_w)
        floor = self.min_flow_fraction * design_flow_m3_s
        flow = max(flow, floor)
        flow = min(flow, design_flow_m3_s)
        # Cube law would be more accurate for a VAV fan; linear is conservative
        # (it under-credits the AI, which reduces airflow).
        return self.fan_w_per_m3_s * flow

    def electric_w(self, cooling_w: float, heating_w: float) -> tuple[float, float]:
        return (
            max(0.0, cooling_w) / self.cooling_cop,
            max(0.0, heating_w) / self.heating_cop,
        )


DEFAULT_PLANT = PlantModel()


# --------------------------------------------------------------------------
# ISO 7730 Fanger PMV/PPD
# --------------------------------------------------------------------------


def fanger_pmv(
    ta: float,
    tr: float | None = None,
    vel: float = 0.12,
    rh: float = 50.0,
    met: float = 1.2,
    clo: float = 0.6,
    wme: float = 0.0,
) -> float:
    """Predicted Mean Vote — ISO 7730 Annex D reference algorithm.

    ``ta`` air temperature (C), ``tr`` mean radiant temperature (defaults to
    ``ta``), ``vel`` relative air velocity (m/s), ``rh`` relative humidity (%),
    ``met`` metabolic rate (met), ``clo`` clothing insulation (clo).

    Returns PMV clipped to the meaningful [-3, +3] scale.
    """
    if tr is None:
        tr = ta
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))  # vapour pressure, Pa

    icl = 0.155 * clo          # thermal insulation, m2K/W
    m = met * 58.15            # metabolic rate, W/m2
    w = wme * 58.15            # external work
    mw = m - w                 # internal heat production

    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(max(vel, 0.0))   # forced convection coefficient
    taa = ta + 273.0
    tra = tr + 273.0

    # Clothing surface temperature, solved by successive substitution.
    tcla = taa + (35.5 - ta) / (3.5 * (6.45 * icl + 0.1))
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4

    xn = tcla / 100.0
    xf = tcla / 50.0
    hc = hcf
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25   # natural convection
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= 1.5e-4:
            break
    tcl = 100.0 * xn - 273.0

    # Heat-loss components (W/m2).
    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)            # skin diffusion
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0          # sweating
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)                   # latent respiration
    hl4 = 0.0014 * m * (34.0 - ta)                            # dry respiration
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)         # radiation
    hl6 = fcl * hc * (tcl - ta)                               # convection

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return max(-3.0, min(3.0, pmv))


def pmv_ppd(pmv: float) -> float:
    """Predicted Percentage Dissatisfied from PMV (ISO 7730)."""
    return 100.0 - 95.0 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)


def summer_clo(month: int) -> float:
    """Seasonal clothing insulation — 0.5 clo summer, 0.9 clo winter."""
    return 0.5 if month in (3, 4, 5, 6, 7, 8, 9, 10) else 0.9


# --------------------------------------------------------------------------
# CO2 mass balance
# --------------------------------------------------------------------------

CO2_PER_PERSON_M3_S = 0.0000052 * 1.0  # 0.0052 L/s at 1.2 met


def co2_next_ppm(
    current_ppm: float,
    occupants: float,
    outdoor_air_m3_s: float,
    volume_m3: float,
    dt_s: float,
    outdoor_ppm: float = 420.0,
) -> float:
    """Well-mixed single-zone CO2 balance, integrated with an exact exponential
    step (unconditionally stable for any timestep)."""
    volume_m3 = max(volume_m3, 1.0)
    gen_ppm_s = (occupants * CO2_PER_PERSON_M3_S / volume_m3) * 1e6
    ach_s = max(outdoor_air_m3_s, 0.0) / volume_m3
    if ach_s < 1e-9:
        return current_ppm + gen_ppm_s * dt_s
    steady = outdoor_ppm + gen_ppm_s / ach_s
    decay = math.exp(-ach_s * dt_s)
    return steady + (current_ppm - steady) * decay


# --------------------------------------------------------------------------
# Grid signal
# --------------------------------------------------------------------------


def grid_signal(hour: int, targets: GridTargets) -> GridSignal:
    carbon = targets.carbon_profile[hour % 24]
    lo, hi = targets.peak_window
    in_peak = lo <= hour < hi
    if in_peak:
        tariff = targets.tariff_peak
    elif 7 <= hour < 18:
        tariff = targets.tariff_mid
    else:
        tariff = targets.tariff_offpeak
    return GridSignal(carbon_g_per_kwh=float(carbon), tariff_inr_per_kwh=tariff, peak_window=in_peak)


# --------------------------------------------------------------------------
# KPI accumulation
# --------------------------------------------------------------------------


@dataclass
class KPI:
    label: str = ""
    engine: str = ""
    steps: int = 0
    sim_hours: float = 0.0
    total_kwh: float = 0.0
    hvac_kwh: float = 0.0
    cooling_kwh: float = 0.0
    heating_kwh: float = 0.0
    fan_kwh: float = 0.0
    plug_light_kwh: float = 0.0
    cost_inr: float = 0.0
    carbon_kg: float = 0.0
    peak_demand_w: float = 0.0
    peak_window_kwh: float = 0.0

    occupied_hours: float = 0.0
    # Zone-hours outside the temperature band during occupancy.
    temp_exceedance_zone_hours: float = 0.0
    pmv_exceedance_zone_hours: float = 0.0
    co2_exceedance_zone_hours: float = 0.0
    worst_pmv: float = 0.0
    worst_co2_ppm: float = 0.0
    mean_abs_pmv_occupied: float = 0.0
    mean_zone_temp_occupied_c: float = 0.0

    decisions: int = 0
    llm_decisions: int = 0
    fallback_decisions: int = 0
    mean_decision_latency_ms: float = 0.0
    p95_decision_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class KPIAccumulator:
    comfort: ComfortTargets = field(default_factory=ComfortTargets)
    label: str = ""
    engine: str = ""

    def __post_init__(self) -> None:
        self.kpi = KPI(label=self.label, engine=self.engine)
        self._pmv_sum = 0.0
        self._pmv_n = 0
        self._temp_sum = 0.0
        self._temp_n = 0
        self._latencies: list[float] = []

    def add(self, snap: Snapshot) -> None:
        k = self.kpi
        dt_h = snap.timestep_hours
        k.steps += 1
        k.sim_hours += dt_h

        cool = snap.hvac_cooling_elec_w * dt_h / 1000.0
        heat = snap.hvac_heating_elec_w * dt_h / 1000.0
        fan = snap.fan_elec_w * dt_h / 1000.0
        plug = (snap.lights_elec_w + snap.equip_elec_w) * dt_h / 1000.0
        total = cool + heat + fan + plug

        k.cooling_kwh += cool
        k.heating_kwh += heat
        k.fan_kwh += fan
        k.plug_light_kwh += plug
        k.hvac_kwh += cool + heat + fan
        k.total_kwh += total
        k.cost_inr += total * snap.grid.tariff_inr_per_kwh
        k.carbon_kg += total * snap.grid.carbon_g_per_kwh / 1000.0
        k.peak_demand_w = max(k.peak_demand_w, snap.total_elec_w)
        if snap.grid.peak_window:
            k.peak_window_kwh += total

        if snap.occupied:
            k.occupied_hours += dt_h
            for z in snap.zones:
                if z.occupants <= 0.05:
                    continue
                self._temp_sum += z.temp_c
                self._temp_n += 1
                self._pmv_sum += abs(z.pmv)
                self._pmv_n += 1
                if not (self.comfort.occupied_temp_min_c <= z.temp_c <= self.comfort.occupied_temp_max_c):
                    k.temp_exceedance_zone_hours += dt_h
                if abs(z.pmv) > (z.pmv_limit or self.comfort.pmv_limit):
                    k.pmv_exceedance_zone_hours += dt_h
                if z.co2_ppm > self.comfort.co2_limit_ppm:
                    k.co2_exceedance_zone_hours += dt_h
                if abs(z.pmv) > abs(k.worst_pmv):
                    k.worst_pmv = z.pmv
                k.worst_co2_ppm = max(k.worst_co2_ppm, z.co2_ppm)

        k.mean_abs_pmv_occupied = self._pmv_sum / self._pmv_n if self._pmv_n else 0.0
        k.mean_zone_temp_occupied_c = self._temp_sum / self._temp_n if self._temp_n else 0.0

    def add_decision(self, source: str, latency_ms: float) -> None:
        k = self.kpi
        k.decisions += 1
        if source == "llm":
            k.llm_decisions += 1
        else:
            k.fallback_decisions += 1
        self._latencies.append(latency_ms)
        self._latencies.sort()
        k.mean_decision_latency_ms = sum(self._latencies) / len(self._latencies)
        idx = max(0, int(0.95 * len(self._latencies)) - 1)
        k.p95_decision_latency_ms = self._latencies[idx]

    def result(self) -> KPI:
        return self.kpi


def _pct(base: float, new: float) -> float:
    if base <= 1e-9:
        return 0.0
    return (base - new) / base * 100.0


def compare(baseline: KPI, ai: KPI) -> dict[str, Any]:
    """The savings table. Positive percentages mean the AI used less."""
    return {
        "total_kwh": {
            "baseline": round(baseline.total_kwh, 3),
            "ai": round(ai.total_kwh, 3),
            "saved_kwh": round(baseline.total_kwh - ai.total_kwh, 3),
            "pct": round(_pct(baseline.total_kwh, ai.total_kwh), 2),
        },
        "hvac_kwh": {
            "baseline": round(baseline.hvac_kwh, 3),
            "ai": round(ai.hvac_kwh, 3),
            "saved_kwh": round(baseline.hvac_kwh - ai.hvac_kwh, 3),
            "pct": round(_pct(baseline.hvac_kwh, ai.hvac_kwh), 2),
        },
        "cost_inr": {
            "baseline": round(baseline.cost_inr, 2),
            "ai": round(ai.cost_inr, 2),
            "saved": round(baseline.cost_inr - ai.cost_inr, 2),
            "pct": round(_pct(baseline.cost_inr, ai.cost_inr), 2),
        },
        "carbon_kg": {
            "baseline": round(baseline.carbon_kg, 3),
            "ai": round(ai.carbon_kg, 3),
            "saved": round(baseline.carbon_kg - ai.carbon_kg, 3),
            "pct": round(_pct(baseline.carbon_kg, ai.carbon_kg), 2),
        },
        "peak_demand_w": {
            "baseline": round(baseline.peak_demand_w, 1),
            "ai": round(ai.peak_demand_w, 1),
            "pct": round(_pct(baseline.peak_demand_w, ai.peak_demand_w), 2),
        },
        "peak_window_kwh": {
            "baseline": round(baseline.peak_window_kwh, 3),
            "ai": round(ai.peak_window_kwh, 3),
            "pct": round(_pct(baseline.peak_window_kwh, ai.peak_window_kwh), 2),
        },
        "comfort": {
            "baseline_temp_exceedance_zone_hours": round(baseline.temp_exceedance_zone_hours, 2),
            "ai_temp_exceedance_zone_hours": round(ai.temp_exceedance_zone_hours, 2),
            "baseline_pmv_exceedance_zone_hours": round(baseline.pmv_exceedance_zone_hours, 2),
            "ai_pmv_exceedance_zone_hours": round(ai.pmv_exceedance_zone_hours, 2),
            "baseline_co2_exceedance_zone_hours": round(baseline.co2_exceedance_zone_hours, 2),
            "ai_co2_exceedance_zone_hours": round(ai.co2_exceedance_zone_hours, 2),
            "baseline_mean_abs_pmv": round(baseline.mean_abs_pmv_occupied, 3),
            "ai_mean_abs_pmv": round(ai.mean_abs_pmv_occupied, 3),
            "baseline_worst_co2_ppm": round(baseline.worst_co2_ppm, 1),
            "ai_worst_co2_ppm": round(ai.worst_co2_ppm, 1),
            # The headline claim: comfort was not traded away.
            "comfort_preserved": (
                ai.pmv_exceedance_zone_hours <= baseline.pmv_exceedance_zone_hours + 1e-6
                and ai.co2_exceedance_zone_hours <= baseline.co2_exceedance_zone_hours + 1e-6
            ),
        },
        "agent": {
            "decisions": ai.decisions,
            "llm_decisions": ai.llm_decisions,
            "fallback_decisions": ai.fallback_decisions,
            "mean_latency_ms": round(ai.mean_decision_latency_ms, 1),
            "p95_latency_ms": round(ai.p95_decision_latency_ms, 1),
        },
    }
