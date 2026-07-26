"""A lumped-capacitance surrogate of the same three-zone building.

Purpose: the repository must be runnable — end to end, closed loop, dashboard
and all — by a reviewer who has not installed a 200 MB simulation engine, and
by CI. It is selected automatically when EnergyPlus is absent, and the
dashboard labels every number produced by it, so surrogate output is never
mistaken for EnergyPlus output.

Model, per zone: a 2R2C network (air node + mass node) driven by conduction,
solar gain through the roof and glazing, internal gains, infiltration and
ventilation, with an ideal HVAC that meets whatever load the dual set-point
band demands. Humidity is a simple latent balance, CO2 the same mass balance
the metrics module uses. Integration is explicit Euler on a 60 s sub-step,
which is stable for these time constants.

It is a surrogate, not a validated model: it reproduces the *shape* and the
*direction* of the response (and therefore exercises the whole control stack),
typically within ~10-15% of the EnergyPlus daily cooling energy for this
building. All headline results in the submission come from EnergyPlus.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from ..bus import STATUS, TELEMETRY, EventBus
from ..config import RunConfig
from ..logs import LogDigest
from ..metrics import (
    DEFAULT_PLANT,
    KPIAccumulator,
    PlantModel,
    co2_next_ppm,
    fanger_pmv,
    grid_signal,
    summer_clo,
)
from ..telemetry import ControlAction, Snapshot, ZoneState
from ..weather import EPW, WeatherRecord, resolve_epw, synthetic_record
from ..agent.guardrails import clamp
from .schedules import minutes_until_occupied, minutes_until_vacant
from .base import DEFAULT_ZONES, ControlPolicy, EngineResult, ZoneSpec, ensure_out_dir

AIR_DENSITY = 1.2
AIR_CP = 1006.0
LATENT_HEAT = 2.45e6      # J/kg water
# Inside surface film coefficient (W/m2K) for a horizontal surface, ASHRAE.
INSIDE_FILM = 8.3


@dataclass
class ZoneThermal:
    """Envelope parameters per zone, matched to models/baseline.idf."""

    spec: ZoneSpec
    ua_envelope_w_k: float      # conduction + glazing, W/K
    ua_air_mass_w_k: float      # air <-> internal mass coupling, W/K
    c_air_j_k: float            # air capacitance
    c_mass_j_k: float           # thermal mass capacitance (brick + slab)
    solar_aperture_m2: float    # effective area x SHGC for glazing
    roof_area_m2: float
    roof_u_w_m2k: float
    roof_absorptance: float
    infiltration_ach: float

    # Surface areas used for the radiant weighting.
    other_surface_area_m2: float = 140.0

    # State
    t_air: float = 28.0
    t_mass: float = 28.5
    t_roof_inner: float = 30.0  # inner surface of the roof slab
    w_air: float = 0.014        # humidity ratio, kg/kg
    co2_ppm: float = 450.0

    def mean_radiant_c(self) -> float:
        """Area-weighted mean radiant temperature.

        This matters more than it looks. An uninsulated RCC roof reaches ~40 C on
        its inner face at midday, which pulls MRT several degrees above air
        temperature and is why EnergyPlus reports the production hall as warm
        even when its air is exactly at set-point. Approximating MRT by the air
        temperature — the obvious shortcut — made this engine disagree with
        EnergyPlus by up to 1.8 PMV, enough that the safety layer spent the day
        fighting the baseline it was supposed to leave alone.
        """
        total = self.roof_area_m2 + self.other_surface_area_m2
        return (
            self.roof_area_m2 * self.t_roof_inner + self.other_surface_area_m2 * self.t_mass
        ) / total


def _zone_thermal(spec: ZoneSpec) -> ZoneThermal:
    """Envelope numbers derived from the IDF constructions.

    Walls: 12 mm plaster + 230 mm brick + 12 mm plaster -> U ~ 2.0 W/m2K.
    Roof:  40 mm screed + 150 mm RCC + plaster           -> U ~ 3.0 W/m2K.
    Glazing: single clear, U 5.8, SHGC 0.82.
    """
    wall_u, glass_u, roof_u = 2.0, 5.8, 3.0
    glazing = {"OFFICE": 9.0 + 3.6, "PROD_HALL": 6.0, "PACK_STORE": 2.0}.get(spec.name, 4.0)
    exterior_wall = {"OFFICE": 8 * 3.6 * 2 + 10 * 3.6, "PROD_HALL": 8 * 3.6 * 2,
                     "PACK_STORE": 8 * 3.6 * 2 + 10 * 3.6}.get(spec.name, 60.0) - glazing
    roof_area = spec.area_m2
    ua = exterior_wall * wall_u + glazing * glass_u
    return ZoneThermal(
        spec=spec,
        ua_envelope_w_k=ua,
        ua_air_mass_w_k=6.0 * spec.area_m2,          # ~6 W/m2K film coupling
        c_air_j_k=spec.volume_m3 * AIR_DENSITY * AIR_CP * 3.0,   # air + furniture
        c_mass_j_k=spec.area_m2 * 1.1e5,             # brick/slab effective mass
        solar_aperture_m2=glazing * 0.82 * 0.45,     # SHGC x incidence factor
        roof_area_m2=roof_area,
        other_surface_area_m2=exterior_wall + glazing + spec.area_m2,   # walls + floor
        roof_u_w_m2k=roof_u,
        roof_absorptance=0.75,
        infiltration_ach={"OFFICE": 0.30, "PROD_HALL": 0.45, "PACK_STORE": 0.35}.get(spec.name, 0.35),
    )


# Occupancy / lighting / equipment fractions mirroring the IDF Schedule:Compact
# objects. Keyed by (weekday flag, hour).
def _fraction(profile: list[tuple[float, float]], hour: float) -> float:
    for until, value in profile:
        if hour < until:
            return value
    return profile[-1][1]


_OCC = {
    "OFFICE": [(9, 0.0), (13, 0.95), (14, 0.5), (18, 0.9), (19, 0.25), (24, 0.0)],
    "PROD_HALL": [(6, 0.0), (10, 1.0), (13, 0.9), (14, 0.4), (18, 0.95), (19, 0.3), (24, 0.0)],
    "PACK_STORE": [(7, 0.0), (12, 0.6), (14, 0.3), (18, 0.7), (24, 0.0)],
}
_OCC_SAT = {
    "OFFICE": [(9, 0.0), (14, 0.4), (24, 0.0)],
    "PROD_HALL": [(6, 0.0), (14, 0.7), (24, 0.0)],
    "PACK_STORE": [(8, 0.0), (14, 0.4), (24, 0.0)],
}
_LIGHT = [(6, 0.05), (19, 0.9), (21, 0.2), (24, 0.05)]
_EQUIP = [(6, 0.15), (18, 0.85), (20, 0.35), (24, 0.15)]
_PEOPLE_COUNT = {"OFFICE": 6, "PROD_HALL": 12, "PACK_STORE": 4}
_MET_W = {"OFFICE": 120.0, "PROD_HALL": 200.0, "PACK_STORE": 165.0}


class SurrogateEngine:
    name = "surrogate"

    def __init__(
        self,
        cfg: RunConfig,
        bus: EventBus,
        zones: tuple[ZoneSpec, ...] = DEFAULT_ZONES,
        plant: PlantModel = DEFAULT_PLANT,
        digest: LogDigest | None = None,
        idf_override: str | Path | None = None,
        snapshot_sink: list[Snapshot] | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.plant = plant
        self.digest = digest or LogDigest()
        self.idf_override = idf_override
        self.zones = zones
        self.thermal = [_zone_thermal(z) for z in zones]
        self._epw: EPW | None = None
        self._snapshots: list[Snapshot] = snapshot_sink if snapshot_sink is not None else []

    # -------------------------------------------------------------- weather
    def _weather(self, month: int, day: int, hour: int, minute: int) -> WeatherRecord:
        if self._epw is not None:
            return self._epw.at(month, day, hour, minute)
        return synthetic_record(month, day, hour, minute)

    # ------------------------------------------------------------------ run
    def run(self, policy: ControlPolicy, label: str) -> EngineResult:
        epw_path = resolve_epw(self.cfg.epw)
        if epw_path is not None:
            try:
                self._epw = EPW.load(epw_path)
            except (OSError, ValueError) as exc:
                self.digest.add_line(f"** Warning ** could not read {epw_path}: {exc}")
                self._epw = None

        out_dir = ensure_out_dir(self.cfg.out_dir / "surrogate" / label)
        acc = KPIAccumulator(self.cfg.comfort, label, self.name)
        self._snapshots.clear()   # clear in place: the agent holds this list
        snapshots = self._snapshots

        steps_per_hour = max(1, self.cfg.timesteps_per_hour)
        dt_h = 1.0 / steps_per_hour
        dt_s = dt_h * 3600.0
        sub_steps = max(1, int(dt_s // 60))
        sub_dt = dt_s / sub_steps

        days = _day_sequence(self.cfg.start_month, self.cfg.start_day, self.cfg.end_month, self.cfg.end_day)
        total_steps = len(days) * 24 * steps_per_hour

        self.bus.publish(
            STATUS, phase="start", label=label, engine=self.name,
            idf=str(self.idf_override or self.cfg.idf), epw=str(epw_path or "synthetic"),
        )
        self.digest.add_line(
            "**   ~~~   ** surrogate engine active (EnergyPlus not installed or explicitly selected)"
        )

        # Initialise zone states from the first weather record.
        first = self._weather(days[0][0], days[0][1], 5, 0)
        for zt in self.thermal:
            zt.t_air = first.drybulb_c - 1.0
            zt.t_mass = first.drybulb_c - 0.5
            zt.t_roof_inner = first.drybulb_c
            zt.w_air = _humidity_ratio(first.drybulb_c, first.rh_pct)
            zt.co2_ppm = 430.0

        started = time.time()
        step = 0
        last_action: ControlAction | None = None
        for month, day, weekday in days:
            for hour in range(24):
                for ts in range(steps_per_hour):
                    minute = int(ts * 60 / steps_per_hour)
                    weather = self._weather(month, day, hour, minute)
                    prev = snapshots[-1] if snapshots else None
                    snap = self._build_snapshot(
                        step, month, day, hour, minute, dt_h, weather, weekday, prev
                    )
                    # Visible to the agent's tools before it decides.
                    snapshots.append(snap)
                    requested = policy.decide(snap)
                    # Same safety layer as the EnergyPlus engine, same place in
                    # the sequence — so a policy behaves identically on both.
                    action = clamp(requested, snap, self.cfg.comfort, previous=last_action)
                    last_action = action
                    notify = getattr(policy, "on_applied", None)
                    if callable(notify):
                        notify(action, snap)
                    snap.guardrail_notes = action.clamped
                    self._integrate(snap, action, weather, weekday, hour, minute, sub_dt, sub_steps)
                    self._finalise(snap, action, acc, prev)
                    self.bus.publish(TELEMETRY, label=label, snapshot=snap.to_dict())
                    if self.cfg.pace_s > 0:
                        time.sleep(self.cfg.pace_s)
                    step += 1
                    if total_steps and step % max(1, total_steps // 20) == 0:
                        self.bus.publish(
                            STATUS, phase="running", label=label,
                            percent=int(100 * step / total_steps),
                        )
        wall = time.time() - started

        (out_dir / "surrogate.log").write_text(self.digest.for_llm(4000), encoding="utf-8")
        self.bus.publish(
            STATUS, phase="done", label=label, ok=True, steps=len(snapshots),
            wall_seconds=round(wall, 2), log=self.digest.summary(),
        )
        return EngineResult(
            label=label,
            engine=self.name,
            kpi=acc.result(),
            snapshots=snapshots,
            ok=bool(snapshots),
            error="" if snapshots else "no timesteps generated",
            idf_used=str(self.idf_override or self.cfg.idf),
            epw_used=str(epw_path or "synthetic"),
            out_dir=str(out_dir),
            wall_seconds=wall,
            severe_count=self.digest.severe_count,
            warning_count=self.digest.warning_count,
        )

    # ------------------------------------------------------------ internals
    def _build_snapshot(
        self,
        step: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        dt_h: float,
        weather: WeatherRecord,
        weekday: int,
        prev: Snapshot | None = None,
    ) -> Snapshot:
        zones: list[ZoneState] = []
        clo = summer_clo(month)
        occupied = False
        for zt in self.thermal:
            spec = zt.spec
            occ_frac = _occupancy(spec.name, hour + minute / 60.0, weekday)
            people = occ_frac * _PEOPLE_COUNT.get(spec.name, 4)
            zs = ZoneState(
                name=spec.name,
                temp_c=zt.t_air,
                rh_pct=_relative_humidity(zt.t_air, zt.w_air),
                co2_ppm=zt.co2_ppm,
                occupants=people,
                area_m2=spec.area_m2,
                minutes_until_occupied=minutes_until_occupied(spec.name, hour, minute, weekday + 1),
                minutes_until_vacant=minutes_until_vacant(spec.name, hour, minute, weekday + 1),
                pmv_limit=spec.pmv_limit,
                met=spec.met,
                clo=clo,
                air_velocity_m_s=spec.air_velocity_m_s,
            )
            zs.pmv = fanger_pmv(
                ta=zt.t_air,
                tr=zt.mean_radiant_c(),
                vel=spec.air_velocity_m_s,
                rh=zs.rh_pct,
                met=spec.met,
                clo=clo,
            )
            if people > 0.05:
                occupied = True
            zones.append(zs)
        snap = Snapshot(
            step=step,
            sim_seconds=step * dt_h * 3600.0,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            weekday=weekday + 1,          # internal 0=Mon -> ISO 1=Mon
            timestep_hours=dt_h,
            outdoor_temp_c=weather.drybulb_c,
            outdoor_rh_pct=weather.rh_pct,
            solar_w_m2=weather.ghi_w_m2,
            wind_speed_m_s=weather.wind_m_s,
            zones=zones,
            grid=grid_signal(hour, self.cfg.grid),
            occupied=occupied,
        )
        if prev is not None:
            # Carry the last measured power and running totals forward, so an
            # agent inspecting this snapshot mid-decision sees real metering
            # rather than zeros (this timestep's HVAC has not run yet).
            snap.hvac_cooling_elec_w = prev.hvac_cooling_elec_w
            snap.hvac_heating_elec_w = prev.hvac_heating_elec_w
            snap.fan_elec_w = prev.fan_elec_w
            snap.lights_elec_w = prev.lights_elec_w
            snap.equip_elec_w = prev.equip_elec_w
            snap.cum_kwh = prev.cum_kwh
            snap.cum_hvac_kwh = prev.cum_hvac_kwh
            snap.cum_cost_inr = prev.cum_cost_inr
            snap.cum_carbon_kg = prev.cum_carbon_kg
            snap.peak_demand_w = prev.peak_demand_w
            for zs, pz in zip(snap.zones, prev.zones):
                zs.cooling_setpoint_c = pz.cooling_setpoint_c
                zs.heating_setpoint_c = pz.heating_setpoint_c
        return snap

    def _integrate(
        self,
        snap: Snapshot,
        action: ControlAction,
        weather: WeatherRecord,
        weekday: int,
        hour: int,
        minute: int,
        sub_dt: float,
        sub_steps: int,
    ) -> None:
        clock = hour + minute / 60.0
        light_frac = _fraction(_LIGHT, clock)
        equip_frac = _fraction(_EQUIP, clock)
        # Sol-air temperature for the roof: outdoor + absorbed solar / h_out.
        sol_air = weather.drybulb_c + 0.75 * weather.ghi_w_m2 / 20.0

        for zt, zs in zip(self.thermal, snap.zones):
            spec = zt.spec
            cool_sp, heat_sp = action.setpoints_for(spec.name)
            oa_fraction = float(
                action.zone_overrides.get(spec.name, {}).get("oa_fraction", action.oa_fraction)
            )
            oa_flow = spec.design_oa_m3_s * max(0.0, min(1.0, oa_fraction))
            # Ventilation only runs when the space is in use.
            if zs.occupants <= 0.05:
                oa_flow *= 0.15

            people_w = zs.occupants * _MET_W.get(spec.name, 150.0)
            lights_w = spec.lights_w_m2 * spec.area_m2 * light_frac
            equip_w = spec.equip_w_m2 * spec.area_m2 * equip_frac
            solar_w = zt.solar_aperture_m2 * weather.ghi_w_m2
            infil_m3_s = zt.infiltration_ach * spec.volume_m3 / 3600.0

            cooling_thermal = 0.0
            heating_thermal = 0.0
            for _ in range(sub_steps):
                q_env = zt.ua_envelope_w_k * (weather.drybulb_c - zt.t_air)
                q_roof = zt.roof_area_m2 * zt.roof_u_w_m2k * (sol_air - zt.t_air)
                q_mass = zt.ua_air_mass_w_k * (zt.t_mass - zt.t_air)
                q_vent = (oa_flow + infil_m3_s) * AIR_DENSITY * AIR_CP * (weather.drybulb_c - zt.t_air)
                q_int = people_w * 0.6 + lights_w + equip_w   # 60% of metabolic is sensible
                q_gain = q_env + q_roof + q_mass + q_vent + q_int + solar_w

                # Ideal HVAC: exactly meet the band, nothing more.
                t_free = zt.t_air + q_gain * sub_dt / zt.c_air_j_k
                q_hvac = 0.0
                if t_free > cool_sp:
                    q_hvac = -(t_free - cool_sp) * zt.c_air_j_k / sub_dt
                    cooling_thermal += -q_hvac
                elif t_free < heat_sp:
                    q_hvac = (heat_sp - t_free) * zt.c_air_j_k / sub_dt
                    heating_thermal += q_hvac

                zt.t_air += (q_gain + q_hvac) * sub_dt / zt.c_air_j_k
                # Mass node: driven by the air node and by absorbed solar.
                q_to_mass = zt.ua_air_mass_w_k * (zt.t_air - zt.t_mass) + 0.35 * solar_w
                zt.t_mass += q_to_mass * sub_dt / zt.c_mass_j_k
                # Roof inner surface: steady-state split of the slab resistance
                # between the outside (sol-air) and inside (film) sides.
                zt.t_roof_inner = zt.t_air + (sol_air - zt.t_air) * (
                    zt.roof_u_w_m2k / (zt.roof_u_w_m2k + INSIDE_FILM)
                )

                # Latent balance: moisture in from OA/infiltration and people,
                # out through the cooling coil at its apparatus dew point.
                w_out = _humidity_ratio(weather.drybulb_c, weather.rh_pct)
                flow_kg_s = (oa_flow + infil_m3_s) * AIR_DENSITY
                mass_air = spec.volume_m3 * AIR_DENSITY
                people_kg_s = zs.occupants * 5.0e-6 * 12.0
                dw = (flow_kg_s * (w_out - zt.w_air) + people_kg_s) * sub_dt / mass_air
                zt.w_air = max(0.004, min(0.025, zt.w_air + dw))
                if cooling_thermal > 0:
                    # Coil dehumidification, capped by the sensible heat ratio.
                    latent_capacity = cooling_thermal * (1 - 0.7) / 0.7
                    dehum_kg_s = latent_capacity / LATENT_HEAT
                    zt.w_air = max(0.006, zt.w_air - dehum_kg_s * sub_dt / mass_air)

                zt.co2_ppm = co2_next_ppm(
                    zt.co2_ppm, zs.occupants, oa_flow + infil_m3_s, spec.volume_m3, sub_dt
                )

            avg_cool = cooling_thermal / sub_steps
            avg_heat = heating_thermal / sub_steps
            # Latent load the coil must also remove (adds to electrical load).
            latent_w = avg_cool * (1 - 0.7) / 0.7 if avg_cool > 0 else 0.0

            zs.temp_c = zt.t_air
            zs.rh_pct = _relative_humidity(zt.t_air, zt.w_air)
            zs.co2_ppm = zt.co2_ppm
            zs.cooling_rate_w = avg_cool + latent_w
            zs.heating_rate_w = avg_heat
            zs.lights_w = lights_w
            zs.equip_w = equip_w
            zs.cooling_setpoint_c = cool_sp
            zs.heating_setpoint_c = heat_sp
            zs.pmv = fanger_pmv(
                ta=zt.t_air,
                tr=zt.mean_radiant_c(),
                vel=spec.air_velocity_m_s,
                rh=zs.rh_pct,
                met=spec.met,
                clo=summer_clo(snap.month),
            )

    def _finalise(
        self,
        snap: Snapshot,
        action: ControlAction,
        acc: KPIAccumulator,
        prev: Snapshot | None,
    ) -> None:
        cooling_w = sum(z.cooling_rate_w for z in snap.zones)
        heating_w = sum(z.heating_rate_w for z in snap.zones)
        fan_w = 0.0
        for zt, zs in zip(self.thermal, snap.zones):
            thermal = zs.cooling_rate_w + zs.heating_rate_w
            fan_w += self.plant.fan_power_w(
                thermal, enabled=thermal > 1.0, design_flow_m3_s=zt.spec.design_flow_m3_s
            )
        cool_elec, heat_elec = self.plant.electric_w(cooling_w, heating_w)
        snap.hvac_cooling_elec_w = cool_elec
        snap.hvac_heating_elec_w = heat_elec
        snap.fan_elec_w = fan_w
        snap.lights_elec_w = sum(z.lights_w for z in snap.zones)
        snap.equip_elec_w = sum(z.equip_w for z in snap.zones)
        snap.control_source = action.source
        snap.decision_id = action.decision_id

        total_kwh = snap.total_elec_w * snap.timestep_hours / 1000.0
        hvac_kwh = (cool_elec + heat_elec + fan_w) * snap.timestep_hours / 1000.0
        snap.cum_kwh = (prev.cum_kwh if prev else 0.0) + total_kwh
        snap.cum_hvac_kwh = (prev.cum_hvac_kwh if prev else 0.0) + hvac_kwh
        snap.cum_cost_inr = (prev.cum_cost_inr if prev else 0.0) + total_kwh * snap.grid.tariff_inr_per_kwh
        snap.cum_carbon_kg = (
            (prev.cum_carbon_kg if prev else 0.0) + total_kwh * snap.grid.carbon_g_per_kwh / 1000.0
        )
        snap.peak_demand_w = max(prev.peak_demand_w if prev else 0.0, snap.total_elec_w)

        occupied_zones = [z for z in snap.zones if z.occupants > 0.05]
        if occupied_zones:
            snap.pmv_worst = max((z.pmv for z in occupied_zones), key=abs)
            snap.co2_worst_ppm = max(z.co2_ppm for z in occupied_zones)
            snap.comfort_violation = (
                snap.co2_worst_ppm > self.cfg.comfort.co2_limit_ppm
                or any(
                    abs(z.pmv) > (z.pmv_limit or self.cfg.comfort.pmv_limit) for z in occupied_zones
                )
                or any(
                    not (
                        self.cfg.comfort.occupied_temp_min_c
                        <= z.temp_c
                        <= self.cfg.comfort.occupied_temp_max_c
                    )
                    for z in occupied_zones
                )
            )
        # The snapshot was appended to the shared list before the policy was
        # asked to decide, so the agent's tools could see it; only the KPI
        # accumulation belongs here.
        acc.add(snap)


# ---------------------------------------------------------------- helpers


def _occupancy(zone: str, clock: float, weekday: int) -> float:
    if weekday == 6:          # Sunday
        return 0.0
    profile = _OCC_SAT if weekday == 5 else _OCC
    return _fraction(profile.get(zone, _OCC["OFFICE"]), clock)


def _saturation_pressure(t_c: float) -> float:
    """Magnus formula, Pa."""
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def _humidity_ratio(t_c: float, rh_pct: float) -> float:
    pv = _saturation_pressure(t_c) * max(0.0, min(100.0, rh_pct)) / 100.0
    return max(0.001, 0.62198 * pv / (101325.0 - pv))


def _relative_humidity(t_c: float, w: float) -> float:
    pv = 101325.0 * w / (0.62198 + w)
    rh = 100.0 * pv / _saturation_pressure(t_c)
    return max(1.0, min(100.0, rh))


_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _day_sequence(
    start_month: int, start_day: int, end_month: int, end_day: int
) -> list[tuple[int, int, int]]:
    """Inclusive (month, day, weekday) list. Weekday 0=Mon .. 6=Sun, anchored so
    the sequence starts on a Tuesday to match the IDF RunPeriod."""
    out: list[tuple[int, int, int]] = []
    month, day = start_month, start_day
    index = 0
    guard = 0
    while guard < 400:
        out.append((month, day, (1 + index) % 7))
        if month == end_month and day == end_day:
            break
        day += 1
        if day > _DAYS_IN_MONTH[month - 1]:
            day = 1
            month = month % 12 + 1
        index += 1
        guard += 1
    return out
