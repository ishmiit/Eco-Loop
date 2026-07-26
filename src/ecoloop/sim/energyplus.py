"""The EnergyPlus engine — feedback out, control in, inside one running sim.

This is the closed loop. EnergyPlus runs *in-process* through its C API
(``libenergyplusapi`` via ``pyenergyplus``), and on every zone timestep our
callback:

    1. reads 35 sensor handles straight out of the running model — ten
       variables per zone plus five site variables (no CSV, no polling);
    2. hands a Snapshot to the control policy;
    3. writes the returned set-points into live ``Schedule:Constant``
       actuators, which the thermostat predictor reads microseconds later in
       the same timestep.

Forward injection is therefore genuinely in-band: the value the agent chose
changes the load EnergyPlus is about to compute, not a file for a later run.

Two details that matter for robustness:

* **Warmup is skipped.** ``warmup_flag`` is true for the first few simulated
  days while EnergyPlus converges initial conditions; recording those would
  double-count energy. We also wait for ``api_data_fully_ready`` before
  resolving handles, because handles do not exist until then.
* **Exceptions never escape the callback.** An exception raised inside an
  EnergyPlus callback crosses a C boundary and aborts the process. Every
  callback body is wrapped; a failure degrades to "hold last action" and is
  recorded, so a long run cannot be killed by one bad timestep.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ..bus import ECM, LOG, STATUS, TELEMETRY, EventBus
from ..config import RunConfig
from ..energyplus_locate import ensure_importable
from ..logs import LogDigest
from ..metrics import DEFAULT_PLANT, KPIAccumulator, PlantModel, grid_signal, summer_clo
from ..telemetry import ControlAction, Snapshot, ZoneState
from ..agent.guardrails import clamp
from ..weather import resolve_epw
from .base import DEFAULT_ZONES, ControlPolicy, EngineResult, ZoneSpec, ensure_out_dir
from .idf import IDF
from .schedules import minutes_until_occupied, minutes_until_vacant

# Output variables requested per zone. (variable name, key kind)
#   "zone"   -> keyed by zone name
#   "ideal"  -> keyed by the ZoneHVAC:IdealLoadsAirSystem name
#   "people" -> keyed by the People object name
_ZONE_VARS: tuple[tuple[str, str, str], ...] = (
    ("temp_c", "Zone Mean Air Temperature", "zone"),
    ("rh_pct", "Zone Air Relative Humidity", "zone"),
    ("co2_ppm", "Zone Air CO2 Concentration", "zone"),
    ("occupants", "Zone People Occupant Count", "zone"),
    ("lights_w", "Zone Lights Electricity Rate", "zone"),
    ("equip_w", "Zone Electric Equipment Electricity Rate", "zone"),
    ("cooling_rate_w", "Zone Ideal Loads Supply Air Total Cooling Rate", "ideal"),
    ("heating_rate_w", "Zone Ideal Loads Supply Air Total Heating Rate", "ideal"),
    ("oa_flow_m3_s", "Zone Ideal Loads Outdoor Air Standard Density Volume Flow Rate", "ideal"),
    ("pmv", "Zone Thermal Comfort Fanger Model PMV", "people"),
)

_SITE_VARS: tuple[tuple[str, str], ...] = (
    ("outdoor_temp_c", "Site Outdoor Air Drybulb Temperature"),
    ("outdoor_rh_pct", "Site Outdoor Air Relative Humidity"),
    ("solar_direct", "Site Direct Solar Radiation Rate per Area"),
    ("solar_diffuse", "Site Diffuse Solar Radiation Rate per Area"),
    ("wind_speed_m_s", "Site Wind Speed"),
)


def _ideal_loads_name(zone: ZoneSpec) -> str:
    """OFFICE -> OFFICE_IDEAL_LOADS, PROD_HALL -> PROD_IDEAL_LOADS."""
    stem = {"OFFICE": "OFFICE", "PROD_HALL": "PROD", "PACK_STORE": "STORE"}.get(
        zone.name, zone.name.split("_")[0]
    )
    return f"{stem}_IDEAL_LOADS"


class EnergyPlusEngine:
    """Runs one simulation under closed-loop control."""

    name = "energyplus"

    def __init__(
        self,
        cfg: RunConfig,
        bus: EventBus,
        zones: tuple[ZoneSpec, ...] = DEFAULT_ZONES,
        plant: PlantModel = DEFAULT_PLANT,
        idf_override: str | Path | None = None,
        digest: LogDigest | None = None,
        snapshot_sink: list[Snapshot] | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.zones = zones
        self.plant = plant
        self.idf_override = Path(idf_override) if idf_override else None
        self.digest = digest or LogDigest()
        # Shared with the agent so its tools read the same telemetry the engine
        # is producing, with no copy and no lag.
        self._snapshots: list[Snapshot] = snapshot_sink if snapshot_sink is not None else []

        self._api: Any = None
        self._state: Any = None
        self._handles: dict[str, int] = {}
        self._actuators: dict[str, int] = {}
        self._ready = False
        self._step = 0
        self._policy: ControlPolicy | None = None
        self._acc: KPIAccumulator | None = None
        self._cum = {"kwh": 0.0, "hvac_kwh": 0.0, "cost": 0.0, "carbon": 0.0, "peak_w": 0.0}
        self._last_action: ControlAction | None = None
        self._callback_errors = 0
        self._label = ""
        self._handle_error = ""
        self._defaults: dict[str, float] = {}

    # ------------------------------------------------------------------ setup
    def prepare_idf(self, label: str) -> Path:
        """Write the exact IDF this run will use into the run's artifacts, so
        every reported number is traceable to a committed input file."""
        source = self.idf_override or Path(self.cfg.idf)
        idf = IDF.load(source)
        idf.set_run_period(
            self.cfg.start_month, self.cfg.start_day, self.cfg.end_month, self.cfg.end_day
        )
        idf.set_timestep(self.cfg.timesteps_per_hour)
        self._defaults = idf.schedule_constants()
        target = ensure_out_dir(self.cfg.out_dir / "idf") / f"{label}.idf"
        idf.save(target)
        return target

    def _request_variables(self, api: Any, state: Any) -> None:
        """``request_variable`` must be called before the run starts, otherwise
        the variable is not instantiated and its handle comes back as -1."""
        for _, var, _ in _ZONE_VARS:
            for zone in self.zones:
                for key in {zone.name, _ideal_loads_name(zone), zone.people_object}:
                    api.exchange.request_variable(state, var, key)
        for _, var in _SITE_VARS:
            api.exchange.request_variable(state, var, "Environment")

    def _resolve_handles(self, api: Any, state: Any) -> None:
        missing: list[str] = []
        for field_name, var, key_kind in _ZONE_VARS:
            for zone in self.zones:
                key = {
                    "zone": zone.name,
                    "ideal": _ideal_loads_name(zone),
                    "people": zone.people_object,
                }[key_kind]
                handle = api.exchange.get_variable_handle(state, var, key)
                self._handles[f"{zone.name}:{field_name}"] = handle
                if handle < 0:
                    missing.append(f"{var}@{key}")
        for field_name, var in _SITE_VARS:
            handle = api.exchange.get_variable_handle(state, var, "Environment")
            self._handles[f"site:{field_name}"] = handle
            if handle < 0:
                missing.append(f"{var}@Environment")

        for zone in self.zones:
            for slot, schedule in (
                ("cool", zone.cooling_sp_schedule),
                ("heat", zone.heating_sp_schedule),
                ("oa", zone.oa_schedule),
            ):
                handle = api.exchange.get_actuator_handle(
                    state, "Schedule:Constant", "Schedule Value", schedule
                )
                self._actuators[f"{zone.name}:{slot}"] = handle
                if handle < 0:
                    missing.append(f"actuator:{schedule}")

        if missing:
            self._handle_error = "unresolved handles: " + ", ".join(missing[:8])
            self.bus.publish(LOG, level="severe", message=self._handle_error, source="engine")
        else:
            self.bus.publish(
                LOG,
                level="info",
                source="engine",
                message=(
                    f"{len(self._handles)} sensor handles and {len(self._actuators)} "
                    f"actuator handles bound"
                ),
            )

    def _get(self, key: str) -> float:
        handle = self._handles.get(key, -1)
        if handle < 0:
            return 0.0
        return float(self._api.exchange.get_variable_value(self._state, handle))

    # ------------------------------------------------------------- callbacks
    def _on_message(self, message: bytes | str) -> None:
        try:
            text = message.decode("utf-8", "replace") if isinstance(message, bytes) else str(message)
            for line in text.splitlines():
                if not line.strip():
                    continue
                level, msg = self.digest.add_line(line)
                if level in ("severe", "fatal"):
                    self.bus.publish(LOG, level=level, message=msg, source="energyplus")
        except Exception:
            pass  # never let logging kill the simulation

    def _on_progress(self, percent: int) -> None:
        try:
            self.bus.publish(STATUS, phase="running", label=self._label, percent=int(percent))
        except Exception:
            pass

    def _on_timestep(self, state: Any) -> None:
        """The closed loop. Wrapped whole: an exception here would abort E+."""
        try:
            api = self._api
            if api.exchange.warmup_flag(state):
                return
            if not self._ready:
                if not api.exchange.api_data_fully_ready(state):
                    return
                self._resolve_handles(api, state)
                self._ready = True

            snap = self._read_snapshot(state)
            # Publish into the shared list *before* asking for a decision, so a
            # tool call made during this decision sees the state it is deciding
            # about rather than the previous timestep.
            self._snapshots.append(snap)
            requested = self._policy.decide(snap) if self._policy else ControlAction()
            # THE SAFETY LAYER. Applied here rather than inside any one policy,
            # so every brain — LLM, heuristic and the rule-based baseline — is
            # held to the identical envelope, and no action can reach an
            # actuator without passing it.
            action = clamp(requested, snap, self.cfg.comfort, previous=self._last_action)
            self._apply(state, action, snap)
            notify = getattr(self._policy, "on_applied", None)
            if callable(notify):
                notify(action, snap)

            snap.control_source = action.source
            snap.decision_id = action.decision_id
            snap.guardrail_notes = action.clamped
            self._last_action = action

            if self._acc is not None:
                self._acc.add(snap)
            self.bus.publish(TELEMETRY, label=self._label, snapshot=snap.to_dict())

            if self.cfg.pace_s > 0:
                time.sleep(self.cfg.pace_s)
        except Exception as exc:
            self._callback_errors += 1
            if self._callback_errors <= 3:
                self.bus.publish(
                    LOG,
                    level="severe",
                    source="engine",
                    message=f"timestep callback failed ({type(exc).__name__}: {exc})",
                    traceback=traceback.format_exc(limit=4),
                )

    # ------------------------------------------------------------ telemetry
    def _read_snapshot(self, state: Any) -> Snapshot:
        api = self._api
        ex = api.exchange
        steps_in_hour = max(1, int(ex.num_time_steps_in_hour(state)))
        hour = int(ex.hour(state))
        # Derive the minute from the zone timestep index rather than from
        # ``minutes()``: that call reports minutes into the hour at the current
        # *system* timestep, which is finer than the zone timestep and produces
        # ragged clocks like 12:23. The timestep index gives exact 0/15/30/45.
        minute = int(round((int(ex.zone_time_step_number(state)) - 1) * 60.0 / steps_in_hour))
        if minute >= 60:
            minute -= 60
            hour += 1
        if hour >= 24:
            hour -= 24
        timestep_hours = 1.0 / steps_in_hour
        # EnergyPlus day_of_week is 1=Sunday..7=Saturday; convert to ISO.
        weekday = ((int(ex.day_of_week(state)) + 5) % 7) + 1

        zones: list[ZoneState] = []
        cooling_w = heating_w = lights_w = equip_w = 0.0
        fan_w = 0.0
        occupied = False
        for zone in self.zones:
            zs = ZoneState(
                name=zone.name,
                area_m2=zone.area_m2,
                pmv_limit=zone.pmv_limit,
                met=zone.met,
                clo=summer_clo(int(ex.month(state))),
                air_velocity_m_s=zone.air_velocity_m_s,
            )
            zs.temp_c = self._get(f"{zone.name}:temp_c")
            zs.rh_pct = self._get(f"{zone.name}:rh_pct")
            zs.co2_ppm = self._get(f"{zone.name}:co2_ppm") or 420.0
            zs.occupants = self._get(f"{zone.name}:occupants")
            zs.pmv = self._get(f"{zone.name}:pmv")
            zs.lights_w = self._get(f"{zone.name}:lights_w")
            zs.equip_w = self._get(f"{zone.name}:equip_w")
            zs.cooling_rate_w = self._get(f"{zone.name}:cooling_rate_w")
            zs.heating_rate_w = self._get(f"{zone.name}:heating_rate_w")
            zs.minutes_until_occupied = minutes_until_occupied(zone.name, hour, minute, weekday)
            zs.minutes_until_vacant = minutes_until_vacant(zone.name, hour, minute, weekday)

            # Report the set-points actually in force (read back from the
            # actuators rather than trusting our own copy).
            # Before our first write the actuator reads back 0 (not overridden),
            # so fall back to the schedule defaults parsed out of the IDF.
            cool_h = self._actuators.get(f"{zone.name}:cool", -1)
            heat_h = self._actuators.get(f"{zone.name}:heat", -1)
            zs.cooling_setpoint_c = self._defaults.get(zone.cooling_sp_schedule, 24.0)
            zs.heating_setpoint_c = self._defaults.get(zone.heating_sp_schedule, 21.0)
            if cool_h >= 0:
                value = float(ex.get_actuator_value(state, cool_h))
                if value > 0.0:
                    zs.cooling_setpoint_c = value
            if heat_h >= 0:
                value = float(ex.get_actuator_value(state, heat_h))
                if value > 0.0:
                    zs.heating_setpoint_c = value

            cooling_w += zs.cooling_rate_w
            heating_w += zs.heating_rate_w
            lights_w += zs.lights_w
            equip_w += zs.equip_w
            hvac_thermal = zs.cooling_rate_w + zs.heating_rate_w
            fan_w += self.plant.fan_power_w(
                hvac_thermal, enabled=hvac_thermal > 1.0, design_flow_m3_s=zone.design_flow_m3_s
            )
            if zs.occupants > 0.05:
                occupied = True
            zones.append(zs)

        cool_elec, heat_elec = self.plant.electric_w(cooling_w, heating_w)
        snap = Snapshot(
            step=self._step,
            sim_seconds=self._step * timestep_hours * 3600.0,
            month=int(ex.month(state)),
            day=int(ex.day_of_month(state)),
            hour=hour,
            minute=minute,
            weekday=weekday,
            timestep_hours=timestep_hours,
            outdoor_temp_c=self._get("site:outdoor_temp_c"),
            outdoor_rh_pct=self._get("site:outdoor_rh_pct"),
            solar_w_m2=self._get("site:solar_direct") + self._get("site:solar_diffuse"),
            wind_speed_m_s=self._get("site:wind_speed_m_s"),
            zones=zones,
            grid=grid_signal(hour, self.cfg.grid),
            hvac_cooling_elec_w=cool_elec,
            hvac_heating_elec_w=heat_elec,
            fan_elec_w=fan_w,
            lights_elec_w=lights_w,
            equip_elec_w=equip_w,
            occupied=occupied,
        )
        self._step += 1

        total_kwh = snap.total_elec_w * timestep_hours / 1000.0
        hvac_kwh = (cool_elec + heat_elec + fan_w) * timestep_hours / 1000.0
        self._cum["kwh"] += total_kwh
        self._cum["hvac_kwh"] += hvac_kwh
        self._cum["cost"] += total_kwh * snap.grid.tariff_inr_per_kwh
        self._cum["carbon"] += total_kwh * snap.grid.carbon_g_per_kwh / 1000.0
        self._cum["peak_w"] = max(self._cum["peak_w"], snap.total_elec_w)
        snap.cum_kwh = self._cum["kwh"]
        snap.cum_hvac_kwh = self._cum["hvac_kwh"]
        snap.cum_cost_inr = self._cum["cost"]
        snap.cum_carbon_kg = self._cum["carbon"]
        snap.peak_demand_w = self._cum["peak_w"]

        occupied_zones = [z for z in zones if z.occupants > 0.05]
        if occupied_zones:
            snap.pmv_worst = max((z.pmv for z in occupied_zones), key=abs)
            snap.co2_worst_ppm = max(z.co2_ppm for z in occupied_zones)
            snap.comfort_violation = (
                snap.co2_worst_ppm > self.cfg.comfort.co2_limit_ppm
                or any(abs(z.pmv) > (z.pmv_limit or self.cfg.comfort.pmv_limit) for z in occupied_zones)
                or any(
                    not (
                        self.cfg.comfort.occupied_temp_min_c
                        <= z.temp_c
                        <= self.cfg.comfort.occupied_temp_max_c
                    )
                    for z in occupied_zones
                )
            )
        return snap

    # --------------------------------------------------------------- control
    def _apply(self, state: Any, action: ControlAction, snap: Snapshot) -> None:
        ex = self._api.exchange
        for zone in self.zones:
            cool, heat = action.setpoints_for(zone.name)
            oa = float(action.zone_overrides.get(zone.name, {}).get("oa_fraction", action.oa_fraction))
            for slot, value in (("cool", cool), ("heat", heat), ("oa", oa)):
                handle = self._actuators.get(f"{zone.name}:{slot}", -1)
                if handle >= 0:
                    ex.set_actuator_value(state, handle, float(value))

    # ------------------------------------------------------------------- run
    def run(self, policy: ControlPolicy, label: str) -> EngineResult:
        install = ensure_importable()
        if install is None:
            return EngineResult(
                label=label,
                engine=self.name,
                kpi=KPIAccumulator(self.cfg.comfort, label, self.name).result(),
                ok=False,
                error="EnergyPlus not found — run scripts/install_energyplus.sh",
            )
        from pyenergyplus.api import EnergyPlusAPI  # noqa: PLC0415 (needs sys.path first)

        epw = resolve_epw(self.cfg.epw)
        if epw is None:
            return EngineResult(
                label=label,
                engine=self.name,
                kpi=KPIAccumulator(self.cfg.comfort, label, self.name).result(),
                ok=False,
                error="no .epw weather file found",
            )

        idf_path = self.prepare_idf(label)
        out_dir = ensure_out_dir(self.cfg.out_dir / "eplus" / label)

        self._label = label
        self._policy = policy
        self._acc = KPIAccumulator(self.cfg.comfort, label, self.name)
        self._snapshots.clear()   # clear in place: the agent holds this list
        self._step = 0
        self._ready = False
        self._handles.clear()
        self._actuators.clear()
        self._cum = {"kwh": 0.0, "hvac_kwh": 0.0, "cost": 0.0, "carbon": 0.0, "peak_w": 0.0}

        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        self._api, self._state = api, state

        api.runtime.set_console_output_status(state, False)
        api.runtime.callback_message(state, self._on_message)
        api.runtime.callback_progress(state, self._on_progress)
        # Sensors are read and set-points written on the zone timestep, after
        # the heat balance is initialised and before the thermostat predictor.
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, self._on_timestep)
        self._request_variables(api, state)

        self.bus.publish(
            STATUS,
            phase="start",
            label=label,
            engine=self.name,
            idf=str(idf_path),
            epw=str(epw),
            energyplus_version=install.version,
        )

        started = time.time()
        try:
            exit_code = api.runtime.run_energyplus(
                state, ["-d", str(out_dir), "-w", str(epw), str(idf_path)]
            )
        except Exception as exc:  # pragma: no cover - defensive
            exit_code = -1
            self.bus.publish(LOG, level="fatal", source="engine", message=f"run_energyplus raised: {exc}")
        finally:
            wall = time.time() - started
            try:
                api.state_manager.delete_state(state)
            except Exception:
                pass
            self._api = self._state = None

        self.digest.add_file(out_dir / "eplusout.err")
        kpi = self._acc.result()
        ok = exit_code == 0 and not self.digest.has_fatal and bool(self._snapshots)
        error = ""
        if exit_code != 0:
            error = f"EnergyPlus exited with code {exit_code}"
        elif not self._snapshots:
            error = "no timesteps were recorded (check the run period)"
        elif self._handle_error:
            error = self._handle_error

        self.bus.publish(
            STATUS,
            phase="done",
            label=label,
            ok=ok,
            error=error,
            steps=len(self._snapshots),
            wall_seconds=round(wall, 2),
            log=self.digest.summary(),
        )
        return EngineResult(
            label=label,
            engine=self.name,
            kpi=kpi,
            snapshots=self._snapshots,
            ok=ok,
            error=error,
            idf_used=str(idf_path),
            epw_used=str(epw),
            out_dir=str(out_dir),
            wall_seconds=wall,
            severe_count=self.digest.severe_count,
            warning_count=self.digest.warning_count,
        )


def run_variant(
    cfg: RunConfig,
    bus: EventBus,
    policy_factory: Callable[[], ControlPolicy],
    label: str,
    idf_override: str | Path | None = None,
) -> EngineResult:
    """Run one IDF variant (used by the ECM verification pass)."""
    engine = EnergyPlusEngine(cfg, bus, idf_override=idf_override, digest=LogDigest())
    policy = policy_factory()
    try:
        result = engine.run(policy, label)
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()
    bus.publish(
        ECM,
        label=label,
        ok=result.ok,
        kwh=round(result.kpi.total_kwh, 3),
        idf=result.idf_used,
        log=engine.digest.for_llm(600),
    )
    return result
