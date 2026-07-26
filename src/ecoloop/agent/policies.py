"""The two non-LLM policies.

``BaselinePolicy`` is the control-group: a conventional rule-based BMS. It is
deliberately *reasonable* rather than a straw man — fixed 24 C cooling / 21 C
heating during operating hours, night setback outside them, and full design
ventilation whenever the space is occupied. That is how these buildings are
actually run, and beating a strawman would make the savings number worthless.

``HeuristicPolicy`` is the fallback brain and the LLM's safety net. It encodes
the same supervisory strategies the agent is prompted to reason about
(setback, pre-cooling, peak-window coasting, demand-controlled ventilation) as
deterministic rules. It exists so that:

* a run with ``--brain heuristic`` gives a like-for-like "rules vs. LLM"
  comparison, isolating what the language model actually contributes;
* an LLM timeout, crash or garbage response degrades to competent control
  rather than to nothing.

Both drive the identical actuator path as the LLM, so any measured difference
between runs is control strategy and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..bus import DECISION, EventBus
from ..config import ComfortTargets, GridTargets
from ..telemetry import ControlAction, Snapshot, ZoneState

OCCUPANCY_EPS = 0.05

# Operating hours of the food unit (see models/baseline.idf schedules).
OPERATING_START = 6
OPERATING_END = 19
# How early to begin pre-cooling ahead of a shift. Two hours, because this is a
# heavy masonry building in a hot climate: at 06:00 the slab is still holding
# the previous afternoon's heat, and an hour of lead time only gets the
# set-point down, not the space.
PRE_START_MIN = 120.0


class BaselinePolicy:
    """Fixed-schedule BMS. No feedback beyond the clock."""

    name = "baseline"

    def __init__(
        self,
        cooling_c: float = 24.0,
        heating_c: float = 21.0,
        setback_cooling_c: float = 30.0,
        setback_heating_c: float = 16.0,
        bus: EventBus | None = None,
    ) -> None:
        self.cooling_c = cooling_c
        self.heating_c = heating_c
        self.setback_cooling_c = setback_cooling_c
        self.setback_heating_c = setback_heating_c
        self._id = 0
        self._bus = bus
        self._last_published = ""

    def _publish(self, action: ControlAction, snap: Snapshot) -> None:
        """Emit a decision event when the strategy changes.

        The deterministic arms decide on every timestep, so publishing each one
        would bury the log; publishing on change gives the dashboard and
        decisions.jsonl a readable feed for the ablation arm too.
        """
        if self._bus is None or action.rationale == self._last_published:
            return
        self._last_published = action.rationale
        self._bus.publish(
            DECISION,
            decision_id=action.decision_id,
            clock=snap.clock,
            source=action.source,
            rationale=action.rationale,
            latency_ms=0.0,
            tool_calls=0,
            model=action.source,
            cache_hit=False,
            requested={
                "cooling_setpoint_c": action.cooling_setpoint_c,
                "heating_setpoint_c": action.heating_setpoint_c,
                "oa_fraction": action.oa_fraction,
                "zones": action.zone_overrides,
            },
            applied=action.zone_overrides,
            clamped=[],
        )

    def decide(self, snap: Snapshot) -> ControlAction:
        self._id += 1
        operating = _in_operating_hours(snap)
        cool = self.cooling_c if operating else self.setback_cooling_c
        heat = self.heating_c if operating else self.setback_heating_c
        action = ControlAction(
            cooling_setpoint_c=cool,
            heating_setpoint_c=heat,
            oa_fraction=1.0,
            source="baseline",
            rationale=(
                "Fixed schedule: 24/21 C during operating hours, night setback outside, "
                "design ventilation at all times."
            ),
            decision_id=self._id,
        )
        for zone in snap.zones:
            action.zone_overrides[zone.name] = {
                "cooling_setpoint_c": cool,
                "heating_setpoint_c": heat,
                # A conventional BMS ventilates on the schedule, not on demand.
                "oa_fraction": 1.0 if operating else 0.35,
            }
        self._publish(action, snap)
        return action

    def close(self) -> None:
        return None


@dataclass
class ZoneRecommendation:
    """The deterministic layer's suggestion for one zone at one timestep."""

    zone: str
    cooling_setpoint_c: float
    heating_setpoint_c: float
    oa_fraction: float
    lever: str          # "A" setback, "B" band edge, "C" pre-cool/coast, "F" optimum start
    state: str          # occupied | pre-occupancy | unoccupied
    reason: str

    def as_dict(self) -> dict[str, float]:
        return {
            "cooling_setpoint_c": round(self.cooling_setpoint_c, 2),
            "heating_setpoint_c": round(self.heating_setpoint_c, 2),
            "oa_fraction": round(self.oa_fraction, 3),
        }


def zone_state(zone: ZoneState) -> tuple[bool, bool]:
    """``(occupied, pre_start)`` for one zone.

    A zone counts as occupied if anyone is in it OR the shift pattern says we
    are inside its occupied period. Both halves are needed: at the exact shift
    boundary the schedule reads "in shift" while the sensor still reports zero
    people, and treating that timestep as empty throws away the whole pre-cool
    one step before the shift starts — precisely when it matters.
    """
    in_shift = zone.minutes_until_occupied is not None and zone.minutes_until_occupied <= 0.0
    occupied = zone.occupants > OCCUPANCY_EPS or in_shift
    pre_start = (
        not occupied
        and zone.minutes_until_occupied is not None
        and 0 < zone.minutes_until_occupied <= PRE_START_MIN
    )
    return occupied, pre_start


def recommend(
    zone: ZoneState, snap: Snapshot, comfort: ComfortTargets, grid: GridTargets
) -> ZoneRecommendation:
    """The whole supervisory strategy for one zone, as one function.

    Shared deliberately between two callers:

    * ``HeuristicPolicy`` applies it verbatim — that is the deterministic
      control arm, and the ablation baseline for "what does the LLM add?";
    * the LLM prompt shows it as a **recommended default** the model may accept,
      adjust or override, and ``LLMPolicy`` measures how often it does.

    One implementation means the ablation is exact: the two arms differ only in
    whether a language model gets to disagree.
    """
    occupied, pre_start = zone_state(zone)
    peak_lo, _ = grid.peak_window
    pre_peak = peak_lo - 2 <= snap.hour < peak_lo
    hot_outside = snap.outdoor_temp_c > 30.0

    if pre_start:
        cool_lo, cool_hi = comfort.cooling_bounds(True)
        heat_lo, _ = comfort.heating_bounds(True)
        return ZoneRecommendation(
            zone=zone.name,
            cooling_setpoint_c=cool_hi - 1.0,
            heating_setpoint_c=heat_lo,
            oa_fraction=comfort.oa_fraction_min,
            lever="F",
            state="pre-occupancy",
            reason=f"occupied in {zone.minutes_until_occupied:.0f} min — pre-cool so the shift does not start hot",
        )

    if not occupied:
        cool_lo, cool_hi = comfort.cooling_bounds(False)
        heat_lo, _ = comfort.heating_bounds(False)
        return ZoneRecommendation(
            zone=zone.name,
            cooling_setpoint_c=cool_hi,
            heating_setpoint_c=heat_lo,
            oa_fraction=comfort.oa_fraction_min,
            lever="A",
            state="unoccupied",
            reason="empty and nobody due — full setback, ventilation to minimum",
        )

    cool_lo, cool_hi = comfort.cooling_bounds(True)
    heat_lo, heat_hi = comfort.heating_bounds(True)
    # Sit NEAR THE TOP of the allowed band. The band top is the comfort
    # envelope, so operating there is comfortable by construction; sitting in
    # the middle leaves half the available saving on the table.
    cool = cool_hi - 0.5
    lever, reason = "B", "occupied with comfort margin — sit high in the band"

    if pre_peak and hot_outside:
        # Gentle on purpose: driving to the floor of the band would create a new
        # demand spike an hour before the peak and give back in demand charges
        # what it saves in energy.
        cool = max(cool_lo, cool - 1.0)
        lever, reason = "C", f"pre-cool the thermal mass ahead of the {peak_lo}:00 peak"
    elif snap.grid.peak_window:
        cool = cool_hi
        lever, reason = "C", "coast through the peak-tariff window"
    elif snap.grid.carbon_g_per_kwh < 400:
        cool = max(cool_lo, cool - 0.5)
        lever, reason = "E", "low-carbon window — do thermal work now"

    # Comfort takes precedence over every energy lever above, measured against
    # this zone's own PMV envelope.
    pmv_limit = zone.pmv_limit or comfort.pmv_limit
    if zone.pmv > pmv_limit - 0.15 or zone.temp_c > comfort.occupied_temp_max_c:
        cool = max(cool_lo, min(cool, zone.temp_c - 0.5))
        lever = "B"
        reason = f"comfort margin thin (PMV {zone.pmv:+.2f} of {pmv_limit:.2f}) — hold cooling down"

    return ZoneRecommendation(
        zone=zone.name,
        cooling_setpoint_c=min(max(cool, cool_lo), cool_hi),
        heating_setpoint_c=min(max(heat_lo, heat_lo), heat_hi),
        oa_fraction=_dcv_fraction(zone, comfort),
        lever=lever,
        state="occupied",
        reason=reason,
    )


def recommend_all(
    snap: Snapshot, comfort: ComfortTargets, grid: GridTargets
) -> dict[str, ZoneRecommendation]:
    return {z.name: recommend(z, snap, comfort, grid) for z in snap.zones}


class HeuristicPolicy:
    """Deterministic supervisory control — the fallback brain and the ablation
    arm. Applies :func:`recommend` verbatim, every zone, every decision."""

    name = "heuristic"

    def __init__(
        self, comfort: ComfortTargets, grid: GridTargets, bus: EventBus | None = None
    ) -> None:
        self.comfort = comfort
        self.grid = grid
        self._id = 0
        self._bus = bus
        self._last_published = ""

    _publish = BaselinePolicy._publish

    def decide(self, snap: Snapshot) -> ControlAction:
        self._id += 1
        action = ControlAction(source="heuristic", decision_id=self._id)
        reasons: list[str] = []
        for zone in snap.zones:
            rec = recommend(zone, snap, self.comfort, self.grid)
            action.zone_overrides[zone.name] = rec.as_dict()
            reasons.append(f"{zone.name} (lever {rec.lever}): {rec.reason}")
        first = next(iter(action.zone_overrides.values()), None)
        if first:
            action.cooling_setpoint_c = first["cooling_setpoint_c"]
            action.heating_setpoint_c = first["heating_setpoint_c"]
            action.oa_fraction = first["oa_fraction"]
        action.rationale = "; ".join(reasons[:3]) or "hold"
        self._publish(action, snap)
        return action

    def close(self) -> None:
        return None


def _dcv_fraction(zone: ZoneState, comfort: ComfortTargets) -> float:
    """Ventilate to the CO2 target, not to the design flow."""
    headroom = comfort.co2_limit_ppm - zone.co2_ppm
    if headroom < 60:
        return comfort.oa_fraction_max
    if headroom < 180:
        return 0.8
    if headroom < 320:
        return 0.6
    return max(comfort.oa_fraction_min, 0.45)


def _in_operating_hours(snap: Snapshot) -> bool:
    """Mon-Fri 06:00-19:00, Saturday half day, Sunday closed — matching the
    occupancy schedules in models/baseline.idf. Keeping the baseline honest
    here matters: a BMS left running on a Sunday would inflate the savings."""
    if snap.weekday >= 7:                       # Sunday
        return False
    end = 15 if snap.weekday == 6 else OPERATING_END
    return OPERATING_START <= snap.hour < end
