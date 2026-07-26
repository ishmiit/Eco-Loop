"""The safety layer between the model and the building.

This is the single most important file for the "did the AI trade comfort for
energy?" question. **Every** action reaching an actuator passes through
:func:`clamp`, including actions the LLM produced minutes ago that are still
being held, because they are re-clamped against the *current* snapshot on every
timestep. Consequences:

* A hallucinated set-point (``"cooling_setpoint_c": 45``) becomes a legal one.
* An action that was safe when it was chosen but has since become unsafe
  (occupants arrived, CO2 climbed, PMV drifted) is corrected without waiting
  for the next model call.
* Comfort limits hold even if the LLM is offline, wedged or adversarial —
  the guarantee is structural, not behavioural.

Every intervention is recorded in ``ControlAction.clamped`` and surfaced in the
dashboard, so the honest story of "how often did the guardrail save us" is
visible rather than hidden.
"""

from __future__ import annotations

import math
from typing import Any

from ..config import ComfortTargets
from ..metrics import fanger_pmv
from ..telemetry import ControlAction, Snapshot, ZoneState

OCCUPANCY_EPS = 0.05
# Safety margin held back from each zone's PMV limit when computing the cap.
PMV_MARGIN = 0.1
# Fallback dPMV/dT if the local derivative cannot be evaluated (K^-1).
DEFAULT_PMV_SENSITIVITY = 0.28


def pmv_sensitivity(zone: ZoneState, delta: float = 0.75) -> float:
    """Local dPMV/dT for this zone's occupants (K^-1).

    Evaluated by central difference on the ISO 7730 model at the zone's own
    metabolic rate, clothing and air speed, which between them move the slope by
    well over half (measured 0.43 K^-1 for the seated office against 0.26 K^-1
    for the 1.7-met production hall — PMV becomes *less* sensitive to air
    temperature as metabolic rate rises, via the ISO scaling factor
    ``0.303*exp(-0.036*M)+0.028``).

    Applied to the *measured* PMV, so the radiant effect of the hot roof — which
    EnergyPlus captures and a bare air-temperature model does not — is preserved.
    """
    try:
        hot = fanger_pmv(
            ta=zone.temp_c + delta, vel=zone.air_velocity_m_s, rh=zone.rh_pct,
            met=zone.met, clo=zone.clo,
        )
        cold = fanger_pmv(
            ta=zone.temp_c - delta, vel=zone.air_velocity_m_s, rh=zone.rh_pct,
            met=zone.met, clo=zone.clo,
        )
    except (ValueError, ZeroDivisionError, OverflowError):
        return DEFAULT_PMV_SENSITIVITY
    slope = (hot - cold) / (2 * delta)
    if not (0.05 <= slope <= 1.0):
        return DEFAULT_PMV_SENSITIVITY
    return slope


def _finite(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def clamp(
    proposed: ControlAction,
    snap: Snapshot,
    comfort: ComfortTargets,
    previous: ControlAction | None = None,
) -> ControlAction:
    """Return a safe action derived from ``proposed``.

    Rules, in order (later rules win — comfort beats energy):

    1. non-finite values fall back to the previous action;
    2. per-zone set-point bounds, chosen by occupancy;
    3. rate limit vs. the set-points actually in force;
    4. minimum dead-band between heating and cooling;
    5. ventilation bounds, then CO2 escalation (IAQ beats energy);
    6. PMV rescue — if an occupied zone is already outside the comfort
       envelope, drive the set-point toward the zone temperature.
    """
    action = ControlAction(
        cooling_setpoint_c=_finite(proposed.cooling_setpoint_c, 24.0),
        heating_setpoint_c=_finite(proposed.heating_setpoint_c, 21.0),
        oa_fraction=_finite(proposed.oa_fraction, 1.0),
        source=proposed.source,
        rationale=proposed.rationale,
        decision_id=proposed.decision_id,
        latency_ms=proposed.latency_ms,
        tool_calls=proposed.tool_calls,
        model=proposed.model,
    )
    notes: list[str] = []

    for zone in snap.zones:
        cool, heat = proposed.setpoints_for(zone.name)
        oa = _finite(
            proposed.zone_overrides.get(zone.name, {}).get("oa_fraction", proposed.oa_fraction),
            1.0,
        )
        prev_cool, prev_heat = (
            previous.setpoints_for(zone.name) if previous else (zone.cooling_setpoint_c, zone.heating_setpoint_c)
        )
        cool = _finite(cool, prev_cool)
        heat = _finite(heat, prev_heat)

        occupied = zone.occupants > OCCUPANCY_EPS
        cool, heat, oa, zone_notes = _clamp_zone(
            zone=zone,
            cool=cool,
            heat=heat,
            oa=oa,
            occupied=occupied,
            comfort=comfort,
            in_force_cool=zone.cooling_setpoint_c,
            in_force_heat=zone.heating_setpoint_c,
        )
        notes.extend(zone_notes)
        action.zone_overrides[zone.name] = {
            "cooling_setpoint_c": round(cool, 2),
            "heating_setpoint_c": round(heat, 2),
            "oa_fraction": round(oa, 3),
        }

    # Keep the scalar fields meaningful (they are what a zone with no override
    # would use): the mean of the per-zone results.
    if action.zone_overrides:
        action.cooling_setpoint_c = round(
            sum(v["cooling_setpoint_c"] for v in action.zone_overrides.values())
            / len(action.zone_overrides),
            2,
        )
        action.heating_setpoint_c = round(
            sum(v["heating_setpoint_c"] for v in action.zone_overrides.values())
            / len(action.zone_overrides),
            2,
        )
        action.oa_fraction = round(
            sum(v["oa_fraction"] for v in action.zone_overrides.values()) / len(action.zone_overrides),
            3,
        )

    # Provenance stays with whoever authored the action; the notes tell the
    # story of what the safety layer had to change about it.
    action.clamped = notes
    return action


def _clamp_zone(
    zone: ZoneState,
    cool: float,
    heat: float,
    oa: float,
    occupied: bool,
    comfort: ComfortTargets,
    in_force_cool: float,
    in_force_heat: float,
) -> tuple[float, float, float, list[str]]:
    notes: list[str] = []

    # (2) hard safety envelope, by occupancy
    cool_lo, cool_hi = comfort.safety_cooling_bounds(occupied)
    heat_lo, heat_hi = comfort.safety_heating_bounds(occupied)
    if cool < cool_lo or cool > cool_hi:
        notes.append(f"{zone.name}: cooling {cool:.1f} -> safety bounds [{cool_lo:.1f},{cool_hi:.1f}]")
        cool = min(max(cool, cool_lo), cool_hi)
    if heat < heat_lo or heat > heat_hi:
        notes.append(f"{zone.name}: heating {heat:.1f} -> safety bounds [{heat_lo:.1f},{heat_hi:.1f}]")
        heat = min(max(heat, heat_lo), heat_hi)

    # (3) rate limit against what is physically in force right now — but only
    # while the operating band has not moved. When occupancy flips, the
    # set-point *should* step: that is what a setback is, and rate-limiting the
    # transition would smear an hour of cooling into an empty building.
    #
    # The test is against the *authority* band rather than the safety band,
    # because the safety band's lower bound is the same in both occupancy
    # states — so a set-point mid-way between them would not register as a
    # transition at all, and the exemption would silently never fire.
    step = comfort.max_step_c
    authority_lo, authority_hi = comfort.cooling_bounds(occupied)
    band_moved = not (authority_lo - 1e-6 <= in_force_cool <= authority_hi + 1e-6)
    if not band_moved:
        if in_force_cool > 0 and abs(cool - in_force_cool) > step:
            limited = in_force_cool + step * (1 if cool > in_force_cool else -1)
            notes.append(
                f"{zone.name}: cooling step {abs(cool - in_force_cool):.1f}K > {step:.1f}K limit"
            )
            cool = min(max(limited, cool_lo), cool_hi)
        if in_force_heat > 0 and abs(heat - in_force_heat) > step and heat_lo <= in_force_heat <= heat_hi:
            limited = in_force_heat + step * (1 if heat > in_force_heat else -1)
            notes.append(f"{zone.name}: heating step limited to {step:.1f}K")
            heat = min(max(limited, heat_lo), heat_hi)

    # (4) dead-band
    if cool - heat < comfort.min_deadband_c:
        heat = cool - comfort.min_deadband_c
        heat = min(max(heat, heat_lo), heat_hi)
        if cool - heat < comfort.min_deadband_c:
            cool = min(heat + comfort.min_deadband_c, cool_hi)
        notes.append(f"{zone.name}: dead-band widened to {comfort.min_deadband_c:.1f}K")

    # (5) ventilation, then IAQ escalation
    if oa < comfort.oa_fraction_min or oa > comfort.oa_fraction_max:
        notes.append(
            f"{zone.name}: OA {oa:.2f} -> bounds "
            f"[{comfort.oa_fraction_min:.2f},{comfort.oa_fraction_max:.2f}]"
        )
        oa = min(max(oa, comfort.oa_fraction_min), comfort.oa_fraction_max)
    if occupied:
        if zone.co2_ppm > comfort.co2_limit_ppm:
            if oa < 1.0:
                notes.append(f"{zone.name}: CO2 {zone.co2_ppm:.0f} ppm over limit -> OA forced to 1.00")
            oa = 1.0
        elif zone.co2_ppm > comfort.co2_limit_ppm - 120 and oa < 0.8:
            notes.append(f"{zone.name}: CO2 {zone.co2_ppm:.0f} ppm approaching limit -> OA >= 0.80")
            oa = 0.8

    # (6) PREDICTIVE PMV cap. The temperature band top is one number for the
    # whole building, but PMV is not: at the same 26.5 C the 1.7-met production
    # hall, the 1.4-met packing area and the seated office are at very different
    # points in their own envelopes. So instead of waiting for a breach, cap the
    # cooling set-point at the temperature where *this* zone would reach *its*
    # PMV limit — linearising PMV about the measured operating point.
    #
    # This is what turns comfort from something repaired after the fact into
    # something that does not happen: it is why band-edge operation shows up as
    # energy saved rather than as comfort hours lost.
    if occupied:
        pmv_limit = zone.pmv_limit or comfort.pmv_limit
        sensitivity = pmv_sensitivity(zone)
        if sensitivity > 1e-3:
            headroom_k = (pmv_limit - PMV_MARGIN - zone.pmv) / sensitivity
            pmv_cap = zone.temp_c + headroom_k
            if pmv_cap < cool - 1e-6:
                target = max(cool_lo, pmv_cap)
                notes.append(
                    f"{zone.name}: PMV cap — measured {zone.pmv:+.2f} vs limit {pmv_limit:.2f} "
                    f"at {zone.temp_c:.1f}C -> cooling {cool:.1f} -> {target:.1f}"
                )
                cool = target

    # (7) rescue, for when the zone is already outside its envelope
    if occupied:
        pmv_limit = zone.pmv_limit or comfort.pmv_limit
        tol = 0.3
        too_warm = zone.pmv > pmv_limit or zone.temp_c > comfort.occupied_temp_max_c + tol
        too_cool = zone.pmv < -pmv_limit or zone.temp_c < comfort.occupied_temp_min_c - tol
        if too_warm:
            target = max(cool_lo, min(cool, zone.temp_c - 0.5))
            if target < cool - 1e-6:
                notes.append(
                    f"{zone.name}: PMV {zone.pmv:+.2f} / {zone.temp_c:.1f}C too warm "
                    f"-> cooling {cool:.1f} -> {target:.1f}"
                )
                cool = target
        elif too_cool:
            target = min(heat_hi, max(heat, zone.temp_c + 0.5))
            if target > heat + 1e-6:
                notes.append(
                    f"{zone.name}: PMV {zone.pmv:+.2f} / {zone.temp_c:.1f}C too cool "
                    f"-> heating {heat:.1f} -> {target:.1f}"
                )
                heat = target
            if cool - heat < comfort.min_deadband_c:
                cool = min(heat + comfort.min_deadband_c, cool_hi)

    return cool, heat, oa, notes


def validate_setpoint_request(
    zone: str,
    cooling: float | None,
    heating: float | None,
    oa_fraction: float | None,
    comfort: ComfortTargets,
) -> dict[str, Any]:
    """Cheap pre-flight check used by the ``set_zone_setpoints`` tool so the
    model gets an immediate, specific complaint instead of a silent clamp."""
    problems: list[str] = []
    if cooling is not None:
        c = _finite(cooling, float("nan"))
        if math.isnan(c):
            problems.append("cooling_setpoint_c is not a number")
        elif not (comfort.unoccupied_heating_min_c <= c <= comfort.unoccupied_cooling_max_c):
            problems.append(
                f"cooling_setpoint_c {c} is outside the physically allowed range "
                f"[{comfort.unoccupied_heating_min_c}, {comfort.unoccupied_cooling_max_c}]"
            )
    if heating is not None:
        h = _finite(heating, float("nan"))
        if math.isnan(h):
            problems.append("heating_setpoint_c is not a number")
        elif not (comfort.unoccupied_heating_min_c <= h <= comfort.heating_setpoint_max_c):
            problems.append(
                f"heating_setpoint_c {h} is outside the allowed range "
                f"[{comfort.unoccupied_heating_min_c}, {comfort.heating_setpoint_max_c}]"
            )
    if oa_fraction is not None:
        o = _finite(oa_fraction, float("nan"))
        if math.isnan(o):
            problems.append("oa_fraction is not a number")
        elif not (0.0 <= o <= 1.0):
            problems.append(f"oa_fraction {o} must be between 0 and 1")
    return {"ok": not problems, "problems": problems}
