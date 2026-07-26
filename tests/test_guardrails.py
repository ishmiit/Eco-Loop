"""The safety layer — the file that decides whether comfort can be traded away.

These tests are the evidence behind "comfort is a structural guarantee, not a
behavioural one": they drive deliberately hostile actions (45 C set-points, NaN,
inverted bands) through ``clamp`` and assert the outcome is always safe.
"""

from __future__ import annotations

import math

import pytest
from conftest import make_zone

from ecoloop.agent.guardrails import DEFAULT_PMV_SENSITIVITY, clamp, pmv_sensitivity
from ecoloop.config import ComfortTargets
from ecoloop.telemetry import ControlAction, Snapshot


def snap_with(*zones) -> Snapshot:
    return Snapshot(month=5, day=12, hour=14, minute=0, weekday=2, zones=list(zones), occupied=True)


def action(cooling: float, heating: float = 20.0, oa: float = 1.0, source: str = "llm") -> ControlAction:
    return ControlAction(
        cooling_setpoint_c=cooling, heating_setpoint_c=heating, oa_fraction=oa, source=source
    )


class TestHostileInput:
    def test_absurdly_high_setpoint_is_clamped(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=25.0, occupants=5.0, pmv=0.3)
        result = clamp(action(45.0), snap_with(zone), comfort)
        applied = result.zone_overrides["OFFICE"]["cooling_setpoint_c"]
        assert applied <= comfort.occupied_temp_max_c
        assert result.clamped

    def test_absurdly_low_setpoint_is_clamped(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=25.0, occupants=5.0, cooling_sp=24.0)
        result = clamp(action(-40.0), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] >= comfort.absolute_cooling_min_c

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "warm"])
    def test_non_numeric_falls_back_to_the_previous_action(self, bad) -> None:
        zone = make_zone(temp=25.0, occupants=5.0, cooling_sp=25.0)
        previous = clamp(action(25.0), snap_with(zone), ComfortTargets())
        result = clamp(
            ControlAction(cooling_setpoint_c=bad, heating_setpoint_c=20.0, oa_fraction=0.6),
            snap_with(zone),
            ComfortTargets(),
            previous=previous,
        )
        value = result.zone_overrides["OFFICE"]["cooling_setpoint_c"]
        assert math.isfinite(value)
        assert ComfortTargets().absolute_cooling_min_c <= value <= 30.0

    def test_inverted_band_is_repaired(self) -> None:
        """Cooling below heating would make the thermostat fight itself."""
        comfort = ComfortTargets()
        zone = make_zone(temp=25.0, occupants=5.0, cooling_sp=25.0, heating_sp=20.0)
        result = clamp(action(cooling=23.0, heating=26.0), snap_with(zone), comfort)
        applied = result.zone_overrides["OFFICE"]
        gap = applied["cooling_setpoint_c"] - applied["heating_setpoint_c"]
        assert gap >= comfort.min_deadband_c - 1e-6

    def test_out_of_range_oa_is_clamped(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(occupants=5.0, co2=600.0)
        low = clamp(action(25.0, oa=-3.0), snap_with(zone), comfort)
        high = clamp(action(25.0, oa=9.0), snap_with(zone), comfort)
        assert low.zone_overrides["OFFICE"]["oa_fraction"] == comfort.oa_fraction_min
        assert high.zone_overrides["OFFICE"]["oa_fraction"] == comfort.oa_fraction_max


class TestComfortProtection:
    def test_co2_over_the_ceiling_forces_full_ventilation(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(occupants=10.0, co2=comfort.co2_limit_ppm + 80)
        result = clamp(action(26.0, oa=0.35), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["oa_fraction"] == 1.0
        assert any("CO2" in note for note in result.clamped)

    def test_co2_approaching_the_ceiling_raises_ventilation(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(occupants=10.0, co2=comfort.co2_limit_ppm - 60)
        result = clamp(action(26.0, oa=0.35), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["oa_fraction"] >= 0.8

    def test_iaq_escalation_does_not_apply_to_an_empty_zone(self) -> None:
        """Nobody is breathing, so ventilating an empty room is pure waste."""
        comfort = ComfortTargets()
        zone = make_zone(occupants=0.0, co2=1400.0, mins_to_occupied=None)
        result = clamp(action(29.0, heating=16.0, oa=0.35), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["oa_fraction"] == comfort.oa_fraction_min

    def test_predictive_pmv_cap_stops_the_setpoint_before_a_breach(self) -> None:
        comfort = ComfortTargets()
        # 0.15 below its limit already: there is almost no headroom left.
        zone = make_zone(temp=25.5, occupants=3.0, pmv=0.95, pmv_limit=1.1, cooling_sp=25.5)
        result = clamp(action(26.5), snap_with(zone), comfort)
        applied = result.zone_overrides["OFFICE"]["cooling_setpoint_c"]
        assert applied < 26.5
        assert any("PMV cap" in note for note in result.clamped)

    def test_pmv_cap_leaves_headroom_alone(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=24.0, occupants=3.0, pmv=0.2, pmv_limit=1.5, cooling_sp=25.5)
        result = clamp(action(26.5), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] == pytest.approx(26.5, abs=0.6)

    def test_rescue_when_already_too_warm(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=28.5, occupants=5.0, pmv=1.6, pmv_limit=0.8, cooling_sp=28.0)
        result = clamp(action(28.0), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] < 28.0

    def test_zones_are_judged_against_their_own_pmv_limits(self) -> None:
        """Identical PMV, different verdicts — this is the whole point of the
        per-zone limits."""
        comfort = ComfortTargets()
        office = make_zone("OFFICE", temp=25.5, occupants=5.0, pmv=1.05, pmv_limit=0.8, cooling_sp=25.5)
        hall = make_zone("PROD_HALL", temp=25.5, occupants=11.0, pmv=1.05, pmv_limit=1.5, cooling_sp=25.5)
        result = clamp(action(26.5), snap_with(office, hall), comfort)
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] < 26.0
        assert result.zone_overrides["PROD_HALL"]["cooling_setpoint_c"] > 25.9


class TestRateLimiting:
    def test_large_step_is_limited(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=25.0, occupants=5.0, cooling_sp=24.0)
        result = clamp(action(26.5), snap_with(zone), comfort)
        applied = result.zone_overrides["OFFICE"]["cooling_setpoint_c"]
        assert applied == pytest.approx(24.0 + comfort.max_step_c, abs=0.01)

    def test_occupancy_transition_is_exempt(self) -> None:
        """A setback is a step change by design; rate-limiting it would smear an
        hour of cooling into an empty building."""
        comfort = ComfortTargets()
        zone = make_zone(temp=26.0, occupants=0.0, cooling_sp=25.0, mins_to_occupied=None)
        result = clamp(action(30.0, heating=16.0), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] == pytest.approx(30.0)

    def test_small_step_passes_untouched(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=25.0, occupants=5.0, pmv=0.2, cooling_sp=25.0, pmv_limit=1.5)
        result = clamp(action(25.5), snap_with(zone), comfort)
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] == pytest.approx(25.5)
        assert not result.clamped


class TestBaselineIsNotQuietlyOptimised:
    """If the safety layer improved the control group, the reported savings
    would be understated — and the comparison would stop being honest."""

    def test_a_reasonable_baseline_action_passes_unchanged(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=24.0, occupants=5.0, pmv=0.3, cooling_sp=24.0, heating_sp=21.0)
        result = clamp(action(24.0, heating=21.0, source="baseline"), snap_with(zone), comfort)
        applied = result.zone_overrides["OFFICE"]
        assert applied["cooling_setpoint_c"] == pytest.approx(24.0)
        assert applied["heating_setpoint_c"] == pytest.approx(21.0)

    def test_baseline_cooling_an_empty_zone_is_not_set_back_for_it(self) -> None:
        comfort = ComfortTargets()
        zone = make_zone(temp=24.0, occupants=0.0, cooling_sp=24.0, mins_to_occupied=None)
        result = clamp(action(24.0, heating=21.0, source="baseline"), snap_with(zone), comfort)
        # 24 C in an empty room wastes energy but is not unsafe, so the safety
        # layer must leave it alone.
        assert result.zone_overrides["OFFICE"]["cooling_setpoint_c"] == pytest.approx(24.0)


class TestProvenanceAndReporting:
    def test_source_survives_clamping(self) -> None:
        result = clamp(action(45.0), snap_with(make_zone(occupants=5.0)), ComfortTargets())
        assert result.source == "llm"

    def test_every_intervention_is_recorded(self) -> None:
        result = clamp(
            action(45.0, heating=44.0, oa=5.0),
            snap_with(make_zone(occupants=5.0, co2=1300.0)),
            ComfortTargets(),
        )
        assert len(result.clamped) >= 2
        assert all(isinstance(note, str) and note for note in result.clamped)

    def test_scalar_fields_summarise_the_zones(self) -> None:
        comfort = ComfortTargets()
        zones = [
            make_zone("OFFICE", occupants=5.0, cooling_sp=25.0, pmv_limit=1.5),
            make_zone("PROD_HALL", occupants=5.0, cooling_sp=25.0, pmv_limit=1.5),
        ]
        result = clamp(action(25.5), snap_with(*zones), comfort)
        values = [v["cooling_setpoint_c"] for v in result.zone_overrides.values()]
        assert result.cooling_setpoint_c == pytest.approx(sum(values) / len(values), abs=0.01)


class TestPMVSensitivity:
    def test_sensitivity_is_positive_and_bounded(self) -> None:
        zone = make_zone(temp=25.0)
        slope = pmv_sensitivity(zone)
        assert 0.05 <= slope <= 1.0

    def test_activity_changes_the_slope(self) -> None:
        """PMV becomes LESS sensitive to air temperature as metabolic rate rises
        (the ISO 7730 factor 0.303*exp(-0.036*M)+0.028 shrinks). Worth pinning:
        it is the opposite of the intuitive guess, and the predictive PMV cap
        divides by this number."""
        seated = make_zone(temp=25.0)
        seated.met = 1.05
        working = make_zone(temp=25.0)
        working.met = 1.7
        assert pmv_sensitivity(working) < pmv_sensitivity(seated)
        assert pmv_sensitivity(seated) == pytest.approx(0.43, abs=0.05)
        assert pmv_sensitivity(working) == pytest.approx(0.26, abs=0.05)

    def test_degenerate_zone_falls_back_to_the_default(self) -> None:
        zone = make_zone(temp=25.0)
        zone.met = 0.0
        zone.clo = 0.0
        assert pmv_sensitivity(zone) == DEFAULT_PMV_SENSITIVITY
