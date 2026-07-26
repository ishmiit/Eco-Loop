"""Physics and KPI accounting."""

from __future__ import annotations

import math

import pytest
from conftest import make_zone

from ecoloop.config import ComfortTargets, GridTargets
from ecoloop.metrics import (
    DEFAULT_PLANT,
    KPIAccumulator,
    co2_next_ppm,
    compare,
    fanger_pmv,
    grid_signal,
    pmv_ppd,
    summer_clo,
)
from ecoloop.telemetry import GridSignal, Snapshot


class TestFangerPMV:
    """Validated against the worked examples in ISO 7730 Annex D."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (dict(ta=22, tr=22, vel=0.1, rh=60, met=1.2, clo=0.5), -0.75),
            (dict(ta=27, tr=27, vel=0.1, rh=60, met=1.2, clo=0.5), 0.77),
            (dict(ta=27, tr=27, vel=0.3, rh=60, met=1.2, clo=0.5), 0.44),
            (dict(ta=23.5, tr=25.5, vel=0.1, rh=60, met=1.2, clo=0.5), -0.01),
        ],
    )
    def test_iso_7730_reference_cases(self, kwargs: dict, expected: float) -> None:
        assert fanger_pmv(**kwargs) == pytest.approx(expected, abs=0.06)

    def test_warmer_air_raises_pmv(self) -> None:
        cool = fanger_pmv(ta=23, rh=55, met=1.2, clo=0.5)
        warm = fanger_pmv(ta=27, rh=55, met=1.2, clo=0.5)
        assert warm > cool

    def test_air_movement_lowers_pmv(self) -> None:
        still = fanger_pmv(ta=27, vel=0.1, rh=55, met=1.2, clo=0.5)
        breezy = fanger_pmv(ta=27, vel=0.8, rh=55, met=1.2, clo=0.5)
        assert breezy < still - 0.2

    def test_activity_raises_pmv(self) -> None:
        """The reason the production hall needs its own PMV limit."""
        seated = fanger_pmv(ta=25, vel=0.4, rh=55, met=1.05, clo=0.5)
        working = fanger_pmv(ta=25, vel=0.4, rh=55, met=1.7, clo=0.5)
        assert working > seated + 0.5

    def test_clipped_to_scale(self) -> None:
        assert fanger_pmv(ta=45, rh=90, met=2.0, clo=1.0) <= 3.0
        assert fanger_pmv(ta=-5, rh=20, met=0.8, clo=0.3) >= -3.0

    def test_no_nan_across_a_wide_sweep(self) -> None:
        for ta in range(10, 45):
            for vel in (0.0, 0.15, 0.5, 1.2):
                value = fanger_pmv(ta=float(ta), vel=vel, rh=50, met=1.4, clo=0.5)
                assert math.isfinite(value)

    def test_ppd_is_minimised_at_neutral(self) -> None:
        assert pmv_ppd(0.0) == pytest.approx(5.0, abs=0.2)
        assert pmv_ppd(1.0) > pmv_ppd(0.5) > pmv_ppd(0.0)

    def test_seasonal_clothing(self) -> None:
        assert summer_clo(5) == 0.5
        assert summer_clo(1) == 0.9


class TestCO2Balance:
    def test_empty_zone_decays_to_outdoor(self) -> None:
        ppm = 1200.0
        for _ in range(240):
            ppm = co2_next_ppm(ppm, occupants=0, outdoor_air_m3_s=0.1, volume_m3=288, dt_s=60)
        # 4 h at 0.1 m3/s through 288 m3 is exp(-5) of the initial 780 ppm excess.
        assert ppm == pytest.approx(420.0 + 780.0 * math.exp(-5.0), abs=1.0)
        for _ in range(600):
            ppm = co2_next_ppm(ppm, occupants=0, outdoor_air_m3_s=0.1, volume_m3=288, dt_s=60)
        assert ppm == pytest.approx(420.0, abs=0.5)

    def test_occupancy_raises_co2_to_a_steady_state(self) -> None:
        ppm = 420.0
        for _ in range(600):
            ppm = co2_next_ppm(ppm, occupants=12, outdoor_air_m3_s=0.134, volume_m3=288, dt_s=60)
        # 12 people at 0.0052 L/s each into 0.134 m3/s of outdoor air.
        expected = 420.0 + (12 * 5.2e-6 / 0.134) * 1e6
        assert ppm == pytest.approx(expected, rel=0.02)

    def test_less_ventilation_means_more_co2(self) -> None:
        full = co2_next_ppm(600, 10, 0.13, 288, 900)
        cut = co2_next_ppm(600, 10, 0.05, 288, 900)
        assert cut > full

    def test_zero_ventilation_is_stable(self) -> None:
        ppm = co2_next_ppm(600, 10, 0.0, 288, 900)
        assert math.isfinite(ppm) and ppm > 600

    def test_large_timestep_stays_bounded(self) -> None:
        """The exponential integrator must not overshoot on a long step."""
        ppm = co2_next_ppm(420, 12, 0.134, 288, dt_s=36000)
        steady = 420.0 + (12 * 5.2e-6 / 0.134) * 1e6
        assert ppm == pytest.approx(steady, rel=0.01)


class TestPlantModel:
    def test_cop_conversion(self) -> None:
        cool, heat = DEFAULT_PLANT.electric_w(3200.0, 0.0)
        assert cool == pytest.approx(1000.0)
        assert heat == 0.0

    def test_negative_load_does_not_produce_power(self) -> None:
        cool, heat = DEFAULT_PLANT.electric_w(-500.0, -500.0)
        assert cool == 0.0 and heat == 0.0

    def test_fan_power_has_a_floor_and_a_ceiling(self) -> None:
        idle = DEFAULT_PLANT.fan_power_w(1.0, enabled=True, design_flow_m3_s=0.95)
        flat_out = DEFAULT_PLANT.fan_power_w(1e7, enabled=True, design_flow_m3_s=0.95)
        assert idle == pytest.approx(0.25 * 0.95 * 1000.0)
        assert flat_out == pytest.approx(0.95 * 1000.0)

    def test_fan_off_when_disabled(self) -> None:
        assert DEFAULT_PLANT.fan_power_w(5000.0, enabled=False, design_flow_m3_s=0.95) == 0.0


class TestGridSignal:
    def test_peak_window_and_tariff(self) -> None:
        targets = GridTargets()
        peak = grid_signal(19, targets)
        assert peak.peak_window is True
        assert peak.tariff_inr_per_kwh == targets.tariff_peak

    def test_midday_is_the_cleanest(self) -> None:
        targets = GridTargets()
        midday = grid_signal(12, targets).carbon_g_per_kwh
        evening = grid_signal(19, targets).carbon_g_per_kwh
        assert midday < evening

    def test_hour_wraps(self) -> None:
        assert grid_signal(25, GridTargets()).carbon_g_per_kwh == grid_signal(1, GridTargets()).carbon_g_per_kwh


class TestKPIAccumulator:
    def _snap(self, **kw) -> Snapshot:
        defaults = dict(
            timestep_hours=0.25,
            hvac_cooling_elec_w=4000.0,
            fan_elec_w=500.0,
            lights_elec_w=1000.0,
            equip_elec_w=2000.0,
            grid=GridSignal(700.0, 8.0, False),
            occupied=True,
            zones=[make_zone()],
        )
        defaults.update(kw)
        return Snapshot(**defaults)

    def test_energy_integration(self) -> None:
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        for _ in range(4):
            acc.add(self._snap())
        kpi = acc.result()
        assert kpi.total_kwh == pytest.approx(7.5, rel=1e-6)      # 7500 W for 1 h
        assert kpi.hvac_kwh == pytest.approx(4.5, rel=1e-6)
        assert kpi.sim_hours == pytest.approx(1.0)
        assert kpi.cost_inr == pytest.approx(7.5 * 8.0)
        assert kpi.carbon_kg == pytest.approx(7.5 * 0.7)

    def test_peak_demand_is_a_maximum(self) -> None:
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        acc.add(self._snap(hvac_cooling_elec_w=1000.0))
        acc.add(self._snap(hvac_cooling_elec_w=9000.0))
        acc.add(self._snap(hvac_cooling_elec_w=2000.0))
        assert acc.result().peak_demand_w == pytest.approx(12500.0)

    def test_per_zone_pmv_limit_is_respected(self) -> None:
        """A 1.1 PMV in the production hall is fine; in the office it is not."""
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        acc.add(self._snap(zones=[make_zone("PROD_HALL", pmv=1.1, pmv_limit=1.5)]))
        assert acc.result().pmv_exceedance_zone_hours == 0.0

        acc2 = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        acc2.add(self._snap(zones=[make_zone("OFFICE", pmv=1.1, pmv_limit=0.8)]))
        assert acc2.result().pmv_exceedance_zone_hours == pytest.approx(0.25)

    def test_unoccupied_hours_do_not_count_against_comfort(self) -> None:
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        acc.add(self._snap(occupied=False, zones=[make_zone(temp=31.0, pmv=2.5, occupants=0.0)]))
        kpi = acc.result()
        assert kpi.pmv_exceedance_zone_hours == 0.0
        assert kpi.occupied_hours == 0.0

    def test_co2_exceedance(self) -> None:
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        acc.add(self._snap(zones=[make_zone(co2=1250.0)]))
        assert acc.result().co2_exceedance_zone_hours == pytest.approx(0.25)

    def test_peak_window_energy_is_tracked_separately(self) -> None:
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        acc.add(self._snap(grid=GridSignal(760.0, 11.5, True)))
        acc.add(self._snap(grid=GridSignal(340.0, 8.0, False)))
        kpi = acc.result()
        assert kpi.peak_window_kwh == pytest.approx(1.875)
        assert kpi.total_kwh == pytest.approx(3.75)

    def test_latency_percentiles(self) -> None:
        acc = KPIAccumulator(ComfortTargets(), "t", "surrogate")
        for value in range(1, 101):
            acc.add_decision("llm", float(value))
        kpi = acc.result()
        assert kpi.decisions == 100
        assert kpi.llm_decisions == 100
        assert kpi.mean_decision_latency_ms == pytest.approx(50.5)
        assert kpi.p95_decision_latency_ms == pytest.approx(95.0)


class TestCompare:
    def _kpi(self, total: float, hvac: float, pmv_exc: float = 0.0, co2_exc: float = 0.0):
        acc = KPIAccumulator(ComfortTargets(), "x", "surrogate")
        kpi = acc.result()
        kpi.total_kwh = total
        kpi.hvac_kwh = hvac
        kpi.pmv_exceedance_zone_hours = pmv_exc
        kpi.co2_exceedance_zone_hours = co2_exc
        return kpi

    def test_percentages(self) -> None:
        table = compare(self._kpi(100.0, 70.0), self._kpi(85.0, 55.0))
        assert table["total_kwh"]["pct"] == pytest.approx(15.0)
        assert table["hvac_kwh"]["pct"] == pytest.approx(21.43, abs=0.01)
        assert table["total_kwh"]["saved_kwh"] == pytest.approx(15.0)

    def test_regression_is_reported_as_negative(self) -> None:
        table = compare(self._kpi(100.0, 70.0), self._kpi(110.0, 80.0))
        assert table["total_kwh"]["pct"] < 0

    def test_comfort_preserved_flag_is_strict(self) -> None:
        better = compare(self._kpi(100, 70, pmv_exc=5.0), self._kpi(80, 55, pmv_exc=1.0))
        assert better["comfort"]["comfort_preserved"] is True
        worse = compare(self._kpi(100, 70, pmv_exc=1.0), self._kpi(80, 55, pmv_exc=2.0))
        assert worse["comfort"]["comfort_preserved"] is False
        # A CO2 regression alone is enough to fail the flag.
        iaq = compare(self._kpi(100, 70, co2_exc=0.0), self._kpi(80, 55, co2_exc=0.5))
        assert iaq["comfort"]["comfort_preserved"] is False

    def test_zero_baseline_does_not_divide_by_zero(self) -> None:
        table = compare(self._kpi(0.0, 0.0), self._kpi(0.0, 0.0))
        assert table["total_kwh"]["pct"] == 0.0
