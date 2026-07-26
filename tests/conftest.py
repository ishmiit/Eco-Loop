"""Shared fixtures.

The suite runs without EnergyPlus, without a GPU and without a network: the
surrogate engine stands in for the simulator and the ``mock`` LLM provider for
the model. Tests that genuinely need EnergyPlus are marked and skipped when it
is absent, so ``pytest`` is green on a bare checkout and *more* thorough on a
machine that has the real thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ecoloop.config import LLMConfig, RunConfig  # noqa: E402
from ecoloop.energyplus_locate import find_energyplus  # noqa: E402
from ecoloop.telemetry import GridSignal, Snapshot, ZoneState  # noqa: E402

HAS_ENERGYPLUS = find_energyplus() is not None
requires_energyplus = pytest.mark.skipif(
    not HAS_ENERGYPLUS, reason="EnergyPlus not installed (./scripts/install_energyplus.sh)"
)


@pytest.fixture
def run_config(tmp_path: Path) -> RunConfig:
    cfg = RunConfig(
        run_id="pytest",
        engine="surrogate",
        start_month=5,
        start_day=12,
        end_month=5,
        end_day=12,
        timesteps_per_hour=2,
        decision_interval_min=60,
        agent_mode="sync",
        brain="heuristic",
        llm=LLMConfig(provider="mock", model="mock"),
    )
    # Redirect artifacts into the test's tmp dir.
    import ecoloop.config as config_module

    config_module.ARTIFACTS_DIR = tmp_path / "artifacts"
    cfg.__class__.out_dir = property(lambda self: tmp_path / "artifacts" / self.run_id)
    return cfg


def make_zone(
    name: str = "OFFICE",
    temp: float = 25.0,
    occupants: float = 4.0,
    pmv: float = 0.4,
    co2: float = 700.0,
    pmv_limit: float = 0.8,
    cooling_sp: float = 25.0,
    heating_sp: float = 20.0,
    mins_to_occupied: float | None = 0.0,
) -> ZoneState:
    return ZoneState(
        name=name,
        temp_c=temp,
        rh_pct=55.0,
        co2_ppm=co2,
        occupants=occupants,
        pmv=pmv,
        cooling_setpoint_c=cooling_sp,
        heating_setpoint_c=heating_sp,
        area_m2=80.0,
        pmv_limit=pmv_limit,
        met=1.2,
        clo=0.5,
        air_velocity_m_s=0.4,
        minutes_until_occupied=mins_to_occupied,
        minutes_until_vacant=0.0 if occupants > 0 else None,
    )


@pytest.fixture
def snapshot() -> Snapshot:
    return Snapshot(
        step=10,
        month=5,
        day=12,
        hour=14,
        minute=0,
        weekday=2,
        timestep_hours=0.25,
        outdoor_temp_c=34.0,
        outdoor_rh_pct=62.0,
        solar_w_m2=780.0,
        zones=[
            make_zone("OFFICE", pmv_limit=0.8),
            make_zone("PROD_HALL", temp=25.8, occupants=11.0, pmv=1.1, co2=980.0, pmv_limit=1.5),
            make_zone("PACK_STORE", temp=24.8, occupants=3.0, pmv=0.9, co2=650.0, pmv_limit=1.1),
        ],
        grid=GridSignal(carbon_g_per_kwh=350.0, tariff_inr_per_kwh=8.0, peak_window=False),
        occupied=True,
    )
