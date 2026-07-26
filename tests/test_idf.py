"""IDF parsing, round-tripping and the ECM library.

Round-trip fidelity is load-bearing: the agent's generated variants are a
submission deliverable, and a diff between ``baseline.idf`` and a variant has to
show only what the agent actually changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ecoloop.config import MODELS_DIR
from ecoloop.sim.idf import ECM_DOCS, ECM_LIBRARY, IDF, apply_ecms

BASELINE = MODELS_DIR / "baseline.idf"

SAMPLE = """\
!- a header comment

Version, 25.2;

Timestep, 4;

ScheduleTypeLimits, Fraction, 0.0, 1.0, CONTINUOUS;

Schedule:Constant, CSP_TEST, Temperature, 24.0;

Zone,
    MY_ZONE,                 !- Name
    0.0,                     !- Direction of Relative North {deg}
    0.0, 0.0, 0.0,           !- X,Y,Z Origin {m}
    1,                       !- Type
    1,                       !- Multiplier
    3.6,                     !- Ceiling Height {m}
    288.0;                   !- Volume {m3}
"""


class TestParser:
    def test_parses_objects_and_fields(self) -> None:
        idf = IDF.parse(SAMPLE)
        assert [o.obj_class for o in idf.objects] == [
            "Version", "Timestep", "ScheduleTypeLimits", "Schedule:Constant", "Zone",
        ]
        zone = idf.require("Zone", "MY_ZONE")
        assert zone.fields == ["MY_ZONE", "0.0", "0.0", "0.0", "0.0", "1", "1", "3.6", "288.0"]

    def test_no_phantom_empty_fields(self) -> None:
        """Padding between a value and its `!-` comment is not another field."""
        idf = IDF.parse(SAMPLE)
        for obj in idf.objects:
            assert "" not in obj.fields, f"{obj.obj_class} has an empty field: {obj.fields}"

    def test_multiple_fields_on_one_line(self) -> None:
        idf = IDF.parse(SAMPLE)
        limits = idf.of_class("ScheduleTypeLimits")[0]
        assert limits.fields == ["Fraction", "0.0", "1.0", "CONTINUOUS"]

    def test_comments_are_preserved(self) -> None:
        idf = IDF.parse(SAMPLE)
        zone = idf.require("Zone", "MY_ZONE")
        assert "Name" in zone.comments[0]
        assert "Volume" in zone.comments[-1]

    def test_round_trip_is_stable(self) -> None:
        once = IDF.parse(SAMPLE).to_text()
        twice = IDF.parse(once).to_text()
        assert once == twice

    def test_unterminated_object_is_kept(self) -> None:
        idf = IDF.parse("Timestep, 6")
        assert idf.of_class("Timestep")[0].fields == ["6"]

    def test_get_is_case_insensitive(self) -> None:
        idf = IDF.parse(SAMPLE)
        assert idf.get("zone", "my_zone") is not None
        assert idf.get("Zone", "nope") is None

    def test_require_raises_for_missing(self) -> None:
        with pytest.raises(KeyError):
            IDF.parse(SAMPLE).require("Zone", "ABSENT")

    def test_set_field_extends(self) -> None:
        idf = IDF.parse(SAMPLE)
        obj = idf.of_class("Timestep")[0]
        obj.set_field(4, "x")
        assert obj.fields == ["4", "", "", "", "x"]


class TestBaselineModel:
    def test_committed_model_parses(self) -> None:
        idf = IDF.load(BASELINE)
        assert len(idf.objects) > 100

    def test_the_agent_control_surface_exists(self) -> None:
        """The nine actuated schedules are the entire control interface; if one
        is renamed, the EnergyPlus engine silently loses an actuator."""
        idf = IDF.load(BASELINE)
        schedules = idf.schedule_constants()
        for name in (
            "CSP_OFFICE", "CSP_PROD", "CSP_STORE",
            "HSP_OFFICE", "HSP_PROD", "HSP_STORE",
            "OAF_OFFICE", "OAF_PROD", "OAF_STORE",
        ):
            assert name in schedules, f"missing actuated schedule {name}"

    def test_zone_names_match_the_engine_metadata(self) -> None:
        from ecoloop.sim.base import DEFAULT_ZONES

        idf = IDF.load(BASELINE)
        names = {o.name for o in idf.of_class("Zone")}
        for spec in DEFAULT_ZONES:
            assert spec.name in names
            assert idf.get("Schedule:Constant", spec.cooling_sp_schedule) is not None
            assert idf.get("Schedule:Constant", spec.heating_sp_schedule) is not None
            assert idf.get("Schedule:Constant", spec.oa_schedule) is not None
            assert idf.get("People", spec.people_object) is not None

    def test_fanger_comfort_is_requested_for_every_people_object(self) -> None:
        idf = IDF.load(BASELINE)
        people = idf.of_class("People")
        assert len(people) == 3
        for obj in people:
            assert any("FANGER" == f.upper() for f in obj.fields), obj.name

    def test_co2_balance_is_enabled(self) -> None:
        idf = IDF.load(BASELINE)
        balance = idf.of_class("ZoneAirContaminantBalance")
        assert balance and balance[0].fields[0].lower() == "yes"

    def test_run_period_and_timestep_rewrite(self) -> None:
        idf = IDF.load(BASELINE)
        idf.set_run_period(6, 1, 6, 4)
        idf.set_timestep(6)
        rp = idf.of_class("RunPeriod")[0]
        assert rp.fields[1:3] == ["6", "1"]
        assert rp.fields[4:6] == ["6", "4"]
        assert idf.of_class("Timestep")[0].fields[0] == "6"

    def test_baseline_round_trips(self) -> None:
        once = IDF.load(BASELINE).to_text()
        assert IDF.parse(once).to_text() == once


class TestECMLibrary:
    def test_every_ecm_is_documented(self) -> None:
        assert set(ECM_LIBRARY) == set(ECM_DOCS)

    @pytest.mark.parametrize("name", sorted(ECM_LIBRARY))
    def test_each_ecm_applies_to_the_baseline(self, name: str) -> None:
        idf = IDF.load(BASELINE)
        before = idf.to_text()
        results = apply_ecms(idf, [{"ecm": name}])
        assert len(results) == 1
        assert results[0].ok, f"{name}: {results[0].detail}"
        assert idf.to_text() != before, f"{name} reported success but changed nothing"

    def test_unknown_ecm_is_reported_not_raised(self) -> None:
        idf = IDF.load(BASELINE)
        results = apply_ecms(idf, [{"ecm": "teleport_the_building"}])
        assert results[0].ok is False
        assert "unknown ECM" in results[0].detail

    def test_parameters_are_clamped(self) -> None:
        idf = IDF.load(BASELINE)
        apply_ecms(idf, [{"ecm": "cool_roof", "params": {"solar_absorptance": -5}}])
        material = idf.require("Material", "ROOF_SCREED_40MM")
        assert float(material.field(7)) == pytest.approx(0.15)   # clamped to the floor

    def test_malformed_parameters_fall_back_to_defaults(self) -> None:
        idf = IDF.load(BASELINE)
        results = apply_ecms(idf, [{"ecm": "led_retrofit", "params": {"reduction": "lots"}}])
        assert results[0].ok
        assert float(idf.require("Lights", "PROD_LIGHTS").field(5)) == pytest.approx(12.0 * 0.65)

    def test_applying_twice_is_detected(self) -> None:
        idf = IDF.load(BASELINE)
        assert apply_ecms(idf, [{"ecm": "roof_insulation"}])[0].ok
        second = apply_ecms(idf, [{"ecm": "roof_insulation"}])[0]
        assert second.ok is False
        assert "already" in second.detail

    def test_roof_insulation_inserts_under_the_screed(self) -> None:
        idf = IDF.load(BASELINE)
        apply_ecms(idf, [{"ecm": "roof_insulation"}])
        layers = idf.require("Construction", "ROOF_UNINSULATED").fields
        assert layers == [
            "ROOF_UNINSULATED", "ROOF_SCREED_40MM", "ROOF_INSUL_50MM", "RCC_150MM", "PLASTER_12MM",
        ]

    def test_heat_recovery_writes_the_right_fields(self) -> None:
        idf = IDF.load(BASELINE)
        apply_ecms(idf, [{"ecm": "heat_recovery", "params": {"sensible_effectiveness": 0.75}}])
        system = idf.require("ZoneHVAC:IdealLoadsAirSystem", "PROD_IDEAL_LOADS")
        assert system.field(24) == "Enthalpy"
        assert float(system.field(25)) == pytest.approx(0.75)

    def test_all_ecms_together_still_round_trip(self) -> None:
        idf = IDF.load(BASELINE)
        results = apply_ecms(idf, [{"ecm": name} for name in sorted(ECM_LIBRARY)])
        assert all(r.ok for r in results)
        text = idf.to_text()
        assert IDF.parse(text).to_text() == text

    def test_a_broken_ecm_does_not_abort_the_batch(self) -> None:
        idf = IDF.load(BASELINE)
        results = apply_ecms(
            idf, [{"ecm": "nonsense"}, {"ecm": "cool_roof"}, {"ecm": "also_nonsense"}]
        )
        assert [r.ok for r in results] == [False, True, False]

    def test_save_and_reload(self, tmp_path: Path) -> None:
        idf = IDF.load(BASELINE)
        apply_ecms(idf, [{"ecm": "glazing_upgrade"}])
        target = idf.save(tmp_path / "variant.idf")
        assert target.exists()
        reloaded = IDF.load(target)
        assert reloaded.require("Construction", "WINDOW_CLEAR").field(1) == "GLAZING_DGU_LOWE"
