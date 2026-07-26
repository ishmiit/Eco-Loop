"""End-to-end: a whole experiment, artifacts and all.

These run on the **surrogate** engine with the **mock** LLM provider, which is
the configuration a reviewer on a bare checkout gets. They are the evidence
behind "it runs end to end without EnergyPlus, a GPU or a network".

The EnergyPlus-backed equivalents are marked and skip when it is absent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from conftest import requires_energyplus

from ecoloop.bus import EventBus, read_events
from ecoloop.config import LLMConfig, RunConfig
from ecoloop.logs import LogDigest, classify, fingerprint
from ecoloop.orchestrator import Orchestrator, select_engine
from ecoloop.weather import EPW, resolve_epw, synthetic_record


def make_config(tmp_path: Path, **overrides) -> RunConfig:
    import ecoloop.config as config_module

    config_module.ARTIFACTS_DIR = tmp_path / "artifacts"
    RunConfig.out_dir = property(  # type: ignore[assignment]
        lambda self: config_module.ARTIFACTS_DIR / self.run_id
    )
    kwargs = dict(
        run_id="e2e",
        engine="surrogate",
        start_month=5,
        start_day=12,
        end_month=5,
        end_day=12,
        timesteps_per_hour=2,
        decision_interval_min=120,
        agent_mode="sync",
        brain="heuristic",
        llm=LLMConfig(provider="mock", model="mock"),
    )
    kwargs.update(overrides)
    return RunConfig(**kwargs)


@pytest.fixture(scope="module")
def outcome(tmp_path_factory) -> tuple[dict, RunConfig]:
    """One surrogate experiment, shared by the whole class — it takes a second,
    and re-running it per test would be pure waste."""
    tmp_path = tmp_path_factory.mktemp("run")
    cfg = make_config(tmp_path)
    results = Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
    return results, cfg


class TestSurrogateExperiment:
    def test_both_arms_succeed(self, outcome) -> None:
        results, _ = outcome
        assert results["baseline"]["ok"], results["baseline"]["error"]
        assert results["ai"]["ok"], results["ai"]["error"]
        assert results["engine"] == "surrogate"

    def test_both_arms_ran_the_same_number_of_timesteps(self, outcome) -> None:
        """Different step counts would make the comparison meaningless."""
        results, _ = outcome
        assert results["baseline"]["steps"] == results["ai"]["steps"] == 48

    def test_energy_is_physically_plausible(self, outcome) -> None:
        results, _ = outcome
        for arm in ("baseline", "ai"):
            kpi = results[arm]["kpi"]
            # 240 m2 hot-humid premises, one day: tens to low hundreds of kWh.
            assert 20.0 < kpi["total_kwh"] < 2000.0, arm
            assert kpi["hvac_kwh"] <= kpi["total_kwh"] + 1e-6
            assert kpi["peak_demand_w"] > 0
            assert kpi["sim_hours"] == pytest.approx(24.0)

    def test_the_agent_arm_beats_the_baseline(self, outcome) -> None:
        results, _ = outcome
        assert results["savings"]["total_kwh"]["pct"] > 0

    def test_comfort_is_not_traded_away(self, outcome) -> None:
        results, _ = outcome
        assert results["savings"]["comfort"]["comfort_preserved"] is True

    def test_the_guardrail_did_not_optimise_the_baseline(self, outcome) -> None:
        """The control group must reach the actuators as authored.

        This is the load-bearing fairness check: if the safety layer set the
        baseline back when a zone emptied, the baseline would silently become a
        smarter controller and the reported savings would be understated.

        Asserted on the *applied* set-points rather than on an intervention
        count — the rate limit legitimately touches both arms at schedule
        transitions, so counting adjustments would be the wrong test.
        """
        _, cfg = outcome
        events = read_events(cfg.out_dir / "events.jsonl", ["telemetry"])
        baseline = [e["snapshot"] for e in events if e["label"] == "baseline"]
        # 10:00-17:00 on a weekday is well inside operating hours and past any
        # post-transition ramp, so BaselinePolicy's fixed 24 C must be intact.
        settled = [s for s in baseline if 10 <= s["hour"] <= 17 and s["weekday"] <= 5]
        assert settled, "no settled operating-hour timesteps to check"
        for snap in settled:
            for zone in snap["zones"]:
                assert zone["cooling_setpoint_c"] == pytest.approx(24.0, abs=0.01), (
                    f"the safety layer moved the baseline at {snap['clock']} in {zone['name']}"
                )

    # -- artifacts ---------------------------------------------------------
    def test_every_promised_artifact_exists(self, outcome) -> None:
        _, cfg = outcome
        for name in (
            "manifest.json",
            "results.json",
            "savings.csv",
            "telemetry_ai.csv",
            "telemetry_baseline.csv",
            "events.jsonl",
            "decisions.jsonl",
        ):
            path = cfg.out_dir / name
            assert path.exists() and path.stat().st_size > 0, name

    def test_telemetry_csv_is_wellformed(self, outcome) -> None:
        _, cfg = outcome
        with (cfg.out_dir / "telemetry_ai.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 48
        for column in (
            "clock", "total_elec_w", "cum_kwh", "control_source",
            "OFFICE_temp_c", "PROD_HALL_pmv", "PACK_STORE_co2_ppm",
        ):
            assert column in rows[0], column
        assert float(rows[-1]["cum_kwh"]) > float(rows[0]["cum_kwh"])

    def test_savings_csv_matches_results_json(self, outcome) -> None:
        results, cfg = outcome
        with (cfg.out_dir / "savings.csv").open() as fh:
            rows = {r["metric"]: r for r in csv.DictReader(fh)}
        assert float(rows["total_kwh"]["pct_reduction"]) == pytest.approx(
            results["savings"]["total_kwh"]["pct"]
        )

    def test_manifest_reproduces_the_run(self, outcome) -> None:
        _, cfg = outcome
        manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
        assert manifest["engine"] == "surrogate"
        assert manifest["window"]["timesteps_per_hour"] == 2
        assert manifest["comfort"]["co2_limit_ppm"] > 0
        assert "version" in manifest

    def test_event_stream_carries_every_kind(self, outcome) -> None:
        _, cfg = outcome
        events = read_events(cfg.out_dir / "events.jsonl")
        kinds = {e["kind"] for e in events}
        assert {"telemetry", "status", "kpi"} <= kinds
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs), "event sequence must be monotonic"

    def test_telemetry_events_are_labelled_per_arm(self, outcome) -> None:
        _, cfg = outcome
        events = read_events(cfg.out_dir / "events.jsonl", ["telemetry"])
        labels = {e["label"] for e in events}
        assert labels == {"baseline", "ai"}


class TestLLMBrainOnTheSurrogate:
    """The mock provider exercises the whole agent path — tool registry, tool
    handler, guardrail, decision events — without a model server."""

    def test_run_completes_and_the_agent_decided(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, run_id="mockllm", brain="llm")
        results = Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        assert results["ai"]["ok"]
        agent = results["agent"]
        assert agent["decisions"] > 0
        assert agent["llm_decisions"] > 0
        assert agent["tool_calls"] >= agent["llm_decisions"]

    def test_decisions_are_logged_with_rationales(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, run_id="mockllm2", brain="llm")
        Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        decisions = [
            json.loads(line)
            for line in (cfg.out_dir / "decisions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert decisions
        # The log covers both arms: the deterministic baseline publishes on
        # strategy change, the agent publishes every decision.
        sources = {d["source"] for d in decisions}
        assert "llm" in sources
        assert sources <= {"llm", "heuristic", "external", "baseline"}
        for decision in decisions:
            assert decision["rationale"]
            assert decision["applied"]

    def test_live_mirror_is_written_for_out_of_process_clients(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, run_id="mockllm3", brain="llm")
        Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        live = cfg.out_dir / "live"
        assert (live / "state.json").exists()
        assert (live / "history.jsonl").exists()
        state = json.loads((live / "state.json").read_text())
        assert state["compact"]["zones"]

    def test_an_external_mcp_client_can_read_the_run(self, tmp_path: Path) -> None:
        from ecoloop.agent.context import FileContext
        from ecoloop.agent.tools import build_registry

        cfg = make_config(tmp_path, run_id="mockllm4", brain="llm")
        Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        registry = build_registry(FileContext(cfg.out_dir))
        state = registry.call("get_building_state", {})
        assert state.ok and state.payload["zones"]
        history = registry.call("get_recent_history", {"minutes": 240, "metric": "co2"})
        assert history.ok and history.payload["points"]


class TestEngineSelection:
    def test_surrogate_is_always_available(self) -> None:
        assert select_engine(RunConfig(engine="surrogate")) == "surrogate"

    def test_auto_prefers_energyplus_when_present(self) -> None:
        from ecoloop.energyplus_locate import find_energyplus

        expected = "energyplus" if find_energyplus() else "surrogate"
        assert select_engine(RunConfig(engine="auto")) == expected

    def test_explicit_energyplus_without_an_install_fails_loudly(self, monkeypatch) -> None:
        """Silently falling back would let a reviewer attribute surrogate numbers
        to EnergyPlus."""
        import ecoloop.orchestrator as orchestrator

        monkeypatch.setattr(orchestrator, "find_energyplus", lambda: None)
        with pytest.raises(RuntimeError, match="EnergyPlus"):
            select_engine(RunConfig(engine="energyplus"))


@requires_energyplus
class TestEnergyPlusExperiment:
    def test_a_real_simulation_closes_the_loop(self, tmp_path: Path) -> None:
        cfg = make_config(
            tmp_path, run_id="ep", engine="energyplus", timesteps_per_hour=4, brain="heuristic"
        )
        results = Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        assert results["engine"] == "energyplus"
        assert results["baseline"]["ok"] and results["ai"]["ok"]
        assert results["baseline"]["steps"] == results["ai"]["steps"] == 96
        assert results["baseline"]["severe_count"] == 0
        assert results["savings"]["total_kwh"]["pct"] > 0
        assert results["savings"]["comfort"]["comfort_preserved"]

    def test_the_actuators_actually_moved_the_building(self, tmp_path: Path) -> None:
        """The proof that control was injected rather than merely computed: the
        two arms must produce different zone temperatures from the same model."""
        cfg = make_config(
            tmp_path, run_id="ep2", engine="energyplus", timesteps_per_hour=4, brain="heuristic"
        )
        Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        events = read_events(cfg.out_dir / "events.jsonl", ["telemetry"])
        baseline = [e["snapshot"] for e in events if e["label"] == "baseline"]
        ai = [e["snapshot"] for e in events if e["label"] == "ai"]
        setpoints_differ = sum(
            1
            for b, a in zip(baseline, ai)
            if abs(b["zones"][0]["cooling_setpoint_c"] - a["zones"][0]["cooling_setpoint_c"]) > 0.1
        )
        temps_differ = sum(
            1 for b, a in zip(baseline, ai) if abs(b["zones"][0]["temp_c"] - a["zones"][0]["temp_c"]) > 0.1
        )
        assert setpoints_differ > 20, "the agent's set-points never diverged from the baseline"
        assert temps_differ > 20, "different set-points did not change the simulated building"

    def test_the_generated_idf_is_written_and_valid(self, tmp_path: Path) -> None:
        from ecoloop.sim.idf import IDF

        cfg = make_config(tmp_path, run_id="ep3", engine="energyplus", brain="heuristic")
        Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        for name in ("baseline", "ai"):
            path = cfg.out_dir / "idf" / f"{name}.idf"
            assert path.exists()
            idf = IDF.load(path)
            run_period = idf.of_class("RunPeriod")[0]
            assert run_period.fields[1] == "5" and run_period.fields[2] == "12"

    def test_energyplus_native_reports_are_kept(self, tmp_path: Path) -> None:
        cfg = make_config(tmp_path, run_id="ep4", engine="energyplus", brain="heuristic")
        Orchestrator(cfg, EventBus(cfg.out_dir / "events.jsonl")).run()
        assert (cfg.out_dir / "eplus" / "ai" / "eplusout.err").exists()
        assert (cfg.out_dir / "eplus" / "ai" / "eplustbl.htm").exists()


class TestWeather:
    def test_a_weather_file_resolves(self) -> None:
        epw = resolve_epw()
        assert epw is None or epw.suffix == ".epw"

    def test_epw_parses_and_interpolates(self) -> None:
        epw_path = resolve_epw()
        if epw_path is None:
            pytest.skip("no weather file available")
        epw = EPW.load(epw_path)
        assert len(epw.records) >= 8760
        at_14 = epw.at(5, 12, 14, 0)
        at_1430 = epw.at(5, 12, 14, 30)
        assert -40 < at_14.drybulb_c < 60
        assert 0 <= at_14.rh_pct <= 100
        assert at_14.ghi_w_m2 >= 0
        assert at_14.drybulb_c != at_1430.drybulb_c or at_14.ghi_w_m2 == at_1430.ghi_w_m2

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_epw("/nonexistent/file.epw")

    def test_synthetic_weather_is_a_sane_fallback(self) -> None:
        midday = synthetic_record(5, 12, 13)
        midnight = synthetic_record(5, 12, 1)
        assert midday.drybulb_c > midnight.drybulb_c
        assert midday.ghi_w_m2 > 0
        assert midnight.ghi_w_m2 == 0


class TestLogDigest:
    def test_classification(self) -> None:
        assert classify("   ** Severe  ** something bad")[0] == "severe"
        assert classify("   ** Warning ** something odd")[0] == "warning"
        assert classify("   ** Fatal  ** giving up")[0] == "fatal"
        assert classify("Starting Simulation")[0] == "info"

    def test_fingerprint_collapses_names_and_numbers(self) -> None:
        a = fingerprint('Zone "OFFICE" temperature 34.12 C exceeds limit')
        b = fingerprint('Zone "PROD_HALL" temperature 31.88 C exceeds limit')
        assert a == b

    def test_ten_thousand_repeats_become_one_entry(self) -> None:
        """The whole point: a runaway error must not blow the prompt budget."""
        digest = LogDigest()
        for i in range(10_000):
            digest.add_line(f'   ** Severe  ** Zone "Z{i}" temperature {20 + i * 0.01:.2f} C is bad')
        assert digest.total == 10_000
        assert len(digest.entries) == 1
        assert digest.severe_count == 10_000
        rendered = digest.for_llm(1200)
        assert len(rendered) <= 1200
        assert "x10000" in rendered

    def test_worst_first_ordering(self) -> None:
        digest = LogDigest()
        digest.add_line("   ** Warning ** a warning")
        digest.add_line("   ** Fatal  ** a fatal")
        digest.add_line("   ** Severe  ** a severe")
        levels = [e.level for e in digest.ranked()]
        assert levels == ["fatal", "severe", "warning"]

    def test_search_falls_back_when_the_regex_is_invalid(self) -> None:
        """An LLM will send an invalid regex sooner or later."""
        digest = LogDigest()
        digest.add_line("   ** Severe  ** GetSurfaceData: Sub-surface is out of bounds")
        assert digest.search("Sub-surface")
        assert digest.search("Sub-surface[")   # unbalanced bracket

    def test_clean_run_says_so(self) -> None:
        digest = LogDigest()
        digest.add_line("Starting Simulation")
        assert "no warnings or errors" in digest.for_llm(400)

    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        digest = LogDigest()
        digest.add_file(tmp_path / "absent.err")
        assert digest.total == 0


class TestExternalControlDuringARun:
    """An external MCP client writing set-points into a *running* simulation.

    This is the claim that the MCP server is a controller and not a viewer, and
    it is also the one place two threads write the agent's pending-action slot:
    the agent mid-decision and the engine draining the control inbox on the same
    timestep. Without a lock around that slot, one caller's action is silently
    consumed as the other's.
    """

    def test_external_writes_and_agent_decisions_coexist(self, tmp_path: Path) -> None:
        import threading
        import time

        from ecoloop.agent.controller import LLMPolicy
        from ecoloop.bus import DECISION, TOOL_CALL
        from ecoloop.sim.surrogate import SurrogateEngine

        cfg = make_config(
            tmp_path,
            run_id="external",
            brain="llm",
            agent_mode="async",
            timesteps_per_hour=4,
            decision_interval_min=30,
        )
        bus = EventBus()
        external_calls: list[dict] = []
        decisions: list[dict] = []
        bus.subscribe(
            lambda e: external_calls.append(e)
            if e["kind"] == TOOL_CALL and e.get("source") == "external"
            else (decisions.append(e) if e["kind"] == DECISION else None)
        )

        sink: list = []
        policy = LLMPolicy(cfg, bus, LogDigest(), sink)
        policy.enable_live_bridge(cfg.out_dir, every_steps=1)
        inbox = cfg.out_dir / "live" / "control_inbox.jsonl"

        stop = threading.Event()

        def writer() -> None:
            for index in range(40):
                if stop.is_set():
                    return
                with inbox.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "kind": "setpoints",
                                "payload": {
                                    "zone": "OFFICE",
                                    "cooling_setpoint_c": 26.0,
                                    "rationale": f"external {index}",
                                },
                            }
                        )
                        + "\n"
                    )
                time.sleep(0.002)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            result = SurrogateEngine(cfg, bus, snapshot_sink=sink).run(policy, "ai")
        finally:
            stop.set()
            thread.join()
            policy.close()

        assert result.ok and len(result.snapshots) == 96
        assert external_calls, "no external write reached the simulation"
        sources = {d["source"] for d in decisions}
        assert "external" in sources and "llm" in sources
        # An operator taking control is not the model failing.
        summary = policy.summary()
        assert summary["external_decisions"] == len(
            [d for d in decisions if d["source"] == "external"]
        )
        assert summary["llm_decisions"] > 0
