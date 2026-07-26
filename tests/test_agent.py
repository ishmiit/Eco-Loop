"""The agent layer: LLM client parsing, tool registry, policies, MCP server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_zone

from ecoloop.agent.context import FileContext, LiveContext
from ecoloop.agent.llm import LLMClient, LLMReply, ToolCall, extract_tool_calls_from_text
from ecoloop.agent.policies import BaselinePolicy, HeuristicPolicy, recommend, zone_state
from ecoloop.agent.tools import SCOPE_ANALYSIS, SCOPE_CONTROL, build_registry
from ecoloop.config import ComfortTargets, GridTargets, LLMConfig, RunConfig
from ecoloop.logs import LogDigest
from ecoloop.mcp.server import MCPServer
from ecoloop.telemetry import GridSignal, Snapshot


# --------------------------------------------------------------------------
# LLM response parsing — where small local models actually break
# --------------------------------------------------------------------------


class TestToolCallParsing:
    def test_arguments_as_dict(self) -> None:
        """Ollama's native API hands back a parsed object."""
        from ecoloop.agent.llm import _coerce_arguments

        assert _coerce_arguments({"cooling_setpoint_c": 26.0}) == {"cooling_setpoint_c": 26.0}

    def test_arguments_as_json_string(self) -> None:
        """The OpenAI schema specifies a string."""
        from ecoloop.agent.llm import _coerce_arguments

        assert _coerce_arguments('{"cooling_setpoint_c": 26.0}') == {"cooling_setpoint_c": 26.0}

    def test_double_encoded_arguments(self) -> None:
        from ecoloop.agent.llm import _coerce_arguments

        assert _coerce_arguments('"{\\"a\\": 1}"') == {"a": 1}

    def test_garbage_arguments_become_empty(self) -> None:
        from ecoloop.agent.llm import _coerce_arguments

        assert _coerce_arguments("not json at all") == {}
        assert _coerce_arguments(None) == {}
        assert _coerce_arguments([1, 2]) == {}

    def test_recovers_a_fenced_json_tool_call(self) -> None:
        """A 3B model often answers with fenced JSON instead of using the tool
        protocol. Discarding those replies throws away a usable decision."""
        text = (
            "Sure, here is my decision:\n```json\n"
            '{"name": "set_zone_setpoints", "arguments": {"cooling_setpoint_c": 26.5, '
            '"rationale": "band edge"}}\n```'
        )
        calls = extract_tool_calls_from_text(text, {"set_zone_setpoints"})
        assert len(calls) == 1
        assert calls[0].name == "set_zone_setpoints"
        assert calls[0].arguments["cooling_setpoint_c"] == 26.5

    def test_recovers_a_bare_argument_object(self) -> None:
        text = '{"cooling_setpoint_c": 29.0, "oa_fraction": 0.35, "rationale": "setback"}'
        calls = extract_tool_calls_from_text(text, {"set_zone_setpoints"})
        assert len(calls) == 1
        assert calls[0].name == "set_zone_setpoints"

    def test_ignores_unknown_tool_names(self) -> None:
        text = '{"name": "launch_missiles", "arguments": {}}'
        assert extract_tool_calls_from_text(text, {"set_zone_setpoints"}) == []

    def test_prose_only_yields_nothing(self) -> None:
        assert extract_tool_calls_from_text("I think we should cool it a bit.", {"x"}) == []

    def test_empty_text_is_safe(self) -> None:
        assert extract_tool_calls_from_text("", {"x"}) == []


class TestAssistantTurnFormatting:
    """Ollama wants an object, OpenAI wants a string — getting this wrong only
    fails on a decision's *second* round trip, which is easy to miss."""

    def _reply(self) -> LLMReply:
        return LLMReply(
            content="", tool_calls=[ToolCall(name="get_control_targets", arguments={"a": 1})]
        )

    def test_ollama_gets_an_object(self) -> None:
        client = LLMClient(LLMConfig(provider="ollama"))
        turn = client.assistant_turn(self._reply())
        assert turn["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_openai_gets_a_string(self) -> None:
        client = LLMClient(LLMConfig(provider="openai_compat"))
        turn = client.assistant_turn(self._reply())
        assert turn["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


class TestMockProvider:
    def test_mock_returns_an_action(self) -> None:
        client = LLMClient(LLMConfig(provider="mock"))
        reply = client.chat([{"role": "user", "content": '{"occupied": true}'}], tools=[])
        assert reply.ok
        assert reply.tool_calls[0].name == "set_zone_setpoints"

    def test_deadline_in_the_past_is_not_attempted(self) -> None:
        client = LLMClient(LLMConfig(provider="mock"))
        reply = client.chat([{"role": "user", "content": "x"}], deadline=0.0)
        assert reply.ok is False
        assert "deadline" in reply.error

    def test_unreachable_endpoint_reports_cleanly(self) -> None:
        client = LLMClient(LLMConfig(provider="ollama", base_url="http://127.0.0.1:1", timeout_s=1))
        health = client.health()
        assert health["ok"] is False
        assert health["error"]


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------


class TestToolRegistry:
    def _registry(self, tmp_path: Path):
        return build_registry(FileContext(tmp_path / "run"))

    def test_scopes_partition_the_tools(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        control = registry.names(SCOPE_CONTROL)
        analysis = registry.names(SCOPE_ANALYSIS)
        assert control and analysis
        assert not (control & analysis)
        assert "set_zone_setpoints" in control
        assert "propose_ecm" in analysis

    def test_openai_and_mcp_render_the_same_tools(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        openai_names = {t["function"]["name"] for t in registry.openai_schema(("control", "analysis"))}
        mcp_names = {t["name"] for t in registry.mcp_schema()}
        assert openai_names == mcp_names

    def test_control_scope_is_served_terse(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        terse = json.dumps(registry.openai_schema(SCOPE_CONTROL))
        full = json.dumps(registry.openai_schema(SCOPE_CONTROL, terse=False))
        assert len(terse) < len(full)

    def test_mcp_keeps_the_full_descriptions(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        for tool in registry.mcp_schema():
            assert len(tool["description"]) > 40, tool["name"]

    def test_read_only_hint_matches_mutating(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        by_name = {t.name: t for t in registry.tools}
        for entry in registry.mcp_schema():
            assert entry["annotations"]["readOnlyHint"] is not by_name[entry["name"]].mutating

    def test_unknown_tool_returns_an_error_not_an_exception(self, tmp_path: Path) -> None:
        result = self._registry(tmp_path).call("no_such_tool", {})
        assert result.ok is False
        assert "unknown tool" in result.error
        assert "available_tools" in result.payload

    def test_handler_exception_is_captured(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        boom = next(t for t in registry.tools if t.name == "get_building_state")
        boom.handler = lambda a: (_ for _ in ()).throw(RuntimeError("kaboom"))
        result = registry.call("get_building_state", {})
        assert result.ok is False
        assert "kaboom" in result.error

    def test_out_of_range_arguments_are_clamped_by_the_wrapper(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        result = registry.call("get_recent_history", {"minutes": 999999, "metric": "co2"})
        assert result.ok
        assert result.payload["window_minutes"] <= 720
        bad_type = registry.call("get_recent_history", {"minutes": "soon"})
        assert bad_type.ok

    def test_tool_result_text_is_capped(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        result = registry.call("list_available_ecms", {})
        assert len(result.as_text(120)) <= 120

    def test_docs_table_covers_every_tool(self, tmp_path: Path) -> None:
        registry = self._registry(tmp_path)
        assert len(registry.docs_table()) == len(registry.tools)


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------


def snap(hour: int, zones, peak: bool = False, carbon: float = 700.0, outdoor: float = 33.0) -> Snapshot:
    return Snapshot(
        month=5, day=12, hour=hour, minute=0, weekday=2, outdoor_temp_c=outdoor,
        zones=list(zones), grid=GridSignal(carbon, 11.5 if peak else 8.0, peak),
        occupied=any(z.occupants > 0.05 for z in zones),
    )


class TestBaselinePolicy:
    def test_operating_hours_use_the_fixed_setpoint(self) -> None:
        action = BaselinePolicy().decide(snap(10, [make_zone(occupants=5.0)]))
        assert action.zone_overrides["OFFICE"]["cooling_setpoint_c"] == 24.0

    def test_night_setback(self) -> None:
        action = BaselinePolicy().decide(snap(2, [make_zone(occupants=0.0)]))
        assert action.zone_overrides["OFFICE"]["cooling_setpoint_c"] == 30.0

    def test_sunday_is_closed(self) -> None:
        sunday = snap(10, [make_zone(occupants=0.0)])
        sunday.weekday = 7
        action = BaselinePolicy().decide(sunday)
        assert action.zone_overrides["OFFICE"]["cooling_setpoint_c"] == 30.0

    def test_saturday_is_a_half_day(self) -> None:
        saturday = snap(17, [make_zone(occupants=0.0)])
        saturday.weekday = 6
        action = BaselinePolicy().decide(saturday)
        assert action.zone_overrides["OFFICE"]["cooling_setpoint_c"] == 30.0

    def test_it_does_not_react_to_the_building(self) -> None:
        """A fixed-schedule BMS is blind by definition; if it reacted, it would
        stop being a valid control group."""
        cool = BaselinePolicy().decide(snap(10, [make_zone(temp=22.0, occupants=5.0, co2=500)]))
        hot = BaselinePolicy().decide(snap(10, [make_zone(temp=29.0, occupants=5.0, co2=1400)]))
        assert (
            cool.zone_overrides["OFFICE"]["cooling_setpoint_c"]
            == hot.zone_overrides["OFFICE"]["cooling_setpoint_c"]
        )


class TestRecommendation:
    comfort = ComfortTargets()
    grid = GridTargets()

    def test_unoccupied_gets_full_setback(self) -> None:
        zone = make_zone(occupants=0.0, mins_to_occupied=None)
        rec = recommend(zone, snap(2, [zone]), self.comfort, self.grid)
        assert rec.lever == "A"
        assert rec.cooling_setpoint_c == self.comfort.unoccupied_cooling_max_c
        assert rec.oa_fraction == self.comfort.oa_fraction_min

    def test_pre_occupancy_gets_optimum_start(self) -> None:
        zone = make_zone(occupants=0.0, mins_to_occupied=45.0)
        rec = recommend(zone, snap(5, [zone]), self.comfort, self.grid)
        assert rec.lever == "F"
        assert rec.cooling_setpoint_c < self.comfort.cooling_setpoint_max_c
        assert rec.cooling_setpoint_c >= self.comfort.cooling_setpoint_min_c

    def test_shift_boundary_counts_as_occupied(self) -> None:
        """At the exact boundary the schedule says in-shift while the sensor
        still reads zero people; treating it as empty discards the pre-cool."""
        zone = make_zone(occupants=0.0, mins_to_occupied=0.0)
        occupied, pre_start = zone_state(zone)
        assert occupied is True and pre_start is False

    def test_occupied_sits_high_in_the_band(self) -> None:
        zone = make_zone(occupants=5.0, pmv=0.2, pmv_limit=1.5)
        rec = recommend(zone, snap(11, [zone]), self.comfort, self.grid)
        assert rec.lever in ("B", "E")
        assert rec.cooling_setpoint_c >= self.comfort.cooling_setpoint_max_c - 1.1

    def test_peak_window_coasts_to_the_top(self) -> None:
        zone = make_zone(occupants=5.0, pmv=0.2, pmv_limit=1.5)
        rec = recommend(zone, snap(19, [zone], peak=True), self.comfort, self.grid)
        assert rec.lever == "C"
        assert rec.cooling_setpoint_c == self.comfort.cooling_setpoint_max_c

    def test_pre_peak_precools(self) -> None:
        zone = make_zone(occupants=5.0, pmv=0.2, pmv_limit=1.5)
        coast = recommend(zone, snap(19, [zone], peak=True), self.comfort, self.grid)
        precool = recommend(zone, snap(17, [zone], outdoor=34.0), self.comfort, self.grid)
        assert precool.lever == "C"
        assert precool.cooling_setpoint_c < coast.cooling_setpoint_c

    def test_thin_comfort_margin_overrides_the_energy_levers(self) -> None:
        zone = make_zone(occupants=5.0, temp=26.0, pmv=1.05, pmv_limit=1.1)
        rec = recommend(zone, snap(19, [zone], peak=True), self.comfort, self.grid)
        assert rec.cooling_setpoint_c < self.comfort.cooling_setpoint_max_c
        assert "comfort" in rec.reason

    def test_dcv_tracks_co2(self) -> None:
        low = make_zone(occupants=5.0, co2=500.0)
        high = make_zone(occupants=5.0, co2=1050.0)
        assert (
            recommend(low, snap(11, [low]), self.comfort, self.grid).oa_fraction
            < recommend(high, snap(11, [high]), self.comfort, self.grid).oa_fraction
        )

    def test_recommendation_is_always_inside_its_own_band(self) -> None:
        for hour in range(24):
            for occupants in (0.0, 6.0):
                for co2 in (450.0, 1090.0):
                    for pmv in (-0.2, 0.5, 1.4):
                        zone = make_zone(
                            occupants=occupants, co2=co2, pmv=pmv, pmv_limit=1.5,
                            mins_to_occupied=0.0 if occupants else None,
                        )
                        rec = recommend(zone, snap(hour, [zone]), self.comfort, self.grid)
                        lo, hi = self.comfort.cooling_bounds(rec.state != "unoccupied")
                        assert lo - 1e-6 <= rec.cooling_setpoint_c <= hi + 1e-6
                        assert (
                            self.comfort.oa_fraction_min
                            <= rec.oa_fraction
                            <= self.comfort.oa_fraction_max
                        )


class TestHeuristicPolicy:
    def test_it_applies_the_recommendation_verbatim(self) -> None:
        """The ablation arm must be exactly the recommendation, or `--brain
        heuristic` stops isolating the language model's contribution."""
        comfort, grid = ComfortTargets(), GridTargets()
        zones = [make_zone("OFFICE", occupants=5.0), make_zone("PROD_HALL", occupants=0.0, mins_to_occupied=None)]
        state = snap(11, zones)
        action = HeuristicPolicy(comfort, grid).decide(state)
        for zone in zones:
            rec = recommend(zone, state, comfort, grid)
            assert action.zone_overrides[zone.name] == rec.as_dict()


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------


class TestMCPServer:
    def _server(self, tmp_path: Path) -> MCPServer:
        run = tmp_path / "artifacts" / "run1"
        (run / "live").mkdir(parents=True)
        return MCPServer(run)

    def _call(self, server: MCPServer, method: str, params: dict | None = None, request_id=1):
        request = {"jsonrpc": "2.0", "method": method}
        if request_id is not None:
            request["id"] = request_id
        if params is not None:
            request["params"] = params
        return server.handle(request)

    def test_initialize_advertises_capabilities(self, tmp_path: Path) -> None:
        response = self._call(self._server(tmp_path), "initialize", {})
        result = response["result"]
        assert result["protocolVersion"]
        assert "tools" in result["capabilities"]
        assert "resources" in result["capabilities"]
        assert result["serverInfo"]["name"]

    def test_tools_list_matches_the_registry(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        tools = self._call(server, "tools/list")["result"]["tools"]
        assert {t["name"] for t in tools} == {t.name for t in server.registry.tools}
        for tool in tools:
            assert "inputSchema" in tool

    def test_tools_call_returns_a_text_content_block(self, tmp_path: Path) -> None:
        response = self._call(
            self._server(tmp_path), "tools/call", {"name": "list_available_ecms", "arguments": {}}
        )
        content = response["result"]["content"]
        assert content[0]["type"] == "text"
        assert response["result"]["isError"] is False
        assert json.loads(content[0]["text"])["ecms"]

    def test_failed_tool_sets_is_error_not_a_jsonrpc_error(self, tmp_path: Path) -> None:
        """A tool failure the model can read and retry beats a protocol error
        that kills the turn."""
        response = self._call(
            self._server(tmp_path), "tools/call", {"name": "nope", "arguments": {}}
        )
        assert "error" not in response
        assert response["result"]["isError"] is True

    def test_notifications_get_no_response(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        assert self._call(server, "notifications/initialized", {}, request_id=None) is None
        assert server.initialized is True

    def test_unknown_method_is_a_jsonrpc_error(self, tmp_path: Path) -> None:
        response = self._call(self._server(tmp_path), "does/not/exist")
        assert response["error"]["code"] == -32601

    def test_ping(self, tmp_path: Path) -> None:
        assert self._call(self._server(tmp_path), "ping")["result"] == {}

    def test_resources_list_includes_the_model(self, tmp_path: Path) -> None:
        resources = self._call(self._server(tmp_path), "resources/list")["result"]["resources"]
        uris = {r["uri"] for r in resources}
        assert "ecoloop://model/baseline.idf" in uris
        assert "ecoloop://runs" in uris

    def test_read_the_baseline_model(self, tmp_path: Path) -> None:
        result = self._call(
            self._server(tmp_path), "resources/read", {"uri": "ecoloop://model/baseline.idf"}
        )["result"]
        assert "Schedule:Constant, CSP_OFFICE" in result["contents"][0]["text"]

    def test_path_traversal_is_refused(self, tmp_path: Path) -> None:
        """A resource URI is attacker-influenced input."""
        response = self._call(
            self._server(tmp_path),
            "resources/read",
            {"uri": "ecoloop://runs/run1/../../../../etc/passwd"},
        )
        assert "error" in response

    def test_unsupported_uri_is_an_error(self, tmp_path: Path) -> None:
        response = self._call(self._server(tmp_path), "resources/read", {"uri": "http://evil"})
        assert "error" in response

    def test_writes_are_queued_to_the_control_inbox(self, tmp_path: Path) -> None:
        """This is what makes an external MCP client a controller rather than a
        viewer: the request lands in the inbox the simulation drains."""
        server = self._server(tmp_path)
        response = self._call(
            server,
            "tools/call",
            {
                "name": "set_zone_setpoints",
                "arguments": {"zone": "OFFICE", "cooling_setpoint_c": 26.0, "rationale": "test"},
            },
        )
        assert response["result"]["isError"] is False
        inbox = server.run_dir / "live" / "control_inbox.jsonl"
        assert inbox.exists()
        record = json.loads(inbox.read_text().strip())
        assert record["kind"] == "setpoints"
        assert record["payload"]["cooling_setpoint_c"] == 26.0

    def test_a_malformed_request_does_not_kill_the_session(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        # tools/call with no params is a *tool* failure, reported as isError so
        # the model can read it and retry — not a protocol error that kills the
        # turn. Either way the session must survive.
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})
        assert response["result"]["isError"] is True
        assert self._call(server, "ping")["result"] == {}


# --------------------------------------------------------------------------
# Contexts
# --------------------------------------------------------------------------


class TestContexts:
    def test_live_context_serves_the_latest_snapshot(self) -> None:
        cfg = RunConfig(run_id="ctx")
        snaps: list[Snapshot] = []
        ctx = LiveContext(
            cfg=cfg, digest=LogDigest(), snapshots=lambda: snaps,
            on_setpoints=lambda r: {"ok": True}, on_hold=lambda r: {"ok": True},
        )
        assert "note" in ctx.building_state()
        snaps.append(snap(14, [make_zone(occupants=5.0)]))
        state = ctx.building_state()
        assert state["zones"][0]["z"] == "OFFICE"
        assert "comfort_flags" in state

    def test_history_reports_a_trend(self) -> None:
        cfg = RunConfig(run_id="ctx")
        snaps = []
        for i in range(12):
            snaps.append(snap(10, [make_zone(temp=24.0 + i * 0.3, occupants=5.0)]))
        ctx = LiveContext(
            cfg=cfg, digest=LogDigest(), snapshots=lambda: snaps,
            on_setpoints=lambda r: {}, on_hold=lambda r: {},
        )
        result = ctx.history(180, "zone_temp")
        assert result["trend"] == "rising"
        assert len(result["points"]) <= 12

    def test_grid_forecast_finds_the_cheapest_hour(self) -> None:
        cfg = RunConfig(run_id="ctx")
        snaps = [snap(16, [make_zone(occupants=5.0)])]
        ctx = LiveContext(
            cfg=cfg, digest=LogDigest(), snapshots=lambda: snaps,
            on_setpoints=lambda r: {}, on_hold=lambda r: {},
        )
        forecast = ctx.grid_forecast(6)
        assert forecast["now_hour"] == 16
        assert forecast["peak_starts_in_hours"] == 2

    def test_file_context_on_a_missing_run_is_graceful(self, tmp_path: Path) -> None:
        ctx = FileContext(tmp_path / "nope")
        assert "note" in ctx.building_state()
        assert ctx.is_live() is False
        result = ctx.apply_setpoints({"cooling_setpoint_c": 25})
        assert result["ok"] is False

    def test_ecm_authoring_is_refused_when_not_enabled(self) -> None:
        cfg = RunConfig(run_id="ctx")
        ctx = LiveContext(
            cfg=cfg, digest=LogDigest(), snapshots=lambda: [],
            on_setpoints=lambda r: {}, on_hold=lambda r: {}, on_ecm=None,
        )
        assert ctx.propose_ecm([{"ecm": "cool_roof"}], "x")["ok"] is False
