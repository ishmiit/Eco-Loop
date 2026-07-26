"""The tool registry — one definition, three consumers.

The same ``Tool`` objects are rendered as:

* OpenAI/Ollama ``tools=[...]`` function schemas for the in-process agent;
* MCP ``tools/list`` entries for any external MCP client (Claude Desktop, an
  IDE, another agent) — see ``ecoloop.mcp.server``;
* the documentation table in ``docs/ARCHITECTURE.md``.

Keeping one registry is what makes the claim "the LLM and an external MCP
client drive the building through the identical interface" true rather than
aspirational.

Tools are **scoped**: the per-decision loop only sees the six tools that are
useful inside a 25-second control decision, while the ECM pass and external MCP
clients see the full set. Handing a 3B model eleven tools when six will do
costs latency and accuracy for nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

SCOPE_CONTROL = "control"     # inside the real-time decision loop
SCOPE_ANALYSIS = "analysis"   # log reading, IDF inspection, ECM authoring
SCOPE_ALL = (SCOPE_CONTROL, SCOPE_ANALYSIS)

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler
    scope: str = SCOPE_CONTROL
    # True if calling this tool changes the building or writes a file.
    mutating: bool = False
    # Terse description for the real-time control loop. Tool schemas sit in the
    # prompt on *every* decision, and prompt evaluation is the single largest
    # component of decision latency (measured: 3.8 s of a 6.2 s decision at a
    # 2339-token prompt). External MCP clients get the full prose; the control
    # loop gets one line, and the strategy guidance lives in the system prompt
    # where it is stated once.
    brief: str = ""
    # Fields dropped from the schema in the control loop for the same reason.
    brief_schema: dict[str, Any] | None = None

    def as_openai(self, terse: bool = False) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (self.brief or self.description) if terse else self.description,
                "parameters": (
                    (self.brief_schema or self.schema) if terse else self.schema
                ),
            },
        }

    def as_mcp(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.schema,
            "annotations": {
                "readOnlyHint": not self.mutating,
                "title": self.name.replace("_", " ").title(),
            },
        }


class RunContext(Protocol):
    """What a tool is allowed to do to the world.

    Implemented twice: ``LiveContext`` (inside the simulation process, direct
    object access) and ``FileContext`` (from the web server or an MCP client,
    via the run's artifact directory). Tools are written once against this
    protocol, so an external MCP client gets exactly the agent's capabilities.
    """

    def building_state(self) -> dict[str, Any]: ...
    def history(self, minutes: int, metric: str) -> dict[str, Any]: ...
    def targets(self) -> dict[str, Any]: ...
    def grid_forecast(self, hours: int) -> dict[str, Any]: ...
    def apply_setpoints(self, request: dict[str, Any]) -> dict[str, Any]: ...
    def hold(self, rationale: str) -> dict[str, Any]: ...
    def read_log(self, level: str, max_chars: int) -> dict[str, Any]: ...
    def search_log(self, pattern: str, limit: int) -> dict[str, Any]: ...
    def list_idf_objects(self, object_class: str, limit: int) -> dict[str, Any]: ...
    def available_ecms(self) -> dict[str, Any]: ...
    def propose_ecm(self, measures: list[dict[str, Any]], rationale: str) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# Schemas. Deliberately flat and small: nested objects and free-form maps are
# where small models fail. "zone" takes a name or the literal "ALL".
# --------------------------------------------------------------------------

_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


def build_registry(ctx: RunContext) -> "ToolRegistry":
    tools: list[Tool] = [
        Tool(
            name="get_building_state",
            description=(
                "Current sensor readings for all three zones: air temperature, relative humidity, "
                "CO2, occupant count, Fanger PMV, the set-points in force, HVAC electrical power, "
                "grid carbon intensity and tariff. Use this if you need to re-read the building; "
                "the latest state is already included in the decision prompt."
            ),
            schema=_NO_ARGS,
            handler=lambda a: ctx.building_state(),
            scope=SCOPE_CONTROL,
            brief="Re-read all zone sensors. The latest state is already in the prompt.",
        ),
        Tool(
            name="get_recent_history",
            description=(
                "Trend for one metric over the last N simulated minutes, down-sampled to at most "
                "12 points, with the min/max/mean and the direction of travel. Use it to tell a "
                "transient apart from a trend before changing set-points."
            ),
            schema={
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Look-back window in simulated minutes (15-720).",
                        "default": 120,
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["zone_temp", "co2", "pmv", "power", "outdoor_temp", "cooling_setpoint"],
                        "description": "Which signal to summarise.",
                        "default": "zone_temp",
                    },
                },
                "additionalProperties": False,
            },
            handler=lambda a: ctx.history(
                int(_num(a.get("minutes"), 120, 15, 720)), str(a.get("metric") or "zone_temp")
            ),
            scope=SCOPE_CONTROL,
            brief="Trend of one metric over the last N minutes, with min/max/mean and direction.",
        ),
        Tool(
            name="get_control_targets",
            description=(
                "The hard constraints you must respect: occupied and unoccupied temperature bands, "
                "the PMV limit, the CO2 ceiling, the allowed set-point range per occupancy state, "
                "the maximum change per decision, and the facility peak demand limit."
            ),
            schema=_NO_ARGS,
            handler=lambda a: ctx.targets(),
            scope=SCOPE_CONTROL,
            brief="Full comfort, ventilation and grid constraints.",
        ),
        Tool(
            name="get_grid_forecast",
            description=(
                "Grid carbon intensity (gCO2/kWh) and tariff (INR/kWh) for the next N hours, "
                "flagging the peak-tariff window. Use it to decide whether to pre-cool now and "
                "coast later."
            ),
            schema={
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Horizon, 1-12 hours.", "default": 4}
                },
                "additionalProperties": False,
            },
            handler=lambda a: ctx.grid_forecast(int(_num(a.get("hours"), 4, 1, 12))),
            scope=SCOPE_CONTROL,
            brief="Grid carbon and tariff for the next N hours; flags the peak window.",
        ),
        Tool(
            name="set_zone_setpoints",
            description=(
                "THE CONTROL ACTION. Write cooling/heating set-points and the outdoor-air fraction "
                "into the running EnergyPlus model. Applies from the next timestep. Values outside "
                "the allowed band are clamped by the safety layer and the clamp is reported back "
                "to you. Call this exactly once per decision."
            ),
            schema={
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "OFFICE, PROD_HALL, PACK_STORE, or ALL for every zone.",
                        "default": "ALL",
                    },
                    "cooling_setpoint_c": {
                        "type": "number",
                        "description": "Cooling set-point in Celsius (23.0-26.5 occupied, 26.0-30.0 unoccupied).",
                    },
                    "heating_setpoint_c": {
                        "type": "number",
                        "description": "Heating set-point in Celsius (19.0-22.0 occupied, 15.0-18.0 unoccupied).",
                    },
                    "oa_fraction": {
                        "type": "number",
                        "description": "Outdoor-air fraction 0.35-1.0. Lower saves energy; raise it when CO2 climbs.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence: which target this serves and why now.",
                    },
                },
                "required": ["cooling_setpoint_c", "rationale"],
                "additionalProperties": False,
            },
            handler=lambda a: ctx.apply_setpoints(a),
            scope=SCOPE_CONTROL,
            mutating=True,
            brief=(
                "THE CONTROL ACTION. Write cooling/heating set-points and outdoor-air fraction "
                "into the running model. Call once per decision."
            ),
            brief_schema={
                "type": "object",
                "properties": {
                    "zone": {"type": "string", "description": "OFFICE, PROD_HALL, PACK_STORE or ALL."},
                    "cooling_setpoint_c": {"type": "number", "description": "Celsius, inside the allowed band."},
                    "heating_setpoint_c": {"type": "number", "description": "Celsius, inside the allowed band."},
                    "oa_fraction": {"type": "number", "description": "0.35-1.0; lower saves energy, raise it as CO2 climbs."},
                    "rationale": {"type": "string", "description": "One sentence: which lever and why now."},
                },
                "required": ["cooling_setpoint_c", "rationale"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="hold_current_strategy",
            description=(
                "Explicitly keep the set-points already in force. The correct action when the "
                "building is inside every target and nothing in the forecast has changed — "
                "unnecessary set-point movement costs energy and comfort."
            ),
            schema={
                "type": "object",
                "properties": {
                    "rationale": {"type": "string", "description": "Why no change is needed."}
                },
                "required": ["rationale"],
                "additionalProperties": False,
            },
            handler=lambda a: ctx.hold(str(a.get("rationale") or "no change required")),
            scope=SCOPE_CONTROL,
            mutating=True,
            brief="Keep the set-points already in force. Correct when nothing has changed.",
        ),
        Tool(
            name="read_simulation_log",
            description=(
                "Deduplicated EnergyPlus log digest, worst-first (fatal, then severe, then "
                "warning) with repeat counts. Use it after a failed simulation to find the cause."
            ),
            schema={
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["all", "fatal", "severe", "warning"],
                        "default": "severe",
                    },
                    "max_chars": {"type": "integer", "default": 1200},
                },
                "additionalProperties": False,
            },
            handler=lambda a: ctx.read_log(
                str(a.get("level") or "severe"), int(_num(a.get("max_chars"), 1200, 200, 8000))
            ),
            scope=SCOPE_ANALYSIS,
        ),
        Tool(
            name="search_simulation_log",
            description=(
                "Regex or substring search across the deduplicated log entries. Use it to check "
                "whether a specific object name or error phrase appears."
            ),
            schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or plain substring."},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=lambda a: ctx.search_log(
                str(a.get("pattern") or ""), int(_num(a.get("limit"), 8, 1, 40))
            ),
            scope=SCOPE_ANALYSIS,
        ),
        Tool(
            name="list_idf_objects",
            description=(
                "Inspect the building model: list objects of one IDF class (Construction, Material, "
                "Lights, Zone, ZoneHVAC:IdealLoadsAirSystem, ...) with their field values. Omit the "
                "class to get the class inventory."
            ),
            schema={
                "type": "object",
                "properties": {
                    "object_class": {"type": "string", "description": "IDF class name, or empty for an inventory."},
                    "limit": {"type": "integer", "default": 12},
                },
                "additionalProperties": False,
            },
            handler=lambda a: ctx.list_idf_objects(
                str(a.get("object_class") or ""), int(_num(a.get("limit"), 12, 1, 60))
            ),
            scope=SCOPE_ANALYSIS,
        ),
        Tool(
            name="list_available_ecms",
            description=(
                "The Energy Conservation Measures you can apply to the building model, with their "
                "parameters. These are capital measures, distinct from the real-time set-point "
                "control done with set_zone_setpoints."
            ),
            schema=_NO_ARGS,
            handler=lambda a: ctx.available_ecms(),
            scope=SCOPE_ANALYSIS,
        ),
        Tool(
            name="propose_ecm",
            description=(
                "Write a modified .idf applying one or more ECMs, which is then simulated and "
                "compared against the baseline. If the generated file fails to simulate you will "
                "be shown the EnergyPlus error so you can correct it and try again."
            ),
            schema={
                "type": "object",
                "properties": {
                    "measures": {
                        "type": "array",
                        "description": 'e.g. [{"ecm": "cool_roof", "params": {"solar_absorptance": 0.3}}]',
                        "items": {
                            "type": "object",
                            "properties": {
                                "ecm": {"type": "string"},
                                "params": {"type": "object"},
                            },
                            "required": ["ecm"],
                        },
                    },
                    "rationale": {"type": "string", "description": "Why these measures, from the run evidence."},
                },
                "required": ["measures"],
                "additionalProperties": False,
            },
            handler=lambda a: ctx.propose_ecm(
                list(a.get("measures") or []), str(a.get("rationale") or "")
            ),
            scope=SCOPE_ANALYSIS,
            mutating=True,
        ),
    ]
    return ToolRegistry(tools)


def _num(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


@dataclass
class ToolResult:
    name: str
    ok: bool
    payload: dict[str, Any]
    latency_ms: float = 0.0
    error: str = ""

    def as_text(self, max_chars: int = 1400) -> str:
        body = self.payload if self.ok else {"error": self.error, **self.payload}
        text = json.dumps(body, separators=(",", ":"), default=str)
        if len(text) > max_chars:
            suffix = '...,"truncated":true}'
            text = text[: max(0, max_chars - len(suffix))] + suffix
        return text


@dataclass
class ToolRegistry:
    tools: list[Tool]
    calls: list[ToolResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_name = {t.name: t for t in self.tools}

    def names(self, scope: str | Iterable[str] = SCOPE_ALL) -> set[str]:
        return {t.name for t in self.select(scope)}

    def select(self, scope: str | Iterable[str] = SCOPE_ALL) -> list[Tool]:
        wanted = {scope} if isinstance(scope, str) else set(scope)
        return [t for t in self.tools if t.scope in wanted]

    def openai_schema(
        self, scope: str | Iterable[str] = SCOPE_CONTROL, terse: bool | None = None
    ) -> list[dict[str, Any]]:
        """``terse`` defaults to True for the control scope, where the schema is
        re-sent on every decision, and False elsewhere."""
        if terse is None:
            terse = scope == SCOPE_CONTROL
        return [t.as_openai(terse=terse) for t in self.select(scope)]

    def mcp_schema(self, scope: str | Iterable[str] = SCOPE_ALL) -> list[dict[str, Any]]:
        return [t.as_mcp() for t in self.select(scope)]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Never raises: an unknown tool or a handler exception comes back as a
        ToolResult the model can read and react to."""
        started = time.monotonic()
        tool = self._by_name.get(name)
        if tool is None:
            result = ToolResult(
                name=name,
                ok=False,
                payload={"available_tools": sorted(self._by_name)},
                error=f"unknown tool {name!r}",
            )
        else:
            try:
                payload = tool.handler(dict(arguments or {}))
                result = ToolResult(name=name, ok=True, payload=payload or {})
            except Exception as exc:
                result = ToolResult(
                    name=name,
                    ok=False,
                    payload={"hint": "check the argument names and types against the schema"},
                    error=f"{type(exc).__name__}: {exc}",
                )
        result.latency_ms = (time.monotonic() - started) * 1000.0
        self.calls.append(result)
        return result

    def docs_table(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "scope": t.scope,
                "mutating": t.mutating,
                "args": sorted((t.schema.get("properties") or {}).keys()),
                "description": t.description,
            }
            for t in self.tools
        ]
