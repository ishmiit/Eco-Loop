"""An MCP server for the building — stdio JSON-RPC 2.0, no dependencies.

The point of this file is that the tools it publishes are *the same objects*
the in-process agent uses. ``ecoloop.agent.tools.build_registry`` is called once
here with a :class:`~ecoloop.agent.context.FileContext`, and the resulting
``Tool`` list is rendered into MCP ``inputSchema`` form. So:

* a local Llama/Qwen model driving the loop, and
* Claude Desktop, an IDE, or another agent connected over MCP

get an identical interface to the identical handlers. An external client can
read live telemetry and **write set-points into a running simulation** — the
request is queued to the run's control inbox, drained by the simulation on its
next timestep, and passed through the same guardrail as the model's own actions.

Wire up (Claude Desktop ``claude_desktop_config.json``, or any MCP client):

    {
      "mcpServers": {
        "ecoloop": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["-m", "ecoloop.mcp.server"],
          "env": {"PYTHONPATH": "/absolute/path/to/HoneyWell/src"}
        }
      }
    }

Protocol notes: ``initialize`` advertises ``tools`` and ``resources``
capabilities; ``tools/call`` returns the handler's JSON as a single text
content block, with ``isError`` set on failure rather than a JSON-RPC error, so
the model sees the message and can retry. Notifications get no response, as
required. Anything unexpected becomes a JSON-RPC error object instead of a
crash — a server that dies takes the client's session with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from ..agent.context import FileContext
from ..agent.tools import SCOPE_ALL, build_registry
from ..config import ARTIFACTS_DIR, REPO_ROOT
from ..energyplus_locate import describe as describe_energyplus
from ..orchestrator import list_runs

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "ecoloop-building-agent"
SERVER_VERSION = "1.0.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603

# Files inside a run directory published as MCP resources.
_RESOURCE_FILES: tuple[tuple[str, str, str], ...] = (
    ("results.json", "application/json", "KPIs, savings table and agent statistics"),
    ("manifest.json", "application/json", "The exact RunConfig used (reproduces the run)"),
    ("savings.csv", "text/csv", "Headline baseline vs AI comparison"),
    ("telemetry_baseline.csv", "text/csv", "Per-timestep telemetry, rule-based baseline"),
    ("telemetry_ai.csv", "text/csv", "Per-timestep telemetry, AI closed loop"),
    ("decisions.jsonl", "application/x-ndjson", "Every agent decision with rationale and latency"),
    ("ecm_report.json", "application/json", "Phase B retrofit attempts and self-corrections"),
    ("ecm_evidence.json", "application/json", "Run evidence given to the retrofit agent"),
)


def resolve_run_dir(run_id: str = "") -> Path:
    """Explicit id, else ``$ECOLOOP_RUN``, else the newest run on disk."""
    import os

    candidate = (run_id or os.environ.get("ECOLOOP_RUN", "")).strip()
    if candidate:
        return ARTIFACTS_DIR / candidate
    runs = list_runs()
    live = [r for r in runs if (ARTIFACTS_DIR / r["run_id"] / "live").is_dir()]
    if live:
        return ARTIFACTS_DIR / live[0]["run_id"]
    if runs:
        return ARTIFACTS_DIR / runs[0]["run_id"]
    return ARTIFACTS_DIR / "none"


class MCPServer:
    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir = run_dir or resolve_run_dir()
        self.context = FileContext(self.run_dir)
        self.registry = build_registry(self.context)
        self.initialized = False

    # ------------------------------------------------------------ dispatch
    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Returns a JSON-RPC response, or None for notifications."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}
        is_notification = request_id is None

        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method in ("notifications/initialized", "initialized"):
                self.initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.registry.mcp_schema(SCOPE_ALL)}
            elif method == "tools/call":
                return self._tools_call(request_id, params)
            elif method == "resources/list":
                result = {"resources": self._resources()}
            elif method == "resources/read":
                result = self._read_resource(str(params.get("uri") or ""))
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method and method.startswith("notifications/"):
                return None
            else:
                if is_notification:
                    return None
                return _error(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")
        except Exception as exc:   # never let one bad request kill the session
            if is_notification:
                return None
            return _error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        self.initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Eco-Loop Building Agents. You are connected to a three-zone FSSAI food "
                f"premises simulated in EnergyPlus. Active run: {self.run_dir.name}. "
                f"{describe_energyplus().splitlines()[0]}\n"
                "Read the building with get_building_state / get_recent_history, then write "
                "set-points with set_zone_setpoints. If a simulation is running, your writes "
                "are applied on its next timestep after the safety-layer clamp; if none is "
                "running, reads serve the last completed run and writes are refused.\n"
                "list_available_ecms and propose_ecm work on the building model itself."
            ),
        }

    def _tools_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        result = self.registry.call(name, arguments)
        payload = result.payload if result.ok else {"error": result.error, **result.payload}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
                "isError": not result.ok,
            },
        }

    # ----------------------------------------------------------- resources
    def _resources(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [
            {
                "uri": "ecoloop://runs",
                "name": "All runs",
                "description": "Every experiment in artifacts/, newest first",
                "mimeType": "application/json",
            },
            {
                "uri": "ecoloop://model/baseline.idf",
                "name": "Baseline building model",
                "description": "The committed EnergyPlus input file for the three-zone premises",
                "mimeType": "text/plain",
            },
            {
                "uri": "ecoloop://docs/architecture",
                "name": "System architecture",
                "description": "Tool-calling architecture, prompt strategy, latency management",
                "mimeType": "text/markdown",
            },
        ]
        for filename, mime, description in _RESOURCE_FILES:
            if (self.run_dir / filename).exists():
                out.append(
                    {
                        "uri": f"ecoloop://runs/{self.run_dir.name}/{filename}",
                        "name": f"{self.run_dir.name}: {filename}",
                        "description": description,
                        "mimeType": mime,
                    }
                )
        for idf in sorted(self.run_dir.glob("idf/*.idf")):
            out.append(
                {
                    "uri": f"ecoloop://runs/{self.run_dir.name}/idf/{idf.name}",
                    "name": f"{self.run_dir.name}: {idf.name}",
                    "description": "Generated EnergyPlus model for this run",
                    "mimeType": "text/plain",
                }
            )
        return out

    def _read_resource(self, uri: str) -> dict[str, Any]:
        def wrap(text: str, mime: str) -> dict[str, Any]:
            return {"contents": [{"uri": uri, "mimeType": mime, "text": text}]}

        if uri == "ecoloop://runs":
            return wrap(json.dumps(list_runs(), indent=2, default=str), "application/json")
        if uri == "ecoloop://model/baseline.idf":
            path = REPO_ROOT / "models" / "baseline.idf"
            if not path.exists():
                raise FileNotFoundError("models/baseline.idf is missing")
            return wrap(path.read_text(encoding="latin-1"), "text/plain")
        if uri == "ecoloop://docs/architecture":
            path = REPO_ROOT / "docs" / "ARCHITECTURE.md"
            if not path.exists():
                raise FileNotFoundError("docs/ARCHITECTURE.md is missing")
            return wrap(path.read_text(encoding="utf-8"), "text/markdown")

        prefix = "ecoloop://runs/"
        if not uri.startswith(prefix):
            raise ValueError(f"unsupported resource uri: {uri}")
        remainder = uri[len(prefix):]
        run_id, _, relative = remainder.partition("/")
        if not run_id or not relative:
            raise ValueError(f"malformed run resource uri: {uri}")
        # Contain the path inside the artifacts tree: a resource URI is
        # attacker-influenced input, and "../../etc/passwd" must not resolve.
        base = (ARTIFACTS_DIR / run_id).resolve()
        target = (base / relative).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            raise FileNotFoundError(f"no such resource: {uri}")
        mime = next((m for f, m, _ in _RESOURCE_FILES if f == relative), None)
        if mime is None:
            mime = "text/plain" if target.suffix in (".idf", ".txt", ".err") else "application/json"
        return wrap(target.read_text(encoding="latin-1"), mime)


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None, run_dir: Path | None = None) -> int:
    """Read line-delimited JSON-RPC from stdin, write responses to stdout."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = MCPServer(run_dir)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, _error(None, PARSE_ERROR, f"invalid JSON: {exc}"))
            continue
        if isinstance(request, list):   # JSON-RPC batch
            for item in request:
                response = server.handle(item) if isinstance(item, dict) else _error(
                    None, INVALID_REQUEST, "batch items must be objects"
                )
                if response is not None:
                    _write(stdout, response)
            continue
        if not isinstance(request, dict):
            _write(stdout, _error(None, INVALID_REQUEST, "request must be an object"))
            continue
        response = server.handle(request)
        if response is not None:
            _write(stdout, response)
    return 0


def _write(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, default=str) + "\n")
    stream.flush()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Eco-Loop MCP server (stdio JSON-RPC 2.0)")
    parser.add_argument("--run", default="", help="run id under artifacts/ (default: newest/live)")
    args = parser.parse_args()
    return serve(run_dir=resolve_run_dir(args.run))


if __name__ == "__main__":
    raise SystemExit(main())
