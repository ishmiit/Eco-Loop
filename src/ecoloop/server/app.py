"""The dashboard's HTTP surface.

Three groups of endpoints:

* **read** — runs, results, telemetry series, decisions;
* **stream** — ``/api/stream`` tails the run's ``events.jsonl`` and pushes
  Server-Sent Events, which is how the dashboard shows a simulation live;
* **act** — ``/api/runs`` launches a run in a *subprocess*, and
  ``/api/tools/*`` exposes the agent's own tool registry over HTTP.

The subprocess boundary is deliberate. EnergyPlus runs in-process through a C
library; hosting it inside the web server would mean a simulation crash takes
the dashboard with it, and would serialise runs behind the event loop. Instead
the run writes ``events.jsonl`` and the server tails it, so either side can die
without the other noticing.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..agent.context import FileContext
from ..agent.llm import LLMClient
from ..agent.tools import SCOPE_ALL, build_registry
from ..bus import read_events
from ..config import ARTIFACTS_DIR, LLMConfig, REPO_ROOT
from ..energyplus_locate import find_energyplus
from ..orchestrator import list_runs
from ..weather import resolve_epw

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Eco-Loop Building Agents", version=__version__)

# Tracks runs this server started, so /api/runs/active can report on them.
_processes: dict[str, subprocess.Popen] = {}


def _run_dir(run_id: str) -> Path:
    """Resolve a run id inside the artifacts tree, rejecting traversal."""
    base = ARTIFACTS_DIR.resolve()
    target = (base / run_id).resolve()
    if not str(target).startswith(str(base)) or not target.is_dir():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return target


# --------------------------------------------------------------------- read


@app.get("/api/health")
def health() -> dict[str, Any]:
    install = find_energyplus()
    epw = resolve_epw()
    llm = LLMClient(LLMConfig.from_env()).health()
    return {
        "version": __version__,
        "energyplus": {
            "found": install is not None,
            "version": install.version if install else None,
            "root": str(install.root) if install else None,
        },
        "weather": epw.name if epw else None,
        "llm": llm,
        "engine": "energyplus" if install else "surrogate",
    }


@app.get("/api/runs")
def runs() -> dict[str, Any]:
    return {"runs": list_runs()}


@app.get("/api/runs/active")
def active_runs() -> dict[str, Any]:
    out = []
    for run_id, process in list(_processes.items()):
        code = process.poll()
        out.append({"run_id": run_id, "running": code is None, "exit_code": code})
        if code is not None and (ARTIFACTS_DIR / run_id / "results.json").exists():
            _processes.pop(run_id, None)
    return {"active": out}


@app.get("/api/runs/{run_id}/results")
def results(run_id: str) -> dict[str, Any]:
    path = _run_dir(run_id) / "results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run has not finished yet")
    return json.loads(path.read_text())


@app.get("/api/runs/{run_id}/manifest")
def manifest(run_id: str) -> dict[str, Any]:
    path = _run_dir(run_id) / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no manifest")
    return json.loads(path.read_text())


@app.get("/api/runs/{run_id}/telemetry")
def telemetry(
    run_id: str,
    label: str = Query("ai", pattern="^(ai|baseline)$"),
    stride: int = Query(1, ge=1, le=64),
) -> dict[str, Any]:
    """Column-oriented series — far smaller over the wire than row objects,
    and directly consumable by the chart code."""
    events = read_events(_run_dir(run_id) / "events.jsonl", ["telemetry"])
    snaps = [e["snapshot"] for e in events if e.get("label") == label and e.get("snapshot")]
    snaps = snaps[::stride]
    if not snaps:
        return {"label": label, "points": 0, "series": {}, "zones": []}
    zone_names = [z["name"] for z in snaps[0]["zones"]]
    series: dict[str, list[Any]] = {
        "clock": [s["clock"] for s in snaps],
        "hour": [s["hour"] + s["minute"] / 60.0 for s in snaps],
        "outdoor_temp_c": [round(s["outdoor_temp_c"], 2) for s in snaps],
        "solar_w_m2": [round(s["solar_w_m2"], 1) for s in snaps],
        "total_kw": [round(s["total_elec_w"] / 1000.0, 3) for s in snaps],
        "hvac_kw": [
            round(
                (s["hvac_cooling_elec_w"] + s["hvac_heating_elec_w"] + s["fan_elec_w"]) / 1000.0, 3
            )
            for s in snaps
        ],
        "cum_kwh": [round(s["cum_kwh"], 3) for s in snaps],
        "cum_cost_inr": [round(s["cum_cost_inr"], 2) for s in snaps],
        "cum_carbon_kg": [round(s["cum_carbon_kg"], 3) for s in snaps],
        "occupied": [1 if s["occupied"] else 0 for s in snaps],
        "peak_window": [1 if s["grid"]["peak_window"] else 0 for s in snaps],
        "carbon_g_kwh": [round(s["grid"]["carbon_g_per_kwh"], 1) for s in snaps],
        "control_source": [s.get("control_source", "") for s in snaps],
        "guardrail": [len(s.get("guardrail_notes") or []) for s in snaps],
    }
    zones: list[dict[str, Any]] = []
    for index, name in enumerate(zone_names):
        zones.append(
            {
                "name": name,
                "pmv_limit": snaps[0]["zones"][index].get("pmv_limit", 0.7),
                "temp_c": [round(s["zones"][index]["temp_c"], 2) for s in snaps],
                "cooling_sp_c": [round(s["zones"][index]["cooling_setpoint_c"], 2) for s in snaps],
                "pmv": [round(s["zones"][index]["pmv"], 3) for s in snaps],
                "co2_ppm": [round(s["zones"][index]["co2_ppm"], 1) for s in snaps],
                "occupants": [round(s["zones"][index]["occupants"], 2) for s in snaps],
            }
        )
    return {"label": label, "points": len(snaps), "series": series, "zones": zones}


@app.get("/api/runs/{run_id}/decisions")
def decisions(run_id: str, limit: int = Query(400, ge=1, le=5000)) -> dict[str, Any]:
    events = read_events(_run_dir(run_id) / "events.jsonl", ["decision"])
    return {"decisions": events[-limit:], "total": len(events)}


@app.get("/api/runs/{run_id}/tool-calls")
def tool_calls(run_id: str, limit: int = Query(300, ge=1, le=5000)) -> dict[str, Any]:
    events = read_events(_run_dir(run_id) / "events.jsonl", ["tool_call"])
    return {"tool_calls": events[-limit:], "total": len(events)}


@app.get("/api/runs/{run_id}/logs")
def logs(run_id: str, limit: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    events = read_events(_run_dir(run_id) / "events.jsonl", ["log", "status", "ecm"])
    return {"events": events[-limit:]}


@app.get("/api/runs/{run_id}/file")
def run_file(run_id: str, path: str = Query(...)) -> FileResponse:
    """Serve an artifact (CSV, IDF, EnergyPlus HTML report) for download."""
    base = _run_dir(run_id)
    target = (base / path).resolve()
    if not str(target).startswith(str(base.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="no such artifact")
    return FileResponse(target, filename=target.name)


# ------------------------------------------------------------------- stream


@app.get("/api/stream")
async def stream(run_id: str = Query(...), from_seq: int = Query(0, ge=0)) -> StreamingResponse:
    """Tail ``events.jsonl`` as Server-Sent Events.

    Waits for the file to appear (a run that has just been launched has not
    written it yet) and stops once the run has finished and the file is fully
    drained, so the browser is not left holding an open request forever.
    """
    run_path = _run_dir(run_id) / "events.jsonl"

    async def generate() -> AsyncIterator[bytes]:
        yield b": connected\n\n"
        position = 0
        seq = from_seq
        idle = 0.0
        finished = False
        while True:
            if run_path.exists():
                try:
                    with run_path.open("r", encoding="utf-8") as fh:
                        fh.seek(position)
                        chunk = fh.read()
                        position = fh.tell()
                except OSError:
                    chunk = ""
                sent = 0
                for line in chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue   # a partially flushed final line
                    if int(event.get("seq", 0)) <= seq:
                        continue
                    seq = int(event.get("seq", seq))
                    sent += 1
                    yield f"event: {event.get('kind', 'message')}\ndata: {json.dumps(event)}\n\n".encode()
                    if event.get("kind") == "status" and event.get("phase") == "run_done":
                        finished = True
                idle = 0.0 if sent else idle + 0.4
            else:
                idle += 0.4

            if finished:
                yield b"event: eof\ndata: {}\n\n"
                return
            process = _processes.get(run_id)
            if process is not None and process.poll() is not None and idle > 2.0:
                yield b"event: eof\ndata: {}\n\n"
                return
            if process is None and idle > 90.0:
                yield b"event: eof\ndata: {}\n\n"
                return
            yield b": keep-alive\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------- act


class RunRequest(BaseModel):
    days: int = Field(2, ge=1, le=14)
    brain: str = Field("llm", pattern="^(llm|heuristic|baseline)$")
    engine: str = Field("auto", pattern="^(auto|energyplus|surrogate)$")
    decision_interval: int = Field(30, ge=5, le=180)
    agent_mode: str = Field("sync", pattern="^(sync|async)$")
    pace: float = Field(0.0, ge=0.0, le=2.0)
    model: str = ""
    ecm_pass: bool = False
    start_month: int = Field(5, ge=1, le=12)
    start_day: int = Field(12, ge=1, le=28)
    run_id: str = ""


@app.post("/api/runs")
def start_run(request: RunRequest) -> dict[str, Any]:
    run_id = request.run_id or f"live_{time.strftime('%Y%m%d_%H%M%S')}"
    if any(p.poll() is None for p in _processes.values()):
        raise HTTPException(
            status_code=409,
            detail="a run is already in progress — one simulation at a time on one machine",
        )
    argv = [
        sys.executable, "-m", "ecoloop", "run",
        "--run-id", run_id,
        "--days", str(request.days),
        "--brain", request.brain,
        "--engine", request.engine,
        "--decision-interval", str(request.decision_interval),
        "--agent-mode", request.agent_mode,
        "--pace", str(request.pace),
        "--start-month", str(request.start_month),
        "--start-day", str(request.start_day),
        "--quiet",
    ]
    if request.model:
        argv += ["--model", request.model]
    if request.ecm_pass:
        argv += ["--ecm-pass"]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    (ARTIFACTS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    log = (ARTIFACTS_DIR / run_id / "run.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        argv, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT
    )
    _processes[run_id] = process
    return {"run_id": run_id, "pid": process.pid, "command": " ".join(argv)}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    process = _processes.get(run_id)
    if process is None or process.poll() is not None:
        return {"run_id": run_id, "stopped": False, "detail": "not running"}
    process.terminate()
    return {"run_id": run_id, "stopped": True}


@app.get("/api/tools")
def tools() -> dict[str, Any]:
    registry = build_registry(FileContext(ARTIFACTS_DIR / "none"))
    return {"tools": registry.mcp_schema(SCOPE_ALL), "docs": registry.docs_table()}


class ToolCall(BaseModel):
    run_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/tools/call")
def call_tool(request: ToolCall) -> JSONResponse:
    """The agent's tool registry over HTTP — the same handlers the LLM and MCP
    clients use, so the dashboard's manual-override panel is not a special case."""
    registry = build_registry(FileContext(_run_dir(request.run_id)))
    result = registry.call(request.name, request.arguments)
    return JSONResponse(
        {
            "ok": result.ok,
            "name": result.name,
            "payload": result.payload,
            "error": result.error,
            "latency_ms": round(result.latency_ms, 2),
        },
        status_code=200 if result.ok else 400,
    )


# -------------------------------------------------------------------- pages


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> int:
    import uvicorn

    print(f"  Eco-Loop dashboard  ->  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload, log_level="warning")
    return 0
