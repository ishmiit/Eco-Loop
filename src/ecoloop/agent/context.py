"""Implementations of :class:`~ecoloop.agent.tools.RunContext`.

``LiveContext`` runs inside the simulation process and is what the LLM policy
uses. ``FileContext`` runs anywhere else — the web server, an MCP client in
another process — and reaches the same live run through its artifact directory:

    artifacts/<run>/live/state.json          latest snapshot + targets (written)
    artifacts/<run>/live/history.jsonl       down-sampled telemetry   (written)
    artifacts/<run>/live/control_inbox.jsonl external control requests (read)

The simulation drains ``control_inbox.jsonl`` on each timestep, so an external
MCP client genuinely steers the building — with the same guardrail clamp
applied to its requests as to the model's.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ..config import ComfortTargets, GridTargets, RunConfig
from ..logs import LogDigest
from ..metrics import grid_signal
from ..sim.base import DEFAULT_ZONES
from ..sim.idf import ECM_DOCS, ECM_LIBRARY
from ..telemetry import Snapshot

MAX_HISTORY = 4000


def _targets_payload(comfort: ComfortTargets, grid: GridTargets) -> dict[str, Any]:
    return {
        "comfort": {
            "occupied_temp_band_c": [comfort.occupied_temp_min_c, comfort.occupied_temp_max_c],
            "unoccupied_temp_band_c": [comfort.unoccupied_temp_min_c, comfort.unoccupied_temp_max_c],
            "pmv_limit_abs": comfort.pmv_limit,
            "co2_limit_ppm": comfort.co2_limit_ppm,
            "rh_max_pct": comfort.rh_max_pct,
        },
        "setpoint_authority": {
            "occupied": {
                "cooling_c": [comfort.cooling_setpoint_min_c, comfort.cooling_setpoint_max_c],
                "heating_c": [comfort.heating_setpoint_min_c, comfort.heating_setpoint_max_c],
            },
            "unoccupied": {
                "cooling_c": [comfort.unoccupied_cooling_min_c, comfort.unoccupied_cooling_max_c],
                "heating_c": [comfort.unoccupied_heating_min_c, comfort.unoccupied_heating_max_c],
            },
            "oa_fraction": [comfort.oa_fraction_min, comfort.oa_fraction_max],
            "min_deadband_c": comfort.min_deadband_c,
            "max_change_per_decision_c": comfort.max_step_c,
        },
        "grid": {
            "peak_demand_limit_w": grid.peak_demand_limit_w,
            "peak_tariff_window_hours": list(grid.peak_window),
            "tariff_inr_per_kwh": {
                "offpeak": grid.tariff_offpeak,
                "mid": grid.tariff_mid,
                "peak": grid.tariff_peak,
            },
        },
        "zones": [
            {"name": z.name, "area_m2": z.area_m2, "comfort_priority": z.priority}
            for z in DEFAULT_ZONES
        ],
    }


def _grid_forecast_payload(hour: int, hours: int, grid: GridTargets) -> dict[str, Any]:
    rows = []
    for offset in range(hours + 1):
        h = (hour + offset) % 24
        signal = grid_signal(h, grid)
        rows.append(
            {
                "hour": h,
                "carbon_g_kwh": round(signal.carbon_g_per_kwh),
                "tariff_inr_kwh": signal.tariff_inr_per_kwh,
                "peak_window": signal.peak_window,
            }
        )
    cheapest = min(rows, key=lambda r: r["tariff_inr_kwh"])
    cleanest = min(rows, key=lambda r: r["carbon_g_kwh"])
    return {
        "now_hour": hour,
        "forecast": rows,
        "cheapest_hour": cheapest["hour"],
        "cleanest_hour": cleanest["hour"],
        "peak_starts_in_hours": next(
            (r["hour"] - hour if r["hour"] >= hour else r["hour"] + 24 - hour)
            for r in rows
            if r["peak_window"]
        )
        if any(r["peak_window"] for r in rows)
        else None,
    }


def _history_payload(
    snapshots: list[Snapshot], minutes: int, metric: str, points: int = 12
) -> dict[str, Any]:
    if not snapshots:
        # Same keys as the populated case: a stable shape is easier for a small
        # model to read than a special case it has to detect.
        return {
            "metric": metric,
            "window_minutes": minutes,
            "points": [],
            "trend": "unknown",
            "note": "no history yet",
        }
    dt_min = max(1.0, snapshots[-1].timestep_hours * 60.0)
    span = max(1, int(minutes / dt_min))
    window = snapshots[-span:]

    def value(snap: Snapshot) -> float:
        if metric == "outdoor_temp":
            return snap.outdoor_temp_c
        if metric == "power":
            return snap.total_elec_w
        occupied = [z for z in snap.zones if z.occupants > 0.05] or snap.zones
        if metric == "co2":
            return max(z.co2_ppm for z in occupied)
        if metric == "pmv":
            return max((z.pmv for z in occupied), key=abs)
        if metric == "cooling_setpoint":
            return sum(z.cooling_setpoint_c for z in snap.zones) / len(snap.zones)
        return sum(z.temp_c for z in occupied) / len(occupied)

    series = [(s.clock, round(value(s), 2)) for s in window]
    stride = max(1, len(series) // points)
    sampled = series[::stride][-points:]
    numbers = [v for _, v in series]
    trend = "flat"
    if len(numbers) >= 4:
        head = sum(numbers[: max(1, len(numbers) // 4)]) / max(1, len(numbers) // 4)
        tail = sum(numbers[-max(1, len(numbers) // 4):]) / max(1, len(numbers) // 4)
        delta = tail - head
        scale = max(0.2, abs(head) * 0.01)
        trend = "rising" if delta > scale else "falling" if delta < -scale else "flat"
    return {
        "metric": metric,
        "window_minutes": minutes,
        "points": [{"t": t, "v": v} for t, v in sampled],
        "min": min(numbers),
        "max": max(numbers),
        "mean": round(sum(numbers) / len(numbers), 2),
        "trend": trend,
    }


class LiveContext:
    """Tool backend inside the simulation process."""

    def __init__(
        self,
        cfg: RunConfig,
        digest: LogDigest,
        snapshots: Callable[[], list[Snapshot]],
        on_setpoints: Callable[[dict[str, Any]], dict[str, Any]],
        on_hold: Callable[[str], dict[str, Any]],
        on_ecm: Callable[[list[dict[str, Any]], str], dict[str, Any]] | None = None,
        idf_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.digest = digest
        self._snapshots = snapshots
        self._on_setpoints = on_setpoints
        self._on_hold = on_hold
        self._on_ecm = on_ecm
        self.idf_path = Path(idf_path) if idf_path else Path(cfg.idf)
        self._idf_cache: Any = None

    # -- reads --------------------------------------------------------------
    def latest(self) -> Snapshot | None:
        snaps = self._snapshots()
        return snaps[-1] if snaps else None

    def building_state(self) -> dict[str, Any]:
        snap = self.latest()
        if snap is None:
            return {"note": "simulation has not produced a timestep yet"}
        payload = snap.compact()
        payload["comfort_flags"] = {
            "worst_pmv": round(snap.pmv_worst, 2),
            "worst_co2_ppm": round(snap.co2_worst_ppm),
            "violation_now": snap.comfort_violation,
        }
        return payload

    def history(self, minutes: int, metric: str) -> dict[str, Any]:
        return _history_payload(self._snapshots(), minutes, metric)

    def targets(self) -> dict[str, Any]:
        return _targets_payload(self.cfg.comfort, self.cfg.grid)

    def grid_forecast(self, hours: int) -> dict[str, Any]:
        snap = self.latest()
        return _grid_forecast_payload(snap.hour if snap else 0, hours, self.cfg.grid)

    def read_log(self, level: str, max_chars: int) -> dict[str, Any]:
        levels = ("fatal", "severe", "warning") if level in ("all", "") else (level,)
        return {
            "summary": self.digest.summary(),
            "digest": self.digest.for_llm(max_chars, levels),
        }

    def search_log(self, pattern: str, limit: int) -> dict[str, Any]:
        return {"pattern": pattern, "matches": self.digest.search(pattern, limit)}

    def _idf(self) -> Any:
        from ..sim.idf import IDF  # local import keeps module import cheap

        if self._idf_cache is None:
            self._idf_cache = IDF.load(self.idf_path)
        return self._idf_cache

    def list_idf_objects(self, object_class: str, limit: int) -> dict[str, Any]:
        idf = self._idf()
        if not object_class:
            return {"file": str(self.idf_path), "classes": idf.classes()}
        objects = idf.of_class(object_class)
        if not objects:
            return {
                "object_class": object_class,
                "objects": [],
                "note": "no objects of that class",
                "available_classes": sorted(idf.classes())[:60],
            }
        return {
            "object_class": object_class,
            "count": len(objects),
            "objects": [
                {"name": o.name, "fields": o.fields[:14]} for o in objects[:limit]
            ],
        }

    def available_ecms(self) -> dict[str, Any]:
        return {
            "ecms": [{"ecm": name, "description": ECM_DOCS.get(name, "")} for name in sorted(ECM_LIBRARY)]
        }

    # -- writes -------------------------------------------------------------
    def apply_setpoints(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._on_setpoints(request)

    def hold(self, rationale: str) -> dict[str, Any]:
        return self._on_hold(rationale)

    def propose_ecm(self, measures: list[dict[str, Any]], rationale: str) -> dict[str, Any]:
        if self._on_ecm is None:
            return {
                "ok": False,
                "error": "ECM authoring is not enabled for this run (use --ecm-pass)",
            }
        return self._on_ecm(measures, rationale)

    # -- live mirror for out-of-process clients ------------------------------
    def publish_live(self, out_dir: Path) -> None:
        snap = self.latest()
        if snap is None:
            return
        live = out_dir / "live"
        live.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_at": time.time(),
            "run_id": self.cfg.run_id,
            "state": snap.to_dict(),
            "compact": self.building_state(),
            "targets": self.targets(),
            "log": self.digest.summary(),
        }
        tmp = live / "state.json.tmp"
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(live / "state.json")


class FileContext:
    """Tool backend for out-of-process clients (web API, MCP server)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.live_dir = self.run_dir / "live"

    # -- helpers ------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        path = self.live_dir / "state.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _snapshots(self) -> list[Snapshot]:
        path = self.live_dir / "history.jsonl"
        out: list[Snapshot] = []
        if not path.exists():
            return out
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(Snapshot.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            return out
        return out[-MAX_HISTORY:]

    def _config(self) -> RunConfig:
        cfg = RunConfig(run_id=self.run_dir.name)
        manifest = self.run_dir / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                comfort = data.get("comfort") or {}
                grid = data.get("grid") or {}
                for key, value in comfort.items():
                    if hasattr(cfg.comfort, key):
                        setattr(cfg.comfort, key, value)
                for key, value in grid.items():
                    if hasattr(cfg.grid, key) and key not in ("peak_window", "carbon_profile"):
                        setattr(cfg.grid, key, value)
                if isinstance(grid.get("peak_window"), list) and len(grid["peak_window"]) == 2:
                    cfg.grid.peak_window = (int(grid["peak_window"][0]), int(grid["peak_window"][1]))
                if isinstance(grid.get("carbon_profile"), list) and len(grid["carbon_profile"]) == 24:
                    cfg.grid.carbon_profile = tuple(float(v) for v in grid["carbon_profile"])
                if data.get("idf"):
                    cfg.idf = data["idf"]
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                pass
        return cfg

    def is_live(self, max_age_s: float = 20.0) -> bool:
        state = self._state()
        return bool(state) and (time.time() - float(state.get("written_at", 0))) < max_age_s

    # -- reads --------------------------------------------------------------
    def building_state(self) -> dict[str, Any]:
        state = self._state()
        if not state:
            return {"note": f"no live state for run {self.run_dir.name!r}"}
        payload = dict(state.get("compact") or {})
        payload["live"] = self.is_live()
        payload["age_seconds"] = round(time.time() - float(state.get("written_at", 0)), 1)
        return payload

    def history(self, minutes: int, metric: str) -> dict[str, Any]:
        return _history_payload(self._snapshots(), minutes, metric)

    def targets(self) -> dict[str, Any]:
        state = self._state()
        if state.get("targets"):
            return state["targets"]
        cfg = self._config()
        return _targets_payload(cfg.comfort, cfg.grid)

    def grid_forecast(self, hours: int) -> dict[str, Any]:
        cfg = self._config()
        state = self._state().get("state") or {}
        return _grid_forecast_payload(int(state.get("hour", 0)), hours, cfg.grid)

    def read_log(self, level: str, max_chars: int) -> dict[str, Any]:
        digest = LogDigest()
        found = False
        for err in sorted(self.run_dir.glob("eplus/*/eplusout.err")):
            digest.add_file(err)
            found = True
        if not found:
            return {"note": "no EnergyPlus error file in this run directory yet"}
        levels = ("fatal", "severe", "warning") if level in ("all", "") else (level,)
        return {"summary": digest.summary(), "digest": digest.for_llm(max_chars, levels)}

    def search_log(self, pattern: str, limit: int) -> dict[str, Any]:
        digest = LogDigest()
        for err in sorted(self.run_dir.glob("eplus/*/eplusout.err")):
            digest.add_file(err)
        return {"pattern": pattern, "matches": digest.search(pattern, limit)}

    def list_idf_objects(self, object_class: str, limit: int) -> dict[str, Any]:
        from ..sim.idf import IDF

        candidates = sorted(self.run_dir.glob("idf/*.idf"))
        path = candidates[0] if candidates else Path(self._config().idf)
        if not path.exists():
            return {"error": f"no IDF found for run {self.run_dir.name!r}"}
        idf = IDF.load(path)
        if not object_class:
            return {"file": str(path), "classes": idf.classes()}
        objects = idf.of_class(object_class)
        return {
            "file": str(path),
            "object_class": object_class,
            "count": len(objects),
            "objects": [{"name": o.name, "fields": o.fields[:14]} for o in objects[:limit]],
        }

    def available_ecms(self) -> dict[str, Any]:
        return {
            "ecms": [{"ecm": name, "description": ECM_DOCS.get(name, "")} for name in sorted(ECM_LIBRARY)]
        }

    # -- writes: queued for the simulation loop to drain --------------------
    def _enqueue(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.live_dir.exists():
            return {"ok": False, "error": f"run {self.run_dir.name!r} is not running"}
        record = {"kind": kind, "queued_at": time.time(), "payload": payload}
        with (self.live_dir / "control_inbox.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return {
            "ok": True,
            "queued": record,
            "note": "applied on the next simulation timestep, after the guardrail clamp",
        }

    def apply_setpoints(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._enqueue("setpoints", request)

    def hold(self, rationale: str) -> dict[str, Any]:
        return self._enqueue("hold", {"rationale": rationale})

    def propose_ecm(self, measures: list[dict[str, Any]], rationale: str) -> dict[str, Any]:
        return self._enqueue("ecm", {"measures": measures, "rationale": rationale})
