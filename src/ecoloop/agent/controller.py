"""The LLM control policy — where prompt latency stops being a problem.

An EnergyPlus timestep takes ~1 ms. A 3B model decision takes 1-4 s. Blocking
the simulation on every model call would be both slow and unlike a real BMS, so
the two run on separate clocks:

    engine thread            agent thread
    ------------             ------------
    decide(snap) ---------->  latest snapshot (overwritten, never queued)
      returns the HELD             |
      action, re-clamped           v
      against THIS snapshot   LLM tool loop (1-3 round trips)
           ^                       |
           +---- new held action <-+

Four consequences, all of them deliberate:

* **The simulation never stalls.** ``decide`` is O(zones) and returns
  immediately, so a wedged or absent model cannot break a 3-day run.
* **Held actions stay safe.** The held action is re-clamped against the current
  snapshot on *every* timestep, so comfort protection tracks the building
  continuously rather than at the decision cadence.
* **Latest-wins, never a backlog.** The snapshot slot is overwritten, so a slow
  decision is answered against fresh state instead of building a queue of stale
  work — the classic failure of naive agent loops.
* **Timeouts degrade, not fail.** Exceed the budget and the heuristic policy
  takes that decision; the run continues and the KPI records who decided.

``agent_mode="sync"`` inverts this for demos and for reviewers who want to see
the model in the loop on every decision: the engine waits (bounded by
``llm.timeout_s``) at each decision point. Same code path, same guardrails.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ..bus import DECISION, LOG, TOOL_CALL, EventBus
from ..config import RunConfig
from ..logs import LogDigest
from ..telemetry import ControlAction, Snapshot
from . import prompts
from .context import LiveContext
from .guardrails import clamp, validate_setpoint_request
from .llm import LLMClient, LLMReply
from .policies import HeuristicPolicy
from .tools import SCOPE_CONTROL, ToolRegistry, build_registry

ACTION_TOOLS = {"set_zone_setpoints", "hold_current_strategy"}
ZONE_ALIASES = {
    "ALL": None,
    "": None,
    "*": None,
    "OFFICE": "OFFICE",
    "PROD_HALL": "PROD_HALL",
    "PRODUCTION": "PROD_HALL",
    "PROD": "PROD_HALL",
    "PACK_STORE": "PACK_STORE",
    "STORE": "PACK_STORE",
    "PACKING": "PACK_STORE",
}


class LLMPolicy:
    """Closed-loop supervisory control driven by an open-source LLM."""

    name = "llm"

    def __init__(
        self,
        cfg: RunConfig,
        bus: EventBus,
        digest: LogDigest,
        snapshots_ref: list[Snapshot],
        on_ecm: Any = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.digest = digest
        self.client = LLMClient(cfg.llm)
        self.fallback = HeuristicPolicy(cfg.comfort, cfg.grid)

        self._snapshots = snapshots_ref
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: Snapshot | None = None

        # The action currently in force, pre-clamp. Seeded with a safe default
        # so the first timesteps are controlled before the model has answered.
        self._held_raw = ControlAction(
            cooling_setpoint_c=24.0,
            heating_setpoint_c=21.0,
            oa_fraction=1.0,
            source="heuristic",
            rationale="startup default until the first model decision lands",
        )
        self._held_clamped: ControlAction | None = None
        self._last_notes: list[str] = []
        self._decision_id = 0
        self._last_decision_sim_minutes = -1e9
        self._last_reasoned_signature: tuple | None = None
        self._cache: dict[tuple, ControlAction] = {}
        self._live_dir: Path | None = None
        self._live_every = 2
        # ``_pending_action`` is written by whichever thread is calling a tool
        # handler. In async mode that can be the agent thread mid-decision *and*
        # the engine thread draining an external MCP request on the same
        # timestep, so the slot needs its own lock — otherwise one caller's
        # action is silently consumed as the other's.
        self._action_lock = threading.Lock()

        self.stats = {
            "decisions": 0,
            "llm_ok": 0,
            "llm_failed": 0,
            "fallback": 0,
            "cache_hits": 0,
            "external": 0,
            "tool_calls": 0,
            "recovered_from_text": 0,
            "guardrail_interventions": 0,
            "accepted_default": 0,
            "deviated_from_default": 0,
            "latencies_ms": [],
        }

        # Tool plumbing. The registry is shared with the MCP server, so an
        # external client calls the identical handlers.
        self._pending_action: ControlAction | None = None
        self.context = LiveContext(
            cfg=cfg,
            digest=digest,
            snapshots=lambda: list(self._snapshots),
            on_setpoints=self._tool_set_setpoints,
            on_hold=self._tool_hold,
            on_ecm=on_ecm,
        )
        self.registry: ToolRegistry = build_registry(self.context)
        self._control_tools = self.registry.openai_schema(SCOPE_CONTROL)
        self._control_names = self.registry.names(SCOPE_CONTROL)

        health = self.client.health()
        self.bus.publish(
            LOG,
            level="info" if health.get("ok") else "severe",
            source="agent",
            message=(
                f"LLM {cfg.llm.provider}/{cfg.llm.model} "
                + ("reachable" if health.get("ok") else f"UNREACHABLE: {health.get('error')}")
            ),
            health=health,
        )
        self.llm_available = bool(health.get("ok"))

        self._thread: threading.Thread | None = None
        if cfg.agent_mode == "async":
            self._thread = threading.Thread(target=self._worker, name="ecoloop-agent", daemon=True)
            self._thread.start()

    # ------------------------------------------------------ live bridge
    def enable_live_bridge(self, out_dir: Path, every_steps: int = 2) -> None:
        """Mirror state to ``<run>/live/`` and honour external control requests.

        This is what makes the MCP server more than a viewer: a client in
        another process (Claude Desktop, an IDE, another agent) can call
        ``set_zone_setpoints`` against a *running* simulation. Its request lands
        in ``control_inbox.jsonl``, is drained here on the next timestep, and
        goes through the identical guardrail as the model's own actions.
        """
        self._live_dir = out_dir / "live"
        self._live_dir.mkdir(parents=True, exist_ok=True)
        (self._live_dir / "control_inbox.jsonl").write_text("")
        self._live_every = max(1, every_steps)

    def _pump_live(self, snap: Snapshot) -> None:
        if self._live_dir is None:
            return
        if snap.step % self._live_every == 0:
            try:
                self.context.publish_live(self._live_dir.parent)
                write_history(self._live_dir / "history.jsonl", self._snapshots, stride=2)
            except OSError:
                pass
        self._drain_inbox(snap)

    def _drain_inbox(self, snap: Snapshot) -> None:
        inbox = self._live_dir / "control_inbox.jsonl" if self._live_dir else None
        if inbox is None or not inbox.exists():
            return
        try:
            if inbox.stat().st_size == 0:
                return
            lines = inbox.read_text(encoding="utf-8").splitlines()
            inbox.write_text("")   # claim the batch
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("kind")
            payload = record.get("payload") or {}
            if kind == "setpoints":
                with self._action_lock:
                    in_flight = self._pending_action
                    self._pending_action = None
                    result = self._tool_set_setpoints(payload)
                    action = self._pending_action
                    self._pending_action = in_flight    # give the agent its slot back
                if action is not None:
                    action.source = "external"
                    action.rationale = (
                        f"external MCP client: {payload.get('rationale', 'no rationale')}"
                    )
                    self._decision_id += 1
                    action.decision_id = self._decision_id
                    self._commit(action, snap)
                self.bus.publish(
                    TOOL_CALL, source="external", tool="set_zone_setpoints",
                    arguments=payload, ok=bool(result.get("ok")), error=result.get("error", ""),
                    clock=snap.clock,
                )
            elif kind == "hold":
                self.bus.publish(
                    TOOL_CALL, source="external", tool="hold_current_strategy",
                    arguments=payload, ok=True, clock=snap.clock,
                )

    # ------------------------------------------------------------------ API
    def decide(self, snap: Snapshot) -> ControlAction:
        """Called by the engine every timestep. Returns promptly, always."""
        self._pump_live(snap)
        sim_minutes = _sim_minutes(snap)
        due = sim_minutes - self._last_decision_sim_minutes >= self.cfg.decision_interval_min
        if not due and self._material_change(snap):
            due = True

        if due:
            self._last_decision_sim_minutes = sim_minutes
            if self.cfg.agent_mode == "sync":
                self._run_decision(snap)
            else:
                with self._lock:
                    self._pending = snap      # latest-wins
                self._wake.set()

        # Return the held action as-is. The engine applies the safety layer to
        # whatever any policy returns, so clamping here would double-apply it —
        # and the guarantee belongs at the actuator boundary, not in one policy.
        with self._lock:
            return self._held_raw.copy()

    def on_applied(self, action: ControlAction, snap: Snapshot) -> None:
        """Engine callback: what the safety layer actually applied this timestep.

        The corrections are fed back into the next prompt, which is how the agent
        learns the shape of its own constraints instead of repeating a rejected
        request every decision.
        """
        self._held_clamped = action
        if action.clamped:
            self.stats["guardrail_interventions"] += 1
            with self._lock:
                self._last_notes = action.clamped

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.cfg.llm.timeout_s))

    # --------------------------------------------------------------- worker
    def _worker(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                snap = self._pending
                self._pending = None
            if snap is not None:
                try:
                    self._run_decision(snap)
                except Exception as exc:   # a decision must never kill the thread
                    self.bus.publish(
                        LOG, level="severe", source="agent",
                        message=f"decision failed: {type(exc).__name__}: {exc}",
                    )

    # ------------------------------------------------------------- decision
    def _run_decision(self, snap: Snapshot) -> None:
        self._decision_id += 1
        snap.decision_id = self._decision_id
        started = time.monotonic()
        deadline = started + self.cfg.llm.timeout_s

        signature = self._signature(snap)
        if self.cfg.llm.enable_cache and signature in self._cache:
            cached = self._cache[signature].copy()
            cached.decision_id = self._decision_id
            cached.source = "llm"
            cached.rationale = f"[cached] {cached.rationale}"
            cached.latency_ms = (time.monotonic() - started) * 1000.0
            self._commit(cached, snap, cache_hit=True)
            return

        if not self.llm_available:
            self._commit(self._fallback_action(snap, "LLM endpoint unreachable"), snap)
            return

        with self._action_lock:
            self._pending_action = None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.decision_prompt(
                    snap,
                    self.cfg.comfort,
                    self.cfg.grid,
                    last_action_summary=self._in_force_summary(),
                    guardrail_notes=self._last_notes,
                ),
            },
        ]

        tool_calls_made = 0
        last_reply: LLMReply | None = None
        for round_index in range(self.cfg.llm.max_tool_rounds):
            reply = self.client.chat(
                messages,
                tools=self._control_tools,
                deadline=deadline,
                valid_names=self._control_names,
            )
            last_reply = reply
            if not reply.ok:
                self._commit(self._fallback_action(snap, reply.error or "model error"), snap, reply=reply)
                return
            if reply.recovered_from_text:
                self.stats["recovered_from_text"] += 1

            if not reply.tool_calls:
                # No tool call and no recoverable JSON: nudge once, then fall back.
                if round_index == 0:
                    messages.append({"role": "assistant", "content": reply.content[:400]})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You must call one action tool now: set_zone_setpoints "
                                "(with cooling_setpoint_c and rationale) or hold_current_strategy."
                            ),
                        }
                    )
                    continue
                self._commit(
                    self._fallback_action(snap, "model produced no tool call"), snap, reply=reply
                )
                return

            messages.append(self.client.assistant_turn(reply))
            for call in reply.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                tool_calls_made += 1
                self.stats["tool_calls"] += 1
                self.bus.publish(
                    TOOL_CALL,
                    decision_id=self._decision_id,
                    tool=call.name,
                    arguments=call.arguments,
                    ok=result.ok,
                    error=result.error,
                    latency_ms=round(result.latency_ms, 1),
                    clock=snap.clock,
                )
                messages.append(
                    {"role": "tool", "name": call.name, "content": result.as_text(1200)}
                )

            with self._action_lock:
                action = self._pending_action
                self._pending_action = None
            if action is not None:
                action.decision_id = self._decision_id
                if (
                    action.rationale in ("", "(no rationale supplied by the model)")
                    and reply.content.strip()
                ):
                    # Models often put the reasoning in the text and omit the
                    # tool's rationale field; keep it rather than lose it — but
                    # not when the "reasoning" is just the prompt read back,
                    # which a 3B model does often enough to pollute the log.
                    text = reply.content.strip().replace("\n", " ")
                    if not _looks_like_echo(text):
                        action.rationale = text[:220]
                action.latency_ms = (time.monotonic() - started) * 1000.0
                action.tool_calls = tool_calls_made
                action.model = reply.model
                if self.cfg.llm.enable_cache:
                    self._cache[signature] = action.copy()
                    if len(self._cache) > 512:
                        self._cache.clear()
                self._commit(action, snap, reply=reply)
                return

            if time.monotonic() > deadline:
                break

        self._commit(
            self._fallback_action(snap, "no action after tool rounds"), snap, reply=last_reply
        )

    def _commit(
        self,
        action: ControlAction,
        snap: Snapshot,
        reply: LLMReply | None = None,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            self._held_raw = action.copy()
        self.stats["decisions"] += 1
        if cache_hit:
            self.stats["cache_hits"] += 1
        if action.source == "llm":
            self.stats["llm_ok"] += 1
        elif action.source == "external":
            # An operator or MCP client taking control is not the model failing.
            self.stats["external"] += 1
        else:
            self.stats["fallback"] += 1
            if reply is not None and not reply.ok:
                self.stats["llm_failed"] += 1
        self.stats["latencies_ms"].append(action.latency_ms)
        self._last_reasoned_signature = self._signature(snap)
        if action.source == "llm":
            # How much did the language model actually change? The prompt carries
            # a deterministic recommendation, so without this counter the claim
            # "the LLM is doing the work" would be unfalsifiable.
            deviations = _divergence(action, snap, self.cfg)
            if deviations:
                self.stats["deviated_from_default"] += 1
                action.deviations = deviations
            else:
                self.stats["accepted_default"] += 1

        preview = clamp(action, snap, self.cfg.comfort, previous=self._held_clamped)
        self.bus.publish(
            DECISION,
            decision_id=action.decision_id,
            clock=snap.clock,
            source=action.source,
            rationale=action.rationale,
            latency_ms=round(action.latency_ms, 1),
            tool_calls=action.tool_calls,
            model=action.model,
            cache_hit=cache_hit,
            deviations=getattr(action, "deviations", []),
            requested={
                "cooling_setpoint_c": action.cooling_setpoint_c,
                "heating_setpoint_c": action.heating_setpoint_c,
                "oa_fraction": action.oa_fraction,
                "zones": action.zone_overrides,
            },
            applied={z: v for z, v in preview.zone_overrides.items()},
            clamped=preview.clamped,
            llm=reply.to_dict() if reply is not None else None,
        )

    # ------------------------------------------------------------- fallback
    def _fallback_action(self, snap: Snapshot, why: str) -> ControlAction:
        action = self.fallback.decide(snap)
        action.source = "heuristic"
        action.rationale = f"[fallback: {why}] {action.rationale}"
        # HeuristicPolicy stamps its own private counter; the decision id must
        # stay on the agent's sequence or the event log double-uses ids.
        action.decision_id = self._decision_id
        return action

    # ----------------------------------------------------- tool handlers
    def _tool_set_setpoints(self, request: dict[str, Any]) -> dict[str, Any]:
        """Backs ``set_zone_setpoints``. Builds the pending action and reports
        back exactly what the safety layer will do with it — that feedback is
        what stops the model repeating an out-of-band request."""
        snap = self._snapshots[-1] if self._snapshots else Snapshot()
        raw_zone = str(request.get("zone") or "ALL").strip().upper().replace(" ", "_")
        zone = ZONE_ALIASES.get(raw_zone, raw_zone if raw_zone in ZONE_ALIASES.values() else None)
        if raw_zone not in ZONE_ALIASES and zone is None:
            return {
                "ok": False,
                "error": f"unknown zone {raw_zone!r}",
                "valid_zones": ["OFFICE", "PROD_HALL", "PACK_STORE", "ALL"],
            }

        cooling = request.get("cooling_setpoint_c")
        heating = request.get("heating_setpoint_c")
        oa = request.get("oa_fraction")
        rationale = str(request.get("rationale") or "").strip()
        # A call with no set-point fields re-applies what is already in force.
        # That is safe, but reporting it as an applied change would mislead the
        # model into thinking it acted — so name it for what it is.
        nothing_requested = cooling is None and heating is None and oa is None
        check = validate_setpoint_request(zone or "ALL", cooling, heating, oa, self.cfg.comfort)
        if not check["ok"]:
            return {"ok": False, "error": "; ".join(check["problems"]), "nothing_applied": True}

        # Authority clamp, against the band this timestep's occupancy allows.
        #
        # Measured trade-off: *rejecting* an out-of-band request is better
        # pedagogy — the model gets a specific complaint and one more round to
        # fix it — but with a 3B model it burned the whole tool-round budget and
        # pushed most decisions into the timeout fallback. Clamping and
        # *reporting* the adjustment costs no round trip, applies a sensible
        # action now, and the correction still reaches the model through the
        # next prompt's guardrail feedback. Autonomy is preserved where it
        # matters: the band comes from occupancy, which is a fact, while the
        # choice of where to sit inside it stays the agent's.
        targets = [z for z in snap.zones if zone is None or z.name == zone]
        cooling, heating, authority_notes = _apply_authority(
            targets, cooling, heating, self.cfg.comfort
        )

        base = self._held_raw
        action = ControlAction(
            cooling_setpoint_c=_first_float(cooling, base.cooling_setpoint_c),
            heating_setpoint_c=_first_float(heating, base.heating_setpoint_c),
            oa_fraction=_first_float(oa, base.oa_fraction),
            source="llm",
            rationale=rationale or "(no rationale supplied by the model)",
        )
        if zone is None:
            for z in snap.zones:
                action.zone_overrides[z.name] = {
                    "cooling_setpoint_c": action.cooling_setpoint_c,
                    "heating_setpoint_c": action.heating_setpoint_c,
                    "oa_fraction": action.oa_fraction,
                }
        else:
            # Keep the other zones on what they already have.
            for z in snap.zones:
                if z.name == zone:
                    action.zone_overrides[z.name] = {
                        "cooling_setpoint_c": action.cooling_setpoint_c,
                        "heating_setpoint_c": action.heating_setpoint_c,
                        "oa_fraction": action.oa_fraction,
                    }
                else:
                    prev = base.zone_overrides.get(z.name, {})
                    action.zone_overrides[z.name] = {
                        "cooling_setpoint_c": prev.get("cooling_setpoint_c", z.cooling_setpoint_c),
                        "heating_setpoint_c": prev.get("heating_setpoint_c", z.heating_setpoint_c),
                        "oa_fraction": prev.get("oa_fraction", base.oa_fraction),
                    }
        self._pending_action = action

        preview = clamp(action, snap, self.cfg.comfort, previous=self._held_clamped)
        adjustments = authority_notes + preview.clamped
        return {
            "ok": True,
            "scope": zone or "ALL",
            "held_unchanged": nothing_requested,
            "note": (
                "no set-point values were supplied, so the current strategy is held"
                if nothing_requested
                else "applied"
            ),
            "applied_after_safety_layer": preview.zone_overrides,
            "adjustments": adjustments or ["none — your values were used as given"],
            "effective_from": "next simulation timestep",
        }

    def _tool_hold(self, rationale: str) -> dict[str, Any]:
        snap = self._snapshots[-1] if self._snapshots else Snapshot()
        action = self._held_raw.copy()
        action.source = "llm"
        action.rationale = f"hold: {rationale}"
        self._pending_action = action
        return {
            "ok": True,
            "held": {
                "cooling_setpoint_c": action.cooling_setpoint_c,
                "heating_setpoint_c": action.heating_setpoint_c,
                "oa_fraction": action.oa_fraction,
            },
            "clock": snap.clock,
        }

    # -------------------------------------------------------------- helpers
    def _in_force_summary(self) -> str:
        action = self._held_clamped or self._held_raw
        parts = [
            f"{zone} cool {vals['cooling_setpoint_c']:.1f}C / heat {vals['heating_setpoint_c']:.1f}C "
            f"/ OA {vals['oa_fraction']:.2f}"
            for zone, vals in list(action.zone_overrides.items())[:3]
        ]
        if not parts:
            parts = [
                f"cool {action.cooling_setpoint_c:.1f}C / heat {action.heating_setpoint_c:.1f}C "
                f"/ OA {action.oa_fraction:.2f}"
            ]
        return "; ".join(parts) + f" (source: {action.source})"

    def _signature(self, snap: Snapshot) -> tuple:
        """A coarse fingerprint of the control situation. Two timesteps with the
        same fingerprint warrant the same decision, so the second one can reuse
        the first's answer instead of paying for another inference."""
        return (
            snap.hour,
            snap.minute // 30,
            snap.grid.peak_window,
            tuple(
                (
                    z.name,
                    round(z.temp_c * 2) / 2,        # 0.5 C buckets
                    round(z.co2_ppm / 100),         # 100 ppm buckets
                    z.occupants > 0.05,
                )
                for z in snap.zones
            ),
            round(snap.outdoor_temp_c),
        )

    def _material_change(self, snap: Snapshot) -> bool:
        """Trigger an early decision when the situation has moved enough to make
        the held action stale — occupancy flip, comfort breach, IAQ escalation."""
        if self._last_reasoned_signature is None:
            return True
        if snap.comfort_violation:
            return True
        current = self._signature(snap)
        old_zones = {z[0]: z for z in self._last_reasoned_signature[3]}
        for zone in current[3]:
            old = old_zones.get(zone[0])
            if old is None:
                return True
            if old[3] != zone[3]:              # occupancy changed
                return True
            if abs(old[1] - zone[1]) >= 1.5:   # temperature moved 1.5 C
                return True
            if abs(old[2] - zone[2]) >= 3:     # CO2 moved 300 ppm
                return True
        return current[2] != self._last_reasoned_signature[2]   # peak window flipped

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.stats["latencies_ms"])
        p95 = lat[max(0, int(0.95 * len(lat)) - 1)] if lat else 0.0
        return {
            "provider": self.cfg.llm.provider,
            "model": self.cfg.llm.model,
            "agent_mode": self.cfg.agent_mode,
            "decision_interval_min": self.cfg.decision_interval_min,
            "decisions": self.stats["decisions"],
            "llm_decisions": self.stats["llm_ok"],
            "fallback_decisions": self.stats["fallback"],
            "external_decisions": self.stats["external"],
            "llm_errors": self.stats["llm_failed"],
            "cache_hits": self.stats["cache_hits"],
            "tool_calls": self.stats["tool_calls"],
            "recovered_from_text": self.stats["recovered_from_text"],
            "guardrail_interventions": self.stats["guardrail_interventions"],
            "accepted_default": self.stats["accepted_default"],
            "deviated_from_default": self.stats["deviated_from_default"],
            "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
            "p95_latency_ms": round(p95, 1),
        }


_ECHO_MARKERS = (
    "decision ",
    "building state",
    "per-zone situation",
    "recommended cooling",
    "allowed cooling",
    "safety layer intervened",
    "peak-demand limit",
    "accept the recommendation",
)


def _looks_like_echo(text: str) -> bool:
    """True when the model has copied the prompt instead of explaining itself."""
    lowered = text.lower().lstrip("-* ")
    return any(lowered.startswith(marker) for marker in _ECHO_MARKERS) or (
        sum(marker in lowered for marker in _ECHO_MARKERS) >= 2
    )


def _divergence(action: ControlAction, snap: Snapshot, cfg: RunConfig) -> list[str]:
    """Where the model's action differs from the recommended default."""
    from .policies import recommend

    out: list[str] = []
    for zone in snap.zones:
        rec = recommend(zone, snap, cfg.comfort, cfg.grid)
        applied = action.zone_overrides.get(zone.name)
        if not applied:
            continue
        cool = _first_float(applied.get("cooling_setpoint_c"), rec.cooling_setpoint_c)
        oa = _first_float(applied.get("oa_fraction"), rec.oa_fraction)
        if abs(cool - rec.cooling_setpoint_c) > 0.25:
            out.append(
                f"{zone.name}: cooling {cool:.1f} vs recommended {rec.cooling_setpoint_c:.1f}"
            )
        if abs(oa - rec.oa_fraction) > 0.05:
            out.append(f"{zone.name}: oa {oa:.2f} vs recommended {rec.oa_fraction:.2f}")
    return out


def authority_band(zone: Any, comfort: Any) -> tuple[tuple[float, float], tuple[float, float], str]:
    """The cooling and heating band this zone may be driven to right now.

    Pre-occupancy counts as occupied so that optimum start is expressible: a
    zone that is empty but fills in under two hours must be allowed to reach
    the occupied band, or it can never be pre-cooled.
    """
    in_shift = zone.minutes_until_occupied is not None and zone.minutes_until_occupied <= 0.0
    pre_start = zone.minutes_until_occupied is not None and 0 < zone.minutes_until_occupied <= 120
    occupied = zone.occupants > 0.05 or in_shift or pre_start
    state = (
        "occupied"
        if (zone.occupants > 0.05 or in_shift)
        else ("pre-occupancy" if pre_start else "unoccupied")
    )
    return comfort.cooling_bounds(occupied), comfort.heating_bounds(occupied), state


def _apply_authority(
    zones: list[Any], cooling: Any, heating: Any, comfort: Any
) -> tuple[Any, Any, list[str]]:
    """Clamp requested set-points into the tightest band across target zones."""
    notes: list[str] = []
    if not zones:
        return cooling, heating, notes
    cool_lo = max(authority_band(z, comfort)[0][0] for z in zones)
    cool_hi = min(authority_band(z, comfort)[0][1] for z in zones)
    heat_lo = max(authority_band(z, comfort)[1][0] for z in zones)
    heat_hi = min(authority_band(z, comfort)[1][1] for z in zones)
    if cool_hi < cool_lo:      # mixed occupancy across a multi-zone request
        cool_lo, cool_hi = min(cool_lo, cool_hi), max(cool_lo, cool_hi)
    if heat_hi < heat_lo:
        heat_lo, heat_hi = min(heat_lo, heat_hi), max(heat_lo, heat_hi)
    states = ", ".join(sorted({authority_band(z, comfort)[2] for z in zones}))

    if cooling is not None:
        value = _first_float(cooling, cool_lo)
        clamped = min(max(value, cool_lo), cool_hi)
        if abs(clamped - value) > 1e-6:
            notes.append(
                f"cooling {value:.1f} -> {clamped:.1f} C (allowed {cool_lo:.1f}-{cool_hi:.1f} "
                f"while {states})"
            )
        cooling = clamped
    if heating is not None:
        value = _first_float(heating, heat_lo)
        clamped = min(max(value, heat_lo), heat_hi)
        if abs(clamped - value) > 1e-6:
            notes.append(
                f"heating {value:.1f} -> {clamped:.1f} C (allowed {heat_lo:.1f}-{heat_hi:.1f} "
                f"while {states})"
            )
        heating = clamped
    return cooling, heating, notes


def _sim_minutes(snap: Snapshot) -> float:
    return ((snap.month * 31 + snap.day) * 24 + snap.hour) * 60.0 + snap.minute


def _first_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_history(path: Path, snapshots: list[Snapshot], stride: int = 1) -> None:
    """Mirror telemetry for out-of-process tool clients (see FileContext)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for snap in snapshots[::stride]:
            fh.write(json.dumps(snap.to_dict(), default=str) + "\n")
