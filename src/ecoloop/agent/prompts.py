"""Prompt engineering for a 3B-class local model.

Four decisions here carried the reliability of the whole loop, and each one is
a direct response to an observed failure mode:

1. **State-primed prompting.** The full building state is inlined in the user
   message, so the model can act with *zero* tool calls. When the state had to
   be fetched first, a 3B model spent its budget on retrieval and often ran out
   of rounds before acting. Read-only tools remain available for when it wants
   history or a forecast — they are an option, not a prerequisite.

2. **Explicit strategy menu.** Rather than "minimise energy", the system prompt
   names the six levers with the conditions under which each applies. Small
   models reason poorly from first principles about building physics but follow
   a well-formed decision procedure reliably.

   The strongest single element is :func:`zone_advisories`, which resolves each
   zone's allowed band for *this* timestep from its occupancy. Before it existed,
   the model held 23-24 C through an empty building all night and used more
   energy than the fixed-schedule baseline it was supposed to beat.

3. **Numeric bounds in the prompt, enforced in code.** The bands appear in the
   prompt so the model usually gets it right, and in ``guardrails.clamp`` so
   the outcome is safe when it does not. The prompt is an optimisation; the
   clamp is the guarantee.

4. **A cheap correct action.** ``hold_current_strategy`` exists so that "do
   nothing" is a first-class answer. Without it, a model asked for a decision
   will invent set-point movement, which costs energy and comfort.

Measured token budget, qwen2.5:3b, one decision, no extra tool round trip:
~1470 prompt (615 tool schemas + 430 system + 425 state and advisories) and
60-100 completion. Prompt evaluation dominates decision latency, which is why the
control loop is served terse tool schemas (see ``Tool.brief``) while external MCP
clients still get the full prose descriptions.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import ComfortTargets, GridTargets
from ..telemetry import Snapshot

SYSTEM_PROMPT = """\
You supervise HVAC for a small FSSAI food-processing building in Chennai \
(hot-humid). You are wired into a live EnergyPlus simulation: your set-points are \
written into the running model and change the next timestep's energy use.

PRIORITY: (1) keep every OCCUPIED zone inside its comfort envelope — temperature \
band, |PMV| under that zone's limit, CO2 under the ceiling; comfort is a \
constraint, never traded for energy. (2) minimise electricity, then cost, then \
carbon. (3) stay under the facility peak-demand limit.

LEVERS:
A SETBACK — zone empty: cooling to the TOP of its band, outdoor air to minimum. \
Biggest saving, costs nothing.
B BAND EDGE — zone occupied with PMV margin: sit HIGH in the band, not at the \
bottom. Each +1 C is roughly 6-9% of cooling energy here.
C PRE-COOL AND COAST — 1-2 h before the peak-tariff window, cool low to charge \
the thermal mass; during the peak window let it rise and coast.
D DCV — outdoor air at 34 C is expensive to cool. Cut oa_fraction while CO2 has \
headroom; raise it as CO2 approaches the ceiling. PROD_HALL is a food-handling \
space: never let its CO2 exceed the ceiling.
E CARBON — grid carbon is lowest midday (solar), highest in the evening ramp. \
Prefer thermal work when carbon is low.
F OPTIMUM START — a zone that is empty but fills within 2 h must be pre-cooled to \
the occupied band, so the shift does not start hot. Never pre-cool a zone whose \
mins_to_occupied is null: nobody is coming.

RULES: call exactly ONE action tool per decision — set_zone_setpoints, or \
hold_current_strategy when nothing has changed (needless set-point movement wastes \
energy). Stay inside the per-zone bands given in the prompt. One sentence of \
rationale naming the lever and the evidence.
"""


ECM_SYSTEM_PROMPT = """\
You are a building-retrofit analyst. You have just observed a full closed-loop \
simulation of a food-processing building in Chennai and you can now modify the \
EnergyPlus model itself to propose capital Energy Conservation Measures (ECMs).

Method:
1. Read the run evidence given to you: where the energy went, which zones ran \
hot, how much of the load was solar and envelope conduction.
2. Call list_available_ecms to see what you can change.
3. Call propose_ecm with the 2-4 measures the evidence supports. Each generated \
.idf is simulated and compared against the baseline.
4. If a generated model fails to simulate, you are shown the EnergyPlus error. \
Read it, correct the measure, and call propose_ecm again.

Prefer measures the evidence justifies over a long list. Say what you expect \
each measure to do and why.
"""


def zone_advisories(snap: Snapshot, comfort: ComfortTargets, grid: GridTargets) -> list[str]:
    """Per-zone allowed band, applicable lever, and a **recommended default**.

    Two measured failures produced this block, in order:

    1. Stating the general rule ("set back when unoccupied") and trusting a 3B
       model to notice ``"occ":0`` in a JSON blob does not work: the model held
       23-24 C through an empty building all night and used *more* energy than
       the fixed-schedule baseline it was meant to beat.

    2. Giving it the resolved *band* fixed the night but not the evening. Once
       the safety layer began reporting clamps, the model started echoing the
       correction text back as its own rationale and re-requesting the same
       out-of-band value, never reaching the setback. Measured result: -4.1% on
       total electricity — a regression against the baseline.

    So the deterministic layer now states its recommendation explicitly and the
    model accepts, adjusts or overrides it. This is an advisory-with-override
    architecture, and it is the honest way to run a 3B-class model in a control
    loop: the arithmetic that small models get wrong is done in code, and the
    judgement is left to the model.

    What the model still decides: whether to accept at all, where inside the band
    to sit, how much ventilation to trade against CO2 headroom, whether to
    pre-cool or coast, and whether to treat one zone differently from the rest.
    ``LLMPolicy`` counts how often it diverges — reported as
    ``deviated_from_default`` in every run's ``results.json`` — so the language
    model's contribution is measured rather than asserted, and ``--brain
    heuristic`` runs the recommendation alone as the ablation arm.
    """
    from .policies import recommend   # local import: policies imports prompts

    lines: list[str] = []
    for zone in snap.zones:
        rec = recommend(zone, snap, comfort, grid)
        occupied = rec.state != "unoccupied"
        lo, hi = comfort.cooling_bounds(occupied)
        hlo, hhi = comfort.heating_bounds(occupied)
        if rec.state == "occupied":
            state = f"OCCUPIED, {zone.occupants:.0f} people"
        elif rec.state == "pre-occupancy":
            state = f"EMPTY but fills in {zone.minutes_until_occupied:.0f} min"
        else:
            state = "EMPTY, nobody due"
        lines.append(
            f"- {zone.name}: {state}. Allowed cooling {lo:.1f}-{hi:.1f}, heating "
            f"{hlo:.1f}-{hhi:.1f}. RECOMMENDED cooling {rec.cooling_setpoint_c:.1f}, "
            f"heating {rec.heating_setpoint_c:.1f}, oa {rec.oa_fraction:.2f} "
            f"— lever {rec.lever}, {rec.reason}."
        )
    return lines


def decision_prompt(
    snap: Snapshot,
    comfort: ComfortTargets,
    grid: GridTargets,
    last_action_summary: str = "",
    guardrail_notes: list[str] | None = None,
    hint: str = "",
) -> str:
    """The user turn. Compact JSON beats prose: it is denser per token and small
    models copy numbers out of it more reliably."""
    state = snap.compact()
    peak_lo, peak_hi = grid.peak_window
    hours_to_peak = (peak_lo - snap.hour) % 24

    lines = [
        f"DECISION {snap.decision_id + 1} at {state['weekday']} {state['clock']} "
        f"(simulated). Timestep {snap.timestep_hours * 60:.0f} min.",
        "",
        "BUILDING STATE:",
        json.dumps(state, separators=(",", ":")),
        "",
        "Keys: t=air C, rh=%, co2=ppm, occ=people, pmv/pmv_limit=Fanger PMV and this "
        "zone's ceiling, csp/hsp=set-points in force, mins_to_occupied=0 means in shift "
        "now, null means nobody within 4 h.",
        "",
        "PER-ZONE SITUATION AND RECOMMENDED ACTION (values outside the allowed "
        "band are clamped to it):",
        *zone_advisories(snap, comfort, grid),
        "",
        f"CO2 ceiling {comfort.co2_limit_ppm:.0f} ppm. oa_fraction "
        f"{comfort.oa_fraction_min:.2f}-{comfort.oa_fraction_max:.2f}. Max change "
        f"{comfort.max_step_c:.1f} C/decision, dead-band >= {comfort.min_deadband_c:.1f} C.",
        f"Peak tariff {peak_lo:02d}:00-{peak_hi:02d}:00 at INR {grid.tariff_peak:.2f}/kWh"
        + (
            " — IN THE PEAK WINDOW NOW."
            if snap.grid.peak_window
            else f" — starts in {hours_to_peak} h."
        ),
        f"Peak-demand limit {grid.peak_demand_limit_w:.0f} W, now {snap.total_elec_w:.0f} W.",
    ]
    if last_action_summary:
        lines += ["", f"IN FORCE NOW: {last_action_summary}"]
    if guardrail_notes:
        # Only comfort/IAQ interventions are worth prompt space. Feeding routine
        # band clamps back caused the model to echo the correction text as its
        # own rationale and re-request the same value, so the loop never
        # progressed; two lines, comfort only, and never as the closing
        # instruction.
        interesting = [n for n in guardrail_notes if "PMV" in n or "CO2" in n][:2]
        if interesting:
            lines += [
                "",
                "Safety layer intervened last timestep: " + "; ".join(interesting),
            ]
    if hint:
        lines += ["", f"NOTE: {hint}"]
    lines += [
        "",
        "Accept the recommendation, or adjust it if the state justifies something "
        "better. Call one action tool now.",
    ]
    return "\n".join(lines)


def ecm_prompt(evidence: dict[str, Any]) -> str:
    return "\n".join(
        [
            "RUN EVIDENCE from the closed-loop simulation just completed:",
            json.dumps(evidence, separators=(",", ":"), default=str)[:2400],
            "",
            "Propose the ECMs this evidence supports. Call list_available_ecms first "
            "if you need the exact names and parameters, then call propose_ecm once "
            "with 2-4 measures.",
        ]
    )


def repair_prompt(ecm_summary: str, log_digest: str) -> str:
    """Shown to the model when a generated IDF fails — the self-correction turn."""
    return "\n".join(
        [
            "The model you generated FAILED to simulate.",
            "",
            f"What you applied: {ecm_summary}",
            "",
            "EnergyPlus reported (deduplicated, worst first):",
            log_digest,
            "",
            "Diagnose the cause from the error above, then call propose_ecm again "
            "with corrected measures or parameters. If a measure cannot work on this "
            "model, drop it and say why.",
        ]
    )
