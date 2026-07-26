# System Architecture

Eco-Loop Building Agents — an autonomous closed-loop supervisory controller in
which EnergyPlus is the building and a local open-source LLM is the brain.

This document covers what the submission asks for: the tool-calling
architecture, the prompt-engineering strategy, prompt-latency management, and
the approach to lengthy simulation logs. It also states plainly which parts of
the decision are made by the language model and which by deterministic code,
because that distinction is the difference between a demo and a result.

---

## 1. The loop

```
                        ┌──────────────────────────────────────────┐
                        │        EnergyPlus 25.2 (in-process)      │
                        │  libenergyplusapi via pyenergyplus       │
                        │  3-zone food premises, Chennai TMYx      │
                        └───────┬───────────────────────▲──────────┘
       every zone timestep      │                       │  Schedule:Constant
       (15 simulated minutes)   │                       │  actuators, applied
                                ▼                       │  in the SAME timestep
                     ┌──────────────────────┐           │
                     │  Snapshot            │           │
                     │  10 vars × 3 zones   │           │
                     │  + 5 site + grid sig.│           │
                     └──────┬───────────────┘           │
                            │                           │
                            ▼                           │
                ┌───────────────────────────┐            │
                │  ControlPolicy            │            │
                │  · LLMPolicy  (agent)     │            │
                │  · HeuristicPolicy (rules)│            │
                │  · BaselinePolicy (control│            │
                │      group, fixed sched.) │            │
                └──────┬────────────────────┘            │
                       │ ControlAction                   │
                       ▼                                 │
                ┌───────────────────────────┐            │
                │  guardrails.clamp()       │────────────┘
                │  THE SAFETY LAYER         │
                │  every action, every brain│
                └───────────────────────────┘
```

The agent side of that box:

```
   LLMPolicy.decide()  ──▶  latest-snapshot slot  ──▶  agent thread
        │ returns the                                     │
        │ HELD action                          ┌──────────▼───────────┐
        │ immediately                          │  tool-calling loop   │
        ▼                                      │  qwen2.5:3b (Ollama) │
   engine continues                            └──────────┬───────────┘
                                                          │
                            ┌─────────────────────────────▼──────────────┐
                            │  ToolRegistry — ONE definition             │
                            │    → OpenAI/Ollama function schemas        │
                            │    → MCP tools/list (external clients)     │
                            │    → the table in section 2                │
                            └────────────────────────────────────────────┘
```

**Forward injection is genuinely in-band.** The callback runs at
`begin_zone_timestep_after_init_heat_balance`, reads sensors with
`get_variable_value`, and writes set-points with `set_actuator_value` on
`Schedule:Constant` / `Schedule Value` handles. The thermostat predictor reads
those schedules microseconds later, in the same timestep. There is no file
round-trip, no restart, and no CSV in the control path.

The nine actuated schedules are the entire control surface:

| Actuator | What it moves |
|---|---|
| `CSP_OFFICE`, `CSP_PROD`, `CSP_STORE` | cooling set-point per zone (°C) |
| `HSP_OFFICE`, `HSP_PROD`, `HSP_STORE` | heating set-point per zone (°C) |
| `OAF_OFFICE`, `OAF_PROD`, `OAF_STORE` | outdoor-air fraction, 0–1 (DCV) |

Ten sensor variables are read per zone, per timestep: mean air temperature,
relative humidity, CO₂ concentration, occupant count, **EnergyPlus's own Fanger
PMV**, ideal-loads cooling rate, ideal-loads heating rate, outdoor-air volume
flow, lighting power and equipment power. Five site variables complete the
picture: dry bulb, relative humidity, direct and diffuse solar, wind speed — 35
handles in total. A synthetic grid signal (carbon intensity, ToD tariff,
peak-window flag) is added in Python.

---

## 2. Tool-calling architecture

One registry, three consumers. `ecoloop/agent/tools.py` defines each tool once
as a `Tool` object; the same objects are rendered into OpenAI/Ollama function
schemas for the in-process agent, into MCP `inputSchema` entries for external
clients, and into this table. That is what makes "a local Qwen model and Claude
Desktop drive the building through the identical interface" a fact about the
code rather than a claim.

`python -m ecoloop tools` prints the live version of this table.

| Tool | Scope | Writes | Purpose |
|---|---|---|---|
| `get_building_state` | control | – | re-read all zone sensors |
| `get_recent_history` | control | – | trend of one metric, down-sampled to ≤12 points, with min/max/mean and direction |
| `get_control_targets` | control | – | comfort envelope, set-point authority, grid limits |
| `get_grid_forecast` | control | – | carbon and tariff for the next N hours; flags the peak window |
| `set_zone_setpoints` | control | ✔ | **the control action** — cooling/heating set-points and OA fraction |
| `hold_current_strategy` | control | ✔ | explicitly keep what is in force |
| `read_simulation_log` | analysis | – | deduplicated EnergyPlus log digest, worst first |
| `search_simulation_log` | analysis | – | regex/substring search over log entries |
| `list_idf_objects` | analysis | – | inspect the building model by IDF class |
| `list_available_ecms` | analysis | – | the retrofit measures available |
| `propose_ecm` | analysis | ✔ | write a modified `.idf`, simulate it, score it |

### Scoping

The real-time decision loop only sees the six `control` tools. The ECM pass and
external MCP clients see all eleven. Tool schemas sit in the prompt on *every*
decision, and prompt evaluation is the dominant cost (section 4) — handing a 3B
model eleven tools when six will do costs latency for nothing.

Control-scope tools are also rendered **terse**: a one-line description instead
of the full prose, and a flattened parameter schema (`Tool.brief` /
`Tool.brief_schema`). That cut the serialised tool block from 3564 to 2457
characters. External MCP clients still receive the full descriptions, because
they have no per-decision budget and their operator benefits from the prose.

### Failure handling

`ToolRegistry.call` never raises. An unknown tool returns the list of valid
names; a handler exception returns the exception text plus a hint. Both come
back to the model as a readable tool result it can act on, rather than as an
exception that ends the turn. The MCP server follows the same principle: a tool
failure is `isError: true` on a successful JSON-RPC response, never a protocol
error.

### MCP server

`python -m ecoloop mcp` speaks JSON-RPC 2.0 over stdio: `initialize`,
`tools/list`, `tools/call`, `resources/list`, `resources/read`, `ping`, and
notifications. Resources expose the baseline model, every run's results,
savings CSV, telemetry CSVs, decision log, and each generated ECM variant.

An external client is a **controller, not a viewer**. `set_zone_setpoints` from
an MCP client writes into `artifacts/<run>/live/control_inbox.jsonl`; the
running simulation drains that inbox on its next timestep and passes the request
through the identical guardrail as the model's own actions. Resource URIs are
path-contained against the artifacts tree, because a URI is
attacker-influenced input.

Claude Desktop configuration:

```json
{
  "mcpServers": {
    "ecoloop": {
      "command": "/abs/path/to/HoneyWell/.venv/bin/python",
      "args": ["-m", "ecoloop.mcp.server"],
      "env": { "PYTHONPATH": "/abs/path/to/HoneyWell/src" }
    }
  }
}
```

---

## 3. Prompt-engineering strategy

Everything below is a response to a measured failure, in the order the failures
appeared. The measurements are from qwen2.5:3b on Ollama, one simulated day,
30-minute decision cadence, EnergyPlus engine.

### 3.1 State-primed prompting

The full building state is inlined in the user message as compact JSON, so the
model can act with **zero** tool calls. When the state had to be fetched first,
the model spent its round-trip budget on retrieval and sometimes ran out before
acting. Read-only tools remain available for history and forecasts — an option,
not a prerequisite.

Compact JSON beats prose here: denser per token, and small models copy numbers
out of it more reliably than out of sentences.

### 3.2 An explicit lever menu

Rather than "minimise energy", the system prompt names six levers with the
condition under which each applies: unoccupied setback, band-edge operation,
pre-cool-and-coast, demand-controlled ventilation, carbon-aware shifting, and
optimum start. Small models reason poorly from first principles about building
physics but follow a well-formed decision procedure reliably.

### 3.3 Per-zone situation and a recommended default

This is the part worth reading carefully, because it is where the honest
accounting lives.

**Attempt 1 — general rules only.** The model held 23–24 °C through an empty
building all night and produced a **regression**: it used more electricity than
the fixed-schedule baseline it was meant to beat. It never noticed `"occ": 0`
in the state JSON.

**Attempt 2 — resolve the allowed band per zone per timestep.** Fixed the
night. Then the safety layer started reporting its clamps back into the prompt,
and the model began echoing the correction text as its own rationale while
re-requesting the same out-of-band value, so it never reached the evening
setback. Measured: **−4.1%** on total electricity. Still a regression.

**Attempt 3 — state the recommendation explicitly.** The deterministic layer
(`agent/policies.py: recommend()`) computes a recommended set-point, heating
set-point and OA fraction per zone, with the lever and the reason, and the
prompt presents it as a default the model may accept, adjust or override:

```
PER-ZONE SITUATION AND RECOMMENDED ACTION (values outside the allowed band are clamped to it):
- OFFICE: EMPTY, nobody due. Allowed cooling 26.0-30.0, heating 15.0-18.0.
  RECOMMENDED cooling 30.0, heating 15.0, oa 0.35 — lever A, empty and nobody due.
- PROD_HALL: OCCUPIED, 11 people. Allowed cooling 23.0-26.5, heating 19.0-22.0.
  RECOMMENDED cooling 26.5, heating 19.0, oa 0.80 — lever C, coast through the peak-tariff window.
```

Result: **+8.0% total electricity, +11.1% HVAC, comfort improved.**

**What this means, stated plainly.** This is an *advisory-with-override*
architecture. The arithmetic a 3B model gets wrong — resolving occupancy into a
band, tracking the shift pattern — is done in code. The judgement is left to the
model: whether to accept at all, where inside the band to sit, how much
ventilation to trade against CO₂ headroom, whether to pre-cool or coast, and
whether to treat one zone differently from the rest.

**And it is measured, not asserted.** Every run counts how often the model
diverged from the recommendation: **140 of 177 decisions (79%)** on the
submission run. `--brain heuristic` applies the recommendation verbatim, so the
ablation is exact — one `recommend()` implementation, two arms, differing only in
whether a language model may disagree with it.

The result of that ablation, three simulated days, same building and weather:

| | Total electricity | HVAC electricity | Peak-window energy |
|---|---|---|---|
| rules only | **8.79% lower** | **12.23% lower** | 7.63% lower |
| qwen2.5:3b in the loop | 5.52% lower | 7.69% lower | 8.00% **higher** |

**The 3B model is behind the rules it is advised by.** Both arms preserve
comfort (PMV exceedance 0.75 → 0.00 zone-hours) and shave peak demand
comparably (10.0% vs 10.9%), but the model's divergences cost energy on net, and
its peak-window shifting was actively counterproductive.

Reported rather than buried, because it is the actual finding and because the
mechanism it demonstrates is separable from the model's current quality: a real
in-band closed loop, a structural comfort guarantee, and a measurement that
would show a better model doing better. The same code path runs any Ollama or
OpenAI-compatible endpoint, so testing that is one flag
(`--model llama3.1:8b`).

### 3.4 A cheap correct action

`hold_current_strategy` exists so "do nothing" is a first-class answer. Without
it, a model asked for a decision invents set-point movement, which costs energy
and comfort.

### 3.5 Small defences that mattered

* **Fenced-JSON recovery.** A 3B model often answers with a JSON object instead
  of using the tool protocol. `extract_tool_calls_from_text` parses those
  replies, including bare argument objects whose tool is inferred from the keys.
  Discarding them would throw away usable decisions.
* **Argument coercion.** Ollama returns tool arguments as a dict, OpenAI as a
  JSON string, and small models sometimes double-encode. All handled in one
  place.
* **Provider-correct assistant turns.** Replaying a tool call requires an
  *object* for Ollama and a *string* for OpenAI. Getting this wrong returned
  HTTP 400 — but only on a decision's second round trip, i.e. only when the
  model had used a read-only tool first. It presented as an intermittent 5%
  failure rate.
* **Echo suppression.** When the tool's `rationale` field is empty the reply
  text is used instead — unless it is just the prompt read back, which a 3B
  model does often enough to pollute the decision log.

### 3.6 Token budget

Measured, one decision, no extra round trip:

| Component | Tokens |
|---|---|
| tool schemas (6 control tools, terse) | ~615 |
| system prompt | ~430 |
| state + advisories + limits | ~425 |
| **prompt total** | **~1470** |
| completion | 60–100 |

Down from 2339 before the trimming pass.

---

## 4. Prompt-latency management

Measured on an M-series Mac, qwen2.5:3b via Ollama:

| | |
|---|---|
| prompt evaluation | ~3.8 s |
| generation | ~2.4 s |
| **mean decision** | **6.2 s** |
| p95 decision | 7.8–10.0 s |
| EnergyPlus zone timestep | ~1 ms |

A decision costs about 6000× a timestep. Five mechanisms handle that.

### 4.1 Decoupled clocks with a latest-wins slot

`LLMPolicy.decide()` is O(zones) and returns immediately, handing back the
**held** action. The model runs on its own thread against a single-snapshot slot
that is **overwritten**, never queued. A slow decision is therefore answered
against fresh state instead of accumulating a backlog of stale work — the
classic failure of naive agent loops. A wedged or absent model cannot stall a
three-day run.

### 4.2 Continuous re-clamping of held actions

The held action passes through `guardrails.clamp` on **every** timestep, against
the *current* snapshot. So comfort protection tracks the building continuously
even though decisions arrive every 30 simulated minutes: an action that was safe
when chosen but has since become unsafe (occupants arrived, CO₂ climbed, PMV
drifted) is corrected without waiting for the next model call.

### 4.3 Bounded budget with graceful degradation

Every decision has a hard wall-clock deadline (`llm.timeout_s`, default 25 s),
enforced across the whole multi-round tool loop — a request that cannot finish
inside the remaining budget is not started. On timeout the deterministic policy
takes that decision and the run continues. The KPI records who decided, so the
fallback rate is visible rather than hidden: the validated run shows 54 model
decisions and 11 fallbacks, the fallbacks caused by CPU contention from a
concurrent test suite.

### 4.4 Event-triggered cadence

Decisions fire on a cadence (default 30 simulated minutes) **or** early on a
material change: an occupancy flip, a comfort breach, a 1.5 °C move, a 300 ppm
CO₂ move, or a peak-window transition. Rare events get a fast response without
paying for inference every timestep.

### 4.5 Prompt-shape economies

* terse tool schemas in the control scope (section 2);
* a compact state view, ~120 tokens instead of a full snapshot;
* a stable prompt prefix so Ollama's KV cache is reused across decisions —
  visible in its logs as `cached n_tokens` covering the system prompt and tool
  block;
* `keep_alive: 30m` so weights stay resident between decisions;
* a coarse situation-fingerprint cache (0.5 °C temperature buckets, 100 ppm CO₂
  buckets, occupancy flags, hour, peak flag): two timesteps with the same
  fingerprint reuse the first's decision.

### 4.6 Two modes

`--agent-mode sync` blocks the engine at each decision point, bounded by the
timeout. Every decision is genuinely the model's, which is what a reviewer wants
to watch; it is the default. `--agent-mode async` is the production shape — the
simulation runs flat out and the agent keeps up as best it can. `--pace` slows
the simulation to wall-clock for a legible live demo.

---

## 5. Handling lengthy simulation logs

A three-day run emits a few thousand EnergyPlus messages; a severe input error
can emit tens of thousands of near-identical ones. `ecoloop/logs.py` puts every
message through a three-stage digest:

1. **classify** — Fatal / Severe / Warning / Info, by EnergyPlus prefix;
2. **fingerprint** — strip quoted names, numbers, paths and timestamps, so
   "Zone OFFICE temperature 34.12 C" and "Zone PROD_HALL temperature 31.88 C"
   collapse to one entry with `count: 2` and retained examples;
3. **rank and cap** — Fatal first, then Severe, then Warning by frequency, and
   render under a caller-supplied character budget.

`read_simulation_log` returns that digest — worst-first, deduplicated,
hard-capped at ~1200 characters by default. A 40 000-line failure becomes ~600
characters that still name the offending object.
`search_simulation_log` handles the specific question, and falls back to
substring matching when the model sends an invalid regex, which it will.

Messages are captured live through `api.runtime.callback_message`, and the
`eplusout.err` file is folded in after the run. The digest is thread-safe
because EnergyPlus calls that callback from its own thread.

---

## 6. The safety layer

`ecoloop/agent/guardrails.py`. Every action from **every** brain — LLM,
heuristic, the rule-based baseline, and external MCP clients — passes
`clamp()` before it reaches an actuator. Applied in the engine, not in any one
policy, so there is no path around it.

Rules, in order (later rules win — comfort beats energy):

1. non-finite values fall back to the previous action;
2. hard safety envelope by occupancy;
3. rate limit vs. the set-points actually in force — skipped when the operating
   band itself moved, because an occupancy transition *should* step;
4. minimum dead-band between heating and cooling;
5. ventilation bounds, then CO₂ escalation (IAQ beats energy);
6. **predictive PMV cap** — cap cooling at the temperature where *this* zone
   would reach *its own* PMV limit, linearising ISO 7730 about the measured
   operating point using the zone's metabolic rate, clothing and air speed;
7. rescue, if the zone is already outside its envelope.

Rule 6 is why band-edge operation shows up as energy saved rather than comfort
hours lost. The band top is one number for the building, but PMV is not: at the
same 26.5 °C the 1.7-met production hall, the 1.4-met packing area and the
seated office sit at very different points in their own envelopes.

**The safety layer deliberately does not optimise.** Its cooling floor is
21 °C, well below the 23 °C recommended authority: cooling an empty room to
24 °C wastes energy but is not *unsafe*, so it is the optimiser's business.
If the guardrail quietly improved the control group, the reported savings would
be understated and the comparison would stop being honest. There is a test for
exactly that (`TestBaselineIsNotQuietlyOptimised`).

Every intervention is recorded in `ControlAction.clamped`, surfaced in the
dashboard, and counted in `results.json`.

---

## 7. Phase B — the agent rewrites the building model

Real-time control is bounded by what the building *is*. `--ecm-pass` lets the
agent change that: it reads the evidence from the run it just observed, calls
`propose_ecm`, and each generated `.idf` is simulated and scored.

The self-correction loop is genuine, not decorative:

```
propose → write .idf → simulate → FAIL → LogDigest → repair prompt → retry
```

When a generated model fails, the EnergyPlus error is deduplicated and handed
back with the measures that produced it; the model diagnoses and calls
`propose_ecm` again, under a hard attempt budget. Every variant is kept in
`artifacts/<run>/idf/` whether it succeeded or not.

On the submission run the repair path did **not** fire: the model called
`list_available_ecms`, then `propose_ecm` with `cool_roof` and `glazing_upgrade`
— precisely what the evidence pointed at, given a 0.75-absorptance uninsulated
roof and single clear glazing — and the generated model simulated first time for
16.2% total / 23.5% HVAC against the same control on the unmodified building.
Stated explicitly because "self-corrections: 0" in `ecm_report.json` should read
as "nothing needed correcting", not as a loop that was never wired up.

ECM variants are scored under the *deterministic* policy, against the same
policy on the unmodified model. Comparing a patched model under the heuristic
against the LLM run would mix a control change into a fabric change.

The ECM library (`sim/idf.py`): `roof_insulation`, `cool_roof`,
`glazing_upgrade`, `led_retrofit`, `window_shading`, `infiltration_sealing`,
`heat_recovery`, `demand_controlled_ventilation`. Parameters are clamped;
malformed parameters fall back to defaults; an unknown measure returns the
valid list rather than raising.

---

## 8. Experimental design

The headline claim is only worth something if exactly one thing differs between
the two runs. Held identical:

* the IDF — the same generated file, byte for byte;
* the weather file and run period;
* the plant model converting thermal load to electricity;
* the safety layer.

Only the control policy differs.

**The baseline is deliberately reasonable, not a straw man.** Fixed 24 °C
cooling / 21 °C heating during operating hours, night and weekend setback,
design ventilation whenever occupied, Sunday closed, Saturday a half day. That
is how these buildings are actually run. Beating a straw man would make the
number worthless.

**Energy accounting.** `ZoneHVAC:IdealLoadsAirSystem` reports thermal load;
`metrics.PlantModel` converts it to electricity with an explicit VRF-class plant
— cooling COP 3.2, heating COP 3.0, fan 1.0 kW per m³/s with a 25% minimum-flow
floor — applied identically to both runs, so reported savings are invariant to
the choice. Circulation fans (ceiling fans in the office and packing areas,
industrial circulators in the production hall) are modelled as real
`ElectricEquipment` with their motor heat, so the comfort benefit of air
movement is not free.

**Comfort accounting.** EnergyPlus's own Fanger PMV, per People object, against
per-zone limits: 0.8 office (1.05 met), 1.5 production hall (1.7 met), 1.1
packing (1.4 met). Exceedance is counted in zone-hours during **occupied** hours
only. `comfort_preserved` is strict: the AI's PMV *and* CO₂ exceedance must both
be ≤ the baseline's.

**Known simplifications**, stated rather than buried: interior partitions are
adiabatic; the grid carbon and tariff profiles are representative shapes, not
live feeds; the surrogate engine is a surrogate and every number it produces is
labelled as such in the dashboard; PMV assumes a fixed 0.5 clo in summer.

---

## 9. Module map

```
src/ecoloop/
  config.py            RunConfig, ComfortTargets, GridTargets, LLMConfig
  telemetry.py         Snapshot / ZoneState / ControlAction — the one contract
  metrics.py           ISO 7730 PMV, CO2 balance, plant model, KPIs, savings
  logs.py              the log digest (section 5)
  bus.py               event fan-out + events.jsonl
  weather.py           EPW resolution and a minimal EPW reader
  energyplus_locate.py finds EnergyPlus, injects pyenergyplus onto sys.path
  orchestrator.py      baseline + AI + savings + artifacts + phase B
  cli.py               python -m ecoloop
  sim/
    base.py            engine contract, zone metadata
    idf.py             dependency-free IDF reader/writer + ECM library
    schedules.py       the operating schedule, shared (occupancy foresight)
    energyplus.py      the EnergyPlus engine (section 1)
    surrogate.py       2R2C fallback so the repo runs without EnergyPlus
  agent/
    llm.py             Ollama / OpenAI-compatible / mock, tool calling
    tools.py           THE tool registry (section 2)
    context.py         LiveContext (in-process) and FileContext (out-of-process)
    prompts.py         prompt strategy (section 3)
    guardrails.py       the safety layer (section 6)
    policies.py        baseline, recommend(), heuristic
    controller.py      LLMPolicy — the latency architecture (section 4)
    ecm_agent.py       phase B and self-correction (section 7)
  mcp/server.py        MCP over stdio
  server/              FastAPI + the live dashboard
```

## 10. Testing

186 tests, no EnergyPlus, GPU or network required — the surrogate engine stands
in for the simulator and the `mock` provider for the model. Tests that need
EnergyPlus are marked and skip cleanly when it is absent.

The suite is weighted towards the claims that matter: ISO 7730 PMV against the
standard's worked examples; the CO₂ integrator's stability at long timesteps;
hostile input through the guardrail (45 °C set-points, NaN, `null`, inverted
bands); that the guardrail does not quietly optimise the baseline; IDF
round-trip fidelity; every ECM applying to the committed model; MCP protocol
conformance including path-traversal refusal. One test drives an external MCP client writing
set-points into a *running* async simulation while the agent is mid-decision,
which is the only place two threads contend for the pending-action slot.
