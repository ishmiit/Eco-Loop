# Eco-Loop Building Agents

An autonomous closed-loop supervisory controller for buildings. **EnergyPlus is
the building; a local open-source LLM is the brain.** Telemetry streams out of a
running simulation, the model reasons against comfort, tariff and grid-carbon
targets, and its set-points are written back into the *same* running simulation
— no file round-trip, no restart.

```
EnergyPlus 25.2  ──telemetry──▶  MCP tool layer  ──▶  qwen2.5:3b (Ollama)
      ▲                                                      │
      └────────── set-points, same timestep ◀── safety layer ─┘
```

The demonstration building is a three-zone FSSAI-licensed food premises in
Chennai (hot-humid): a front office, a production hall and a packing store,
240 m², uninsulated RCC roof and 230 mm brick walls — representative of Indian
light-industrial stock.

---

## Result

Three simulated days, Chennai TMYx weather, EnergyPlus 25.2, qwen2.5:3b running
locally in Ollama, against a conventional fixed-schedule BMS.

<!-- RESULTS:START -->
| Metric | Baseline BMS | AI closed loop | Change |
|---|---|---|---|
| Total electricity | 780.0 kWh | 737.0 kWh | **5.5% lower** |
| HVAC electricity | 560.4 kWh | 517.3 kWh | **7.7% lower** |
| Cost | 6,441 INR | 6,076 INR | **5.7% lower** |
| Carbon | 386.8 kg | 381.6 kg | **1.3% lower** |
| Peak demand | 19,425 W | 17,306 W | **10.9% lower** |
| Peak-window energy | 100.9 kWh | 109.0 kWh | 8.0% higher |
| PMV exceedance | 0.75 zone-h | 0.00 zone-h | **better** |
| CO₂ exceedance | 0.00 zone-h | 0.00 zone-h | equal |

**Comfort: preserved.** 3.0 simulated days, 288 timesteps, energyplus 25.2.0, IND_TN_Chennai.Intl.AP.432790_TMYx.2009-2023.epw.

**Where it did worse:** peak-window energy. Reported rather than dropped — the agent shifts thermal work in time, and on this run that moved some load into the evening tariff window even while total consumption fell.

**Agent:** qwen2.5:3b · 177 decisions (177 by the model, 0 deterministic fallback) · 163 tool calls · mean 7038 ms, p95 10925 ms · safety layer intervened on 118 timesteps, diverged from the deterministic recommendation on 140/177 (79%).

**Retrofit pass (phase B):** 1 attempt, 1 verified by simulation, no generated model failed, so the repair loop was not exercised. Best: cool_roof, glazing_upgrade → +16.2% total, +23.5% HVAC against the same control on the unmodified model.

### What the language model contributes

Identical building, weather, window, plant model and safety layer. The only difference is whether a language model may disagree with the deterministic recommendation it is shown.

| | Total electricity | HVAC electricity | Peak-window energy | Comfort |
|---|---|---|---|---|
| Rules only (`--brain heuristic`) | **8.79% lower** | **12.23% lower** | +7.63% | preserved |
| qwen2.5:3b in the loop | 5.52% lower | 7.69% lower | -8.00% | preserved |

**Read this honestly: on this run the 3B model is behind the rules it is advised by.** It holds comfort just as well and shaves peak demand comparably, but its divergences from the recommendation cost energy on net. The architecture is what is being demonstrated — a real closed loop with a real safety guarantee and a measurable ablation — and the same code runs a larger model behind one flag (`--model llama3.1:8b`).

Reproduce: `python -m ecoloop run --days 3 --brain llm --ecm-pass`  ·  this exact run is committed at `artifacts/example_submission/` (open it in the dashboard on a fresh clone, no simulation required)
<!-- RESULTS:END -->

`comfort_preserved` is strict: the AI's PMV **and** CO₂ exceedance hours must
both be no worse than the baseline's. Saving energy by quietly spending comfort
counts as a failure here.

---

## Quick start

```bash
./scripts/setup.sh          # venv, deps, EnergyPlus, Chennai weather, Ollama + model
make doctor                 # four green checks
make run                    # baseline + AI closed loop + retrofit pass
make serve                  # live dashboard on http://127.0.0.1:8765
```

`setup.sh` is idempotent and needs no sudo. It downloads an official NREL
EnergyPlus build into `./vendor/` (~200 MB); nothing is installed system-wide
except the Ollama binary, and only if you let it.

Already have EnergyPlus? Point at it instead:

```bash
export ECOLOOP_ENERGYPLUS_DIR=/Applications/EnergyPlus-25-2-0
```

Prefer vLLM, llama.cpp, LM Studio or TGI over Ollama? Any OpenAI-compatible
endpoint works:

```bash
export ECOLOOP_LLM_PROVIDER=openai_compat
export ECOLOOP_LLM_URL=http://localhost:8000
export ECOLOOP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

**No EnergyPlus and no GPU?** It still runs end to end. A 2R2C surrogate of the
same building stands in, and the dashboard labels every number it produces as
surrogate output. The test suite runs in this mode.

---

## What it actually does

**Reads the building.** Ten sensor variables per zone every timestep, straight
out of the running model through the EnergyPlus C API: temperature, humidity,
CO₂, occupancy, EnergyPlus's own Fanger PMV, ideal-loads cooling and heating
rate, outdoor-air flow, lighting and equipment power. Plus five site variables
and a grid signal (carbon intensity, time-of-day tariff, peak window) — 35
sensor handles in total.

**Reasons with six levers.** Unoccupied setback, band-edge operation,
pre-cool-and-coast around the peak-tariff window, demand-controlled ventilation,
carbon-aware shifting, and optimum start using the known shift pattern.

**Writes back in-band.** Nine `Schedule:Constant` actuators — cooling set-point,
heating set-point and outdoor-air fraction per zone — written with
`set_actuator_value` and read by the thermostat predictor microseconds later in
the same timestep.

**Cannot trade comfort for energy.** Every action from every brain passes a
safety layer before it reaches an actuator: per-zone bounds, rate limit,
dead-band, CO₂ escalation, and a **predictive PMV cap** that stops the set-point
where *that* zone would reach *its own* PMV limit, linearising ISO 7730 about the
measured operating point. It is applied in the engine, not in a policy, so there
is no path around it.

**Then rewrites the building.** With `--ecm-pass` the agent reads the run
evidence, writes modified `.idf` files, and each is simulated and scored. When
one fails, the EnergyPlus error is deduplicated and handed back — it diagnoses
and retries. Every variant is kept, including the failures.

---

## Honest accounting

This is the part worth reading before judging the numbers.

**The baseline is not a straw man.** Fixed 24 °C cooling / 21 °C heating during
operating hours, night and weekend setback, design ventilation whenever occupied,
Sunday closed, Saturday a half day. That is how these buildings are run.

**Only the control policy differs between the two runs** — same IDF byte for
byte, same weather, same run period, same plant model, same safety layer.

**The LLM is advised, not autonomous-from-scratch.** A 3B model given only
general rules produced a *regression*: it held 23–24 °C through an empty building
all night and used more electricity than the baseline. So the deterministic layer
now computes a recommended set-point per zone per timestep and the prompt
presents it as a default the model may accept, adjust or override. The arithmetic
small models get wrong is done in code; the judgement is the model's.

**And the model's contribution is measured, not asserted — including when the
measurement is unflattering.** The ablation above is generated from two real
runs: on this building, qwen2.5:3b at 5.5% is *behind* the deterministic rules
at 8.8%. It diverged from the recommendation on 79% of decisions, and on net
those divergences cost energy. Both arms preserve comfort and shave peak demand
comparably.

That is the honest state of a 3B model in this loop, and it is why the ablation
is a first-class output rather than a footnote: the two arms share one
`recommend()` implementation and differ only in whether a language model may
disagree with it, so the comparison is exact. `make ablation` regenerates it, and
`--model llama3.1:8b` runs the same code against a larger model.

**The safety layer deliberately does not optimise.** Its cooling floor is 21 °C,
well below the 23 °C recommended authority, because over-cooling an empty room
wastes energy but is not *unsafe* — that is the optimiser's job. If the guardrail
quietly improved the control group, the reported savings would be understated.
There is a test for exactly that.

**Known simplifications**, stated rather than buried: interior partitions are
adiabatic; ideal-loads thermal output is converted to electricity by an explicit
plant model (cooling COP 3.2, heating COP 3.0, fan 1.0 kW per m³/s) applied
identically to both runs; the grid carbon and tariff profiles are representative
Indian shapes, not live feeds; PMV assumes 0.5 clo in summer.

Full detail, including the measurements behind each design decision, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## The dashboard

`make serve` → <http://127.0.0.1:8765>

Start a run from the browser and watch it stream: live zone tiles, the decision
feed with each rationale and latency, and a slide-over per decision showing the
model's raw tool call, what was applied per zone, and what the safety layer
changed. When the run finishes, the comparison view loads over it — cumulative
electricity, demand, per-zone temperature against the comfort band, PMV against
per-zone limits, CO₂ against the ceiling, the comfort ledger, the retrofit
attempts, and every artifact as a download.

Built to the FoodRaksha CRM design system (titanium palette, SF Pro type scale,
980px capsules, inset grouped lists) — see
[docs/DESIGN-SYSTEM.md](docs/DESIGN-SYSTEM.md). Light and dark. No CDN: it works
on an air-gapped laptop.

---

## MCP

The eleven tools are defined **once**. The same registry is rendered as
OpenAI/Ollama function schemas for the in-process agent and as MCP `tools/list`
entries for any external client, so a local Qwen model and Claude Desktop drive
the building through the identical interface.

```bash
make tools           # the registry, as documentation
make mcp             # JSON-RPC 2.0 over stdio
```

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

An external client is a **controller, not a viewer**: `set_zone_setpoints` from
MCP is queued to the run's control inbox, drained by the simulation on its next
timestep, and clamped by the same guardrail as the model's own actions.

---

## CLI

```
ecoloop doctor                    check EnergyPlus, weather, model, LLM
ecoloop run [options]             baseline + AI closed loop, then the savings
ecoloop serve                     live dashboard
ecoloop mcp                       MCP server on stdio
ecoloop runs                      list runs
ecoloop report <run_id>           re-print a savings table
ecoloop tools                     the tool registry
```

Useful `run` options:

| Option | Meaning |
|---|---|
| `--days N` | simulated days (default 3) |
| `--brain llm\|heuristic\|baseline` | which controller drives the AI arm |
| `--model qwen2.5:3b` | any Ollama or OpenAI-compatible model |
| `--decision-interval 30` | simulated minutes between decisions |
| `--agent-mode sync\|async` | block on the model per decision, or decouple |
| `--pace 0.15` | seconds of wall clock per timestep, for a live demo |
| `--ecm-pass` | also run the retrofit phase |
| `--engine surrogate` | force the fallback engine |
| `--verbose` | print every decision as it happens |

---

## Artifacts

Every run writes `artifacts/<run_id>/`:

| File | What it is |
|---|---|
| `savings.csv` | the headline comparison table |
| `results.json` | all KPIs, savings, agent and guardrail statistics |
| `telemetry_ai.csv`, `telemetry_baseline.csv` | per-timestep data for both runs |
| `decisions.jsonl` | every decision with rationale, latency and clamps |
| `events.jsonl` | the full event stream (telemetry, decisions, tool calls, logs) |
| `manifest.json` | the exact configuration — reproduces the run |
| `idf/baseline.idf`, `idf/ai.idf` | the models that were simulated |
| `idf/ecm_attempt_*.idf` | agent-generated retrofit variants |
| `eplus/*/eplustbl.htm` | EnergyPlus's own summary report |
| `ecm_report.json` | retrofit attempts, self-corrections, results |
| `report/*.png`, `report/report.pdf` | static charts (`make report`) |

---

## Tests

```bash
make test        # 186 tests, no EnergyPlus / GPU / network needed
```

Weighted towards the claims that matter: ISO 7730 PMV against the standard's
worked examples, CO₂ integrator stability at long timesteps, hostile input
through the guardrail (45 °C set-points, NaN, `null`, inverted bands), proof the
guardrail does not quietly optimise the baseline, IDF round-trip fidelity, every
ECM applying cleanly to the committed model, and MCP protocol conformance
including path-traversal refusal.

---

## Repository layout

```
models/baseline.idf        the building — heavily commented, this is the contract
models/weather/            Chennai TMYx EPW
src/ecoloop/
  sim/                     EnergyPlus engine, surrogate engine, IDF toolkit
  agent/                   LLM client, tool registry, guardrails, policies, phase B
  mcp/                     MCP server
  server/                  FastAPI + dashboard
docs/ARCHITECTURE.md       tool calling, prompts, latency, log handling
docs/DESIGN-SYSTEM.md      the visual system
docs/DEMO-SCRIPT.md        3-minute video shot list
docs/PRESENTATION-OUTLINE.md  slide-by-slide deck content
scripts/                   setup, EnergyPlus install, report export
tests/                     186 tests
```

## Requirements

Python 3.10+, ~1 GB disk for EnergyPlus, ~2 GB for the model. macOS and Linux;
`arm64` and `x86_64`. No GPU required — qwen2.5:3b runs on CPU at ~6 s per
decision, which the architecture is built to absorb.
