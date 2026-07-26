# PoC demonstration video — 3-minute script

The brief asks for a maximum three-minute recording that shows the loop in
action: data moving live from EnergyPlus to the LLM, and control actions
updating the model automatically. This is a shot list that fits in 3:00 and
proves that, rather than describing it.

## Before you record

```bash
python -m ecoloop doctor          # all four checks green
```

Pre-warm the model so the first decision is not a cold-start outlier:

```bash
ollama run qwen2.5:3b "ready" >/dev/null
```

Have two terminals and a browser ready:

* **T1** — `python -m ecoloop serve` (leave running)
* **T2** — for the CLI run
* **Browser** — `http://127.0.0.1:8765`

Record at 1280×800 or larger. The dashboard is legible down to 900px wide.

---

## 0:00–0:20 — What this is

Browser on the dashboard, before starting a run.

> "A three-zone FSSAI-licensed food premises in Chennai, simulated in
> EnergyPlus 25.2. Every fifteen simulated minutes it streams zone temperature,
> humidity, CO₂, occupancy and Fanger PMV to Qwen 2.5 3B running locally in
> Ollama. The model writes set-points straight back into the running
> simulation."

Point at the two pills in the header: **EnergyPlus 25.2.0** and **qwen2.5:3b**,
both green — the environment is real, not stubbed.

## 0:20–0:50 — Start the loop, show it live

Click **Start run** (2 days, LLM brain, decision every 30 min, pacing
0.15 s/timestep so it is watchable).

The live tiles appear and begin updating. Narrate over them:

> "Left to right: outdoor conditions, facility power, grid carbon and tariff,
> and which brain is in control. Then one tile per zone — air temperature, the
> set-point in force, PMV against that zone's own limit, and CO₂."

Wait for the decision feed to appear below and scroll to it.

> "Each row is one model decision: its rationale, the set-point it chose, the
> latency, and how many tool calls it made."

## 0:50–1:30 — The loop, closed, in one row

Click a decision row to open the slide-over. This is the money shot: it shows
telemetry in, reasoning, and the action out, for one timestep.

> "The model's own words. Below that, exactly what was applied to the running
> EnergyPlus model, per zone. Below that, the raw tool call it made —
> `set_zone_setpoints` — and the full model response."

If the decision has safety-layer adjustments, point at them:

> "And here is the safety layer. Every action from every brain passes a
> predictive PMV cap, CO₂ escalation, a rate limit and a dead-band before it
> reaches an actuator. Comfort is a structural guarantee here, not something we
> hope the model respects."

## 1:30–2:00 — Terminal proof, in parallel

Switch to **T2** while the run continues. Show the loop from the other side:

```bash
python -m ecoloop tools
```

> "Eleven tools, defined once. The same registry is served to the local model as
> function schemas and to any MCP client over stdio."

Then demonstrate that an external MCP client is a *controller*, not a viewer:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_building_state","arguments":{}}}' \
  | python -m ecoloop mcp
```

> "That is live state, out of the running simulation, over MCP."

## 2:00–2:35 — The result

Back to the browser once the run has finished; the finished view loads over the
live one.

> "Baseline versus AI: the baseline is a conventional fixed-schedule BMS —
> 24 °C in operating hours, night setback, design ventilation. Not a straw man."

Point at the graphite card, then the charts:

> "Total electricity down, HVAC down, peak demand down. The gap between the two
> lines is the saving. And the comfort table — PMV and CO₂ exceedance hours are
> lower than the baseline, not higher. It did not buy the energy with comfort."

Scroll to the agent stats:

> "Decisions by the model, fallbacks, mean and p95 latency, and how often the
> model diverged from the deterministic recommendation it was shown — which is
> how we measure what the language model actually contributed rather than
> claiming it."

## 2:35–3:00 — Phase B and close

Scroll to the retrofit section (needs `--ecm-pass`; use a pre-recorded run if
the live one skipped it).

> "After the control run, the agent modifies the building model itself. It reads
> the run evidence, writes new IDF files, and each one is simulated and scored.
> When a generated model fails, the EnergyPlus error is deduplicated and handed
> back — it diagnoses the failure and tries again. Every variant is kept."

Finish on the artifacts list.

> "Everything is on disk: the savings CSV, per-timestep telemetry for both runs,
> every decision with its rationale and latency, the exact IDF that was
> simulated, and EnergyPlus's own summary report."

---

## Fallback plan

If the LLM is slow on the recording machine, record the run with
`--brain heuristic` for the live segment (it is instant and drives the identical
actuator path), and use a pre-recorded LLM run for the decision-feed and
slide-over segments. Say which is which — do not narrate a rules run as a model
run.

If EnergyPlus is unavailable, the surrogate engine keeps the whole demo working
and the dashboard labels it "Surrogate engine". Do not claim EnergyPlus numbers
from a surrogate run.

## Recording commands, for reference

```bash
# a paced 2-day LLM run, watchable in real time
python -m ecoloop run --days 2 --brain llm --decision-interval 30 --pace 0.15 --verbose

# the full submission run, with the retrofit pass
python -m ecoloop run --run-id submission --days 3 --brain llm --ecm-pass

# the ablation: identical everything, deterministic brain
python -m ecoloop run --run-id ablation --days 3 --brain heuristic

# static charts for the deck
python scripts/export_report.py --run submission
```
