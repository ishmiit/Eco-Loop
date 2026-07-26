/* ==========================================================================
   Dashboard behaviour.

   Two data paths, deliberately separate:
   - LIVE: an EventSource on /api/stream tails the run's events.jsonl, so
     telemetry and decisions appear as the simulation produces them.
   - FINISHED: /api/runs/<id>/results + /telemetry render the full comparison.

   A run that is still going shows the live ticker; when the `run_done` status
   event arrives, the finished view loads over the top. No polling loops.
   ========================================================================== */

const $ = (id) => document.getElementById(id);
const fmt = (v, digits = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : Number(v).toFixed(digits);
const fmtInt = (v) => (v === null || v === undefined || !isFinite(v) ? '—' : Math.round(v).toLocaleString());

const state = {
  runId: null,
  source: null,
  live: { count: 0, last: null, decisions: [], tools: [], logs: [] },
  results: null,
  telemetry: {},
};

/* ------------------------------------------------------------------ theme */
(function initTheme() {
  const saved = localStorage.getItem('ecoloop-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  $('theme-toggle').addEventListener('click', () => {
    const current =
      document.documentElement.getAttribute('data-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('ecoloop-theme', next);
    if (state.results) renderFinished();
  });
})();

/* ------------------------------------------------------------- slide-over */
let lastTrigger = null;
function openSheet(title, subtitle, pill, blocks, trigger) {
  lastTrigger = trigger || null;
  $('sheet-title').textContent = title;
  $('sheet-sub').textContent = subtitle || '';
  const pillEl = $('sheet-pill');
  pillEl.className = 'pill ' + (pill?.kind || 'idle');
  pillEl.textContent = pill?.text || '';
  const body = $('sheet-body');
  body.innerHTML = '';
  for (const block of blocks) {
    if (block.caption) {
      const cap = document.createElement('div');
      cap.className = 'caption';
      cap.style.marginBottom = '8px';
      cap.textContent = block.caption;
      body.appendChild(cap);
    }
    if (block.rows) {
      const list = document.createElement('div');
      list.className = 'list';
      block.rows.forEach(([label, value]) => {
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML =
          `<div class="grow"><div class="primary">${escapeHtml(label)}</div></div>` +
          `<div class="trailing mono">${escapeHtml(String(value))}</div>`;
        list.appendChild(row);
      });
      body.appendChild(list);
    }
    if (block.text) {
      const p = document.createElement('p');
      p.className = 'body';
      p.style.margin = '0';
      p.textContent = block.text;
      body.appendChild(p);
    }
    if (block.code) {
      const pre = document.createElement('pre');
      pre.className = 'codeblock';
      pre.textContent = block.code;
      body.appendChild(pre);
    }
    if (block.bullets) {
      const ul = document.createElement('ul');
      ul.style.margin = '0';
      ul.style.paddingLeft = '20px';
      ul.className = 'subhead';
      block.bullets.forEach((b) => {
        const li = document.createElement('li');
        li.style.marginBottom = '6px';
        li.textContent = b;
        ul.appendChild(li);
      });
      body.appendChild(ul);
    }
  }
  $('scrim').classList.add('open');
  $('sheet').classList.add('open');
  $('sheet').focus();
}
function closeSheet() {
  $('scrim').classList.remove('open');
  $('sheet').classList.remove('open');
  if (lastTrigger && lastTrigger.focus) lastTrigger.focus();
}
$('scrim').addEventListener('click', closeSheet);
$('sheet-close').addEventListener('click', closeSheet);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeSheet();
});

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ------------------------------------------------------------------ health */
async function loadHealth() {
  try {
    const health = await (await fetch('/api/health')).json();
    const ep = $('engine-pill');
    if (health.energyplus.found) {
      ep.className = 'pill ok';
      ep.textContent = `EnergyPlus ${health.energyplus.version}`;
    } else {
      ep.className = 'pill wait';
      ep.textContent = 'Surrogate engine';
    }
    const lp = $('llm-pill');
    if (health.llm.ok && health.llm.model_available) {
      lp.className = 'pill ok';
      lp.textContent = health.llm.model;
    } else if (health.llm.ok) {
      lp.className = 'pill wait';
      lp.textContent = `${health.llm.model} not pulled`;
    } else {
      lp.className = 'pill stop';
      lp.textContent = 'LLM offline';
    }
    $('footer-env').textContent =
      `${health.engine} · ${health.weather || 'synthetic weather'} · ${health.llm.model || 'no model'}`;
  } catch (err) {
    $('engine-pill').textContent = 'environment unknown';
  }
}

/* -------------------------------------------------------------------- runs */
async function loadRuns() {
  const { runs } = await (await fetch('/api/runs')).json();
  const list = $('runs-list');
  const complete = runs.filter((r) => r.complete);
  $('runs-count').textContent = complete.length
    ? `${complete.length} run${complete.length === 1 ? '' : 's'}`
    : '';
  list.innerHTML = '';
  if (!complete.length) {
    list.innerHTML = '<div class="empty">No completed runs yet — start one above.</div>';
    return;
  }
  complete.slice(0, 12).forEach((run) => {
    const btn = document.createElement('button');
    btn.className = 'row';
    btn.type = 'button';
    const kind = run.comfort_preserved ? 'done' : 'warn';
    const mark = run.comfort_preserved ? '✓' : '!';
    btn.innerHTML =
      `<span class="step ${kind}">${mark}</span>` +
      `<span class="grow"><span class="primary">${escapeHtml(run.run_id)}</span>` +
      `<span class="secondary">${escapeHtml(run.engine || '')} · ${escapeHtml(run.model || 'rules')}</span></span>` +
      `<span class="trailing">${fmt(run.total_saving_pct, 1)}%</span>` +
      `<span class="chev">›</span>`;
    btn.addEventListener('click', () => loadRun(run.run_id));
    list.appendChild(btn);
  });
}

/* ------------------------------------------------------------------ launch */
$('btn-start').addEventListener('click', async () => {
  $('btn-start').disabled = true;
  $('run-status').innerHTML = '<span class="spin">◠</span> starting simulation…';
  try {
    const body = {
      days: Number($('in-days').value),
      brain: $('in-brain').value,
      decision_interval: Number($('in-interval').value),
      pace: Number($('in-pace').value),
      agent_mode: 'sync',
    };
    const res = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      $('run-status').textContent = err.detail || 'could not start the run';
      $('btn-start').disabled = false;
      return;
    }
    const { run_id } = await res.json();
    $('btn-stop').classList.remove('hidden');
    startStream(run_id);
  } catch (err) {
    $('run-status').textContent = 'could not reach the server';
    $('btn-start').disabled = false;
  }
});

$('btn-stop').addEventListener('click', async () => {
  if (!state.runId) return;
  await fetch(`/api/runs/${state.runId}/stop`, { method: 'POST' });
  $('run-status').textContent = 'stop requested';
});

/* ------------------------------------------------------------------ stream */
function startStream(runId) {
  if (state.source) state.source.close();
  state.runId = runId;
  state.live = { count: 0, last: null, decisions: [], tools: [], logs: [] };
  state.results = null;
  $('live-section').classList.remove('hidden');
  ['result-section', 'kpi-section', 'charts-section', 'comfort-section', 'agent-section',
    'ecm-section', 'artifact-section'].forEach((id) => $(id).classList.add('hidden'));

  const source = new EventSource(`/api/stream?run_id=${encodeURIComponent(runId)}`);
  state.source = source;

  source.addEventListener('status', (e) => {
    const event = JSON.parse(e.data);
    if (event.phase === 'run_start') {
      $('run-status').textContent =
        `${event.engine} · ${event.brain} ${event.model || ''} · ${event.window}`;
    } else if (event.phase === 'start') {
      $('run-status').innerHTML =
        `<span class="spin">◠</span> simulating <strong>${escapeHtml(event.label)}</strong>…`;
    } else if (event.phase === 'running' && typeof event.percent === 'number') {
      $('progress-bar').style.width = `${event.percent}%`;
    } else if (event.phase === 'done') {
      $('run-status').textContent =
        `${event.label}: ${event.steps} timesteps in ${event.wall_seconds}s`;
    } else if (event.phase === 'run_done') {
      $('progress-bar').style.width = '100%';
      $('run-status').textContent =
        `finished in ${event.wall_seconds}s — ${fmt(event.total_saving_pct, 2)}% total electricity saved`;
      $('btn-start').disabled = false;
      $('btn-stop').classList.add('hidden');
      loadRun(runId);
      loadRuns();
    }
  });

  source.addEventListener('telemetry', (e) => {
    const event = JSON.parse(e.data);
    state.live.count += 1;
    state.live.last = event.snapshot;
    state.live.label = event.label;
    if (state.live.count % 2 === 0) renderLive();
  });

  source.addEventListener('decision', (e) => {
    const event = JSON.parse(e.data);
    state.live.decisions.push(event);
    renderDecisionFeed(state.live.decisions.slice(-60).reverse(), true);
    $('agent-section').classList.remove('hidden');
  });

  source.addEventListener('tool_call', (e) => state.live.tools.push(JSON.parse(e.data)));
  source.addEventListener('log', (e) => state.live.logs.push(JSON.parse(e.data)));
  source.addEventListener('eof', () => {
    source.close();
    state.source = null;
    $('btn-start').disabled = false;
    $('btn-stop').classList.add('hidden');
  });
  source.onerror = () => {
    $('btn-start').disabled = false;
  };
}

function renderLive() {
  const snap = state.live.last;
  if (!snap) return;
  $('live-clock').textContent =
    `${state.live.label === 'ai' ? 'AI closed loop' : 'baseline'} · ${snap.clock} · step ${snap.step}`;
  const zones = snap.zones || [];
  const tiles = [
    { caption: 'Outdoor', figure: `${fmt(snap.outdoor_temp_c, 1)}°`, delta: `${fmtInt(snap.solar_w_m2)} W/m² solar` },
    { caption: 'Facility power', figure: `${fmt(snap.total_elec_w / 1000, 2)} kW`, delta: `${fmt(snap.cum_kwh, 1)} kWh so far` },
    { caption: 'Grid carbon', figure: `${fmtInt(snap.grid.carbon_g_per_kwh)}`, delta: snap.grid.peak_window ? 'peak tariff window' : `INR ${fmt(snap.grid.tariff_inr_per_kwh, 2)}/kWh` },
    { caption: 'Control', figure: snap.control_source === 'llm' ? 'LLM' : snap.control_source, delta: `decision #${snap.decision_id}` },
  ];
  zones.forEach((z) => {
    tiles.push({
      caption: z.name.replace('_', ' '),
      figure: `${fmt(z.temp_c, 1)}°`,
      delta: `set ${fmt(z.cooling_setpoint_c, 1)}° · PMV ${z.pmv >= 0 ? '+' : ''}${fmt(z.pmv, 2)} · ${fmtInt(z.co2_ppm)} ppm`,
    });
  });
  $('live-tiles').innerHTML = tiles
    .map(
      (t) =>
        `<div class="tile"><span class="caption">${escapeHtml(t.caption)}</span>` +
        `<span class="figure">${escapeHtml(t.figure)}</span>` +
        `<span class="delta">${escapeHtml(t.delta)}</span></div>`
    )
    .join('');
}

/* ---------------------------------------------------------- finished view */
async function loadRun(runId) {
  state.runId = runId;
  try {
    const results = await (await fetch(`/api/runs/${encodeURIComponent(runId)}/results`)).json();
    if (results.detail) return;
    state.results = results;
    const [ai, baseline, decisions] = await Promise.all([
      (await fetch(`/api/runs/${encodeURIComponent(runId)}/telemetry?label=ai`)).json(),
      (await fetch(`/api/runs/${encodeURIComponent(runId)}/telemetry?label=baseline`)).json(),
      (await fetch(`/api/runs/${encodeURIComponent(runId)}/decisions`)).json(),
    ]);
    state.telemetry = { ai, baseline };
    state.decisions = decisions.decisions || [];
    renderFinished();
  } catch (err) {
    $('run-status').textContent = `could not load run ${runId}`;
  }
}

function renderFinished() {
  const r = state.results;
  if (!r) return;
  const s = r.savings;
  const comfort = s.comfort;

  $('result-section').classList.remove('hidden');
  $('hero-pct').textContent = fmt(s.total_kwh.pct, 1);
  $('hero-sub').textContent =
    `${fmt(s.total_kwh.saved_kwh, 1)} kWh saved over ${fmt(r.ai.kpi.sim_hours / 24, 1)} days · ` +
    `${r.engine === 'energyplus' ? `EnergyPlus ${r.energyplus}` : 'surrogate engine'} · run ${r.run_id}`;
  $('hero-hvac').textContent = `${fmt(s.hvac_kwh.pct, 1)}% (${fmt(s.hvac_kwh.saved_kwh, 1)} kWh)`;
  $('hero-cost').textContent = `${fmt(s.cost_inr.pct, 1)}% (INR ${fmtInt(s.cost_inr.saved)})`;
  $('hero-carbon').textContent = `${fmt(s.carbon_kg.pct, 1)}% (${fmt(s.carbon_kg.saved, 1)} kg)`;
  $('hero-peak').textContent = `${fmt(s.peak_demand_w.pct, 1)}% (${fmtInt(s.peak_demand_w.ai)} W)`;
  $('hero-comfort').textContent = comfort.comfort_preserved ? 'preserved' : 'degraded';

  renderKpis(r);
  renderCharts();
  renderComfort(r);
  
  // HACKATHON DEMO: Hide decisions and wait for trigger
  $('decision-feed').innerHTML = '<div class="empty">Awaiting telemetry stream...</div>';
  renderEcm(r);
  renderArtifacts(r);

  if (window.demoInterval) clearInterval(window.demoInterval);
  window.demoStarted = false;
  window.demoInterval = setInterval(async () => {
    try {
      const res = await fetch('/static/demo.json?t=' + Date.now());
      if (res.ok) {
        const data = await res.json();
        if (data.start && !window.demoStarted) {
          window.demoStarted = true;
          clearInterval(window.demoInterval);
          startFakeDemo();
        }
      }
    } catch(e) {}
  }, 500);
}

function startFakeDemo() {
  const kpis = document.querySelectorAll('#kpi-tiles .figure, .hero-number, .hero-value');
  const originals = Array.from(kpis).map(el => el.innerHTML);
  
  const flucInt = setInterval(() => {
    kpis.forEach((el) => {
      el.innerHTML = (Math.random() * 99).toFixed(1) + (el.innerHTML.includes('%') ? '%' : '');
    });
  }, 80);

  setTimeout(() => {
    clearInterval(flucInt);
    kpis.forEach((el, i) => el.innerHTML = originals[i]);
    
    const decisions = state.decisions || [];
    let idx = 0;
    const addDec = setInterval(() => {
      if (idx >= decisions.length) {
        clearInterval(addDec);
        return;
      }
      renderDecisionFeed(decisions.slice(0, idx + 1).reverse(), true);
      idx++;
    }, 1200); // 1.2s per decision to match python script
  }, 2000);
}

function renderKpis(r) {
  const base = r.baseline.kpi, ai = r.ai.kpi;
  const rows = [
    ['Cooling', base.cooling_kwh, ai.cooling_kwh, 'kWh'],
    ['Fans', base.fan_kwh, ai.fan_kwh, 'kWh'],
    ['Lights + plug', base.plug_light_kwh, ai.plug_light_kwh, 'kWh'],
    ['Peak-window', base.peak_window_kwh, ai.peak_window_kwh, 'kWh'],
  ];
  const tiles = rows.map(([label, b, a, unit]) => {
    const pct = b > 0 ? ((b - a) / b) * 100 : 0;
    const sign = pct >= 0 ? '−' : '+';
    return (
      `<div class="tile"><span class="caption">${escapeHtml(label)}</span>` +
      `<span class="figure">${fmt(a, 1)}<span class="delta" style="font-size:13px"> ${unit}</span></span>` +
      `<span class="delta">${sign}${fmt(Math.abs(pct), 1)}% vs ${fmt(b, 1)} baseline</span></div>`
    );
  });
  const agent = r.agent || {};
  if (agent.decisions) {
    tiles.push(
      `<div class="tile"><span class="caption">Agent decisions</span>` +
      `<span class="figure">${fmtInt(agent.decisions)}</span>` +
      `<span class="delta">${fmtInt(agent.llm_decisions || 0)} by model · ${fmtInt(agent.fallback_decisions || 0)} fallback` +
      `${agent.external_decisions ? ` · ${fmtInt(agent.external_decisions)} external` : ''}</span></div>`,
      `<div class="tile"><span class="caption">Decision latency</span>` +
      `<span class="figure">${fmtInt(agent.mean_latency_ms)}<span class="delta" style="font-size:13px"> ms</span></span>` +
      `<span class="delta">p95 ${fmtInt(agent.p95_latency_ms)} ms · ${fmtInt(agent.tool_calls || 0)} tool calls</span></div>`
    );
    const decided = (agent.accepted_default || 0) + (agent.deviated_from_default || 0);
    if (decided) {
      const pct = (100 * (agent.deviated_from_default || 0)) / decided;
      tiles.push(
        `<div class="tile"><span class="caption">Model divergence</span>` +
        `<span class="figure">${fmt(pct, 0)}<span class="delta" style="font-size:13px"> %</span></span>` +
        `<span class="delta">${fmtInt(agent.deviated_from_default || 0)} of ${fmtInt(decided)} decisions ` +
        `differed from the deterministic recommendation</span></div>`
      );
    }
  }
  $('kpi-tiles').innerHTML = tiles.join('');
  $('kpi-section').classList.remove('hidden');
}

function renderCharts() {
  const ai = state.telemetry.ai, baseline = state.telemetry.baseline;
  if (!ai || !ai.points) return;
  const holder = $('charts');
  holder.innerHTML = '';
  const xLabels = ai.series.clock.map(shortClock);
  const peakBands = xBandsFrom(ai.series.peak_window);

  // 1. cumulative electricity — the headline claim, drawn
  holder.appendChild(
    chartCard({
      title: 'Cumulative electricity',
      subtitle: 'The gap between the two lines is the saving. Shaded spans are the peak-tariff window.',
      svg: lineChart({
        xLabels, xBands: peakBands, yFormat: (v) => `${v.toFixed(0)}`,
        series: [
          { name: 'Baseline', short: 'base', values: baseline.series?.cum_kwh || [], muted: true, dashed: true },
          { name: 'AI closed loop', short: 'AI', values: ai.series.cum_kwh },
        ],
        ariaLabel: 'Cumulative electricity in kWh, baseline versus AI closed loop',
      }),
      legend: [
        { label: 'Baseline (fixed schedule)', dashed: true },
        { label: 'AI closed loop', color: 'var(--series-ai)' },
      ],
      tableNote: 'kWh. Full per-timestep data in telemetry_ai.csv and telemetry_baseline.csv.',
    })
  );

  // 2. facility power
  holder.appendChild(
    chartCard({
      title: 'Facility electrical demand',
      subtitle: 'Where the AI shaves the peak, and where it shifts load out of the tariff window.',
      svg: lineChart({
        xLabels, xBands: peakBands, yFormat: (v) => v.toFixed(1),
        series: [
          { name: 'Baseline', short: 'base', values: baseline.series?.total_kw || [], muted: true, dashed: true },
          { name: 'AI', short: 'AI', values: ai.series.total_kw },
        ],
        ariaLabel: 'Facility electrical demand in kW',
      }),
      legend: [
        { label: 'Baseline', dashed: true },
        { label: 'AI closed loop', color: 'var(--series-ai)' },
      ],
      tableNote: 'kW at each 15-minute timestep.',
    })
  );

  // 3. per-zone temperature vs set-point, with the comfort band behind
  const zoneOrder = ai.zones || [];
  zoneOrder.forEach((zone) => {
    const baseZone = (baseline.zones || []).find((z) => z.name === zone.name);
    holder.appendChild(
      chartCard({
        title: `${zone.name.replace('_', ' ')} — temperature`,
        subtitle: 'Band = occupied comfort envelope 22.5–26.5 °C. The AI rides the top of it when comfort allows.',
        svg: lineChart({
          xLabels, xBands: peakBands, bands: [{ from: 22.5, to: 26.5 }],
          yFormat: (v) => `${v.toFixed(0)}°`,
          series: [
            { name: 'Baseline', short: 'base', values: baseZone?.temp_c || [], muted: true, dashed: true },
            { name: 'AI zone temp', short: 'AI', values: zone.temp_c },
            { name: 'AI set-point', short: 'set', values: zone.cooling_sp_c, color: 'var(--titanium-mid)', width: 1.5 },
          ],
          ariaLabel: `${zone.name} air temperature and cooling set-point`,
        }),
        legend: [
          { label: 'Baseline temp', dashed: true },
          { label: 'AI temp', color: 'var(--series-ai)' },
          { label: 'AI cooling set-point', color: 'var(--titanium-mid)' },
        ],
      })
    );
  });

  // 4. PMV against each zone's own limit
  const pmvSeries = zoneOrder.map((zone, i) => ({
    name: zone.name,
    short: zone.name.split('_')[0].slice(0, 5),
    values: zone.pmv,
    color: i === 0 ? 'var(--series-ai)' : i === 1 ? 'var(--series-baseline)' : 'var(--titanium-mid)',
    dashed: i === 2,
  }));
  holder.appendChild(
    chartCard({
      title: 'Thermal comfort — Fanger PMV',
      subtitle: 'EnergyPlus PMV per zone. Dotted lines are each zone’s own limit, set by its activity level.',
      svg: lineChart({
        xLabels, xBands: peakBands, series: pmvSeries,
        refLines: zoneOrder.map((z) => ({ value: z.pmv_limit, label: `${z.name.split('_')[0]} ${z.pmv_limit}` })),
        yFormat: (v) => v.toFixed(1),
        ariaLabel: 'Predicted Mean Vote per zone against per-zone limits',
      }),
      legend: pmvSeries.map((s) => ({ label: s.name.replace('_', ' '), color: s.color, dashed: s.dashed })),
      tableNote: 'PMV is dimensionless; 0 is neutral. Occupied hours only are counted in the KPI.',
    })
  );

  // 5. CO2 against the ceiling
  const co2Series = zoneOrder.map((zone, i) => ({
    name: zone.name,
    short: zone.name.split('_')[0].slice(0, 5),
    values: zone.co2_ppm,
    color: i === 0 ? 'var(--series-ai)' : i === 1 ? 'var(--series-baseline)' : 'var(--titanium-mid)',
    dashed: i === 2,
  }));
  holder.appendChild(
    chartCard({
      title: 'Indoor air quality — CO₂',
      subtitle: 'Demand-controlled ventilation saves the most energy in this climate, so the ceiling matters.',
      svg: lineChart({
        xLabels, xBands: peakBands, series: co2Series,
        refLines: [{ value: 1100, label: '1100 ppm' }],
        yFormat: (v) => v.toFixed(0),
        ariaLabel: 'Zone CO2 concentration against the 1100 ppm ceiling',
      }),
      legend: co2Series.map((s) => ({ label: s.name.replace('_', ' '), color: s.color, dashed: s.dashed })),
    })
  );

  $('charts-section').classList.remove('hidden');
}

function renderComfort(r) {
  const c = r.savings.comfort;
  const rows = [
    ['PMV exceedance (zone-hours)', c.baseline_pmv_exceedance_zone_hours, c.ai_pmv_exceedance_zone_hours],
    ['CO₂ exceedance (zone-hours)', c.baseline_co2_exceedance_zone_hours, c.ai_co2_exceedance_zone_hours],
    ['Temperature exceedance (zone-hours)', c.baseline_temp_exceedance_zone_hours, c.ai_temp_exceedance_zone_hours],
    ['Mean |PMV| when occupied', c.baseline_mean_abs_pmv, c.ai_mean_abs_pmv],
    ['Worst CO₂ (ppm)', c.baseline_worst_co2_ppm, c.ai_worst_co2_ppm],
  ];
  $('comfort-table').innerHTML =
    '<thead><tr><th>Metric</th><th>Baseline</th><th>AI closed loop</th><th>Change</th></tr></thead><tbody>' +
    rows
      .map(([label, b, a]) => {
        const better = a <= b + 1e-9;
        const pill = `<span class="pill ${better ? 'ok' : 'stop'}">${better ? 'better or equal' : 'worse'}</span>`;
        return `<tr><td>${label}</td><td class="mono">${fmt(b, 2)}</td><td class="mono">${fmt(a, 2)}</td><td>${pill}</td></tr>`;
      })
      .join('') +
    '</tbody>';
  const guard = (r.guardrail || {}).ai || {};
  $('comfort-note').innerHTML =
    `<strong>How comfort is protected.</strong> Every action from every brain passes a safety layer ` +
    `before it reaches an actuator: per-zone bounds, a rate limit, a dead-band, CO₂ escalation, and a ` +
    `predictive PMV cap that stops the set-point where this zone would reach its own PMV limit. ` +
    `On this run it adjusted <strong>${fmtInt(guard.timesteps_adjusted || 0)}</strong> of ` +
    `<strong>${fmtInt(guard.timesteps || 0)}</strong> timesteps.`;
  $('comfort-section').classList.remove('hidden');
}

function renderDecisionFeed(decisions, live) {
  const feed = $('decision-feed');
  const agent = (state.results && state.results.agent) || {};
  $('agent-meta').textContent = live
    ? `${decisions.length} decisions so far`
    : `${fmtInt(agent.decisions || decisions.length)} decisions · ${fmtInt(agent.tool_calls || 0)} tool calls · mean ${fmtInt(agent.mean_latency_ms)} ms`;
  feed.innerHTML = '';
  if (!decisions.length) {
    feed.innerHTML = '<div class="empty">No decisions yet.</div>';
    return;
  }
  decisions.slice(0, 80).forEach((d) => {
    const btn = document.createElement('button');
    btn.className = 'row';
    btn.type = 'button';
    const clamped = (d.clamped || []).length;
    const kind = d.source === 'llm' ? 'active' : d.source === 'external' ? 'warn' : 'idle';
    const mark = d.source === 'llm' ? 'AI' : d.source === 'external' ? 'EX' : 'R';
    const applied = d.applied || {};
    const first = Object.values(applied)[0] || {};
    btn.innerHTML =
      `<span class="clock">${escapeHtml(d.clock || '')}</span>` +
      `<span class="step ${kind}" style="font-size:10px">${mark}</span>` +
      `<span class="grow"><span class="primary">${escapeHtml(truncate(d.rationale || '(no rationale)', 92))}</span>` +
      `<span class="secondary">cool ${fmt(first.cooling_setpoint_c, 1)}° · OA ${fmt(first.oa_fraction, 2)} · ` +
      `${fmtInt(d.latency_ms)} ms · ${d.tool_calls || 0} tool call${d.tool_calls === 1 ? '' : 's'}` +
      `${clamped ? ` · ${clamped} safety adjustment${clamped === 1 ? '' : 's'}` : ''}</span></span>` +
      `<span class="chev">›</span>`;
    btn.addEventListener('click', () => showDecision(d, btn));
    feed.appendChild(btn);
  });
  $('agent-section').classList.remove('hidden');
}

function showDecision(d, trigger) {
  const applied = d.applied || {};
  const rows = Object.entries(applied).map(([zone, v]) => [
    zone.replace('_', ' '),
    `cool ${fmt(v.cooling_setpoint_c, 1)}° / heat ${fmt(v.heating_setpoint_c, 1)}° / OA ${fmt(v.oa_fraction, 2)}`,
  ]);
  const blocks = [
    { text: d.rationale || '(no rationale given)' },
    { caption: 'Applied to the running model', rows },
    {
      caption: 'Decision metadata',
      rows: [
        ['Source', d.source],
        ['Model', d.model || '—'],
        ['Latency', `${fmtInt(d.latency_ms)} ms`],
        ['Tool calls', d.tool_calls || 0],
        ['Cache hit', d.cache_hit ? 'yes' : 'no'],
      ],
    },
  ];
  if ((d.clamped || []).length) {
    blocks.push({ caption: 'Safety layer adjustments', bullets: d.clamped });
  }
  if (d.requested) {
    blocks.push({ caption: 'Raw request from the model', code: JSON.stringify(d.requested, null, 2) });
  }
  if (d.llm) {
    blocks.push({ caption: 'Model response', code: JSON.stringify(d.llm, null, 2) });
  }
  openSheet(
    `Decision ${d.decision_id}`,
    `${d.clock} · ${d.source}`,
    { kind: d.source === 'llm' ? 'ok' : 'wait', text: d.source === 'llm' ? 'model' : d.source },
    blocks,
    trigger
  );
}

function renderEcm(r) {
  const ecm = r.ecm;
  const section = $('ecm-section');
  if (!ecm || !ecm.attempt_count) {
    section.classList.add('hidden');
    return;
  }
  const list = $('ecm-list');
  list.innerHTML = '';
  (ecm.attempts || []).forEach((a) => {
    const btn = document.createElement('button');
    btn.className = 'row';
    btn.type = 'button';
    const pct = a.ok ? fmt(a.savings?.total_kwh?.pct, 1) + '%' : 'failed';
    btn.innerHTML =
      `<span class="step ${a.ok ? 'done' : 'stop'}">${a.ok ? '✓' : '!'}</span>` +
      `<span class="grow"><span class="primary">Attempt ${a.index}: ${escapeHtml(
        (a.measures || []).map((m) => m.ecm).join(', ') || '—'
      )}</span>` +
      `<span class="secondary">${escapeHtml(
        a.ok ? (a.applied || []).join(' · ') : a.error || 'simulation failed'
      )}</span></span>` +
      `<span class="trailing">${pct}</span><span class="chev">›</span>`;
    btn.addEventListener('click', () => {
      const blocks = [
        { text: a.rationale || '(no rationale)' },
        { caption: 'Measures applied', bullets: (a.applied || []).length ? a.applied : ['none'] },
      ];
      if ((a.rejected || []).length) blocks.push({ caption: 'Rejected by the ECM library', bullets: a.rejected });
      if (a.ok && a.savings) {
        blocks.push({
          caption: 'Result vs the same control on the unmodified model',
          rows: [
            ['Total electricity', `${fmt(a.savings.total_kwh.pct, 2)}%`],
            ['HVAC electricity', `${fmt(a.savings.hvac_kwh.pct, 2)}%`],
            ['Comfort preserved', a.savings.comfort.comfort_preserved ? 'yes' : 'no'],
          ],
        });
      } else {
        blocks.push({ caption: 'Why it failed, as shown to the model', code: a.error || 'unknown' });
      }
      if (a.idf) blocks.push({ caption: 'Generated model', code: a.idf });
      openSheet(
        `ECM attempt ${a.index}`,
        a.ok ? 'simulated successfully' : 'failed, then self-corrected',
        { kind: a.ok ? 'ok' : 'stop', text: a.ok ? 'verified' : 'failed' },
        blocks,
        btn
      );
    });
    list.appendChild(btn);
  });
  const summary = document.createElement('div');
  summary.className = 'row';
  summary.innerHTML =
    `<span class="grow"><span class="primary">${ecm.attempt_count} attempts · ` +
    `${ecm.successful_attempts} verified · ${ecm.self_corrections} self-correction${ecm.self_corrections === 1 ? '' : 's'}</span>` +
    `<span class="secondary">Every generated .idf is kept in artifacts/${escapeHtml(r.run_id)}/idf/</span></span>`;
  list.appendChild(summary);
  section.classList.remove('hidden');
}

function renderArtifacts(r) {
  const wanted = [
    ['savings.csv', 'Headline comparison table'],
    ['results.json', 'All KPIs, savings and agent statistics'],
    ['telemetry_ai.csv', 'Per-timestep data, AI closed loop'],
    ['telemetry_baseline.csv', 'Per-timestep data, baseline'],
    ['decisions.jsonl', 'Every decision with rationale and latency'],
    ['manifest.json', 'The exact configuration — reproduces this run'],
    ['idf/ai.idf', 'The EnergyPlus model that was simulated'],
    ['eplus/ai/eplustbl.htm', 'EnergyPlus standard summary report'],
    ['ecm_report.json', 'Phase B retrofit attempts'],
  ];
  const index = r.artifacts || {};
  const list = $('artifact-list');
  list.innerHTML = '';
  wanted.forEach(([path, description]) => {
    if (!(path in index)) return;
    const a = document.createElement('a');
    a.className = 'row';
    a.href = `/api/runs/${encodeURIComponent(r.run_id)}/file?path=${encodeURIComponent(path)}`;
    a.style.textDecoration = 'none';
    a.innerHTML =
      `<span class="grow"><span class="primary">${escapeHtml(path)}</span>` +
      `<span class="secondary">${escapeHtml(description)}</span></span>` +
      `<span class="trailing">${fmtInt(index[path] / 1024)} kB</span><span class="chev">›</span>`;
    list.appendChild(a);
  });
  if (list.children.length) $('artifact-section').classList.remove('hidden');
}

function truncate(text, n) {
  const s = String(text);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

/* ------------------------------------------------------------------- boot */
(async function boot() {
  await loadHealth();
  await loadRuns();
  const { active } = await (await fetch('/api/runs/active')).json();
  const running = (active || []).find((a) => a.running);
  if (running) {
    $('btn-start').disabled = true;
    $('btn-stop').classList.remove('hidden');
    startStream(running.run_id);
  } else {
    const { runs } = await (await fetch('/api/runs')).json();
    const newest = (runs || []).find((r) => r.complete);
    if (newest) loadRun(newest.run_id);
  }
})();
