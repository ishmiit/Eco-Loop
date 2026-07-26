/* ==========================================================================
   Hand-rolled SVG charts. No CDN, no chart library — the dashboard has to work
   on an air-gapped laptop during a demo.

   Design rules, following the project's design system and the data-viz method:
   - Two series separated by LIGHTNESS, dash pattern and a direct end label, not
     by hue: the interface is titanium and graphite, and colour is reserved for
     application status. The pair was validated for colour-vision separation
     (worst adjacent OKLab dE 18.9), and because both marks are near-neutral the
     legend plus direct labels carry identity — never colour alone.
   - Thin marks (2px lines), recessive grid, 4px rounded data ends.
   - Comfort bands and the peak-tariff window are faint fills behind the data,
     so a breach is visible as geometry rather than needing a colour alarm.
   - One y-axis per chart. Two measures of different scale get two charts.
   ========================================================================== */

const CH = {
  W: 640, H: 260,
  pad: { top: 16, right: 62, bottom: 30, left: 46 },
};

function svgEl(name, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  return el;
}

function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0, 1];
  if (min === max) { min -= 0.5; max += 0.5; }
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 0.001; v += step) {
    out.push(Number(v.toFixed(6)));
  }
  return out.length >= 2 ? out : [min, max];
}

/**
 * Line chart with optional bands.
 * @param {object} spec
 *   series: [{ name, values, dashed, muted, labelSuffix }]
 *   x: array of x labels (hour-of-run floats used for spacing)
 *   bands: [{ from, to, kind }]           horizontal comfort bands
 *   xBands: [{ fromIndex, toIndex }]      vertical spans (peak window)
 *   refLines: [{ value, label }]
 *   yFormat: fn
 */
function lineChart(spec) {
  const { pad } = CH;
  const width = CH.W, height = spec.height || CH.H;
  const svg = svgEl('svg', {
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-label': spec.ariaLabel || spec.title || 'chart',
  });
  const series = (spec.series || []).filter((s) => s.values && s.values.length);
  if (!series.length) return svg;

  const n = Math.max(...series.map((s) => s.values.length));
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  let min = Infinity, max = -Infinity;
  for (const s of series) {
    for (const v of s.values) {
      if (v === null || !isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  for (const b of spec.bands || []) { min = Math.min(min, b.from); max = Math.max(max, b.to); }
  for (const r of spec.refLines || []) { min = Math.min(min, r.value); max = Math.max(max, r.value); }
  if (spec.yMin !== undefined) min = Math.min(min, spec.yMin);
  if (spec.yMax !== undefined) max = Math.max(max, spec.yMax);
  if (!isFinite(min)) { min = 0; max = 1; }
  const span = (max - min) || 1;
  min -= span * 0.08;
  max += span * 0.08;

  const x = (i) => pad.left + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
  const y = (v) => pad.top + plotH - ((v - min) / (max - min)) * plotH;

  // vertical spans (peak-tariff window) behind everything
  for (const band of spec.xBands || []) {
    svg.appendChild(svgEl('rect', {
      x: x(band.fromIndex), y: pad.top,
      width: Math.max(1, x(band.toIndex) - x(band.fromIndex)), height: plotH,
      fill: 'var(--peak-fill)',
    }));
  }
  // horizontal comfort bands
  for (const band of spec.bands || []) {
    const top = y(band.to), bottom = y(band.from);
    svg.appendChild(svgEl('rect', {
      x: pad.left, y: top, width: plotW, height: Math.max(1, bottom - top),
      fill: 'var(--band-fill)',
    }));
  }

  // y grid + ticks
  for (const t of niceTicks(min, max, spec.yTicks || 4)) {
    if (t < min || t > max) continue;
    svg.appendChild(svgEl('line', {
      x1: pad.left, x2: pad.left + plotW, y1: y(t), y2: y(t), class: 'axis-line',
    }));
    const label = svgEl('text', { x: pad.left - 8, y: y(t) + 4, class: 'tick-text', 'text-anchor': 'end' });
    label.textContent = spec.yFormat ? spec.yFormat(t) : String(t);
    svg.appendChild(label);
  }

  // reference lines (limits) — dotted, labelled in the right gutter and stacked
  // so several per-zone limits do not print on top of each other
  const refLabels = [];
  for (const ref of spec.refLines || []) {
    svg.appendChild(svgEl('line', {
      x1: pad.left, x2: pad.left + plotW, y1: y(ref.value), y2: y(ref.value),
      stroke: 'var(--stop-fg)', 'stroke-width': 1, 'stroke-dasharray': '2 3', opacity: 0.65,
    }));
    refLabels.push({ y: y(ref.value) + 4, text: ref.label });
  }
  refLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < refLabels.length; i += 1) {
    if (refLabels[i].y - refLabels[i - 1].y < 11) refLabels[i].y = refLabels[i - 1].y + 11;
  }
  refLabels.forEach((r) => {
    const label = svgEl('text', {
      x: pad.left + plotW + 6, y: r.y, class: 'tick-text', fill: 'var(--stop-fg)',
    });
    label.textContent = r.text;
    svg.appendChild(label);
  });

  // x ticks: day boundaries / every 6 h from the clock strings
  const xLabels = spec.xLabels || [];
  const strideTarget = Math.max(1, Math.round(n / 6));
  for (let i = 0; i < n; i += strideTarget) {
    const text = xLabels[i];
    if (!text) continue;
    const label = svgEl('text', {
      x: x(i), y: height - 10, class: 'tick-text', 'text-anchor': 'middle',
    });
    label.textContent = text;
    svg.appendChild(label);
  }
  svg.appendChild(svgEl('line', {
    x1: pad.left, x2: pad.left + plotW, y1: pad.top + plotH, y2: pad.top + plotH, class: 'axis-line',
  }));

  // Direct end labels, collected first so they can be nudged apart: two series
  // that finish at similar values would otherwise print on top of each other,
  // and these labels are what carries identity in a near-neutral palette.
  const endLabels = [];

  // series
  series.forEach((s) => {
    let d = '';
    let started = false;
    s.values.forEach((v, i) => {
      if (v === null || !isFinite(v)) { started = false; return; }
      d += `${started ? 'L' : 'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`;
      started = true;
    });
    if (!d) return;
    svg.appendChild(svgEl('path', {
      d, fill: 'none',
      stroke: s.color || (s.muted ? 'var(--series-baseline)' : 'var(--series-ai)'),
      'stroke-width': s.width || 2,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      'stroke-dasharray': s.dashed ? '6 3' : null,
      opacity: s.opacity || 1,
    }));
    const lastIndex = [...s.values.keys()].reverse().find((i) => isFinite(s.values[i]));
    if (lastIndex !== undefined && s.name) {
      endLabels.push({
        x: Math.min(x(lastIndex) + 7, width - 4),
        y: Math.max(pad.top + 9, Math.min(y(s.values[lastIndex]) + 4, pad.top + plotH)),
        text: s.short || s.name,
        fill: s.color || (s.muted ? 'var(--series-baseline)' : 'var(--series-ai)'),
      });
    }
  });

  const MIN_GAP = 12;
  endLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i += 1) {
    const gap = endLabels[i].y - endLabels[i - 1].y;
    if (gap < MIN_GAP) endLabels[i].y = endLabels[i - 1].y + MIN_GAP;
  }
  // If nudging pushed the stack off the bottom, shift the whole stack up.
  const overflow = endLabels.length
    ? endLabels[endLabels.length - 1].y - (pad.top + plotH)
    : 0;
  if (overflow > 0) endLabels.forEach((l) => { l.y -= overflow; });
  endLabels.forEach((l) => {
    const label = svgEl('text', { x: l.x, y: l.y, class: 'series-label', fill: l.fill });
    label.textContent = l.text;
    svg.appendChild(label);
  });

  return svg;
}

function chartCard({ title, subtitle, svg, legend, tableNote }) {
  const card = document.createElement('div');
  card.className = 'chart-card';
  const h = document.createElement('h3');
  h.textContent = title;
  card.appendChild(h);
  if (subtitle) {
    const p = document.createElement('p');
    p.className = 'footnote';
    p.textContent = subtitle;
    card.appendChild(p);
  }
  const holder = document.createElement('div');
  holder.className = 'chart-holder';
  holder.appendChild(svg);
  card.appendChild(holder);
  if (legend && legend.length) {
    const wrap = document.createElement('div');
    wrap.className = 'legend';
    legend.forEach((item) => {
      const el = document.createElement('span');
      el.className = 'legend-item';
      const sw = document.createElement('span');
      sw.className = 'swatch' + (item.dashed ? ' dashed' : '');
      if (!item.dashed) sw.style.background = item.color;
      el.appendChild(sw);
      el.appendChild(document.createTextNode(item.label));
      wrap.appendChild(el);
    });
    card.appendChild(wrap);
  }
  if (tableNote) {
    const p = document.createElement('p');
    p.className = 'footnote';
    p.style.marginTop = '8px';
    p.textContent = tableNote;
    card.appendChild(p);
  }
  return card;
}

function xBandsFrom(flags) {
  const out = [];
  let start = null;
  (flags || []).forEach((v, i) => {
    if (v && start === null) start = i;
    if (!v && start !== null) { out.push({ fromIndex: start, toIndex: i }); start = null; }
  });
  if (start !== null) out.push({ fromIndex: start, toIndex: flags.length - 1 });
  return out;
}

function shortClock(clock) {
  if (!clock) return '';
  const parts = String(clock).split(' ');
  return parts.length > 1 ? parts[1] : parts[0];
}
