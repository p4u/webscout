// Hand-written SVG charts for the statistics view. No chart library: every chart is a
// function that draws into a host element at the host's current width, redraws when the
// host is resized, and returns a disposer. Labels and tooltips are built with DOM APIs
// only (textContent), never innerHTML, so API strings cannot inject markup.

const SVG_NS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else node.setAttribute(k, String(v));
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function text(x, y, value, attrs = {}) {
  const t = svg('text', { x, y, ...attrs });
  t.textContent = value;
  return t;
}

/* ------------------------------------------------------------------ ticks */

/** A "nice" step (1, 2, 2.5, 5 × 10^k) near `raw`. */
function niceStep(raw, integer) {
  if (!(raw > 0)) return 1;
  const exp = Math.floor(Math.log10(raw));
  const base = 10 ** exp;
  const f = raw / base;
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  let step = nf * base;
  if (integer) step = Math.max(1, Math.ceil(step));
  return step;
}

/** Ticks from 0 to a rounded-up maximum, about `target` intervals. */
export function niceTicks(max, { target = 4, integer = false } = {}) {
  if (!(max > 0)) max = integer ? target : 1;
  const step = niceStep(max / target, integer);
  const top = Math.ceil(max / step - 1e-9) * step;
  const ticks = [];
  for (let v = 0; v <= top + step / 2; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

/** Rough text width for tick labels (12px system font, tabular figures). */
function textWidth(s, px = 11.5) {
  return String(s).length * px * 0.58;
}

/* ----------------------------------------------------------------- tooltip */

class Tooltip {
  constructor(host) {
    this.host = host;
    this.el = document.createElement('div');
    this.el.className = 'sv-tip';
    this.el.setAttribute('role', 'status');
    this.el.hidden = true;
    host.append(this.el);
  }

  /** content: { title, rows: [{ color, label, value, strong }] , foot } */
  show(content, x, y) {
    const el = this.el;
    el.replaceChildren();
    if (content.title) {
      const h = document.createElement('div');
      h.className = 'sv-tip__title';
      h.textContent = content.title;
      el.append(h);
    }
    for (const r of content.rows ?? []) {
      const row = document.createElement('div');
      row.className = 'sv-tip__row' + (r.strong ? ' sv-tip__row--strong' : '');
      const sw = document.createElement('span');
      sw.className = 'sv-tip__swatch';
      if (r.color) sw.style.background = r.color;
      else sw.style.visibility = 'hidden';
      const label = document.createElement('span');
      label.className = 'sv-tip__label';
      label.textContent = r.label;
      const value = document.createElement('span');
      value.className = 'sv-tip__value';
      value.textContent = r.value;
      row.append(sw, label, value);
      el.append(row);
    }
    if (content.foot) {
      const f = document.createElement('div');
      f.className = 'sv-tip__foot';
      f.textContent = content.foot;
      el.append(f);
    }
    el.hidden = false;
    const hostW = this.host.clientWidth;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    // Prefer the right of the pointer; flip left near the edge.
    let left = x + 14;
    if (left + w > hostW - 4) left = x - w - 14;
    left = Math.max(4, Math.min(left, hostW - w - 4));
    const top = Math.max(0, y - h / 2);
    el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }

  hide() {
    this.el.hidden = true;
  }

  remove() {
    this.el.remove();
  }
}

/* -------------------------------------------------------------- responsive */

/**
 * Draw into `host` at its width, and again whenever that width changes.
 * `draw(width)` returns the SVG to mount. Returns a disposer.
 */
function responsive(host, draw) {
  host.classList.add('sv-chart');
  let lastW = -1;
  let raf = 0;
  const render = () => {
    raf = 0;
    const w = Math.floor(host.clientWidth);
    if (w <= 0 || w === lastW) return;
    lastW = w;
    const node = draw(w);
    const old = host.querySelector(':scope > svg');
    if (old) old.replaceWith(node);
    else host.prepend(node);
  };
  const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
    if (!raf) raf = requestAnimationFrame(render);
  }) : null;
  ro?.observe(host);
  render();
  if (lastW <= 0 && !ro) {
    // Hidden at mount with no ResizeObserver: draw at a sane default.
    lastW = 0;
    host.prepend(draw(600));
  }
  return () => {
    ro?.disconnect();
    if (raf) cancelAnimationFrame(raf);
  };
}

/* ------------------------------------------------------------ time charts */

/**
 * Pick which bucket indices carry an x label so labels never collide.
 * `preferred(i)` lets the caller favour round boundaries (midnight, Mondays…).
 */
function labelIndices(n, band, minGap, preferred) {
  const every = Math.max(1, Math.ceil(minGap / Math.max(band, 0.001)));
  const out = new Set();
  let last = Infinity;
  // Walk from the newest bucket backwards so the latest round boundary is labelled.
  for (let i = n - 1; i >= 0; i--) {
    const ok = preferred ? preferred(i, every) : (n - 1 - i) % every === 0;
    if (ok && last - i >= every) {
      out.add(i);
      last = i;
    }
  }
  if (out.size === 0) for (let i = n - 1; i >= 0; i -= every) out.add(i);
  return out;
}

function frame(width, height, { left, right = 12, top = 12, bottom = 26 }) {
  return {
    W: width,
    x0: left,
    x1: width - right,
    y0: top,
    y1: height - bottom,
    get w() {
      return this.x1 - this.x0;
    },
    get h() {
      return this.y1 - this.y0;
    },
  };
}

function yAxis(g, f, ticks, fmt, side = 'left') {
  const top = ticks[ticks.length - 1] || 1;
  for (const v of ticks) {
    const y = f.y1 - (v / top) * f.h;
    if (side === 'left') {
      g.append(svg('line', { class: 'sv-grid', x1: f.x0, x2: f.x1, y1: y, y2: y }));
      g.append(text(f.x0 - 8, y, fmt(v), { class: 'sv-axis', 'text-anchor': 'end', dy: '0.35em' }));
    } else {
      g.append(text(f.x1 + 8, y, fmt(v), { class: 'sv-axis sv-axis--right', 'text-anchor': 'start', dy: '0.35em' }));
    }
  }
}

function xAxis(g, f, n, band, xLabel, preferred) {
  const widest = Math.max(...Array.from({ length: Math.min(n, 8) }, (_, i) => textWidth(xLabel(i))), 30);
  const idx = labelIndices(n, band, widest + 14, preferred);
  for (const i of idx) {
    const cx = f.x0 + band * (i + 0.5);
    // Centre under the bucket; only a label that would overflow the SVG is pinned to
    // its edge, and it moves no further than it has to.
    let anchor = 'middle';
    let x = cx;
    const half = textWidth(xLabel(i)) / 2;
    if (cx - half < 2) {
      anchor = 'start';
      x = 2;
    } else if (cx + half > f.W - 2) {
      anchor = 'end';
      x = f.W - 2;
    }
    g.append(text(x, f.y1 + 17, xLabel(i), { class: 'sv-axis', 'text-anchor': anchor }));
  }
  g.append(svg('line', { class: 'sv-baseline', x1: f.x0, x2: f.x1, y1: f.y1, y2: f.y1 }));
}

/** Transparent columns that drive the tooltip and the hover highlight. */
function hoverColumns(root, host, tip, f, n, band, onHover) {
  const layer = svg('g', { class: 'sv-hover' });
  const hi = svg('rect', { class: 'sv-hover__band', x: 0, y: f.y0, width: band, height: f.h, visibility: 'hidden' });
  layer.append(hi);
  const cols = svg('g');
  for (let i = 0; i < n; i++) {
    cols.append(svg('rect', { x: f.x0 + band * i, y: f.y0, width: Math.max(band, 1), height: f.h + 20, fill: 'transparent', 'data-i': i }));
  }
  layer.append(cols);
  const place = (evt) => {
    const t = evt.target.closest?.('[data-i]');
    if (!t) return;
    const i = +t.getAttribute('data-i');
    hi.setAttribute('x', f.x0 + band * i);
    hi.setAttribute('visibility', 'visible');
    const rect = root.getBoundingClientRect();
    const hostRect = host.getBoundingClientRect();
    const scale = rect.width / (root.viewBox.baseVal.width || rect.width);
    const x = rect.left - hostRect.left + (f.x0 + band * (i + 0.5)) * scale;
    const content = onHover(i, true);
    tip.show(content, x, rect.top - hostRect.top + (f.y0 + f.h * 0.35) * scale);
  };
  layer.addEventListener('pointermove', place);
  layer.addEventListener('pointerdown', place);
  layer.addEventListener('pointerleave', () => {
    hi.setAttribute('visibility', 'hidden');
    tip.hide();
    onHover(-1, false);
  });
  return layer;
}

function baseSvg(width, height, label) {
  const root = svg('svg', {
    viewBox: `0 0 ${width} ${height}`,
    width: '100%',
    height,
    role: 'img',
    'aria-label': label,
    preserveAspectRatio: 'xMidYMid meet',
  });
  const t = svg('title');
  t.textContent = label;
  root.append(t);
  return root;
}

/**
 * Stacked bars over time.
 * opts: { series: [row], keys: [{ key, label, color }], xLabel(i), preferred(i, every),
 *         tipTitle(i), height, label, total(i) }
 */
export function stackedBars(host, opts) {
  const { series, keys, xLabel, tipTitle, height = 230, label = 'Stacked bar chart', preferred } = opts;
  const tip = new Tooltip(host);
  const totals = series.map((row) => keys.reduce((s, k) => s + (row[k.key] || 0), 0));
  const ticks = niceTicks(Math.max(0, ...totals), { integer: true });
  const top = ticks[ticks.length - 1];
  const fmtTick = (v) => formatCompact(v);

  const dispose = responsive(host, (width) => {
    const left = Math.max(...ticks.map((t) => textWidth(fmtTick(t)))) + 14;
    const f = frame(width, height, { left });
    const root = baseSvg(width, height, label);
    const g = svg('g');
    root.append(g);
    yAxis(g, f, ticks, fmtTick);
    const n = series.length;
    const band = f.w / Math.max(n, 1);
    const gap = band > 14 ? Math.min(band * 0.28, 10) : band > 5 ? 1.5 : 0.5;
    const barW = Math.max(band - gap, 0.8);
    const bars = svg('g', { class: 'sv-bars' });
    series.forEach((row, i) => {
      let y = f.y1;
      const x = f.x0 + band * i + (band - barW) / 2;
      const stack = keys.filter((k) => row[k.key] > 0);
      stack.forEach((k, j) => {
        const h = (row[k.key] / top) * f.h;
        y -= h;
        const isTop = j === stack.length - 1;
        const r = isTop ? Math.min(3, barW / 2, h) : 0;
        bars.append(svg('path', { d: roundedTop(x, y, barW, h, r), style: { fill: k.color } }));
      });
    });
    g.append(bars);
    xAxis(g, f, n, band, xLabel, preferred);
    root.append(
      hoverColumns(root, host, tip, f, n, band, (i) => {
        if (i < 0) return null;
        const row = series[i];
        const rows = keys
          .filter((k) => row[k.key] > 0)
          .map((k) => ({ color: k.color, label: k.label, value: formatInt(row[k.key]) }));
        return {
          title: tipTitle(i),
          rows: [{ label: 'Searches', value: formatInt(totals[i]), strong: true }, ...rows],
          foot: totals[i] === 0 ? 'No searches' : null,
        };
      }),
    );
    return root;
  });
  return () => {
    dispose();
    tip.remove();
  };
}

function roundedTop(x, y, w, h, r) {
  if (h <= 0) return '';
  if (r <= 0) return `M${x},${y + h}V${y}H${x + w}V${y + h}Z`;
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

/**
 * Lines (optionally filled) over time, with an optional right-hand axis.
 * opts: { series, lines: [{ values, label, color, area, axis: 'left'|'right', fmt }],
 *         leftFmt, rightFmt, xLabel, preferred, tipTitle, tipRows(i), height, label }
 * A null value is a gap in the line.
 */
export function lineChart(host, opts) {
  const { lines, xLabel, tipTitle, tipRows, height = 230, label = 'Line chart', preferred } = opts;
  const n = opts.length;
  const leftFmt = opts.leftFmt ?? formatCompact;
  const rightFmt = opts.rightFmt ?? formatCompact;
  const tip = new Tooltip(host);
  const maxOf = (side) =>
    Math.max(0, ...lines.filter((l) => (l.axis ?? 'left') === side).flatMap((l) => l.values.filter((v) => v != null)));
  const leftTicks = (opts.ticks ?? ((m) => niceTicks(m, { integer: opts.integer })))(maxOf('left'));
  const hasRight = lines.some((l) => l.axis === 'right');
  const rightTicks = hasRight ? niceTicks(maxOf('right')) : null;

  const dispose = responsive(host, (width) => {
    const left = Math.max(...leftTicks.map((t) => textWidth(leftFmt(t)))) + 14;
    const right = hasRight ? Math.max(...rightTicks.map((t) => textWidth(rightFmt(t)))) + 16 : 14;
    const f = frame(width, height, { left, right });
    const root = baseSvg(width, height, label);
    const g = svg('g');
    root.append(g);
    yAxis(g, f, leftTicks, leftFmt);
    if (hasRight) yAxis(g, f, rightTicks, rightFmt, 'right');
    const band = f.w / Math.max(n, 1);
    const xAt = (i) => f.x0 + band * (i + 0.5);
    const dots = [];
    for (const line of lines) {
      const ticks = line.axis === 'right' ? rightTicks : leftTicks;
      const top = ticks[ticks.length - 1] || 1;
      const yAt = (v) => f.y1 - (v / top) * f.h;
      // Split into runs of non-null values.
      const runs = [];
      let cur = [];
      line.values.forEach((v, i) => {
        if (v == null && line.connectGaps) return;
        if (v == null) {
          if (cur.length) runs.push(cur);
          cur = [];
        } else cur.push([xAt(i), yAt(v)]);
      });
      if (cur.length) runs.push(cur);
      for (const run of runs) {
        const d = run.map(([x, y], j) => `${j ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
        if (line.area && run.length > 1) {
          const a = `${d}L${run[run.length - 1][0].toFixed(1)},${f.y1}L${run[0][0].toFixed(1)},${f.y1}Z`;
          g.append(svg('path', { d: a, class: 'sv-area', style: { fill: line.color } }));
        }
        if (line.markers) {
          for (const [x, y] of run) g.append(svg('circle', { cx: x, cy: y, r: 2, style: { fill: line.color } }));
        }
        if (run.length === 1) {
          g.append(svg('circle', { cx: run[0][0], cy: run[0][1], r: 2.5, style: { fill: line.color } }));
        } else {
          g.append(
            svg('path', {
              d,
              class: 'sv-line' + (line.dashed ? ' sv-line--dashed' : ''),
              style: { stroke: line.color, strokeWidth: line.width ?? 2 },
            }),
          );
        }
      }
      const dot = svg('circle', { r: 4, class: 'sv-dot', visibility: 'hidden', style: { stroke: line.color } });
      dots.push({ dot, line, yAt });
    }
    xAxis(g, f, n, band, xLabel, preferred);
    const hover = hoverColumns(root, host, tip, f, n, band, (i, on) => {
      for (const { dot, line, yAt } of dots) {
        const v = i >= 0 ? line.values[i] : null;
        if (!on || v == null) dot.setAttribute('visibility', 'hidden');
        else {
          dot.setAttribute('cx', xAt(i));
          dot.setAttribute('cy', yAt(v));
          dot.setAttribute('visibility', 'visible');
        }
      }
      if (i < 0) return null;
      return { title: tipTitle(i), rows: tipRows(i) };
    });
    root.append(hover);
    // Dots go above the hover layer visually but must not steal the pointer.
    const dotLayer = svg('g', { style: { pointerEvents: 'none' } });
    for (const { dot } of dots) dotLayer.append(dot);
    root.append(dotLayer);
    return root;
  });
  return () => {
    dispose();
    tip.remove();
  };
}

/* ------------------------------------------------------------------ donut */

/**
 * opts: { segments: [{ label, value, color }], center, centerLabel, size, label, fmtValue }
 * Returns a disposer. Fixed size; the host lays it out.
 */
export function donut(host, opts) {
  const { segments, center, centerLabel, size = 168, label = 'Donut chart' } = opts;
  host.classList.add('sv-chart');
  const tip = new Tooltip(host);
  const total = segments.reduce((s, x) => s + x.value, 0);
  const r = size / 2;
  const thick = Math.max(16, size * 0.14);
  const ri = r - thick;
  const root = baseSvg(size, size, label);
  root.setAttribute('width', size);
  root.classList.add('sv-donut');
  const g = svg('g', { transform: `translate(${r},${r})` });
  root.append(g);
  g.append(svg('circle', { r: r - thick / 2, class: 'sv-donut__track', 'stroke-width': thick, fill: 'none' }));
  let a0 = -Math.PI / 2;
  const visible = segments.filter((s) => s.value > 0);
  const padA = visible.length > 1 ? 0.012 : 0;
  for (const s of visible) {
    const sweep = (s.value / total) * Math.PI * 2;
    const a1 = a0 + sweep;
    const path = svg('path', {
      d: arc(r - 1, ri, a0 + padA / 2, a1 - padA / 2),
      class: 'sv-donut__seg',
      style: { fill: s.color },
      tabindex: 0,
      'aria-label': `${s.label}: ${formatInt(s.value)} (${formatPct(s.value / total)})`,
    });
    const show = (evt) => {
      const hostRect = host.getBoundingClientRect();
      const pt = evt.clientX != null && evt.type !== 'focus'
        ? [evt.clientX - hostRect.left, evt.clientY - hostRect.top]
        : [size / 2, size / 2];
      tip.show(
        { title: s.label, rows: [{ color: s.color, label: 'Searches', value: `${formatInt(s.value)} · ${formatPct(s.value / total)}` }] },
        pt[0],
        pt[1],
      );
    };
    path.addEventListener('pointermove', show);
    path.addEventListener('focus', show);
    path.addEventListener('pointerleave', () => tip.hide());
    path.addEventListener('blur', () => tip.hide());
    g.append(path);
    a0 = a1;
  }
  g.append(text(0, -2, center, { class: 'sv-donut__value', 'text-anchor': 'middle' }));
  g.append(text(0, 18, centerLabel, { class: 'sv-donut__label', 'text-anchor': 'middle' }));
  host.prepend(root);
  return () => tip.remove();
}

function arc(ro, ri, a0, a1) {
  // A full circle cannot be one arc; draw two halves.
  if (a1 - a0 >= Math.PI * 2 - 1e-6) {
    const m = a0 + Math.PI;
    return arc(ro, ri, a0, m) + arc(ro, ri, m, a1);
  }
  const large = a1 - a0 > Math.PI ? 1 : 0;
  const p = (rad, a) => `${(rad * Math.cos(a)).toFixed(2)},${(rad * Math.sin(a)).toFixed(2)}`;
  return `M${p(ro, a0)}A${ro},${ro} 0 ${large} 1 ${p(ro, a1)}L${p(ri, a1)}A${ri},${ri} 0 ${large} 0 ${p(ri, a0)}Z`;
}

/* -------------------------------------------------------------- sparkline */

/** A tiny trend line; stretches to its box. */
export function sparkline(values, { color = 'var(--accent)', height = 36, label = 'Trend' } = {}) {
  const n = values.length;
  const w = 100;
  const max = Math.max(1e-9, ...values);
  const root = svg('svg', {
    viewBox: `0 0 ${w} ${height}`,
    preserveAspectRatio: 'none',
    width: '100%',
    height,
    role: 'img',
    'aria-label': label,
    class: 'sv-spark',
  });
  if (n < 2) return root;
  const pad = 3;
  const pts = values.map((v, i) => [(i / (n - 1)) * w, height - pad - (v / max) * (height - pad * 2)]);
  const d = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join('');
  root.append(svg('path', { d: `${d}L${w},${height}L0,${height}Z`, class: 'sv-spark__area', style: { fill: color } }));
  root.append(svg('path', { d, class: 'sv-spark__line', style: { stroke: color } }));
  return root;
}

/* ---------------------------------------------------------------- formats */

const intFmt = new Intl.NumberFormat('en-US');

export function formatInt(n) {
  return intFmt.format(Math.round(n || 0));
}

export function formatCompact(n) {
  if (n == null || !isFinite(n)) return '–';
  const a = Math.abs(n);
  if (a >= 1e6) return `${+(n / 1e6).toFixed(1)}M`;
  if (a >= 1e4) return `${+(n / 1e3).toFixed(0)}k`;
  if (a >= 1e3) return `${+(n / 1e3).toFixed(1)}k`;
  return `${+n.toFixed(2)}`;
}

export function formatPct(x, digits = 0) {
  if (x == null || !isFinite(x)) return '–';
  const v = x * 100;
  // Keep one decimal for small but non-zero shares so they do not read as 0%.
  const d = digits || (v > 0 && v < 1 ? 1 : 0);
  return `${v.toFixed(d)}%`;
}

/** USD: four decimals under $1, two otherwise. */
export function formatUsd(v) {
  if (v == null || !isFinite(v)) return '–';
  if (v === 0) return '$0.00';
  const a = Math.abs(v);
  const s = a < 1 ? a.toFixed(4) : intFmtUsd.format(a);
  return (v < 0 ? '−$' : '$') + s;
}

const intFmtUsd = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** Short USD for axis ticks. */
export function formatUsdTick(v) {
  if (v === 0) return '$0';
  if (v < 0.01) return `$${+v.toFixed(4)}`;
  if (v < 1) return `$${+v.toFixed(3)}`;
  if (v < 1000) return `$${+v.toFixed(2)}`;
  return `$${formatCompact(v)}`;
}

/** Seconds as "8.4 s", "3m 12s", "1h 04m". */
export function formatDuration(s) {
  if (s == null || !isFinite(s)) return '–';
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    return r ? `${m}m ${String(r).padStart(2, '0')}s` : `${m}m`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.round((s % 3600) / 60);
  return `${h}h ${String(m).padStart(2, '0')}m`;
}

/** Duration axis ticks on round seconds, minutes or hours. */
export function durationTicks(max) {
  if (max < 120) return niceTicks(max);
  if (max < 7200) return niceTicks(max / 60, { integer: true }).map((m) => m * 60);
  return niceTicks(max / 3600).map((h) => h * 3600);
}

/** Axis ticks in seconds. */
export function formatDurationTick(s) {
  if (s === 0) return '0';
  if (s < 60) return `${+s.toFixed(1)}s`;
  if (s < 3600) return `${+(s / 60).toFixed(1)}m`;
  return `${+(s / 3600).toFixed(1)}h`;
}
