// Statistics view: usage, outcomes, spend and latency of the searches this server ran.
// Mounted by main.js as `mountStats(container)`; returns { refresh, destroy }.
// Every string that comes from the API (queries above all) is inserted with textContent.
import './stats.css';
import {
  stackedBars,
  lineChart,
  donut,
  sparkline,
  formatInt,
  formatPct,
  formatUsd,
  formatUsdTick,
  formatDuration,
  formatDurationTick,
  formatCompact,
  durationTicks,
} from './charts.js';

const REFRESH_MS = 30_000;
const RECENT_FOLDED = 12;
const STORE_KEY = 'webscout.stats.range';

const RANGES = [
  { id: '24h', label: '24 h', long: 'last 24 hours', bucket: 'hour', days: 1 },
  { id: '7d', label: '7 d', long: 'last 7 days', bucket: 'day', days: 7, hourly: true },
  { id: '30d', label: '30 d', long: 'last 30 days', bucket: 'day', days: 30 },
  { id: '12w', label: '12 wk', long: 'last 12 weeks', bucket: 'week', days: 84 },
  { id: '1y', label: '1 yr', long: 'last year', bucket: 'week', days: 365 },
];

export const OUTCOMES = [
  { key: 'complete', label: 'Complete', color: 'var(--sv-complete)' },
  { key: 'partial', label: 'Partial', color: 'var(--sv-partial)' },
  { key: 'truncated', label: 'Truncated', color: 'var(--sv-truncated)' },
  { key: 'empty', label: 'Empty', color: 'var(--sv-empty)' },
  { key: 'failed', label: 'Failed', color: 'var(--sv-failed)' },
  { key: 'cancelled', label: 'Cancelled', color: 'var(--sv-cancelled)' },
];

const SPEND = [
  { key: 'jev_usd', label: 'Jev (judge)', color: 'var(--sv-jev)' },
  { key: 'writer_usd', label: 'Writer LLM', color: 'var(--sv-writer)' },
  { key: 'planner_usd', label: 'Planner LLM', color: 'var(--sv-planner)' },
];

const KIND_LABEL = { answer: 'Answer', harvest: 'List', unknown: 'Unknown' };

/* ------------------------------------------------------------------ DOM */

function h(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'dataset') Object.assign(node.dataset, v);
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else if (k === 'style') Object.assign(node.style, v);
      else node.setAttribute(k, v === true ? '' : String(v));
    }
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const ICON_REFRESH =
  'M17.65 6.35A7.96 7.96 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z';

function icon(d) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('aria-hidden', 'true');
  s.setAttribute('class', 'sv-icon');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', d);
  s.append(p);
  return s;
}

/* -------------------------------------------------------------- time fmt */

const MONTH_DAY_UTC = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const WEEKDAY_UTC = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' });
const MONTH_DAY = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const WEEKDAY = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
const HM = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
const HMS = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
const FULL = new Intl.DateTimeFormat('en-US', {
  weekday: 'short',
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
});
const DATE_LONG = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: 'numeric' });

// Hour buckets are shown in the viewer's local time (an hour is an hour anywhere);
// day and week buckets are UTC-aligned server-side, so they are labelled in UTC to
// keep "Sep 25" meaning the bucket the server counted.
function bucketLabel(date, bucket) {
  if (bucket === 'hour') return date.getHours() === 0 ? MONTH_DAY.format(date) : HM.format(date);
  return MONTH_DAY_UTC.format(date);
}

function bucketTitle(date, bucket) {
  if (bucket === 'hour') {
    const end = new Date(date.getTime() + 3600_000);
    return `${WEEKDAY.format(date)}, ${HM.format(date)}–${HM.format(end)}`;
  }
  if (bucket === 'day') return `${WEEKDAY_UTC.format(date)} (UTC)`;
  const end = new Date(date.getTime() + 6 * 86400_000);
  return `Week of ${MONTH_DAY_UTC.format(date)} – ${MONTH_DAY_UTC.format(end)}`;
}

/** Round label boundaries: hours on multiples of 1/2/3/4/6/12/24. */
function preferredFor(dates, bucket) {
  if (bucket !== 'hour') return undefined;
  return (i, every) => {
    const step = [1, 2, 3, 4, 6, 12, 24].find((s) => s >= every) ?? 24;
    const d = dates[i];
    return d.getMinutes() === 0 && d.getHours() % step === 0;
  };
}

function relativeTime(date, now = Date.now()) {
  const s = Math.round((now - date.getTime()) / 1000);
  if (s < 45) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const hr = Math.round(m / 60);
  if (hr < 24) return `${hr} h ago`;
  const d = Math.round(hr / 24);
  if (d === 1) return 'yesterday';
  if (d < 7) return `${d} days ago`;
  return MONTH_DAY.format(date);
}

function uptime(secs) {
  if (!(secs >= 0)) return '';
  const d = Math.floor(secs / 86400);
  const hr = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  if (d) return `${d}d ${hr}h`;
  if (hr) return `${hr}h ${m}m`;
  return `${m}m`;
}

function num(v) {
  return typeof v === 'number' && isFinite(v) ? v : 0;
}

/* ------------------------------------------------------------------ view */

export function mountStats(container) {
  const saved = (() => {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
    } catch {
      return null;
    }
  })();
  const state = {
    range: RANGES.find((r) => r.id === saved?.range) ?? RANGES[2],
    hourly: !!saved?.hourly,
    data: null,
    loading: false,
    error: null, // { kind: 'auth' | 'http' | 'network', message }
    updatedAt: null,
    controller: null,
    seq: 0,
    showAllRecent: false,
  };
  let disposers = [];
  let timer = 0;
  let destroyed = false;

  const root = h('section', { class: 'sv', 'aria-labelledby': 'sv-title' });
  const updated = h('span', { class: 'sv-updated' });
  const refreshBtn = h(
    'button',
    { class: 'sv-refresh', type: 'button', title: 'Refresh now', 'aria-label': 'Refresh statistics', onclick: () => load() },
    icon(ICON_REFRESH),
  );
  const seg = h('div', { class: 'sv-seg', role: 'radiogroup', 'aria-label': 'Time range' });
  const hourlyToggle = h('label', { class: 'sv-hourly' });
  const hourlyInput = h('input', {
    type: 'checkbox',
    onchange: () => {
      state.hourly = hourlyInput.checked;
      persist();
      load();
    },
  });
  hourlyToggle.append(hourlyInput, h('span', { text: 'Hourly' }));

  for (const r of RANGES) {
    const b = h('button', {
      type: 'button',
      role: 'radio',
      class: 'sv-seg__btn',
      dataset: { range: r.id },
      text: r.label,
      title: r.long,
      onclick: () => {
        if (state.range === r) return;
        state.range = r;
        persist();
        syncControls();
        load();
      },
    });
    seg.append(b);
  }
  seg.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    e.preventDefault();
    const i = RANGES.indexOf(state.range);
    const j = (i + (e.key === 'ArrowRight' ? 1 : -1) + RANGES.length) % RANGES.length;
    seg.children[j].click();
    seg.children[j].focus();
  });

  const header = h(
    'header',
    { class: 'sv-head' },
    h('h2', { class: 'sv-title', id: 'sv-title', text: 'Statistics' }),
    h('div', { class: 'sv-controls' }, seg, hourlyToggle, h('div', { class: 'sv-refreshbox' }, updated, refreshBtn)),
  );
  const body = h('div', { class: 'sv-body' });
  root.append(header, body);
  container.replaceChildren(root);

  function persist() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ range: state.range.id, hourly: state.hourly }));
    } catch {
      /* storage may be unavailable; the choice just is not remembered */
    }
  }

  function syncControls() {
    for (const b of seg.children) {
      const on = b.dataset.range === state.range.id;
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
    }
    hourlyToggle.hidden = !state.range.hourly;
    hourlyInput.checked = state.hourly;
    refreshBtn.classList.toggle('is-spinning', state.loading);
    refreshBtn.disabled = state.loading;
    updated.textContent = state.updatedAt ? `Updated ${HMS.format(state.updatedAt)}` : state.loading ? 'Loading…' : '';
  }

  function query() {
    const r = state.range;
    const bucket = r.hourly && state.hourly ? 'hour' : r.bucket;
    return { bucket, days: r.days };
  }

  async function load() {
    if (destroyed) return;
    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    const seq = ++state.seq;
    state.loading = true;
    syncControls();
    if (!state.data) renderSkeleton();
    const { bucket, days } = query();
    try {
      const res = await fetch(`/api/stats?bucket=${bucket}&days=${days}`, {
        credentials: 'same-origin',
        headers: { accept: 'application/json' },
        signal: controller.signal,
      });
      if (seq !== state.seq) return;
      if (res.status === 401) {
        state.error = { kind: 'auth' };
        state.data = null;
        window.dispatchEvent(new CustomEvent('webscout:login-required'));
      } else if (!res.ok) {
        let message = `The server answered ${res.status}${res.statusText ? ` ${res.statusText}` : ''}.`;
        try {
          const j = await res.json();
          if (j?.message) message = String(j.message);
        } catch {
          /* non-JSON error body */
        }
        state.error = { kind: 'http', message };
      } else {
        const data = await res.json();
        if (seq !== state.seq) return;
        state.data = data;
        state.error = null;
        state.updatedAt = new Date();
      }
    } catch (err) {
      if (err?.name === 'AbortError' || seq !== state.seq) return;
      state.error = { kind: 'network', message: 'Could not reach the server.' };
    } finally {
      if (seq === state.seq) {
        state.loading = false;
        state.controller = null;
      }
    }
    if (destroyed || seq !== state.seq) return;
    syncControls();
    render();
  }

  function clearCharts() {
    for (const d of disposers) {
      try {
        d();
      } catch {
        /* a disposer must never stop the others */
      }
    }
    disposers = [];
  }

  function renderSkeleton() {
    clearCharts();
    body.replaceChildren(
      h(
        'div',
        { class: 'sv-kpis', 'aria-hidden': 'true' },
        Array.from({ length: 6 }, () => h('div', { class: 'sv-card sv-kpi sv-skel' })),
      ),
      h('div', { class: 'sv-card sv-skel sv-skel--chart', 'aria-hidden': 'true' }),
    );
  }

  function render() {
    clearCharts();
    if (state.error?.kind === 'auth') {
      body.replaceChildren(
        h(
          'div',
          { class: 'sv-message' },
          h('p', { class: 'sv-message__title', text: 'Log in to see statistics' }),
          h('p', { class: 'sv-message__text', text: 'Statistics are available once you are signed in.' }),
        ),
      );
      return;
    }
    const banner = state.error
      ? h(
          'div',
          { class: 'sv-error', role: 'alert' },
          h('span', { text: state.data ? `Could not refresh: ${state.error.message} Showing the last data.` : `Could not load statistics. ${state.error.message}` }),
          h('button', { class: 'btn btn--tonal btn--small', type: 'button', text: 'Retry', onclick: () => load() }),
        )
      : null;
    if (!state.data) {
      body.replaceChildren(banner);
      return;
    }
    const d = state.data;
    if (num(d.all_time?.runs) === 0 && num(d.totals?.runs) === 0) {
      body.replaceChildren(...[banner, emptyState(d)].filter(Boolean));
      return;
    }
    body.replaceChildren(...[banner, ...dashboard(d)].filter(Boolean));
  }

  /* ------------------------------------------------------------ sections */

  function emptyState(d) {
    return h(
      'div',
      { class: 'sv-empty' },
      h('div', { class: 'sv-empty__art', 'aria-hidden': 'true' }, emptyArt()),
      h('p', { class: 'sv-empty__title', text: 'No searches yet' }),
      h('p', {
        class: 'sv-empty__text',
        text:
          'Every search run on this server — from this web page or from an MCP client such as Claude — is recorded here: how many ran, how they ended, how long they took and what they cost.',
      }),
      h('p', { class: 'sv-empty__hint', text: 'Run a search and come back; this page refreshes itself every 30 seconds.' }),
      serverLine(d),
    );
  }

  function dashboard(d) {
    const t = d.totals ?? {};
    const series = Array.isArray(d.series) ? d.series : [];
    const bucket = d.range?.bucket ?? query().bucket;
    const dates = series.map((s) => new Date(s.t));
    const xLabel = (i) => bucketLabel(dates[i], bucket);
    const tipTitle = (i) => bucketTitle(dates[i], bucket);
    const preferred = preferredFor(dates, bucket);
    const rangeText = state.range.long + (bucket === 'hour' && state.range.id !== '24h' ? ', hourly' : '');

    const out = [];
    out.push(kpis(d, series, rangeText));

    // --- Searches over time + outcomes
    const runsHost = h('div', { class: 'sv-plot' });
    const donutHost = h('div', { class: 'sv-donutbox' });
    out.push(
      h(
        'div',
        { class: 'sv-grid' },
        card('Searches over time', `By outcome, ${bucketNoun(bucket)}`, [legend(OUTCOMES), runsHost], 'sv-span-8'),
        card('Outcomes', rangeText, [donutHost, outcomeLegend(t)], 'sv-span-4'),
      ),
    );
    later(() => {
      disposers.push(
        stackedBars(runsHost, {
          series,
          keys: OUTCOMES,
          xLabel,
          tipTitle,
          preferred,
          label: `Searches per ${bucket}, stacked by outcome, ${rangeText}. ${formatInt(t.runs)} in total.`,
        }),
      );
      disposers.push(
        donut(donutHost, {
          segments: OUTCOMES.map((o) => ({ label: o.label, value: num(t[o.key]), color: o.color })),
          center: formatInt(t.runs),
          centerLabel: num(t.runs) === 1 ? 'search' : 'searches',
          label: `Outcomes: ${OUTCOMES.map((o) => `${formatInt(t[o.key])} ${o.label.toLowerCase()}`).join(', ')}.`,
        }),
      );
    });

    // --- Spend over time + where the money goes
    const spendHost = h('div', { class: 'sv-plot' });
    let cum = 0;
    const cost = series.map((s) => num(s.cost_usd));
    const cumulative = cost.map((c) => (cum += c));
    out.push(
      h(
        'div',
        { class: 'sv-grid' },
        card(
          'Spend over time',
          `USD ${bucketNoun(bucket)}; cumulative on the right axis`,
          [
            legend([
              { label: `Spend per ${bucket}`, color: 'var(--sv-spend)' },
              { label: `Cumulative · ${formatUsd(cum)}`, color: 'var(--sv-spend-cum)', dashed: true },
            ]),
            spendHost,
          ],
          'sv-span-8',
        ),
        card('Where the money goes', `Total ${formatUsd(num(d.cost?.total_usd ?? t.cost_usd))}`, spendBreakdown(d), 'sv-span-4'),
      ),
    );
    later(() => {
      disposers.push(
        lineChart(spendHost, {
          length: series.length,
          lines: [
            { values: cost, label: 'Spend', color: 'var(--sv-spend)', area: true },
            { values: cumulative, label: 'Cumulative', color: 'var(--sv-spend-cum)', axis: 'right', dashed: true, width: 1.5 },
          ],
          leftFmt: formatUsdTick,
          rightFmt: formatUsdTick,
          xLabel,
          tipTitle,
          preferred,
          tipRows: (i) => [
            { color: 'var(--sv-spend)', label: 'Spend', value: formatUsd(cost[i]), strong: true },
            { label: 'Searches', value: formatInt(series[i].runs) },
            { label: 'Per search', value: series[i].runs ? formatUsd(cost[i] / series[i].runs) : '–' },
            { color: 'var(--sv-spend-cum)', label: 'Cumulative', value: formatUsd(cumulative[i]) },
          ],
          label: `Spend per ${bucket}, ${rangeText}: ${formatUsd(cum)} in total.`,
        }),
      );
    });

    // --- Answers vs lists (+ presets, sources) + duration
    const durHost = h('div', { class: 'sv-plot' });
    const avg = series.map((s) => (num(s.runs) > 0 ? num(s.avg_elapsed_s) : null));
    out.push(
      h(
        'div',
        { class: 'sv-grid' },
        card('Answers vs lists', rangeText, kindsBlock(d), 'sv-span-6'),
        card(
          'Duration',
          `Average per ${bucket}`,
          [
            h(
              'div',
              { class: 'sv-durstats' },
              durStat('Median', t.p50_elapsed_s),
              durStat('90th pct', t.p90_elapsed_s),
              durStat('Mean', t.avg_elapsed_s),
            ),
            durHost,
          ],
          'sv-span-6',
        ),
      ),
    );
    later(() => {
      disposers.push(
        lineChart(durHost, {
          length: series.length,
          height: 200,
          lines: [{ values: avg, label: 'Average', color: 'var(--sv-duration)', connectGaps: true, markers: true }],
          leftFmt: formatDurationTick,
          ticks: durationTicks,
          xLabel,
          tipTitle,
          preferred,
          tipRows: (i) => [
            { color: 'var(--sv-duration)', label: 'Average', value: avg[i] == null ? '–' : formatDuration(avg[i]), strong: true },
            { label: 'Searches', value: formatInt(series[i].runs) },
          ],
          label: `Average search duration per ${bucket}, ${rangeText}. Median ${formatDuration(t.p50_elapsed_s)}, 90th percentile ${formatDuration(t.p90_elapsed_s)}.`,
        }),
      );
    });

    out.push(recentTable(d));
    out.push(serverLine(d));
    return out;
  }

  // Charts measure their host, so they are drawn after the DOM is attached.
  function later(fn) {
    queueMicrotask(() => {
      if (!destroyed) fn();
    });
  }

  function bucketNoun(bucket) {
    return bucket === 'hour' ? 'per hour' : bucket === 'day' ? 'per day (UTC)' : 'per week (UTC, Mon–Sun)';
  }

  function card(title, sub, children, cls = '') {
    return h(
      'section',
      { class: `sv-card ${cls}` },
      h('header', { class: 'sv-card__head' }, h('h3', { class: 'sv-card__title', text: title }), sub ? h('p', { class: 'sv-card__sub', text: sub }) : null),
      children,
    );
  }

  function legend(items) {
    return h(
      'ul',
      { class: 'sv-legend' },
      items.map((i) =>
        h(
          'li',
          null,
          h('span', { class: 'sv-swatch' + (i.dashed ? ' sv-swatch--line' : ''), style: { background: i.color, color: i.color } }),
          i.label,
        ),
      ),
    );
  }

  function kpis(d, series, rangeText) {
    const t = d.totals ?? {};
    const runs = num(t.runs);
    const running = num(d.server?.running_now);
    const rate = runs ? num(t.success_rate) : null;

    const searches = kpi('Searches', formatInt(runs), rangeText);
    const spark = h('div', { class: 'sv-kpi__spark' }, sparkline(series.map((s) => num(s.runs)), { label: 'Searches trend' }));
    searches.append(spark);

    const success = kpi('Success rate', rate == null ? '–' : formatPct(rate), `${formatInt(t.complete)} of ${formatInt(runs)} complete`);
    if (rate != null) {
      const bar = h('div', { class: 'sv-meter', role: 'presentation' });
      for (const o of OUTCOMES) {
        const v = num(t[o.key]);
        if (v > 0) bar.append(h('span', { style: { width: `${(v / runs) * 100}%`, background: o.color }, title: `${o.label}: ${formatInt(v)}` }));
      }
      success.append(bar);
    }

    const spend = kpi('Spend', formatUsd(num(t.cost_usd)), rangeText);
    const per = kpi('Cost per search', runs ? formatUsd(num(t.avg_cost_usd)) : '–', kindCostLine(d));
    const dur = kpi('Duration', runs ? formatDuration(num(t.p50_elapsed_s)) : '–', runs ? `median · p90 ${formatDuration(num(t.p90_elapsed_s))}` : 'median');

    const live = kpi('Running now', formatInt(running), running ? (running === 1 ? 'search in progress' : 'searches in progress') : 'idle');
    live.classList.toggle('is-live', running > 0);
    const dot = h('span', { class: 'sv-pulse', 'aria-hidden': 'true' });
    live.querySelector('.sv-kpi__label').prepend(dot);

    return h('div', { class: 'sv-kpis' }, searches, success, spend, per, dur, live);
  }

  function kindCostLine(d) {
    const a = d.by_kind?.answer;
    const hv = d.by_kind?.harvest;
    const parts = [];
    if (num(a?.runs)) parts.push(`answer ${formatUsd(num(a.avg_cost_usd))}`);
    if (num(hv?.runs)) parts.push(`list ${formatUsd(num(hv.avg_cost_usd))}`);
    return parts.join(' · ') || 'average';
  }

  function kpi(label, value, sub) {
    return h(
      'div',
      { class: 'sv-card sv-kpi' },
      h('div', { class: 'sv-kpi__label', text: label }),
      h('div', { class: 'sv-kpi__value', text: value }),
      h('div', { class: 'sv-kpi__sub', text: sub }),
    );
  }

  function outcomeLegend(t) {
    const runs = num(t.runs);
    return h(
      'table',
      { class: 'sv-otable' },
      h(
        'tbody',
        null,
        OUTCOMES.map((o) => {
          const v = num(t[o.key]);
          return h(
            'tr',
            { class: v ? '' : 'is-zero' },
            h('td', null, h('span', { class: 'sv-swatch', style: { background: o.color } }), o.label),
            h('td', { class: 'sv-num', text: formatInt(v) }),
            h('td', { class: 'sv-num sv-muted', text: runs ? formatPct(v / runs) : '–' }),
          );
        }),
      ),
    );
  }

  function spendBreakdown(d) {
    const c = d.cost ?? {};
    const tk = d.tokens ?? {};
    const total = SPEND.reduce((s, x) => s + num(c[x.key]), 0);
    const bar = h('div', { class: 'sv-stackbar', role: 'img', 'aria-label': SPEND.map((x) => `${x.label} ${formatUsd(num(c[x.key]))}`).join(', ') });
    for (const x of SPEND) {
      const v = num(c[x.key]);
      if (v > 0 && total > 0) bar.append(h('span', { style: { width: `${(v / total) * 100}%`, background: x.color }, title: `${x.label}: ${formatUsd(v)}` }));
    }
    const tokens = {
      jev_usd: `${formatCompact(num(tk.jev_input))} input tokens`,
      writer_usd: `${formatCompact(num(tk.writer_prompt))} in · ${formatCompact(num(tk.writer_completion))} out`,
      planner_usd: `${formatCompact(num(tk.planner_prompt))} in · ${formatCompact(num(tk.planner_completion))} out`,
    };
    const rows = SPEND.map((x) => {
      const v = num(c[x.key]);
      return h(
        'li',
        { class: 'sv-spendrow' },
        h('span', { class: 'sv-swatch', style: { background: x.color } }),
        h('span', { class: 'sv-spendrow__label' }, h('span', { text: x.label }), h('small', { text: tokens[x.key] })),
        h('span', { class: 'sv-spendrow__value' }, h('span', { class: 'sv-num', text: formatUsd(v) }), h('small', { class: 'sv-num', text: total ? formatPct(v / total) : '–' })),
      );
    });
    const note = h('p', {
      class: 'sv-fine',
      text: 'Writer and planner costs are counted only when the model endpoint reports a price.',
    });
    return [bar, h('ul', { class: 'sv-spendlist' }, rows), note];
  }

  function kindsBlock(d) {
    const t = d.totals ?? {};
    const k = d.by_kind ?? {};
    const block = (name, desc, s) =>
      h(
        'div',
        { class: 'sv-kind' },
        h('div', { class: 'sv-kind__head' }, h('span', { class: 'sv-kind__name', text: name }), h('span', { class: 'sv-kind__desc', text: desc })),
        h('div', { class: 'sv-kind__runs sv-num', text: formatInt(num(s?.runs)) }),
        h(
          'dl',
          { class: 'sv-kind__dl' },
          dlRow('Success', num(s?.runs) ? formatPct(num(s.success_rate)) : '–'),
          dlRow('Avg time', num(s?.runs) ? formatDuration(num(s.avg_elapsed_s)) : '–'),
          dlRow('Avg cost', num(s?.runs) ? formatUsd(num(s.avg_cost_usd)) : '–'),
        ),
      );
    const presets = d.by_preset ?? {};
    return [
      h('div', { class: 'sv-kinds' }, block('Answers', 'one question', k.answer), block('Lists', 'many records', k.harvest)),
      splitBar('Preset', [
        { label: 'Quick', value: num(presets.quick), color: 'var(--sv-quick)' },
        { label: 'Standard', value: num(presets.standard), color: 'var(--sv-standard)' },
        { label: 'Thorough', value: num(presets.thorough), color: 'var(--sv-thorough)' },
      ]),
      splitBar('Source', [
        { label: 'Web UI', value: num(t.ui), color: 'var(--sv-ui)' },
        { label: 'MCP', value: num(t.mcp), color: 'var(--sv-mcp)' },
      ]),
    ];
  }

  function dlRow(k, v) {
    return h('div', null, h('dt', { text: k }), h('dd', { class: 'sv-num', text: v }));
  }

  function splitBar(title, parts) {
    const total = parts.reduce((s, p) => s + p.value, 0);
    const bar = h('div', { class: 'sv-stackbar sv-stackbar--thin', role: 'img', 'aria-label': `${title}: ${parts.map((p) => `${p.label} ${p.value}`).join(', ')}` });
    for (const p of parts) if (p.value > 0) bar.append(h('span', { style: { width: `${(p.value / total) * 100}%`, background: p.color }, title: `${p.label}: ${formatInt(p.value)}` }));
    return h(
      'div',
      { class: 'sv-split' },
      h('div', { class: 'sv-split__title', text: title }),
      bar,
      h(
        'ul',
        { class: 'sv-split__legend' },
        parts.map((p) =>
          h(
            'li',
            null,
            h('span', { class: 'sv-swatch', style: { background: p.color } }),
            h('span', { text: p.label }),
            h('span', { class: 'sv-num', text: formatInt(p.value) }),
            h('span', { class: 'sv-num sv-muted', text: total ? formatPct(p.value / total) : '' }),
          ),
        ),
      ),
    );
  }

  function durStat(label, v) {
    return h('div', { class: 'sv-durstat' }, h('span', { class: 'sv-durstat__value sv-num', text: v ? formatDuration(num(v)) : '–' }), h('span', { class: 'sv-durstat__label', text: label }));
  }

  function recentTable(d) {
    const recent = Array.isArray(d.recent) ? d.recent : [];
    const now = Date.now();
    const head = h(
      'thead',
      null,
      h(
        'tr',
        null,
        ['When', 'Query', 'Kind', 'Source', 'Outcome', 'Duration', 'Cost', 'Results'].map((c, i) =>
          h('th', { scope: 'col', class: i >= 5 ? 'sv-num' : '', text: c }),
        ),
      ),
    );
    const rows = recent.map((r) => {
      const when = new Date(r.started_at);
      const valid = !isNaN(when.getTime());
      const q = String(r.query ?? '');
      const outcome = OUTCOMES.some((o) => o.key === r.outcome) ? r.outcome : 'empty';
      const oLabel = OUTCOMES.find((o) => o.key === outcome).label;
      const source = r.source === 'mcp' ? 'mcp' : 'ui';
      const results = r.kind === 'harvest' ? `${formatInt(r.records)} rec` : `${formatInt(r.pages)} pages`;
      return h(
        'tr',
        null,
        h('td', { class: 'sv-when' }, h('time', { datetime: valid ? when.toISOString() : '', title: valid ? FULL.format(when) : '', text: valid ? relativeTime(when, now) : '–' })),
        h('td', { class: 'sv-query' }, h('span', { title: q, text: q || '(empty query)' })),
        h('td', { class: 'sv-kindcell', text: KIND_LABEL[r.kind] ?? String(r.kind ?? '–') }),
        h('td', null, h('span', { class: 'sv-src', dataset: { source }, text: source === 'mcp' ? 'MCP' : 'UI' })),
        h('td', null, h('span', { class: 'sv-badge', dataset: { outcome }, text: oLabel })),
        h('td', { class: 'sv-num', text: formatDuration(num(r.elapsed_s)) }),
        h('td', { class: 'sv-num', text: formatUsd(num(r.cost_usd)) }),
        h('td', { class: 'sv-num sv-muted', title: `${formatInt(r.records)} records · ${formatInt(r.pages)} pages`, text: results }),
      );
    });
    const tbody = h('tbody', null, rows);
    const table = rows.length
      ? h('div', { class: 'sv-tablewrap', tabindex: 0, role: 'region', 'aria-label': 'Recent searches' }, h('table', { class: 'sv-table' }, head, tbody))
      : h('p', { class: 'sv-none', text: 'No searches in this range.' });
    let more = null;
    if (rows.length > RECENT_FOLDED) {
      const apply = () => {
        rows.forEach((r, i) => (r.hidden = !state.showAllRecent && i >= RECENT_FOLDED));
        more.textContent = state.showAllRecent ? 'Show fewer' : `Show all ${rows.length}`;
        more.setAttribute('aria-expanded', String(state.showAllRecent));
      };
      more = h('button', {
        class: 'btn btn--text btn--small sv-more',
        type: 'button',
        onclick: () => {
          state.showAllRecent = !state.showAllRecent;
          apply();
        },
      });
      apply();
    }
    const sub = rows.length ? (rows.length >= 50 ? 'Latest 50, newest first' : `${rows.length} in range, newest first`) : '';
    return card('Recent searches', sub, [table, more], 'sv-recent');
  }

  function serverLine(d) {
    const a = d.all_time ?? {};
    const s = d.server ?? {};
    const parts = [];
    const first = a.first_run_at ? new Date(a.first_run_at) : null;
    parts.push(
      `All time: ${formatInt(a.runs)} ${num(a.runs) === 1 ? 'search' : 'searches'} · ${formatUsd(num(a.cost_usd))}` +
        (first && !isNaN(first) ? ` since ${DATE_LONG.format(first)}` : ''),
    );
    if (s.version) parts.push(`webscout ${s.version}`);
    if (s.uptime_secs != null) parts.push(`up ${uptime(s.uptime_secs)}`);
    if (s.mcp_enabled != null) parts.push(s.mcp_enabled ? 'MCP on' : 'MCP off');
    return h('p', { class: 'sv-footer', text: parts.join('  ·  ') });
  }

  /* ------------------------------------------------------------ lifecycle */

  function tick() {
    if (destroyed || state.loading) return;
    if (document.visibilityState !== 'visible') return;
    // Logged out: do not re-raise the login prompt every 30 s; refresh() retries.
    if (state.error?.kind === 'auth') return;
    load();
  }

  function onVisibility() {
    if (document.visibilityState !== 'visible' || destroyed) return;
    if (!state.updatedAt || Date.now() - state.updatedAt.getTime() >= REFRESH_MS) load();
  }

  // The relative times in the table ("3 min ago") go stale between fetches; the
  // 30 s refresh redraws them, which is fine-grained enough.
  timer = window.setInterval(tick, REFRESH_MS);
  document.addEventListener('visibilitychange', onVisibility);
  syncControls();
  load();

  return {
    refresh() {
      load();
    },
    destroy() {
      destroyed = true;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisibility);
      state.controller?.abort();
      clearCharts();
      root.remove();
    },
  };
}

function emptyArt() {
  const ns = 'http://www.w3.org/2000/svg';
  const s = document.createElementNS(ns, 'svg');
  s.setAttribute('viewBox', '0 0 120 72');
  s.setAttribute('width', '120');
  s.setAttribute('height', '72');
  const bars = [18, 30, 22, 44, 34, 56];
  bars.forEach((hgt, i) => {
    const r = document.createElementNS(ns, 'rect');
    r.setAttribute('x', String(8 + i * 18));
    r.setAttribute('y', String(64 - hgt));
    r.setAttribute('width', '12');
    r.setAttribute('height', String(hgt));
    r.setAttribute('rx', '3');
    r.setAttribute('class', i === bars.length - 1 ? 'sv-empty__bar sv-empty__bar--hi' : 'sv-empty__bar');
    s.append(r);
  });
  const base = document.createElementNS(ns, 'line');
  base.setAttribute('x1', '2');
  base.setAttribute('x2', '118');
  base.setAttribute('y1', '66');
  base.setAttribute('y2', '66');
  base.setAttribute('class', 'sv-empty__base');
  s.append(base);
  return s;
}
