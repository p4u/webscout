import './styles.css';
import { ApiError, downloadUrl, fetchOptions, streamSearch } from './api.js';
import { createControls } from './options.js';
import { renderMarkdown, renderPlain, wrapTables } from './markdown.js';

const $ = (id) => document.getElementById(id);

const el = {
  app: $('app'),
  form: $('search-form'),
  query: $('query'),
  clearQuery: $('clear-query'),
  submit: $('submit'),
  stop: $('stop'),
  basic: $('basic-controls'),
  advancedDetails: $('advanced'),
  advanced: $('advanced-controls'),
  changedCount: $('changed-count'),
  reset: $('reset'),
  brand: $('brand'),

  errorPanel: $('error-panel'),
  errorTitle: $('error-title'),
  errorDetail: $('error-detail'),
  retry: $('retry'),

  activity: $('activity'),
  activityStatus: $('activity-status'),
  activityList: $('activity-list'),
  activityToggle: $('activity-toggle'),
  timer: $('timer'),

  usage: $('usage'),
  usageRows: $('usage-rows'),
  usageTotal: $('usage-total'),

  result: $('result'),
  outcome: $('outcome'),
  stats: $('stats'),
  notes: $('notes'),
  prose: $('prose'),
  copy: $('copy'),
  downloadTrigger: $('download-trigger'),
  downloadList: $('download-list'),

  toast: $('toast'),
};

const DOWNLOAD_FORMATS = [
  { format: 'markdown', label: 'Markdown', ext: '.md' },
  { format: 'json', label: 'JSON', ext: '.json' },
  { format: 'csv', label: 'CSV', ext: '.csv' },
  { format: 'jsonl', label: 'JSON Lines', ext: '.jsonl' },
];

const state = {
  controls: null,
  controller: null,
  running: false,
  startedAt: 0,
  timerId: 0,
  toastId: 0,
  lastRunId: null,
  lastContent: '',
  lastFormat: 'markdown',
  lastQuery: '',
  lastStats: null,
  lastUsage: null,
  // Incremented on every run. A late event from a superseded run is ignored rather
  // than allowed to overwrite the current one.
  token: 0,
};

// ---------------------------------------------------------------- bootstrap

init();

async function init() {
  wireStaticHandlers();
  await loadOptions();

  // ?q=... makes a search linkable and re-runnable.
  const initial = new URLSearchParams(location.search).get('q');
  if (initial && initial.trim()) {
    el.query.value = initial;
    el.clearQuery.hidden = false;
    void run(initial);
    return;
  }
  el.query.focus();
}

async function loadOptions() {
  try {
    const schema = await fetchOptions();
    state.controls = createControls(schema, () => updateChangedCount());
    state.controls.renderBasic(el.basic);
    state.controls.renderAdvanced(el.advanced);
    updateChangedCount();
    clearError();
  } catch (err) {
    // A missing control surface must not leave a blank screen: the query field
    // still works, and retry re-fetches the schema.
    showError('Could not load search options', messageOf(err), () => loadOptions());
  }
}

function wireStaticHandlers() {
  // The full placeholder does not fit a phone; swap it rather than clip it.
  const narrow = window.matchMedia('(max-width: 560px)');
  const applyPlaceholder = () => {
    el.query.placeholder = narrow.matches
      ? 'Ask anything'
      : 'Ask anything, or describe a list to harvest';
  };
  applyPlaceholder();
  narrow.addEventListener('change', applyPlaceholder);

  el.form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (state.running) return;
    void run(el.query.value);
  });

  // Enter submits. Handled explicitly as well as via the form's submit event so a
  // stray implicit submission can never navigate away mid-run.
  el.query.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || event.isComposing) return;
    event.preventDefault();
    if (!state.running) void run(el.query.value);
  });

  el.query.addEventListener('input', () => {
    el.clearQuery.hidden = el.query.value.length === 0;
  });

  el.clearQuery.addEventListener('click', () => {
    el.query.value = '';
    el.clearQuery.hidden = true;
    el.query.focus();
  });

  el.stop.addEventListener('click', () => stopRun());

  el.reset.addEventListener('click', () => {
    state.controls?.reset();
    updateChangedCount();
    toast('Settings reset to defaults');
  });

  el.brand.addEventListener('click', (event) => {
    if (state.running) return;
    event.preventDefault();
    resetToIdle();
  });

  el.activityToggle.addEventListener('click', () => {
    setActivityLogOpen(el.activityList.hidden);
  });

  el.copy.addEventListener('click', () => void copyContent());

  el.downloadTrigger.addEventListener('click', (event) => {
    event.stopPropagation();
    toggleDownloadMenu(el.downloadList.hidden);
  });

  document.addEventListener('click', () => toggleDownloadMenu(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') toggleDownloadMenu(false);
  });

  // An unhandled rejection anywhere still ends up as a readable message.
  window.addEventListener('unhandledrejection', (event) => {
    event.preventDefault();
    showError('Unexpected error', messageOf(event.reason));
  });
}

function updateChangedCount() {
  const n = state.controls?.changedCount() ?? 0;
  el.changedCount.hidden = n === 0;
  el.changedCount.textContent = String(n);
}

// ----------------------------------------------------------------- the run

async function run(rawQuery) {
  const query = rawQuery.trim();
  if (!query) {
    el.query.focus();
    showError('Enter a query', 'Type what you want to find, then press Enter.');
    return;
  }

  clearError();
  hideResult();
  state.lastQuery = query;
  try {
    history.replaceState(null, '', `?q=${encodeURIComponent(query)}`);
  } catch {
    /* file:// or a sandboxed frame — the URL is cosmetic */
  }

  const options = state.controls?.changed() ?? {};
  state.lastFormat = String(state.controls?.get('format') ?? 'markdown');

  setPhase('running');
  startTimer();
  el.activity.hidden = false;
  el.activity.dataset.live = 'true';
  el.activityList.replaceChildren();
  setActivityLogOpen(true);
  setStatus('Contacting webscout…');
  // Show the meter at zero from the first instant, so it is a thing that fills
  // rather than a thing that appears.
  renderUsage(null);

  const controller = new AbortController();
  state.controller = controller;
  const token = ++state.token;
  const current = () => state.token === token;
  setRunning(true);

  let sawTerminal = false;

  try {
    for await (const event of streamSearch({ query, options, signal: controller.signal })) {
      if (!current()) return; // stopped or superseded; the reader is cancelled on exit
      switch (event.type) {
        case 'accepted':
          state.lastRunId = event.run_id ?? null;
          logLine('accepted', `Run accepted${event.run_id ? ` (${event.run_id})` : ''}`);
          setStatus('Searching…');
          break;
        case 'progress':
          handleProgress(event);
          break;
        case 'stats':
          state.lastStats = { ...(state.lastStats ?? {}), ...event };
          break;
        case 'usage':
          // Arrives about once a second while the run is in flight, and once
          // more just before the terminal event with the settled numbers.
          renderUsage(event);
          break;
        case 'result':
          sawTerminal = true;
          showResult(event);
          break;
        case 'error':
          sawTerminal = true;
          throw new ApiError(event.message || 'The run failed without a message.');
        default:
          logLine(event.type || 'event', summarise(event));
      }
    }

    if (!sawTerminal) {
      throw new ApiError(
        'The connection closed before the run produced a result. The server may have restarted.',
      );
    }
  } catch (err) {
    if (!current()) return;
    if (err?.name === 'AbortError') {
      setStatus('Stopped');
      logLine('stopped', 'Run stopped by you.');
    } else {
      setStatus('Failed');
      logLine('error', messageOf(err));
      showError(titleFor(err), messageOf(err), () => void run(state.lastQuery));
    }
  } finally {
    if (current()) finishRun();
  }
}

/** Stop is immediate: the UI settles now, the aborted stream unwinds on its own. */
function stopRun() {
  if (!state.running) return;
  state.token += 1; // any event still in flight belongs to a dead run
  state.controller?.abort(new DOMException('stopped by the user', 'AbortError'));
  setStatus('Stopped');
  logLine('stopped', 'Run stopped by you.');
  finishRun();
}

function finishRun() {
  stopTimer();
  setRunning(false);
  state.controller = null;
  el.activity.dataset.live = 'false';
  setPhase('done');
}

function handleProgress(event) {
  const stage = String(event.stage ?? 'run');
  const round = Number.isFinite(event.round) ? `round ${event.round}` : '';
  const message = String(event.message ?? stage);
  logLine(stage, message, event.counts, round);
  setStatus(round ? `${message} · ${round}` : message);
  if (event.counts && typeof event.counts === 'object') {
    state.lastStats = {
      ...(state.lastStats ?? {}),
      pages_fetched: event.counts.pages ?? state.lastStats?.pages_fetched,
      records: event.counts.records ?? state.lastStats?.records,
    };
  }
}

function logLine(stage, message, counts, round) {
  const li = document.createElement('li');
  li.className = 'activity__item';

  const time = document.createElement('span');
  time.className = 'activity__time';
  time.textContent = elapsedLabel();

  const stageEl = document.createElement('span');
  stageEl.className = 'activity__stage';
  stageEl.textContent = stage;

  const msg = document.createElement('span');
  msg.className = 'activity__msg';
  msg.textContent = round && !String(message).includes(round) ? `${message} (${round})` : message;

  li.append(time, stageEl, msg);

  if (counts && typeof counts === 'object') {
    const parts = Object.entries(counts)
      .filter(([, v]) => typeof v === 'number')
      .map(([k, v]) => `${k} ${v}`);
    if (parts.length) {
      const c = document.createElement('span');
      c.className = 'activity__counts';
      c.textContent = parts.join(' · ');
      li.appendChild(c);
    }
  }

  const atBottom =
    el.activityList.scrollTop + el.activityList.clientHeight >= el.activityList.scrollHeight - 24;
  el.activityList.appendChild(li);
  if (atBottom) el.activityList.scrollTop = el.activityList.scrollHeight;
}

function setActivityLogOpen(open) {
  el.activityList.hidden = !open;
  el.activityToggle.textContent = open ? 'Hide log' : 'Show log';
  el.activityToggle.setAttribute('aria-expanded', String(open));
  if (open) el.activityList.scrollTop = el.activityList.scrollHeight;
}

function setStatus(text) {
  el.activityStatus.textContent = text;
}

function summarise(event) {
  if (typeof event.message === 'string') return event.message;
  try {
    return JSON.stringify(event).slice(0, 200);
  } catch {
    return '(unreadable event)';
  }
}

// -------------------------------------------------------------------- timer

function startTimer() {
  state.startedAt = performance.now();
  el.timer.textContent = '0.0s';
  stopTimer();
  state.timerId = window.setInterval(() => {
    el.timer.textContent = elapsedLabel();
  }, 100);
}

function stopTimer() {
  if (state.timerId) {
    window.clearInterval(state.timerId);
    state.timerId = 0;
  }
}

function elapsedLabel() {
  const ms = performance.now() - state.startedAt;
  return formatDuration(ms);
}

function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${String(Math.floor(s % 60)).padStart(2, '0')}s`;
}

// ------------------------------------------------------------------- result

function showResult(event) {
  const outcome = String(event.outcome ?? 'complete').toLowerCase();
  const format = String(event.format ?? state.lastFormat ?? 'markdown').toLowerCase();
  const content = typeof event.content === 'string' ? event.content : '';

  state.lastContent = content;
  state.lastFormat = format;
  state.lastStats = { ...(state.lastStats ?? {}), ...(event.stats ?? {}) };

  el.outcome.dataset.outcome = outcome;
  el.outcome.textContent = outcome;
  el.outcome.title = OUTCOME_HELP[outcome] ?? '';

  renderStats(state.lastStats);
  // Settle the meter on the run's own accounting, so the panel and the result
  // can never disagree about what was spent.
  renderUsage(usageFromStats(event.stats));
  renderNotes(collectNotes(event));

  el.prose.innerHTML =
    format === 'markdown' || format === 'terminal'
      ? renderMarkdown(content)
      : renderPlain(content, format === 'jsonl' ? 'json' : format);
  wrapTables(el.prose);

  renderDownloads();
  el.copy.textContent = format === 'markdown' ? 'Copy markdown' : `Copy ${format}`;
  el.copy.disabled = content.length === 0;

  el.result.hidden = false;
  setStatus(`Finished · ${outcome}`);
  // Give the answer the page; the log stays one click away.
  setActivityLogOpen(false);
}

const OUTCOME_HELP = {
  complete: 'The run satisfied the request.',
  partial: 'The web ran out before the target did, or some claims were unsupported.',
  truncated: 'The round ceiling hit while results were still arriving. Set a higher max rounds for more.',
  empty: 'Nothing verifiable was found. This is a result, not a failure.',
};

function renderStats(stats) {
  el.stats.replaceChildren();
  if (!stats) return;

  const elapsedMs = pick(stats, ['elapsed_ms', 'total_ms', 'duration_ms']);
  const rows = [
    ['Elapsed', elapsedMs != null ? formatDuration(Number(elapsedMs)) : null],
    ['Pages', pick(stats, ['pages_fetched', 'pages_read', 'pages'])],
    ['Records', pick(stats, ['records', 'records_found', 'record_count'])],
    ['Jev', pick(stats, ['jev_requests', 'typesafe_requests'])],
    ['LLM', pick(stats, ['llm_requests'])],
    ['Rounds', pick(stats, ['rounds'])],
  ];

  for (const [label, value] of rows) {
    if (value === null || value === undefined) continue;
    const stat = document.createElement('span');
    stat.className = 'stat';
    const v = document.createElement('span');
    v.className = 'stat__value';
    v.textContent = typeof value === 'number' ? formatNumber(value) : String(value);
    const l = document.createElement('span');
    l.textContent = label.toLowerCase();
    stat.append(v, l);
    el.stats.appendChild(stat);
  }
}

function pick(object, keys) {
  for (const key of keys) {
    const value = object?.[key];
    if (value !== null && value !== undefined) return value;
  }
  return null;
}

function formatNumber(n) {
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(2);
}

// ------------------------------------------------------------- tokens & cost

/** Thousands separators, and never `NaN` on the screen. */
function count(n) {
  const v = Number(n);
  return Number.isFinite(v) ? Math.round(v).toLocaleString() : '0';
}

/** Four decimals, because a whole run often costs less than a cent. */
function usd(n) {
  const v = Number(n);
  return Number.isFinite(v) ? `$${v.toFixed(4)}` : null;
}

/**
 * The three rows of the meter, in the order they spend.
 *
 * A cost of `undefined` means the endpoint never reported one — only OpenRouter
 * does — and that row simply shows no money rather than a zero that would read
 * as "this was free". Jev's cost is always known: it is tokens times a measured
 * constant.
 */
function usageRows(event) {
  const jev = event?.jev ?? {};
  const llm = event?.llm ?? {};
  const planner = event?.planner ?? {};
  return [
    {
      name: 'Jev',
      requests: jev.requests ?? 0,
      tokens: `${count(jev.input_tokens)} in`,
      reasoning: 0,
      cost: jev.cost_usd,
    },
    {
      name: 'Writer',
      requests: llm.requests ?? 0,
      tokens: `${count(llm.prompt_tokens)} in · ${count(llm.completion_tokens)} out`,
      reasoning: llm.reasoning_tokens ?? 0,
      cost: llm.cost_usd,
    },
    {
      name: 'Planner',
      requests: planner.requests ?? 0,
      tokens: `${count(planner.prompt_tokens)} in · ${count(planner.completion_tokens)} out`,
      reasoning: planner.reasoning_tokens ?? 0,
      cost: planner.cost_usd,
      // The planner shares the writer's model unless one was named, in which
      // case it never reports separately and an empty row is just noise.
      hideWhenIdle: true,
    },
  ];
}

/**
 * What the run has cost so far, or `null` if a component that actually ran
 * never reported a price. A partial total would be read as a total.
 */
function usageTotal(rows) {
  let sum = 0;
  for (const row of rows) {
    if (row.requests === 0) continue;
    if (typeof row.cost !== 'number' || !Number.isFinite(row.cost)) return null;
    sum += row.cost;
  }
  return sum;
}

function renderUsage(event) {
  state.lastUsage = event;
  el.usage.hidden = false;
  el.usageRows.replaceChildren();

  const rows = usageRows(event);
  for (const row of rows) {
    if (row.hideWhenIdle && row.requests === 0) continue;

    const line = document.createElement('div');
    line.className = 'usage__row';

    const name = document.createElement('span');
    name.className = 'usage__name';
    name.textContent = row.name;

    const metrics = document.createElement('span');
    metrics.className = 'usage__metrics';
    let text = `${count(row.requests)} req · ${row.tokens} tokens`;
    if (row.reasoning > 0) text += ` · ${count(row.reasoning)} thinking`;
    metrics.textContent = text;

    const cost = document.createElement('span');
    cost.className = 'usage__cost';
    cost.textContent = usd(row.cost) ?? '—';
    if (usd(row.cost) === null) cost.title = 'This endpoint does not report a cost.';

    line.append(name, metrics, cost);
    el.usageRows.appendChild(line);
  }

  const total = usageTotal(rows);
  el.usageTotal.textContent = total === null ? '' : usd(total);
}

/**
 * The same shape as a `usage` event, read out of a finished run's stats.
 *
 * Used to settle the meter at the end: the server already sends a final `usage`
 * built from these very numbers, and this makes the panel right even if that
 * line was missed.
 */
function usageFromStats(stats) {
  if (!stats) return state.lastUsage;
  return {
    jev: {
      requests: stats.jev_requests ?? 0,
      input_tokens: stats.jev_input_tokens ?? 0,
      cost_usd: stats.jev_cost_usd,
    },
    llm: {
      requests: stats.llm_requests ?? 0,
      prompt_tokens: stats.llm_prompt_tokens ?? 0,
      completion_tokens: stats.llm_completion_tokens ?? 0,
      reasoning_tokens: stats.llm_reasoning_tokens ?? 0,
      cost_usd: stats.llm_cost_usd ?? undefined,
    },
    planner: {
      requests: stats.planner_requests ?? 0,
      prompt_tokens: stats.planner_prompt_tokens ?? 0,
      completion_tokens: stats.planner_completion_tokens ?? 0,
      reasoning_tokens: stats.planner_reasoning_tokens ?? 0,
      cost_usd: stats.planner_cost_usd ?? undefined,
    },
  };
}

/** Notes may sit on the event, inside stats, or inside mission; accept all three. */
function collectNotes(event) {
  const buckets = [event?.notes, event?.stats?.notes, event?.mission?.notes];
  const out = [];
  for (const bucket of buckets) {
    if (!Array.isArray(bucket)) continue;
    for (const note of bucket) {
      const text = typeof note === 'string' ? note : note?.message ?? note?.text;
      if (typeof text === 'string' && text.trim() && !out.includes(text)) out.push(text.trim());
    }
  }
  return out;
}

function renderNotes(notes) {
  el.notes.replaceChildren();
  el.notes.hidden = notes.length === 0;
  for (const note of notes) {
    const li = document.createElement('li');
    li.textContent = note;
    el.notes.appendChild(li);
  }
}

function renderDownloads() {
  el.downloadList.replaceChildren();
  const runId = state.lastRunId;
  el.downloadTrigger.disabled = !runId;
  if (!runId) return;

  for (const { format, label, ext } of DOWNLOAD_FORMATS) {
    const a = document.createElement('a');
    a.className = 'menu__item';
    a.setAttribute('role', 'menuitem');
    a.href = downloadUrl(runId, format);
    a.download = `webscout-${runId}${ext}`;
    a.textContent = label;
    const extEl = document.createElement('span');
    extEl.className = 'menu__ext';
    extEl.textContent = ext;
    a.appendChild(extEl);
    a.addEventListener('click', () => toggleDownloadMenu(false));
    el.downloadList.appendChild(a);
  }
}

function toggleDownloadMenu(open) {
  el.downloadList.hidden = !open;
  el.downloadTrigger.setAttribute('aria-expanded', String(open));
}

async function copyContent() {
  if (!state.lastContent) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(state.lastContent);
    } else {
      legacyCopy(state.lastContent);
    }
    toast('Copied to clipboard');
  } catch {
    // Clipboard permission can be denied (or absent over plain http on a LAN).
    try {
      legacyCopy(state.lastContent);
      toast('Copied to clipboard');
    } catch {
      toast('Could not copy — select the text and copy manually');
    }
  }
}

function legacyCopy(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.select();
  const ok = document.execCommand('copy');
  area.remove();
  if (!ok) throw new Error('copy rejected');
}

function hideResult() {
  el.result.hidden = true;
  el.prose.replaceChildren();
  el.notes.hidden = true;
  state.lastRunId = null;
  state.lastContent = '';
  state.lastStats = null;
}

// -------------------------------------------------------------------- state

function setPhase(phase) {
  el.app.dataset.phase = phase;
}

function resetToIdle() {
  hideResult();
  clearError();
  el.activity.hidden = true;
  el.activityList.replaceChildren();
  el.usage.hidden = true;
  el.usageRows.replaceChildren();
  el.usageTotal.textContent = '';
  state.lastUsage = null;
  el.query.value = '';
  el.clearQuery.hidden = true;
  setPhase('idle');
  try {
    history.replaceState(null, '', location.pathname);
  } catch {
    /* cosmetic */
  }
  el.query.focus();
}

function setRunning(running) {
  state.running = running;
  el.submit.disabled = running;
  el.submit.hidden = running;
  el.stop.hidden = !running;
  el.query.readOnly = running;
  el.reset.disabled = running;
  for (const control of el.form.querySelectorAll('.chip__control, .field__input, .switch input')) {
    control.disabled = running;
  }
}

// ------------------------------------------------------------------- errors

function showError(title, detail, onRetry) {
  el.errorTitle.textContent = title;
  el.errorDetail.textContent = detail || '';
  el.errorPanel.hidden = false;
  el.retry.hidden = typeof onRetry !== 'function';
  el.retry.onclick = () => {
    clearError();
    onRetry?.();
  };
  if (el.app.dataset.phase === 'idle' && onRetry) setPhase('done');
}

function clearError() {
  el.errorPanel.hidden = true;
  el.retry.onclick = null;
}

function titleFor(err) {
  if (err instanceof ApiError && err.status === 400) return 'The API rejected the request';
  if (err instanceof ApiError && err.status >= 500) return 'The API failed';
  return 'Search failed';
}

function messageOf(err) {
  if (!err) return 'Unknown error.';
  if (typeof err === 'string') return err;
  if (err instanceof Error) return err.message || String(err);
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

// -------------------------------------------------------------------- toast

function toast(text) {
  el.toast.textContent = text;
  el.toast.hidden = false;
  window.clearTimeout(state.toastId);
  state.toastId = window.setTimeout(() => {
    el.toast.hidden = true;
  }, 2400);
}
