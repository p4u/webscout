import './styles.css';
import {
  ApiError,
  downloadUrl,
  fetchMcpInfo,
  fetchOptions,
  fetchSession,
  login as apiLogin,
  logout as apiLogout,
  streamSearch,
} from './api.js';
import { createControls } from './options.js';
import { renderMarkdown, renderPlain, wrapTables } from './markdown.js';
import { EXAMPLE_GROUPS, TIPS } from './examples.js';
import { CLIENTS, TOOL_NOTES, maskedToken, mcpUrl, tokenOrPlaceholder } from './connect.js';
import { mountStats } from './stats.js';

const $ = (id) => document.getElementById(id);

const el = {
  app: $('app'),
  form: $('search-form'),
  query: $('query'),
  clearQuery: $('clear-query'),
  submit: $('submit'),
  searchBtn: $('search-btn'),
  lucky: $('lucky'),
  stop: $('stop'),
  toolsToggle: $('tools-toggle'),
  toolsDot: $('tools-dot'),
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
  usageToggle: $('usage-toggle'),
  timer: $('timer'),

  usage: $('usage'),
  usageRows: $('usage-rows'),
  usageTotal: $('usage-total'),

  result: $('result'),
  outcome: $('outcome'),
  summary: $('summary'),
  stats: $('stats'),
  notes: $('notes'),
  prose: $('prose'),
  took: $('took'),
  sourcesToggle: $('sources-toggle'),
  sources: $('sources'),
  sourcesList: $('sources-list'),
  quarantine: $('quarantine'),
  quarantineList: $('quarantine-list'),

  help: $('help'),
  helpOpen: $('help-open'),
  helpClose: $('help-close'),
  helpGroups: $('help-groups'),
  helpTips: $('help-tips'),

  connect: $('connect'),
  connectOpen: $('connect-open'),
  connectClose: $('connect-close'),
  connectStatus: $('connect-status'),
  connectUrl: $('connect-url'),
  connectToken: $('connect-token'),
  connectTokenPaste: $('connect-token-paste'),
  connectTokenServer: $('connect-token-server'),
  connectTokenValue: $('connect-token-value'),
  connectTokenReveal: $('connect-token-reveal'),
  connectTokenCopy: $('connect-token-copy'),
  connectTabs: $('connect-tabs'),
  connectPanel: $('connect-panel'),
  connectTools: $('connect-tools'),
  copy: $('copy'),
  downloadTrigger: $('download-trigger'),
  downloadList: $('download-list'),

  toast: $('toast'),

  tabs: $('tabs'),
  tabButtons: [$('tab-search'), $('tab-stats')],
  viewSearch: $('view-search'),
  viewStats: $('view-stats'),
  logout: $('logout'),

  login: $('login'),
  loginForm: $('login-form'),
  loginPassword: $('login-password'),
  loginEye: $('login-eye'),
  loginError: $('login-error'),
  loginSubmit: $('login-submit'),
  loginSubmitText: $('login-submit-text'),
};

const SEARCH_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M15.5 15.5 L21 21" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

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
  // Tokens and cost are for the curious; remembered across visits once shown.
  showUsage: readPref('webscout.showUsage'),
  // Which client's setup the "Connect AI tools" dialog shows.
  connectClient: 'claude',
  // The server's MCP token, when a password-protected server hands it to a
  // logged-in person. Memory only: never stored, never logged, dropped on logout.
  mcpToken: null,
  tokenRevealed: false,

  // Which tab is showing: 'search' or 'stats' (mirrored in the URL hash).
  view: 'search',
  // The statistics view, mounted the first time its tab is opened.
  statsView: null,
  // A ?q= search that arrived while the Statistics tab was showing; it runs
  // when Search is first shown rather than spending money out of sight.
  pendingQuery: '',

  // Login. `loginPromise` is pending while the card is up; everything that hit a
  // 401 awaits it and then carries on where it was.
  authRequired: false,
  loginPromise: null,
  loginResolve: null,
  loggingIn: false,
  // Incremented on every run. A late event from a superseded run is ignored rather
  // than allowed to overwrite the current one.
  token: 0,
};

// ---------------------------------------------------------------- bootstrap

init();

async function init() {
  wireStaticHandlers();
  wireLogin();
  wireTabs();
  renderHelp();
  renderConnect();
  applyUsageVisibility();

  // A private server shows the login card before anything else. An older
  // server without /api/session reads as open (see fetchSession).
  const session = await fetchSession();
  state.authRequired = session.auth_required;
  el.logout.hidden = !state.authRequired;
  document.body.dataset.boot = 'ready';
  if (session.auth_required && !session.authenticated) await requireLogin();

  applyView(viewFromHash());
  await loadOptions();

  // ?q=... makes a search linkable and re-runnable.
  const initial = new URLSearchParams(location.search).get('q');
  if (initial && initial.trim()) {
    el.query.value = initial;
    el.clearQuery.hidden = false;
    if (state.view === 'search') void run(initial);
    else state.pendingQuery = initial;
    return;
  }
  if (state.view === 'search') el.query.focus();
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
    if (isLoginRequired(err)) {
      await requireLogin();
      return loadOptions();
    }
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

  // "Surprise me": Google's "I'm Feeling Lucky" — a random known-good example,
  // run straight away.
  el.lucky.addEventListener('click', () => {
    if (state.running) return;
    const all = EXAMPLE_GROUPS.flatMap((g) => g.examples);
    const pick = all[Math.floor(Math.random() * all.length)];
    el.query.value = pick;
    el.clearQuery.hidden = false;
    void run(pick);
  });

  // "Tools" in the results header shows or hides the search options row.
  el.toolsToggle.addEventListener('click', () => {
    setToolsOpen(el.app.dataset.tools !== 'open');
  });

  // Footer and top-bar shortcuts: open a dialog, or switch to a view.
  document.addEventListener('click', (event) => {
    const opener = event.target.closest?.('[data-open]');
    if (opener) {
      if (opener.dataset.open === 'help') openHelp();
      else if (opener.dataset.open === 'connect') void openConnect();
      return;
    }
    const viewLink = event.target.closest?.('[data-view-link]');
    if (viewLink) {
      event.preventDefault();
      selectView(viewLink.dataset.viewLink);
    }
  });

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
    event.preventDefault();
    if (state.view !== 'search') selectView('search');
    if (!state.running) resetToIdle();
  });

  el.activityToggle.addEventListener('click', () => {
    setActivityLogOpen(el.activityList.hidden);
  });

  el.usageToggle.addEventListener('click', () => {
    state.showUsage = !state.showUsage;
    writePref('webscout.showUsage', state.showUsage);
    applyUsageVisibility();
  });

  el.sourcesToggle.addEventListener('click', () => {
    setSourcesOpen(el.sources.hidden);
  });

  // A citation in the answer opens the sources box at the page it names.
  el.prose.addEventListener('click', (event) => {
    const cite = event.target.closest?.('a.cite');
    if (!cite) return;
    event.preventDefault();
    setSourcesOpen(true);
    const item = document.getElementById(`source-${cite.dataset.source}`);
    if (!item) return;
    item.scrollIntoView({ behavior: 'smooth', block: 'center' });
    item.classList.remove('is-flash');
    void item.offsetWidth; // restart the highlight animation
    item.classList.add('is-flash');
  });

  el.helpOpen.addEventListener('click', () => openHelp());
  el.connectOpen.addEventListener('click', () => void openConnect());
  el.connectClose.addEventListener('click', () => el.connect.close());
  el.connect.addEventListener('click', (event) => {
    if (event.target === el.connect) el.connect.close();
    const copy = event.target.closest?.('[data-copy-target]');
    if (copy) void copyText(document.getElementById(copy.dataset.copyTarget)?.value ?? '');
    // Built afresh rather than read off the screen: a masked server token must
    // still reach the clipboard as the real value.
    const copyStep = event.target.closest?.('[data-copy-step]');
    if (copyStep) void copyText(snippetText(Number(copyStep.dataset.copyStep), realToken()));
  });
  el.connect.addEventListener('close', () => {
    state.tokenRevealed = false;
    applyTokenUi();
    renderConnectPanel();
  });
  el.connectToken.addEventListener('input', () => renderConnectPanel());
  el.connectTokenReveal.addEventListener('click', () => {
    state.tokenRevealed = !state.tokenRevealed;
    applyTokenUi();
    renderConnectPanel();
  });
  el.connectTokenCopy.addEventListener('click', () => {
    if (state.mcpToken) void copyText(state.mcpToken);
  });
  el.helpClose.addEventListener('click', () => el.help.close());
  // A click on the backdrop (outside the dialog box) closes it.
  el.help.addEventListener('click', (event) => {
    if (event.target === el.help) el.help.close();
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
  // The results header keeps the options behind "Tools"; a dot there says
  // something differs from the defaults.
  el.toolsDot.hidden = n === 0;
}

function setToolsOpen(open) {
  el.app.dataset.tools = open ? 'open' : 'closed';
  el.toolsToggle.setAttribute('aria-expanded', String(open));
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
    history.replaceState(null, '', `?q=${encodeURIComponent(query)}${location.hash}`);
  } catch {
    /* file:// or a sandboxed frame — the URL is cosmetic */
  }

  const options = state.controls?.changed() ?? {};
  state.lastFormat = String(state.controls?.get('format') ?? 'markdown');

  setPhase('running');
  setToolsOpen(false);
  startTimer();
  el.activity.hidden = false;
  el.activity.dataset.live = 'true';
  el.activityList.replaceChildren();
  setActivityLogOpen(true);
  setStatus('Contacting webscout…');
  // The meter starts at zero from the first instant, so when someone opens it
  // mid-run it is a thing that fills rather than a thing that appears.
  renderUsage(null);
  el.stats.replaceChildren();

  const controller = new AbortController();
  state.controller = controller;
  const token = ++state.token;
  const current = () => state.token === token;
  setRunning(true);

  let sawTerminal = false;
  let loginNeeded = false;

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
    } else if (isLoginRequired(err)) {
      // The session ran out. Nothing ran server-side; log in and go again.
      loginNeeded = true;
      setStatus('Log in to continue');
      logLine('login', 'The session has expired. Log in and the search starts again.');
    } else {
      setStatus('Failed');
      logLine('error', messageOf(err));
      showError(titleFor(err), messageOf(err), () => void run(state.lastQuery));
    }
  } finally {
    if (current()) finishRun();
  }

  if (loginNeeded) {
    await requireLogin();
    if (current() && !state.running) void run(query);
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
  const sources = Array.isArray(event.sources) ? event.sources : [];

  state.lastContent = content;
  state.lastFormat = format;
  state.lastStats = { ...(state.lastStats ?? {}), ...(event.stats ?? {}) };

  el.result.dataset.outcome = outcome;
  el.outcome.dataset.outcome = outcome;
  el.outcome.textContent = OUTCOME_LABEL[outcome] ?? outcome;
  el.outcome.title = OUTCOME_HELP[outcome] ?? '';
  el.summary.textContent = sentence(event.summary);

  renderStats(state.lastStats);
  // Settle the meter on the run's own accounting, so the panel and the result
  // can never disagree about what was spent.
  renderUsage(usageFromStats(event.stats));
  renderNotes(collectNotes(event));

  // Formatted text shows the reply alone — sources, notes and run figures have
  // their own places. `body` is absent from an older API; fall back to the
  // whole document then.
  if (format === 'markdown' || format === 'terminal') {
    const body = typeof event.body === 'string' ? event.body : content;
    el.prose.innerHTML = body.trim()
      ? renderMarkdown(body)
      : `<p class="reply__empty">${escapeText(emptyText(event))}</p>`;
    linkCitations(el.prose, sources);
  } else {
    el.prose.innerHTML = renderPlain(content, format === 'jsonl' ? 'json' : format);
  }
  wrapTables(el.prose);

  renderSources(sources, Array.isArray(event.quarantined) ? event.quarantined : []);
  renderTook(state.lastStats);

  renderDownloads();
  el.copy.disabled = content.length === 0;

  el.result.hidden = false;
  setStatus(`Finished · ${(OUTCOME_LABEL[outcome] ?? outcome).toLowerCase()}`);
  // Give the answer the page; the log stays one click away.
  setActivityLogOpen(false);
}

const OUTCOME_LABEL = {
  complete: 'Complete',
  partial: 'Partial',
  truncated: 'Stopped early',
  empty: 'Nothing found',
};

const OUTCOME_HELP = {
  complete: 'The run satisfied the request.',
  partial: 'Part of the request was answered; the rest could not be found or verified.',
  truncated: 'The round limit was reached while results were still arriving. Allow more search rounds for more.',
  empty: 'Nothing verifiable was found. This is a result, not a failure.',
};

function emptyText(event) {
  const kind = String(event.kind ?? event.mission?.kind ?? '');
  return kind === 'harvest'
    ? 'No items could be verified. Try wording the list differently, or a broader description.'
    : 'No answer could be verified from the pages found. Try rephrasing, or naming the organisation or product more precisely.';
}

/** "the evidence answers the question" → "The evidence answers the question." */
function sentence(text) {
  if (typeof text !== 'string' || !text.trim()) return '';
  const t = text.trim();
  return t[0].toUpperCase() + t.slice(1) + (/[.!?]$/.test(t) ? '' : '.');
}

function escapeText(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

/**
 * Turn the answer's `[n]` citations into links to source n. Walks text nodes
 * only, so nothing the sanitiser removed can come back, and leaves code and
 * existing links alone.
 */
function linkCitations(root, sources) {
  if (!sources.length) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement?.closest('a, code, pre') || !/\[\d{1,3}\]/.test(node.nodeValue)
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  for (const node of nodes) {
    const frag = document.createDocumentFragment();
    let last = 0;
    const text = node.nodeValue;
    for (const m of text.matchAll(/\[(\d{1,3})\]/g)) {
      const n = Number(m[1]);
      const source = sources[n - 1];
      if (!source) continue;
      // "[1], [2]" reads as a row of badges; the comma between two of them
      // is noise once they are drawn as badges.
      const between = text.slice(last, m.index);
      if (!(last > 0 && /^\s*,?\s*$/.test(between))) frag.append(between);
      const a = document.createElement('a');
      a.className = 'cite';
      a.href = `#source-${n}`;
      a.dataset.source = String(n);
      a.textContent = String(n);
      a.title = source.title || hostOf(source.url);
      frag.append(a);
      last = m.index + m[0].length;
    }
    if (last === 0) continue;
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return String(url ?? '');
  }
}

/** Only web links become clickable; anything else is shown as text. */
function safeHref(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'http:' || u.protocol === 'https:' ? u.href : null;
  } catch {
    return null;
  }
}

function supportLabel(p) {
  const v = Number(p);
  if (!Number.isFinite(v)) return null;
  if (v >= 0.8) return ['strong', 'Strong support'];
  if (v >= 0.5) return ['medium', 'Supports'];
  return ['weak', 'Weak support'];
}

/** "https://www.vodafone.com/about/who" → ["vodafone.com", "https://www.vodafone.com › about › who"] */
function crumbOf(url) {
  try {
    const u = new URL(url);
    const parts = u.pathname.split('/').filter(Boolean).map((p) => {
      try {
        return decodeURIComponent(p);
      } catch {
        return p;
      }
    });
    return [`${u.protocol}//${u.host}`, ...parts].join(' › ');
  } catch {
    return String(url ?? '');
  }
}

/** A site's first letter, for the favicon-like circle. */
function initialOf(host) {
  const name = host.replace(/^(www|m|en|es)\./, '');
  return (name.match(/[a-z0-9]/i)?.[0] ?? '?').toUpperCase();
}

/** A stable colour per site, so the circles tell sites apart. */
const FAVICON_COLOURS = ['#4285f4', '#ea4335', '#f9ab00', '#34a853', '#a142f4', '#24c1e0', '#e8710a', '#1a73e8'];
function colourOf(host) {
  let h = 0;
  for (const ch of host) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return FAVICON_COLOURS[h % FAVICON_COLOURS.length];
}

function renderSources(sources, quarantined) {
  el.sourcesList.replaceChildren();
  el.quarantineList.replaceChildren();

  sources.forEach((source, i) => {
    const li = document.createElement('li');
    li.className = 'source';
    li.id = `source-${i + 1}`;
    const host = hostOf(source.url);
    const href = safeHref(source.url);

    // Google's result header: favicon, site name, breadcrumb URL.
    const site = document.createElement('div');
    site.className = 'source__site';
    const fav = document.createElement('span');
    fav.className = 'source__fav';
    fav.textContent = initialOf(host);
    fav.style.background = colourOf(host);
    fav.setAttribute('aria-hidden', 'true');
    const names = document.createElement('div');
    names.className = 'source__names';
    const siteName = document.createElement('span');
    siteName.className = 'source__host';
    siteName.textContent = host;
    const crumb = document.createElement('cite');
    crumb.className = 'source__crumb';
    crumb.textContent = crumbOf(source.url);
    names.append(siteName, crumb);
    // The citation number rides on the circle, so [n] in the answer and this
    // result are visibly the same thing.
    const num = document.createElement('span');
    num.className = 'source__num';
    num.textContent = String(i + 1);
    num.title = `Citation ${i + 1}`;
    const mark = document.createElement('span');
    mark.className = 'source__mark';
    mark.append(fav, num);
    site.append(mark, names);

    const title = document.createElement(href ? 'a' : 'span');
    title.className = 'source__title';
    title.textContent = (source.title || '').trim() || host;
    if (href) {
      title.href = href;
      title.target = '_blank';
      title.rel = 'noopener noreferrer';
    }
    const heading = document.createElement('h3');
    heading.className = 'source__heading';
    heading.appendChild(title);

    li.append(site, heading);
    const support = supportLabel(source.supports);
    if (support) {
      const meta = document.createElement('p');
      meta.className = `source__support source__support--${support[0]}`;
      meta.textContent = `${support[1]} · ${Math.round(Number(source.supports) * 100)}%`;
      meta.title = 'How strongly this page supports the answer';
      li.appendChild(meta);
    }
    el.sourcesList.appendChild(li);
  });

  for (const url of quarantined) {
    const li = document.createElement('li');
    li.textContent = String(url);
    el.quarantineList.appendChild(li);
  }
  el.quarantine.hidden = quarantined.length === 0;
  el.sourcesList.hidden = sources.length === 0;

  const count = sources.length + quarantined.length;
  el.sourcesToggle.hidden = count === 0;
  el.sourcesToggle.dataset.count = String(sources.length || quarantined.length);
  setSourcesOpen(false);
}

function setSourcesOpen(open) {
  const hasAny = !el.sourcesToggle.hidden;
  el.sources.hidden = !(open && hasAny);
  const n = el.sourcesToggle.dataset.count ?? '';
  el.sourcesToggle.textContent = `${open ? 'Hide' : 'Show'} sources${n ? ` (${n})` : ''}`;
  el.sourcesToggle.setAttribute('aria-expanded', String(open && hasAny));
}

/** One quiet line under the reply: how long it took and how much was read. */
function renderTook(stats) {
  if (!stats) {
    el.took.textContent = '';
    return;
  }
  const parts = [];
  const ms = pick(stats, ['elapsed_ms', 'total_ms', 'duration_ms']);
  const secs = pick(stats, ['elapsed_secs']);
  if (ms != null) parts.push(`took ${formatDuration(Number(ms))}`);
  else if (secs != null) parts.push(`took ${formatDuration(Number(secs) * 1000)}`);
  const pages = pick(stats, ['pages_fetched', 'pages_read', 'pages']);
  if (pages != null) parts.push(`${formatNumber(Number(pages))} pages read`);
  el.took.textContent = parts.length ? sentence(parts.join(' · ')) : '';
}

function renderStats(stats) {
  el.stats.replaceChildren();
  if (!stats) return;

  const secs = pick(stats, ['elapsed_secs']);
  const elapsedMs =
    pick(stats, ['elapsed_ms', 'total_ms', 'duration_ms']) ?? (secs != null ? Number(secs) * 1000 : null);
  const rows = [
    ['Elapsed', elapsedMs != null ? formatDuration(Number(elapsedMs)) : null],
    ['Pages', pick(stats, ['pages_fetched', 'pages_read', 'pages'])],
    // An answer has no records; "0 records" would read as a failure.
    ['Records', pick(stats, ['records', 'records_found', 'record_count']) || null],
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
  await copyText(state.lastContent);
}

async function copyText(text) {
  if (!text) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      legacyCopy(text);
    }
    toast('Copied to clipboard');
  } catch {
    // Clipboard permission can be denied (or absent over plain http on a LAN).
    try {
      legacyCopy(text);
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
  el.sources.hidden = true;
  el.sourcesList.replaceChildren();
  el.summary.textContent = '';
  el.took.textContent = '';
  state.lastRunId = null;
  state.lastContent = '';
  state.lastStats = null;
}

// -------------------------------------------------------------------- state

function setPhase(phase) {
  el.app.dataset.phase = phase;
  updateLayout();
}

/**
 * `hero` centres the search under the product name; `compact` is the sticky
 * header. The Statistics tab always uses the compact header: a centred empty
 * masthead above a dashboard would push the numbers below the fold.
 */
function updateLayout() {
  const hero = el.app.dataset.phase === 'idle' && state.view === 'search';
  el.app.dataset.layout = hero ? 'hero' : 'compact';
}

function resetToIdle() {
  hideResult();
  clearError();
  el.activity.hidden = true;
  el.activityList.replaceChildren();
  el.usageRows.replaceChildren();
  el.stats.replaceChildren();
  el.usageTotal.textContent = '';
  state.lastUsage = null;
  el.query.value = '';
  el.clearQuery.hidden = true;
  setPhase('idle');
  try {
    history.replaceState(null, '', location.pathname + location.hash);
  } catch {
    /* cosmetic */
  }
  el.query.focus();
}

function setRunning(running) {
  state.running = running;
  el.submit.disabled = running;
  el.submit.hidden = running;
  el.searchBtn.disabled = running;
  el.lucky.disabled = running;
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

// ------------------------------------------------------------ usage toggle

/** The meter exists from the start of a run; it is only shown on request. */
function applyUsageVisibility() {
  const show = state.showUsage;
  el.usage.hidden = !show;
  el.usageToggle.textContent = show ? 'Hide tokens & cost' : 'Show tokens & cost';
  el.usageToggle.setAttribute('aria-expanded', String(show));
}

function readPref(key) {
  try {
    return localStorage.getItem(key) === '1';
  } catch {
    return false;
  }
}

function writePref(key, on) {
  try {
    localStorage.setItem(key, on ? '1' : '0');
  } catch {
    /* private mode: the choice lasts for this page only */
  }
}

// --------------------------------------------------------------------- help

function renderHelp() {
  el.helpGroups.replaceChildren();
  for (const group of EXAMPLE_GROUPS) {
    const section = document.createElement('section');
    section.className = 'help__group';
    const h = document.createElement('h3');
    h.className = 'help__group-title';
    h.textContent = group.title;
    const blurb = document.createElement('p');
    blurb.className = 'help__blurb';
    blurb.textContent = group.blurb;
    const list = document.createElement('ul');
    list.className = 'help__examples';
    for (const example of group.examples) {
      const item = document.createElement('li');
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'help__example';
      b.innerHTML = SEARCH_ICON;
      const text = document.createElement('span');
      text.textContent = example;
      b.appendChild(text);
      b.addEventListener('click', () => useExample(example));
      item.appendChild(b);
      list.appendChild(item);
    }
    section.append(h, blurb, list);
    el.helpGroups.appendChild(section);
  }
  el.helpTips.replaceChildren(
    ...TIPS.map((tip) => {
      const li = document.createElement('li');
      li.textContent = tip;
      return li;
    }),
  );
}

function openHelp() {
  if (typeof el.help.showModal === 'function') el.help.showModal();
  else el.help.setAttribute('open', '');
}

function useExample(text) {
  el.help.close();
  if (state.running) return;
  el.query.value = text;
  el.clearQuery.hidden = false;
  el.query.focus();
  el.query.setSelectionRange(text.length, text.length);
}

// ------------------------------------------------------------ connect (MCP)

function renderConnect() {
  el.connectUrl.value = mcpUrl();
  el.connectTabs.replaceChildren(
    ...CLIENTS.map((c) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'connect__tab';
      b.setAttribute('role', 'tab');
      b.dataset.client = c.id;
      b.textContent = c.name;
      b.addEventListener('click', () => {
        state.connectClient = c.id;
        renderConnectPanel();
      });
      return b;
    }),
  );
  el.connectTools.replaceChildren(
    ...TOOL_NOTES.map(([name, text]) => {
      const li = document.createElement('li');
      const code = document.createElement('code');
      code.textContent = name;
      li.append(code, ` ${text}`);
      return li;
    }),
  );
  renderConnectPanel();
}

function renderConnectPanel() {
  const client = CLIENTS.find((c) => c.id === state.connectClient) ?? CLIENTS[0];
  for (const tab of el.connectTabs.children) {
    const on = tab.dataset.client === client.id;
    tab.setAttribute('aria-selected', String(on));
    tab.classList.toggle('is-active', on);
  }
  const url = mcpUrl();
  const token = state.mcpToken
    ? state.tokenRevealed
      ? state.mcpToken
      : maskedToken()
    : tokenOrPlaceholder(el.connectToken.value);
  el.connectPanel.replaceChildren(
    ...client.steps.map((step, i) => {
      const wrap = document.createElement('div');
      wrap.className = 'connect__step';
      const p = document.createElement('p');
      p.className = 'connect__where';
      p.textContent = step.where;
      const pre = document.createElement('pre');
      pre.className = 'connect__code';
      const code = document.createElement('code');
      code.textContent = step.code(url, token);
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.className = 'btn btn--tonal btn--small connect__copy';
      copy.dataset.copyStep = String(i);
      copy.textContent = 'Copy';
      pre.append(code, copy);
      wrap.append(p, pre);
      return wrap;
    }),
  );
}

/** The token a copied snippet carries: the server's, else what was pasted, else the placeholder. */
function realToken() {
  return state.mcpToken ?? tokenOrPlaceholder(el.connectToken.value);
}

function snippetText(index, token) {
  const client = CLIENTS.find((c) => c.id === state.connectClient) ?? CLIENTS[0];
  const step = client.steps[index];
  return step ? step.code(mcpUrl(), token) : '';
}

/** Server token (read-only, masked, Reveal/Copy) or the paste-to-fill field. */
function applyTokenUi() {
  const token = state.mcpToken;
  el.connectTokenPaste.hidden = Boolean(token);
  el.connectTokenServer.hidden = !token;
  el.connectTokenValue.value = token ?? '';
  el.connectTokenValue.type = token && state.tokenRevealed ? 'text' : 'password';
  el.connectTokenReveal.textContent = state.tokenRevealed ? 'Hide' : 'Reveal';
  el.connectTokenReveal.setAttribute('aria-pressed', String(state.tokenRevealed));
  // The label points at whichever field is showing.
  el.connect
    .querySelector('label[for^="connect-token"]')
    ?.setAttribute('for', token ? 'connect-token-value' : 'connect-token');
}

/**
 * Ask the server whether the endpoint is on. A password-protected server also
 * hands a logged-in person its token; an open server never sends it.
 */
async function openConnect() {
  if (typeof el.connect.showModal === 'function') el.connect.showModal();
  else el.connect.setAttribute('open', '');
  state.tokenRevealed = false;
  applyTokenUi();
  renderConnectPanel();
  el.connectStatus.dataset.state = 'unknown';
  el.connectStatus.textContent = 'Checking the MCP server…';
  let info;
  try {
    info = await fetchMcpInfo();
  } catch (err) {
    if (isLoginRequired(err)) {
      el.connect.close();
      await requireLogin();
      return openConnect();
    }
    el.connectStatus.dataset.state = 'off';
    el.connectStatus.textContent = 'Could not ask the server whether MCP is on.';
    return;
  }
  const token = typeof info?.token === 'string' ? info.token.trim() : '';
  state.mcpToken = token || null;
  applyTokenUi();
  renderConnectPanel();
  if (info?.enabled) {
    el.connectStatus.dataset.state = 'on';
    el.connectStatus.textContent = state.mcpToken
      ? 'The MCP server is on. Your clients need the token below.'
      : 'The MCP server is on and requires a token.';
  } else {
    el.connectStatus.dataset.state = 'off';
    el.connectStatus.textContent =
      'The MCP server is off: set WEBSCOUT_MCP_TOKEN in the server\'s .env and restart it.';
  }
}

// --------------------------------------------------------------------- tabs

function viewFromHash() {
  return location.hash === '#stats' ? 'stats' : 'search';
}

function wireTabs() {
  for (const tab of el.tabButtons) {
    tab.addEventListener('click', () => selectView(tab.dataset.view));
  }
  // Arrow keys move between tabs, as the tablist pattern expects.
  el.tabs.addEventListener('keydown', (event) => {
    const keys = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
    if (!(event.key in keys)) return;
    event.preventDefault();
    const i = el.tabButtons.findIndex((t) => t.dataset.view === state.view);
    const step = keys[event.key];
    const n = el.tabButtons.length;
    const next = Number.isFinite(step) ? (i + step + n) % n : step < 0 ? 0 : n - 1;
    selectView(el.tabButtons[next].dataset.view);
    el.tabButtons[next].focus();
  });
  // Back and forward move between the tabs too.
  window.addEventListener('popstate', () => applyView(viewFromHash()));
  window.addEventListener('hashchange', () => applyView(viewFromHash()));
}

/** A tab click: record it in the history (so Back returns), then show it. */
function selectView(view) {
  if (view === state.view) return;
  const url = `${location.pathname}${location.search}${view === 'stats' ? '#stats' : ''}`;
  try {
    history.pushState(null, '', url);
  } catch {
    /* cosmetic */
  }
  applyView(view);
}

function applyView(view) {
  const changed = view !== state.view;
  state.view = view;
  el.app.dataset.view = view;
  for (const tab of el.tabButtons) {
    const on = tab.dataset.view === view;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
  }
  // The search panel keeps its state (a finished result stays) while hidden.
  el.viewSearch.hidden = view !== 'search';
  el.viewStats.hidden = view !== 'stats';
  updateLayout();
  toggleDownloadMenu(false);

  if (view === 'stats') {
    document.title = 'Statistics · webscout';
    if (!state.statsView) state.statsView = mountStats(el.viewStats);
    else if (changed) state.statsView.refresh();
    return;
  }
  document.title = 'webscout';
  if (state.pendingQuery) {
    const q = state.pendingQuery;
    state.pendingQuery = '';
    if (!state.running) void run(q);
  }
}

// -------------------------------------------------------------------- login

function isLoginRequired(err) {
  return err instanceof ApiError && err.status === 401;
}

/**
 * Show the login card (once, however many callers hit a 401) and resolve when
 * the person has logged in. The app underneath is hidden, not torn down, so a
 * finished result, the open tab and the options are all still there after.
 */
function requireLogin() {
  if (!state.loginPromise) {
    state.loginPromise = new Promise((resolve) => {
      state.loginResolve = resolve;
    });
    showLogin();
  }
  return state.loginPromise;
}

function showLogin() {
  for (const dialog of [el.help, el.connect]) if (dialog.open) dialog.close();
  toggleDownloadMenu(false);
  document.body.dataset.boot = 'ready';
  el.app.hidden = true;
  el.login.hidden = false;
  el.loginPassword.value = '';
  setPasswordVisible(false);
  setLoginError('');
  setLoginBusy(false);
  document.title = 'Log in · webscout';
  el.loginPassword.focus();
}

function hideLogin() {
  el.loginPassword.value = '';
  setPasswordVisible(false);
  el.login.hidden = true;
  el.app.hidden = false;
  document.title = state.view === 'stats' ? 'Statistics · webscout' : 'webscout';
  if (state.view === 'search' && !state.running) el.query.focus();
  const resolve = state.loginResolve;
  state.loginPromise = null;
  state.loginResolve = null;
  resolve?.();
}

function wireLogin() {
  el.loginForm.addEventListener('submit', (event) => {
    event.preventDefault();
    void submitLogin();
  });
  el.loginEye.addEventListener('change', () => setPasswordVisible(el.loginEye.checked));
  el.loginPassword.addEventListener('input', () => {
    if (el.loginError.textContent) setLoginError('');
  });
  el.logout.addEventListener('click', () => void logOut());

  // The statistics view reports its own 401s this way. Once logged in again it
  // is refreshed, since the request that failed is its own.
  let refreshQueued = false;
  window.addEventListener('webscout:login-required', () => {
    if (refreshQueued) return;
    refreshQueued = true;
    void requireLogin().then(() => {
      refreshQueued = false;
      if (state.view === 'stats') state.statsView?.refresh();
    });
  });
}

async function submitLogin() {
  if (state.loggingIn) return;
  const password = el.loginPassword.value;
  if (!password) {
    setLoginError('Enter the password.');
    el.loginPassword.focus();
    return;
  }
  setLoginError('');
  setLoginBusy(true);
  try {
    await apiLogin(password);
  } catch (err) {
    setLoginBusy(false);
    setLoginError(loginErrorText(err));
    el.loginPassword.select();
    el.loginPassword.focus();
    return;
  }
  setLoginBusy(false);
  hideLogin();
}

function loginErrorText(err) {
  const status = err instanceof ApiError ? err.status : 0;
  if (status === 401) return 'Wrong password.';
  if (status === 429) return 'Too many attempts. Wait a minute and try again.';
  if (status >= 500) return `The server could not log you in: ${messageOf(err)}`;
  return messageOf(err);
}

function setLoginError(text) {
  el.loginError.textContent = text;
  el.login.classList.toggle('has-error', Boolean(text));
  el.loginPassword.setAttribute('aria-invalid', String(Boolean(text)));
  if (text) {
    // Restart the nudge so a second wrong password is felt as well as read.
    el.login.classList.remove('is-shaking');
    void el.login.offsetWidth;
    el.login.classList.add('is-shaking');
  }
}

function setLoginBusy(busy) {
  state.loggingIn = busy;
  el.loginSubmit.disabled = busy;
  el.loginSubmit.setAttribute('aria-busy', String(busy));
  el.loginSubmitText.textContent = busy ? 'Logging in…' : 'Log in';
  el.loginPassword.readOnly = busy;
}

function setPasswordVisible(visible) {
  el.loginPassword.type = visible ? 'text' : 'password';
  el.loginEye.checked = visible;
}

/** Log out, forget anything secret this page holds, and show the card. */
async function logOut() {
  el.logout.disabled = true;
  if (state.running) stopRun();
  await apiLogout();
  state.mcpToken = null;
  state.tokenRevealed = false;
  el.connectToken.value = '';
  applyTokenUi();
  renderConnectPanel();
  el.logout.disabled = false;
  await requireLogin();
  if (state.view === 'stats') state.statsView?.refresh();
}
