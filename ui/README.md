# webscout UI

A single static page that drives the webscout API. Plain HTML, CSS and ES modules,
bundled by Vite. The only runtime dependencies are [`marked`](https://marked.js.org)
for markdown and [`dompurify`](https://github.com/cure53/DOMPurify) for sanitising it.
No framework, no CSS library, and nothing is loaded from a CDN — the build inlines
everything, so the container works with no outbound network.

## Development

The UI talks to the API over same-origin `/api/...` requests. In development Vite
proxies that prefix to a locally running API:

```bash
# terminal 1 — the API, from the repo root
source env
cargo run --release -- --api --api-port 8080

# terminal 2 — the UI
cd ui
npm ci          # or npm install
npm run dev     # http://localhost:5173
```

`npm run dev` proxies `/api` to `http://127.0.0.1:8080`. Point it somewhere else with
`WEBSCOUT_API_ORIGIN`:

```bash
WEBSCOUT_API_ORIGIN=http://api-host:8080 npm run dev
```

`?q=<query>` in the URL runs that search on load, which makes a result linkable and
re-runnable. The UI also writes the current query into the URL as you search.

## Production build

```bash
npm ci
npm run build       # writes ui/dist/
npm run preview     # serves ui/dist/ on :4173 with the same /api proxy
```

`npm run build` emits a hashed JS bundle, a hashed CSS file and `index.html` into
`ui/dist/`. That directory is the whole deliverable: serve it from any static server
with `/api/` proxied to the API so the browser makes same-origin requests and no CORS
configuration is needed. `ui/Dockerfile` (owned by the packaging work) does exactly
that — node builder, then nginx serving `dist/` with an `/api/` proxy to the `api`
service.

## What it does

- **It looks like Google Search.** The home page is Google's: text links and a round
  log-out button in the top-right corner, the name as a large four-colour wordmark
  (Google's blue, red, yellow and green; system font stack, no web font), the 584px
  pill field, and two grey buttons under it — "webscout Search" and "Surprise me",
  which runs a random question from `src/examples.js` the way "I'm Feeling Lucky"
  skips the results. The search options sit under the buttons, and a grey footer
  closes the page. Once a search runs the page becomes a results page: a sticky
  header with the small wordmark, the field and Search / Statistics tabs underlined
  in blue, with the options behind "Tools"; the answer in a bordered block like an
  AI Overview; and each source drawn as a search result (site circle, name and
  breadcrumb URL, blue title, support level).

- **Options are not hardcoded.** On load the UI fetches `GET /api/options` and renders
  every control from the returned schema: the `basic` group as chips under the search
  buttons (behind "Tools" on the results page), the `advanced` group inside the collapsed "More options" disclosure, each
  option laid out by its `type` (`integer`/`number` → number input honouring
  `min`/`max`, `boolean` → switch, `enum` → select showing `value_labels` when given,
  `string` → text). The `expert` group (thresholds, batch sizes, concurrency, models)
  is not rendered: those knobs are for API callers, and a person running a search has
  no reason to touch them. An option added to `basic` or `advanced` appears here with
  no change to this code.
- **The reply is its own block.** The `result` event carries the result taken apart:
  `body` (the answer prose or the records table, no frame), `summary` (what the
  outcome means), `sources` (backing the answer's `[n]` citations, in order),
  `quarantined` and `notes`. The body is shown in a bordered "Verified answer" block
  with the outcome, copy and download; each `[n]` becomes a numbered badge that opens
  the sources at that page. Sources are listed under the block as search results,
  hidden until "Show sources" is pressed; the same number rides on each result's site
  circle. Non-markdown formats show the full
  document as an escaped code block instead.
- **Examples & help.** A dialog lists example questions for each kind of request
  (quick facts, current information, yes-or-no checks, lists, comparisons, history);
  picking one fills the search field without running it. Examples live in
  `src/examples.js`.
- **Only changed options are sent.** `POST /api/search` carries the query plus the
  options that differ from their schema default. "Reset to defaults" clears them, and
  a badge on the disclosure counts how many are set.
- **The stream is read incrementally.** The NDJSON response is consumed with a
  `fetch` reader, split on newlines with partial lines buffered across chunks.
  `progress` events become a timestamped activity log, `stats` feed the counters, and
  the `result` event ends the run. Stop aborts through an `AbortController`, which
  drops the connection and so aborts the run server-side.
- **Tokens and cost are available, not imposed.** The panel is hidden until "Show
  tokens & cost" in the activity bar is pressed (remembered in `localStorage`). The
  API sends a `usage` event about once a second; the panel shows requests and
  tokens for Jev, the writer and the planner, plus a running total in USD. It appears
  at zero the moment a run starts and settles on the finished run's own stats, so the
  final numbers are the same ones the `result` event carries. The planner row is
  hidden when it made no requests (it shares the writer's model unless one was
  named), reasoning tokens are shown only when some were spent, and a component whose
  endpoint does not report a cost shows `—` rather than `$0.0000` — only OpenRouter
  returns a real price, and inventing one would be worse than admitting the gap.
- **Rendered markdown is treated as hostile.** Answers embed text scraped from the
  web, so `marked` output goes through DOMPurify before it reaches the DOM, scripts
  and event-handler attributes are stripped, and every link is forced to
  `target="_blank" rel="noopener noreferrer"`. Non-markdown formats (json, csv, jsonl)
  are shown as escaped code blocks, never parsed as markup.
- **Login.** On load the page asks `GET /api/session`. When the server has a password
  (`WEBSCOUT_PASSWORD`) and there is no session, a login card replaces the app: one
  password field in a Google-style sign-in card (floating label, "Show password"
  checkbox), Enter submits `POST /api/login`, and a wrong
  password (401), throttling (429) or an unreachable server is shown inline. The
  session is an HttpOnly cookie, so the page never handles it; every request sends it
  with `credentials: 'same-origin'`. Any later 401 (options, search, `/api/mcp`, or
  the statistics view's `webscout:login-required` event) brings the card back, and
  after logging in the interrupted action carries on: the search starts again, the
  Connect dialog reopens, the statistics refresh. The app is hidden, not torn down,
  so results and the open tab survive. The round log-out button in the top-right
  corner (only shown when a password is set) calls `POST /api/logout`. A server without `/api/session` (older
  build, 404) or with no password boots straight into the app.
- **Search and Statistics tabs.** On the home page Statistics is a link in the
  top-right corner and in the footer; on the results page the Search and Statistics
  tabs sit under the field. Statistics (`src/stats.js`, mounted on
  first open, `refresh()`ed on every return) is kept in the URL hash (`#stats`), so
  reload and Back/Forward keep the view; the search form and results are hidden, not
  reset, while it shows. A `?q=` link opened on `#stats` waits to run until the Search
  tab is shown.
- **Connect AI tools.** A dialog shows the MCP address (`<this page's origin>/mcp`),
  whether the server has MCP enabled (`GET /api/mcp`), and setup snippets for Claude
  Code, opencode, pi and others. On a password-protected server a logged-in person
  gets the server's token in that response (`token`): it is shown read-only and
  masked with Reveal and Copy buttons, and filled into every snippet (masked on
  screen until revealed; the snippet Copy buttons always copy the real value). It is
  kept in memory only, never logged, and dropped on logout. An open server never sends
  the token; a pasted one fills in the snippets instead and stays in the page.
- **Every failure is visible.** An `error` event, a non-200, a dropped connection, a
  malformed NDJSON line and an unhandled rejection all land in the same inline panel
  with a retry button. There is no state that renders a blank page.

## Layout of the source

| File | Contains |
|---|---|
| `index.html` | The whole page skeleton; nothing is created from JS that could be static |
| `src/main.js` | App controller: run lifecycle, streaming, result rendering, errors |
| `src/api.js` | `GET /api/options`, the NDJSON `POST /api/search` generator, session/login/logout, `GET /api/mcp`, download URLs |
| `src/stats.js`, `src/charts.js`, `src/stats.css` | The Statistics tab (`mountStats(container)` → `{ refresh, destroy }`) |
| `src/options.js` | Generic control rendering from the options schema, dirty tracking |
| `src/examples.js` | The help dialog's example questions and tips |
| `src/connect.js` | The "Connect AI tools" dialog: MCP URL and per-client setup snippets |
| `src/markdown.js` | `marked` + DOMPurify, link hardening, table wrapping |
| `src/styles.css` | All styling except the statistics view; Google-like tokens (colours, fonts, the results column's `--lead` and `--col`) at the top |

## Notes

- Light theme only. The palette lives in `:root` in `src/styles.css`; the accent is a
  single custom property (`--accent`), and the statistics view reads the same tokens.
- Responsive down to 360px, following Google's phone layout: a smaller wordmark and
  icon buttons in the corner on the home page; on the results page the wordmark is
  centred above a full-width field, the tabs scroll sideways, and dialogs open as
  bottom sheets.
- Buttons and chips centre their contents with flex and symmetric padding (a chip's
  caret is drawn inside its select, so label, value and caret are one centred run).
  The text centres of every button, link-button, tab and chip were measured against
  their boxes at 1280px and 390px: all within 0.5px.
- The run has phases (`data-phase` on `#app`: `idle`, `running`, `done`) and the page
  has a tab (`data-view`: `search`, `stats`). Together they set `data-layout`, which
  the CSS keys on: `hero` (idle Search tab) is the Google home page; `compact` (a run,
  or the Statistics tab) is the results page's sticky header. `data-tools` on `#app`
  (`open`/`closed`) shows the options row under the tabs there.
