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

- **A search page with its own face.** One brand colour, a deep teal (`--accent`),
  marks everything that is webscout's: the logo, links, focus rings, the active tab
  and the primary button. The logo is an original mark — a "w" drawn as two check
  strokes, the last one rising like a tick (web search that checks what it returns) —
  in a rounded teal square, beside the name with "web" in ink and "scout" in teal
  (system font stack, no web font). The mark is one inline SVG `<symbol>` in
  `index.html`, used by the home page, the results header and the login card; the
  favicon is the same drawing as a data URI. The home page centres the logo, a pill
  search field and a single primary **Search** button (Enter searches too), with the
  search options under it, quiet text links in the top-right corner and a one-line
  footer. Once a search runs the page becomes a results page: a sticky header with
  the small logo, the field, the tokens-and-cost pill and the Search / Statistics
  tabs, with the options behind "Tools"; the answer in a bordered "Verified answer"
  block; and each source listed with its site circle, name and breadcrumb URL, title
  link and support level.

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
- **Tokens and cost sit in the header.** On the results page a pill right of the
  search field shows the run's total cost and how many tokens went through the models
  ("$0.0053 | 51k tokens"). It appears the moment a search starts and fills from the
  `usage` events the API sends about once a second (a breathing dot while the run is
  live), then settles on the finished run's own stats, so the final numbers are the
  ones the `result` event carries. It keeps the last run's numbers until the next
  search or a return to the home page, and is not shown on the idle home page or the
  Statistics tab. Clicking it opens a popover (Escape, the close button or a click
  outside closes it) with a table per component — Jev, the writer and the planner:
  requests, tokens in and out, reasoning tokens and cost, plus a total row — and the
  run's elapsed time, pages and rounds. The planner row is hidden when it made no
  requests (it shares the writer's model unless one was named). Jev has no output or
  reasoning tokens and shows a dash there. A cost that is not known shows `—`, never
  `$0.0000`: a component whose endpoint does not report a price (only OpenRouter
  does) makes the total unknown rather than a partial sum, and so does a run that
  has not reported usage yet. On a phone the pill shows the icon and the amount only,
  and the breakdown opens as a bottom sheet.
- **Rendered markdown is treated as hostile.** Answers embed text scraped from the
  web, so `marked` output goes through DOMPurify before it reaches the DOM, scripts
  and event-handler attributes are stripped, and every link is forced to
  `target="_blank" rel="noopener noreferrer"`. Non-markdown formats (json, csv, jsonl)
  are shown as escaped code blocks, never parsed as markup.
- **Login.** On load the page asks `GET /api/session`. When the server has a password
  (`WEBSCOUT_PASSWORD`) and there is no session, a login card replaces the app: one
  password field in a sign-in card under the logo (floating label, "Show password"
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
| `src/styles.css` | All styling except the statistics view; the design tokens (the teal accent and neutrals, fonts, the results column's `--lead` and `--col`) at the top |

## Notes

- Light theme only. The palette lives in `:root` in `src/styles.css`; the accent is a
  single custom property (`--accent`, with `--accent-hover`, `--accent-ink`,
  `--accent-wash` and `--accent-line` derived by hand), and the statistics view reads
  the same tokens: its spend and depth series are ramps of the accent, and "truncated"
  is a slate blue so it never reads as teal beside "complete" green.
- Responsive down to 360px: a smaller logo and icon buttons in the corner on the home
  page; on the results page the logo, the cost pill and the icon buttons share the
  top row above a full-width field, the tabs scroll sideways, and dialogs and the cost
  breakdown open as bottom sheets.
- Buttons and chips centre their contents with flex and symmetric padding (a chip's
  caret is drawn inside its select, so label, value and caret are one centred run).
  The text centres of every button, link-button, tab and chip were measured against
  their boxes at 1280px and 390px (the home Search button and the cost pill included):
  all within 0.5px.
- The run has phases (`data-phase` on `#app`: `idle`, `running`, `done`) and the page
  has a tab (`data-view`: `search`, `stats`). Together they set `data-layout`, which
  the CSS keys on: `hero` (idle Search tab) is the centred home page; `compact` (a run,
  or the Statistics tab) is the results page's sticky header. `data-tools` on `#app`
  (`open`/`closed`) shows the options row under the tabs there.
