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

- **Options are not hardcoded.** On load the UI fetches `GET /api/options` and renders
  every control from the returned schema: the `basic` group as chips under the search
  field, the `advanced` group inside the collapsed "More options" disclosure, each
  option laid out by its `type` (`integer`/`number` → number input honouring
  `min`/`max`, `boolean` → switch, `enum` → select showing `value_labels` when given,
  `string` → text). The `expert` group (thresholds, batch sizes, concurrency, models)
  is not rendered: those knobs are for API callers, and a person running a search has
  no reason to touch them. An option added to `basic` or `advanced` appears here with
  no change to this code.
- **The reply is its own card.** The `result` event carries the result taken apart:
  `body` (the answer prose or the records table, no frame), `summary` (what the
  outcome means), `sources` (backing the answer's `[n]` citations, in order),
  `quarantined` and `notes`. The body is shown in a raised card; each `[n]` becomes a
  numbered badge that opens the sources box at that page. Sources sit in a separate
  box, hidden until "Show sources" is pressed. Non-markdown formats show the full
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
- **Connect AI tools.** A dialog shows the MCP address (`<this page's origin>/mcp`),
  whether the server has MCP enabled (`GET /api/mcp`), and setup snippets for Claude
  Code, opencode, pi and others. A pasted token fills in the snippets; it stays in the
  page and is never sent anywhere or stored.
- **Every failure is visible.** An `error` event, a non-200, a dropped connection, a
  malformed NDJSON line and an unhandled rejection all land in the same inline panel
  with a retry button. There is no state that renders a blank page.

## Layout of the source

| File | Contains |
|---|---|
| `index.html` | The whole page skeleton; nothing is created from JS that could be static |
| `src/main.js` | App controller: run lifecycle, streaming, result rendering, errors |
| `src/api.js` | `GET /api/options`, the NDJSON `POST /api/search` generator, download URLs |
| `src/options.js` | Generic control rendering from the options schema, dirty tracking |
| `src/examples.js` | The help dialog's example questions and tips |
| `src/connect.js` | The "Connect AI tools" dialog: MCP URL and per-client setup snippets |
| `src/markdown.js` | `marked` + DOMPurify, link hardening, table wrapping |
| `src/styles.css` | All styling; one accent colour, design tokens at the top |

## Notes

- Light theme only. The palette lives in `:root` in `src/styles.css`; the accent is a
  single custom property (`--accent`).
- Responsive down to 360px: chips wrap, the advanced grid becomes one column, and the
  result actions go full width.
- The layout has two phases, driven by `data-phase` on `#app`: `idle` centres the
  search field under the product name, anything else collapses it into a compact
  sticky header.
