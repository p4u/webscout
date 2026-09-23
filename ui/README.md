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
  field, the `advanced` group inside a collapsed disclosure, each option laid out by
  its `type` (`integer`/`number` → number input honouring `min`/`max`, `boolean` →
  switch, `enum` → select, `string` → text). An option added to the API appears here
  with no change to this code.
- **Only changed options are sent.** `POST /api/search` carries the query plus the
  options that differ from their schema default. "Reset to defaults" clears them, and
  a badge on the disclosure counts how many are set.
- **The stream is read incrementally.** The NDJSON response is consumed with a
  `fetch` reader, split on newlines with partial lines buffered across chunks.
  `progress` events become a timestamped activity log, `stats` feed the counters, and
  the `result` event ends the run. Stop aborts through an `AbortController`, which
  drops the connection and so aborts the run server-side.
- **Tokens and cost are visible while they are being spent.** The API sends a `usage`
  event about once a second; the panel under the activity log shows requests and
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
