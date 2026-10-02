---
name: webscout
description: How to research the web with the webscout MCP server (tools search, start_search, get_search_result and friends), which returns verified answers with sources and verified lists of organisations, people or products. Use this skill whenever webscout tools are available and the task needs anything from the web — a fact, a current officeholder, a date, an organisation profile or contact details, a comparison, checking a claim, or building a list of companies, associations, projects or leads with their websites and emails — even if the user does not say "webscout" or "search". Prefer it over generic web search or page fetching for anything that must be correct and cited.
---

# Using webscout

webscout is a research service behind an MCP server. You give it one request in plain language; it plans searches, reads the pages it finds, checks every fact against its source, and returns only what it could verify:

- a **question** ("Who is the current CEO of GitLab?") gets a written **answer with numbered sources**;
- a **list request** ("Find 20 coworking spaces in Barcelona with a public email") gets a **table of items**, each with its source and a status.

Because the checking happens on the server, a webscout result is safer to rely on than your own reading of search snippets. It is also slower than a plain search: questions usually take 15 seconds to 2 minutes, lists 5 to 20 minutes. Plan for that, and do not reach for webscout for things you already know or that do not need the web.

## The basic loop

1. Call `search` with the request. It waits about 10 seconds; a quick question may come back finished in that same call.
2. If the response says `"status": "running"`, call `get_search_result` with its `request_id`. Each call waits about 10 seconds for the search. **Call it again immediately** while the status is still `running` — there is no need to sleep between calls, the waiting happens server-side.
3. When the status is `finished`, read the result (see "Reading results").

Never start the same search twice to "speed it up" or because it is taking a while: the first one is still running and a duplicate only doubles the cost. If you lose a `request_id`, `list_searches` shows the server's recent searches with their ids, newest first.

Use `start_search` instead of `search` when you want a long search (usually a list) to run in the background while you do other work, or to run several searches in parallel: it returns the `request_id` at once. Collect each result later with `get_search_result`. The server runs only a few searches at once (4 by default); if you get a "searches are already running" error, wait for one to finish or cancel one.

`get_search_status` reports progress without waiting (stage, round, pages read, items found, cost so far, recent progress lines) — useful to tell the user how a long list is going. `cancel_search` stops a search; nothing it found is kept.

## Writing a good request

webscout reads your request literally, the way a careful researcher would read a brief. A clear brief is the biggest lever you have on quality.

**Name the subject precisely.** Use the organisation's full name and, if it has one, its acronym: "the Il·lustre Col·legi de l'Advocacia de Barcelona (ICAB)", not "the Barcelona bar". Ambiguous names get ambiguous evidence.

**Spell out every part you need.** A multi-part question is checked part by part, and the search keeps going until each part is found or ruled out. "Who is the Director General of ANFAC, since when, and what is the published email of its communications director?" gets three parts searched for; "tell me about ANFAC" gets whatever turns up.

**Say what you mean by time.** webscout knows today's date. "The current dean", "the next election" and "the latest release" are understood, and values it has to work out (the next election from a four-year mandate that began in July 2025) come back labelled as estimates.

**For lists, give the count, the criteria and the fields.** "Find 10 professional colleges (colegios profesionales) in Aragón that will hold Junta de Gobierno elections in 2026 or 2027. For each: official name, website, general email, and the election year with the source." The count is when to stop; the criteria are checked per item; the fields are what is looked up for each item.

**Ask for the kind of contact you want.** "General contact email" gets the organisation's general address ahead of press or department addresses.

**One request per search.** Several unrelated questions in one request dilute each other. Run separate searches (in parallel with `start_search` if you like).

Any language works; use the one natural to the subject if that helps ("colegios profesionales", "Junta de Govern").

## Reading results

`get_search_result` returns the result text by default (`detail: "simple"`): the answer, or the table of items. Ask for `detail: "full"` when you need the sources with their support scores, the notes, the cost, the run statistics or the structured records.

The **outcome** says how far the result got:

| outcome | meaning | what to do |
|---|---|---|
| `complete` | Every part of the question is answered (or the list target was reached) from verified sources. | Use it. |
| `partial` | Some parts are answered; the notes say which are missing ("Not found in the evidence: contact email"). | Report what was found, say what was not, and consider a narrower follow-up search for the missing part. |
| `truncated` | A list stopped at its round limit while still finding items. | Re-run with a higher `max_rounds` if the user needs more. |
| `empty` | Nothing could be verified. | This is absence of evidence, not evidence of absence. Rephrase, or tell the user nothing reliable was found. |

Signals inside an answer:

- **Citations** like `[3]` point to the numbered sources (`detail: "full"` lists them). Keep them when you relay facts, or name the source.
- **`(estimated: …)`** marks a value worked out from stated facts ("2029 (estimated: the four-year mandate began in July 2025)"). Pass the label on; do not present it as a stated fact.
- **`[unsupported]`** after a sentence means no source backs that sentence. Treat it as unverified.
- **"not found in the sources"** in the answer means exactly that for that part.
- The **notes** explain caveats: sources that disagree, superseded information, pages ignored because they tried to instruct the AI ("quarantined").

For a **list**, each row carries a **status**: `complete` (every requested field found and every criterion verified), or what is missing or unverified ("missing: website", "unverified: will hold elections in 2026 or 2027"). Complete rows come first. Only complete rows have been verified against all the criteria — if you present the other rows, say what each one lacks. The list is cut to the requested count by default; pass `limit: 0` to get every row found, or a number for fewer. `records_total` and `records_complete` (in `detail: "full"`) give the counts.

To hand the user a file, use `format`: `csv` (a spreadsheet, with a source column per field), `markdown`, `json` (everything, including every passage read — large) or `jsonl`. Exports are never cut by `limit`. For reading the result yourself, use `detail` instead.

## Options

`search` and `start_search` take a few options directly. Leave them at their defaults unless you have a reason:

| option | values | when to change it |
|---|---|---|
| `preset` | `quick`, `standard` (default), `thorough` | `quick` for a simple, stable fact when speed matters; `thorough` reads more pages per round, for hard questions or sparse topics — it costs more and is not reliably better, so try `standard` first. |
| `max_rounds` | `auto` (default) or 1–200 | `auto` stops by itself when the answer is complete or progress levels off. Give a number to allow exactly up to that many rounds — e.g. to push a `truncated` list further. |
| `no_enrich` | true / false | For lists: skip looking up each item's missing details (email, website…) and keep only what list pages say. Much faster, much less complete. |
| `no_follow` | true / false | For lists: do not open the links found on list pages. Faster, may find fewer items. |
| `no_search_cache` | true / false | Ignore search results cached in the last few hours. Use it when something changed very recently. |

Everything else (search engines, thresholds, batch sizes, models) goes in the `options` object and is rarely worth touching; `list_search_options` describes every option with its default and range. The ones occasionally useful: `search_engines` (`auto`, `ddg`, `jina`) to pin one engine, and `llm_model` / `planner_model` / `extract_model` to try different models for writing, planning or list extraction.

## Cost and time

Each search costs real money on the server (language models plus verification calls). As a rough guide: a question costs about $0.01–0.04 and takes 15 seconds to 2 minutes; a 10–30 item list costs about $0.10–0.50 and takes 5–20 minutes. `detail: "full"` reports the exact cost. Prefer one well-written request to several loose ones, do not repeat a finished search hoping for a different result, and use `no_enrich` when the user only needs names.

## Patterns that work

- **Profile one organisation**: one `search` naming it precisely and listing every field wanted (members, leadership and since when, mandate, next election, general email, phone). Follow up with a narrower search only for parts the notes say are missing.
- **Build a list, then profile**: a list search to find the candidates, then individual searches for the few rows that matter most.
- **Long list while you keep working**: `start_search`, tell the user it is running, check `get_search_status` when they ask for progress, collect with `get_search_result` when done.
- **Check a claim**: phrase it as a question ("Is info@example.org the official contact address of Example Org?"). A resolved "no" — the organisation's own contact page lists a different address — is a valid answer.
- **Relay honestly**: give the facts with their sources or citations, keep estimate labels, and say plainly what was not found. The user chose webscout because the answer is verified; do not blend in unverified guesses of your own.
