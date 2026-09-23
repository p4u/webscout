# webscout

Search the web and return **verified** results.

A generative model does the writing — inventing search queries, pulling records out
of page text, composing prose. [TypeSafe](https://docs.typesafe.ai)'s Jev model does
the judging and writes nothing. [Obscura](https://github.com/h4ckf0r0day/obscura) is
compiled into the binary and does the fetching.

```bash
webscout "what is the current stable release of Go"
webscout -f csv "at least 100 cooperative names with their email addresses"
webscout -vv -f json --thorough "EU AI Act obligations for GPAI providers" > out.json
```

## Why two models

A generative model asked to pull a hundred email addresses out of scraped text will,
somewhere in the tail, produce one that looks entirely plausible and does not exist.
You cannot tell which one by reading the output — that is the whole problem with
using an LLM as a research tool.

So generation is fenced on both sides. Nothing reaches the generative model before
Jev clears the input, and nothing leaves it before Jev checks the output against the
page it supposedly came from:

```
        +------------ obscura (embedded) ------------+
        |   search DDG          render pages         |
        +--------+------------------+----------------+
                 v                  v
            Jev: triage        Jev: injection screen ---> quarantine
                 |                  |
                 v                  v
            worth reading?     safe to show the model?
                                    |
                                    v
                          qwen: extract records
                                    |
                                    v
                      Jev: is each value actually
                         written in the source?  ---------> rejected
                                    |
                                    v
                              verified records
```

Jev returns calibrated probabilities and never text. qwen writes and never decides.
Rust owns the loop, the thresholds, and the output.

## The loop

There is **no time limit**. A clock would truncate a run that was still producing,
which is exactly the run you wanted to finish. It stops on one of four conditions:

1. the target count is reached,
2. N consecutive rounds find nothing new (`--max-barren-rounds`, default 3),
3. the planner runs out of genuinely new search angles,
4. the round ceiling (`--max-rounds`, default 40) — reported as `truncated` if it
   was still finding things, so you know to re-run with a higher ceiling.

For **answer** missions: plan queries → search → Jev-triage the hits → fetch the
best → screen, extract, verify each chunk → dedupe into the store → replan against
what is still missing. The planner is told which queries were already tried and which
domains have been productive, because the failure mode of a naive loop is re-issuing
paraphrases of a query that already worked and mining the same page forever.

For **harvest** missions the round has more moving parts, described below.

## Two mission shapes

The request is classified automatically, and the distinction matters more than it
looks. "Give me at least 100 cooperatives with emails" is not a question — it is an
enumeration that is not finished until it has 100 rows. Treating it as a question is
why naive research agents reply "here are some cooperatives" and stop at six.

| | **answer** | **harvest** |
|---|---|---|
| Trigger | explanation, fact, comparison | a list of entities, usually with a count |
| Output | cited prose + quoted passages | deduplicated records, each with a source |
| Done when | evidence answers the question | target count reached, or web exhausted |

The classifier is itself checked: qwen parses the request, then Jev is asked whether
the parse is faithful to what was written. If Jev scores the parse below 0.5 the
classifier re-runs with the low-confidence verdict included, and the higher-scoring
parse wins. Misreading the mission is the most expensive possible error, because
every later stage inherits it.

### Harvest rounds

Each harvest round runs in up to four phases.

**1. Discover.** Triage and extraction use a stage-specific goal — "Find pages that
name or list {topic}{, that meet each constraint}" — rather than the full user
request. The difference is measurable: on the query `Provide a list of 100
Ayuntamientos from Spain, with their contact e-mail address, that have executed
"Presupuestos Participativos" at least once`, the full request returned relevance 0.21
against the best real candidate page; the narrowed discovery goal returned 0.91
against the same page. Running triage with the full request produced 92 search hits,
0 pages fetched, 0 records — because no single page would deliver the whole request.

A constraint that merely names a contact field ("publish a contact email for
participation matters") is dropped from the discovery goal: no directory page
demonstrates it in a snippet, and demanding it rejected every candidate —
measured 2026-09-21, a "municipalities that ran citizen consultations and
publish a participation email" run scored every hit 0.03–0.13 relevance, read
zero pages, and ended `empty`. The contact detail stays a field (enrichment
fetches it per entity) and the constraint stays on the record, where the
post-enrichment re-check upgrades it from the enriched fact.

The same drop applies to a constraint that **mirrors a field's own
name-tokens** ("publishes pricing publicly" against the field
`publishes_pricing_publicly`, "claims end-to-end verifiability" against
`claims_end_to_end_verifiability`, "is grouped by its autonomous community"
against `autonomous_community`): the property is a fact of the entity's own
page, which enrichment fetches, not something a listing states about each
item. The rule is a token-subset test — every name-token of the field must
appear in the constraint — so `founded_year` against "founded after 2020"
keeps its constraint listable ("year" is absent, and a founding year IS a
directory-listable fact), as does `seed_round_size` against "raised a seed
round". As with contact details, the constraint stays on the record and the
post-enrichment re-check settles it from the enriched fact.

A third drop is a **judgment, not a token rule**. At harvest start one batched
Jev ask (`judge_listability`) puts one noul per constraint — *would a page that
lists many of these entities (a directory, register, or member list) state
this property per item?* — and anything scoring below 0.5 is flagged
unlistable in `Mission::unlistable_constraints` and left out of every stage
goal, exactly like the two heuristics. A failed ask changes nothing: every
constraint stays listable and the heuristics still apply. One verdict is
overridden in code: a constraint carrying one of the mission's anchor tokens
names the set itself — the register of CDTI NEOTEC 2024 grantees *is* the list
of things satisfying "awarded grants in the CDTI NEOTEC 2024 call", demonstrated
by inclusion rather than restated per row — so `constraint_names_the_set` keeps
it in every goal regardless of the per-item reading (bare-year and common-word
anchor tokens carry no set identity and never trigger the override; the ask's
wording teaches the by-inclusion case too). Measured 2026-09-21 (q81): without
the override the ask read both constraints unlistable, the goal collapsed to
the bare entity type, triage scored the ministry's own resolution pages
0.08–0.13, and a generic companies directory supplied 146 records of which
exactly 1 was complete.

The token rules above approximate that judgment, and both miss whole shapes.
Measured 2026-09-21 (q84, "member companies of the AEI Cyber Security
association with more than one office in Spain"): the constraint names the
fact — the parse dutifully created the field `office_count_in_spain` — but
"count" never appears in the constraint wording, so the mirror rule does not
fire, and no association member list states office counts per member. Every
stage goal therefore demanded it: triage scored the association's own
member-list pages 0.07–0.14 relevance, thirteen rounds fetched and re-fetched
those very pages while steer reported `no_sources`, and the run ended
`empty`. The topic itself carried the qualifier too ("… with multiple offices
in Spain"), so the parse prompt now forbids per-item factual qualifiers
(counts of offices or employees, founding years, revenue, prices) in `topic`
for the same reason it already forbids contact details.

Stage goals go further than the constraint drop: they render through a
**listing core** rather than the topic itself. The same harvest-start request
that judges listability also carries one `choice` over code-built clause
prefixes of the topic (cut at " that ", " which ", " who ", ", ", " and ",
" with " — the markers a classifier uses when it folds a request's conditions
into the topic), with the full topic always the last option. The selected
core is stored in `Mission::listing_core` and used by `discovery_goal` and
`extraction_goal`; query building and steer keep the full topic, so searches
stay specific while the accept-side goal names a list that can actually
exist. The judge selects among prefixes the code built — it is never offered
a string it could invent.

Why that is needed even with the prompt rule: measured 2026-09-21 (q63,
"organizations using Decidim that ran a vote in 2025 and the organization
responsible for managing that vote") — the parse, running the new prompt,
still folded both conditions into the topic; triage scored
decidim.org/installations/ 0.19–0.23 for nine rounds (a clean goal on that
page measures 0.91) and the run returned 1 record. A failed ask or a `full`
pick leaves the goals on the topic, which is the previous behaviour.

Record **extraction** runs on the same filtered goal (`extraction_goal`),
not on the raw request. Its prompt once carried `mission.query` plus every
constraint verbatim, and an extractor asked for entries that "publish a
contact email" from a page that lists names only correctly returned zero
records — measured 2026-09-21 (q41 rerun): five pages read,
`records_extracted=0` on every one, run `empty` in two minutes while triage
and screening (which did use the filtered goal) worked normally. Judging
the unlistable constraints is grounding's job (five-way choice against the
passage) and the post-enrichment re-check's (enriched facts inlined); a
second, blunter gate in the extractor only starves the pipeline.

Extraction during discovery treats every field except the entity name as optional:
if a page lists municipality names without emails, those names are kept and enriched
later. Grounding then checks three things in one request per chunk:

- **Entity presence** — is the entity name written in this passage?
- **Field association** — is this value given *as the field of this entity*, not just
  present somewhere on the page? (`"Ayuntamiento de Getafe"` and `"Getafe"` are judged
  separately; Jev reads literally.)
- **Constraint support** — does the passage state that this entity meets the mission
  constraint? Each check uses a one-sentence gloss of equivalent wordings (`constraint_glosses`,
  populated by `parse_mission`) inlined into the noul. Measured on decidim.org/partners/
  (2026-09-18): the bare constraint "integrators of Decidim.org" scored 0.22–0.26 per
  record on a page that says "Service providers … collaborate with Decidim"; the
  context-aware gloss scored 0.91. A separate **page-level** `p{j}` noul asks whether
  the page as a whole lists the target kind of entity (scored 0.93 on the same page).
  A record below `constraint_floor` is rescued when the page-level score is ≥ 0.70 and
  the per-record score is still ≥ 0.25 — the page's title, headings and surrounding
  context make the constraint plain for every listed entity, even when the per-item
  passage uses different words.

Before consulting Jev about a regex-detectable field (email, URL, phone), the code
confirms the candidate value literally appears in the passage text; Jev is only asked
when the code-level check passes.

**2. Follow links.** Pages that yielded two or more entities get their outbound
links Jev-scored in one request per page: for each link, a `noul` question asks
whether following it would reach more entities for the goal or the next page of the
same list. Links scoring at or above `follow_floor` (default 0.60) are queued ahead
of search results in the next round — they bypass search and triage because they have
already been judged. The link text and href are both shown to Jev; the destination is
not fetched again at this stage. Successful link follows count in `stats.links_followed`.

Seed pages (depth 1) and anchor-domain pages (depth 2) are handed to this phase
regardless of entity yield — the vendor's own site is worth navigating even when the
entry-point page names no entities directly. The per-page link cap is 20 for
anchor-domain pages (vs. the default 12), because the vendor's navigation is the most
trustworthy index of its own content. The Decidim run (2026-09-18) reached
`/partners/` from the `decidim.org` homepage via one link-selection request
(Jev probability 0.61–0.64), in a run where no search result had contained that URL.

Language-variant URLs are deduplicated before Jev sees them: `/es/partners/`,
`/ca/partners/`, and further translations collapse to one key (`lang_dedup_key` strips
the leading `/xx/` or `/xx-YY/` path segment). The first variant is followed; the
rest are dropped. The Decidim run saw 11 language variants deduplicated this way.

**3. Enrich.** After discovery, entities that still lack a required field are searched
for directly. A query template is produced once per run by the LLM — for example
`"{entity} correo electrónico contacto"` for a Spanish email field — then all
per-entity searches for that field are issued in one batch with `enrich_results_per_query`
results each (default 5). The top results are Jev-triaged with the enrich-specific
goal — "Find the official website or contact page of {entity} that gives its {field}"
— and the best `enrich_read` pages (default 2) are fetched.

For regex-detectable fields, candidates are extracted from the page text first.
A `choice` question then lists those candidates plus a `none` option ("None of these
is the {field} of {entity}"), with the page text as state. A chosen value whose
confidence meets `select_confidence` (default 0.50) is accepted; its per-field
provenance records the source URL and Jev's confidence as the grounding score. No LLM
call is made on this path. A post-pick `noul` then confirms the value is genuinely
this entity's field, not a co-resident value on a shared contact page.

For non-regex fields, the LLM extracts a single value from the page text and Jev
checks field association as in discovery.

**4. Steer.** At the end of each round, one Jev request asks: is the request
satisfied (`p ≥ 0.70` → stop), is the web exhausted (`p ≥ 0.70` with no queued
queries → stop), and what is the bottleneck (`no_sources` / `missing_fields` /
`wrong_entities` / `none`). The code maps the bottleneck verdict to an action —
raise the enrich batch, widen the search, tighten the constraint wording — and asks
the LLM only to write the new query strings, never to decide whether to continue.

### Near-collision merge

When a newly discovered entity's normalized key does not match any existing key
exactly but is a prefix, suffix, or substring of one — or shares enough tokens — Jev
is asked one `score` question with three levels: "different entities" / "possibly the
same, ambiguous" / "the same entity". The `score` field is a **level index** (0, 1,
or 2): a score ≥ 1.5 triggers a merge that keeps the better-grounded value for each
field and unions provenance. These checks are batched when several near-collisions
arrive in one round.
Before any pair is offered, **category echoes** are dropped: a record whose
name is made only of the entity type's own words ("crm" for a mission about
"open-source CRM projects", "Barcelona" for one about coworking spaces in
Barcelona) is the category or the place, not an entity. Leaving one in is not
neutral: containment matches it against every name containing the word, and
transitive merges then swallow unrelated real entities through it. Measured
2026-09-21, an open-source-CRM harvest: one bare `crm` record chained EspoCRM,
CiviCRM, ChurchCRM, DayByDayCRM and more into a single merged group, and
CiviCRM — one of the largest open-source CRMs — vanished from the output.
Drops are counted in `stats.category_echoes_dropped`. The drop alone is not
enough: pair candidates must be built over the post-drop key set. An echo
removed from the store but left in the key list still serves as a union-find
bridge — `crm` chains every *CRM name into one group through containment pairs,
and the merge absorbs them into an arbitrary survivor while the echo itself is
gone from the output (measured 2026-09-21: `krayin` kept, sixteen real projects
including CiviCRM and SuiteCRM dropped into it).

Names are one collision signal; an **identical published email** is another, and
the stronger one. A brand's directory page names each branch and every branch
carries the brand's one contact address, so "CREC Gràcia" and "CREC Cerdà" —
one token in common, invisible to the 60% overlap rule — both publish
`info@crec.cc`. Records sharing a normalised email address are paired
star-shaped (n−1 questions for a group of n, not n²/2) and asked a question
that states the shared address: one operator with several locations, or
independent organisations that share a mailbox? The distinction is why the
merge is never automatic on email equality — a federation directory that stamps
its own address on every member would otherwise collapse fifty cooperatives
into one. Measured 2026-09-21, "10 coworking spaces in Barcelona with a public
contact email": without the signal, 9 CREC branches and 2 Aurea branches made
the run report `complete` at 14 records while the emails named 4 operators;
with it, the group collapses and the run keeps searching.

The question carries the mission's entity type as context — *"These are names of
pricing plans, so an appended number, price or qualifier that does not change what
is named is decoration, not identity."* Measured on linear.app/pricing
(2026-09-20): without the context, `Basic $10` against `Basic` scored below the
merge floor — to a literal reader a bare number is identity-carrying — and the run
emitted the same plan twice.

Two floors, one per signal: name pairs merge from "the same" (≥ 1.5), email
pairs from "ambiguous" (≥ 1.0). The identical published address is itself
evidence, so a literal reader's "branch or unrelated — could be either" plus
that address is a merge; measured 2026-09-21, two listings of one operator
sharing `info@coworking-bcn.es` scored 1.44 and stayed apart under the single
1.5 floor. Genuinely different organisations still score 0 and never merge.

The merge runs twice per round: before enrichment (so the field-fill budget is
not spent twice on rows that turn out to be the same) and again after it, when
enrichment actually filled something. The second pass exists because a
shared-email pair can only arm once enrichment has attached the operator's one
address to every branch, and a run that satisfies after a single round
otherwise ends before any pass sees it (measured 2026-09-21: four CREC
branches all enriched to `info@crec.cc`, but the round's pre-enrichment merge
had already run).

### Anchor-site enumeration corroboration

When the mission is about one organisation's own enumeration — its plans, its
products, its members — that organisation's site is an authority: whatever the
site's complete list omits, a third party listing it is asserting, not
corroborating.

Pages on the anchor's site (`on_anchor_site`: a domain match, or a host label
equal to the anchor stem) that name at least `enum_min_yield` (3) distinct
entities are asked one completeness noul — *"Taken as a whole, do these passages
constitute a complete enumeration of all {entity_type}, none deliberately
omitted, rather than a partial list, a set of examples, or one region's or one
federation's members?"* A page scoring at least `enum_completeness_floor` (0.80)
becomes an enumeration authority, up to `enum_max_pages` (3). At merge time,
records whose only provenance is third-party are checked against those
authorities with batched presence nouls; a record no authority mentions — max
presence at or below `enum_absence_ceiling` (0.20) — is dropped and counted in
`stats.anchor_unmentioned`. An unanswered ask reads as presence 1.0: a failed
guard is never an open door, and it can never cost a record its place.

Measured on "list Linear's current paid plans" (2026-09-20): the pricing page's
completeness scored **0.80**, against **0.04** for a changelog and **0.14** for
the docs billing page — the floor separates the real enumeration from pages that
merely mention plans in passing. The gate dropped six aggregator phantoms (Plus,
Standard, Linear Plus, Linear Standard, Pro Plan) while keeping Basic, Business
and Enterprise.

### Research plan

Once per harvest run, one **planner LLM** call framed as a research librarian
answers "where on the web would a complete list of *X* be published, and why?"
It returns up to eight sources (each with `kind`, `why`, an optional seed URL,
1–3 natural-language queries and a language), one fan-out axis
(`regions` / `ecosystem_terms` / `none`) with its values, and the languages
sources are likely written in. A single Jev batch then scores each source with a
three-level `score` question ("unlikely to list them" / "lists some of them" /
"lists most of them"), so the queue is ordered by expected completeness rather
than by LLM enthusiasm.

Round 1 issues the raw request, the seed query `{anchor_stem} {entity_type}`,
and then the planner's per-source queries in score-descending order. Any
non-null http(s) URLs from the plan are pushed straight into the fetch queue
so the seed pages are read directly rather than rediscovered. Additionally,
every domain-shaped anchor (e.g. `Decidim.org`) contributes `https://{domain}/`
as a depth-1 seed — the anchor's own homepage is fetched and navigated even when
search never surfaces it. The Decidim run (2026-09-18) reached `/partners/` from
the homepage via one `follow_links` pass without any search result pointing there.
Later rounds
fan out with `{anchor} {value} {source_type}` combinations from the chosen
axis. Every query is passed through `ensure_anchor` (which appends the
anchor's stem — `Decidim`, not `Decidim.org` — when it's missing) and then
through the P4 **query gate**: one batched Jev call scores each candidate on
whether it will surface pages that list the entities, and queries below
`query_gate_floor` (0.4 by default) are dropped, but never fewer than two of
the best. Disable the whole thing with `--no-plan` to hand query generation
back to the planner LLM's turn-by-turn output.

**Identity is guaranteed.** Whatever the classifier returns, a harvest's
fields always include one that names each item: `parse_mission` prepends
`name` when every field is an attribute (`Mission::needs_identity_field`).
Without it, a comparison whose fields are its dimensions keys records by the
first dimension's value — one fragment per column, duplicated per page
language (measured 2026-09-21 on "compare 10 voting providers by pricing,
target market, …").

**Facts as fields.** The parse prompt requires a concrete per-entity fact the
request imposes on every item — a year, number, price, version ("founded
after 2020", "fewer than 100 employees", "at least 500 members") — to become
a field (`founded_year`, `employee_count`, `membership`) in addition to a
constraint. Fields are enriched and grounded per entity, so the fact is
checked against the entity's own page, not against the directory that listed
it. Measured 2026-09-21 on "30 Spanish SaaS companies founded after 2020":
with the fact as a constraint only, a startup-list page vouched for nothing
specific and a company founded in 2013 passed the filter.

The constraint verdict then follows the fact. Verdicts are decided at
discovery against the *listing* passage — which is usually silent per item and
can vouch wrongly — so after enrichment fills values, the constraint questions
are re-asked with the enriched facts spelled out in the question itself
(`employee_count=518` against "fewer than 100 employees"). A `contradicts`
excludes the record exactly as in discovery; a `supports` upgrades a stale
`not_addressed`; a weaker new verdict never downgrades an existing support —
facts silent about "Spanish" must not un-verify it. Measured 2026-09-21 on the
same run: enrichment attached `employee_count=518`/`916`/`391` to records
whose "<100 employees" verdict still read `supports` from the directory page,
and every `founded after 2020` sat at `not_addressed` beside a grounded
`founded_year=2024` — 0 of 69 records complete, violators counted as
satisfied. A failed re-check batch changes nothing, which is the safe
direction for an upgrade-only gate.

**Field guard.** After `parse_mission`, `guard_fields` drops any proposed field that
cannot be matched to a keyword in the user's request. The classifier sometimes
proposes fields the request never mentioned — a plain "list of integrator names" came
back with `["name","type","location","website"]`. Each extra field forces unnecessary
enrichment queries and can misclassify the mission's shape. The entity field is never
dropped.

The planner LLM is a separate slot from the writer LLM: use `--planner-model`
(or `WEBSCOUT_PLANNER_MODEL`) to route research-plan, mission-parse, re-aim
and query-writing calls through a reasoning-friendly model while the writer
model handles extraction and prose. When unset the writer model is used for
both.

**Why the planner slot exists — measured.** A "companies integrating
Decidim" run on the pre-plan pipeline produced **130 hits and 0 pages
kept** because no generated query contained the word `Decidim` — the
the prior planner forbade product names and the anchor was never appended.
A plain `decidim.org partners` search returns
[decidim.org/partners/](https://decidim.org/partners/) as its first
result, and Jev triage scores that page **0.68** against **< 0.15**
for the other hits under the same goal. Asked "where would this list
be published", `qwen/qwen3.8-27b` (thinking on) named that URL first
in **8 s**; `google/gemma-4-31b-it` took **45 s** and volunteered
nothing specific. That gap — accuracy and latency — is why the
planner slot is separately routable: extraction is mechanical and
belongs on the writer model, planning is not and does not.

The planner talks to OpenRouter over the `reasoning` extension —
`qwen3.8-27b` reasons by default even when the caller does not want
it to (measured 2026-09-18: with no `reasoning` param, an
`enable_thinking: false` call spent 800+ tokens reasoning and
returned empty content). `src/llm.rs` therefore sends
`reasoning: {"effort": "medium"}` for thinking calls and
`reasoning: {"enabled": false}` for non-thinking calls, only when the
endpoint host is OpenRouter's.

### The judge selects, it does not generate

Borrowed from the sibling jev-search project: Jev picks from options the code
built, never from options it invented. Applied here to query selection.

Round 1's primary query is chosen from a set of up to five candidates that
`build_query_candidates` assembles from the mission's anchors, entity type,
scope and constraint words. One Jev `choice` question (`select_query`) then
picks the best candidate. The worst case is a suboptimal but still code-built,
still anchor-carrying query — Jev cannot produce a candidate that was not
offered to it, so no query comes out anchorless. Measured picks on 2026-09-18:
`Decidim integrators Decidim.org` at probability 0.68 out of five candidates;
`Presupuestos Participativos Ayuntamientos Spain` at 0.75 out of five.

The same pattern applies to enrichment queries. For each (entity, field) pair,
code builds three candidates (primary render, alternate render, bare `{entity}
{field}`), and one batched Jev `choice` per pair (`select_enrich_queries`) picks
which to search. Previously, query phrasing during enrichment was a function of
the attempt count, so repeated failures changed the query in an undirected way.

### Completion accounting

A record is *complete* when every mission field is non-empty **and** every mission
constraint is `supports` (directly or by page-level rescue). The `satisfied_by`
counter and the `outcome` field both use the complete count, so a run that found 80
entities but where only 60 meet both conditions reports `partial` at 60, not 80.

Records that are bound to a `related` organisation (`entity_binding: related`) are
never complete: the page described a different body, even if its contact details look
plausible. Records with unverified constraints (`not_addressed`, `ambiguous`, or
`mixed` with no page-level rescue) are also never complete, even when every field is
populated.

All such records are still included in the output — they are real, verified,
partially-useful results — sorted complete-first, then by grounding. The enrichment
phase closes many field-level gaps automatically; unverified constraints and
wrong-entity bindings are genuine limits of what the web stated.

## What still uses the LLM

| Task | Where |
|---|---|
| Classify the request (mission kind, topic, constraints, entity field, fields) | `parse_mission` |
| Plan search queries (new angles, `site:` exploits) | `fetch_round`, `steer` |
| Generate enrichment query template (once per run) | `enrich_round` |
| Extract non-regex field values from page text during enrichment | `enrich_round` (non-regex path only) |
| Build the research plan (sources + fan-out axis, once per run) | `plan_research` (planner slot) |
| Write answer prose | `verify_round` (answer missions only) |

Everything else — triage, injection screening, entity-presence grounding, field
association, constraint support, link utility, near-collision scoring, steer
decisions (satisfied / exhausted / bottleneck) — is Jev. The LLM sees no URLs or
source text except what Jev has already cleared.

## Verification, concretely

Two numbers do the real work.

**`injection`** — screens every chunk *before* the generative model sees it. Measured
against the live API on a passage that both answers a question and carries an
injection: `supports` 0.98, `injection` 0.99. On relevance alone it would have sailed
straight into the evidence. Quarantined pages are reported with their URL rather than
silently dropped, because which site did it is information you want.

**`grounding`** — every extracted record is checked back against the exact text it
came from: *is every value in this record actually written in that passage?* Below
`--grounding-floor` (default 0.60) the record is discarded and counted. The count
appears in the stats line, so a run that rejected forty rows tells you so.

Grounding checks are batched — one request carries a question per record, and Jev
runs them in parallel — which is what keeps verifying hundreds of rows affordable.

**`constraint_support`** — harvest grounding batches carry two levels of constraint
question. Per-record `c{i}_{j}` questions use the full context-aware gloss; page-level
`p{j}` nouls ask whether the page as a whole lists the target kind of entity. A record
whose per-record score is below `constraint_floor` (default 0.50) is rescued when the
page-level score is ≥ 0.70 and the per-record score is still ≥ 0.25. The
decidim.org/partners/ run (2026-09-18): with bare constraint wording all 12 records
were rejected (0.22–0.26); with the gloss + page-level rescue all 12 were kept (0.91),
and the run returned them in 229 s for $0.023 of Jev.

The per-record constraint question is a **five-way `choice`** (`supports` /
`contradicts` / `not_addressed` / `ambiguous` / `mixed`), replacing the binary noul it
used to be. The distinction matters because the consequences differ in kind, not in
degree:

| Verdict | What it means | Consequence |
|---|---|---|
| `supports` | passage states or clearly implies the entity meets the criterion | record counts toward target |
| `contradicts` | passage states something that cannot be true if the criterion is met | record excluded; **page-level rescue cannot override** |
| `not_addressed` | passage says nothing either way | record kept, unverified; eligible for page-level rescue |
| `ambiguous` | criterion touched on, but meaning or subject is unclear | record kept, unverified; eligible for page-level rescue |
| `mixed` | both supporting and conflicting statements present | record kept, unverified; eligible for page-level rescue |

A failed Jev call reads as `not_addressed`, never `supports` — a failed guard is never
an open door.

Measured against the live API (2026-09-20, decidim.org/partners/):

- Pokecode vs "are integrators of Decidim.org": **`supports` 0.98**. The old binary
  noul scored 0.22–0.26 on the same page.
- A Decidim documentation page that never mentions Pokecode: **`not_addressed` 1.00**.
  The binary noul could not distinguish silence from contradiction at all — one live
  harvest discarded 129 records with no diagnostic to separate "the page refuted it"
  from "the page never mentioned it".
- "El Ayuntamiento de Ejemplo anuncia que tiene la intención de iniciar presupuestos
  participativos el próximo año" vs "has executed participatory budgeting at least
  once": **`not_addressed` 0.65 / `contradicts` 0.33 / `supports` ≈ 0.00** — an
  announced intention can no longer pass as an execution.

An unverified record (`not_addressed`, `ambiguous`, or `mixed` with no page-level
rescue) is reported in the output but does not count toward the target and cannot
produce `outcome: complete`. A contradicted record is excluded outright and counted in
`stats.excluded_contradicted`.

### Entity binding in enrichment

Each enrichment page describes an organisation. That organisation may be the target
entity itself, a related body, or something unrelated — and before this check, all
three were silently treated the same way. A federation's contact details were attached
to a member municipality as if they were the municipality's own.

After a candidate value is chosen on the regex path (or extracted on the non-regex
path), a **four-way `choice`** question (`same` / `related` / `different` /
`unresolved`) asks what relationship the page's organisation has to the target entity:

| Verdict | Meaning | Gate |
|---|---|---|
| `same` | the entity itself, or an official office of it | accept on the usual confidence floor |
| `related` | a parent, federation, subsidiary, supplier, or contractor | reject |
| `different` | unrelated, or a same-named body elsewhere | reject |
| `unresolved` | the passage does not establish which organisation it describes | fall back to a stricter floor |

Acceptance is on the **argmax label only**, never a probability threshold. Measured
(2026-09-20): Terrassa vs the insurer Egarsat came back `related` 0.53 / `different`
0.36 — a margin that any confidence gate would have let through. Forcing an argmax
decision closes that gap entirely. Confirmed correct attributions scored `same` 1.00
(Bilbao's town-hall page, Ajuntament de Sant Adrià de Besòs' OAC page).

Measured misattributions caught in a live harvest (2026-09-20):

| Entity | Page | Verdict | Address the tool previously accepted |
|---|---|---|---|
| Jumilla | Consejo Regulador D.O. Jumilla | `related` 0.73 | `info@vinosdejumilla.org` (the wine regulatory council) |
| Terrassa | Egarsat (insurer) | `related` 0.53 | the insurer's contact address |
| Elche | Diputación de Albacete | `different` 0.70 | the provincial council's address |

Six wrong-entity values were rejected in one harvest run with this gate in place.

**This question is deliberately NOT asked during discovery.** On a listing page that
names forty municipalities the question is ill-posed: the passage describes the list,
not one organisation. Adding it to discovery collapsed a harvest from around 38
entities to 4, with rounds logging pages yielding records but zero gained. The binding
check belongs only in enrichment, where exactly one entity is being looked up on
exactly one page.

### Per-claim checking on the answer path

Harvest discards an ungrounded record. The answer path could not do the same thing —
you cannot discard a third of a paragraph — so for a while it did the weaker thing:
one whole-answer `unsupported` noul, a note appended, and the answer returned as
`complete` anyway. Detection without consequence.

The fix splits the finished answer into claims **in code** (`split_claims`: sentence
boundaries that survive citation markers, decimals, `e.g.`/`U.S.`/`Inc.`, ellipses,
quotations, and fenced code blocks) and asks Jev one noul per claim in a single
request: *"Is the claim in `claims[i]` supported by `evidence`?"*, false branch *"No
evidence item states this; it may be true in the world but it is not in the
evidence."* — because unsupported is not the same as untrue, and a plausible
fabrication scores as supported the moment the question drifts toward world
knowledge.

Measured against the live API (2026-09-19). State: two evidence items (Wikipedia
saying three seasons of *The White Lotus* are out, an HBO release saying a fourth is
renewed) and a four-sentence answer whose last two sentences were invented ("The
fourth season will premiere in March 2027 on HBO Max", "The series has won 15 Emmy
Awards to date"):

| check | result |
|---|---|
| whole-answer `unsupported` | **0.98** — correct, and silent about which sentence |
| per-claim `k0`…`k3` | **0.98 / 0.98 / 0.03 / 0.03** — names them exactly |

Detection was never the problem; localisation and consequence were. Claims scoring
below `claim_floor` (default 0.50 — a *support* probability, not a truth probability)
get ` [unsupported]` appended to the sentence in the rendered answer, are counted in
`stats.unsupported_claims` against `stats.claims_checked`, are listed in a note, and
downgrade `Outcome::Complete` to `Outcome::Partial`. Marking rather than deleting:
removing a sentence breaks the bracketed citations that follow it and leaves prose
that no longer reads.

The check runs on the **final** draft, after the staleness retry has settled, so a
re-drafted answer is the one verified. A claim whose Jev batch failed is returned
unchecked — neither marked nor counted — and a check that fails outright adds a note
saying so, because silence from a guard must never read as a pass.

One verdict upstream of the claim check deserves the same care. The run-level
`answered` noul that grades the answer complete (≥ 0.70) / partial (≥ 0.35) /
empty licenses a **resolved negative**: when `evidence` is the authoritative
place the asked-about fact would appear and the fact is absent there — the
organisation's own contact page lists its addresses and does not list the one
asked about — the answer is "no", stated from the evidence. Without this
license a correct "no" graded as `empty`: a negative answer is still an answer
(measured 2026-09-21, decidim.org's contact page against an `info@` directory
claim: "no mention of info@decidim.org; the email listed to contact Decidim is
hola@decidim.org").

## Output

Results to **stdout**, logs to **stderr**, always. `webscout -vvv -f json ... > out.json`
gives a clean document at any verbosity, which is what makes this safe to drive from
another program.

| `-f` | For |
|---|---|
| `terminal` | reading (default) |
| `json` | one document, every raw judgment including rejected candidates |
| `jsonl` | one record per line, then a `{"type":"summary"}` line |
| `markdown` | a document or a PR |
| `csv` | a spreadsheet; columns follow the order you asked for |

Every harvest record carries:

- `source_url` — the page the entity was first found on
- `grounding` — minimum Jev grounding across all fields
- `constraint_support` — Jev's probability that the record meets the mission
  constraints (1.0 when no constraints were specified); the numeric score kept for
  backward compatibility; the categorical verdict is in `constraint_status`
- `constraint_status` — per-constraint array, parallel to `mission.constraints`,
  each entry one of `supports` / `contradicts` / `not_addressed` / `ambiguous` /
  `mixed`. Empty on records from missions with no constraints. A record with any
  non-`supports` entry is included in the output but is not complete and is not
  counted toward the target.
- `entity_binding` — relationship between the page the value came from and the target
  entity: `same` / `related` / `different` / `unresolved`. Omitted (i.e. `unresolved`)
  when no binding check ran (discovery records, records produced before this field
  existed). A `related` record is never complete.
- `provenance` — a per-field map of `{source_url, grounding}` for each field value
  that was verified on a page different from the entity's primary source (e.g. an
  email retrieved from the municipality's own website while the entity was found on
  a directory page)

In JSON and JSONL these serialize automatically. In CSV, the columns are mission
fields, then `source_url`, `grounding`, `constraint_support`, then one
`{field}_source_url` column for each field whose provenance source differs from
`source_url` in any record. In terminal and markdown output, per-field sources are
shown inline only when they differ from the row's primary source (e.g. `email ←
https://…`).

Verbosity: `-v` info, `-vv` debug, `-vvv` trace, or `--log-level` explicitly.
`--log-json` emits structured log lines for an agent parsing progress.
`WEBSCOUT_LOG` overrides everything, `RUST_LOG`-style.

Exit code is 0 whenever the run completed. Finding nothing is a **result**, not a
failure — it is reported as `"outcome": "empty"` in the payload. Reserve nonzero for
actual breakage.

## For agents

```bash
webscout -f jsonl --log-json -vv "housing co-ops in Catalonia with contact emails"
```

Read `outcome` first: `complete` | `partial` | `truncated` | `empty`. `partial` means
the web ran out before the target did; `truncated` means the round ceiling hit while
results were still arriving, and re-running with `--max-rounds` higher will find
more. Every record carries `source_url`, `grounding`, `constraint_support`, and
per-field `provenance`, so a downstream consumer can apply a stricter bar than the
tool did without re-running anything.

## Performance, and the limits that shape it

Two hard constraints were found by measurement, not by reading docs:

**Jev's per-request limits, measured 2026-09-18: 64k tokens total per request, and
32k tokens for the state plus the longest single question.** The full-request limit
(`MAX_REQUEST_TOKENS = 65,536`) is what actually binds when question count is high;
the state limit (`MAX_STATE_TOKENS = 32,768`) binds when a large page text is passed
in verbatim. Both reject with HTTP 400 `{"detail":{"error_type":"max_tokens_exceeded"}}`.
That error reads like a billing problem and is not one — the dashboard still shows
credit while every oversized call fails. The previous assumption of a single 32k
whole-request limit was wasting half the usable capacity on question-heavy batches.

Converting those token limits into a character budget is where it gets interesting.
English prose runs about 3.7 characters per token, so 3.0 looks conservative — but
the pages this tool exists to read are not prose. A public register of cooperative
names and email addresses measured **2.00 characters per token**: an address like
`bagnusarc2021@gmail.com` shatters into many tokens apiece. At 3.0 those requests
passed the local check and were rejected by the server, and a rejected screening
batch is not retried — it is treated as unsafe and its chunks are **dropped**. One
run quarantined 77 chunks and lost two thirds of its records that way.

The budget is now adaptive: after each successful request, the observed
`body_bytes / input_tokens` ratio is folded into an exponential moving average
(alpha 0.3, clamped 2.0–3.5). Early requests use the conservative 2.0 seed; as the
EMA climbs toward the true ratio for the current content type, batches grow to use
the available capacity. Batch sizes are computed by serializing each item rather than
scaling its length by a guessed factor.

When a batch overruns despite the adaptive budget (the EMA climbed on prose, then a
denser batch hit the server limit), two things happen simultaneously: the EMA resets
to the 2.0 seed so the next estimate is conservative again, and `split_on_oversize`
halves the batch and retries both halves. Halving recurses up to four levels deep; a
batch is only dropped when a single-item sub-batch is still oversized and cannot be
split further. The earlier behaviour — dropping the whole rejected batch — was losing
entire rounds of records; the split-and-retry recovers all but that pathological case.

**Obscura's embedded V8 is not thread-safe.** Any combination of two or more browser
threads with two or more concurrent jobs segfaults. Obscura's own CLI parallelises
with separate worker *processes* for the same reason. So `--browsers` defaults to 1
and page rendering is serialised.

That made rendering the bottleneck — roughly eight seconds a page, one at a time. The
fix is to not render most pages at all:

| | before | after |
|---|---|---|
| page read | ~8,000 ms | 320–1,800 ms |
| Jev requests (same query) | 54 | 22 |
| end to end | ~165 s | ~37 s |

Three changes got there, and each is a different kind of saving:

1. **Plain HTTP first.** Most of the web ships its content in the initial HTML.
   The browser is kept for pages that genuinely need it.
2. **Batched Jev calls.** Questions in one request run in parallel server-side, so
   screening twenty chunks is one request rather than twenty. Verified against the
   API that batched and individual answers agree: a directory page scored `has_items`
   0.97 both ways, an injected passage 0.99 both ways.
3. **Concurrency everywhere HTTP-only.** Searches, screening, extraction and
   verification all overlap; only rendering and round boundaries are serial.

### The fast path can lie, so it is checked

A page can ship plenty of prose and still inject the data you want by script. The
register that yields 37 records in testing has **zero email addresses in its HTML
source** and all of them after JavaScript runs — a naive "does this page have text?"
check passes it happily and silently loses every record.

So the decision is made on evidence rather than on a guess about the markup: if a
page passed triage and then yielded nothing, it is re-read with the browser before
being written off. That costs a render only for pages that already looked worth
reading, and it recovers exactly the cases the fast path gets wrong.

## Automatic steering

On by default; `--no-auto` disables it. After every round the model sees what has
been found, what the last round added, and the current settings, then decides
whether to keep going, whether the request is actually fulfilled, and how to retune
the search.

The fixed rules it replaces are blunt — three barren rounds and a round ceiling suit
a fact lookup and badly underserve "two hundred municipalities with contact
addresses", where the right depth only becomes clear once the first rounds return.

```
auto keep_going=true satisfied=false reason="0 of 200 found; need comprehensive
     registries of Spanish municipalities implementing participatory budgeting"
auto retuned changes=queries_per_round 5 -> 8
auto retuned changes=results_per_query 15 -> 20
```

Every decision is logged with its reason and lands in `notes`, so a run that spent
twelve rounds can be read back and understood. Proposals are **clamped** —
`read_per_query` to 1..=10, `results_per_query` to 4..=30, chunks to 4..=120 — so a
bad decision costs a round rather than your budget. If the model fails to answer, the
fixed rules take over rather than the run ending.

Use `--no-auto` when you want strictly predictable cost.

## Defaults

Tuned so a trivial lookup stays quick and a deep one is not quietly truncated. Both
matter, and they pull in opposite directions, so the defaults differ by *what kind of
question* is being asked rather than compromising between them.

| Default | Value | Why |
|---|---|---|
| `--fetch-concurrency` | your core count, clamped 4..=16 | Rendering is CPU-bound, so the useful ceiling is the hardware. A laptop no longer thrashes and a large box is no longer left idle. |
| stealth | **on** (`--no-stealth` disables) | Costs nothing measurable; better odds on sites that push back, which is the reason to run obscura at all. |
| `read_per_query` | 3 | More independent sources per round, now that fetching is batched and parallel. |
| chunks per page (answer) | 10 | An answer needs two or three good passages; screening the rest confirms what you already have. |
| chunks per page (harvest) | 40 | A harvest's value *is* the long tail. |
| `--enrich-batch` | 25 | Entities enriched per round (halved by `--quick`, doubled by `--thorough`). Keeps one enrich round bounded (25 × ~2 fields × 2 pages ≈ 100 fetches) while visibly closing the field-fill gap. |
| `constraint_floor` | 0.50 | Minimum Jev constraint-support probability to keep a discovered entity. Lenient by design: mission constraints are often paraphrases a source will not echo verbatim, so a middling score is what a qualifying entity typically gets. |
| `follow_floor` | 0.60 | Minimum Jev link-utility probability before following. Higher than grounding because a wrong link is a whole page wasted, and the decision is made without seeing the destination. |
| `max_follow_per_page` | 12 | Hard cap on links followed from one listing page. A register with 400 entity links would blow through the round budget in one pass; the top 12 are almost always the ones worth reading. |
| `enrich_read` | 2 | Pages fetched per (entity, field) search. The official homepage almost always ranks first and a second page catches the one time it does not. |
| `select_confidence` | 0.50 | Minimum choice-confidence for accepting a regex candidate as the entity's field value. Matches a calibrated "more likely than not" and lets a page with two plausible emails still succeed when Jev picks the general one. |
| `enrich_results_per_query` | 5 | Search results per enrichment query. Smaller than the general `results_per_query` because enrichment already knows which entity it wants. |
| `enum_*` gates | 0.80 / 0.20 / 3 / 3 | Anchor-site enumeration corroboration (see its section): completeness floor for an anchor page to count as an authority, presence ceiling below which an unmentioned record is dropped, minimum entities named to arm the gate, maximum authority pages kept.

### Harvest flags

| Flag | Default | Effect |
|---|---|---|
| `--no-follow` | off | Skip the link-following phase; use only search results. Useful for narrow, cost-bounded runs. |
| `--no-enrich` | off | Skip enrichment; output only what listing pages directly stated. |
| `--no-plan` | off | Skip the research plan; hand all query generation to the LLM planner. |
| `--planner-model MODEL` | writer model | Use a separate model for research plan, parse, re-aim and query writing. Same endpoint / key as the writer model. |
| `--enrich-batch N` | 25 | Override the profile default for entities enriched per round. |
| `--thinking-control MODE` | `auto` | How to tell the writer endpoint to stop reasoning. `auto` uses the OpenRouter form for `openrouter.ai`, nothing elsewhere. `vllm` sends `chat_template_kwargs.enable_thinking` — measured 2026-09-18: switching this off cut an extraction call from **58.3 s / 7,713 tokens** (23,631 chars of reasoning) to **1.4 s / 48 tokens** with identical records. `effort` sends `reasoning_effort: medium/none`. `openrouter` sends `reasoning.effort/enabled`. `off` sends nothing. Also `$WEBSCOUT_THINKING_CONTROL`. |
| `--planner-thinking-control MODE` | writer's value | Same options as `--thinking-control`, for the planner transport. Also `$WEBSCOUT_PLANNER_THINKING_CONTROL`. |
| `--search-engines auto\|ddg\|jina` | `auto` | Which search engines to run. `auto` uses every available lane (DuckDuckGo when obscura is configured, Jina when a key is present). Pin to one engine to compare them on the same mission or to skip a dependency. |
| `--no-search-cache` | off | Do not read or write the on-disk search cache. Useful when the cached results may be stale for time-sensitive queries. |
| `--search-cache-ttl SECONDS` | 21600 | How long a cached search response stays usable. Defaults to 6 hours. Also `$WEBSCOUT_SEARCH_CACHE_TTL`. |
| `--url URL` (repeatable) | off | Read this URL directly instead of searching. On an answer mission the given pages are the whole evidence base (still injection-screened — a page you name gets no exemption); on a harvest they are depth-1 seeds beside the search. |

That last split is the one that mattered most. The same public register yielded **18
records when the cap truncated its table and 37 when it did not** — every field
populated, grounding mean 0.91 against 0.93 before, so the extra rows are as solid as
the originals. The cap was the limit, not the web.

A single-fact lookup is unaffected: it still skips planning, reads three pages and
answers in ~15s, because effort is matched to the question rather than to a global
setting.

## How fast can it go

No ceiling is imposed anywhere in this code — `--fetch-concurrency`, `--concurrency`
and `--max-questions-per-request` all accept whatever you give them. That is
deliberate, because the interesting limits turned out not to be ours. Measured on a
12-core box:

**Rendering is CPU-bound.** 32 Wikipedia pages through `obscura scrape`: 74s at
concurrency 4, 32s at 8, 37s at 16, 29s at 32. Past roughly your core count the curve
is flat and noisy. Launching 32 browsers works fine — 33 MB each, so ~1 GB — it just
does not buy much on pages this heavy. Lighter pages or a bigger box will go further.

**Verification is request-bound, not parallelism-bound.** Jev's latency is flat at
about 0.9s no matter how many calls are in flight, but throughput plateaus near **4
requests/second**, well under the documented 1,200/minute. Running 32 in parallel
takes 8.5s for 32 calls; running 8 takes 2.3s for 8.

**But questions are nearly free.** 192 questions in one request answered in 0.98s,
against 0.80s for 16. That asymmetry is the whole optimisation: pack hard, send
seldom. Batching triage from one request per search hit into one per *batch* cut a
run from 24 Jev requests to 11 and from 72s to 54s.

So raising the knobs does not help, and can hurt — the same query at
`--fetch-concurrency 32 --concurrency 32` took **82s** against **51s** at the
defaults, with identical work done.

### Where a run's time actually goes

A single-round query lands around 50 seconds, split roughly:

| Stage | Time | Bound by |
|---|---|---|
| Search batch | ~8s | one obscura call, parallel inside |
| Fetch batch | ~8s | CPU cores |
| Jev stages | ~15s | ~0.9s each, in sequence |
| Generative calls | ~18s | reasoning: synthesis is 10.1s with thinking, 2.1s without |

The critical path is **sequential stages**, not insufficient parallelism. You cannot
plan before the mission is parsed, search before planning, or judge before fetching.
More workers do not move that floor.

### Effort matched to the question

"Who is the CEO of X" and "compare the AI Act and GDPR" are not the same job, and
treating them identically made the trivial one embarrassing: five planned queries,
ten pages fetched, ninety-one chunks screened and a reasoning-heavy synthesis — to
establish something the first search snippet already said. Thirty seconds for a
one-line lookup.

The mission classifier now also decides whether a request is a single-fact lookup,
and three things change when it is:

- **No query planning.** The user's own phrasing is what a person would type and is
  already the best first search. Asking a reasoning model to improve it cost ~6s and
  improved nothing.
- **Three pages, not ten.** A widely agreed fact needs two sources that agree. The
  rest cost the screening of every chunk on every one of them.
- **No chain of thought when writing.** Synthesis is 10.1s with reasoning and 2.1s
  without, for the same one-line answer. Weighing sources earns reasoning; naming an
  officeholder does not.

| "who is the current CEO of Vodafone" | before | after |
|---|---|---|
| wall clock | 30.8 s | **12.5 s** |
| queries / pages / chunks | 5 / 10 / 91 | 2 / 3 / 25 |
| Jev requests | 15 | 8 |
| cost | $0.0075 | $0.0022 |

A harvest is never treated as simple, and complex questions are unaffected: the AI
Act comparison above still classifies as `simple=false` and runs 5 rounds over 32
pages for a 1,231-word answer. The flag is logged with the mission, so you can always
see which path a run took.

### Pipelined rounds

What does move it is overlapping the phases that need different resources. A round
splits cleanly in two: `fetch_round` talks to the web (plan, search, triage, fetch),
`verify_round` talks to the models (screen, extract, ground). Neither needs the
other's capacity, so round N+1's fetching runs *while* round N is still being
verified:

```
searching round=1
searching round=2          <- starts before round 1 finishes
round complete round=1
searching round=3
round complete round=2
```

The next round therefore plans against knowledge one round stale. That is the
deliberate trade: a slightly less informed round that runs for free beats a better
informed one the pipeline had to wait for. Queries already issued and URLs already
fetched *are* shared before the handoff, so it never re-searches or re-reads what the
current round just did — only the productive-domain feedback lags.

Two things override the prefetch: if the reviewer names a specific gap, its targeted
searches replace the speculative round, and a satisfied target stops the pipeline
rather than fetching a round nobody will read.

### Measured harvest results

Two reference queries, 2026-09-18. The speculative search stage ran concurrently
with `plan_research` and added 3.2 s to the wall clock while the plan was being
built.

| Query | Before | After |
|---|---|---|
| Decidim integrators (worldwide) | 12 records, partial, 3 rounds, 229 s | 19 verified integrators, **complete**, 1 round, 216 s, $0.019 Jev — new sources reached including comptoir-du-libre.org and decidim.org/installations |
| Ayuntamientos Presupuestos Participativos (Spain) | 41 entities, 20 complete, 4 rounds, 499 s | 38 entities, 17 complete, 2 rounds, 296 s — roughly the same yield in half the rounds |

## What a run cost

Any verbosity (`-v` and up) prints a summary to stderr when the run ends. It goes to
stderr deliberately: the stats footer only appears in the terminal and markdown
renderings, so anyone using `-f json` never saw it.

```
  ── run summary ──────────────────────────────────────────────
  Jev       7 requests      22,189 input tokens   $0.0009
  LLM       2 requests       2,887 prompt + 176 completion
  Time    10.1s wall   (browser 3.9s · jev 3.5s · llm 3.8s)
  Work    1 round · 2 queries · 3 pages · 10 chunks
  Guards  2 passage(s) quarantined · 0 ungrounded record(s) rejected
          · 3 contradicted record(s) excluded · 2 wrong-entity value(s) rejected
          · 5 constraint(s) unverified
  Harvest 47 entities discovered · 38 enriched · 12 links followed
  ─────────────────────────────────────────────────────────────
```

The browser/jev/llm split is derived from the stage timings, so it tells you which
subsystem a slow run is waiting on — and since the three overlap, they sum to more
than the wall clock. `Guards` only appears when something was actually caught, and each
sub-counter within it is printed only when it fired:

- **`contradicted record(s) excluded`** — records whose source passage explicitly
  contradicted a mission constraint (`stats.excluded_contradicted`). Separate from
  `ungrounded record(s) rejected`: those were not in the text; these were in the text
  and the text said no.
- **`wrong-entity value(s) rejected`** — values discarded because the page described a
  related or different organisation rather than the target entity
  (`stats.wrong_entity_rejected`).
- **`constraint(s) unverified`** — (record, constraint) pairs kept in the output but
  not counted toward the target because the passage said nothing, was ambiguous, or was
  mixed, and no page-level rescue applied (`stats.constraints_unverified`).

`Harvest` only appears on harvest missions and only when any of the three counters is
non-zero.

## Profiling

`--profile` prints a per-stage table to stderr when the run ends, so it never
contaminates the payload. The same numbers are always in the JSON under
`stats.stage_ms`, whether or not the flag is set.

```
webscout --profile "who is the current CEO of Vodafone"
```

Stage times sum to **more** than the wall clock, on purpose: stages overlap, and the
excess is exactly what the concurrency and round pipelining are buying. The share
column is measured against wall clock, not against the sum — so one stage at 57% is
genuinely holding the run up, while several at 20% merely ran at the same time.

## Reacting to a failing Jev

- **Transient** (429, 5xx, 529 overloaded, `model_unavailable`) — retried with
  exponential backoff plus jitter, so a burst of concurrent calls does not retry in
  lockstep against a struggling backend.
- **Verdicts on the request** (4xx such as `max_tokens_exceeded`) — not retried at
  all. Resending identical bytes cannot change the answer, and retrying turned an
  instant failure into a slow one with the cause buried under backoff.
- **Sustained outage** — a circuit breaker opens after 10 consecutive failures and
  fails fast with a message naming the outage. Without it, an outage costs every
  call in the pipeline a full retry ladder and a run grinds for twenty minutes
  before saying anything useful.
- **A failed screen is never an open door.** If Jev cannot judge a chunk, that chunk
  is treated as unsafe and withheld from the generative model rather than passed
  through unchecked. The same applies to grounding: an unverifiable record is
  discarded, not trusted.

## Quality control

Jev checks each item; a reader has to check the *set*. Jev can say a record is
grounded and that evidence answers a question, but not that every row came from one
country when the request implied several, or that a column you asked for is empty
everywhere.

So before a harvest gives up, the generative model reads a summary — counts, how many
of each field are populated, which domains contributed, sample rows — and says
whether it genuinely satisfies the request. When it finds a specific gap it also
proposes the searches that would close it, and the loop takes those instead of
stopping. A run that would have ended at "no new records" gets one more pass aimed at
what is actually missing.

Answers get the equivalent check: the finished prose is compared against the evidence
it was built from, and any claim the evidence does not support is flagged in the
report.

## Running the web UI

The repository ships a Docker Compose setup that runs a self-contained API
container and an nginx container that serves the static UI and proxies all
`/api/` requests to it.  The browser never needs to contact the API directly,
so no CORS configuration is required.

### Quick start

```bash
# 1. Copy the example env file and fill in the two required keys.
cp .env.example .env
$EDITOR .env          # set TYPESAFE_API_KEY and WEBSCOUT_LLM_API_KEY at minimum

# 2. Build images and start.
docker compose up --build

# 3. Open http://localhost:3000
```

The API listens on port 8080 **inside its container only** — it is deliberately
not published on the host, because it holds the credentials and accepts queries
that could be expensive.  Only the UI is reachable from the host, on port 3000.

### Credentials note

The repo's gitignored `env` file uses `export KEY=value` syntax and is for
shell use (`source env`).  Docker Compose's `env_file` directive requires plain
`KEY=value` form with no `export` prefix.  `.env.example` (and therefore `.env`)
uses that plain form specifically so compose can read it.

### Obscura and search lanes

The API container does **not** ship obscura.  Without it only the Jina search
lane is active, which requires `JINA_API_KEY` to be set.  The DuckDuckGo lane
uses obscura and is therefore unavailable unless a host obscura is mounted.

To supply a host obscura, mount the directory containing the `obscura` and
`obscura-worker` binaries onto `/opt/obscura` inside the container.  The image
adds `/opt/obscura` to `PATH` automatically.  Uncomment the example in
`docker-compose.yml`:

```yaml
volumes:
  - /usr/local/bin:/opt/obscura:ro
```

### Search cache

A named Docker volume (`search_cache`) persists the 6-hour DuckDuckGo/Jina
result cache across container restarts, so repeated or similar queries do not
hammer the search engines.

### The event stream

`POST /api/search` answers with `application/x-ndjson`: one JSON object per
line, flushed as it happens.  `type` says what each line is.

| `type` | When | Carries |
|---|---|---|
| `accepted` | once, first | `run_id`, `query`, `started_at` |
| `progress` | round starts, stage transitions | `stage`, `round`, `message`, `counts` |
| `usage` | about once a second, and once more before the ending | tokens and cost so far |
| `stats` | once, at the end | the five coarse counters |
| `result` / `error` | exactly one, last | the payload, or why there isn't one |

The `usage` event is what the UI's token meter is made of:

```json
{"type":"usage","elapsed_ms":41000,
 "jev":{"requests":12,"input_tokens":88213,"cost_usd":0.0037},
 "llm":{"requests":3,"prompt_tokens":14788,"completion_tokens":1190,"cost_usd":0.0021},
 "planner":{"requests":1,"prompt_tokens":2288,"completion_tokens":4177,
            "reasoning_tokens":3900,"cost_usd":0.0104}}
```

Three things about it are load-bearing:

- **`cost_usd` on `llm` and `planner` is absent, not zero, when the endpoint
  does not report one.**  OpenRouter returns a real `usage.cost` on every
  completion (verified 2026-09-21: `2.18e-06` on a 17-token call, no special
  request parameter); a plain vLLM deployment returns nothing.  No price table
  is inferred from a model name — a guessed invoice is worse than an honest
  gap.  Jev's `cost_usd` is always present: it is input tokens times a measured
  constant.
- **`reasoning_tokens` appears only when some were spent.**  It is the slice of
  `completion_tokens` that went to chain of thought, and a large share of it on
  extraction calls is the signature of the wrong `--thinking-control` for that
  endpoint — measured once at 7,713 completion tokens of pure thought with
  empty `content`.
- **The last `usage` event is built from the finished report**, not from another
  sample, so the numbers a client is left showing cannot disagree with the
  `result` event's `stats` by a request that landed in between.

Sampling costs nothing and stops when you do: the counters are atomics on the
Jev and LLM clients, and the timer lives inside the response stream rather than
in a spawned task, so closing the connection drops the run and the sampling
with it.

## Building

```bash
cargo build --release
cargo test
```

Requires the `obscura` and `obscura-worker` binaries on PATH — webscout drives them
as subprocesses. Get them from
[obscura's releases](https://github.com/h4ckf0r0day/obscura/releases); the archives
ship both together, and they must stay in the same directory. Point elsewhere with
`--obscura-bin`. Their absence is reported at startup rather than as a confusing
fetch failure later.

### Why obscura is a subprocess, not a library

Obscura *can* be linked in — the crate exists and works. It was, and then it was
measured, and the numbers decided it:

| Fetching 8 pages | Time |
|---|---|
| Embedded, serialised (one thread) | 151.6 s |
| One `obscura fetch` process per URL | 45.0 s |
| **One `obscura scrape --concurrency 8`** | **6.2 s** |

Embedded, obscura's V8 is not safe to drive from several threads in one process:
any combination of two or more browsers with two or more concurrent jobs segfaults,
so every render serialises. `obscura scrape` runs persistent `obscura-worker`
*processes*, each with its own address space and its own V8, which sidesteps the
constraint entirely — and is exactly why obscura's own CLI is built that way.

Spawning one `fetch` per URL is not the same thing: each pays full V8 startup, which
is why it only reaches 45 s. The win comes specifically from batching into one
`scrape` call, so this tool batches a whole round's searches into one invocation and
a whole round's pages into another.

Dropping the embedded engine also took the binary from **62 MB to 5.8 MB**.

### Plain HTTP first

Rendering is still the expensive step, so it is avoided where it is not needed. Most
of the web — references, documentation, registers, news — ships its content in the
initial HTML, and a plain HTTP fetch returns it in well under a second.

```
page batch read  requested=6 http=6 rendered=0  ms=1253
```

The catch is that HTML can look complete and still be missing the data. The register
this tool was built against has **zero email addresses in its HTML source** and all
37 after JavaScript runs; a naive "does this page have text?" check passes it and
silently loses every record.

So HTTP is only a pre-filter, and the decision is made on evidence rather than on a
guess about the markup: anything thin goes straight to the browser, and anything that
passes but then yields nothing is re-rendered in a single batch and reprocessed.

```
page batch read  requested=6 http=2 rendered=4  ms=11656
re-reading empty pages with the browser  count=2
records extracted  url=.../list-of-cda-accredited-cooperatives/  found=37
```

Cheap when it works, self-correcting when it does not.

### Static linking

Now that V8 is no longer linked in, a fully static musl build is within reach — it
was impossible before, since `rusty_v8` ships a glibc-built static library.

What remains is ordinary: `ring` (the TLS crypto provider) contains assembly and
needs a C cross-compiler. Install one and it should build:

```bash
sudo apt install musl-tools          # provides musl-gcc
CC_x86_64_unknown_linux_musl=musl-gcc \
  cargo build --release --target x86_64-unknown-linux-musl
```

The default provider (`aws-lc-rs`) is C as well, so the manifest selects `ring`
through `rustls-no-provider`, which keeps everything else pure Rust.

## Credentials

**Nothing is compiled in.** Every key comes from a flag or an environment variable,
with the flag winning, so a one-off run can override your shell without editing it. A
key baked into a binary shows up in `strings`, cannot be rotated without a rebuild,
and makes the artifact itself the secret.

| Flag | Environment | Required |
|---|---|---|
| `--typesafe-key` | `TYPESAFE_API_KEY` | yes |
| `--typesafe-endpoint` | `TYPESAFE_ENDPOINT` | no |
| `--llm-key` | `WEBSCOUT_LLM_API_KEY` | yes |
| `--llm-endpoint` | `WEBSCOUT_LLM_ENDPOINT` | no |
| `--llm-model` | `WEBSCOUT_LLM_MODEL` | no |
| `--planner-model` | `WEBSCOUT_PLANNER_MODEL` | no — routes plan / parse / re-aim / query writing to a separate model |
| `--planner-endpoint` | `WEBSCOUT_PLANNER_ENDPOINT` | no — separate host for the planner; defaults to the writer's endpoint |
| `--planner-key` | `WEBSCOUT_PLANNER_KEY` | no — API key for the planner endpoint; defaults to the writer's key |
| `--jina-key` | `JINA_API_KEY` | no — selects the Jina backend |

Missing required keys fail before any work starts, naming both ways to supply them.
Non-HTTPS endpoints are rejected unless they are localhost, since credentials ride on
every request.

**Any OpenAI-compatible server works.** `--llm-endpoint` and `--planner-endpoint`
require the full standard URL ending in `/chat/completions`, e.g.
`https://openrouter.ai/api/v1/chat/completions` (the default). A wrong URL — a base
URL, a path ending in `/v1`, a trailing `/chat/` — is rejected at startup with the
corrected form suggested in the error message, so there is no silent rewriting and no
trial-and-error.

Some self-hosted servers always reason regardless of what the caller asks. One self-hosted vLLM
deployment (measured 2026-09-18) is one: it ignores
`reasoning: {"enabled": false}` (the OpenRouter form) and returns its chain of thought
in a field named `reasoning_content` rather than `reasoning`. webscout accepts both
field names and reports the correct byte count in the starvation diagnostic. When the
reasoning budget fills `max_tokens` before any `content` appears
(`finish_reason: "length"`), the max-tokens escalation kicks in automatically — it is
not gated on OpenRouter or on any other specific host.

To switch reasoning off entirely on vLLM-backed servers, pass
`--thinking-control vllm`. This sends `chat_template_kwargs.enable_thinking: false` on
non-thinking calls. Measured against the same vLLM deployment: an extraction call that
took **58.3 s and 7,713 completion tokens** with reasoning on dropped to **1.4 s and
48 completion tokens** with it off, producing identical records. Without this flag the
starvation escalation still saves the run by raising `max_tokens`, but at a large cost
in tokens and latency.

## PDFs

A URL whose path ends `.pdf` never goes through the browser: Chromium's PDF viewer
exposes no document text to a page read, and obscura sometimes hands back the raw
byte stream instead, which screens as junk (measured on the CDTI NEOTEC 2024
resolution: 627,825 characters of PDF bytes, ten rounds, zero records). PDF URLs are
fetched with `obscura fetch --dump original` — the plain HTTP body, so the same
stealth flags and session cookies apply — checked for the `%PDF` magic, and parsed
with `pdf-extract` into page text that flows through the same screening and
grounding as any HTML. Measured on the same resolution after the fix: all 62
beneficiaries extracted from the official PDF in one 83-second run, grounding
0.86–0.99.

## Two fetch backends

Fetching runs through obscura by default. **Supplying a Jina key switches to Jina's
hosted search and reader** — the key's presence is the whole switch, so there is no
second flag that could contradict it.

|  | obscura (default) | Jina (`--jina-key`) |
|---|---|---|
| Runs | locally, as a subprocess | hosted |
| Rendering cost | your CPU — the largest slice of most runs | theirs |
| Needs installing | `obscura` + `obscura-worker` on PATH | nothing |
| Cost | free, unmetered | metered per call |
| JS-rendered pages | yes | yes |
| Bot-protected sites | reads them, with `--stealth` | **often blocked** |

### Measured, same queries, both backends

`--quick --max-rounds 1`, one run each:

| Case | Backend | Wall | Browser | Pages | Outcome | Result |
|---|---|---|---|---|---|---|
| Simple fact | obscura | 22.6s | 3.9s | 3 | complete | answer |
| Simple fact | **jina** | **8.8s** | **1.8s** | 3 | complete | answer |
| Bot-protected site | **obscura** | 12.3s | 6.4s | 3 | **complete** | answer |
| Bot-protected site | jina | 10.2s | 7.2s | **0** | **empty** | **nothing** |
| JS table harvest | obscura | 38.8s | 10.3s | 5 | truncated | 18 records |
| JS table harvest | **jina** | **28.9s** | **4.6s** | 2 | truncated | **36 records** |

Three things worth reading carefully.

**Jina is roughly twice as fast at fetching** when it works — 1.8s against 3.9s, 4.6s
against 10.3s. Rendering runs on their hardware instead of competing for your cores.

**Jina returns nothing at all on bot-protected sites.** Not slower, not partial:
0 pages, `empty`, no answer. Obscura read the same company's own leadership page
without trouble. If your sources push back, this is disqualifying rather than a
tuning problem.

**Jina found twice the records from fewer pages** on the JS-rendered register — 36
from 2 pages against 18 from 5. Its markdown is far more compact (8 chunks against
30), so much more of the table survives the per-page chunk cap. But grounding
confidence was *lower*: mean 0.80, min 0.65, against obscura's 0.93 and 0.89. Jina's
markdown conversion reformats table cells, which makes Jev's "is this value written
here, character for character" check harder to satisfy. More rows, each slightly less
certain.

Caveat: these are single runs against live services, and variance is real — the LLM
column swung 16.0s to 3.5s between two runs of the *same* query. Treat the browser
column and the outcomes as signal, the totals as indicative.

Both backends sit behind the same plain-HTTP pre-filter, so a page that needs no
JavaScript costs neither of them anything.

### Search lanes

Fetching and searching are separate concerns. The fetch backend (obscura or Jina)
reads pages; search lanes query search engines. Several lanes run concurrently —
currently DuckDuckGo and Jina — and their results are merged by URL before any
Jev triage runs. One slow or failing lane never discards another's results: each
lane has a 20 s deadline of its own, inside whatever overall budget the run imposes.

Measured 2026-09-18: `html.duckduckgo.com` returns 10 clean results per query.
`www.bing.com` returns 10 nodes whose content is unrelated to the query (French
microphone tutorials for a Decidim query) behind `bing.com/ck/a?` redirects.
Brave, Mojeek, Startpage and Ecosia return interstitials or near-empty bodies
(438–4,086 characters, zero result selectors). So only DuckDuckGo and Jina are
lanes today. Adding a third engine when one becomes usable is a `LaneEngine`
variant, not a refactor.

Previously a Jina key silently disabled DuckDuckGo: one enum chose both the
fetcher and the searcher. Fetching and searching are now separate, so supplying
a Jina key gives you Jina's reader *and* both engines' search results.

`--search-engines auto|ddg|jina` — `auto` (default) runs every lane that is
available; pin to one engine to compare them on the same mission.

### Merging and agreement

`canonical_url` gives each page a stable identity regardless of how different
engines or sites express its URL: the scheme is dropped, a leading `www.` is
stripped, trailing slashes are removed, and tracking parameters (`utm_*`,
`fbclid`, `s`, `t`, and a short blocklist) are discarded. Two hits that differ
only in those ways collapse to one.

`title_key` catches the same story syndicated under different URLs: the trailing
"— Reuters" or "| El País" segment is stripped, then the first eight alphanumeric
words form the headline identity. A hit whose title key was already claimed by a
different URL is dropped as a syndicated copy — but only when the key is
headline-length (at least four words). A two- or three-word title is a
programme or category name ("NEOTEC 2024"), and distinct government pages about
the same programme legitimately share one; measured on the CDTI NEOTEC harvest
(2026-09-21), the short-title collision dropped the ministry's resolution page
and two other distinct NEOTEC pages as "syndicated copies" — 232 drops in a
single run, including the very list the mission was asking for.

When two lanes return the same URL their engine sets are unioned into
`Hit.engines`. The number of agreeing engines is a tiebreak in triage ranking:
15% per extra engine beyond the first, capped at a 30% boost total. That boost
is applied *after* the keep/drop decision, so it reorders results but never
reclassifies them — a result Jev wants to drop cannot be rescued by engine
count — and it costs no Jev tokens.

### Search cache

Successful, non-empty search responses are cached on disk, keyed by lane id,
query string and result limit. The default TTL is 6 hours — long enough that a
re-run or a second mission on the same subject reuses what is already on disk,
short enough that a query like "what changed this morning" is still answerable.

Empty and failed responses are never cached: an empty result is what a transient
block looks like, and caching it would hold the failure for the whole TTL.

Cache files live under `$XDG_CACHE_HOME/webscout/search` (falling back to
`$HOME/.cache/webscout/search` and then to the system temp directory, so it
works in containers too).

Measured 2026-09-18: an identical repeat run reported 8 hits, 0 misses, 16,487
bytes served from cache — the search phase ran in negligible time. The end-of-run
summary line shows hit and miss counts when either is non-zero.

`--no-search-cache` disables the cache entirely. `--search-cache-ttl SECONDS`
overrides the TTL; `$WEBSCOUT_SEARCH_CACHE_TTL` is the equivalent environment
variable.

## Layout

```
src/config.rs      endpoints, keys, and every tunable threshold in one place
src/types.rs       Mission, Record, Outcome, Stats; normalize_entity
src/llm.rs         qwen client: reasoning toggle, strict json_schema decoding
src/typesafe.rs    Jev client: noul / choice / score, batched questions, adaptive EMA
src/browser.rs     obscura subprocess batches, plain-HTTP pre-filter, link extraction, search lanes
src/search_cache.rs  on-disk cache for search-lane responses; SHA-256 keyed, 6h TTL
src/candidates.rs  regex candidate extraction and normalization for email, URL, phone
src/scout.rs       the loop: plan, triage, screen, extract, verify, enrich, replan
src/output.rs      five renderers; per-field provenance columns in CSV
src/main.rs        CLI, logging to stderr, wiring
```

## Known limits

- **Search is a scrape** of DuckDuckGo's HTML endpoint. It is server-rendered and
  stable, but the markup can change without notice; empty results with a working
  network is the tell. The selector lives in `SEARCH_JS` in `src/browser.rs`.
- **Field names are chosen by the classifier, not by you.** Asking for "email
  addresses" may yield a field called `email` or `email_address`. Read
  `mission.fields` from the output and key off that — hardcoding a guess is how you
  end up reporting empty columns that are in fact populated.
- **Incomplete records are kept but sorted after complete ones.** A record is
  *complete* only when every mission field is non-empty. Incomplete records (e.g.
  name without email) are included in the output and in `source_url` / `grounding`
  provenance, but `outcome` and the satisfied-count use only complete records. The
  enrichment phase closes many of these gaps automatically; what remains after
  enrichment is a genuine limit of what the web stated publicly.
- **Thresholds are untuned.** They are applied to calibrated probabilities, but where
  each line sits is a product decision that deserves evaluation against your own
  queries, not a constant to trust.
- **Harvest quality tracks the web's.** If contact details are not on public pages,
  no amount of looping invents them — and by design, nothing invents them.
