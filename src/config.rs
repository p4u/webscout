//! Settings: credentials, endpoints, and the knobs that govern how hard the scout
//! digs.
//!
//! Nothing here is compiled in. Every credential arrives from a flag or an
//! environment variable, because a key baked into a binary is readable with
//! `strings`, cannot be rotated without a rebuild, and makes the artifact itself the
//! secret. Flags win over the environment, so a one-off run can override a shell
//! profile without editing it.

use anyhow::{Result, bail};
use std::time::Duration;

pub const DEFAULT_TYPESAFE_ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
pub const TYPESAFE_MODEL: &str = "jev-latest";
pub const DEFAULT_LLM_ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
pub const DEFAULT_LLM_MODEL: &str = "google/gemma-4-31b-it";

/// Where each credential and endpoint may come from, flags first.
#[derive(Default)]
pub struct Sources<'a> {
    pub typesafe_key: Option<&'a str>,
    pub typesafe_endpoint: Option<&'a str>,
    pub llm_key: Option<&'a str>,
    pub llm_endpoint: Option<&'a str>,
    pub llm_model: Option<&'a str>,
    /// Model used for planning-shaped calls (mission parse, research plan,
    /// query writing, re-aim, review). Defaults to the writer model when
    /// unset. Kept separate because planning benefits from reasoning while
    /// extraction wants a mechanical, cheap model — see P7b in the README.
    pub planner_model: Option<&'a str>,
    /// Endpoint for planner calls. Defaults to `llm_endpoint` when unset,
    /// so behaviour is identical to before when the flag is absent.
    pub planner_endpoint: Option<&'a str>,
    /// API key for the planner endpoint. Defaults to `llm_key` when unset.
    pub planner_key: Option<&'a str>,
    pub jina_key: Option<&'a str>,
}

/// Resolved credentials and endpoints for one run.
#[derive(Debug, Clone)]
pub struct Credentials {
    pub typesafe_key: String,
    pub typesafe_endpoint: String,
    pub llm_key: String,
    pub llm_endpoint: String,
    pub llm_model: String,
    /// Model used for planning; defaults to `llm_model` when the caller
    /// supplied nothing.
    pub planner_model: String,
    /// Endpoint for planner calls; defaults to `llm_endpoint`.
    pub planner_endpoint: String,
    /// API key for the planner endpoint; defaults to `llm_key`.
    pub planner_key: String,
    /// Present only when supplied. Its presence is what selects the Jina backend.
    pub jina_key: Option<String>,
}

/// A configured value without surrounding whitespace or one matching pair of
/// quotes. `.env` files quote values, and Compose strips the quotes, but
/// `docker run --env-file` and the Railway / DigitalOcean variable screens pass
/// them through literally: a quoted endpoint arrived as `'https://…'` and failed
/// the https check at startup (measured 2026-09-25).
pub(crate) fn unquote(v: &str) -> &str {
    let t = v.trim();
    for q in ['\'', '"'] {
        if t.len() >= 2 && t.starts_with(q) && t.ends_with(q) {
            return t[1..t.len() - 1].trim();
        }
    }
    t
}

fn pick(flag: Option<&str>, env: &str) -> Option<String> {
    flag.map(|v| unquote(v).to_string())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            std::env::var(env)
                .ok()
                .map(|v| unquote(&v).to_string())
                .filter(|v| !v.is_empty())
        })
}

impl Credentials {
    /// Resolve every credential, preferring a flag over the environment.
    ///
    /// Missing required keys fail here, before any work starts, with a message that
    /// names both ways of supplying them. Discovering a missing key on the first API
    /// call means the failure arrives after a search has already run.
    pub fn resolve(s: &Sources<'_>) -> Result<Self> {
        let typesafe_key = pick(s.typesafe_key, "TYPESAFE_API_KEY").ok_or_else(|| {
            anyhow::anyhow!(
                "no TypeSafe API key. Pass --typesafe-key or set TYPESAFE_API_KEY. \
                 Keys are never compiled in; get one at \
                 https://console.typesafe.ai/settings/keys"
            )
        })?;

        let llm_key = pick(s.llm_key, "WEBSCOUT_LLM_API_KEY").ok_or_else(|| {
            anyhow::anyhow!(
                "no generative model API key. Pass --llm-key or set WEBSCOUT_LLM_API_KEY. \
                 The default endpoint is OpenRouter; --llm-endpoint and --llm-model point \
                 it elsewhere."
            )
        })?;

        let llm_model = pick(s.llm_model, "WEBSCOUT_LLM_MODEL")
            .unwrap_or_else(|| DEFAULT_LLM_MODEL.to_string());
        let planner_model =
            pick(s.planner_model, "WEBSCOUT_PLANNER_MODEL").unwrap_or_else(|| llm_model.clone());

        let llm_endpoint = pick(s.llm_endpoint, "WEBSCOUT_LLM_ENDPOINT")
            .unwrap_or_else(|| DEFAULT_LLM_ENDPOINT.to_string());
        validate_chat_endpoint("--llm-endpoint", &llm_endpoint)?;

        let planner_key =
            pick(s.planner_key, "WEBSCOUT_PLANNER_KEY").unwrap_or_else(|| llm_key.clone());

        let planner_endpoint = pick(s.planner_endpoint, "WEBSCOUT_PLANNER_ENDPOINT")
            .unwrap_or_else(|| llm_endpoint.clone());
        validate_chat_endpoint("--planner-endpoint", &planner_endpoint)?;

        let creds = Self {
            typesafe_key,
            typesafe_endpoint: pick(s.typesafe_endpoint, "TYPESAFE_ENDPOINT")
                .unwrap_or_else(|| DEFAULT_TYPESAFE_ENDPOINT.to_string()),
            llm_key,
            llm_endpoint,
            llm_model,
            planner_model,
            planner_endpoint,
            planner_key,
            jina_key: pick(s.jina_key, "JINA_API_KEY"),
        };

        validate_endpoint("--typesafe-endpoint", &creds.typesafe_endpoint)?;
        Ok(creds)
    }
}

/// Reject an endpoint that is not HTTPS unless it is plainly local.
///
/// Credentials ride on every one of these requests, so an `http://` endpoint would
/// put them on the wire in clear text. Localhost is exempt because that is how you
/// point the tool at a model you are running yourself.
pub fn validate_endpoint(name: &str, url: &str) -> Result<()> {
    if url.starts_with("https://")
        || url.starts_with("http://127.0.0.1")
        || url.starts_with("http://localhost")
    {
        return Ok(());
    }
    bail!("{name} must be https:// (or a localhost address); got {url}")
}

/// Require the URL to be a full OpenAI-compatible chat completions endpoint.
///
/// Called for the LLM endpoint and the planner endpoint. NOT called for the TypeSafe
/// endpoint, which lives at `/v1/systemone` and has a different API shape.
///
/// Validation steps:
/// 1. Apply the existing scheme check (https, or http on localhost/127.0.0.1).
/// 2. Require the URL path to end with `/chat/completions`, allowing one optional
///    trailing slash and ignoring any query string.
///
/// On failure the error names the flag, shows the URL as given, states what is
/// required, and suggests the corrected form (built by `suggest_chat_endpoint`).
/// Suggesting is fine; using it silently is not.
pub fn validate_chat_endpoint(name: &str, url: &str) -> Result<()> {
    validate_endpoint(name, url)?;

    // Strip query string to inspect the path only.
    let path_part = match url.find('?') {
        Some(i) => &url[..i],
        None => url,
    };
    let path_no_slash = path_part.trim_end_matches('/');

    if path_no_slash.ends_with("/chat/completions") {
        return Ok(());
    }

    let suggestion = suggest_chat_endpoint(url);
    bail!(
        "{name} requires the full OpenAI-compatible chat completions URL; got: {url}\n\
         The path must end with /chat/completions. Did you mean:\n  {suggestion}"
    )
}

/// Build an error hint for a URL that is missing the `/chat/completions` suffix.
///
/// This function only constructs a suggestion to include in an error message — it does
/// NOT rewrite or silently accept any URL on the actual request path.
///
/// Shapes handled:
/// - Ends with `/chat` or `/chat/` → append `completions`
/// - Has some path after the host → append `/chat/completions`
/// - Bare host (no path, or only `/`) → append `/v1/chat/completions`
pub fn suggest_chat_endpoint(url: &str) -> String {
    // Strip query string so we only inspect the path.
    let (base, qs) = match url.find('?') {
        Some(i) => (&url[..i], &url[i..]),
        None => (url, ""),
    };
    let base_no_slash = base.trim_end_matches('/');

    // Decide whether there is any path component after scheme://host.
    let after_scheme = base_no_slash
        .find("://")
        .map(|i| &base_no_slash[i + 3..])
        .unwrap_or(base_no_slash);
    let has_path = after_scheme.contains('/');

    if base_no_slash.ends_with("/chat") {
        format!("{base_no_slash}/completions{qs}")
    } else if !has_path {
        // Bare host: suggest the canonical /v1/chat/completions path.
        format!("{base_no_slash}/v1/chat/completions{qs}")
    } else {
        format!("{base_no_slash}/chat/completions{qs}")
    }
}

#[cfg(test)]
mod config_tests {
    use super::{suggest_chat_endpoint, unquote, validate_chat_endpoint};

    #[test]
    fn unquote_strips_one_matching_pair_and_whitespace() {
        assert_eq!(
            unquote("'https://x/v1/chat/completions'"),
            "https://x/v1/chat/completions"
        );
        assert_eq!(unquote(" \"abc\" "), "abc");
        assert_eq!(unquote("'abc\""), "'abc\"");
        assert_eq!(unquote("plain"), "plain");
        assert_eq!(unquote("''"), "");
    }

    // --- validate_chat_endpoint: acceptance cases ---

    #[test]
    fn accepts_openrouter_full_url() {
        assert!(
            validate_chat_endpoint(
                "--llm-endpoint",
                "https://openrouter.ai/api/v1/chat/completions"
            )
            .is_ok()
        );
    }

    #[test]
    fn accepts_trailing_slash_on_completions() {
        assert!(
            validate_chat_endpoint(
                "--llm-endpoint",
                "https://openrouter.ai/api/v1/chat/completions/"
            )
            .is_ok()
        );
    }

    #[test]
    fn accepts_localhost_http() {
        assert!(
            validate_chat_endpoint(
                "--llm-endpoint",
                "http://localhost:8000/v1/chat/completions"
            )
            .is_ok()
        );
    }

    #[test]
    fn accepts_query_string() {
        assert!(
            validate_chat_endpoint(
                "--llm-endpoint",
                "https://example.com/v1/chat/completions?key=x"
            )
            .is_ok()
        );
    }

    // --- validate_chat_endpoint: rejection cases ---

    #[test]
    fn rejects_chat_slash() {
        let err = validate_chat_endpoint("--llm-endpoint", "https://vllm.example.org/v1/chat/")
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("vllm.example.org"), "{msg}");
        assert!(
            msg.contains("vllm.example.org/v1/chat/completions"),
            "{msg}"
        );
    }

    #[test]
    fn rejects_v1_only() {
        assert!(validate_chat_endpoint("--llm-endpoint", "https://example.com/v1").is_err());
    }

    #[test]
    fn rejects_bare_host() {
        assert!(validate_chat_endpoint("--llm-endpoint", "https://example.com").is_err());
    }

    #[test]
    fn rejects_http_on_non_local() {
        assert!(
            validate_chat_endpoint("--llm-endpoint", "http://example.com/v1/chat/completions")
                .is_err()
        );
    }

    // --- suggest_chat_endpoint ---

    #[test]
    fn suggest_from_chat_slash() {
        assert_eq!(
            suggest_chat_endpoint("https://vllm.example.org/v1/chat/"),
            "https://vllm.example.org/v1/chat/completions"
        );
    }

    #[test]
    fn suggest_from_v1() {
        assert_eq!(
            suggest_chat_endpoint("https://host/v1"),
            "https://host/v1/chat/completions"
        );
    }

    #[test]
    fn suggest_from_bare_host() {
        assert_eq!(
            suggest_chat_endpoint("https://host"),
            "https://host/v1/chat/completions"
        );
    }
}

/// Which search lanes a run may use.
///
/// Lives here rather than in `main.rs` because both the CLI and the `--api`
/// server need it, and a request option that names an engine must resolve to
/// exactly the same set of lanes a flag would.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "lower")]
pub enum SearchEngines {
    /// Every lane that is available in this configuration.
    Auto,
    /// DuckDuckGo only (needs obscura).
    Ddg,
    /// Jina only (needs a Jina key).
    Jina,
}

/// Everything that decides how deep the scout digs and where it draws its lines.
///
/// The thresholds are applied to calibrated probabilities from Jev, but where to put
/// each line is a product decision, not a model output. Keeping them together means
/// you can retune without rerunning anything, and makes it honest that they are
/// starting points rather than tuned constants.
#[derive(Debug, Clone)]
pub struct Tunables {
    // --- Search breadth ---
    /// Search queries issued per round.
    pub queries_per_round: usize,
    /// Results requested from each search.
    pub results_per_query: usize,
    /// Pages actually fetched per query, after triage ranks them.
    pub read_per_query: usize,

    // --- Depth / termination -------------------------------------------------
    // There is no wall-clock limit by design. The scout stops when it has what it
    // came for, or when it stops making progress — not when a timer fires, since a
    // timer would truncate a run that was still producing.
    /// Hard ceiling on rounds, so a pathological run cannot spin forever.
    ///
    /// Under `auto_rounds` this is only a safety net — the preset's ceiling —
    /// and the run is expected to stop on a judged verdict well before it.
    /// A number the caller chose is honoured exactly, on both paths.
    pub max_rounds: usize,
    /// Let the run decide when to stop instead of running to `max_rounds`.
    ///
    /// On by default. A harvest with a target then also stops, once the target
    /// is met, when Jev judges progress has levelled off (`plateaued`, read
    /// from a per-round history) — q81 classified 41 of its 62 grantees in 5
    /// rounds against 39 in 10. Open-ended harvests do not: measured
    /// 2026-09-24, the verdict ran high on nearly every round of a bursty
    /// "find all" discovery and cut q65 to 52 records and q101 to 15 against
    /// fixed-40 baselines of 141 and 217. An answer keeps its own short cap. A
    /// caller who names a number opts out: they asked for depth, and a
    /// plateau stop would second-guess them.
    pub auto_rounds: bool,
    /// Consecutive rounds yielding zero new verified items before giving up.
    pub max_barren_rounds: usize,

    // --- Jev thresholds ---
    /// Minimum `answers_query` for a search hit to be worth fetching.
    pub keep_relevance: f64,
    /// Above this `is_slop`, discard however relevant it looks.
    pub slop_ceiling: f64,
    /// At or above this `injection`, the chunk never reaches the generative model.
    pub injection_ceiling: f64,
    /// Minimum grounding for an extracted record to be kept.
    ///
    /// This is the number that decides whether the tool hallucinates. A record the
    /// generative model produced but Jev cannot find in the source text is exactly
    /// what a fabricated email address looks like.
    pub grounding_floor: f64,
    /// Minimum `supports` for a passage to be cited in an answer.
    pub keep_support: f64,

    /// Minimum `has_items` before a chunk is sent for extraction.
    ///
    /// Deliberately low. The asymmetry is the point: a false positive wastes one
    /// cheap call, while a false negative silently discards the only page that had
    /// the data. An early build ran this at 0.30, rejected all 208 chunks of a
    /// 32-page run, never called extraction once, and reported "empty".
    pub has_items_floor: f64,

    /// Questions packed into one verification request.
    ///
    /// Measured: 192 questions in a single request answered in 0.98s, against 0.80s
    /// for 16. Question count is very nearly free — only the token budget genuinely
    /// binds — while *requests* are scarce, since throughput plateaus near 4 per
    /// second however much you parallelise. So pack hard and send seldom.
    pub max_questions_per_request: usize,

    // --- Chunking ---
    pub chunk_chars: usize,
    /// Chunks screened per page for an answer mission: the head, plus the page's
    /// last chunk once the page runs past the cap (see `head_and_tail`).
    ///
    /// An answer needs two or three passages that settle the question; screening the
    /// rest of a long page is money spent to confirm what you already have. Measured
    /// 2026-09-24 over 30 answer runs, every final evidence passage traced back to its
    /// chunk and every answer sentence to its citations: a cap of 7 screens 17% fewer
    /// chunks and loses no sentence whose only support was cut; 6 loses four such
    /// sentences, 5 loses seven. Default 7 (was 10); the quick and thorough presets
    /// were not measured and keep their values.
    pub max_chunks_per_page: usize,
    /// Chunks screened per page for a harvest.
    ///
    /// Much higher, because a harvest's value *is* the long tail. Measured: the same
    /// public register yielded 18 records when the cap truncated its table and 36
    /// when more of it survived — the cap, not the web, was the limit.
    pub harvest_max_chunks_per_page: usize,

    // --- Concurrency and transport ---
    pub concurrency: usize,
    pub page_timeout: Duration,
    pub http_timeout: Duration,
    pub max_retries: usize,

    // --- Package B: harvest v2 tunables --------------------------------------
    /// Minimum Jev probability that a record satisfies the mission constraints
    /// before it is kept. 0.5 is deliberately lenient: mission constraints are
    /// often paraphrases ("that ran participatory budgeting") a source will not
    /// echo verbatim, so a middling probability is what a genuinely qualifying
    /// entity typically scores. Raise it if the constraint wording is precise.
    pub constraint_floor: f64,
    /// Minimum Jev probability that following a link would reach another list
    /// of entities. Higher than grounding because a wrong link is a whole page
    /// wasted, and follow decisions are made without seeing the page yet.
    pub follow_floor: f64,
    /// How many links to follow from one listing page. Capped hard because a
    /// register with 400 entity links would blow through the round budget in
    /// one pass; the top 12 are almost always the ones worth reading.
    pub max_follow_per_page: usize,
    /// Entities whose fields we enrich per round. 25 keeps one round bounded
    /// (25 entities × ~2 fields × 2 pages = ~100 fetches) while still visibly
    /// closing the field-fill gap between rounds.
    pub enrich_batch: usize,
    /// Enrichment pages fetched per (entity, field) search. Two is enough for a
    /// contact page: the official homepage almost always makes the top rank and
    /// a second page catches the one time it does not.
    pub enrich_read: usize,
    /// Minimum choice-confidence for accepting the regex-candidate answer as
    /// the entity's official field value. 0.5 matches a calibrated "more likely
    /// than not" and lets a page with two plausible emails still succeed when
    /// Jev picks the general one.
    pub select_confidence: f64,
    /// Search results requested per enrichment query. Smaller than the general
    /// results_per_query because enrichment already knows which entity it
    /// wants and rarely needs the long tail.
    pub enrich_results_per_query: usize,

    /// Run a research plan (one planner-LLM call) at the start of a harvest
    /// and draw from it before falling back to the LLM planner.
    /// Measured motivation: on a "100 municipalities" run, rounds 2–3 fetched
    /// zero pages because the LLM planner returned one or two generic queries
    /// after round 1's discovery. Research-plan queries keep discovery moving
    /// while the planner is reserved for exploit (`site:`) angles.
    pub research_plan: bool,

    /// Minimum Jev probability that a generated query would name entities
    /// matching the anchors and filters, before we bother searching it. P4:
    /// spending a search on a query Jev says would be about the wrong topic
    /// wastes both the search and the triage that follows. 0.4 is
    /// deliberately lenient — the gate is meant to drop the plainly-off
    /// queries, not to second-guess the planner's borderline calls. The
    /// gate never drops below two queries per round; keeping the two best
    /// beats an empty round when Jev disagrees with everything the planner
    /// wrote.
    pub query_gate_floor: f64,

    // --- Anchor enumeration corroboration (harvest) ------------------------
    /// A page on the mission's anchor domain counts as an authoritative
    /// enumeration only if Jev reads it as complete at or above this
    /// probability. Measured on Linear's pricing page (2026-09-21): it is a
    /// complete listing of the plans, and two SEO-blog rows ("Plus",
    /// "Standard") were absent from it — exactly the pollution the gate
    /// exists to drop. Directories of a *subset* (one federation's members)
    /// read as incomplete, which keeps the gate off broad discovery missions.
    pub enum_completeness_floor: f64,
    /// A determination field (is it SaaS, does it offer an API) records "no"
    /// when the entity's own page — entity-bound, the page a company writes
    /// about itself — was read and Jev puts the probability of "yes" at or
    /// below this ceiling. Decisive silence on the authoritative page is a
    /// resolved negative, the field-level mirror of the answer path's
    /// negative license; between this ceiling and the grounding floor the
    /// field honestly stays empty.
    pub determination_no_ceiling: f64,
    /// An entity is treated as absent from an enumeration page when Jev's
    /// presence probability is at or below this. Only the joint condition —
    /// a complete enumeration that never mentions the entity — drops a
    /// record; absence from a partial listing proves nothing.
    pub enum_absence_ceiling: f64,
    /// Minimum records extracted from enumeration pages before the gate
    /// arms. An enumeration that has only named one or two entities has not
    /// demonstrated that it enumerates.
    pub enum_min_yield: usize,
    /// Cap on enumeration pages kept for corroboration, so a mission whose
    /// anchor domain is a sprawling site cannot spend the state budget on
    /// chunks. Kept in first-read order.
    pub enum_max_pages: usize,

    /// Deadline for one search lane's batch of queries, inside whatever
    /// overall budget the round has. Each lane gets its own so a wedged
    /// engine costs only its own results — the point of running lanes
    /// concurrently is that no single engine can starve a round. Twenty
    /// seconds is roughly three times the observed p50 for a five-query
    /// DuckDuckGo batch, so it fires only on a lane that is genuinely stuck.
    pub search_lane_timeout: Duration,

    /// Whether successful search responses are cached on disk.
    pub search_cache: bool,

    /// How long a cached search response stays usable. Six hours: long
    /// enough that a re-run or a second mission on the same subject pays
    /// nothing for its searches, short enough that a question about this
    /// morning still reaches the web.
    pub search_cache_ttl: Duration,

    /// Maximum number of re-aim rounds per run. P5: when a round returns
    /// hits but Jev rejects every page, the planner writes corrected
    /// searches once (twice by default) before the barren rule kicks in.
    /// A cap because a mission whose vocabulary genuinely does not match
    /// the web must not keep spending planner calls forever.
    pub max_reaims: usize,

    /// Minimum `currency` for a passage to survive on a time-sensitive answer.
    ///
    /// A low bar on purpose. The `cur{slot}` noul asks whether the passage
    /// contradicts today — whether it describes a state of affairs that has since
    /// changed — not whether it is fresh for its own sake. A 2019 page stating a
    /// founding date is perfectly current; a 2019 page stating who the current CEO
    /// is may not be. Set high, this would throw away every older page regardless
    /// of what it claims, which is the opposite of the intent, so the floor only
    /// catches passages Jev is fairly sure are stale. Ordering (currency as a
    /// ranking multiplier) does the rest of the work.
    pub currency_floor: f64,

    /// Minimum Jev probability that a single claim in the written answer is
    /// stated by the gathered evidence, before the claim is left unmarked.
    ///
    /// This is a **support** probability, not a truth probability: a claim below
    /// the floor is one nothing in the evidence backs, which is not the same as a
    /// claim that is false. The consequence is calibrated to that difference — the
    /// sentence is marked and counted, not deleted.
    ///
    /// 0.5 is "more likely than not" on a calibrated noul. Measured 2026-09-19
    /// against the live API on a four-sentence answer with two invented sentences,
    /// the per-claim nouls returned 0.98 / 0.98 / 0.03 / 0.03 — the separation is
    /// wide enough that the exact floor does not matter much, and picking the
    /// midpoint avoids tuning to one example.
    pub claim_floor: f64,
}

impl Default for Tunables {
    fn default() -> Self {
        Self {
            queries_per_round: 5,
            results_per_query: 8,
            read_per_query: 3,

            max_rounds: 40,

            auto_rounds: true,
            max_barren_rounds: 3,

            keep_relevance: 0.35,
            slop_ceiling: 0.75,
            injection_ceiling: 0.55,
            grounding_floor: 0.60,
            keep_support: 0.45,
            has_items_floor: 0.12,
            max_questions_per_request: 192,

            chunk_chars: 4000,
            max_chunks_per_page: 7,
            harvest_max_chunks_per_page: 40,

            concurrency: 8,
            page_timeout: Duration::from_secs(45),
            http_timeout: Duration::from_secs(180),
            max_retries: 4,

            constraint_floor: 0.5,
            follow_floor: 0.6,
            max_follow_per_page: 12,
            enrich_batch: 25,
            enrich_read: 2,
            select_confidence: 0.5,
            enrich_results_per_query: 5,
            research_plan: true,

            search_lane_timeout: Duration::from_secs(20),
            search_cache: true,
            search_cache_ttl: crate::search_cache::DEFAULT_TTL,

            query_gate_floor: 0.4,

            enum_completeness_floor: 0.8,
            determination_no_ceiling: 0.1,
            enum_absence_ceiling: 0.2,
            enum_min_yield: 3,
            enum_max_pages: 3,

            max_reaims: 2,
            currency_floor: 0.25,
            claim_floor: 0.5,
        }
    }
}

impl Tunables {
    /// Widen everything for a run that should leave no stone unturned.
    pub fn thorough() -> Self {
        Self {
            queries_per_round: 10,
            results_per_query: 20,
            read_per_query: 8,
            max_rounds: 100,
            auto_rounds: true,
            max_barren_rounds: 5,
            max_chunks_per_page: 20,
            harvest_max_chunks_per_page: 80,
            enrich_batch: 50,
            ..Default::default()
        }
    }

    /// Narrow everything for a quick look.
    pub fn quick() -> Self {
        Self {
            queries_per_round: 3,
            results_per_query: 6,
            read_per_query: 2,
            max_rounds: 3,
            auto_rounds: true,
            max_barren_rounds: 1,
            max_chunks_per_page: 6,
            harvest_max_chunks_per_page: 16,
            enrich_batch: 12,
            ..Default::default()
        }
    }
}

impl Tunables {
    /// Chunk cap for this mission.
    ///
    /// A harvest and an answer want opposite things from a long page, so they get
    /// different caps rather than a compromise that serves neither.
    pub fn chunk_cap(&self, harvest: bool) -> usize {
        if harvest {
            self.harvest_max_chunks_per_page
        } else {
            self.max_chunks_per_page
        }
    }
}

/// How many pages to render at once, by default.
///
/// Rendering is CPU-bound, so the useful ceiling is the core count: measured on a
/// 12-core box, 32 pages took 74s at concurrency 4, 32s at 8 and 29s at 32 — the
/// gain past the core count is noise. Deriving it from the hardware means a laptop
/// does not thrash and a large machine is not left idle.
///
/// Clamped at both ends: below 4 the pipeline stalls on a single slow page, and
/// above 16 you are queueing work the CPU cannot start.
pub fn default_fetch_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .clamp(4, 16)
}

/// Jev 1.13 bills input tokens at $42 per billion; output tokens are free.
/// <https://docs.typesafe.ai/models>
pub const TYPESAFE_USD_PER_INPUT_TOKEN: f64 = 42.0 / 1e9;
