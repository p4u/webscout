//! The `--api` server: one HTTP endpoint per thing the CLI already does.
//!
//! Three rules shape everything here.
//!
//! **No credentials over the wire.** Keys, endpoints and the obscura binary path
//! are resolved once from the server's own environment at startup, exactly as the
//! CLI resolves them, and are never readable or settable by a request. The option
//! catalogue below is the complete set of things a request may change; anything
//! else is rejected as unknown, which is what makes "`llm_key` in the body" a 400
//! rather than a quietly ignored field.
//!
//! **The catalogue is the contract.** `/api/options` is generated from
//! `Tunables::default()` and the preset functions, and request validation is driven
//! from the same table, so the UI's idea of the parameter surface and the server's
//! idea of it cannot drift apart.
//!
//! **A disconnect stops the work.** The run future lives *inside* the response
//! stream rather than in a spawned task, so dropping the response — which is what a
//! client disconnect does — drops the future and cancels the run. A spawned task
//! would keep burning Jev and LLM budget for a browser tab that closed.

use std::collections::{BTreeMap, VecDeque};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::Json;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::browser::{Backend, Fetcher, Jina, Obscura, SearchLane};
use crate::config::{Credentials, SearchEngines, Tunables, default_fetch_concurrency};
use crate::llm::{Llm, LlmUsage, ThinkingControl};
use crate::output::{self, Format};
use crate::scout::Scout;
use crate::search_cache::SearchCache;
use crate::types::ScoutReport;
use crate::typesafe::Jev;

/// Port the server binds when nothing says otherwise.
pub const DEFAULT_API_PORT: u16 = 8080;

/// Completed runs kept for the download endpoint.
///
/// Twenty is enough for a person clicking around in the UI and small enough that a
/// long harvest's four renderings cannot grow the process without bound.
const MAX_STORED_RUNS: usize = 20;

/// Progress events buffered before the run starts dropping them.
///
/// Generous, because the events are tiny and a browser on a slow link should not
/// lose the story of a run. Overflow drops rather than blocks — the run is the
/// thing that matters, not the commentary.
const PROGRESS_BUFFER: usize = 256;

/// How often the stream samples the clients' counters while a run is in flight.
///
/// One second is fast enough that a person watching sees the numbers move and
/// slow enough that a long harvest adds a few hundred tiny lines, not thousands.
/// Sampling is three atomic loads, so the interval is chosen for the reader's
/// benefit rather than the run's.
const USAGE_INTERVAL: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------- progress --

/// One coarse, human-readable step in a run.
///
/// Deliberately not a mirror of the tracing spans: this is what a person watching
/// a progress list wants to read, not what an engineer debugging a stage wants to
/// grep. The tracing logs are untouched and remain the authoritative record.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    /// Coarse stage name: `mission`, `search`, `extract`, `enrich`,
    /// `round-complete`, `steer`, `synthesize`, `verify`, `done`.
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round: Option<usize>,
    pub message: String,
    pub counts: ProgressCounts,
}

/// Running totals carried on every progress event so a UI never has to add up.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ProgressCounts {
    pub pages: usize,
    pub records: usize,
}

// ----------------------------------------------------------------- usage --

/// What a run has spent so far, read straight from the clients' counters.
///
/// Every field here already exists as an atomic inside `Jev` and `Llm`, so a
/// sample is a handful of relaxed loads with no lock and no counter of our own —
/// which is the only reason it is safe to take one every second while the run is
/// mid-flight. Deliberately *not* a `Stats`: `Stats` is the finished record of a
/// run, and half of it (rounds, pages, rejections) is not knowable from the
/// clients alone.
///
/// Jev's cost is always known: it is its input tokens times a measured constant.
/// The writer and planner costs are whatever their endpoint reported, which is a
/// real figure on OpenRouter and nothing at all elsewhere — hence `Option`, and
/// hence a key that is simply absent from the wire rather than a zero that would
/// read as "this run was free".
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageSnapshot {
    pub jev_requests: usize,
    pub jev_input_tokens: usize,
    pub jev_cost_usd: f64,
    pub llm: LlmUsage,
    pub planner: LlmUsage,
}

impl UsageSnapshot {
    /// Read the three clients a scout is holding.
    pub fn sample(scout: &Scout) -> Self {
        let (jev_requests, jev_input_tokens, jev_cost_usd) = scout.jev.stats();
        Self {
            jev_requests,
            jev_input_tokens,
            jev_cost_usd,
            llm: scout.llm.stats(),
            planner: scout.planner.stats(),
        }
    }

    /// The same numbers as a finished run reports.
    ///
    /// The last `usage` event of a run is built from the report rather than from
    /// another sample, so the panel a person is left looking at cannot disagree
    /// with the `result` event's stats by a request that landed in between.
    pub fn from_stats(st: &crate::types::Stats) -> Self {
        Self {
            jev_requests: st.jev_requests,
            jev_input_tokens: st.jev_input_tokens,
            jev_cost_usd: st.jev_cost_usd,
            llm: LlmUsage {
                requests: st.llm_requests,
                prompt_tokens: st.llm_prompt_tokens,
                completion_tokens: st.llm_completion_tokens,
                reasoning_tokens: st.llm_reasoning_tokens,
                cost_usd: st.llm_cost_usd,
            },
            planner: LlmUsage {
                requests: st.planner_requests,
                prompt_tokens: st.planner_prompt_tokens,
                completion_tokens: st.planner_completion_tokens,
                reasoning_tokens: st.planner_reasoning_tokens,
                cost_usd: st.planner_cost_usd,
            },
        }
    }
}

/// One component's slice of a `usage` event.
///
/// `cost_usd` and `reasoning_tokens` are omitted when there is nothing to say:
/// an absent cost means the endpoint does not report one, and an absent
/// reasoning count means none was spent. Both are cleaner for a reader than a
/// zero that has to be interpreted.
fn llm_usage_json(u: &LlmUsage) -> Value {
    let mut o = json!({
        "requests": u.requests,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
    });
    if let Some(cost) = u.cost_usd {
        o["cost_usd"] = json!(cost);
    }
    if u.reasoning_tokens > 0 {
        o["reasoning_tokens"] = json!(u.reasoning_tokens);
    }
    o
}

// --------------------------------------------------------------- run ids --

/// Monotonic tiebreaker, so two runs started inside the same nanosecond tick — or
/// on a platform whose clock is coarse — still get distinct ids.
static RUN_SEQ: AtomicU64 = AtomicU64::new(0);

/// Format a run id from a timestamp and a sequence number.
///
/// Hex of the nanosecond clock plus a counter. A UUID crate would add a dependency
/// to produce a string that is no more unique for this purpose: ids are process
/// local, live for at most twenty runs, and never leave this server.
fn format_run_id(nanos: u128, seq: u64) -> String {
    format!("{nanos:x}-{seq:x}")
}

fn new_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format_run_id(nanos, RUN_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// Render a Unix timestamp as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// Reuses the calendar arithmetic already in `clock`, for the same reason that
/// module exists: four lines of integer maths is not worth a date crate.
fn rfc3339(secs: u64) -> String {
    let day = crate::clock::format_civil(crate::clock::civil_from_days((secs / 86_400) as i64));
    let rem = secs % 86_400;
    format!(
        "{day}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn now_rfc3339() -> String {
    rfc3339(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
}

// --------------------------------------------------------------- presets --

/// Which `Tunables` preset a run starts from, before per-option overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Quick,
    Standard,
    Thorough,
}

impl Preset {
    pub fn as_str(self) -> &'static str {
        match self {
            Preset::Quick => "quick",
            Preset::Standard => "standard",
            Preset::Thorough => "thorough",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "quick" => Some(Preset::Quick),
            "standard" => Some(Preset::Standard),
            "thorough" => Some(Preset::Thorough),
            _ => None,
        }
    }

    /// The preset's tunables. Straight from the functions the CLI uses, so a
    /// preset cannot mean one thing on the command line and another over HTTP.
    pub fn tunables(self) -> Tunables {
        match self {
            Preset::Quick => Tunables::quick(),
            Preset::Standard => Tunables::default(),
            Preset::Thorough => Tunables::thorough(),
        }
    }
}

// ------------------------------------------------------- option catalogue --

/// The JSON type of one option, as the UI should render it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    Integer,
    Number,
    Boolean,
    Enum,
    String,
}

impl OptionType {
    fn noun(self) -> &'static str {
        match self {
            OptionType::Integer => "whole number",
            OptionType::Number => "number",
            OptionType::Boolean => "boolean",
            OptionType::Enum => "one of the listed values",
            OptionType::String => "string",
        }
    }
}

/// One settable parameter, described well enough for a UI to render it blind.
#[derive(Debug, Clone, Serialize)]
pub struct OptionSpec {
    pub name: &'static str,
    pub label: &'static str,
    #[serde(rename = "type")]
    pub kind: OptionType,
    pub default: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<&'static str>>,
    /// One line, always present. An option a person cannot understand from the
    /// form is an option they will set wrongly.
    pub help: &'static str,
}

/// A rendered group of options.
#[derive(Debug, Clone, Serialize)]
pub struct OptionGroup {
    pub id: &'static str,
    pub label: &'static str,
    pub options: Vec<OptionSpec>,
}

fn int_opt(
    name: &'static str,
    label: &'static str,
    default: usize,
    min: u64,
    max: u64,
    help: &'static str,
) -> OptionSpec {
    OptionSpec {
        name,
        label,
        kind: OptionType::Integer,
        default: json!(default),
        min: Some(json!(min)),
        max: Some(json!(max)),
        values: None,
        help,
    }
}

fn num_opt(
    name: &'static str,
    label: &'static str,
    default: f64,
    min: f64,
    max: f64,
    help: &'static str,
) -> OptionSpec {
    OptionSpec {
        name,
        label,
        kind: OptionType::Number,
        default: json!(default),
        min: Some(json!(min)),
        max: Some(json!(max)),
        values: None,
        help,
    }
}

fn bool_opt(
    name: &'static str,
    label: &'static str,
    default: bool,
    help: &'static str,
) -> OptionSpec {
    OptionSpec {
        name,
        label,
        kind: OptionType::Boolean,
        default: json!(default),
        min: None,
        max: None,
        values: None,
        help,
    }
}

fn enum_opt(
    name: &'static str,
    label: &'static str,
    default: &'static str,
    values: &[&'static str],
    help: &'static str,
) -> OptionSpec {
    OptionSpec {
        name,
        label,
        kind: OptionType::Enum,
        default: json!(default),
        min: None,
        max: None,
        values: Some(values.to_vec()),
        help,
    }
}

fn string_opt(name: &'static str, label: &'static str, help: &'static str) -> OptionSpec {
    OptionSpec {
        name,
        label,
        kind: OptionType::String,
        // Empty means "whatever the server was started with". The server's own
        // model name is configuration, and echoing it here would be the first
        // step towards echoing the endpoint it goes with.
        default: json!(""),
        min: None,
        max: None,
        values: None,
        help,
    }
}

/// Output formats the API will render and serve.
///
/// `terminal` is deliberately absent: it is a rendering for a TTY with a person in
/// front of it, not something a browser or a download has any use for.
pub const FORMATS: [&str; 4] = ["markdown", "json", "csv", "jsonl"];

const PRESETS: [&str; 3] = ["quick", "standard", "thorough"];
const ENGINES: [&str; 3] = ["auto", "ddg", "jina"];
const THINKING: [&str; 5] = ["auto", "openrouter", "vllm", "effort", "off"];

/// The whole settable surface, generated from the defaults the code actually uses.
///
/// Every number here reads its default out of `Tunables::default()`, so retuning a
/// threshold in `config.rs` moves the API's default with it and no one has to
/// remember this file exists.
pub fn catalogue() -> Vec<OptionGroup> {
    let d = Tunables::default();
    vec![
        OptionGroup {
            id: "basic",
            label: "Search",
            options: vec![
                enum_opt(
                    "preset",
                    "Preset",
                    Preset::Standard.as_str(),
                    &PRESETS,
                    "Starting point for every other setting: quick looks, standard digs, thorough leaves no stone unturned.",
                ),
                int_opt(
                    "max_rounds",
                    "Max rounds",
                    d.max_rounds,
                    1,
                    200,
                    "Hard ceiling on research rounds. There is no time limit; this is what bounds the work.",
                ),
                enum_opt(
                    "format",
                    "Format",
                    "markdown",
                    &FORMATS,
                    "How the finished result is rendered.",
                ),
                enum_opt(
                    "search_engines",
                    "Search engines",
                    "auto",
                    &ENGINES,
                    "Which search lanes to run: auto uses every lane available, or pin a single engine.",
                ),
            ],
        },
        OptionGroup {
            id: "advanced",
            label: "Advanced",
            options: vec![
                int_opt(
                    "queries_per_round",
                    "Queries per round",
                    d.queries_per_round,
                    1,
                    50,
                    "How many search queries each round issues.",
                ),
                int_opt(
                    "results_per_query",
                    "Results per query",
                    d.results_per_query,
                    1,
                    100,
                    "How many results each search asks the engine for.",
                ),
                int_opt(
                    "read_per_query",
                    "Pages read per query",
                    d.read_per_query,
                    1,
                    50,
                    "How many pages are actually fetched per query, after triage ranks them.",
                ),
                int_opt(
                    "max_barren_rounds",
                    "Max barren rounds",
                    d.max_barren_rounds,
                    1,
                    20,
                    "Stop after this many consecutive rounds that find nothing new.",
                ),
                int_opt(
                    "fetch_concurrency",
                    "Fetch concurrency",
                    default_fetch_concurrency(),
                    1,
                    64,
                    "Pages rendered in parallel within one batch; also the politeness knob.",
                ),
                int_opt(
                    "concurrency",
                    "Verification concurrency",
                    d.concurrency,
                    1,
                    64,
                    "Concurrent verification requests. Throughput plateaus near 8; fatter requests help more.",
                ),
                int_opt(
                    "max_questions_per_request",
                    "Questions per request",
                    d.max_questions_per_request,
                    1,
                    512,
                    "Questions packed into one verification request. The token budget still caps an oversized batch.",
                ),
                num_opt(
                    "grounding_floor",
                    "Grounding floor",
                    d.grounding_floor,
                    0.0,
                    1.0,
                    "Minimum confidence that an extracted value really appears in its source before it is kept.",
                ),
                num_opt(
                    "constraint_floor",
                    "Constraint floor",
                    d.constraint_floor,
                    0.0,
                    1.0,
                    "Minimum confidence that a record satisfies the request's constraints.",
                ),
                num_opt(
                    "claim_floor",
                    "Claim floor",
                    d.claim_floor,
                    0.0,
                    1.0,
                    "Minimum support for a sentence of the written answer before it is left unmarked.",
                ),
                num_opt(
                    "currency_floor",
                    "Currency floor",
                    d.currency_floor,
                    0.0,
                    1.0,
                    "Minimum currency for a passage to survive on a time-sensitive question.",
                ),
                num_opt(
                    "query_gate_floor",
                    "Query gate floor",
                    d.query_gate_floor,
                    0.0,
                    1.0,
                    "Generated queries scoring below this are dropped before they are searched.",
                ),
                num_opt(
                    "follow_floor",
                    "Follow floor",
                    d.follow_floor,
                    0.0,
                    1.0,
                    "Minimum confidence that following a link would reach more entities.",
                ),
                num_opt(
                    "select_confidence",
                    "Select confidence",
                    d.select_confidence,
                    0.0,
                    1.0,
                    "Minimum confidence for accepting a detected candidate as an entity's field value.",
                ),
                int_opt(
                    "enrich_batch",
                    "Enrich batch",
                    d.enrich_batch,
                    1,
                    500,
                    "Entities whose missing fields are chased per round.",
                ),
                int_opt(
                    "enrich_read",
                    "Enrich pages read",
                    d.enrich_read,
                    1,
                    20,
                    "Pages fetched per enrichment search.",
                ),
                int_opt(
                    "max_follow_per_page",
                    "Max links followed per page",
                    d.max_follow_per_page,
                    0,
                    100,
                    "How many outbound links one listing page may contribute to the next round.",
                ),
                bool_opt(
                    "no_plan",
                    "Disable research plan",
                    false,
                    "Skip the one-off research plan and rely on the round-by-round planner instead.",
                ),
                bool_opt(
                    "no_enrich",
                    "Disable enrichment",
                    false,
                    "Keep only what listing pages already stated; do not chase missing fields.",
                ),
                bool_opt(
                    "no_follow",
                    "Disable link following",
                    false,
                    "Do not queue outbound links from productive listing pages.",
                ),
                bool_opt(
                    "no_auto",
                    "Disable auto steering",
                    false,
                    "Use the fixed stopping rules instead of letting the judge steer depth.",
                ),
                bool_opt(
                    "no_search_cache",
                    "Disable search cache",
                    false,
                    "Do not read or write the on-disk cache of search responses.",
                ),
                int_opt(
                    "search_cache_ttl",
                    "Search cache TTL (seconds)",
                    d.search_cache_ttl.as_secs() as usize,
                    0,
                    604_800,
                    "How long a cached search response stays usable, in seconds.",
                ),
                string_opt(
                    "llm_model",
                    "Writer model",
                    "Model name for extraction and prose. Blank uses the server's configured model.",
                ),
                string_opt(
                    "planner_model",
                    "Planner model",
                    "Model name for mission parsing and planning. Blank uses the server's configured model.",
                ),
                enum_opt(
                    "thinking_control",
                    "Thinking control",
                    "auto",
                    &THINKING,
                    "How reasoning is switched on for the writer endpoint; the wrong form can burn the whole budget.",
                ),
                enum_opt(
                    "planner_thinking_control",
                    "Planner thinking control",
                    "auto",
                    &THINKING,
                    "How reasoning is switched on for the planner endpoint.",
                ),
            ],
        },
    ]
}

/// Flatten the catalogue into a name → spec lookup.
fn spec_index() -> BTreeMap<&'static str, OptionSpec> {
    catalogue()
        .into_iter()
        .flat_map(|g| g.options)
        .map(|o| (o.name, o))
        .collect()
}

// ----------------------------------------------------------- run options --

/// Everything one request may change about a run.
///
/// Note what is *not* here: no key, no endpoint, no binary path. Those come from
/// the server's environment and nowhere else.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub preset: Preset,
    pub format: Format,
    /// `Auto` means "whatever the server was started with", which may itself be
    /// auto. A request can pin an engine; it cannot invent one the server has no
    /// credentials or binary for, and asking for one is a 400 at scout-build time.
    pub engines: SearchEngines,
    pub tune: Tunables,
    /// `None` means "whatever the process was started with"; the catalogue
    /// advertises that startup default so the UI still shows a number.
    pub fetch_concurrency: Option<usize>,
    pub auto: bool,
    pub follow: bool,
    pub enrich: bool,
    pub llm_model: Option<String>,
    pub planner_model: Option<String>,
    pub thinking_control: Option<ThinkingControl>,
    pub planner_thinking_control: Option<ThinkingControl>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            preset: Preset::Standard,
            format: Format::Markdown,
            engines: SearchEngines::Auto,
            tune: Tunables::default(),
            fetch_concurrency: None,
            auto: true,
            follow: true,
            enrich: true,
            llm_model: None,
            planner_model: None,
            thinking_control: None,
            planner_thinking_control: None,
        }
    }
}

fn type_error(spec: &OptionSpec, got: &Value) -> String {
    format!(
        "option \"{}\" must be a {}; got {}",
        spec.name,
        spec.kind.noun(),
        got
    )
}

fn want_u64(spec: &OptionSpec, v: &Value) -> Result<u64, String> {
    // `as_u64` rejects floats, negatives, strings and booleans in one go, which is
    // exactly the contract: 3.5 rounds is not a smaller mistake than "three".
    let n = v.as_u64().ok_or_else(|| type_error(spec, v))?;
    let min = spec.min.as_ref().and_then(Value::as_u64).unwrap_or(0);
    let max = spec
        .max
        .as_ref()
        .and_then(Value::as_u64)
        .unwrap_or(u64::MAX);
    if n < min || n > max {
        return Err(format!(
            "option \"{}\" must be between {min} and {max}; got {n}",
            spec.name
        ));
    }
    Ok(n)
}

fn want_f64(spec: &OptionSpec, v: &Value) -> Result<f64, String> {
    if v.is_boolean() {
        return Err(type_error(spec, v));
    }
    let n = v.as_f64().ok_or_else(|| type_error(spec, v))?;
    let min = spec
        .min
        .as_ref()
        .and_then(Value::as_f64)
        .unwrap_or(f64::MIN);
    let max = spec
        .max
        .as_ref()
        .and_then(Value::as_f64)
        .unwrap_or(f64::MAX);
    if n < min || n > max {
        return Err(format!(
            "option \"{}\" must be between {min} and {max}; got {n}",
            spec.name
        ));
    }
    Ok(n)
}

fn want_bool(spec: &OptionSpec, v: &Value) -> Result<bool, String> {
    v.as_bool().ok_or_else(|| type_error(spec, v))
}

fn want_enum(spec: &OptionSpec, v: &Value) -> Result<String, String> {
    let s = v.as_str().ok_or_else(|| type_error(spec, v))?;
    let allowed = spec.values.clone().unwrap_or_default();
    if !allowed.contains(&s) {
        return Err(format!(
            "option \"{}\" must be one of {}; got \"{s}\"",
            spec.name,
            allowed.join(", ")
        ));
    }
    Ok(s.to_string())
}

fn want_string(spec: &OptionSpec, v: &Value) -> Result<String, String> {
    Ok(v.as_str().ok_or_else(|| type_error(spec, v))?.to_string())
}

/// Parse a thinking-control name through clap's own value parser.
///
/// Reusing the `ValueEnum` derive means the API and the flag accept exactly the
/// same names forever, with no second table to keep in step.
fn parse_thinking(s: &str) -> Option<ThinkingControl> {
    <ThinkingControl as clap::ValueEnum>::from_str(s, false).ok()
}

fn parse_engines(s: &str) -> Option<SearchEngines> {
    <SearchEngines as clap::ValueEnum>::from_str(s, false).ok()
}

fn parse_format(s: &str) -> Option<Format> {
    <Format as clap::ValueEnum>::from_str(s, false).ok()
}

impl RunOptions {
    /// Validate a request's `options` object and fold it onto a preset.
    ///
    /// Nothing is clamped: a value out of range is the caller's bug and silently
    /// moving it would make the run's behaviour disagree with the request that
    /// asked for it. Unknown names are an error rather than an ignored field, so a
    /// typo — or an attempt to pass `llm_key` — fails loudly.
    pub fn from_map(opts: &Map<String, Value>) -> Result<Self, String> {
        let index = spec_index();

        // Report every unknown name at once. Fixing a form one round-trip per typo
        // is a bad experience, and the whole set is already known here.
        let mut unknown: Vec<&str> = opts
            .keys()
            .map(String::as_str)
            .filter(|k| !index.contains_key(k))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(format!(
                "unknown option{}: {}. Credentials, endpoints and binary paths are \
                 server configuration and cannot be set per request.",
                if unknown.len() == 1 { "" } else { "s" },
                unknown
                    .iter()
                    .map(|u| format!("\"{u}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // The preset decides the baseline every other option is folded onto, so it
        // has to be read before anything else regardless of key order.
        let mut out = RunOptions::default();
        if let Some(v) = opts.get("preset") {
            let spec = &index["preset"];
            let name = want_enum(spec, v)?;
            out.preset = Preset::parse(&name)
                .ok_or_else(|| format!("option \"preset\" has no such preset: \"{name}\""))?;
        }
        out.tune = out.preset.tunables();

        // Apply in catalogue order rather than request order, so two requests with
        // the same options always produce byte-identical tunables.
        for (name, spec) in &index {
            let Some(v) = opts.get(*name) else { continue };
            match *name {
                "preset" => {}
                "format" => {
                    let s = want_enum(spec, v)?;
                    out.format = parse_format(&s)
                        .ok_or_else(|| format!("option \"format\" has no such format: \"{s}\""))?;
                }
                "search_engines" => {
                    let s = want_enum(spec, v)?;
                    out.engines = parse_engines(&s).ok_or_else(|| {
                        format!("option \"search_engines\" has no such engine: \"{s}\"")
                    })?;
                }
                "thinking_control" => {
                    let s = want_enum(spec, v)?;
                    out.thinking_control = Some(parse_thinking(&s).ok_or_else(|| {
                        format!("option \"thinking_control\" has no such mode: \"{s}\"")
                    })?);
                }
                "planner_thinking_control" => {
                    let s = want_enum(spec, v)?;
                    out.planner_thinking_control = Some(parse_thinking(&s).ok_or_else(|| {
                        format!("option \"planner_thinking_control\" has no such mode: \"{s}\"")
                    })?);
                }
                "max_rounds" => out.tune.max_rounds = want_u64(spec, v)? as usize,
                "max_barren_rounds" => out.tune.max_barren_rounds = want_u64(spec, v)? as usize,
                "queries_per_round" => out.tune.queries_per_round = want_u64(spec, v)? as usize,
                "results_per_query" => out.tune.results_per_query = want_u64(spec, v)? as usize,
                "read_per_query" => out.tune.read_per_query = want_u64(spec, v)? as usize,
                "concurrency" => out.tune.concurrency = want_u64(spec, v)? as usize,
                "max_questions_per_request" => {
                    out.tune.max_questions_per_request = want_u64(spec, v)? as usize
                }
                "enrich_batch" => out.tune.enrich_batch = want_u64(spec, v)? as usize,
                "enrich_read" => out.tune.enrich_read = want_u64(spec, v)? as usize,
                "max_follow_per_page" => out.tune.max_follow_per_page = want_u64(spec, v)? as usize,
                "fetch_concurrency" => out.fetch_concurrency = Some(want_u64(spec, v)? as usize),
                "search_cache_ttl" => {
                    out.tune.search_cache_ttl = Duration::from_secs(want_u64(spec, v)?)
                }
                "grounding_floor" => out.tune.grounding_floor = want_f64(spec, v)?,
                "constraint_floor" => out.tune.constraint_floor = want_f64(spec, v)?,
                "claim_floor" => out.tune.claim_floor = want_f64(spec, v)?,
                "currency_floor" => out.tune.currency_floor = want_f64(spec, v)?,
                "query_gate_floor" => out.tune.query_gate_floor = want_f64(spec, v)?,
                "follow_floor" => out.tune.follow_floor = want_f64(spec, v)?,
                "select_confidence" => out.tune.select_confidence = want_f64(spec, v)?,
                "no_plan" => out.tune.research_plan = !want_bool(spec, v)?,
                "no_search_cache" => out.tune.search_cache = !want_bool(spec, v)?,
                "no_follow" => out.follow = !want_bool(spec, v)?,
                "no_enrich" => out.enrich = !want_bool(spec, v)?,
                "no_auto" => out.auto = !want_bool(spec, v)?,
                "llm_model" => {
                    let s = want_string(spec, v)?;
                    out.llm_model = Some(s).filter(|s| !s.trim().is_empty());
                }
                "planner_model" => {
                    let s = want_string(spec, v)?;
                    out.planner_model = Some(s).filter(|s| !s.trim().is_empty());
                }
                // Unreachable: the unknown-name sweep above already rejected
                // anything not in the catalogue. Kept as an error rather than a
                // no-op so that adding a catalogue entry without wiring it here
                // fails a test instead of silently doing nothing.
                other => return Err(format!("option \"{other}\" is not applied by the server")),
            }
        }

        Ok(out)
    }
}

// --------------------------------------------------------------- lane setup --

/// The search lanes a configuration can serve.
///
/// Shared with the CLI so a pinned engine means the same thing in both, including
/// the error text when the engine is unavailable.
pub fn select_lanes(
    engines: SearchEngines,
    obscura: Option<&Obscura>,
    jina: Option<&Jina>,
    deadline: Duration,
) -> Result<Vec<SearchLane>> {
    let mut lanes: Vec<SearchLane> = Vec::new();
    match engines {
        SearchEngines::Auto => {
            if let Some(o) = obscura {
                lanes.push(SearchLane::ddg(o.clone(), deadline));
            }
            if let Some(j) = jina {
                lanes.push(SearchLane::jina(j.clone(), deadline));
            }
        }
        SearchEngines::Ddg => match obscura {
            Some(o) => lanes.push(SearchLane::ddg(o.clone(), deadline)),
            None => {
                anyhow::bail!("search engine \"ddg\" needs obscura; it is not available here")
            }
        },
        SearchEngines::Jina => match jina {
            Some(j) => lanes.push(SearchLane::jina(j.clone(), deadline)),
            None => {
                anyhow::bail!("search engine \"jina\" needs a Jina key; none is configured here")
            }
        },
    }
    if lanes.is_empty() {
        anyhow::bail!("no search engine is available; install obscura or supply a Jina key");
    }
    Ok(lanes)
}

/// Which backend fetches pages. Jina wins when present, exactly as on the CLI.
pub fn select_backend(obscura: Option<&Obscura>, jina: Option<&Jina>) -> Result<Backend> {
    match (jina, obscura) {
        (Some(j), _) => Ok(Backend::Jina(j.clone())),
        (None, Some(o)) => Ok(Backend::Obscura(o.clone())),
        (None, None) => anyhow::bail!("no fetch backend is available"),
    }
}

// ---------------------------------------------------------------- run store --

/// A finished run, pre-rendered in every format the download endpoint offers.
///
/// Rendered once at completion rather than on demand, because re-running a search
/// to satisfy a download would be both slow and a different answer.
#[derive(Debug, Clone)]
struct CompletedRun {
    id: String,
    formats: BTreeMap<&'static str, String>,
}

/// The last `MAX_STORED_RUNS` completed runs, oldest evicted first.
#[derive(Debug, Default)]
struct RunStore {
    runs: VecDeque<CompletedRun>,
}

impl RunStore {
    fn insert(&mut self, run: CompletedRun) {
        while self.runs.len() >= MAX_STORED_RUNS {
            self.runs.pop_front();
        }
        self.runs.push_back(run);
    }

    fn get(&self, id: &str) -> Option<CompletedRun> {
        self.runs.iter().find(|r| r.id == id).cloned()
    }
}

/// Render every downloadable format for a finished report.
///
/// A format that fails to render is omitted rather than faked; the download
/// endpoint reports its absence instead of serving an error document that looks
/// like data.
fn render_all(report: &ScoutReport) -> BTreeMap<&'static str, String> {
    let mut out = BTreeMap::new();
    for name in FORMATS {
        let Some(fmt) = parse_format(name) else {
            continue;
        };
        match output::render(report, fmt) {
            Ok(body) => {
                out.insert(name, body);
            }
            Err(e) => tracing::warn!(format = name, error = %e, "could not render this format"),
        }
    }
    out
}

/// File extension for a download of this format.
fn extension_for(format: &str) -> &'static str {
    match format {
        "markdown" => "md",
        "csv" => "csv",
        "jsonl" => "jsonl",
        _ => "json",
    }
}

/// MIME type for a download of this format.
fn content_type_for(format: &str) -> &'static str {
    match format {
        "markdown" => "text/markdown; charset=utf-8",
        "csv" => "text/csv; charset=utf-8",
        "jsonl" => "application/x-ndjson",
        _ => "application/json",
    }
}

// ----------------------------------------------------------------- state --

/// Everything the server resolved once, at startup, and reuses for every run.
pub struct AppState {
    pub creds: Credentials,
    pub obscura: Option<Obscura>,
    pub jina: Option<Jina>,
    /// The engine pin the process was started with. A request that says `auto`
    /// inherits this; a request that names an engine overrides it.
    pub engines: SearchEngines,
    pub thinking_control: ThinkingControl,
    pub planner_thinking_control: ThinkingControl,
    pub fetch_concurrency: usize,
    runs: Mutex<RunStore>,
}

impl AppState {
    pub fn new(
        creds: Credentials,
        obscura: Option<Obscura>,
        jina: Option<Jina>,
        engines: SearchEngines,
        thinking_control: ThinkingControl,
        planner_thinking_control: ThinkingControl,
        fetch_concurrency: usize,
    ) -> Self {
        Self {
            creds,
            obscura,
            jina,
            engines,
            thinking_control,
            planner_thinking_control,
            fetch_concurrency,
            runs: Mutex::new(RunStore::default()),
        }
    }

    /// Build a scout for one request.
    ///
    /// Fresh clients every time, not clones: `Jev` and `Llm` share their counters
    /// through an `Arc`, so a cloned client would report the previous run's token
    /// spend as part of this one's.
    fn build_scout(
        &self,
        opts: &RunOptions,
        tx: tokio::sync::mpsc::Sender<ProgressEvent>,
    ) -> Result<Scout> {
        let tune = opts.tune.clone();
        let fetch_concurrency = opts
            .fetch_concurrency
            .unwrap_or(self.fetch_concurrency)
            .max(1);

        // Concurrency is a per-request knob, and both fetchers bake it in at
        // construction, so the handles are rebuilt rather than shared.
        let jina = match (&self.jina, &self.creds.jina_key) {
            (Some(_), Some(key)) => Some(Jina::new(
                key.clone(),
                fetch_concurrency,
                tune.page_timeout,
            )?),
            _ => None,
        };
        let obscura = self.obscura.as_ref().map(|o| {
            let mut o = o.clone();
            o.concurrency = fetch_concurrency;
            o.timeout = tune.page_timeout;
            o
        });

        let engines = match opts.engines {
            SearchEngines::Auto => self.engines,
            pinned => pinned,
        };
        let lanes = select_lanes(
            engines,
            obscura.as_ref(),
            jina.as_ref(),
            tune.search_lane_timeout,
        )?;
        let backend = select_backend(obscura.as_ref(), jina.as_ref())?;

        let search_cache = tune
            .search_cache
            .then(|| SearchCache::discover(tune.search_cache_ttl));

        let jev = Jev::new(
            self.creds.typesafe_endpoint.clone(),
            self.creds.typesafe_key.clone(),
            tune.max_retries,
            tune.http_timeout,
        )?;

        let writer_tc = opts.thinking_control.unwrap_or(self.thinking_control);
        let planner_tc = opts
            .planner_thinking_control
            .unwrap_or(self.planner_thinking_control);

        let llm = Llm::new(
            self.creds.llm_endpoint.clone(),
            self.creds.llm_key.clone(),
            opts.llm_model
                .clone()
                .unwrap_or_else(|| self.creds.llm_model.clone()),
            tune.max_retries,
            tune.http_timeout,
            writer_tc,
        )?;
        let planner = Llm::new(
            self.creds.planner_endpoint.clone(),
            self.creds.planner_key.clone(),
            opts.planner_model
                .clone()
                .unwrap_or_else(|| self.creds.planner_model.clone()),
            tune.max_retries,
            tune.http_timeout,
            planner_tc,
        )?;

        let fetcher = Fetcher::new(backend, Duration::from_secs(20))?
            .with_lanes(lanes)
            .with_search_cache(search_cache);

        Ok(Scout {
            auto: opts.auto,
            follow: opts.follow,
            enrich: opts.enrich,
            jev,
            llm,
            planner,
            fetcher,
            tune: std::sync::RwLock::new(tune),
            timings: Default::default(),
            today: crate::clock::today_utc(),
            progress: None,
            // The HTTP API does not expose pinned URLs yet; an empty list keeps
            // every request on the search-driven path exactly as before.
            seed_urls: Vec::new(),
        }
        .with_progress(tx))
    }
}

// -------------------------------------------------------------- responses --

fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    let message = message.into();
    (status, Json(json!({"type": "error", "message": message}))).into_response()
}

/// Serialise one NDJSON line, terminator included.
fn ndjson_line(v: &Value) -> String {
    match serde_json::to_string(v) {
        Ok(mut s) => {
            s.push('\n');
            s
        }
        // Serialising a `Value` cannot fail in practice, but a stream must always
        // emit something parseable rather than truncate mid-run.
        Err(e) => format!("{{\"type\":\"error\",\"message\":\"{e}\"}}\n"),
    }
}

fn progress_line(ev: &ProgressEvent) -> String {
    ndjson_line(&json!({
        "type": "progress",
        "stage": ev.stage,
        "round": ev.round,
        "message": ev.message,
        "counts": {"pages": ev.counts.pages, "records": ev.counts.records},
    }))
}

/// The live token-and-cost line.
///
/// Grouped by the actor that spent it, because that is the distinction that
/// matters here: `jev` judges, `llm` writes, `planner` plans.
fn usage_line(u: &UsageSnapshot, elapsed_ms: u64) -> String {
    ndjson_line(&json!({
        "type": "usage",
        "elapsed_ms": elapsed_ms,
        "jev": {
            "requests": u.jev_requests,
            "input_tokens": u.jev_input_tokens,
            "cost_usd": u.jev_cost_usd,
        },
        "llm": llm_usage_json(&u.llm),
        "planner": llm_usage_json(&u.planner),
    }))
}

fn stats_line(report: &ScoutReport) -> String {
    let st = &report.stats;
    ndjson_line(&json!({
        "type": "stats",
        "jev_requests": st.jev_requests,
        "llm_requests": st.llm_requests + st.planner_requests,
        "pages_fetched": st.pages_fetched,
        "records": report.records.len(),
        "elapsed_ms": (st.elapsed_secs * 1000.0).round() as u64,
    }))
}

fn result_line(report: &ScoutReport, format: Format, content: &str) -> String {
    ndjson_line(&json!({
        "type": "result",
        "outcome": report.outcome.as_str(),
        "format": format_name(format),
        "content": content,
        "stats": report.stats,
        "mission": report.mission,
    }))
}

/// The catalogue's name for a format. `Terminal` is never offered over HTTP, so it
/// falls back to markdown rather than inventing a fifth name for the UI to handle.
fn format_name(format: Format) -> &'static str {
    match format {
        Format::Json => "json",
        Format::Jsonl => "jsonl",
        Format::Csv => "csv",
        Format::Markdown | Format::Terminal => "markdown",
    }
}

// --------------------------------------------------------------- handlers --

async fn health() -> Response {
    Json(json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")})).into_response()
}

async fn options_handler() -> Response {
    Json(json!({"groups": catalogue()})).into_response()
}

#[derive(Deserialize)]
struct SearchRequest {
    #[serde(default)]
    query: String,
    #[serde(default)]
    options: Map<String, Value>,
}

/// What the NDJSON stream is walking through, one line at a time.
struct StreamState {
    rx: tokio::sync::mpsc::Receiver<ProgressEvent>,
    /// The live run. `None` once it has finished; dropping the whole state — which
    /// is what a client disconnect does — cancels it.
    fut: Option<Pin<Box<dyn std::future::Future<Output = Result<ScoutReport>> + Send>>>,
    pending: VecDeque<String>,
    format: Format,
    run_id: String,
    store: Arc<AppState>,
    /// Kept alive alongside the future so the scout (and its clients) drop exactly
    /// when the stream does. Also what the usage timer samples: the counters are
    /// atomics on these clients, so reading them mid-run costs nothing and takes
    /// no lock the run could be waiting on.
    scout: Arc<Scout>,
    /// Fires the live `usage` events. It lives *in* the stream state, so it is
    /// polled only while the stream is polled and is dropped with it — there is
    /// no spawned task to outlive a closed browser tab.
    usage_tick: tokio::time::Interval,
    /// Wall clock for `usage.elapsed_ms` before the report exists to supply it.
    started: std::time::Instant,
}

async fn search(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: SearchRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return json_error(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}"));
        }
    };

    let query = req.query.trim().to_string();
    if query.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "query must not be empty");
    }

    let opts = match RunOptions::from_map(&req.options) {
        Ok(o) => o,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, msg),
    };

    let (tx, rx) = tokio::sync::mpsc::channel(PROGRESS_BUFFER);
    let scout = match state.build_scout(&opts, tx) {
        Ok(s) => Arc::new(s),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, e.to_string()),
    };

    let run_id = new_run_id();
    tracing::info!(run_id = %run_id, query = %query, "api run starting");

    let fut = {
        let scout = scout.clone();
        let query = query.clone();
        Box::pin(async move { scout.run(&query).await })
            as Pin<Box<dyn std::future::Future<Output = Result<ScoutReport>> + Send>>
    };

    let mut pending = VecDeque::new();
    pending.push_back(ndjson_line(&json!({
        "type": "accepted",
        "run_id": run_id,
        "query": query,
        "started_at": now_rfc3339(),
    })));

    // `Burst` would fire every missed tick back to back after a slow consumer
    // catches up, which for a usage panel is a flood of identical lines. Skipping
    // to the next whole interval reports the same numbers once.
    let mut usage_tick = tokio::time::interval(USAGE_INTERVAL);
    usage_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let st = StreamState {
        rx,
        fut: Some(fut),
        pending,
        format: opts.format,
        run_id,
        store: state,
        scout,
        usage_tick,
        started: std::time::Instant::now(),
    };

    let stream = futures::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(line) = st.pending.pop_front() {
                return Some((Ok::<Bytes, std::convert::Infallible>(Bytes::from(line)), st));
            }
            // No future left and nothing queued: the run is over and so is the
            // stream. Exactly one terminal event has already been emitted.
            st.fut.as_ref()?;

            // Progress first (`biased`), so the last events before completion are
            // not overtaken by the result line.
            let done = {
                let fut = st.fut.as_mut().expect("checked just above");
                let rx = &mut st.rx;
                let tick = &mut st.usage_tick;
                tokio::select! {
                    biased;
                    Some(ev) = rx.recv() => { st.pending.push_back(progress_line(&ev)); None }
                    res = fut => Some(res),
                    // Last branch: a tick never delays the story or the ending.
                    _ = tick.tick() => {
                        let u = UsageSnapshot::sample(&st.scout);
                        let ms = st.started.elapsed().as_millis() as u64;
                        st.pending.push_back(usage_line(&u, ms));
                        None
                    }
                }
            };

            let Some(res) = done else { continue };
            st.fut = None;

            // Drain whatever the run emitted while we were awaiting its result, so
            // the story is complete before the ending is told.
            while let Ok(ev) = st.rx.try_recv() {
                st.pending.push_back(progress_line(&ev));
            }

            match res {
                Ok(report) => {
                    let formats = render_all(&report);
                    let wanted = format_name(st.format);
                    let content = formats.get(wanted).cloned().unwrap_or_default();
                    st.pending.push_back(stats_line(&report));
                    // Built from the report, not from another sample, so the
                    // last usage event and the result's stats are the same
                    // numbers by construction.
                    st.pending.push_back(usage_line(
                        &UsageSnapshot::from_stats(&report.stats),
                        (report.stats.elapsed_secs * 1000.0).round() as u64,
                    ));
                    st.pending
                        .push_back(result_line(&report, st.format, &content));
                    // Sync lock, sync body: never held across an await.
                    if let Ok(mut store) = st.store.runs.lock() {
                        store.insert(CompletedRun {
                            id: st.run_id.clone(),
                            formats,
                        });
                    }
                    tracing::info!(run_id = %st.run_id, outcome = report.outcome.as_str(), "api run finished");
                }
                Err(e) => {
                    tracing::warn!(run_id = %st.run_id, error = %e, "api run failed");
                    // A failed run still spent money; report it before the error.
                    st.pending.push_back(usage_line(
                        &UsageSnapshot::sample(&st.scout),
                        st.started.elapsed().as_millis() as u64,
                    ));
                    st.pending.push_back(ndjson_line(
                        &json!({"type": "error", "message": e.to_string()}),
                    ));
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        // Nothing downstream should buffer a progress stream into uselessness.
        .header(header::CACHE_CONTROL, "no-store")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[derive(Deserialize)]
struct DownloadQuery {
    format: Option<String>,
}

async fn download(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Query(q): Query<DownloadQuery>,
) -> Response {
    let format = q.format.unwrap_or_else(|| "markdown".to_string());
    if !FORMATS.contains(&format.as_str()) {
        return json_error(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown format \"{format}\"; expected one of {}",
                FORMATS.join(", ")
            ),
        );
    }

    let run = {
        let Ok(store) = state.runs.lock() else {
            return json_error(StatusCode::INTERNAL_SERVER_ERROR, "run store is poisoned");
        };
        store.get(&run_id)
    };
    let Some(run) = run else {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("no completed run with id \"{run_id}\""),
        );
    };

    let Some(body) = run.formats.get(format.as_str()) else {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("this run could not be rendered as {format}"),
        );
    };

    let filename = format!("webscout-{}.{}", run.id, extension_for(&format));
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type_for(&format).to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{filename}\""),
            ),
        ],
        body.clone(),
    )
        .into_response()
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/options", get(options_handler))
        .route("/api/search", post(search))
        .route("/api/runs/{run_id}/download", get(download))
        .with_state(state)
}

/// Bind and serve until the process is terminated.
///
/// `0.0.0.0` because the intended deployment is a container whose port is reached
/// from a sibling service; the compose file is what decides whether anything
/// outside the host network can see it.
pub async fn serve(port: u16, state: AppState) -> Result<()> {
    let app = router(Arc::new(state));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(port, "api listening on 0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ------------------------------------------------------------------ tests --

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    // --- run ids ---

    #[test]
    fn run_id_is_hex_of_clock_and_counter() {
        assert_eq!(format_run_id(255, 16), "ff-10");
        assert_eq!(format_run_id(0, 0), "0-0");
    }

    #[test]
    fn run_ids_differ_within_one_tick() {
        assert_ne!(format_run_id(42, 0), format_run_id(42, 1));
    }

    #[test]
    fn generated_run_ids_are_unique() {
        let a = new_run_id();
        let b = new_run_id();
        assert_ne!(a, b);
        assert!(a.contains('-'));
    }

    // --- timestamps ---

    #[test]
    fn rfc3339_formats_the_epoch() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn rfc3339_formats_a_known_instant() {
        assert_eq!(rfc3339(1_789_216_496), "2026-09-12T12:34:56Z");
        // A day boundary, where an off-by-one in the division would show.
        assert_eq!(rfc3339(86_399), "1970-01-01T23:59:59Z");
        assert_eq!(rfc3339(86_400), "1970-01-02T00:00:00Z");
    }

    #[test]
    fn now_rfc3339_has_the_right_shape() {
        let s = now_rfc3339();
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.ends_with('Z'), "{s}");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], "T");
    }

    // --- presets ---

    #[test]
    fn presets_round_trip() {
        for name in PRESETS {
            let p = Preset::parse(name).expect(name);
            assert_eq!(p.as_str(), name);
        }
        assert!(Preset::parse("blazing").is_none());
    }

    #[test]
    fn preset_tunables_match_the_config_functions() {
        assert_eq!(
            Preset::Quick.tunables().max_rounds,
            Tunables::quick().max_rounds
        );
        assert_eq!(
            Preset::Thorough.tunables().max_rounds,
            Tunables::thorough().max_rounds
        );
        assert_eq!(
            Preset::Standard.tunables().max_rounds,
            Tunables::default().max_rounds
        );
    }

    // --- catalogue ---

    /// Every option the shared API contract names, in the group it names.
    const SPEC_BASIC: [&str; 4] = ["preset", "max_rounds", "format", "search_engines"];
    const SPEC_ADVANCED: [&str; 27] = [
        "queries_per_round",
        "results_per_query",
        "read_per_query",
        "max_barren_rounds",
        "fetch_concurrency",
        "concurrency",
        "max_questions_per_request",
        "grounding_floor",
        "constraint_floor",
        "claim_floor",
        "currency_floor",
        "query_gate_floor",
        "follow_floor",
        "select_confidence",
        "enrich_batch",
        "enrich_read",
        "max_follow_per_page",
        "no_plan",
        "no_enrich",
        "no_follow",
        "no_auto",
        "no_search_cache",
        "search_cache_ttl",
        "llm_model",
        "planner_model",
        "thinking_control",
        "planner_thinking_control",
    ];

    #[test]
    fn catalogue_has_the_two_contract_groups() {
        let groups = catalogue();
        let ids: Vec<&str> = groups.iter().map(|g| g.id).collect();
        assert_eq!(ids, vec!["basic", "advanced"]);
    }

    #[test]
    fn every_contract_option_is_present_in_its_group() {
        let groups = catalogue();
        let basic: Vec<&str> = groups[0].options.iter().map(|o| o.name).collect();
        let advanced: Vec<&str> = groups[1].options.iter().map(|o| o.name).collect();
        for name in SPEC_BASIC {
            assert!(basic.contains(&name), "basic group is missing {name}");
        }
        for name in SPEC_ADVANCED {
            assert!(advanced.contains(&name), "advanced group is missing {name}");
        }
    }

    #[test]
    fn every_option_carries_a_non_empty_help() {
        for group in catalogue() {
            for opt in group.options {
                assert!(
                    !opt.help.trim().is_empty(),
                    "option {} has no help",
                    opt.name
                );
                assert!(
                    !opt.label.trim().is_empty(),
                    "option {} has no label",
                    opt.name
                );
            }
        }
    }

    #[test]
    fn enum_options_declare_their_values_and_default_to_one() {
        for group in catalogue() {
            for opt in group.options {
                if opt.kind != OptionType::Enum {
                    assert!(opt.values.is_none(), "{} should not list values", opt.name);
                    continue;
                }
                let values = opt.values.clone().unwrap_or_default();
                assert!(!values.is_empty(), "{} lists no values", opt.name);
                let default = opt.default.as_str().unwrap_or_default();
                assert!(
                    values.contains(&default),
                    "{} defaults to {default}, which is not one of its values",
                    opt.name
                );
            }
        }
    }

    #[test]
    fn numeric_options_declare_a_sane_range_containing_their_default() {
        for group in catalogue() {
            for opt in group.options {
                match opt.kind {
                    OptionType::Integer | OptionType::Number => {}
                    _ => {
                        assert!(opt.min.is_none() && opt.max.is_none(), "{}", opt.name);
                        continue;
                    }
                }
                let min = opt.min.as_ref().and_then(Value::as_f64).expect(opt.name);
                let max = opt.max.as_ref().and_then(Value::as_f64).expect(opt.name);
                let default = opt.default.as_f64().expect(opt.name);
                assert!(min <= max, "{} has min above max", opt.name);
                assert!(
                    (min..=max).contains(&default),
                    "{} defaults to {default}, outside {min}..={max}",
                    opt.name
                );
            }
        }
    }

    #[test]
    fn defaults_are_read_from_tunables_not_copied() {
        let index = spec_index();
        let d = Tunables::default();
        assert_eq!(index["max_rounds"].default, json!(d.max_rounds));
        assert_eq!(index["grounding_floor"].default, json!(d.grounding_floor));
        assert_eq!(index["enrich_batch"].default, json!(d.enrich_batch));
        assert_eq!(
            index["search_cache_ttl"].default,
            json!(d.search_cache_ttl.as_secs() as usize)
        );
    }

    #[test]
    fn no_option_names_a_credential_or_an_endpoint() {
        for group in catalogue() {
            for opt in group.options {
                let n = opt.name;
                assert!(!n.contains("key"), "{n} looks like a credential");
                assert!(!n.contains("endpoint"), "{n} looks like an endpoint");
                assert!(!n.contains("bin"), "{n} looks like a binary path");
            }
        }
    }

    #[test]
    fn options_payload_serialises_with_the_contract_shape() {
        let v = serde_json::to_value(json!({"groups": catalogue()})).unwrap();
        let first = &v["groups"][0]["options"][0];
        assert!(first["name"].is_string());
        assert!(first["label"].is_string());
        assert!(first["type"].is_string());
        assert!(first["help"].is_string());
        assert!(v["groups"][1]["options"].as_array().unwrap().len() > 10);
    }

    // --- validation ---

    #[test]
    fn empty_options_yield_the_standard_preset() {
        let o = RunOptions::from_map(&Map::new()).unwrap();
        assert_eq!(o.preset, Preset::Standard);
        assert_eq!(o.tune.max_rounds, Tunables::default().max_rounds);
        assert_eq!(o.format, Format::Markdown);
        assert!(o.auto && o.follow && o.enrich);
    }

    #[test]
    fn every_catalogue_option_is_applied() {
        // Feeding each option its own advertised default must be accepted. This is
        // what stops a catalogue entry existing with no code behind it.
        for group in catalogue() {
            for opt in group.options {
                let m = opts(&[(opt.name, opt.default.clone())]);
                RunOptions::from_map(&m)
                    .unwrap_or_else(|e| panic!("option {} was rejected: {e}", opt.name));
            }
        }
    }

    #[test]
    fn preset_is_applied_before_overrides_whatever_the_key_order() {
        let o = RunOptions::from_map(&opts(&[
            ("max_rounds", json!(7)),
            ("preset", json!("thorough")),
        ]))
        .unwrap();
        assert_eq!(o.preset, Preset::Thorough);
        assert_eq!(o.tune.max_rounds, 7);
        // Everything the override did not name still comes from the preset.
        assert_eq!(
            o.tune.queries_per_round,
            Tunables::thorough().queries_per_round
        );
    }

    #[test]
    fn unknown_option_is_named_in_the_error() {
        let err = RunOptions::from_map(&opts(&[("max_roundz", json!(3))])).unwrap_err();
        assert!(err.contains("max_roundz"), "{err}");
        assert!(err.contains("unknown option"), "{err}");
    }

    #[test]
    fn credentials_endpoints_and_binaries_are_rejected_as_unknown() {
        let err = RunOptions::from_map(&opts(&[
            ("llm_key", json!("sk-secret")),
            ("typesafe_endpoint", json!("https://evil.example/v1")),
            ("obscura_bin", json!("/tmp/pwn")),
        ]))
        .unwrap_err();
        assert!(err.contains("llm_key"), "{err}");
        assert!(err.contains("typesafe_endpoint"), "{err}");
        assert!(err.contains("obscura_bin"), "{err}");
        // And nothing leaked a value back.
        assert!(!err.contains("sk-secret"), "{err}");
    }

    #[test]
    fn several_unknown_options_are_all_named() {
        let err = RunOptions::from_map(&opts(&[("zzz", json!(1)), ("aaa", json!(1))])).unwrap_err();
        assert!(err.contains("aaa") && err.contains("zzz"), "{err}");
    }

    #[test]
    fn wrong_type_is_rejected_by_name() {
        let err = RunOptions::from_map(&opts(&[("max_rounds", json!("lots"))])).unwrap_err();
        assert!(err.contains("max_rounds"), "{err}");
        assert!(err.contains("whole number"), "{err}");
    }

    #[test]
    fn a_float_is_not_a_whole_number() {
        assert!(RunOptions::from_map(&opts(&[("max_rounds", json!(3.5))])).is_err());
    }

    #[test]
    fn a_boolean_is_not_a_number() {
        assert!(RunOptions::from_map(&opts(&[("grounding_floor", json!(true))])).is_err());
    }

    #[test]
    fn a_number_is_not_a_boolean() {
        assert!(RunOptions::from_map(&opts(&[("no_plan", json!(1))])).is_err());
    }

    #[test]
    fn out_of_range_names_the_bound_and_is_not_clamped() {
        let err = RunOptions::from_map(&opts(&[("max_rounds", json!(9999))])).unwrap_err();
        assert!(err.contains("max_rounds"), "{err}");
        assert!(err.contains("200"), "{err}");
        let err = RunOptions::from_map(&opts(&[("grounding_floor", json!(1.5))])).unwrap_err();
        assert!(err.contains("between 0 and 1"), "{err}");
    }

    #[test]
    fn zero_is_below_the_floor_for_a_count() {
        assert!(RunOptions::from_map(&opts(&[("max_rounds", json!(0))])).is_err());
        // But a link cap of zero is a legitimate "follow nothing".
        assert!(RunOptions::from_map(&opts(&[("max_follow_per_page", json!(0))])).is_ok());
    }

    #[test]
    fn negative_values_are_rejected() {
        assert!(RunOptions::from_map(&opts(&[("max_rounds", json!(-1))])).is_err());
        assert!(RunOptions::from_map(&opts(&[("grounding_floor", json!(-0.1))])).is_err());
    }

    #[test]
    fn unknown_enum_value_lists_the_allowed_ones() {
        let err = RunOptions::from_map(&opts(&[("search_engines", json!("bing"))])).unwrap_err();
        assert!(err.contains("bing"), "{err}");
        assert!(err.contains("ddg") && err.contains("jina"), "{err}");
    }

    #[test]
    fn terminal_is_not_an_offered_format() {
        assert!(RunOptions::from_map(&opts(&[("format", json!("terminal"))])).is_err());
    }

    #[test]
    fn negation_options_invert_the_flag() {
        let o = RunOptions::from_map(&opts(&[
            ("no_auto", json!(true)),
            ("no_follow", json!(true)),
            ("no_enrich", json!(true)),
            ("no_plan", json!(true)),
            ("no_search_cache", json!(true)),
        ]))
        .unwrap();
        assert!(!o.auto && !o.follow && !o.enrich);
        assert!(!o.tune.research_plan);
        assert!(!o.tune.search_cache);
    }

    #[test]
    fn model_names_are_accepted_and_blank_means_server_default() {
        let o = RunOptions::from_map(&opts(&[
            ("llm_model", json!("google/gemma-4-31b-it")),
            ("planner_model", json!("   ")),
        ]))
        .unwrap();
        assert_eq!(o.llm_model.as_deref(), Some("google/gemma-4-31b-it"));
        assert_eq!(o.planner_model, None);
    }

    #[test]
    fn thinking_controls_parse_through_the_same_value_enum_as_the_flag() {
        let o = RunOptions::from_map(&opts(&[
            ("thinking_control", json!("vllm")),
            ("planner_thinking_control", json!("off")),
        ]))
        .unwrap();
        assert_eq!(o.thinking_control, Some(ThinkingControl::Vllm));
        assert_eq!(o.planner_thinking_control, Some(ThinkingControl::Off));
    }

    #[test]
    fn engines_and_format_parse_to_their_enums() {
        let o = RunOptions::from_map(&opts(&[
            ("search_engines", json!("ddg")),
            ("format", json!("csv")),
        ]))
        .unwrap();
        assert_eq!(o.engines, SearchEngines::Ddg);
        assert_eq!(o.format, Format::Csv);
    }

    #[test]
    fn fetch_concurrency_falls_back_to_the_servers_choice() {
        let o = RunOptions::from_map(&Map::new()).unwrap();
        assert_eq!(o.fetch_concurrency, None);
        let o = RunOptions::from_map(&opts(&[("fetch_concurrency", json!(3))])).unwrap();
        assert_eq!(o.fetch_concurrency, Some(3));
        // The advertised default is still the number this process would use.
        assert_eq!(
            spec_index()["fetch_concurrency"].default,
            json!(default_fetch_concurrency())
        );
    }

    // --- lane selection ---

    #[test]
    fn pinning_an_unavailable_engine_names_what_is_missing() {
        let d = Duration::from_secs(1);
        let err = select_lanes(SearchEngines::Ddg, None, None, d)
            .unwrap_err()
            .to_string();
        assert!(err.contains("ddg") && err.contains("obscura"), "{err}");
        let err = select_lanes(SearchEngines::Jina, None, None, d)
            .unwrap_err()
            .to_string();
        assert!(err.contains("jina") && err.contains("key"), "{err}");
        assert!(select_lanes(SearchEngines::Auto, None, None, d).is_err());
        assert!(select_backend(None, None).is_err());
    }

    #[test]
    fn search_cache_ttl_becomes_a_duration() {
        let o = RunOptions::from_map(&opts(&[("search_cache_ttl", json!(60))])).unwrap();
        assert_eq!(o.tune.search_cache_ttl, Duration::from_secs(60));
    }

    // --- run store ---

    fn run(id: &str) -> CompletedRun {
        let mut formats = BTreeMap::new();
        formats.insert("markdown", format!("# {id}"));
        CompletedRun {
            id: id.to_string(),
            formats,
        }
    }

    #[test]
    fn store_returns_what_it_kept() {
        let mut s = RunStore::default();
        s.insert(run("a"));
        assert_eq!(s.get("a").unwrap().formats["markdown"], "# a");
        assert!(s.get("b").is_none());
    }

    #[test]
    fn store_evicts_the_oldest_past_the_cap() {
        let mut s = RunStore::default();
        for i in 0..MAX_STORED_RUNS + 5 {
            s.insert(run(&i.to_string()));
        }
        assert_eq!(s.runs.len(), MAX_STORED_RUNS);
        assert!(s.get("0").is_none(), "oldest should have been evicted");
        assert!(s.get("4").is_none());
        assert!(s.get("5").is_some(), "newest {MAX_STORED_RUNS} are kept");
        assert!(s.get(&(MAX_STORED_RUNS + 4).to_string()).is_some());
    }

    // --- download metadata ---

    #[test]
    fn extensions_and_types_cover_every_offered_format() {
        for f in FORMATS {
            assert!(!extension_for(f).is_empty());
            assert!(content_type_for(f).contains('/'));
        }
        assert_eq!(extension_for("markdown"), "md");
        assert_eq!(extension_for("jsonl"), "jsonl");
        assert_eq!(content_type_for("csv"), "text/csv; charset=utf-8");
    }

    #[test]
    fn every_offered_format_parses_to_a_renderer() {
        for f in FORMATS {
            let parsed = parse_format(f).unwrap_or_else(|| panic!("{f} has no renderer"));
            assert_eq!(format_name(parsed), f);
        }
    }

    // --- ndjson lines ---

    #[test]
    fn ndjson_lines_end_in_exactly_one_newline() {
        let line = ndjson_line(&json!({"type": "accepted"}));
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
    }

    #[test]
    fn progress_line_carries_the_contract_fields() {
        let ev = ProgressEvent {
            stage: "search".into(),
            round: Some(1),
            message: "searching round 1".into(),
            counts: ProgressCounts {
                pages: 0,
                records: 0,
            },
        };
        let v: Value = serde_json::from_str(progress_line(&ev).trim()).unwrap();
        assert_eq!(v["type"], "progress");
        assert_eq!(v["stage"], "search");
        assert_eq!(v["round"], 1);
        assert_eq!(v["message"], "searching round 1");
        assert_eq!(v["counts"]["pages"], 0);
        assert_eq!(v["counts"]["records"], 0);
    }

    // --- usage lines ---

    fn sample_usage() -> UsageSnapshot {
        UsageSnapshot {
            jev_requests: 12,
            jev_input_tokens: 88_213,
            jev_cost_usd: 0.0037,
            llm: LlmUsage {
                requests: 3,
                prompt_tokens: 14_788,
                completion_tokens: 1_190,
                reasoning_tokens: 0,
                cost_usd: Some(0.0021),
            },
            planner: LlmUsage {
                requests: 1,
                prompt_tokens: 2_288,
                completion_tokens: 4_177,
                reasoning_tokens: 3_900,
                cost_usd: Some(0.0104),
            },
        }
    }

    #[test]
    fn usage_line_carries_the_contract_fields() {
        let v: Value = serde_json::from_str(usage_line(&sample_usage(), 41_000).trim()).unwrap();
        assert_eq!(v["type"], "usage");
        assert_eq!(v["elapsed_ms"], 41_000);
        assert_eq!(v["jev"]["requests"], 12);
        assert_eq!(v["jev"]["input_tokens"], 88_213);
        assert_eq!(v["jev"]["cost_usd"], 0.0037);
        assert_eq!(v["llm"]["requests"], 3);
        assert_eq!(v["llm"]["prompt_tokens"], 14_788);
        assert_eq!(v["llm"]["completion_tokens"], 1_190);
        assert_eq!(v["planner"]["requests"], 1);
        assert_eq!(v["planner"]["prompt_tokens"], 2_288);
        assert_eq!(v["planner"]["completion_tokens"], 4_177);
    }

    #[test]
    fn usage_line_is_one_ndjson_line() {
        let line = usage_line(&UsageSnapshot::default(), 0);
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
    }

    #[test]
    fn a_cost_the_endpoint_never_reported_is_absent_not_zero() {
        let mut u = sample_usage();
        u.llm.cost_usd = None;
        let v: Value = serde_json::from_str(usage_line(&u, 1).trim()).unwrap();
        assert!(
            v["llm"].get("cost_usd").is_none(),
            "an unknown cost must not be published as 0"
        );
        // A reported zero, on the other hand, is a fact worth sending.
        u.llm.cost_usd = Some(0.0);
        let v: Value = serde_json::from_str(usage_line(&u, 1).trim()).unwrap();
        assert_eq!(v["llm"]["cost_usd"], 0.0);
    }

    #[test]
    fn reasoning_tokens_appear_only_when_some_were_spent() {
        let u = sample_usage();
        let v: Value = serde_json::from_str(usage_line(&u, 1).trim()).unwrap();
        assert!(v["llm"].get("reasoning_tokens").is_none());
        assert_eq!(v["planner"]["reasoning_tokens"], 3_900);
    }

    #[test]
    fn the_final_usage_event_and_the_result_stats_agree() {
        // The last usage line is built from the report, so the panel a person
        // is left looking at cannot disagree with the terminal event.
        let stats = crate::types::Stats {
            jev_requests: 9,
            jev_input_tokens: 1_234,
            jev_cost_usd: 0.0012,
            llm_requests: 4,
            llm_prompt_tokens: 500,
            llm_completion_tokens: 60,
            llm_reasoning_tokens: 12,
            llm_cost_usd: Some(0.0009),
            planner_requests: 2,
            planner_prompt_tokens: 70,
            planner_completion_tokens: 8,
            planner_cost_usd: None,
            ..Default::default()
        };

        let line: Value =
            serde_json::from_str(usage_line(&UsageSnapshot::from_stats(&stats), 1_000).trim())
                .unwrap();
        let from_result = serde_json::to_value(&stats).unwrap();

        assert_eq!(line["jev"]["requests"], from_result["jev_requests"]);
        assert_eq!(line["jev"]["input_tokens"], from_result["jev_input_tokens"]);
        assert_eq!(line["jev"]["cost_usd"], from_result["jev_cost_usd"]);
        assert_eq!(line["llm"]["requests"], from_result["llm_requests"]);
        assert_eq!(
            line["llm"]["prompt_tokens"],
            from_result["llm_prompt_tokens"]
        );
        assert_eq!(
            line["llm"]["completion_tokens"],
            from_result["llm_completion_tokens"]
        );
        assert_eq!(line["llm"]["cost_usd"], from_result["llm_cost_usd"]);
        assert_eq!(
            line["llm"]["reasoning_tokens"],
            from_result["llm_reasoning_tokens"]
        );
        assert_eq!(line["planner"]["requests"], from_result["planner_requests"]);
        assert!(line["planner"].get("cost_usd").is_none());
    }

    #[test]
    fn a_planner_that_never_ran_reports_zeroes_the_ui_can_hide() {
        let v: Value =
            serde_json::from_str(usage_line(&UsageSnapshot::default(), 0).trim()).unwrap();
        assert_eq!(v["planner"]["requests"], 0);
        assert!(v["planner"].get("cost_usd").is_none());
    }

    #[test]
    fn the_usage_interval_is_about_a_second() {
        assert_eq!(USAGE_INTERVAL, Duration::from_secs(1));
    }

    // --- endpoints, driven in-process ---

    /// Credentials that go nowhere. No handler under test opens a connection:
    /// health, options and download never touch them, and every `/api/search`
    /// case here is rejected before a client is built.
    fn test_state() -> Arc<AppState> {
        Arc::new(AppState::new(
            Credentials {
                typesafe_key: "test-key".into(),
                typesafe_endpoint: "https://api.example/v1/systemone".into(),
                llm_key: "test-key".into(),
                llm_endpoint: "https://api.example/v1/chat/completions".into(),
                llm_model: "test-model".into(),
                planner_model: "test-model".into(),
                planner_endpoint: "https://api.example/v1/chat/completions".into(),
                planner_key: "test-key".into(),
                jina_key: None,
            },
            None,
            None,
            SearchEngines::Auto,
            ThinkingControl::Off,
            ThinkingControl::Off,
            4,
        ))
    }

    async fn call(state: Arc<AppState>, req: axum::http::Request<Body>) -> (StatusCode, String) {
        use tower::ServiceExt;
        let res = router(state).oneshot(req).await.expect("router call");
        let status = res.status();
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, String::from_utf8_lossy(&body).into_owned())
    }

    fn get(uri: &str) -> axum::http::Request<Body> {
        axum::http::Request::builder()
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    fn post_json(uri: &str, body: &str) -> axum::http::Request<Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn health_reports_ok_and_the_crate_version() {
        let (status, body) = call(test_state(), get("/api/health")).await;
        assert_eq!(status, StatusCode::OK);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn options_endpoint_serves_the_catalogue() {
        let (status, body) = call(test_state(), get("/api/options")).await;
        assert_eq!(status, StatusCode::OK);
        let v: Value = serde_json::from_str(&body).unwrap();
        let groups = v["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        for group in groups {
            for opt in group["options"].as_array().unwrap() {
                assert!(!opt["help"].as_str().unwrap().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn an_empty_query_is_rejected() {
        let (status, body) =
            call(test_state(), post_json("/api/search", r#"{"query":"  "}"#)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("query must not be empty"), "{body}");
    }

    #[tokio::test]
    async fn a_malformed_body_is_a_400_not_a_422() {
        let (status, body) = call(test_state(), post_json("/api/search", "{not json")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("invalid JSON body"), "{body}");
    }

    #[tokio::test]
    async fn credentials_in_the_request_body_are_a_400() {
        let (status, body) = call(
            test_state(),
            post_json(
                "/api/search",
                r#"{"query":"who is the CEO of Vodafone","options":{
                     "llm_key":"sk-secret",
                     "typesafe_endpoint":"https://evil.example/v1",
                     "obscura_bin":"/tmp/pwn"}}"#,
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("llm_key"), "{body}");
        assert!(body.contains("typesafe_endpoint"), "{body}");
        assert!(body.contains("obscura_bin"), "{body}");
        assert!(!body.contains("sk-secret"), "{body}");
    }

    #[tokio::test]
    async fn an_out_of_range_option_is_a_400_naming_the_bound() {
        let (status, body) = call(
            test_state(),
            post_json(
                "/api/search",
                r#"{"query":"x","options":{"max_rounds":9999}}"#,
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.contains("max_rounds") && body.contains("200"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn download_of_an_unknown_run_is_a_404() {
        let (status, body) = call(test_state(), get("/api/runs/nope/download?format=json")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("nope"), "{body}");
    }

    #[tokio::test]
    async fn download_of_an_unknown_format_is_a_400() {
        let (status, body) = call(test_state(), get("/api/runs/nope/download?format=pdf")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("pdf"), "{body}");
    }

    #[tokio::test]
    async fn download_serves_a_stored_run_as_an_attachment() {
        let state = test_state();
        {
            let mut store = state.runs.lock().unwrap();
            let mut formats = BTreeMap::new();
            formats.insert("markdown", "# hello".to_string());
            formats.insert("csv", "a,b\n".to_string());
            store.insert(CompletedRun {
                id: "abc-1".into(),
                formats,
            });
        }
        use tower::ServiceExt;
        let res = router(state.clone())
            .oneshot(get("/api/runs/abc-1/download?format=csv"))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let disposition = res.headers()[header::CONTENT_DISPOSITION].to_str().unwrap();
        assert_eq!(disposition, "attachment; filename=\"webscout-abc-1.csv\"");
        assert!(
            res.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/csv")
        );
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"a,b\n");
    }

    #[tokio::test]
    async fn download_defaults_to_markdown() {
        let state = test_state();
        {
            let mut store = state.runs.lock().unwrap();
            let mut formats = BTreeMap::new();
            formats.insert("markdown", "# hello".to_string());
            store.insert(CompletedRun {
                id: "abc-2".into(),
                formats,
            });
        }
        let (status, body) = call(state, get("/api/runs/abc-2/download")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "# hello");
    }

    #[test]
    fn a_multiline_message_stays_one_ndjson_line() {
        let ev = ProgressEvent {
            stage: "extract".into(),
            round: None,
            message: "line one\nline two".into(),
            counts: ProgressCounts::default(),
        };
        let line = progress_line(&ev);
        assert_eq!(line.matches('\n').count(), 1, "{line}");
    }
}
