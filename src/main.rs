//! webscout — search the web and return verified results.
//!
//! A generative model writes (queries, extraction, prose) and TypeSafe's Jev
//! verifies (relevance, injection screening, grounding). Obscura does the fetching,
//! driven as a subprocess. The point of the arrangement is that nothing
//! the generative model produces reaches the output without being checked against
//! the page it supposedly came from.
//!
//! Designed to be driven by another agent as much as by a person: the payload goes
//! to stdout in whatever format you ask for, and every log line goes to stderr, so
//! `webscout -f json "..." > out.json` yields a clean document regardless of
//! verbosity.

mod api;
mod browser;
mod candidates;
mod clock;
mod config;
mod llm;
mod output;
mod scout;
mod search_cache;
mod types;
mod typesafe;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::browser::{Fetcher, Jina, Obscura};
use crate::config::Tunables;
use crate::config::{Credentials, SearchEngines, Sources};
use crate::llm::{Llm, ThinkingControl};
use crate::output::Format;
use crate::scout::Scout;
use crate::search_cache::SearchCache;
use crate::types::Outcome;
use crate::typesafe::Jev;

#[derive(Parser, Debug)]
#[command(
    name = "webscout",
    version,
    about = "Search the web and return verified results.",
    long_about = "Search the web and return verified results.\n\n\
        A generative model plans searches, extracts records, and writes prose. \
        TypeSafe's Jev verifies every step: which results are worth reading, \
        whether a page is trying to address the model, and whether each extracted \
        value actually appears in its source. Obscura renders the pages and is \
        driven as a subprocess, so `obscura` and `obscura-worker` must be on PATH.\n\n\
        Results go to stdout; logs go to stderr.",
    after_help = "EXAMPLES:\n  \
        webscout \"what is the current stable release of Go\"\n  \
        webscout -f csv \"at least 100 cooperative names with their email addresses\"\n  \
        webscout -vv --thorough -f json \"EU AI Act obligations for GPAI providers\" > out.json\n  \
        webscout -f jsonl --max-rounds 100 \"housing co-ops in Catalonia with contact emails\""
)]
struct Cli {
    /// What to search for. Plain language; a request for a list is detected
    /// automatically, including any quantity it names.
    ///
    /// Optional only under `--api`, where the queries arrive over HTTP instead.
    /// Passing both is rejected rather than guessed at.
    #[arg(num_args = 1..)]
    query: Vec<String>,

    /// Serve the HTTP API instead of running one query.
    ///
    /// Credentials are resolved once at startup, exactly as for a CLI run, and are
    /// never settable per request: a request may change search depth, thresholds,
    /// model names and output format, and nothing else.
    #[arg(long)]
    api: bool,

    /// Port the API binds on 0.0.0.0. Falls back to $WEBSCOUT_API_PORT.
    #[arg(
        long,
        value_name = "PORT",
        env = "WEBSCOUT_API_PORT",
        default_value_t = crate::api::DEFAULT_API_PORT
    )]
    api_port: u16,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "terminal")]
    format: Format,

    /// Write results here instead of stdout.
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,

    /// Increase verbosity: -v info, -vv debug, -vvv trace. Logs go to stderr.
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Set the log level explicitly. Overrides -v.
    #[arg(long, value_parser = ["off", "error", "warn", "info", "debug", "trace"])]
    log_level: Option<String>,

    /// Emit logs as JSON objects, one per line — for agents that parse progress.
    #[arg(long)]
    log_json: bool,

    /// Search harder: more queries, more results, more pages, more rounds.
    #[arg(long, conflicts_with = "quick")]
    thorough: bool,

    /// Take a quick look instead of digging.
    #[arg(long, conflicts_with = "thorough")]
    quick: bool,

    /// Maximum search rounds, or `auto` (the default) to let the run decide.
    ///
    /// `auto` stops the run when it has what it came for or when progress
    /// levels off, with the preset's ceiling kept only as a safety net. A
    /// number is a hard ceiling honoured exactly; the run still stops early
    /// when it is done, but never for diminishing returns. There is no time
    /// limit either way.
    #[arg(long, value_name = "N|auto", value_parser = parse_round_limit)]
    max_rounds: Option<RoundLimit>,

    /// Stop after this many consecutive rounds that find nothing new.
    #[arg(long)]
    max_barren_rounds: Option<usize>,

    /// Search queries per round.
    #[arg(long)]
    queries_per_round: Option<usize>,

    /// Results requested per search query.
    #[arg(long)]
    results_per_query: Option<usize>,

    /// Pages fetched per query, after triage ranks them.
    #[arg(long)]
    read_per_query: Option<usize>,

    /// Path to the obscura binary.
    ///
    /// webscout drives obscura as a subprocess rather than linking it in. Obscura
    /// embeds V8, which is not safe to drive from several threads in one process —
    /// linked in, every page render serialises at roughly eight seconds each. Run as
    /// a binary it parallelises across worker processes: 24x faster on the same
    /// batch, with identical output.
    #[arg(long, default_value = "obscura")]
    obscura_bin: String,

    /// Pages rendered in parallel within one batch. Also the politeness knob.
    ///
    /// Defaults to your core count, clamped to 4..=16, because rendering is
    /// CPU-bound: on a 12-core box, 32 pages took 74s at concurrency 4, 32s at 8 and
    /// 29s at 32 — the gain past the core count is noise. Nothing caps what you pass
    /// here; raise it if your pages are light or your machine is larger.
    #[arg(long, default_value_t = crate::config::default_fetch_concurrency())]
    fetch_concurrency: usize,

    /// Concurrent verification requests.
    ///
    /// No ceiling is imposed here. Measured: Jev's latency is flat at roughly 0.9s
    /// however many run at once, but throughput plateaus near 4 requests/second, so
    /// raising this past ~8 buys little. Fewer, fatter requests is the lever that
    /// actually works — see --max-questions-per-request.
    #[arg(long)]
    concurrency: Option<usize>,

    /// Questions packed into a single verification request.
    ///
    /// 192 questions answered in 0.98s in testing, against 0.80s for 16 — count is
    /// nearly free, requests are not. The token budget still caps an oversized
    /// request regardless of this value.
    #[arg(long)]
    max_questions_per_request: Option<usize>,

    /// Minimum confidence that an extracted value really appears in its source.
    /// Lower it to keep more and verify by hand; raise it to keep only the certain.
    #[arg(long)]
    grounding_floor: Option<f64>,

    /// Turn off obscura's stealth mode.
    ///
    /// Stealth is on by default: a consistent browser fingerprint plus tracker
    /// blocking. Measured on sites that do not block, it neither helped nor hurt —
    /// page yield and timing were the same within noise. It is the default because
    /// reading sites that push back is the reason to choose obscura over a hosted
    /// reader at all, and paying nothing for better odds there is the right trade.
    /// Ignored when fetching through Jina.
    #[arg(long)]
    no_stealth: bool,

    /// Honour robots.txt when fetching pages. Off by default, matching obscura.
    #[arg(long)]
    obey_robots: bool,

    /// TypeSafe API key. Falls back to $TYPESAFE_API_KEY.
    ///
    /// Nothing is compiled in: a key baked into a binary shows up in `strings`,
    /// cannot be rotated without a rebuild, and makes the binary itself the secret.
    /// A flag beats the environment, so a one-off run can override your shell.
    #[arg(long, value_name = "KEY")]
    typesafe_key: Option<String>,

    /// TypeSafe endpoint. Falls back to $TYPESAFE_ENDPOINT.
    #[arg(long, value_name = "URL")]
    typesafe_endpoint: Option<String>,

    /// Generative model API key. Falls back to $WEBSCOUT_LLM_API_KEY.
    #[arg(long, value_name = "KEY")]
    llm_key: Option<String>,

    /// Generative model endpoint. Falls back to $WEBSCOUT_LLM_ENDPOINT.
    ///
    /// Must be the full OpenAI-compatible chat completions URL, e.g.
    /// https://openrouter.ai/api/v1/chat/completions. A wrong URL (base URL, missing
    /// path segment, etc.) is rejected at startup with the corrected form suggested.
    #[arg(long, value_name = "URL")]
    llm_endpoint: Option<String>,

    /// Generative model name. Falls back to $WEBSCOUT_LLM_MODEL.
    #[arg(long, value_name = "MODEL")]
    llm_model: Option<String>,

    /// Planner model name. Falls back to $WEBSCOUT_PLANNER_MODEL, then to
    /// `--llm-model`. Used for mission parsing, research planning, query
    /// writing, re-aim and review — the reasoning-shaped calls that
    /// benefit from a larger thinking model. The writer still handles
    /// extraction, enrichment templates and answer synthesis, which are
    /// mechanical and cheap.
    #[arg(long, value_name = "MODEL")]
    planner_model: Option<String>,

    /// Planner endpoint. Falls back to $WEBSCOUT_PLANNER_ENDPOINT, then to
    /// `--llm-endpoint`. Lets you route planning calls to a different host (e.g. a
    /// self-hosted reasoning model) while the writer uses another. Must be the full
    /// OpenAI-compatible chat completions URL, e.g.
    /// https://openrouter.ai/api/v1/chat/completions. A wrong URL is rejected at
    /// startup with the corrected form suggested.
    #[arg(long, value_name = "URL")]
    planner_endpoint: Option<String>,

    /// API key for the planner endpoint. Falls back to $WEBSCOUT_PLANNER_KEY,
    /// then to `--llm-key`. Required only when the planner endpoint uses a
    /// different key than the writer.
    #[arg(long, value_name = "KEY")]
    planner_key: Option<String>,

    /// How to control reasoning / thinking on the writer endpoint.
    ///
    /// `auto` (default): uses the OpenRouter form for openrouter.ai, nothing
    /// for all other hosts. `openrouter`: `reasoning.effort/enabled`. `vllm`:
    /// `chat_template_kwargs.enable_thinking` — measured 2026-09-18 on the
    /// self-hosted vLLM deployment, switching this off cut a realistic extraction
    /// call from 58.3 s (7,713 completion tokens, 23,631 chars of reasoning)
    /// to 1.4 s (48 completion tokens) with identical output. `effort`:
    /// `reasoning_effort: "medium"/"none"`. `off`: send no reasoning parameter.
    #[arg(long, env = "WEBSCOUT_THINKING_CONTROL", default_value = "auto")]
    thinking_control: ThinkingControl,

    /// How to control reasoning on the planner endpoint. Falls back to
    /// $WEBSCOUT_PLANNER_THINKING_CONTROL, then to `--thinking-control`.
    #[arg(long, env = "WEBSCOUT_PLANNER_THINKING_CONTROL")]
    planner_thinking_control: Option<ThinkingControl>,

    /// Jina API key. Falls back to $JINA_API_KEY.
    ///
    /// Supplying one switches fetching from a local obscura to Jina's hosted search
    /// and reader. That moves page rendering off this machine — the largest cost in
    /// most runs, and CPU-bound — in exchange for a metered dependency. Both handle
    /// JavaScript-rendered pages; without a key, obscura is used.
    #[arg(long, value_name = "KEY")]
    jina_key: Option<String>,

    /// Turn off link-following (Package B).
    ///
    /// Links on a listing page ("next", other member profiles) are Jev-judged
    /// and queued ahead of search results in the following round. This is the
    /// difference between finding one page of a register and finding all of
    /// them; disable only for narrow, cost-bounded runs.
    #[arg(long)]
    no_follow: bool,

    /// Read this URL directly instead of searching for the answer. Repeatable.
    ///
    /// On an answer mission the given pages are the whole evidence base: their
    /// chunks are injection-screened like any other page's (a page you name
    /// gets no exemption from being checked) and no search runs, so the answer
    /// comes from these pages or the run reports what is missing. On a harvest
    /// they are read first, as depth-1 seeds, and search continues beside them.
    #[arg(long = "url", value_name = "URL", action = clap::ArgAction::Append)]
    url: Vec<String>,

    /// Turn off enrichment (Package B).
    ///
    /// Enrichment searches for entities that were discovered without every
    /// requested field and reads their official pages. Off for a run that
    /// wants only what listing pages already stated.
    #[arg(long)]
    no_enrich: bool,

    /// Entities enriched per round. Overrides the profile default.
    #[arg(long)]
    enrich_batch: Option<usize>,

    /// Turn off the research plan.
    ///
    /// Once per harvest one planner call enumerates the sources most likely
    /// to publish a complete list (partners pages, registries, federations,
    /// directories, sponsor lists, GitHub organisations, Wikipedia
    /// categories) and Jev scores each source by likely completeness. The
    /// resulting queries plus seed URLs drive discovery in preference to
    /// the round-by-round LLM planner. Disable to use the LLM planner
    /// exclusively.
    #[arg(long)]
    no_plan: bool,

    /// Turn off automatic steering.
    ///
    /// By default the model reviews each round and decides whether to keep going,
    /// how deep to read, and whether the request is actually fulfilled — then tunes
    /// the search accordingly. The fixed rules it replaces are blunt: three barren
    /// rounds and a round ceiling suit a fact lookup and badly underserve "two
    /// hundred municipalities with contact addresses", where the right depth only
    /// becomes apparent once the first rounds come back.
    ///
    /// Its proposals are clamped, so a bad decision costs a round, not your budget.
    /// Turn this off for strictly predictable cost.
    #[arg(long)]
    no_auto: bool,

    /// Which search engines to use.
    ///
    /// `auto` (default) runs every lane that is available: DuckDuckGo
    /// whenever obscura is configured, Jina whenever a Jina key is present.
    /// Two engines agreeing on a URL is a free prior that costs no Jev
    /// tokens. Pin this to one engine to compare them on the same mission.
    ///
    /// Measured 2026-09-18: these are the only two engines that return usable
    /// results. Bing scraped through obscura returns unrelated content behind
    /// `bing.com/ck/a?` redirects, and Brave, Mojeek, Startpage and Ecosia all
    /// return interstitials or near-empty bodies.
    #[arg(
        long,
        value_enum,
        env = "WEBSCOUT_SEARCH_ENGINES",
        default_value = "auto"
    )]
    search_engines: SearchEngines,

    /// Do not read or write the on-disk search cache.
    ///
    /// Successful, non-empty responses are otherwise cached per engine and
    /// query under $XDG_CACHE_HOME/webscout/search, which makes a re-run of
    /// the same mission skip its searches entirely.
    #[arg(long)]
    no_search_cache: bool,

    /// How long a cached search response stays usable, in seconds.
    /// Defaults to 21600 (6 hours). Falls back to $WEBSCOUT_SEARCH_CACHE_TTL.
    #[arg(long, value_name = "SECONDS", env = "WEBSCOUT_SEARCH_CACHE_TTL")]
    search_cache_ttl: Option<u64>,

    /// Print a per-stage timing table when the run finishes.
    ///
    /// Goes to stderr, so it never contaminates the payload. Stage times sum to more
    /// than the wall clock because stages overlap — the excess is what the
    /// concurrency and round pipelining are saving you.
    #[arg(long)]
    profile: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(&cli);

    // rustls needs a crypto provider chosen before first use. We pick ring rather
    // than the default aws-lc-rs because ring is pure Rust, which keeps the whole
    // dependency tree free of C and lets this build as a static musl binary.
    if rustls::crypto::ring::default_provider()
        .install_default()
        .is_err()
    {
        tracing::debug!("a rustls crypto provider was already installed");
    }

    // A multi-thread runtime for the orchestration. Browser work happens in
    // obscura's own processes, so nothing here is pinned to a thread.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;

    let code = rt.block_on(run(cli))?;
    std::process::exit(code);
}

/// What `--max-rounds` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundLimit {
    Auto,
    Fixed(usize),
}

/// `auto`, or a whole number of rounds of at least one. Zero is rejected
/// rather than read as "no limit": the way to ask for no fixed limit is
/// `auto`, which still keeps a safety ceiling.
fn parse_round_limit(raw: &str) -> Result<RoundLimit, String> {
    let t = raw.trim();
    if t.eq_ignore_ascii_case("auto") {
        return Ok(RoundLimit::Auto);
    }
    match t.parse::<usize>() {
        Ok(0) => Err("must be at least 1, or `auto`".into()),
        Ok(n) => Ok(RoundLimit::Fixed(n)),
        Err(_) => Err(format!("`{t}` is not a number of rounds or `auto`")),
    }
}

/// Reject the two argument shapes that cannot mean anything.
///
/// Split out from `run` so the rule is testable without a runtime: a mode switch
/// that is only exercised by starting a server is a mode switch nobody tests.
fn check_query_args(api: bool, query: &[String]) -> Result<()> {
    match (api, query.is_empty()) {
        (true, false) => anyhow::bail!(
            "--api conflicts with a query argument: in API mode queries arrive over HTTP. \
             Drop the query, or drop --api to run it once here."
        ),
        (false, true) => anyhow::bail!(
            "no query given. Pass what to search for, or pass --api to serve the HTTP API."
        ),
        _ => Ok(()),
    }
}

async fn run(cli: Cli) -> Result<i32> {
    check_query_args(cli.api, &cli.query)?;
    let query = cli.query.join(" ");

    let mut tune = if cli.thorough {
        Tunables::thorough()
    } else if cli.quick {
        Tunables::quick()
    } else {
        Tunables::default()
    };

    match cli.max_rounds {
        Some(RoundLimit::Fixed(v)) => {
            tune.max_rounds = v;
            tune.auto_rounds = false;
        }
        // `auto` keeps the preset's ceiling as the safety net.
        Some(RoundLimit::Auto) | None => tune.auto_rounds = true,
    }
    if let Some(v) = cli.max_barren_rounds {
        tune.max_barren_rounds = v;
    }
    if let Some(v) = cli.queries_per_round {
        tune.queries_per_round = v;
    }
    if let Some(v) = cli.results_per_query {
        tune.results_per_query = v;
    }
    if let Some(v) = cli.read_per_query {
        tune.read_per_query = v;
    }
    if let Some(v) = cli.concurrency {
        tune.concurrency = v;
    }
    if let Some(v) = cli.max_questions_per_request {
        tune.max_questions_per_request = v;
    }
    if let Some(v) = cli.grounding_floor {
        tune.grounding_floor = v;
    }
    if let Some(v) = cli.enrich_batch {
        tune.enrich_batch = v;
    }
    if cli.no_plan {
        tune.research_plan = false;
    }
    if cli.no_search_cache {
        tune.search_cache = false;
    }
    if let Some(v) = cli.search_cache_ttl {
        tune.search_cache_ttl = std::time::Duration::from_secs(v);
    }

    if cli.api {
        tracing::info!("starting in API mode");
    } else {
        tracing::info!(query = %query, "starting");
    }
    tracing::debug!(?tune, "tunables");

    let creds = Credentials::resolve(&Sources {
        typesafe_key: cli.typesafe_key.as_deref(),
        typesafe_endpoint: cli.typesafe_endpoint.as_deref(),
        llm_key: cli.llm_key.as_deref(),
        llm_endpoint: cli.llm_endpoint.as_deref(),
        llm_model: cli.llm_model.as_deref(),
        planner_model: cli.planner_model.as_deref(),
        planner_endpoint: cli.planner_endpoint.as_deref(),
        planner_key: cli.planner_key.as_deref(),
        jina_key: cli.jina_key.as_deref(),
    })?;

    // Fetching and searching are separate decisions. A Jina key still means
    // "fetch through Jina"; it no longer means "and therefore stop asking
    // DuckDuckGo", which is what the old either/or did — a key silently cost
    // the run its second engine.
    let jina = match creds.jina_key.clone() {
        Some(key) => Some(Jina::new(key, cli.fetch_concurrency, tune.page_timeout)?),
        None => None,
    };

    // Obscura is required only when something actually needs it: it fetches
    // when there is no Jina key, and it drives the DuckDuckGo lane. A Jina
    // user who pinned --search-engines jina never has to install a browser.
    let needs_obscura = jina.is_none() || cli.search_engines != SearchEngines::Jina;
    let obscura = if needs_obscura {
        match Obscura::detect(&cli.obscura_bin, cli.fetch_concurrency, tune.page_timeout) {
            Ok(mut o) => {
                o.stealth = !cli.no_stealth;
                o.obey_robots = cli.obey_robots;
                Some(o)
            }
            // Fatal only when there is no other way to work.
            Err(e) if jina.is_none() => return Err(e),
            Err(e) => {
                tracing::warn!(error = %e, "obscura is unavailable; the DuckDuckGo lane is off");
                None
            }
        }
    } else {
        None
    };

    let backend = api::select_backend(obscura.as_ref(), jina.as_ref())?;
    tracing::info!(backend = backend.name(), "fetching pages");

    // Search lanes: every available one by default, or the single engine the
    // caller pinned for a like-for-like comparison. Shared with the API server so
    // a pinned engine resolves identically in both.
    let lanes = api::select_lanes(
        cli.search_engines,
        obscura.as_ref(),
        jina.as_ref(),
        tune.search_lane_timeout,
    )?;
    tracing::info!(
        lanes = %lanes.iter().map(|l| l.id).collect::<Vec<_>>().join(","),
        deadline_s = tune.search_lane_timeout.as_secs(),
        "search lanes enabled"
    );

    // API mode diverges here: everything above is the same startup a CLI run does,
    // which is the point — a missing key must fail the same way and with the same
    // message whether the process is about to run one query or serve thousands.
    if cli.api {
        let state = api::AppState::new(
            creds,
            obscura,
            jina,
            cli.search_engines,
            cli.thinking_control,
            cli.planner_thinking_control.unwrap_or(cli.thinking_control),
            cli.fetch_concurrency,
        );
        api::serve(cli.api_port, state).await?;
        return Ok(0);
    }

    let search_cache = if tune.search_cache {
        let cache = SearchCache::discover(tune.search_cache_ttl);
        tracing::debug!(
            dir = %cache.dir().display(),
            ttl_s = cache.ttl().as_secs(),
            "search cache enabled"
        );
        Some(cache)
    } else {
        tracing::debug!("search cache disabled");
        None
    };

    tracing::debug!(model = %creds.llm_model, endpoint = %creds.llm_endpoint, "generative model");

    let jev = Jev::new(
        creds.typesafe_endpoint.clone(),
        creds.typesafe_key.clone(),
        tune.max_retries,
        tune.http_timeout,
    )?;
    let planner_tc = cli.planner_thinking_control.unwrap_or(cli.thinking_control);

    let llm = Llm::new(
        creds.llm_endpoint.clone(),
        creds.llm_key.clone(),
        creds.llm_model.clone(),
        tune.max_retries,
        tune.http_timeout,
        cli.thinking_control,
    )?;
    let planner = Llm::new(
        creds.planner_endpoint.clone(),
        creds.planner_key.clone(),
        creds.planner_model.clone(),
        tune.max_retries,
        tune.http_timeout,
        planner_tc,
    )?;
    tracing::debug!(
        planner_model = %creds.planner_model,
        planner_endpoint = %creds.planner_endpoint,
        "planner model"
    );

    let fetcher = Fetcher::new(backend, std::time::Duration::from_secs(20))?
        .with_lanes(lanes)
        .with_search_cache(search_cache);
    // Fail before a single token is spent: a mistyped `--url` that silently
    // dropped would answer a slightly different question than the one asked.
    let seed_urls = Scout::validate_seed_urls(&cli.url)?;
    let scout = Scout {
        auto: !cli.no_auto,
        follow: !cli.no_follow,
        enrich: !cli.no_enrich,
        jev,
        llm,
        planner,
        fetcher,
        tune: std::sync::RwLock::new(tune),
        timings: Default::default(),
        // Read the clock exactly once, here. Every stage that needs to know what
        // day it is reads this string, so a run that crosses midnight cannot
        // disagree with itself halfway through.
        today: clock::today_utc(),
        // A CLI run has no second reader: the tracing logs already say everything
        // the progress channel would.
        progress: None,
        seed_urls,
    };
    let report = scout.run(&query).await?;
    scout.fetcher.log_search_cache_summary();

    // Verbose already means "tell me what happened"; the cost and timing of the run
    // is part of that, and it goes to stderr so it survives any output format.
    if cli.verbose > 0
        || cli
            .log_level
            .as_deref()
            .is_some_and(|l| l != "off" && l != "error" && l != "warn")
    {
        eprint!("{}", run_summary(&report));
    }
    if cli.profile {
        eprint!("{}", profile_table(&report));
    }
    let rendered = output::render(&report, cli.format)?;

    match &cli.output {
        Some(path) => {
            std::fs::write(path, &rendered)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => print!("{rendered}"),
    }

    // Exit codes describe the *run*, not the findings, so a caller can tell a
    // crash from an honest "the web does not contain this". Finding nothing is a
    // valid result and is reported in the payload, not as a failure.
    Ok(match report.outcome {
        Outcome::Complete | Outcome::Partial | Outcome::Truncated | Outcome::Empty => 0,
    })
}

/// Which subsystem a stage's time belongs to.
///
/// Derived from the stage label rather than tracked separately, so a new stage shows
/// up in the right column simply by being named consistently. Anything unrecognised
/// lands in "other", which is the signal that a label needs adding here.
fn component_of(stage: &str) -> &'static str {
    if stage.contains("search") || stage.contains("fetch") {
        "browser"
    } else if stage.contains("classify (LLM)")
        || stage.contains("plan")
        || stage.contains("extract")
        || stage.contains("synthesize")
    {
        "llm"
    } else if stage.contains("audit")
        || stage.contains("triage")
        || stage.contains("screen")
        || stage.contains("ground")
        || stage.contains("assess")
        || stage.contains("verify")
    {
        "jev"
    } else {
        "other"
    }
}

/// Thousands separators, because six-digit token counts are unreadable without them.
fn commas(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The tail of an LLM summary line: reported cost and reasoning tokens.
///
/// Both are omitted when there is nothing to report. An endpoint that never
/// sends `usage.cost` (anything that is not OpenRouter) gets no dollar figure
/// rather than `$0.0000`, and a run that did no thinking says nothing about
/// thinking.
fn llm_extras(cost_usd: Option<f64>, reasoning_tokens: usize) -> String {
    let mut out = String::new();
    if let Some(c) = cost_usd {
        out.push_str(&format!("   ${c:.4}"));
    }
    if reasoning_tokens > 0 {
        out.push_str(&format!("   ({} reasoning)", commas(reasoning_tokens)));
    }
    out
}

/// What the whole run cost, or `None` if any component's endpoint stayed quiet.
///
/// A partial total would be read as a total. An endpoint that reported nothing
/// makes the sum unknowable, not smaller — except when it made no requests at
/// all, in which case it contributed nothing and the sum still stands.
fn total_cost_usd(st: &crate::types::Stats) -> Option<f64> {
    let part = |requests: usize, cost: Option<f64>| match (requests, cost) {
        (0, _) => Some(0.0),
        (_, some) => some,
    };
    Some(
        st.jev_cost_usd
            + part(st.llm_requests, st.llm_cost_usd)?
            + part(st.planner_requests, st.planner_cost_usd)?,
    )
}

/// What a run cost, in tokens and seconds, split by subsystem.
///
/// Printed to stderr under any verbosity, so it is visible regardless of output
/// format — the stats footer only appears in the terminal and markdown renderings,
/// which means `-f json` users never saw it at all.
fn run_summary(report: &crate::types::ScoutReport) -> String {
    use std::fmt::Write as _;
    let st = &report.stats;

    let mut by_component: std::collections::BTreeMap<&str, u64> = Default::default();
    for (stage, t) in &st.stage_ms {
        *by_component.entry(component_of(stage)).or_insert(0) += t.ms;
    }
    let part = |k: &str| by_component.get(k).copied().unwrap_or(0) as f64 / 1000.0;

    let mut s = String::from("\n  ── run summary ──────────────────────────────────────────────\n");
    let _ = writeln!(
        s,
        "  Jev     {:>3} requests   {:>9} input tokens   ${:.4}",
        st.jev_requests,
        commas(st.jev_input_tokens),
        st.jev_cost_usd
    );
    let _ = writeln!(
        s,
        "  LLM     {:>3} requests   {:>9} prompt + {} completion{}",
        st.llm_requests,
        commas(st.llm_prompt_tokens),
        commas(st.llm_completion_tokens),
        llm_extras(st.llm_cost_usd, st.llm_reasoning_tokens)
    );
    if st.planner_requests > 0 {
        let _ = writeln!(
            s,
            "  Planner {:>3} requests   {:>9} prompt + {} completion{}",
            st.planner_requests,
            commas(st.planner_prompt_tokens),
            commas(st.planner_completion_tokens),
            llm_extras(st.planner_cost_usd, st.planner_reasoning_tokens)
        );
    }
    if let Some(total) = total_cost_usd(st) {
        let _ = writeln!(s, "  Cost    ${total:.4} total");
    }
    let _ = writeln!(
        s,
        "  Time    {:.1}s wall   (browser {:.1}s · jev {:.1}s · llm {:.1}s)",
        st.elapsed_secs,
        part("browser"),
        part("jev"),
        part("llm")
    );
    if part("other") > 0.05 {
        let _ = writeln!(s, "  {:>39}unattributed {:.1}s", "", part("other"));
    }
    let _ = writeln!(
        s,
        "  Work    {} round{} · {} queries · {} pages · {} chunks",
        st.rounds,
        if st.rounds == 1 { "" } else { "s" },
        st.queries_issued,
        st.pages_fetched,
        st.chunks_examined
    );
    if st.quarantined > 0
        || st.rejected_ungrounded > 0
        || st.unsupported_claims > 0
        || st.excluded_contradicted > 0
        || st.constraints_unverified > 0
        || st.wrong_entity_rejected > 0
    {
        let _ = write!(
            s,
            "  Guards  {} passage(s) quarantined · {} ungrounded record(s) rejected",
            st.quarantined, st.rejected_ungrounded
        );
        // Each of these is only printed when it fired: a run that saw no
        // contradiction should not read as though it checked and found one.
        if st.excluded_contradicted > 0 {
            let _ = write!(
                s,
                " · {} contradicted record(s) excluded",
                st.excluded_contradicted
            );
        }
        if st.wrong_entity_rejected > 0 {
            let _ = write!(
                s,
                " · {} wrong-entity value(s) rejected",
                st.wrong_entity_rejected
            );
        }
        if st.constraints_unverified > 0 {
            let _ = write!(
                s,
                " · {} constraint(s) unverified",
                st.constraints_unverified
            );
        }
        // Only when it fired: on a clean answer the line should not imply the
        // check found something.
        if st.unsupported_claims > 0 {
            let _ = write!(
                s,
                " · {} of {} answer claim(s) unsupported",
                st.unsupported_claims, st.claims_checked
            );
        }
        s.push('\n');
    }
    s.push_str("  ─────────────────────────────────────────────────────────────\n");
    s
}

/// Render the per-stage timing table.
///
/// Sorted by time spent, because the question this answers is always "what should I
/// attack first". The share column is against wall clock rather than against the sum
/// of stages, so a stage at 90% is genuinely holding the run up, while several
/// stages at 40% simply mean they ran at the same time.
fn profile_table(report: &crate::types::ScoutReport) -> String {
    use std::fmt::Write as _;
    let wall_ms = (report.stats.elapsed_secs * 1000.0).max(1.0);

    let mut rows: Vec<(&String, &crate::types::StageTime)> = report.stats.stage_ms.iter().collect();
    rows.sort_by(|a, b| b.1.ms.cmp(&a.1.ms));

    let mut s = String::from("\n  stage                    time      share   calls   per call\n");
    s.push_str("  ────────────────────────────────────────────────────────────\n");
    for (name, t) in &rows {
        let per = if t.calls > 0 {
            t.ms / t.calls as u64
        } else {
            0
        };
        let _ = writeln!(
            s,
            "  {:<22} {:>7.1}s  {:>5.0}%  {:>6}  {:>7.1}s",
            name,
            t.ms as f64 / 1000.0,
            100.0 * t.ms as f64 / wall_ms,
            t.calls,
            per as f64 / 1000.0
        );
    }
    let summed: u64 = rows.iter().map(|(_, t)| t.ms).sum();
    let _ = writeln!(
        s,
        "  ────────────────────────────────────────────────────────────\n  \
         {:<22} {:>7.1}s          (sum of stages)\n  {:<22} {:>7.1}s\n  \
         note: concurrent stages sum above wall clock; a stage total\n  \
         above wall clock is overlap, not dominance.\n",
        "overlapped work",
        summed as f64 / 1000.0,
        "wall clock",
        wall_ms / 1000.0
    );
    s
}

/// The log level a run gets when `--log-level` says nothing.
///
/// A CLI run is quiet by default because its output is the payload and anything
/// else on the terminal is noise. A server is the opposite: it has no payload on
/// stdout, runs unattended, and a container whose logs say nothing about which
/// search lanes came up is a container nobody can debug. So `--api` starts at
/// info, and `-v` still raises it from there.
fn default_log_level(api: bool, verbose: u8) -> &'static str {
    match verbose {
        0 if api => "info",
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    }
}

/// Logs always go to stderr so stdout carries only the payload. That separation is
/// what makes `-f json > file` safe to run at any verbosity, which matters when
/// the caller is another program.
fn init_logging(cli: &Cli) {
    let level = cli
        .log_level
        .clone()
        .unwrap_or_else(|| default_log_level(cli.api, cli.verbose).to_string());

    // Our own crate at the chosen level; dependencies stay quiet unless tracing
    // deliberately, since reqwest and hyper at debug bury everything else.
    let filter = EnvFilter::try_from_env("WEBSCOUT_LOG").unwrap_or_else(|_| {
        EnvFilter::new(if level == "trace" {
            format!("webscout={level},obscura=debug")
        } else {
            format!("webscout={level}")
        })
    });

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false);

    if cli.log_json {
        builder.json().init();
    } else {
        builder
            .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
            .init();
    }
}

#[cfg(test)]
mod main_tests {
    use super::{
        RoundLimit, check_query_args, commas, component_of, default_log_level, llm_extras,
        parse_round_limit, run_summary, total_cost_usd,
    };

    #[test]
    fn max_rounds_takes_a_number_or_auto() {
        assert_eq!(parse_round_limit("auto"), Ok(RoundLimit::Auto));
        assert_eq!(parse_round_limit(" AUTO "), Ok(RoundLimit::Auto));
        assert_eq!(parse_round_limit("40"), Ok(RoundLimit::Fixed(40)));
        assert!(
            parse_round_limit("0").is_err(),
            "zero is not a way to say auto"
        );
        assert!(parse_round_limit("lots").is_err());
        assert!(parse_round_limit("-3").is_err());
    }

    #[test]
    fn api_mode_is_not_silent_by_default() {
        assert_eq!(default_log_level(false, 0), "warn");
        assert_eq!(default_log_level(true, 0), "info");
        // -v and beyond mean the same thing in both modes.
        assert_eq!(default_log_level(true, 2), "debug");
        assert_eq!(default_log_level(false, 2), "debug");
        assert_eq!(default_log_level(false, 9), "trace");
    }

    #[test]
    fn a_query_without_api_is_fine() {
        assert!(check_query_args(false, &["who".into(), "is".into()]).is_ok());
    }

    #[test]
    fn api_without_a_query_is_fine() {
        assert!(check_query_args(true, &[]).is_ok());
    }

    #[test]
    fn api_plus_a_query_names_the_conflict() {
        let err = check_query_args(true, &["who".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--api"), "{err}");
        assert!(err.contains("conflicts"), "{err}");
        assert!(err.contains("query"), "{err}");
    }

    #[test]
    fn neither_a_query_nor_api_is_an_error_that_suggests_both() {
        let err = check_query_args(false, &[]).unwrap_err().to_string();
        assert!(err.contains("no query"), "{err}");
        assert!(err.contains("--api"), "{err}");
    }

    #[test]
    fn commas_group_thousands() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1_000), "1,000");
        assert_eq!(commas(1_234_567), "1,234,567");
    }

    #[test]
    fn llm_extras_says_nothing_when_there_is_nothing_to_say() {
        // A plain vLLM endpoint that did no thinking: no dollars, no reasoning.
        assert_eq!(llm_extras(None, 0), "");
    }

    #[test]
    fn llm_extras_prints_a_reported_cost_to_four_decimals() {
        assert_eq!(llm_extras(Some(0.00218), 0), "   $0.0022");
        // A free model that reported its freeness is not the same as silence.
        assert_eq!(llm_extras(Some(0.0), 0), "   $0.0000");
    }

    #[test]
    fn llm_extras_surfaces_reasoning_tokens_with_separators() {
        // The measured 2026-09-18 blow-up: 7,713 completion tokens, all thought.
        assert_eq!(llm_extras(None, 7_713), "   (7,713 reasoning)");
        assert_eq!(
            llm_extras(Some(0.01), 7_713),
            "   $0.0100   (7,713 reasoning)"
        );
    }

    #[test]
    fn total_cost_adds_every_component_that_reported_one() {
        let st = crate::types::Stats {
            jev_cost_usd: 0.0037,
            llm_requests: 3,
            llm_cost_usd: Some(0.0021),
            planner_requests: 1,
            planner_cost_usd: Some(0.0104),
            ..Default::default()
        };
        let total = total_cost_usd(&st).expect("all three reported");
        assert!((total - 0.0162).abs() < 1e-9, "{total}");
    }

    #[test]
    fn total_cost_is_unknown_when_an_endpoint_that_ran_stayed_quiet() {
        let st = crate::types::Stats {
            jev_cost_usd: 0.0037,
            llm_requests: 3,
            llm_cost_usd: None,
            ..Default::default()
        };
        assert_eq!(
            total_cost_usd(&st),
            None,
            "a partial total would be read as a total"
        );
    }

    #[test]
    fn a_component_that_never_ran_does_not_make_the_total_unknown() {
        // The usual single-model run: the planner shares the writer's client and
        // never reports separately, which must not hide the total.
        let st = crate::types::Stats {
            jev_cost_usd: 0.0037,
            llm_requests: 3,
            llm_cost_usd: Some(0.0021),
            planner_requests: 0,
            planner_cost_usd: None,
            ..Default::default()
        };
        let total = total_cost_usd(&st).expect("the planner spent nothing");
        assert!((total - 0.0058).abs() < 1e-9, "{total}");
    }

    #[test]
    fn the_run_summary_shows_cost_and_reasoning() {
        let mut report = crate::types::ScoutReport {
            query: "q".into(),
            mission: Default::default(),
            outcome: crate::types::Outcome::Empty,
            records: Vec::new(),
            answer: None,
            evidence: Vec::new(),
            sources: Vec::new(),
            quarantined_sources: Vec::new(),
            notes: Vec::new(),
            stats: Default::default(),
        };
        report.stats.jev_cost_usd = 0.0037;
        report.stats.llm_requests = 3;
        report.stats.llm_cost_usd = Some(0.0021);
        report.stats.llm_reasoning_tokens = 7_713;
        let s = run_summary(&report);
        assert!(s.contains("$0.0021"), "{s}");
        assert!(s.contains("(7,713 reasoning)"), "{s}");
        assert!(s.contains("$0.0058 total"), "{s}");
    }

    #[test]
    fn the_run_summary_omits_a_cost_no_endpoint_reported() {
        let mut report = crate::types::ScoutReport {
            query: "q".into(),
            mission: Default::default(),
            outcome: crate::types::Outcome::Empty,
            records: Vec::new(),
            answer: None,
            evidence: Vec::new(),
            sources: Vec::new(),
            quarantined_sources: Vec::new(),
            notes: Vec::new(),
            stats: Default::default(),
        };
        report.stats.llm_requests = 3;
        let s = run_summary(&report);
        assert!(!s.contains("total"), "{s}");
        assert!(!s.contains("reasoning"), "{s}");
    }

    #[test]
    fn stages_land_in_the_right_column() {
        assert_eq!(component_of("3 search"), "browser");
        assert_eq!(component_of("6 ground"), "jev");
        assert_eq!(component_of("10 synthesize"), "llm");
        assert_eq!(component_of("something new"), "other");
    }
}
