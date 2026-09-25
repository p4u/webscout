//! The MCP server: webscout as a tool any AI client can call.
//!
//! Served by the `--api` process at `/mcp` over the Streamable HTTP transport,
//! answering every POST with a single JSON body. Server-initiated streams are
//! not offered (`GET` is 405), which the specification allows and which every
//! client handles; nothing here needs to push to a client.
//!
//! Three decisions shape the tools.
//!
//! **Searches are jobs, not calls.** A question takes fifteen seconds to two
//! minutes and a list can take half an hour, far past any client's tool
//! timeout. `start_search` returns a request id at once; the run lives in a
//! spawned task, independent of the connection that started it, and is read
//! back with `get_search_status` / `get_search_result`. The result call can
//! long-poll (`wait_seconds`), so a client that just wants the answer makes a
//! handful of calls instead of polling every second. This is the opposite of
//! the NDJSON endpoint, where a disconnect cancels the run: here disconnecting
//! is the normal way to use it, and `cancel_search` is the way to stop one.
//!
//! **A token is required, always.** One shared bearer token, from
//! `WEBSCOUT_MCP_TOKEN` (or `--mcp-token`). Unset means the endpoint refuses
//! every call with 503 — a missing guard is never an open door. The token is
//! compared in constant time and never logged.
//!
//! **Two levels of result.** `detail: "simple"` is the result text alone — the
//! answer prose, or the records table — for a client that only wants to read
//! it. `detail: "full"` is everything the web UI shows: sources with support
//! scores, notes, quarantined pages, per-actor tokens and cost, run stats, the
//! parsed mission and structured records. `format` returns one of the export
//! renderings (markdown, json, csv, jsonl) verbatim.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};

use crate::api::{
    AppState, FORMATS, OptionSpec, OptionType, ProgressEvent, RunOptions, UsageSnapshot, catalogue,
    llm_usage_json, new_run_id, now_rfc3339, render_all,
};
use crate::scout::Scout;
use crate::types::ScoutReport;

/// Protocol revisions this server speaks, newest first. The subset used here —
/// initialize, ping, tools/list, tools/call — is the same in all of them.
const PROTOCOL_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// Searches remembered, running or finished. Oldest finished ones go first;
/// a running search is never evicted.
const MAX_JOBS: usize = 100;

/// Searches allowed to run at once across all MCP clients. Each one is a
/// research run with its own Jev, LLM and browser budget; a client that loops
/// on `start_search` must not be able to fan out without bound.
pub const DEFAULT_MAX_RUNNING: usize = 4;

/// Longest a single `get_search_result` call may block. Short on purpose: a
/// client checking every ten seconds or so is cheap for this server, and a
/// short call keeps well inside any client's tool timeout.
const MAX_WAIT_SECS: u64 = 15;

/// Default blocking time for `get_search_result`: a check about every ten
/// seconds while a search runs.
const DEFAULT_WAIT_SECS: u64 = 10;

/// How long `search` waits before handing back a request id. A quick
/// question can finish inside it; anything longer is picked up by
/// `get_search_result`, one short check at a time.
const SEARCH_WAIT_SECS: u64 = 10;

/// Items a list result returns when the caller sets no `limit` and the
/// request named no number. A 342-row harvest rendered whole was 52,789
/// characters — past Claude Code's tool-output limit, so the client spilled it
/// to a file (measured 2026-09-25). The export formats are never capped.
const DEFAULT_ROW_LIMIT: usize = 50;

/// Progress lines kept per search for `get_search_status`.
const RECENT_PROGRESS: usize = 12;

const INSTRUCTIONS: &str = "webscout searches the web, reads the pages it finds, and returns only \
facts it could verify against them: a direct answer with numbered sources for a question, or a \
table of items for a list request (\"Find 20 coworking spaces in Barcelona with a public email\"). \
To use it, call `search` with the request in plain language. It waits about 10 seconds; a \
quick question may be answered in that call. Otherwise the result has status \"running\" and a \
request_id: call `get_search_result` with it, and keep calling while the status is still \
\"running\" — each call waits about 10 seconds for the search, so call again straight away, \
without sleeping. Questions usually take 15 seconds to 2 minutes; lists can take many minutes. Never start the same search twice. Use detail=\"full\" on \
get_search_result for sources, notes, cost and run statistics. An outcome of \"empty\" means \
nothing could be verified, not that the thing does not exist. `start_search` returns at once \
without waiting, for long searches you want to run in the background while doing other work.";

// ------------------------------------------------------------------- state --

/// Everything the MCP endpoint keeps between calls.
pub struct McpState {
    /// `None` disables the endpoint entirely.
    token: Option<String>,
    max_running: usize,
    jobs: Mutex<VecDeque<Arc<Job>>>,
}

impl McpState {
    pub fn new(token: Option<String>, max_running: usize) -> Self {
        let token = token
            .map(|t| crate::config::unquote(&t).to_string())
            .filter(|t| !t.is_empty());
        Self {
            token,
            max_running: max_running.max(1),
            jobs: Mutex::new(VecDeque::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.token.is_some()
    }

    fn find(&self, id: &str) -> Option<Arc<Job>> {
        let jobs = self.jobs.lock().ok()?;
        jobs.iter().find(|j| j.id == id).cloned()
    }

    fn running(&self) -> usize {
        self.jobs
            .lock()
            .map(|jobs| jobs.iter().filter(|j| j.is_running()).count())
            .unwrap_or(0)
    }

    fn insert(&self, job: Arc<Job>) {
        let Ok(mut jobs) = self.jobs.lock() else {
            return;
        };
        while jobs.len() >= MAX_JOBS {
            match jobs.iter().position(|j| !j.is_running()) {
                Some(i) => {
                    jobs.remove(i);
                }
                None => break,
            }
        }
        jobs.push_back(job);
    }

    fn snapshot(&self) -> Vec<Arc<Job>> {
        self.jobs
            .lock()
            .map(|jobs| jobs.iter().cloned().collect())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Running,
    Finished,
    Failed,
    Cancelled,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Finished => "finished",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }
}

struct JobState {
    status: Status,
    last: Option<ProgressEvent>,
    recent: VecDeque<(u64, String)>,
    report: Option<Arc<ScoutReport>>,
    formats: BTreeMap<&'static str, String>,
    error: Option<String>,
    elapsed_ms: Option<u64>,
    /// Usage frozen at the end of a failed or cancelled run.
    final_usage: Option<UsageSnapshot>,
}

/// One search, from `start_search` until it is evicted.
struct Job {
    id: String,
    query: String,
    options: Map<String, Value>,
    started_at: String,
    started: Instant,
    scout: Arc<Scout>,
    state: Mutex<JobState>,
    done: tokio::sync::watch::Sender<bool>,
    abort: Mutex<Option<tokio::task::AbortHandle>>,
}

impl Job {
    fn is_running(&self) -> bool {
        self.state
            .lock()
            .map(|s| s.status == Status::Running)
            .unwrap_or(false)
    }

    fn elapsed_ms(&self) -> u64 {
        self.state
            .lock()
            .ok()
            .and_then(|s| s.elapsed_ms)
            .unwrap_or_else(|| self.started.elapsed().as_millis() as u64)
    }

    fn record(&self, ev: ProgressEvent) {
        let at = self.started.elapsed().as_secs();
        if let Ok(mut s) = self.state.lock() {
            let line = match ev.round {
                Some(r) => format!("[{}] round {r}: {}", ev.stage, ev.message),
                None => format!("[{}] {}", ev.stage, ev.message),
            };
            s.recent.push_back((at, line));
            while s.recent.len() > RECENT_PROGRESS {
                s.recent.pop_front();
            }
            s.last = Some(ev);
        }
    }

    /// Settle a run. Only the first settlement counts: a cancel that raced a
    /// finishing run keeps whichever landed first.
    fn settle(&self, f: impl FnOnce(&mut JobState)) {
        if let Ok(mut s) = self.state.lock() {
            if s.status != Status::Running {
                return;
            }
            f(&mut s);
            s.elapsed_ms = Some(self.started.elapsed().as_millis() as u64);
        }
        let _ = self.done.send(true);
    }

    fn usage(&self) -> UsageSnapshot {
        if let Ok(s) = self.state.lock() {
            if let Some(r) = &s.report {
                return UsageSnapshot::from_stats(&r.stats);
            }
            if let Some(u) = s.final_usage {
                return u;
            }
        }
        UsageSnapshot::sample(&self.scout)
    }
}

// ------------------------------------------------------------------- auth --

/// Compare without an early exit, so response timing says nothing about how
/// much of a guessed token was right.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The bearer token a request carries, if any.
fn bearer(headers: &HeaderMap) -> Option<&str> {
    let v = headers.get(header::AUTHORIZATION)?.to_str().ok()?.trim();
    let (scheme, rest) = v.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| rest.trim())
        .filter(|t| !t.is_empty())
}

/// `None` when the request may proceed; otherwise the refusal to send.
fn check_auth(mcp: &McpState, headers: &HeaderMap) -> Option<Response> {
    let Some(expected) = &mcp.token else {
        return Some((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "mcp_disabled",
                "message": "The MCP endpoint is disabled: set WEBSCOUT_MCP_TOKEN on the server to enable it."
            })),
        )
            .into_response());
    };
    match bearer(headers) {
        Some(got) if constant_time_eq(got.as_bytes(), expected.as_bytes()) => None,
        got => {
            let (code, message) = if got.is_none() {
                (
                    "invalid_request",
                    "Missing token. Send the header `Authorization: Bearer <token>`, where the token is WEBSCOUT_MCP_TOKEN from the server's .env.",
                )
            } else {
                ("invalid_token", "The bearer token is not valid.")
            };
            let mut res = (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": code, "message": message})),
            )
                .into_response();
            if let Ok(v) = format!("Bearer realm=\"webscout\", error=\"{code}\"").parse() {
                res.headers_mut().insert(header::WWW_AUTHENTICATE, v);
            }
            Some(res)
        }
    }
}

// --------------------------------------------------------------- endpoint --

/// `POST /mcp`: one JSON-RPC message, or a batch of them.
pub async fn handle(State(app): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(res) = check_auth(&app.mcp, &headers) {
        return res;
    }
    let msg: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(rpc_error(Value::Null, -32700, &format!("parse error: {e}"))),
            )
                .into_response();
        }
    };

    match msg {
        Value::Array(batch) => {
            if batch.is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(rpc_error(Value::Null, -32600, "empty batch")),
                )
                    .into_response();
            }
            let mut out = Vec::new();
            for m in batch {
                if let Some(r) = handle_message(&app, m).await {
                    out.push(r);
                }
            }
            if out.is_empty() {
                StatusCode::ACCEPTED.into_response()
            } else {
                Json(Value::Array(out)).into_response()
            }
        }
        m => match handle_message(&app, m).await {
            Some(r) => Json(r).into_response(),
            // Notifications and client responses get 202 and no body.
            None => StatusCode::ACCEPTED.into_response(),
        },
    }
}

/// `GET /mcp`: no server-initiated stream is offered.
pub async fn no_stream(State(app): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(res) = check_auth(&app.mcp, &headers) {
        return res;
    }
    (
        StatusCode::METHOD_NOT_ALLOWED,
        [(header::ALLOW, "POST")],
        "this server does not open server-initiated streams; POST JSON-RPC messages instead",
    )
        .into_response()
}

/// `GET /api/mcp`: what the UI needs to explain how to connect. No secrets.
pub async fn info(State(app): State<Arc<AppState>>) -> Response {
    Json(json!({
        "enabled": app.mcp.enabled(),
        "path": "/mcp",
        "transport": "streamable-http",
        "auth": "bearer",
        "protocol_versions": PROTOCOL_VERSIONS,
        "tools": tool_definitions()
            .into_iter()
            .map(|t| json!({"name": t["name"], "title": t["title"]}))
            .collect::<Vec<_>>(),
    }))
    .into_response()
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

async fn handle_message(app: &Arc<AppState>, msg: Value) -> Option<Value> {
    let Value::Object(obj) = msg else {
        return Some(rpc_error(
            Value::Null,
            -32600,
            "a message must be a JSON object",
        ));
    };
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        // A response to something we never send; nothing to answer.
        return None;
    };
    // No id: a notification. None of them needs an answer or an action here.
    let id = obj.get("id").cloned()?;
    let params = obj.get("params").cloned().unwrap_or(Value::Null);

    let result = match method {
        "initialize" => Ok(initialize(&params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_definitions()})),
        "tools/call" => call_tool(app, &params).await,
        // Asked by some clients whatever the capabilities say; an empty list
        // is a better answer than an error that aborts their startup.
        "resources/list" => Ok(json!({"resources": []})),
        "resources/templates/list" => Ok(json!({"resourceTemplates": []})),
        "prompts/list" => Ok(json!({"prompts": []})),
        "logging/setLevel" => Ok(json!({})),
        other => Err((-32601, format!("method not found: {other}"))),
    };
    Some(match result {
        Ok(v) => rpc_result(id, v),
        Err((code, message)) => rpc_error(id, code, &message),
    })
}

fn initialize(params: &Value) -> Value {
    let asked = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or("");
    // Echo a version we speak; otherwise offer our newest and let the client decide.
    let version = PROTOCOL_VERSIONS
        .iter()
        .find(|v| **v == asked)
        .copied()
        .unwrap_or(PROTOCOL_VERSIONS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {
            "name": "webscout",
            "title": "webscout — verified web search",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "instructions": INSTRUCTIONS,
    })
}

// ------------------------------------------------------------------ tools --

/// JSON Schema for one catalogue option, for a tool's input schema.
fn option_schema(spec: &OptionSpec) -> Value {
    let mut description = format!("{} {}", spec.label, spec.help);
    if let (Some(values), Some(labels)) = (&spec.values, &spec.value_labels) {
        let named: Vec<String> = values
            .iter()
            .zip(labels)
            .map(|(v, l)| format!("{v} = {l}"))
            .collect();
        description.push_str(&format!(" Values: {}.", named.join(", ")));
    }
    let mut s = match spec.kind {
        OptionType::Integer if spec.auto => json!({
            "anyOf": [
                {"type": "integer", "minimum": spec.min, "maximum": spec.max},
                {"type": "string", "enum": ["auto"]},
            ]
        }),
        OptionType::Integer => json!({"type": "integer", "minimum": spec.min, "maximum": spec.max}),
        OptionType::Number => json!({"type": "number", "minimum": spec.min, "maximum": spec.max}),
        OptionType::Boolean => json!({"type": "boolean"}),
        OptionType::Enum => json!({"type": "string", "enum": spec.values}),
        OptionType::String => json!({"type": "string"}),
    };
    s["description"] = json!(description);
    s["default"] = spec.default.clone();
    s
}

/// The options a person would set, lifted to top-level tool arguments. The
/// rest of the catalogue goes through `options`; `format` is chosen when the
/// result is read, not when the search starts.
fn common_options() -> Vec<OptionSpec> {
    catalogue()
        .into_iter()
        .filter(|g| g.id == "basic" || g.id == "advanced")
        .flat_map(|g| g.options)
        .filter(|o| o.name != "format")
        .collect()
}

fn request_id_schema() -> Value {
    json!({"type": "string", "description": "The request_id returned by start_search."})
}

pub fn tool_definitions() -> Vec<Value> {
    let mut start_props = Map::new();
    start_props.insert(
        "query".into(),
        json!({
            "type": "string",
            "description": "What to find, in plain language. A question (\"Who is the CEO of GitLab?\") gets a direct, cited answer; a list request (\"Find 15 open-source CRM projects with a release in the last 12 months, with their website\") gets a table. For lists, say how many items and which details you want.",
        }),
    );
    for spec in common_options() {
        start_props.insert(spec.name.into(), option_schema(&spec));
    }
    start_props.insert(
        "options".into(),
        json!({
            "type": "object",
            "description": "Expert tuning: any other option by name (thresholds, batch sizes, models, search engines). See list_search_options. Rarely needed.",
            "additionalProperties": true,
        }),
    );

    let mut search_props = start_props.clone();
    search_props.insert("limit".into(), json!({
            "type": "integer",
            "minimum": 0,
            "description": "For a list result: most items to return, complete ones (every requested detail found) first. Default: as many as the request asked for, or 50. 0 returns all. Exports via format are never cut.",
        }));
    search_props.insert(
        "detail".into(),
        json!({
            "type": "string",
            "enum": ["simple", "full"],
            "default": "simple",
            "description": "simple: the result text only. full: also sources, notes, cost and run statistics, as JSON.",
        }),
    );

    vec![
        json!({
            "name": "search",
            "title": "Search the web (verified)",
            "description": "Search the web and get a verified answer: webscout reads the pages it finds and returns only what it could check against them — a cited answer for a question, or a table for a list request. Use this for any web lookup. It waits about 10 s; a quick question may be answered in this call. Otherwise the response has status \"running\" and a request_id: call get_search_result with it, and call again straight away while the status is still \"running\" (each call waits about 10 s). Questions usually take 15 s to 2 min; lists can take many minutes. Do not start the same search again.",
            "inputSchema": {
                "type": "object",
                "properties": search_props,
                "required": ["query"],
                "additionalProperties": false,
            },
            "annotations": {
                "title": "Search the web (verified)",
                "readOnlyHint": false,
                "destructiveHint": false,
                "idempotentHint": false,
                "openWorldHint": true,
            },
        }),
        json!({
            "name": "start_search",
            "title": "Start a web search",
            "description": "Like search, but returns a request_id immediately without waiting. Use it to run a long search (a big list) in the background while doing other work; read it later with get_search_result, which waits for the result. For an ordinary lookup, use search instead.",
            "inputSchema": {
                "type": "object",
                "properties": start_props,
                "required": ["query"],
                "additionalProperties": false,
            },
            "annotations": {
                "title": "Start a web search",
                "readOnlyHint": false,
                "destructiveHint": false,
                "idempotentHint": false,
                "openWorldHint": true,
            },
        }),
        json!({
            "name": "get_search_result",
            "title": "Get a search result",
            "description": "Return the result of a search started with search or start_search, waiting up to wait_seconds (default 10, max 15) for it to finish. If it is still running, returns status \"running\" with progress: call again with the same request_id — no need to sleep in between. detail=\"simple\" returns only the result text (the answer, or the table of items); detail=\"full\" adds sources with support scores, notes, pages ignored for prompt injection, tokens and cost, run statistics and structured records. format returns an export (markdown, json, csv, jsonl) verbatim instead.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "request_id": request_id_schema(),
                    "detail": {
                        "type": "string",
                        "enum": ["simple", "full"],
                        "default": "simple",
                        "description": "simple: the result text only. full: everything, as JSON.",
                    },
                    "format": {
                        "type": "string",
                        "enum": FORMATS,
                        "description": "Only to save or hand over a file: returns that export document verbatim (large; json includes every passage read). Leave it out to read the result — detail covers that.",
                    },
                    "limit": json!({
                        "type": "integer",
                        "minimum": 0,
                        "description": "For a list result: most items to return, complete ones (every requested detail found) first. Default: as many as the request asked for, or 50. 0 returns all. Exports via format are never cut.",
                    }),
                    "wait_seconds": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_WAIT_SECS,
                        "default": DEFAULT_WAIT_SECS,
                        "description": "How long to wait for a running search to finish before returning a progress note. 0 returns at once.",
                    },
                },
                "required": ["request_id"],
                "additionalProperties": false,
            },
            "annotations": {"title": "Get a search result", "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false},
        }),
        json!({
            "name": "get_search_status",
            "title": "Check a search's progress",
            "description": "Return a search's status without waiting: running, finished, failed or cancelled, with elapsed time, the current stage, pages read, items found, cost so far and the latest progress messages.",
            "inputSchema": {
                "type": "object",
                "properties": {"request_id": request_id_schema()},
                "required": ["request_id"],
                "additionalProperties": false,
            },
            "annotations": {"title": "Check a search's progress", "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false},
        }),
        json!({
            "name": "cancel_search",
            "title": "Cancel a search",
            "description": "Stop a running search. Nothing it found so far is kept. Has no effect on a search that already finished.",
            "inputSchema": {
                "type": "object",
                "properties": {"request_id": request_id_schema()},
                "required": ["request_id"],
                "additionalProperties": false,
            },
            "annotations": {"title": "Cancel a search", "readOnlyHint": false, "destructiveHint": true, "idempotentHint": true, "openWorldHint": false},
        }),
        json!({
            "name": "list_searches",
            "title": "List recent searches",
            "description": "List the searches this server remembers (the most recent 100, running ones included), newest first, with their request_id, status and query. Useful to find a request_id again.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
            "annotations": {"title": "List recent searches", "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false},
        }),
        json!({
            "name": "list_search_options",
            "title": "List search options",
            "description": "Describe every option start_search accepts, including expert tuning passed through its `options` argument: name, type, default, allowed values or range, and what it does.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false},
            "annotations": {"title": "List search options", "readOnlyHint": true, "idempotentHint": true, "openWorldHint": false},
        }),
    ]
}

/// A tool-level failure: reported inside the result so the model can read it
/// and recover, as the protocol asks, rather than as a JSON-RPC error.
fn tool_error(message: impl Into<String>) -> Value {
    json!({"content": [{"type": "text", "text": message.into()}], "isError": true})
}

fn tool_text(text: impl Into<String>) -> Value {
    json!({"content": [{"type": "text", "text": text.into()}]})
}

/// A JSON result: serialised as text for every client, and as
/// `structuredContent` for clients that read it.
fn tool_json(v: Value) -> Value {
    let text = serde_json::to_string_pretty(&v).unwrap_or_default();
    json!({"content": [{"type": "text", "text": text}], "structuredContent": v})
}

async fn call_tool(app: &Arc<AppState>, params: &Value) -> Result<Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((-32602, "tools/call needs a tool name".to_string()))?;
    let args = match params.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => return Err((-32602, "arguments must be an object".into())),
    };
    tracing::info!(tool = name, "mcp tool call");
    Ok(match name {
        "search" => search(app, &args).await,
        "start_search" => start_search(app, &args),
        "get_search_result" => get_search_result(app, &args).await,
        "get_search_status" => with_job(app, &args, |job| tool_json(status_json(job))),
        "cancel_search" => with_job(app, &args, cancel),
        "list_searches" => list_searches(app),
        "list_search_options" => list_search_options(),
        other => return Err((-32602, format!("unknown tool: {other}"))),
    })
}

fn with_job(
    app: &Arc<AppState>,
    args: &Map<String, Value>,
    f: impl FnOnce(&Job) -> Value,
) -> Value {
    match lookup(app, args) {
        Ok(job) => f(&job),
        Err(e) => e,
    }
}

fn lookup(app: &Arc<AppState>, args: &Map<String, Value>) -> Result<Arc<Job>, Value> {
    let Some(id) = args
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::trim)
    else {
        return Err(tool_error("request_id is required."));
    };
    app.mcp.find(id).ok_or_else(|| {
        tool_error(format!(
            "No search with request_id \"{id}\". The server remembers the last {MAX_JOBS} searches until it restarts; list_searches shows them."
        ))
    })
}

fn start_search(app: &Arc<AppState>, args: &Map<String, Value>) -> Value {
    match launch(app, args) {
        Ok(job) => tool_json(json!({
            "request_id": job.id,
            "status": "running",
            "query": job.query,
            "next": "Call get_search_result with this request_id; it waits for the result. Repeat while status is running.",
        })),
        Err(e) => e,
    }
}

/// `search`: start, then wait in the same call. Most questions finish inside
/// the wait, so a client gets its answer from one call without knowing the
/// search is a job; a longer one returns its request_id and the next step.
async fn search(app: &Arc<AppState>, args: &Map<String, Value>) -> Value {
    let mut run_args = args.clone();
    let mut read_args = Map::new();
    for key in ["detail", "limit"] {
        if let Some(v) = run_args.remove(key) {
            read_args.insert(key.into(), v);
        }
    }
    match launch(app, &run_args) {
        Ok(job) => result_for(job, &read_args, SEARCH_WAIT_SECS).await,
        Err(e) => e,
    }
}

/// Validate a start request and spawn its run. `Err` is the tool error to return.
fn launch(app: &Arc<AppState>, args: &Map<String, Value>) -> Result<Arc<Job>, Value> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if query.is_empty() {
        return Err(tool_error("query is required and must not be empty."));
    }

    // Top-level options and the `options` bag fold into one map, validated by
    // the same catalogue as the HTTP API: unknown names are an error, never
    // silently ignored.
    let mut options = match args.get("options") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => {
            return Err(tool_error(
                "options must be an object of option names to values.",
            ));
        }
    };
    for (k, v) in args {
        if k == "query" || k == "options" || v.is_null() {
            continue;
        }
        options.insert(k.clone(), v.clone());
    }
    // A model may send "12" for a number; the catalogue wants 12.
    if let Some(Value::String(s)) = options.get("max_rounds")
        && let Ok(n) = s.trim().parse::<u64>()
    {
        options.insert("max_rounds".into(), json!(n));
    }
    let opts = match RunOptions::from_map(&options) {
        Ok(o) => o,
        Err(msg) => {
            return Err(tool_error(format!(
                "{msg} Call list_search_options for the full list."
            )));
        }
    };

    let running = app.mcp.running();
    if running >= app.mcp.max_running {
        return Err(tool_error(format!(
            "{running} searches are already running, the most this server allows at once. Wait for one to finish (get_search_result) or stop one (cancel_search), then try again."
        )));
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel(crate::api::PROGRESS_BUFFER);
    let scout = match app.build_scout(&opts, tx) {
        Ok(s) => Arc::new(s),
        Err(e) => return Err(tool_error(format!("could not start the search: {e}"))),
    };

    let id = new_run_id();
    let (done, _) = tokio::sync::watch::channel(false);
    let job = Arc::new(Job {
        id: id.clone(),
        query: query.to_string(),
        options,
        started_at: now_rfc3339(),
        started: Instant::now(),
        scout: scout.clone(),
        state: Mutex::new(JobState {
            status: Status::Running,
            last: None,
            recent: VecDeque::new(),
            report: None,
            formats: BTreeMap::new(),
            error: None,
            elapsed_ms: None,
            final_usage: None,
        }),
        done,
        abort: Mutex::new(None),
    });
    app.mcp.insert(job.clone());
    tracing::info!(run_id = %id, query = %query, "mcp search starting");

    let task = {
        let job = job.clone();
        let app = app.clone();
        let query = query.to_string();
        tokio::spawn(async move {
            let fut = scout.run(&query);
            tokio::pin!(fut);
            let res = loop {
                tokio::select! {
                    biased;
                    Some(ev) = rx.recv() => job.record(ev),
                    res = &mut fut => break res,
                }
            };
            while let Ok(ev) = rx.try_recv() {
                job.record(ev);
            }
            match res {
                Ok(report) => {
                    let formats = render_all(&report);
                    // Downloadable from the UI's endpoint too, like a UI run.
                    app.store_run(job.id.clone(), formats.clone());
                    tracing::info!(run_id = %job.id, outcome = report.outcome.as_str(), "mcp search finished");
                    job.settle(|s| {
                        s.status = Status::Finished;
                        s.report = Some(Arc::new(report));
                        s.formats = formats;
                    });
                }
                Err(e) => {
                    tracing::warn!(run_id = %job.id, error = %e, "mcp search failed");
                    let usage = UsageSnapshot::sample(&scout);
                    job.settle(|s| {
                        s.status = Status::Failed;
                        s.error = Some(e.to_string());
                        s.final_usage = Some(usage);
                    });
                }
            }
        })
    };
    if let Ok(mut a) = job.abort.lock() {
        *a = Some(task.abort_handle());
    }

    Ok(job)
}

fn cancel(job: &Job) -> Value {
    if !job.is_running() {
        return tool_json(json!({
            "request_id": job.id,
            "status": job.state.lock().map(|s| s.status.as_str()).unwrap_or("unknown"),
            "message": "The search had already ended; nothing was cancelled.",
        }));
    }
    let usage = UsageSnapshot::sample(&job.scout);
    if let Ok(mut a) = job.abort.lock()
        && let Some(h) = a.take()
    {
        h.abort();
    }
    job.settle(|s| {
        s.status = Status::Cancelled;
        s.final_usage = Some(usage);
    });
    tracing::info!(run_id = %job.id, "mcp search cancelled");
    tool_json(json!({"request_id": job.id, "status": "cancelled"}))
}

fn list_searches(app: &Arc<AppState>) -> Value {
    let jobs: Vec<Value> = app
        .mcp
        .snapshot()
        .iter()
        .rev()
        .map(|j| {
            let (status, outcome) = j
                .state
                .lock()
                .map(|s| {
                    (
                        s.status.as_str(),
                        s.report.as_ref().map(|r| r.outcome.as_str()),
                    )
                })
                .unwrap_or(("unknown", None));
            json!({
                "request_id": j.id,
                "status": status,
                "outcome": outcome,
                "query": j.query,
                "started_at": j.started_at,
                "elapsed_seconds": secs(j.elapsed_ms()),
            })
        })
        .collect();
    tool_json(json!({"searches": jobs}))
}

fn list_search_options() -> Value {
    let groups: Vec<Value> = catalogue()
        .into_iter()
        .map(|g| {
            let options: Vec<Value> = g
                .options
                .iter()
                .filter(|o| o.name != "format")
                .map(|o| {
                    let mut v = json!({
                        "name": o.name,
                        "type": if o.auto { "integer or \"auto\"" } else { o.kind.noun() },
                        "default": o.default,
                        "description": format!("{}. {}", o.label, o.help),
                    });
                    if let Some(values) = &o.values {
                        v["values"] = json!(values);
                    }
                    if let Some(min) = &o.min {
                        v["min"] = min.clone();
                    }
                    if let Some(max) = &o.max {
                        v["max"] = max.clone();
                    }
                    v
                })
                .collect();
            json!({
                "group": g.id,
                "how_to_pass": if g.id == "expert" {
                    "inside start_search's `options` object"
                } else {
                    "as a top-level start_search argument"
                },
                "options": options,
            })
        })
        .collect();
    tool_json(json!({"groups": groups}))
}

fn secs(ms: u64) -> f64 {
    (ms as f64 / 100.0).round() / 10.0
}

/// Total spend, or `None` when a component that ran never reported a price:
/// a partial sum would be read as the total.
fn total_cost(u: &UsageSnapshot) -> Option<f64> {
    let mut sum = u.jev_cost_usd;
    for part in [&u.llm, &u.planner] {
        if part.requests == 0 {
            continue;
        }
        sum += part.cost_usd?;
    }
    Some((sum * 1e6).round() / 1e6)
}

fn cost_json(u: &UsageSnapshot) -> Value {
    json!({
        "total_usd": total_cost(u),
        "jev": {
            "requests": u.jev_requests,
            "input_tokens": u.jev_input_tokens,
            "cost_usd": u.jev_cost_usd,
        },
        "writer": llm_usage_json(&u.llm),
        "planner": llm_usage_json(&u.planner),
    })
}

fn status_json(job: &Job) -> Value {
    let usage = job.usage();
    let Ok(s) = job.state.lock() else {
        return json!({"request_id": job.id, "status": "unknown"});
    };
    let mut v = json!({
        "request_id": job.id,
        "status": s.status.as_str(),
        "query": job.query,
        "started_at": job.started_at,
        "elapsed_seconds": secs(s.elapsed_ms.unwrap_or_else(|| job.started.elapsed().as_millis() as u64)),
        "cost_so_far_usd": total_cost(&usage),
    });
    if !job.options.is_empty() {
        v["options"] = Value::Object(job.options.clone());
    }
    if let Some(ev) = &s.last {
        v["stage"] = json!(ev.stage);
        v["round"] = json!(ev.round);
        v["message"] = json!(ev.message);
        v["pages_read"] = json!(ev.counts.pages);
        v["items_found"] = json!(ev.counts.records);
    }
    v["recent_progress"] = json!(
        s.recent
            .iter()
            .map(|(at, line)| format!("{at}s {line}"))
            .collect::<Vec<_>>()
    );
    match s.status {
        Status::Running => {
            v["next"] =
                json!("Still running. Call get_search_result with this request_id to wait for it.")
        }
        Status::Finished => {
            if let Some(r) = &s.report {
                v["outcome"] = json!(r.outcome.as_str());
                v["summary"] = json!(crate::output::gloss(r));
            }
            v["next"] = json!("Finished. Call get_search_result to read it.");
        }
        Status::Failed => v["error"] = json!(s.error),
        Status::Cancelled => {}
    }
    v
}

fn parse_wait(args: &Map<String, Value>) -> u64 {
    let raw = match args.get("wait_seconds") {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(DEFAULT_WAIT_SECS as f64),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(DEFAULT_WAIT_SECS as f64),
        _ => DEFAULT_WAIT_SECS as f64,
    };
    raw.clamp(0.0, MAX_WAIT_SECS as f64) as u64
}

async fn get_search_result(app: &Arc<AppState>, args: &Map<String, Value>) -> Value {
    match lookup(app, args) {
        Ok(job) => result_for(job, args, parse_wait(args)).await,
        Err(e) => e,
    }
}

/// Wait up to `wait` seconds for a search, then report its result or progress.
async fn result_for(job: Arc<Job>, args: &Map<String, Value>, wait: u64) -> Value {
    let detail = args
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("simple");
    if detail != "simple" && detail != "full" {
        return tool_error("detail must be \"simple\" or \"full\".");
    }
    let format = args.get("format").and_then(Value::as_str);
    if let Some(f) = format
        && !FORMATS.contains(&f)
    {
        return tool_error(format!("format must be one of {}.", FORMATS.join(", ")));
    }

    if wait > 0 && job.is_running() {
        let mut rx = job.done.subscribe();
        let _ = tokio::time::timeout(Duration::from_secs(wait), rx.wait_for(|d| *d)).await;
    }

    let (status, report, rendered, error) = {
        let Ok(s) = job.state.lock() else {
            return tool_error("internal error: search state is unavailable");
        };
        (
            s.status,
            s.report.clone(),
            format.and_then(|f| s.formats.get(f).cloned()),
            s.error.clone(),
        )
    };

    match status {
        Status::Running => {
            let st = status_json(&job);
            let progress = st["message"].as_str().unwrap_or("starting");
            tool_json(json!({
                "request_id": job.id,
                "status": "running",
                "query": job.query,
                "elapsed_seconds": st["elapsed_seconds"],
                "progress": progress,
                "pages_read": st["pages_read"],
                "items_found": st["items_found"],
                "next": "Not finished yet. Call get_search_result again with the same request_id.",
            }))
        }
        Status::Failed => tool_error(format!(
            "The search failed: {}",
            error.unwrap_or_else(|| "unknown error".into())
        )),
        Status::Cancelled => tool_error("The search was cancelled; it has no result."),
        Status::Finished => {
            let Some(report) = report else {
                return tool_error("internal error: finished search has no report");
            };
            if let Some(f) = format {
                return match rendered {
                    Some(doc) => tool_text(doc),
                    None => tool_error(format!("this result could not be rendered as {f}")),
                };
            }
            let shown = shape(&report, row_limit(args, report.mission.target_count));
            if detail == "full" {
                tool_json(full_json(&job, &shown))
            } else {
                tool_text(simple_text(&shown))
            }
        }
    }
}

/// How many list items to return: `None` is all of them.
fn row_limit(args: &Map<String, Value>, target: Option<usize>) -> Option<usize> {
    match args.get("limit").and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    }) {
        Some(0) => None,
        Some(n) => Some(n as usize),
        None => Some(target.unwrap_or(DEFAULT_ROW_LIMIT).max(1)),
    }
}

/// A report cut to what a client should read.
struct Shown {
    report: ScoutReport,
    total: usize,
    complete: usize,
}

/// Complete records first (every requested field found, constraints
/// supported), in the run's own order within each group, then cut to `limit`.
fn shape(report: &ScoutReport, limit: Option<usize>) -> Shown {
    let mut r = report.clone();
    let total = r.records.len();
    let mission = r.mission.clone();
    let complete = r
        .records
        .iter()
        .filter(|x| crate::scout::is_complete(x, &mission))
        .count();
    // Stable: ties keep the run's order.
    r.records
        .sort_by_key(|x| !crate::scout::is_complete(x, &mission));
    if let Some(n) = limit {
        r.records.truncate(n);
    }
    Shown {
        report: r,
        total,
        complete,
    }
}

/// The result text alone. An empty result says what it means rather than
/// returning nothing, which a model would read as a failure. A cut list says
/// so, and how to get the rest.
fn simple_text(shown: &Shown) -> String {
    let report = &shown.report;
    let mut body = crate::output::markdown_body(report);
    if report.records.len() < shown.total {
        body.push_str(&format!(
            "\nShowing {} of {} items found ({} complete, with every requested detail; complete ones first). \
             Call get_search_result with limit 0 for all of them, or format \"csv\" for a spreadsheet.\n",
            report.records.len(),
            shown.total,
            shown.complete
        ));
    }
    if body.trim().is_empty() {
        let g = crate::output::gloss(report);
        let mut c = g.chars();
        return match c.next() {
            Some(first) => format!("{}{}.", first.to_uppercase(), c.as_str()),
            None => String::new(),
        };
    }
    body
}

fn full_json(job: &Job, shown: &Shown) -> Value {
    let report = &shown.report;
    let usage = UsageSnapshot::from_stats(&report.stats);
    let mut v = json!({
        "request_id": job.id,
        "status": "finished",
        "query": report.query,
        "kind": report.mission.kind,
        "outcome": report.outcome.as_str(),
        "summary": crate::output::gloss(report),
        "text": crate::output::markdown_body(report),
        "sources": report.evidence.iter().enumerate().map(|(i, p)| json!({
            "n": i + 1,
            "url": p.url,
            "title": p.title,
            "supports": p.supports,
        })).collect::<Vec<_>>(),
        "notes": report.notes,
        "quarantined_sources": report.quarantined_sources,
        "cost": cost_json(&usage),
        "elapsed_seconds": secs((report.stats.elapsed_secs * 1000.0).round() as u64),
        "stats": report.stats,
        "mission": report.mission,
        "downloads": FORMATS.iter().map(|f| (f.to_string(), json!(format!("/api/runs/{}/download?format={f}", job.id)))).collect::<Map<_, _>>(),
    });
    if let Some(answer) = &report.answer {
        v["answer"] = json!(answer);
    }
    if !report.records.is_empty() || report.mission.kind == crate::types::MissionKind::Harvest {
        v["records"] = json!(report.records);
        v["records_total"] = json!(shown.total);
        v["records_complete"] = json!(shown.complete);
        v["records_shown"] = json!(report.records.len());
    }
    v
}

// ------------------------------------------------------------------ tests --

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(auth: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(a) = auth {
            h.insert(header::AUTHORIZATION, a.parse().unwrap());
        }
        h
    }

    #[test]
    fn constant_time_eq_compares_whole_values() {
        assert!(constant_time_eq(b"secret-token", b"secret-token"));
        assert!(!constant_time_eq(b"secret-token", b"secret-tokeN"));
        assert!(!constant_time_eq(b"secret", b"secret-token"));
    }

    #[test]
    fn bearer_is_read_case_insensitively_and_trimmed() {
        assert_eq!(bearer(&headers(Some("Bearer abc"))), Some("abc"));
        assert_eq!(bearer(&headers(Some("bearer  abc "))), Some("abc"));
        assert_eq!(bearer(&headers(Some("Basic abc"))), None);
        assert_eq!(bearer(&headers(Some("Bearer "))), None);
        assert_eq!(bearer(&headers(None)), None);
    }

    /// No token configured is a closed door, not an open one.
    #[test]
    fn an_unset_token_disables_the_endpoint() {
        let m = McpState::new(None, 4);
        assert!(!m.enabled());
        let err = check_auth(&m, &headers(Some("Bearer anything"))).unwrap();
        assert_eq!(err.status(), StatusCode::SERVICE_UNAVAILABLE);
        let m = McpState::new(Some("   ".into()), 4);
        assert!(!m.enabled(), "a blank token is no token");
    }

    #[test]
    fn a_wrong_or_missing_token_is_401_with_a_challenge() {
        let m = McpState::new(Some("right".into()), 4);
        for h in [headers(None), headers(Some("Bearer wrong"))] {
            let err = check_auth(&m, &h).unwrap();
            assert_eq!(err.status(), StatusCode::UNAUTHORIZED);
            assert!(err.headers().contains_key(header::WWW_AUTHENTICATE));
        }
        assert!(check_auth(&m, &headers(Some("Bearer right"))).is_none());
    }

    #[test]
    fn initialize_echoes_a_known_version_and_offers_the_newest_otherwise() {
        let r = initialize(&json!({"protocolVersion": "2025-03-26"}));
        assert_eq!(r["protocolVersion"], "2025-03-26");
        let r = initialize(&json!({"protocolVersion": "1999-01-01"}));
        assert_eq!(r["protocolVersion"], PROTOCOL_VERSIONS[0]);
        assert_eq!(r["serverInfo"]["name"], "webscout");
        assert!(r["capabilities"]["tools"].is_object());
    }

    #[test]
    fn every_tool_has_a_schema_description_and_annotations() {
        let tools = tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "search",
                "start_search",
                "get_search_result",
                "get_search_status",
                "cancel_search",
                "list_searches",
                "list_search_options"
            ]
        );
        for t in &tools {
            assert_eq!(t["inputSchema"]["type"], "object", "{}", t["name"]);
            assert!(t["description"].as_str().unwrap().len() > 40);
            assert!(t["annotations"].is_object());
        }
    }

    /// The UI's options are start_search arguments; `format` is not (it is
    /// chosen when reading), and expert knobs go through `options`.
    #[test]
    fn start_search_lifts_the_ui_options_to_arguments() {
        let tools = tool_definitions();
        let props = &tools[1]["inputSchema"]["properties"];
        assert_eq!(tools[1]["name"], "start_search");
        // `search` takes the same arguments plus how much detail to return.
        let search = &tools[0]["inputSchema"]["properties"];
        assert!(search.get("detail").is_some());
        assert!(search.get("preset").is_some());
        for name in [
            "query",
            "preset",
            "max_rounds",
            "no_enrich",
            "no_follow",
            "no_search_cache",
            "options",
        ] {
            assert!(props.get(name).is_some(), "missing {name}");
        }
        assert!(props.get("format").is_none());
        assert!(props.get("grounding_floor").is_none());
        assert_eq!(
            props["preset"]["enum"],
            json!(["quick", "standard", "thorough"])
        );
        assert!(props["max_rounds"]["anyOf"].is_array());
    }

    #[test]
    fn total_cost_is_unknown_when_a_component_that_ran_did_not_report() {
        let mut u = UsageSnapshot {
            jev_cost_usd: 0.002,
            ..Default::default()
        };
        u.llm.requests = 1;
        u.llm.cost_usd = Some(0.0005);
        assert_eq!(total_cost(&u), Some(0.0025));
        u.planner.requests = 1;
        assert_eq!(total_cost(&u), None);
    }

    /// A list returns what was asked for, 50 when nothing was, all on 0.
    #[test]
    fn row_limit_follows_the_request() {
        let with = |v: Value| {
            let mut a = Map::new();
            a.insert("limit".into(), v);
            a
        };
        assert_eq!(row_limit(&Map::new(), Some(8)), Some(8));
        assert_eq!(row_limit(&Map::new(), None), Some(DEFAULT_ROW_LIMIT));
        assert_eq!(row_limit(&with(json!(0)), Some(8)), None);
        assert_eq!(row_limit(&with(json!(3)), Some(8)), Some(3));
        assert_eq!(row_limit(&with(json!("20")), None), Some(20));
    }

    #[test]
    fn wait_is_clamped_and_defaults() {
        let m = |v: Value| {
            let mut a = Map::new();
            a.insert("wait_seconds".into(), v);
            parse_wait(&a)
        };
        assert_eq!(parse_wait(&Map::new()), DEFAULT_WAIT_SECS);
        assert_eq!(m(json!(500)), MAX_WAIT_SECS);
        assert_eq!(m(json!(-3)), 0);
        assert_eq!(m(json!("7")), 7);
    }
}
