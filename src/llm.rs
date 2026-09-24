//! Client for the OpenAI-compatible generative endpoint.
//!
//! This model is the only component that can write. It proposes search queries,
//! pulls structured records out of page text, and composes the final answer.
//! Everything it produces passes through Jev before it reaches the user, which is
//! the whole arrangement: generation is useful and untrustworthy, so it is fenced.
//!
//! Two endpoint behaviours drive the design here, both confirmed against the live
//! service rather than assumed:
//!
//! 1. It is a reasoning model. Tokens go to a separate `reasoning` field and
//!    `content` comes back `null` if the budget is exhausted first. A caller that
//!    reads `content` and forgets the budget gets silent empty strings.
//! 2. `chat_template_kwargs.enable_thinking` switches reasoning off entirely, and
//!    `response_format: json_schema` with `strict` constrains decoding. Extraction
//!    wants both: no wasted reasoning, guaranteed-parseable output.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// How this `Llm` instance controls thinking / reasoning on its endpoint.
///
/// Different OpenAI-compatible servers expose reasoning control through different
/// fields. Measured against live endpoints (2026-09-18):
///
/// - **OpenRouter** (`openrouter.ai`): `"reasoning": {"effort": "medium"}` /
///   `{"enabled": false}`. Sending this to an arbitrary server can trigger a
///   400, so it is restricted to the OpenRouter host.
/// - **vLLM** (`chat_template_kwargs.enable_thinking`): required for vLLM-hosted
///   reasoning models. Toggling it off on a measured endpoint cut latency from
///   10.1 s to 2.1 s. Sending provider-specific keys to an unrecognised provider
///   collects a 400, so this stays opt-in.
/// - **Effort** (`"reasoning_effort": "medium"` / `"none"`): an alternative form
///   observed on some self-hosted servers. Measured: `"reasoning_effort": "none"`
///   → 0.36 s on a trivial call; `"reasoning": {"enabled": false}` (OpenRouter
///   form) was silently ignored by the same server.
/// - **Off**: send no reasoning parameter at all (safe default for unknown servers).
/// - **Auto** (the default): behaves as `openrouter` when the endpoint host is
///   `openrouter.ai`, otherwise as `off`. Existing behaviour is preserved for
///   every user who has not set the flag.
#[derive(Debug, Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum ThinkingControl {
    /// OpenRouter form: `reasoning.effort` / `reasoning.enabled`.
    #[value(name = "openrouter")]
    OpenRouter,
    /// vLLM form: `chat_template_kwargs.enable_thinking`.
    ///
    /// Measured 2026-09-18 on a self-hosted vLLM deployment: switching this off cut
    /// a realistic extraction call from 58.3 s (7,713 completion tokens, 23,631
    /// chars of reasoning) to 1.4 s (48 completion tokens) with identical records.
    #[value(name = "vllm")]
    Vllm,
    /// Effort string form: `"reasoning_effort": "medium"` / `"none"`.
    #[value(name = "effort")]
    Effort,
    /// Send no reasoning parameter (safe for servers that do not understand any).
    #[value(name = "off")]
    Off,
    /// Use `openrouter` for OpenRouter hosts, `off` everywhere else (default).
    #[value(name = "auto")]
    Auto,
}

impl Default for ThinkingControl {
    fn default() -> Self {
        Self::Auto
    }
}

/// Resolve `Auto` to a concrete control based on the endpoint host.
pub(crate) fn effective_control(control: ThinkingControl, endpoint: &str) -> ThinkingControl {
    match control {
        ThinkingControl::Auto => {
            if is_openrouter(endpoint) {
                ThinkingControl::OpenRouter
            } else {
                ThinkingControl::Off
            }
        }
        other => other,
    }
}

/// Apply the appropriate reasoning parameter(s) to the request body.
///
/// Pure function — takes the resolved (non-Auto) control and the desired
/// thinking state and mutates only the relevant key(s) in `body`. Used both
/// at call-setup time and during the starvation-escalation retry loop when
/// thinking is toggled off.
pub(crate) fn apply_thinking_to_body(control: ThinkingControl, thinking: bool, body: &mut Value) {
    match control {
        ThinkingControl::OpenRouter => {
            body["reasoning"] = if thinking {
                json!({"effort": "medium"})
            } else {
                json!({"enabled": false})
            };
        }
        ThinkingControl::Vllm => {
            body["chat_template_kwargs"] = json!({"enable_thinking": thinking});
        }
        ThinkingControl::Effort => {
            body["reasoning_effort"] = if thinking {
                json!("medium")
            } else {
                json!("none")
            };
        }
        // Off and Auto add nothing. Auto should have been resolved before this call.
        ThinkingControl::Off | ThinkingControl::Auto => {}
    }
}

#[derive(Debug, Default)]
struct Counters {
    requests: AtomicUsize,
    prompt_tokens: AtomicUsize,
    completion_tokens: AtomicUsize,
    /// Of `completion_tokens`, the part the model spent thinking.
    ///
    /// Worth its own counter: measured 2026-09-18, one extraction call burned
    /// 7,713 completion tokens on 23,631 characters of chain of thought and
    /// returned empty content. A run whose reasoning share is most of its
    /// completions is a run with the wrong `--thinking-control`.
    reasoning_tokens: AtomicUsize,
    /// Accumulated cost in nano-USD (1e-9).
    ///
    /// Integer nanodollars rather than a bit-cast `f64`, because this is summed
    /// from concurrent requests and a compare-and-swap loop on a float is a
    /// worse answer than fixed point at a resolution no invoice has. OpenRouter
    /// reports costs around 2e-06 USD per small call, so a nanodollar is three
    /// orders of magnitude finer than the smallest thing being added.
    cost_nano_usd: AtomicU64,
    /// Whether any response carried a `usage.cost` at all.
    ///
    /// Absent is not zero: a self-hosted vLLM endpoint reports no cost, and
    /// printing "$0.0000" for it would be a claim rather than a gap.
    cost_reported: AtomicBool,
}

/// Convert a reported USD cost to the integer nanodollars the counter holds.
///
/// Negative and non-finite values are dropped rather than wrapped: a provider
/// that reports `NaN` for a cost has told us nothing, and `as u64` would turn
/// that nothing into a zero or a garbage total.
fn nano_usd(cost: f64) -> Option<u64> {
    if !cost.is_finite() || cost < 0.0 {
        return None;
    }
    Some((cost * 1e9).round() as u64)
}

/// Turn accumulated nanodollars back into USD.
fn usd_from_nano(nano: u64) -> f64 {
    nano as f64 / 1e9
}

#[derive(Clone)]
pub struct Llm {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    max_retries: usize,
    counters: Arc<Counters>,
    /// Resolved (non-Auto) control — Auto is collapsed to Off or OpenRouter at
    /// construction time so the hot path in `chat` never branches on endpoint strings.
    thinking_control: ThinkingControl,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: Option<String>,
    /// Chain of thought from vLLM / OpenRouter deployments (`reasoning`).
    #[serde(default)]
    reasoning: Option<String>,
    /// Alternative field name used by some servers (e.g. a self-hosted vLLM
    /// deployment measured 2026-09-18): same semantics, different key.
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl Message {
    /// Return the chain-of-thought bytes, accepting either field name.
    fn reasoning_bytes(&self) -> usize {
        self.reasoning
            .as_deref()
            .or(self.reasoning_content.as_deref())
            .unwrap_or("")
            .len()
    }
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
    /// What the provider charged for this call, in USD.
    ///
    /// Verified live against openrouter.ai 2026-09-21: the `usage` object of a
    /// normal completion carries `cost` with no special request parameter.
    /// Absent on everything else (a self-hosted vLLM endpoint, plain vLLM), and
    /// absent must mean "unknown", never zero — hence `Option`.
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Deserialize, Default)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: usize,
}

/// A reading of one `Llm`'s counters.
///
/// `cost_usd` is `None` when no response from this endpoint has ever reported a
/// cost, and `Some(0.0)` when one did and the model is free. Callers must keep
/// that distinction: "we do not know" and "it was free" are different claims.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LlmUsage {
    pub requests: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub reasoning_tokens: usize,
    pub cost_usd: Option<f64>,
}

/// How a single call should be run.
pub struct Ask {
    pub system: Option<String>,
    pub prompt: String,
    pub max_tokens: usize,
    /// Let the model reason first. Worth it for planning, wasteful for extraction.
    pub thinking: bool,
    pub temperature: f32,
    /// A strict JSON schema to constrain decoding.
    pub schema: Option<Value>,
    /// Stream the response and abort it when it stalls in a whitespace loop
    /// (see `STALL_WHITESPACE`). Off by default; enabled where the loop was
    /// measured.
    pub stall_guard: bool,
}

impl Ask {
    /// A call whose output is prose for a human.
    pub fn prose(prompt: impl Into<String>) -> Self {
        Self {
            system: None,
            prompt: prompt.into(),
            max_tokens: 4096,
            thinking: true,
            temperature: 0.3,
            schema: None,
            stall_guard: false,
        }
    }

    /// A call whose output is a JSON document matching `schema`.
    ///
    /// Thinking defaults off: extraction is mechanical, and reasoning tokens on a
    /// per-chunk call multiply across hundreds of chunks for no gain in accuracy.
    pub fn structured(prompt: impl Into<String>, schema: Value) -> Self {
        Self {
            system: None,
            prompt: prompt.into(),
            max_tokens: 8192,
            thinking: false,
            temperature: 0.0,
            schema: Some(schema),
            stall_guard: false,
        }
    }

    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    pub fn stall_guard(mut self, on: bool) -> Self {
        self.stall_guard = on;
        self
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }
}

impl Llm {
    pub fn new(
        endpoint: String,
        api_key: String,
        model: String,
        max_retries: usize,
        timeout: std::time::Duration,
        thinking_control: ThinkingControl,
    ) -> Result<Self> {
        let resolved_control = effective_control(thinking_control, &endpoint);
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .context("building HTTP client for the generative endpoint")?,
            endpoint,
            api_key,
            model,
            max_retries,
            counters: Arc::new(Counters::default()),
            thinking_control: resolved_control,
        })
    }

    /// Run a call and return the text content.
    pub async fn chat(&self, ask: Ask) -> Result<String> {
        let mut messages = Vec::new();
        if let Some(system) = &ask.system {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": ask.prompt}));

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": ask.max_tokens,
            "temperature": ask.temperature,
        });

        // Apply the thinking-control parameters for the initial attempt.
        // `self.thinking_control` has already been resolved from Auto at
        // construction time, so this is a simple match with no host-string
        // comparison in the hot path.
        let mut effective_thinking = ask.thinking;
        apply_thinking_to_body(self.thinking_control, effective_thinking, &mut body);

        if let Some(schema) = &ask.schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": true, "schema": schema},
            });
        }

        let mut last_err = None;
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let backoff = std::time::Duration::from_millis(500 * (1 << (attempt - 1)).min(16));
                tokio::time::sleep(backoff).await;
            }

            let result = if ask.stall_guard {
                self.attempt_streaming(&body).await
            } else {
                self.attempt(&body).await
            };
            match result {
                Ok(text) => return Ok(text),
                Err(e) => {
                    // The reasoning budget can swallow the whole allowance, leaving
                    // `content` empty with `finish_reason: "length"`. Retrying
                    // unchanged just burns the same tokens again, so escalate
                    // instead: give it more room, and if that is still not enough,
                    // take thinking away entirely. Observed live — a synthesis call
                    // spent 14,246 bytes reasoning and never produced an answer,
                    // failing the whole run at its very last step. The same
                    // starvation can occur on self-hosted servers that always reason
                    // (e.g. a self-hosted vLLM deployment: 58.3 s / 7,713 tokens with
                    // thinking on, 1.4 s / 48 tokens with vllm-style thinking off).
                    if is_starved(&e) {
                        let current = body["max_tokens"].as_u64().unwrap_or(4096);
                        if current < 32_000 {
                            let raised = (current * 3).min(32_000);
                            tracing::debug!(
                                from = current,
                                to = raised,
                                "reasoning consumed the budget; raising max_tokens"
                            );
                            body["max_tokens"] = json!(raised);
                        } else if effective_thinking
                            && self.thinking_control != ThinkingControl::Off
                        {
                            tracing::debug!(
                                control = ?self.thinking_control,
                                "still starved at the ceiling; disabling thinking"
                            );
                            effective_thinking = false;
                            apply_thinking_to_body(self.thinking_control, false, &mut body);
                        }
                        last_err = Some(e);
                        continue;
                    }
                    tracing::debug!(attempt, error = %e, "generative call failed, retrying");
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("generative call failed with no error")))
    }

    /// Add one response's usage to this client's counters.
    fn record_usage(&self, usage: &Usage) {
        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        self.counters
            .prompt_tokens
            .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
        self.counters
            .completion_tokens
            .fetch_add(usage.completion_tokens, Ordering::Relaxed);
        if let Some(d) = &usage.completion_tokens_details {
            self.counters
                .reasoning_tokens
                .fetch_add(d.reasoning_tokens, Ordering::Relaxed);
        }
        // A reported cost of exactly zero (a free model) still counts as
        // reported: the distinction the UI needs is "the endpoint told us"
        // versus "the endpoint never says".
        if let Some(nano) = usage.cost.and_then(nano_usd) {
            self.counters.cost_reported.store(true, Ordering::Relaxed);
            self.counters
                .cost_nano_usd
                .fetch_add(nano, Ordering::Relaxed);
        }
    }

    /// One streamed attempt, aborted as soon as the output stalls.
    ///
    /// JSON-mode decoding sometimes falls into a whitespace loop: the model
    /// writes `{` or `{"records": [` and then only whitespace until the token
    /// ceiling — 8,192 tokens, about 200 s of nothing. It was the dominant
    /// extraction failure: 93 of 113 failed decodes across 108 harvests
    /// (measured 2026-09-24), each losing its pack's records and holding its
    /// round for minutes. A buffered request can only see this at the end;
    /// streamed, it is visible within seconds (`STALL_WHITESPACE`).
    ///
    /// On a stall the partial text is returned when it holds anything past
    /// the opening structure, so `structured`'s truncation salvage keeps the
    /// records written before the loop; otherwise the attempt fails with a
    /// stall error and the caller's retry loop tries again. An aborted
    /// stream sends no usage block: its output tokens are estimated and
    /// counted, its cost cannot be (see the note below).
    async fn attempt_streaming(&self, body: &Value) -> Result<String> {
        let mut body = body.clone();
        body["stream"] = json!(true);
        // Without this the stream carries no usage, and cost reporting for
        // every guarded call would silently go blank. Verified against
        // openrouter.ai 2026-09-24: the final chunk carries usage with cost.
        body["stream_options"] = json!({"include_usage": true});

        let mut resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .header("HTTP-Referer", "https://github.com/typesafe-ai")
            .header("X-Title", "webscout")
            .json(&body)
            .send()
            .await
            .context("sending request to the generative endpoint")?;

        let status = resp.status();
        if !status.is_success() {
            let raw = resp
                .text()
                .await
                .context("reading generative response body")?;
            bail!(
                "generative endpoint returned HTTP {status}: {}",
                truncate(&raw, 400)
            );
        }

        let mut pending: Vec<u8> = Vec::new();
        let mut content = String::new();
        let mut usage: Option<Usage> = None;
        let mut ws_run = 0usize;
        let mut stalled = false;
        'read: while let Some(bytes) = resp
            .chunk()
            .await
            .context("reading generative response body")?
        {
            pending.extend_from_slice(&bytes);
            // Lines split on `\n`, which never occurs inside a multi-byte
            // UTF-8 sequence, so a chunk boundary cannot corrupt text.
            while let Some(nl) = pending.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = pending.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = sse_data(&line) else {
                    continue;
                };
                if data == "[DONE]" {
                    break 'read;
                }
                let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) else {
                    continue;
                };
                if let Some(err) = chunk.error {
                    bail!(
                        "generative endpoint streamed an error: {}",
                        truncate(&err.to_string(), 400)
                    );
                }
                if let Some(u) = chunk.usage {
                    usage = Some(u);
                }
                for choice in &chunk.choices {
                    if let Some(piece) = &choice.delta.content {
                        ws_run = whitespace_run(ws_run, piece);
                        content.push_str(piece);
                    }
                }
                if ws_run >= STALL_WHITESPACE {
                    stalled = true;
                    break 'read;
                }
            }
        }

        match &usage {
            Some(u) => self.record_usage(u),
            // An aborted stream never reaches its usage block. The request
            // and its output are still counted — tokens at the conventional
            // four characters each — but no cost is added: the price is not
            // known here, and an invented one would be read as measured.
            None => {
                self.counters.requests.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .completion_tokens
                    .fetch_add(content.len() / 4, Ordering::Relaxed);
            }
        }

        if stalled {
            let kept = content.trim_end();
            let progressed = stall_progressed(kept);
            tracing::info!(
                chars = content.len(),
                kept = kept.len(),
                progressed,
                "structured output stalled in a whitespace loop; stream aborted"
            );
            if progressed {
                return Ok(kept.to_string());
            }
            bail!("{STALL_TAG}the model stalled in a whitespace loop before writing any output");
        }
        if content.trim().is_empty() {
            bail!("generative endpoint returned no content (streamed)");
        }
        Ok(content)
    }

    async fn attempt(&self, body: &Value) -> Result<String> {
        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            // OpenRouter uses these for attribution; other providers ignore them.
            .header("HTTP-Referer", "https://github.com/typesafe-ai")
            .header("X-Title", "webscout")
            .json(body)
            .send()
            .await
            .context("sending request to the generative endpoint")?;

        let status = resp.status();
        let raw = resp
            .text()
            .await
            .context("reading generative response body")?;
        if !status.is_success() {
            bail!(
                "generative endpoint returned HTTP {status}: {}",
                truncate(&raw, 400)
            );
        }

        let parsed: ChatResponse = serde_json::from_str(&raw)
            .with_context(|| format!("decoding generative response: {}", truncate(&raw, 400)))?;

        if let Some(usage) = &parsed.usage {
            self.record_usage(usage);
        }

        let choice = parsed
            .choices
            .first()
            .context("generative response had no choices")?;
        let content = choice.content_text();

        if content.trim().is_empty() {
            // Almost always the reasoning budget swallowing the whole allowance.
            // Saying so beats returning an empty string that fails mysteriously
            // three stages later. Accept either the standard `reasoning` field or
            // the `reasoning_content` alias used by some servers.
            let reasoned = choice.message.reasoning_bytes();
            bail!(
                "generative model returned no content (finish_reason={:?}, {reasoned} bytes of \
                 reasoning). Raise max_tokens or disable thinking for this call.",
                choice.finish_reason
            );
        }
        Ok(content)
    }

    /// Run a call constrained to `schema` and deserialize the result.
    ///
    /// Two robustness layers around the OpenAI-compatible `response_format`:
    /// guided decoding rarely fails on providers that support it, but the
    /// deployment landscape is not uniform.
    ///
    /// 1. If the provider returns 400 with a body mentioning `response_format`,
    ///    `json_schema`, or "unsupported", retry ONCE without response_format,
    ///    prepending the schema to the prompt and asking for a bare JSON
    ///    object. Cheap insurance for any endpoint that has not implemented
    ///    guided decoding: better a slightly-flaky JSON parse than a hard
    ///    failure at the first classify.
    /// 2. If the returned text parses but is wrapped in prose (a `{...}` block
    ///    embedded in a preamble), extract the first balanced `{...}` block
    ///    and retry the parse. Observed on OpenRouter reasoning models that
    ///    leak a chain-of-thought preamble into `content` when
    ///    `response_format` was ignored.
    pub async fn structured<T: DeserializeOwned>(
        &self,
        prompt: impl Into<String>,
        schema: Value,
    ) -> Result<T> {
        self.structured_ask(Ask::structured(prompt, schema)).await
    }

    /// `structured` with a caller-built `Ask`, for calls that need a setting
    /// the plain form does not expose — extraction's `stall_guard`. The ask
    /// must carry a schema; its settings travel to the schema-in-prompt
    /// fallback too.
    pub async fn structured_ask<T: DeserializeOwned>(&self, ask: Ask) -> Result<T> {
        let prompt = ask.prompt.clone();
        let schema = ask.schema.clone().unwrap_or_else(|| json!({}));
        let guard = ask.stall_guard;
        let attempt = self.chat(ask).await;
        let text = match attempt {
            Ok(t) => t,
            Err(e) if is_response_format_unsupported(&e) => {
                tracing::info!(
                    error = %e,
                    "endpoint rejected response_format; retrying schema-in-prompt"
                );
                let schema_str = serde_json::to_string(&schema).unwrap_or_default();
                let framed = format!(
                    "Return ONLY a JSON object matching this schema, no prose, no code fence:\n\
                     {schema_str}\n\n{prompt}"
                );
                self.chat(Ask::prose(framed).thinking(false).stall_guard(guard))
                    .await?
            }
            Err(e) => return Err(e),
        };
        // Guided decoding makes this reliable, but a model can still wrap JSON in a
        // fence when the schema is loose, so strip that before giving up.
        let cleaned = strip_code_fence(&text);
        match serde_json::from_str(cleaned) {
            Ok(v) => Ok(v),
            Err(_) => {
                // Second attempt: pull the first balanced {...} out of prose.
                let candidate: &str = match extract_first_json_object(cleaned) {
                    Some(inner) => inner,
                    None => cleaned,
                };
                if let Ok(v) = serde_json::from_str(candidate) {
                    return Ok(v);
                }
                // Third attempt: the response was cut mid-array — degenerate
                // fill, a max-tokens ceiling, or both. Salvage the objects
                // that did complete rather than losing the whole page.
                // Measured 2026-09-21 (q101): a model padding a response
                // with DEL runs until truncation cost that extraction its
                // entire page, seven times in one run.
                if let Some(salvaged) =
                    salvage_truncated_json(candidate).and_then(|s| serde_json::from_str(&s).ok())
                {
                    tracing::debug!("salvaged truncated structured output");
                    return Ok(salvaged);
                }
                serde_json::from_str(candidate).with_context(|| {
                    format!("decoding structured output: {}", truncate(candidate, 400))
                })
            }
        }
    }

    /// What this client has spent so far.
    ///
    /// Every field is a relaxed load of an atomic, so this is cheap enough to
    /// call on a timer while requests are in flight — which is exactly what the
    /// API's live `usage` event does.
    pub fn stats(&self) -> LlmUsage {
        LlmUsage {
            requests: self.counters.requests.load(Ordering::Relaxed),
            prompt_tokens: self.counters.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.counters.completion_tokens.load(Ordering::Relaxed),
            reasoning_tokens: self.counters.reasoning_tokens.load(Ordering::Relaxed),
            cost_usd: self
                .counters
                .cost_reported
                .load(Ordering::Relaxed)
                .then(|| usd_from_nano(self.counters.cost_nano_usd.load(Ordering::Relaxed))),
        }
    }
}

/// One server-sent event of a streamed completion.
#[derive(Deserialize, Default)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize, Default)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
}

#[derive(Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Consecutive whitespace characters that mark a stalled structured output.
///
/// Legitimate JSON never holds this much whitespace in a row — even
/// pretty-printed, the run between two tokens is a newline and an indent —
/// while a whitespace loop reaches it within seconds.
const STALL_WHITESPACE: usize = 256;

/// Non-whitespace characters a stalled output may hold and still count as
/// having written nothing: the opening `{"records": [` is 13.
const STALL_OPENING: usize = 16;

/// Error-message tag for a stall that produced nothing usable.
const STALL_TAG: &str = "[stalled] ";

/// The payload of one server-sent-event line, or `None` for blanks,
/// comments (OpenRouter sends `: OPENROUTER PROCESSING` keep-alives) and
/// other fields.
fn sse_data(line: &str) -> Option<&str> {
    line.trim_end_matches(['\r', '\n'])
        .strip_prefix("data:")
        .map(str::trim)
}

/// The whitespace run after appending `piece`: extended by trailing
/// whitespace, reset by any non-whitespace character inside it.
fn whitespace_run(run: usize, piece: &str) -> usize {
    match piece.rfind(|c: char| !c.is_whitespace()) {
        Some(i) => {
            let after = &piece[i..];
            after.chars().skip(1).count()
        }
        None => run + piece.chars().count(),
    }
}

/// Did a stalled output get past its opening structure? Only then is it
/// worth handing to the truncation salvage.
fn stall_progressed(kept: &str) -> bool {
    kept.chars().filter(|c| !c.is_whitespace()).count() > STALL_OPENING
}

impl Choice {
    fn content_text(&self) -> String {
        self.message.content.clone().unwrap_or_default()
    }
}

/// True when the call produced no content because reasoning ate the token budget.
/// Matched on the message this module raises, which names both conditions.
fn is_starved(e: &anyhow::Error) -> bool {
    let msg = e.to_string();
    msg.contains("returned no content") && msg.contains("finish_reason")
}

/// True when the endpoint host is OpenRouter. Used to decide whether it is
/// safe to send the `reasoning` extension — a random OpenAI-compatible
/// endpoint that has never seen the field is within its rights to 400.
pub(crate) fn is_openrouter(endpoint: &str) -> bool {
    endpoint.contains("openrouter.ai")
}

/// True when the error looks like the endpoint rejected `response_format`.
///
/// The `chat` path raises `"generative endpoint returned HTTP 400: <body>"`
/// on any 4xx; we look for the specific keywords that identify a schema
/// rejection so a genuine bad prompt (also 400) still surfaces as a hard
/// error. Case-insensitive; body texts vary between OpenRouter, Together,
/// Fireworks, DeepInfra.
pub(crate) fn is_response_format_unsupported(e: &anyhow::Error) -> bool {
    let msg = e.to_string().to_lowercase();
    if !msg.contains("http 400") {
        return false;
    }
    msg.contains("response_format")
        || msg.contains("json_schema")
        || msg.contains("json schema")
        || msg.contains("unsupported")
        || msg.contains("does not support")
}

/// Repair a truncated JSON document by cutting at the last completely
/// closed inner object and closing whatever containers are still open.
///
/// This is the recovery of last resort in `structured`: a response that hit
/// its token ceiling mid-array (sometimes after degenerating into filler)
/// still contains every object that finished before the cut. Losing those
/// because the tail is broken was measured at a page per occurrence
/// (2026-09-21, q101: seven extraction calls returned DEL-padding until
/// truncation and each lost its page's records).
///
/// Returns `None` when the document is not truncated (the cut point would be
/// the end of the input — nothing to salvage) or contains no closed inner
/// object.
pub(crate) fn salvage_truncated_json(s: &str) -> Option<String> {
    let mut in_str = false;
    let mut escape = false;
    let mut stack: Vec<char> = Vec::new();
    // Byte offset just past the last `}` that closed an object nested inside
    // an still-open container, with the container stack at that moment.
    let mut last_safe: Option<(usize, Vec<char>)> = None;
    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match c {
            '\\' if in_str => escape = true,
            '"' => in_str = !in_str,
            '{' | '[' if !in_str => stack.push(c),
            '}' | ']' if !in_str => {
                stack.pop();
                if c == '}' && !stack.is_empty() {
                    last_safe = Some((i + c.len_utf8(), stack.clone()));
                }
            }
            _ => {}
        }
    }
    let (cut, open) = last_safe?;
    if cut >= s.len() {
        // The document ends at a complete inner object: it either parses
        // already or is broken some other way truncation repair cannot help.
        return None;
    }
    let mut out = s[..cut].to_string();
    for c in open.iter().rev() {
        out.push(if *c == '{' { '}' } else { ']' });
    }
    Some(out)
}

/// Extract the first balanced `{...}` block from `s`, honouring string
/// literals so an unmatched `{` inside a JSON string does not confuse the
/// scanner. Returns `None` if no balanced object is found.
///
/// This is a fallback for the rare case where the model wraps JSON in a prose
/// preamble that the code-fence stripper does not cover ("Here is the JSON:
/// { ... }"). It is deliberately not clever: it does not try to reconstruct
/// broken JSON, only to find the object boundary.
pub(crate) fn extract_first_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut start: Option<usize> = None;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => {
                if start.is_none() {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0
                    && let Some(st) = start
                {
                    return Some(&s[st..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn strip_code_fence(s: &str) -> &str {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim();
        }
    }
    t
}

pub fn truncate(s: &str, n: usize) -> &str {
    if s.len() <= n {
        s
    } else {
        // Respect char boundaries; page text is full of multi-byte characters and
        // slicing blind panics on them.
        let mut end = n;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only `data:` lines carry payload; OpenRouter's `: OPENROUTER
    /// PROCESSING` keep-alives and blank separators must be skipped.
    #[test]
    fn sse_data_reads_payload_lines_only() {
        assert_eq!(sse_data("data: {\"a\":1}\n"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data: [DONE]\r\n"), Some("[DONE]"));
        assert_eq!(sse_data(": OPENROUTER PROCESSING\n"), None);
        assert_eq!(sse_data("\n"), None);
        assert_eq!(sse_data("event: ping\n"), None);
    }

    /// The run counts only trailing whitespace, carries across stream
    /// chunks, and any real character resets it — pretty-printed JSON never
    /// comes near the stall threshold.
    #[test]
    fn whitespace_run_tracks_trailing_whitespace_across_chunks() {
        let mut run = 0;
        for piece in ["{\"records\": [", "\n", "  ", "\n\n"] {
            run = whitespace_run(run, piece);
        }
        assert_eq!(run, 5, "a loop's whitespace accumulates across chunks");
        run = whitespace_run(run, "  {\"name\": \"A\"}\n    ");
        assert_eq!(run, 5, "a real token resets the run to what follows it");
        // A pretty-printed record never approaches the stall threshold.
        let pretty = "{\n  \"records\": [\n    {\n      \"name\": \"A\"\n    }\n  ]\n}";
        let mut run = 0;
        let mut worst = 0;
        for c in pretty.chars() {
            run = whitespace_run(run, &c.to_string());
            worst = worst.max(run);
        }
        assert!(worst < STALL_WHITESPACE / 10, "worst run {worst}");
        // A whitespace loop gets there.
        let mut run = 0;
        for _ in 0..300 {
            run = whitespace_run(run, "\n");
        }
        assert!(run >= STALL_WHITESPACE);
    }

    /// A stall right after the opening structure has nothing to salvage; one
    /// after complete records does, and must go to the salvage path.
    #[test]
    fn a_stall_counts_as_progress_only_past_the_opening() {
        assert!(!stall_progressed("{"));
        assert!(!stall_progressed("{\"records\": ["));
        assert!(!stall_progressed("{ \"records\" : [ "));
        assert!(stall_progressed(
            "{\"records\": [{\"name\": \"CREC Gracia\", \"contact_email\": \"\"}"
        ));
    }

    #[test]
    fn nano_usd_round_trips_an_openrouter_cost() {
        // The exact figure observed on openrouter.ai 2026-09-21 for a 17-token call.
        let nano = nano_usd(2.18e-06).expect("a real cost converts");
        assert_eq!(nano, 2_180);
        assert!((usd_from_nano(nano) - 2.18e-06).abs() < 1e-12);
    }

    #[test]
    fn nano_usd_accumulates_without_drift() {
        // A thousand small calls must still add up to the arithmetic answer;
        // this is the whole reason the counter is fixed point.
        let one = nano_usd(2.18e-06).unwrap();
        assert!((usd_from_nano(one * 1000) - 2.18e-03).abs() < 1e-12);
    }

    #[test]
    fn nano_usd_rejects_nonsense_rather_than_wrapping() {
        assert_eq!(nano_usd(-1.0), None);
        assert_eq!(nano_usd(f64::NAN), None);
        assert_eq!(nano_usd(f64::INFINITY), None);
        // Free is a real answer, not a missing one.
        assert_eq!(nano_usd(0.0), Some(0));
    }

    #[test]
    fn usage_reads_openrouter_cost_and_reasoning_tokens() {
        // Verbatim shape of an openrouter.ai usage object.
        let u: Usage = serde_json::from_str(
            r#"{"prompt_tokens":15,"completion_tokens":2,"total_tokens":17,
                "cost":2.18e-06,
                "cost_details":{"upstream_inference_cost":2.18e-06},
                "completion_tokens_details":{"reasoning_tokens":7713}}"#,
        )
        .unwrap();
        assert_eq!(u.prompt_tokens, 15);
        assert_eq!(u.completion_tokens, 2);
        assert_eq!(u.cost, Some(2.18e-06));
        assert_eq!(
            u.completion_tokens_details
                .map(|d| d.reasoning_tokens)
                .unwrap_or(0),
            7_713
        );
    }

    #[test]
    fn usage_without_a_cost_is_unknown_not_zero() {
        // What a plain vLLM deployment (a self-hosted vLLM endpoint) sends.
        let u: Usage =
            serde_json::from_str(r#"{"prompt_tokens":15,"completion_tokens":2}"#).unwrap();
        assert_eq!(u.cost, None, "absent must not decay to 0.0");
        assert!(u.completion_tokens_details.is_none());
    }

    #[test]
    fn a_fresh_client_reports_no_cost_at_all() {
        let llm = Llm::new(
            "https://openrouter.ai/api/v1/chat/completions".into(),
            "k".into(),
            "m".into(),
            1,
            std::time::Duration::from_secs(1),
            ThinkingControl::Off,
        )
        .unwrap();
        let u = llm.stats();
        assert_eq!(u, LlmUsage::default());
        assert_eq!(
            u.cost_usd, None,
            "an endpoint that has said nothing has reported no cost"
        );
    }

    #[test]
    fn is_openrouter_detects_host() {
        assert!(is_openrouter(
            "https://openrouter.ai/api/v1/chat/completions"
        ));
        assert!(!is_openrouter("https://api.openai.com/v1/chat/completions"));
        assert!(!is_openrouter("http://localhost:8080/v1/chat/completions"));
    }

    #[test]
    fn is_response_format_unsupported_matches_400_bodies() {
        // Bodies observed in the wild from various OpenAI-compatible providers.
        let e =
            anyhow::anyhow!("generative endpoint returned HTTP 400: response_format not supported");
        assert!(is_response_format_unsupported(&e));
        let e = anyhow::anyhow!(
            "generative endpoint returned HTTP 400: this model does not support json_schema"
        );
        assert!(is_response_format_unsupported(&e));
        let e = anyhow::anyhow!("generative endpoint returned HTTP 400: some other bad request");
        assert!(!is_response_format_unsupported(&e));
        let e = anyhow::anyhow!("generative endpoint returned HTTP 500: server error");
        assert!(!is_response_format_unsupported(&e));
    }

    #[test]
    fn salvage_recovers_records_before_the_cut() {
        // Cut mid-object, as a max-tokens ceiling leaves it.
        let broken = r#"{"records": [{"name": "A"}, {"name": "B"}, {"na"#;
        let fixed = salvage_truncated_json(broken).expect("should salvage");
        assert_eq!(fixed, r#"{"records": [{"name": "A"}, {"name": "B"}]}"#);
        let v: serde_json::Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(v["records"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn salvage_skips_the_partial_object_after_a_comma() {
        let broken = r#"{"records": [{"name": "A"}, {"name": "tru"#;
        let fixed = salvage_truncated_json(broken).expect("should salvage");
        assert_eq!(fixed, r#"{"records": [{"name": "A"}]}"#);
    }

    #[test]
    fn salvage_honours_braces_inside_strings() {
        let broken = r#"{"records": [{"note": "a } brace and \" quote"}, {"x"#;
        let fixed = salvage_truncated_json(broken).expect("should salvage");
        let v: serde_json::Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(v["records"][0]["note"], r#"a } brace and " quote"#);
    }

    #[test]
    fn salvage_returns_none_when_there_is_nothing_to_cut() {
        assert!(salvage_truncated_json(r#"{"records": []}"#).is_none());
        // No closed inner object at all: nothing to cut at.
        assert!(salvage_truncated_json(r#"{"records": [{"na"#).is_none());
        // The last inner object ends at the last byte: the break is
        // elsewhere (here, an unclosed outer), and truncation repair
        // cannot help — the caller reports the parse error.
        assert!(salvage_truncated_json(r#"{"a": {"b": 1}"#).is_none());
    }

    #[test]
    fn salvage_recovers_before_degenerate_filler() {
        // q101's shape: a value degenerates into DEL padding, then truncates.
        let filler: String = "\u{7f}".repeat(200);
        let broken =
            format!(r#"{{"records": [{{"name": "Consul"}}, {{"name": "X", "note": "{filler}"#);
        let fixed = salvage_truncated_json(&broken).expect("should salvage");
        let v: serde_json::Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(v["records"].as_array().unwrap().len(), 1);
        assert_eq!(v["records"][0]["name"], "Consul");
    }

    #[test]
    fn extract_first_json_object_pulls_out_wrapped_json() {
        let s = "Sure, here is the JSON: {\"a\":1,\"b\":\"c\"} — hope this helps";
        assert_eq!(extract_first_json_object(s), Some("{\"a\":1,\"b\":\"c\"}"));

        // Nested braces stay balanced.
        let s = "prefix {\"outer\": {\"inner\": 2}} suffix";
        assert_eq!(
            extract_first_json_object(s),
            Some("{\"outer\": {\"inner\": 2}}")
        );

        // Unbalanced closing brace inside a string does not confuse it.
        let s = r#"{"k": "a } b", "n": 1}"#;
        assert_eq!(
            extract_first_json_object(s),
            Some(r#"{"k": "a } b", "n": 1}"#)
        );

        // Missing close → None.
        assert_eq!(extract_first_json_object("prefix {\"a\": 1"), None);
        // No object at all → None.
        assert_eq!(extract_first_json_object("no braces here"), None);
    }

    #[test]
    fn strip_code_fence_handles_json_marker() {
        assert_eq!(strip_code_fence("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fence("plain text"), "plain text");
    }

    #[test]
    fn message_reasoning_bytes_prefers_reasoning_field() {
        let m = Message {
            content: Some("hi".into()),
            reasoning: Some("abc".into()),
            reasoning_content: Some("longer".into()),
        };
        // When both are present, `reasoning` wins — it is the primary field.
        assert_eq!(m.reasoning_bytes(), 3);
    }

    #[test]
    fn message_reasoning_bytes_falls_back_to_reasoning_content() {
        let m = Message {
            content: None,
            reasoning: None,
            reasoning_content: Some("chain of thought".into()),
        };
        assert_eq!(m.reasoning_bytes(), "chain of thought".len());
    }

    #[test]
    fn message_reasoning_bytes_zero_when_neither() {
        let m = Message {
            content: Some("x".into()),
            reasoning: None,
            reasoning_content: None,
        };
        assert_eq!(m.reasoning_bytes(), 0);
    }

    #[test]
    fn message_deserializes_reasoning_content_alias() {
        // Simulate a response from a server that uses `reasoning_content` instead
        // of `reasoning` (e.g. a self-hosted vLLM deployment).
        let json = r#"{"content":"","reasoning_content":"some chain of thought"}"#;
        let m: Message = serde_json::from_str(json).unwrap();
        assert_eq!(m.reasoning, None);
        assert_eq!(
            m.reasoning_content.as_deref(),
            Some("some chain of thought")
        );
        assert_eq!(m.reasoning_bytes(), "some chain of thought".len());
    }

    // --- ThinkingControl tests ---

    fn body_for_control(control: ThinkingControl, thinking: bool) -> Value {
        let mut body = json!({"model": "x", "messages": []});
        apply_thinking_to_body(control, thinking, &mut body);
        body
    }

    #[test]
    fn openrouter_thinking_on() {
        let b = body_for_control(ThinkingControl::OpenRouter, true);
        assert_eq!(b["reasoning"], json!({"effort": "medium"}));
        assert!(b.get("chat_template_kwargs").is_none());
        assert!(b.get("reasoning_effort").is_none());
    }

    #[test]
    fn openrouter_thinking_off() {
        let b = body_for_control(ThinkingControl::OpenRouter, false);
        assert_eq!(b["reasoning"], json!({"enabled": false}));
    }

    #[test]
    fn vllm_thinking_on() {
        let b = body_for_control(ThinkingControl::Vllm, true);
        assert_eq!(b["chat_template_kwargs"], json!({"enable_thinking": true}));
        assert!(b.get("reasoning").is_none());
        assert!(b.get("reasoning_effort").is_none());
    }

    #[test]
    fn vllm_thinking_off() {
        let b = body_for_control(ThinkingControl::Vllm, false);
        assert_eq!(b["chat_template_kwargs"], json!({"enable_thinking": false}));
    }

    #[test]
    fn effort_thinking_on() {
        let b = body_for_control(ThinkingControl::Effort, true);
        assert_eq!(b["reasoning_effort"], json!("medium"));
        assert!(b.get("reasoning").is_none());
        assert!(b.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn effort_thinking_off() {
        let b = body_for_control(ThinkingControl::Effort, false);
        assert_eq!(b["reasoning_effort"], json!("none"));
    }

    #[test]
    fn off_adds_nothing() {
        let b = body_for_control(ThinkingControl::Off, true);
        assert!(b.get("reasoning").is_none());
        assert!(b.get("chat_template_kwargs").is_none());
        assert!(b.get("reasoning_effort").is_none());
        let b = body_for_control(ThinkingControl::Off, false);
        assert!(b.get("reasoning").is_none());
    }

    #[test]
    fn auto_adds_nothing_on_non_openrouter() {
        let b = body_for_control(ThinkingControl::Auto, true);
        assert!(b.get("reasoning").is_none());
        assert!(b.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn effective_control_auto_resolves_for_openrouter() {
        let ctrl = effective_control(
            ThinkingControl::Auto,
            "https://openrouter.ai/api/v1/chat/completions",
        );
        assert_eq!(ctrl, ThinkingControl::OpenRouter);
    }

    #[test]
    fn effective_control_auto_resolves_for_other_host() {
        let ctrl = effective_control(
            ThinkingControl::Auto,
            "https://vllm.example.org/v1/chat/completions",
        );
        assert_eq!(ctrl, ThinkingControl::Off);
        let ctrl = effective_control(
            ThinkingControl::Auto,
            "http://localhost:11434/v1/chat/completions",
        );
        assert_eq!(ctrl, ThinkingControl::Off);
    }

    #[test]
    fn effective_control_non_auto_passes_through() {
        assert_eq!(
            effective_control(ThinkingControl::Vllm, "https://anything.example.com"),
            ThinkingControl::Vllm
        );
        assert_eq!(
            effective_control(
                ThinkingControl::Effort,
                "https://openrouter.ai/api/v1/chat/completions"
            ),
            ThinkingControl::Effort
        );
    }

    #[test]
    fn is_starved_triggers_on_reasoning_content_server_error() {
        // The error message is the same regardless of which reasoning field the
        // server used — only the byte count differs in the text.
        let e = anyhow::anyhow!(
            "generative model returned no content (finish_reason=Some(\"length\"), \
             14246 bytes of reasoning). Raise max_tokens or disable thinking for this call."
        );
        assert!(is_starved(&e));
    }
}
