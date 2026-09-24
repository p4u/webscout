//! Client for TypeSafe's System One API — the verification layer.
//!
//! Jev returns calibrated judgments, never text. In this tool it does four jobs
//! the generative model cannot be trusted with:
//!
//! - deciding which search results are worth fetching,
//! - refusing to let a page that addresses the model reach the model at all,
//! - checking that every extracted record is actually present in its source,
//! - deciding whether the collected evidence really answers the question.
//!
//! The third is the important one. A generative model asked to pull a hundred
//! email addresses out of page text will, somewhere around the eightieth, produce
//! one that looks right and is not there. Jev finds those.
//!
//! One property shapes every call site: questions batched into a single request
//! run in parallel server-side and cost a fraction of the same questions sent
//! separately. So grounding checks for every record from one page go out together.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::config;

/// Hard limit on the state portion of one request, measured 2026-09-18.
///
/// A state of 31,462 tokens with one question succeeded; the state plus the
/// longest single question must together fit in this window. Using this as the
/// ceiling for the state-only serialisation catches the common overrun where a
/// large page text is dropped in verbatim.
pub const MAX_STATE_TOKENS: usize = 32_768;

/// Hard limit on a full request (state + all questions), measured 2026-09-18.
///
/// Requests of 44,485 and 56,375 input tokens succeeded; the previous assumption
/// of a 32k whole-request limit was wrong and was wasting half the usable capacity
/// on question-heavy batches. This is a *request size* limit, not an account-credit
/// limit — the dashboard still shows funds while every oversized call fails.
pub const MAX_REQUEST_TOKENS: usize = 65_536;

/// Retained for callers compiled before the 2026-09-18 measurement updated the
/// limit model. Equals `MAX_STATE_TOKENS`; prefer the named constants in new code.
pub const MAX_INPUT_TOKENS: usize = MAX_STATE_TOKENS;

/// Characters allowed in one request body, as a conservative fallback.
///
/// The divisor is the measured worst case: a public register of cooperative names
/// and email addresses came in at **2.00 characters per token** consistently —
/// addresses like `bagnusarc2021@gmail.com` shatter into many tokens apiece.
/// English prose is closer to 3.7, but budgeting at 3.0 felt safe until those
/// email-dense pages drove the server to reject requests that had passed the local
/// check. One run quarantined 77 chunks and lost two thirds of its records.
///
/// `Jev::request_budget_chars` / `state_budget_chars` use an adaptive EMA that
/// climbs toward the observed ratio as runs accumulate, so ordinary-prose batches
/// eventually get their full size back. This free function is the conservative
/// seed, used by tests that have no `Jev` handle.
#[cfg(test)]
pub fn budget_chars() -> usize {
    MAX_INPUT_TOKENS * 2
}

/// Update one step of the chars-per-token ratio EMA and clamp to the safe range.
///
/// Alpha 0.3 adapts quickly enough to a new content type (≈10 requests to close
/// 90 % of the gap to the true ratio) while damping one-off noise. The clamp
/// bounds are the measured extremes: 2.0 for email-dense registers (measured worst
/// case), 3.5 for ordinary English prose; outside that range we have no
/// calibration data and conservatism wins.
///
/// A pure function so the EMA math is testable without a network.
pub(crate) fn update_ratio_ema(current: f64, observed: f64) -> f64 {
    const ALPHA: f64 = 0.3;
    (current * (1.0 - ALPHA) + observed * ALPHA).clamp(2.0, 3.5)
}

/// Marks an error as the server's verdict on this request, so the retry loop knows
/// resending is pointless. A sentinel in the message rather than a typed error
/// because `anyhow` carries these across several layers and a string survives that
/// without threading a custom type through every call site.
const CLIENT_ERROR_TAG: &str = "[client-error] ";

fn is_client_error(e: &anyhow::Error) -> bool {
    e.to_string().contains(CLIENT_ERROR_TAG)
}

/// Marks a request the server rejected for size, so a batching caller can split
/// the batch and resend instead of treating every item as failed.
const OVERSIZED_TAG: &str = "[oversized] ";

pub fn is_oversized(e: &anyhow::Error) -> bool {
    e.to_string().contains(OVERSIZED_TAG)
}

/// Serialized size of one question, for batch planning.
///
/// Questions travel in the same request as the state and count against the same
/// limit, and their size scales with the number of items — a batch of 40 chunks
/// carries 80 questions. Budgeting only the state and reserving a flat headroom for
/// the questions works until the batch grows, then silently overruns: raising the
/// chunk cap to 40 produced 107,231-character requests against a ~98,304 ceiling.
///
/// Measuring beats estimating here, because the questions are already built.
pub fn question_cost(id: &str, q: &Value) -> usize {
    // +8 covers the JSON key quoting, colon and comma around the entry.
    id.len() + q.to_string().len() + 8
}

/// Serialized size of a value destined for the state.
pub fn state_cost(v: &impl serde::Serialize) -> usize {
    serde_json::to_string(v).map(|s| s.len()).unwrap_or(0)
}

/// Build a yes/no question. `yes` and `no` describe the two outcomes; filling them
/// in is most of what separates a calibrated noul from a vague one, because the
/// model is applying your definition rather than guessing at one.
pub fn noul(instructions: &str, yes: &str, no: &str) -> Value {
    json!({
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": yes, "false": no},
    })
}

/// Build a question that selects one labelled option.
pub fn choice(instructions: &str, options: &[(&str, &str)]) -> Value {
    let criteria: Map<String, Value> = options
        .iter()
        .map(|(k, v)| ((*k).to_string(), Value::String((*v).to_string())))
        .collect();
    json!({"type": "choice", "instructions": instructions, "criteria": criteria})
}

/// Build the entity-binding question: what relationship does the organisation
/// a passage describes have to the entity we asked about?
///
/// `subject_ref` names the state key holding the text under judgment
/// (`passage` in discovery grounding, `text` in enrichment), so the question
/// points Jev at the same material the caller put in the state.
///
/// This exists because "is this value attributed to this entity" is not the
/// same question as "is this the right organisation", and only the first was
/// ever asked. Measured live (2026-09-20): Jumilla vs the Consejo Regulador
/// de la Denominación de Origen Jumilla scores `related` 0.73 / `different`
/// 0.25; Terrassa vs the insurer Egarsat `related` 0.53 / `different` 0.36;
/// Elche vs the Diputación de Albacete `different` 0.70 / `related` 0.26;
/// Bilbao vs Bilboko Udala and Sant Adrià de Besòs vs its own OAC page both
/// `same` 1.00. All three shipped misattributions fall out on `related` /
/// `different`; both correct ones are unambiguous. Note the Terrassa margin
/// (0.53 vs 0.36) — callers must accept on `same` being the chosen option,
/// never on a probability threshold over the rejecting options.
pub fn entity_binding(entity: &str, topic: &str, subject_ref: &str) -> Value {
    choice(
        &format!(
            "What relationship does the organisation described in `{subject_ref}` have to \
             `{entity}`, a {topic}?"
        ),
        &[
            (
                "same",
                "It is that organisation itself, including an official department or office of it.",
            ),
            (
                "related",
                "It is a parent, member, subsidiary, federation, association, supplier, \
                 contractor or other related but different organisation.",
            ),
            (
                "different",
                "It is an unrelated organisation, or a different place or body that merely has \
                 a similar name.",
            ),
            (
                "unresolved",
                "The passage does not establish which organisation it is about.",
            ),
        ],
    )
}

/// Build a question that rates against ordered levels.
///
/// Levels must describe concrete situations that stand on their own. "Medium
/// quality" conveys nothing; "a personal blog post with no sources" does.
pub fn score(instructions: &str, levels: &[&str]) -> Value {
    json!({"type": "score", "instructions": instructions, "criteria": levels})
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct Answer {
    #[serde(default)]
    pub noul: f64,
    #[serde(default)]
    pub choice: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub probabilities: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct SystemOneResponse {
    #[serde(default)]
    answers: HashMap<String, Answer>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Debug, Deserialize, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: usize,
}

/// Answers keyed by the question id you supplied.
#[derive(Debug, Clone, Default)]
pub struct Answers(HashMap<String, Answer>);

impl Answers {
    /// Probability of yes, or 0 when the question was not answered.
    ///
    /// A missing answer reads as 0 rather than an error because speculative
    /// questions are asked that may never be consumed; a hard failure there would
    /// make batching unusable.
    ///
    /// Use this only where 0 is the *safe* reading. For a question whose low end
    /// means "allow", reach for [`Answers::noul_or`] instead and pass the cautious
    /// default — a missing `injection` answer silently meaning "not an injection"
    /// is precisely the failure this distinction exists to prevent.
    pub fn noul(&self, id: &str) -> f64 {
        self.0.get(id).map(|a| a.noul).unwrap_or(0.0)
    }

    /// Probability of yes, or `missing` when the question was not answered.
    ///
    /// Jev is asked many questions in one request and answers are keyed by id. If a
    /// key is absent — a malformed response, a truncated batch, an id that did not
    /// round-trip — the caller decides what an absent answer means, rather than
    /// inheriting a default that happens to be permissive.
    pub fn noul_or(&self, id: &str, missing: f64) -> f64 {
        self.0.get(id).map(|a| a.noul).unwrap_or(missing)
    }

    /// Reject a response whose numbers are not usable.
    ///
    /// Borrowed from jev-ultrafast's `validate_choice`: a probability outside 0..1,
    /// a NaN, or an infinity means something went wrong upstream, and acting on it
    /// is worse than not acting. Cheap to check, and it turns a silent wrong
    /// decision into a visible one.
    pub fn is_sane(&self, id: &str) -> bool {
        self.0.get(id).is_some_and(|a| {
            let ok = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
            ok(a.noul) && ok(a.confidence) && a.probabilities.values().copied().all(ok)
        })
    }

    pub fn score(&self, id: &str) -> f64 {
        self.0.get(id).map(|a| a.score).unwrap_or(0.0)
    }

    pub fn choice(&self, id: &str) -> String {
        self.0.get(id).map(|a| a.choice.clone()).unwrap_or_default()
    }

    pub fn confidence(&self, id: &str) -> f64 {
        self.0.get(id).map(|a| a.confidence).unwrap_or(0.0)
    }

    /// Probability the model assigned to a specific choice option, or 0 when
    /// the answer or option is absent. Used by Package B enrichment: the
    /// grounding stored for a picked value is that option's probability,
    /// not the top-level `confidence` (which is a self-rating).
    pub fn probability(&self, id: &str, option: &str) -> f64 {
        self.0
            .get(id)
            .and_then(|a| a.probabilities.get(option).copied())
            .unwrap_or(0.0)
    }
}

#[derive(Debug)]
struct Counters {
    requests: AtomicUsize,
    input_tokens: AtomicUsize,
    /// Calls that exhausted their retries back to back. Reset by any success.
    consecutive_failures: AtomicUsize,
    /// Exponential moving average of observed chars-per-token, stored as f64 bits.
    ///
    /// Seeded at 2.0 — the measured worst case (email-dense register pages). The
    /// EMA rises toward the true ratio as requests succeed, letting prose-heavy
    /// runs recover batch headroom over time. See `update_ratio_ema` and
    /// `Jev::budget_chars` for the formula and clamp bounds.
    ratio_ema: AtomicU64,
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            requests: AtomicUsize::new(0),
            input_tokens: AtomicUsize::new(0),
            consecutive_failures: AtomicUsize::new(0),
            // 2.0 is the measured worst case; see `budget_chars`.
            ratio_ema: AtomicU64::new(2.0_f64.to_bits()),
        }
    }
}

/// How many calls in a row may fail before we stop trying.
///
/// Retrying is right for a blip and wrong for an outage. When the service is down
/// — `model_unavailable`, or overloaded for a sustained stretch — every call in the
/// pipeline pays the full retry ladder, and a run that should fail in seconds
/// instead grinds for twenty minutes before saying anything. Past this many
/// consecutive failures the breaker opens and calls fail immediately, so the run
/// reports the outage while it is still useful to hear about.
const BREAKER_THRESHOLD: usize = 10;

#[derive(Clone)]
pub struct Jev {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
    max_retries: usize,
    counters: Arc<Counters>,
}

impl Jev {
    pub fn new(
        endpoint: String,
        api_key: String,
        max_retries: usize,
        timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .context("building HTTP client for TypeSafe")?,
            endpoint,
            api_key,
            model: config::TYPESAFE_MODEL.to_string(),
            max_retries,
            counters: Arc::new(Counters::default()),
        })
    }

    /// Evaluate `state` against `questions`.
    ///
    /// Batch everything independent into one call: the questions run in parallel
    /// and cannot see each other's answers, so splitting them buys nothing and
    /// costs a round trip each.
    pub async fn ask(&self, state: Value, questions: Map<String, Value>) -> Result<Answers> {
        if questions.is_empty() {
            bail!("at least one question is required");
        }
        let body = json!({"state": state, "model": self.model, "questions": questions});

        // Catch oversized requests here rather than paying a round trip to be
        // told. Two independent limits apply (measured 2026-09-18):
        //   state budget  — state + longest question ≤ 32k tokens
        //   request budget — full body (state + all questions) ≤ 64k tokens
        // The server's error names a token limit, which reads like a billing
        // problem and is not one; the messages below name the actual constraint.
        let encoded_state = serde_json::to_string(&body["state"]).unwrap_or_default();
        let state_budget = self.state_budget_chars();
        if encoded_state.len() > state_budget {
            bail!(
                "{OVERSIZED_TAG}state is {} characters, over the ~{} the {MAX_STATE_TOKENS}-token \
                 state limit allows (measured 2026-09-18). Trim the state before \
                 sending — the remaining headroom is for questions. (TypeSafe rejects \
                 these with HTTP 400 max_tokens_exceeded.)",
                encoded_state.len(),
                state_budget,
            );
        }
        let encoded = serde_json::to_string(&body).unwrap_or_default();
        let request_budget = self.request_budget_chars();
        if encoded.len() > request_budget {
            bail!(
                "{OVERSIZED_TAG}request is {} characters, over the ~{} the {MAX_REQUEST_TOKENS}-token \
                 full-request limit allows (measured 2026-09-18). This is a request-size \
                 limit, not a credit problem: reduce state or batch fewer questions. \
                 (TypeSafe rejects these with HTTP 400 max_tokens_exceeded.)",
                encoded.len(),
                request_budget,
            );
        }

        // Refuse to start if the service has been failing consistently. The error
        // names the cause so it is not mistaken for a problem with this request.
        let failures = self.counters.consecutive_failures.load(Ordering::Relaxed);
        if failures >= BREAKER_THRESHOLD {
            bail!(
                "TypeSafe has failed {failures} calls in a row; treating it as unavailable \
                 and not retrying further. Check https://api.typesafe.ai status — a 503 \
                 model_unavailable or sustained overload is an outage on their side, not a \
                 problem with this request or your credit."
            );
        }

        let mut last_err = None;
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                // Exponential backoff with jitter. Without jitter, a burst of
                // concurrent calls that all hit an overloaded backend would retry in
                // lockstep and hammer it again at exactly the same moment.
                let base = 500u64 * (1u64 << (attempt - 1)).min(16);
                let jitter = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_millis() as u64)
                    .unwrap_or(0))
                    % base.max(1);
                tokio::time::sleep(std::time::Duration::from_millis(base + jitter)).await;
            }
            match self.attempt(&body, encoded.len()).await {
                Ok(a) => {
                    self.counters
                        .consecutive_failures
                        .store(0, Ordering::Relaxed);
                    return Ok(a);
                }
                Err(e) => {
                    // A 4xx is a verdict on the request itself; sending the identical
                    // bytes again cannot change it. Retrying anyway turned an instant
                    // failure into a slow one and buried the cause under backoff.
                    if is_client_error(&e) {
                        // A verdict on this one request says nothing about service
                        // health, so it must not push the breaker toward opening.
                        self.counters
                            .consecutive_failures
                            .store(0, Ordering::Relaxed);
                        tracing::debug!(error = %e, "jev rejected the request; not retrying");
                        return Err(e);
                    }
                    tracing::debug!(attempt, error = %e, "jev call failed, retrying");
                    last_err = Some(e);
                }
            }
        }
        let n = self
            .counters
            .consecutive_failures
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if n == BREAKER_THRESHOLD {
            tracing::error!(
                failures = n,
                "TypeSafe appears to be down; failing fast from here on"
            );
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("jev call failed with no error")))
    }

    async fn attempt(&self, body: &Value, encoded_len: usize) -> Result<Answers> {
        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .context("sending request to TypeSafe")?;

        let status = resp.status();
        let raw = resp
            .text()
            .await
            .context("reading TypeSafe response body")?;
        if !status.is_success() {
            // 429 is a rate limit, which is worth retrying; every other 4xx is a
            // statement about this request and will say the same thing next time.
            //
            // Transient conditions are the exception: an overloaded or unavailable
            // backend sometimes answers with a 4xx, and that *is* worth retrying —
            // the request was fine, the service was not. Matching on the body rather
            // than the status keeps us honest about which is which.
            let transient = raw.contains("overloaded")
                || raw.contains("capacity")
                || raw.contains("try_again")
                || raw.contains("temporarily");
            let client_error = status.is_client_error() && status.as_u16() != 429 && !transient;
            let oversized = raw.contains("max_tokens_exceeded");
            let hint = if oversized {
                // The adaptive ratio climbed on prose and a denser batch then
                // overran the server limit (observed live: a screening batch was
                // rejected mid-run and its chunks dropped). Fall back to the
                // measured worst case at once; the EMA will climb again.
                self.counters
                    .ratio_ema
                    .store(2.0_f64.to_bits(), Ordering::Relaxed);
                format!(
                    " — the request exceeded the {MAX_INPUT_TOKENS}-token per-request \
                     limit. This is a size limit, not an account-credit problem. The \
                     chars-per-token estimate has been reset to the conservative seed."
                )
            } else {
                String::new()
            };
            bail!(
                "{}{}TypeSafe returned HTTP {status}: {}{hint}",
                if client_error { CLIENT_ERROR_TAG } else { "" },
                if oversized { OVERSIZED_TAG } else { "" },
                crate::llm::truncate(&raw, 400),
            );
        }

        let parsed: SystemOneResponse = serde_json::from_str(&raw).with_context(|| {
            format!(
                "decoding TypeSafe response: {}",
                crate::llm::truncate(&raw, 400)
            )
        })?;

        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        self.counters
            .input_tokens
            .fetch_add(parsed.usage.input_tokens, Ordering::Relaxed);
        // Per-request usage, labelled by the request's question ids, so a
        // run's Jev spend can be attributed to the stage that asked: the ids
        // name it (`answered` is the evidence assessment, `k0…` the claim
        // check, `c0` enumeration completeness). The stats only carried the
        // run total, which cannot say where 156k tokens per answer went.
        if tracing::enabled!(tracing::Level::DEBUG) {
            let ids: Vec<&str> = body["questions"]
                .as_object()
                .map(|q| q.keys().map(String::as_str).collect())
                .unwrap_or_default();
            tracing::debug!(
                input_tokens = parsed.usage.input_tokens,
                questions = ids.len(),
                first = ids.first().copied().unwrap_or(""),
                state_chars = body["state"].to_string().len(),
                "jev usage"
            );
        }

        // Update the chars-per-token EMA so budget_chars() can grow toward the
        // observed ratio. Guard against zero (the field is default-zero and some
        // test doubles omit it) to avoid a div-by-zero in the ratio.
        if parsed.usage.input_tokens > 0 {
            let observed = encoded_len as f64 / parsed.usage.input_tokens as f64;
            let current = f64::from_bits(self.counters.ratio_ema.load(Ordering::Relaxed));
            let updated = update_ratio_ema(current, observed);
            self.counters
                .ratio_ema
                .store(updated.to_bits(), Ordering::Relaxed);
        }

        Ok(Answers(parsed.answers))
    }

    /// Characters the state (alone) may occupy in one request.
    ///
    /// Measured 2026-09-18: state + longest single question must fit within
    /// 32,768 tokens. Applies a 0.9 margin against transient serialisation
    /// variance. Callers that build the state should compare against this before
    /// adding questions.
    pub fn state_budget_chars(&self) -> usize {
        (MAX_STATE_TOKENS as f64 * self.last_ratio() * 0.9) as usize
    }

    /// Characters that the full serialised body (state + all questions) may use.
    ///
    /// Measured 2026-09-18: requests of 44k and 56k tokens succeeded; the old
    /// 32k whole-request ceiling was wrong and was wasting half the capacity on
    /// question-heavy batches. The pre-flight check in `ask` enforces this limit.
    pub fn request_budget_chars(&self) -> usize {
        (MAX_REQUEST_TOKENS as f64 * self.last_ratio() * 0.9) as usize
    }

    /// The current chars-per-token EMA, clamped to [2.0, 3.5].
    ///
    /// Exposed for the stats footer (`-vv`). A value near 2.0 means the run is
    /// processing token-dense content (emails, codes); near 3.5 means mostly prose.
    pub fn last_ratio(&self) -> f64 {
        f64::from_bits(self.counters.ratio_ema.load(Ordering::Relaxed))
    }

    /// (requests, input tokens, estimated USD)
    pub fn stats(&self) -> (usize, usize, f64) {
        let requests = self.counters.requests.load(Ordering::Relaxed);
        let tokens = self.counters.input_tokens.load(Ordering::Relaxed);
        (
            requests,
            tokens,
            tokens as f64 * config::TYPESAFE_USD_PER_INPUT_TOKEN,
        )
    }
}

/// Convenience for building a question map without ceremony at the call site.
pub fn questions(pairs: Vec<(String, Value)>) -> Map<String, Value> {
    pairs.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_converges_upward_and_clamps_at_ceiling() {
        // Starting at 2.0 (seed), sustained English-prose pages (ratio ~3.7)
        // should drive the EMA to the 3.5 ceiling and not exceed it.
        let mut r = 2.0_f64;
        for _ in 0..60 {
            r = update_ratio_ema(r, 3.7);
        }
        assert!(
            (r - 3.5).abs() < 1e-6,
            "expected clamp at 3.5 after convergence, got {r}"
        );
    }

    #[test]
    fn ema_clamps_at_floor_for_low_observed_ratio() {
        // Observed ratio below 2.0 (e.g. very token-dense binary content) must
        // not push the EMA under the 2.0 floor — below that we have no data.
        let r = update_ratio_ema(2.0, 1.2);
        assert!(r >= 2.0, "EMA must not fall below the 2.0 floor, got {r}");
    }

    #[test]
    fn ema_single_step_math() {
        // Verify the formula: alpha=0.3, current=2.0, observed=3.0 → 2.3, within bounds.
        let r = update_ratio_ema(2.0, 3.0);
        let expected = 2.0 * 0.7 + 3.0 * 0.3; // 1.4 + 0.9 = 2.3
        assert!((r - expected).abs() < 1e-10, "expected {expected}, got {r}");
    }

    /// The binding question's exact wording and options are load-bearing:
    /// they were probed against the live API and discriminate the three
    /// shipped misattributions from the two correct attributions.
    #[test]
    fn entity_binding_question_wording() {
        let q = entity_binding("Jumilla", "Spanish municipality", "text");
        assert_eq!(q["type"], "choice");
        assert_eq!(
            q["instructions"].as_str().unwrap(),
            "What relationship does the organisation described in `text` have to `Jumilla`, \
             a Spanish municipality?"
        );
        let c = q["criteria"].as_object().unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(
            c["same"].as_str().unwrap(),
            "It is that organisation itself, including an official department or office of it."
        );
        assert_eq!(
            c["related"].as_str().unwrap(),
            "It is a parent, member, subsidiary, federation, association, supplier, contractor \
             or other related but different organisation."
        );
        assert_eq!(
            c["different"].as_str().unwrap(),
            "It is an unrelated organisation, or a different place or body that merely has a \
             similar name."
        );
        assert_eq!(
            c["unresolved"].as_str().unwrap(),
            "The passage does not establish which organisation it is about."
        );
    }

    /// The same builder serves both call sites; only the state key it points
    /// at changes (`passage` in discovery grounding, `text` in enrichment).
    #[test]
    fn entity_binding_question_names_the_subject_key() {
        let q = entity_binding("Pokecode", "company", "passage");
        assert!(
            q["instructions"]
                .as_str()
                .unwrap()
                .contains("described in `passage`")
        );
    }

    #[test]
    fn budget_chars_free_fn_is_conservative_seed() {
        // The free function returns MAX_INPUT_TOKENS * 2 (the 2.0 seed), which
        // equals the conservative default a fresh Jev starts at before any
        // observed data.
        assert_eq!(budget_chars(), MAX_INPUT_TOKENS * 2);
        assert_eq!(budget_chars(), MAX_STATE_TOKENS * 2);
    }
}
