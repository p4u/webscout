//! Web access, by driving the `obscura` binary.
//!
//! Obscura is invoked as a subprocess rather than linked in, and the reason is
//! measured rather than aesthetic. Obscura embeds V8, and V8 isolates are pinned to
//! their creating thread — `Browser` and `Page` are `!Send`, and driving two
//! browsers from two threads in one process segfaults. Linked in, every page render
//! therefore serialises: eight pages took 151 seconds.
//!
//! `obscura scrape` sidesteps that entirely by running persistent `obscura-worker`
//! *processes*, each with its own address space and its own V8. The same eight pages
//! take 6.2 seconds — 24x faster, with identical output, including the
//! JavaScript-injected contact tables that plain HTTP cannot see.
//!
//! Spawning one `obscura fetch` per URL is *not* the same thing: that took 45
//! seconds for the same batch, because every process pays full V8 startup. The win
//! comes specifically from batching into one `scrape` call, which is why this module
//! is built around batch operations.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::types::Hit;

/// A link found on a page. Directory sites are the main reason these are kept:
/// a listing page's value is usually its links, not its prose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub href: String,
    pub text: String,
}

/// A fetched page, reduced to what later stages actually consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageContent {
    pub url: String,
    /// The URL we asked for, before any redirect. `url` is the FINAL location
    /// after redirects; downstream stages (enrichment, verify_round retry)
    /// map fetched pages back to what they requested, and the final location
    /// often differs (a canonical redirect, HTTPS upgrade, or a language
    /// switch). Keeping both lets the mapping stay a lookup instead of a
    /// heuristic.
    #[serde(default)]
    pub requested_url: String,
    pub title: String,
    /// Rendered text, as a reader would see it. Not HTML: tags would dominate the
    /// token budget of every judgment and extraction downstream.
    pub text: String,
    #[serde(default)]
    pub links: Vec<Link>,
    /// True when obscura rendered this, false when plain HTTP was enough.
    ///
    /// Callers need to know, because HTML that looks complete can still be missing
    /// the data: a page may ship plenty of prose and inject its table by script. The
    /// register this tool was built against has zero email addresses in its HTML
    /// source and all 37 after JavaScript runs. When a page yields nothing, this says
    /// whether rendering it is still worth a try.
    #[serde(default)]
    pub rendered: bool,
}

/// Scrapes DuckDuckGo's HTML endpoint. Runs inside the page, so it stays ES5-ish.
///
/// Sponsored results are dropped here rather than downstream: an ad is not a search
/// result, and paying a model to judge its relevance is wasted money.
const SEARCH_JS: &str = r#"(function () {
  var out = [];
  var nodes = document.querySelectorAll('.result, .web-result');
  for (var i = 0; i < nodes.length; i++) {
    var r = nodes[i];
    if (/result--ad|result--sponsored/.test(r.className)) continue;
    var a = r.querySelector('.result__a, .result-link, h2 a');
    var s = r.querySelector('.result__snippet, .result-snippet');
    if (!a) continue;
    var href = a.href || a.getAttribute('href') || '';
    var title = (a.textContent || '').trim();
    if (!href || !title) continue;
    out.push({title: title, url: href, snippet: s ? (s.textContent || '').trim().replace(/\s+/g, ' ') : ''});
  }
  return JSON.stringify(out);
})()"#;

/// Extracts readable text plus the link graph in one pass.
///
/// `innerText` rather than `textContent` because it respects layout: it skips hidden
/// nodes and preserves line breaks, which keeps a rendered table readable instead of
/// collapsing it into one run-on line. That matters more than it sounds — the
/// contact registers this tool exists to read are tables.
const CONTENT_JS: &str = r#"(function () {
  // Drop non-content elements before reading text. React apps (GitHub
  // measured 2026-09-21) ship megabytes of <style> inside <body>; innerText
  // on /releases came back 1.39 MB of CSS, and the chunk cap downstream kept
  // nothing but CSS. Removing the nodes beats hoping they are display:none by
  // the time eval runs.
  var junk = document.querySelectorAll('style, script, noscript, svg, template');
  for (var j = 0; j < junk.length; j++) {
    if (junk[j].parentNode) junk[j].parentNode.removeChild(junk[j]);
  }
  var links = [];
  var seen = {};
  var as = document.querySelectorAll('a[href]');
  for (var i = 0; i < as.length && links.length < 300; i++) {
    var h = as[i].href;
    if (!h || seen[h] || h.indexOf('javascript:') === 0) continue;
    seen[h] = 1;
    links.push({href: h, text: (as[i].textContent || '').trim().replace(/\s+/g, ' ').slice(0, 160)});
  }
  return JSON.stringify({
    title: document.title || '',
    text: (document.body ? (document.body.innerText || document.body.textContent || '') : ''),
    links: links
  });
})()"#;

/// One entry in `obscura scrape --format json`.
#[derive(Debug, Deserialize)]
struct ScrapeResult {
    url: String,
    #[serde(default)]
    title: String,
    /// Our `--eval` payload, as a JSON *string* that still needs parsing. Null when
    /// the page failed.
    #[serde(default)]
    eval: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ScrapeOutput {
    #[serde(default)]
    results: Vec<ScrapeResult>,
}

/// What `CONTENT_JS` returns.
#[derive(Debug, Deserialize)]
struct Extracted {
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    links: Vec<Link>,
}

/// Client for the external `obscura` binary.
#[derive(Debug, Clone)]
pub struct Obscura {
    bin: String,
    pub stealth: bool,
    pub obey_robots: bool,
    /// Parallel workers inside one `scrape` call. This is both the throughput knob
    /// and the politeness knob — it bounds how hard a batch hits the sites in it.
    pub concurrency: usize,
    pub timeout: Duration,
}

/// PDF bytes as page text, with every failure — including the parser's own
/// panics — reported as an Err reason instead of a crash.
///
/// pdf-extract panics on malformed objects rather than returning Err
/// (measured 2026-09-23, q102: `first arg must be a name: ObjectType
/// { expected: "Name", found: "Reference" }` at pdf-extract 0.12.1 lib.rs:1459
/// killed the whole harvest run). Catching the panic keeps the page a dropped
/// fetch, not a crashed process: a failed guard is never an open door, and
/// neither is a poisoned one.
pub(crate) fn pdf_text(bytes: &[u8]) -> Result<String, String> {
    let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem(bytes)
    }));
    match parsed {
        Err(p) => Err(format!("parser panicked: {}", payload_downcast(&p))),
        Ok(Err(e)) => Err(format!("could not be parsed: {e}")),
        Ok(Ok(t)) if t.trim().is_empty() => Err("no text".to_string()),
        Ok(Ok(t)) => Ok(t),
    }
}

/// A panic payload as a short string for the drop log. `&'static str` is the
/// common case for `panic!` literals; anything else (String, custom types)
/// falls back to the type name rather than being lost as "Box<dyn Any>".
fn payload_downcast(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

impl Obscura {
    /// Locate the binary and confirm it runs.
    ///
    /// Checked once at startup rather than on first use, so a missing dependency is
    /// an immediate, actionable message instead of every fetch failing later for
    /// reasons the log makes look like network trouble.
    pub fn detect(bin: &str, concurrency: usize, timeout: Duration) -> Result<Self> {
        let out = std::process::Command::new(bin)
            .arg("--version")
            .output()
            .with_context(|| {
                format!(
                    "could not run `{bin}`. webscout drives the obscura browser as a \
                     subprocess; install it from https://github.com/h4ckf0r0day/obscura \
                     and put `obscura` and `obscura-worker` on PATH (or pass --obscura-bin)"
                )
            })?;
        let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
        tracing::debug!(%version, "obscura found");

        // `scrape` is the whole point of using the binary, and it needs its worker
        // sibling. Saying so now beats a confusing failure on the first batch.
        if let Ok(path) = which(bin) {
            let worker = path.with_file_name("obscura-worker");
            if !worker.exists() {
                tracing::warn!(
                    expected = %worker.display(),
                    "obscura-worker not found next to obscura; parallel scraping may fail. \
                     Release archives ship both together."
                );
            }
        }

        Ok(Self {
            bin: bin.to_string(),
            stealth: false,
            obey_robots: false,
            concurrency: concurrency.max(1),
            timeout,
        })
    }

    /// Global flags, which obscura expects *before* the subcommand. A flag in the
    /// wrong position is silently ignored rather than rejected.
    fn global_flags(&self) -> Vec<String> {
        let mut f = Vec::new();
        if self.stealth {
            f.push("--stealth".into());
        }
        if self.obey_robots {
            f.push("--obey-robots".into());
        }
        f
    }

    /// Run several searches in one batch.
    ///
    /// Search result pages are just pages, so they go through the same `scrape` call
    /// as everything else: one subprocess for a whole round's searches rather than
    /// one per query.
    pub async fn search_many(&self, queries: &[String], limit: usize) -> Vec<(String, Vec<Hit>)> {
        if queries.is_empty() {
            return Vec::new();
        }
        let urls: Vec<String> = queries
            .iter()
            .map(|q| format!("https://html.duckduckgo.com/html/?q={}", urlencode(q)))
            .collect();

        let results = match self.scrape(&urls, SEARCH_JS).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "search batch failed");
                return Vec::new();
            }
        };

        // Map back by position: obscura returns one result per requested URL.
        let mut out = Vec::new();
        for (i, query) in queries.iter().enumerate() {
            let hits = results
                .get(i)
                .and_then(|r| r.eval.as_deref())
                .and_then(|raw| serde_json::from_str::<Vec<Hit>>(raw).ok())
                .map(|hits| dedupe_hits(hits, limit))
                .unwrap_or_default();
            if hits.is_empty() {
                tracing::debug!(query = %query, "search returned nothing");
            }
            out.push((query.clone(), hits));
        }
        out
    }

    /// Render many pages in one batch, returning whatever succeeded.
    ///
    /// Failures are dropped rather than raised: one dead link should not cost the
    /// round its other twenty pages.
    ///
    /// PDF URLs are routed away from the browser before this runs: Chromium's
    /// PDF viewer exposes no document text to the eval, and on some sites
    /// obscura substitutes the raw byte stream, which is worse than nothing —
    /// measured 2026-09-23, q81: the CDTI NEOTEC 2024 resolution, the one
    /// document that names all 62 beneficiaries, came back as 627,825 chars
    /// of PDF bytes, screened as binary junk (`has_items` never reached its
    /// 0.12 floor on any chunk), and ten rounds ended empty. See `fetch_pdf`.
    pub async fn fetch_many(&self, urls: &[String]) -> Vec<PageContent> {
        if urls.is_empty() {
            return Vec::new();
        }
        let started = std::time::Instant::now();

        let (pdf_urls, html_urls): (Vec<&String>, Vec<&String>) =
            urls.iter().partition(|u| is_pdf_url(u));
        let mut pages: Vec<PageContent> = Vec::new();
        for url in &pdf_urls {
            if let Some(p) = self.fetch_pdf(url).await {
                pages.push(p);
            }
        }
        if html_urls.is_empty() {
            tracing::info!(
                requested = urls.len(),
                read = pages.len(),
                pdf = pdf_urls.len(),
                ms = started.elapsed().as_millis(),
                "page batch read"
            );
            return pages;
        }
        let urls: Vec<String> = html_urls.into_iter().cloned().collect();
        let urls = &urls[..];

        let results = match self.scrape(urls, CONTENT_JS).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, urls = urls.len(), "fetch batch failed");
                return Vec::new();
            }
        };
        // URLs at debug: counts alone made a 2026-09-21 answer-path failure
        // invisible — five pages were read but which five (and whether the
        // authoritative one was among them) had to be inferred from timing.
        tracing::debug!(urls = ?urls, "obscura render batch");

        let mut pages = Vec::new();
        let by_position = results.len() == urls.len();
        for (i, r) in results.into_iter().enumerate() {
            let Some(raw) = r.eval.as_deref() else {
                tracing::debug!(url = %r.url, "page produced no content");
                continue;
            };
            let Ok(extracted) = serde_json::from_str::<Extracted>(raw) else {
                tracing::debug!(url = %r.url, "could not parse page content");
                continue;
            };
            if extracted.text.trim().is_empty() {
                tracing::debug!(url = %r.url, "page rendered with no readable text");
                continue;
            }
            // Position mapping: obscura preserves input order in its results
            // array, so index i on the requested-urls side lines up with r.
            // If lengths ever diverge, fall back to the final URL — callers
            // still work, they just cannot notice a redirect.
            let requested_url = if by_position {
                urls[i].clone()
            } else {
                r.url.clone()
            };
            pages.push(PageContent {
                url: r.url,
                requested_url,
                title: if extracted.title.is_empty() {
                    r.title
                } else {
                    extracted.title
                },
                text: extracted.text,
                links: extracted.links,
                rendered: true,
            });
        }

        tracing::info!(
            requested = urls.len(),
            read = pages.len(),
            ms = started.elapsed().as_millis(),
            "page batch read"
        );
        pages
    }

    /// `obscura fetch --dump original` streams the HTTP body verbatim with the
    /// same stealth flags as a render, which is what gets a PDF past the bot
    /// protection that guards most public registers. The `%PDF` magic is
    /// checked before parsing: an anti-bot challenge page is HTML, and one
    /// retry covers the intermittency of those challenges (measured
    /// 2026-09-23: three manual fetches of the same CDTI PDF, one challenge,
    /// two clean). Extraction failures drop the page rather than pass junk
    /// downstream — a failed guard is never an open door.
    async fn fetch_pdf(&self, url: &str) -> Option<PageContent> {
        let bytes = self.fetch_pdf_bytes(url).await?;
        match pdf_text(&bytes) {
            Ok(text) => {
                tracing::debug!(
                    url = %url,
                    bytes = bytes.len(),
                    chars = text.len(),
                    "pdf extracted"
                );
                Some(PageContent {
                    url: url.to_string(),
                    requested_url: url.to_string(),
                    title: pdf_title(url),
                    text,
                    links: Vec::new(),
                    rendered: false,
                })
            }
            Err(reason) => {
                tracing::debug!(url = %url, bytes = bytes.len(), reason = %reason, "pdf dropped");
                None
            }
        }
    }

    /// One PDF's bytes, or `None`. The dump is retried once because the raw
    /// path cannot solve a challenge itself — it inherits whatever cookies
    /// the browser session already holds.
    async fn fetch_pdf_bytes(&self, url: &str) -> Option<Vec<u8>> {
        for attempt in 0..2 {
            let mut args = self.global_flags();
            args.push("fetch".into());
            args.push("--dump".into());
            args.push("original".into());
            args.push(url.to_string());
            let output = tokio::time::timeout(
                self.timeout + Duration::from_secs(15),
                tokio::process::Command::new(&self.bin)
                    .args(&args)
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .ok()?
            .ok()?;
            if !output.status.success() {
                tracing::debug!(url = %url, attempt, "obscura fetch failed for pdf");
                continue;
            }
            let stdout = output.stdout;
            // The magic may sit behind a small junk prefix; a full 1 KiB window
            // covers every producer that prepends download wrappers.
            if stdout[..stdout.len().min(1024)]
                .windows(4)
                .any(|w| w == b"%PDF")
            {
                return Some(stdout);
            }
            tracing::debug!(url = %url, attempt, bytes = stdout.len(), "pdf fetch returned non-pdf bytes");
        }
        None
    }

    /// Invoke `obscura scrape` and parse its JSON.
    async fn scrape(&self, urls: &[String], eval: &str) -> Result<Vec<ScrapeResult>> {
        let mut args = self.global_flags();
        args.push("scrape".into());
        args.extend(urls.iter().cloned());
        args.push("--quiet".into());
        args.push("--format".into());
        args.push("json".into());
        args.push("--concurrency".into());
        args.push(self.concurrency.to_string());
        args.push("--timeout".into());
        args.push(self.timeout.as_secs().to_string());
        args.push("--eval".into());
        args.push(eval.to_string());

        // Bound the whole batch generously: obscura applies its own per-page timeout,
        // so this only catches a wedged process.
        //
        // The budget scales with the actual batch, divided by how many pages run at
        // once. Capping the growth (this was once `min(8)`) silently killed large
        // batches: 32 URLs at 30 seconds each needs far more than the 4 minutes that
        // cap allowed, and the whole batch was lost rather than one slow page.
        let waves = urls.len().div_ceil(self.concurrency.max(1)) as u64;
        let budget = self.timeout + Duration::from_secs(30 * waves.max(1));

        let output = tokio::time::timeout(
            budget,
            tokio::process::Command::new(&self.bin)
                .args(&args)
                .kill_on_drop(true)
                .output(),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "obscura scrape exceeded {budget:?} for {} url(s)",
                urls.len()
            )
        })?
        .context("running obscura scrape")?;

        if !output.status.success() {
            bail!(
                "obscura scrape exited {}: {}",
                output.status,
                crate::llm::truncate(&String::from_utf8_lossy(&output.stderr), 300)
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parsed: ScrapeOutput = serde_json::from_str(stdout.trim()).with_context(|| {
            format!(
                "parsing obscura scrape output: {}",
                crate::llm::truncate(stdout.trim(), 300)
            )
        })?;
        Ok(parsed.results)
    }
}

/// Does this URL point at a PDF, by its path?
///
/// Content-type would be more truthful, but the browser path never sees one;
/// the extension is how every public register names its documents, and the
/// `%PDF` magic inside `fetch_pdf_bytes` is the real gate this hint feeds.
fn is_pdf_url(url: &str) -> bool {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.path().rsplit('/').next().map(str::to_string))
        .map(|last| last.to_ascii_lowercase().ends_with(".pdf"))
        .unwrap_or(false)
}

/// A PDF has no `<title>`; the filename is the closest thing and it is what
/// the triage state otherwise shows for a page.
fn pdf_title(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.path().rsplit('/').next().map(str::to_string))
        .map(|last| percent_encoding_recover(&last))
        .unwrap_or_else(|| url.to_string())
}

/// Undo percent-encoding in a path segment, best effort.
fn percent_encoding_recover(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = &s[i + 1..i + 3];
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Fetching through Jina's hosted search and reader APIs.
///
/// An alternative to running obscura locally, selected by supplying a Jina key. The
/// trade is straightforward: page rendering moves off this machine — the single
/// largest cost in every profile, and CPU-bound, so it does not scale with anything
/// you can configure — in exchange for a metered dependency on someone else's
/// service.
///
/// It clears the bar that matters. `r.jina.ai` renders JavaScript server-side, so the
/// public register whose 37 email addresses exist only after its scripts run comes
/// back complete, which is the case that defeats every naive fetcher.
#[derive(Debug, Clone)]
pub struct Jina {
    http: reqwest::Client,
    key: String,
    pub concurrency: usize,
}

/// One hit from `s.jina.ai`.
#[derive(Debug, Deserialize)]
struct JinaSearchItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct JinaSearchResponse {
    #[serde(default)]
    data: Vec<JinaSearchItem>,
}

/// One page from `r.jina.ai`.
#[derive(Debug, Deserialize)]
struct JinaReadData {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
    /// The status the *origin* returned, which is not the status of the Jina call.
    ///
    /// Jina answers 200 even when the site it read refused. Without this, a
    /// Cloudflare interstitial arrives as ordinary content and "Why have I been
    /// blocked? This website is using a security service…" gets screened, chunked
    /// and offered to the model as evidence.
    #[serde(rename = "httpStatus", default)]
    http_status: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct JinaReadResponse {
    #[serde(default)]
    data: Option<JinaReadData>,
}

impl Jina {
    pub fn new(key: String, concurrency: usize, timeout: Duration) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .context("building the Jina HTTP client")?,
            key,
            concurrency: concurrency.max(1),
        })
    }

    /// Run several searches, one request each, concurrently.
    ///
    /// Unlike obscura there is no batch endpoint, but these are ordinary HTTP calls
    /// rather than browser renders, so running them in parallel costs nothing local.
    pub async fn search_many(&self, queries: &[String], limit: usize) -> Vec<(String, Vec<Hit>)> {
        use futures::stream::{self, StreamExt};

        // `buffer_unordered` yields results as they finish; without the sort
        // below callers zipping this against `queries` by position would pair
        // each pair with some *other* pair's hits. The scout enrich stage
        // learned this the hard way. Callers should key by query string
        // regardless — this sort is belt-and-braces.
        let mut results: Vec<(usize, String, Vec<Hit>)> =
            stream::iter(queries.iter().cloned().enumerate())
                .map(|(idx, q)| async move {
                    let hits = match self.search_one(&q, limit).await {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::warn!(query = %q, error = %e, "jina search failed");
                            Vec::new()
                        }
                    };
                    if hits.is_empty() {
                        tracing::debug!(query = %q, "search returned nothing");
                    }
                    (idx, q, hits)
                })
                .buffer_unordered(self.concurrency)
                .collect()
                .await;
        results.sort_by_key(|(i, _, _)| *i);
        results.into_iter().map(|(_, q, h)| (q, h)).collect()
    }

    async fn search_one(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let resp = self
            .http
            .get(format!("https://s.jina.ai/?q={}", urlencode(query)))
            .bearer_auth(&self.key)
            .header("Accept", "application/json")
            // Snippets are all triage needs, and full page content here would cost
            // tokens on results we are about to discard.
            .header("X-Respond-With", "no-content")
            .send()
            .await
            .context("sending Jina search request")?;

        let status = resp.status();
        let raw = resp.text().await.context("reading Jina search response")?;
        if !status.is_success() {
            bail!(
                "jina search returned HTTP {status}: {}",
                crate::llm::truncate(&raw, 300)
            );
        }

        let parsed: JinaSearchResponse = serde_json::from_str(&raw).with_context(|| {
            format!(
                "parsing Jina search response: {}",
                crate::llm::truncate(&raw, 300)
            )
        })?;

        let hits = parsed
            .data
            .into_iter()
            .filter(|i| !i.url.is_empty() && !i.title.is_empty())
            .map(|i| Hit {
                title: i.title,
                url: i.url,
                snippet: if i.description.is_empty() {
                    crate::llm::truncate(&i.content, 300).to_string()
                } else {
                    i.description
                },
                engines: vec!["jina".to_string()],
            })
            .collect();
        Ok(dedupe_hits(hits, limit))
    }

    /// Read many pages concurrently.
    pub async fn fetch_many(&self, urls: &[String]) -> Vec<PageContent> {
        use futures::stream::{self, StreamExt};
        if urls.is_empty() {
            return Vec::new();
        }
        let started = std::time::Instant::now();

        let pages: Vec<Option<PageContent>> = stream::iter(urls.to_vec())
            .map(|u| async move {
                match self.read_one(&u).await {
                    Ok(p) => Some(p),
                    Err(e) => {
                        tracing::debug!(url = %u, error = %e, "jina read failed");
                        None
                    }
                }
            })
            .buffer_unordered(self.concurrency)
            .collect()
            .await;

        let pages: Vec<PageContent> = pages.into_iter().flatten().collect();
        tracing::info!(
            requested = urls.len(),
            read = pages.len(),
            ms = started.elapsed().as_millis(),
            "page batch read (jina)"
        );
        pages
    }

    async fn read_one(&self, url: &str) -> Result<PageContent> {
        let resp = self
            .http
            .get(format!("https://r.jina.ai/{url}"))
            .bearer_auth(&self.key)
            .header("Accept", "application/json")
            // Ask r.jina.ai to append a "Links/Buttons" section to the
            // returned markdown. This is what makes follow_links work with
            // the Jina backend — without it PageContent.links was empty on
            // every discovery page and links_followed stayed at zero.
            .header("X-With-Links-Summary", "true")
            .send()
            .await
            .context("sending Jina read request")?;

        let status = resp.status();
        let raw = resp.text().await.context("reading Jina read response")?;
        if !status.is_success() {
            bail!(
                "jina read returned HTTP {status}: {}",
                crate::llm::truncate(&raw, 300)
            );
        }

        let parsed: JinaReadResponse = serde_json::from_str(&raw)
            .with_context(|| format!("parsing Jina read response for {url}"))?;
        let data = parsed
            .data
            .ok_or_else(|| anyhow!("jina returned no data for {url}"))?;

        if let Some(code) = data.http_status
            && !(200..300).contains(&code)
        {
            bail!("{url} returned HTTP {code} to the reader (likely blocked or missing)");
        }
        if data.content.trim().is_empty() {
            bail!("{url} returned no readable text");
        }
        let final_url = if data.url.is_empty() {
            url.to_string()
        } else {
            data.url
        };
        // Resolve relative markdown links against the *final* URL where
        // possible; fall back to the requested URL if the final one won't
        // parse as a base.
        let link_base = if url::Url::parse(&final_url).is_ok() {
            final_url.as_str()
        } else {
            url
        };
        let links = extract_markdown_links(&data.content, link_base);
        Ok(PageContent {
            url: final_url,
            requested_url: url.to_string(),
            title: data.title,
            text: data.content,
            links,
            // Jina renders server-side, so its output already reflects JavaScript.
            // Marking it rendered stops the caller spending a second read trying to
            // "recover" a page that is already complete.
            rendered: true,
        })
    }
}

/// One search engine, as a lane that can run beside the others.
///
/// Fetching and searching used to be the same either/or switch: supplying a
/// Jina key moved *both* to Jina and silently turned DuckDuckGo off, so a run
/// with a key never saw a second engine's opinion. `Backend` still decides who
/// fetches pages; lanes decide who searches, and several can run at once.
///
/// Measured 2026-09-18, which is why there are only two lanes: `html.duckduckgo.com`
/// returns 10 clean results per query, and Jina's endpoint works. Bing scraped
/// through obscura returns 10 nodes of unrelated content behind `bing.com/ck/a?`
/// redirects; Brave, Mojeek, Startpage and Ecosia return interstitials or
/// near-empty bodies (438–4086 chars, zero result selectors). Adding a third
/// engine when one becomes usable is a `LaneEngine` variant, not a refactor.
#[derive(Debug, Clone)]
pub struct SearchLane {
    /// Short, stable identifier. Recorded in `Hit.engines` and used as part of
    /// the cache key, so changing one invalidates that lane's cached entries.
    pub id: &'static str,
    engine: LaneEngine,
    /// Per-lane deadline, inside whatever overall budget the caller imposes.
    /// A lane that hangs must cost its own results, never another lane's.
    pub deadline: Duration,
}

#[derive(Debug, Clone)]
enum LaneEngine {
    /// DuckDuckGo's HTML endpoint, scraped through obscura.
    Ddg(Obscura),
    /// Jina's hosted search endpoint.
    Jina(Jina),
}

/// Default per-lane deadline. Twenty seconds is roughly three times the
/// observed p50 for a five-query DuckDuckGo batch, so it only fires on a lane
/// that is genuinely stuck.
pub const DEFAULT_LANE_DEADLINE: Duration = Duration::from_secs(20);

impl SearchLane {
    pub fn ddg(obscura: Obscura, deadline: Duration) -> Self {
        Self {
            id: "ddg",
            engine: LaneEngine::Ddg(obscura),
            deadline,
        }
    }

    pub fn jina(jina: Jina, deadline: Duration) -> Self {
        Self {
            id: "jina",
            engine: LaneEngine::Jina(jina),
            deadline,
        }
    }

    /// How many of this lane's queries run at once.
    fn concurrency(&self) -> usize {
        match &self.engine {
            LaneEngine::Ddg(o) => o.concurrency,
            LaneEngine::Jina(j) => j.concurrency,
        }
    }

    /// Run every query on this lane, tagging each hit with the lane id.
    pub async fn search(&self, queries: &[String], limit: usize) -> Vec<(String, Vec<Hit>)> {
        let mut out = match &self.engine {
            LaneEngine::Ddg(o) => o.search_many(queries, limit).await,
            LaneEngine::Jina(j) => j.search_many(queries, limit).await,
        };
        for (_, hits) in out.iter_mut() {
            for h in hits.iter_mut() {
                if !h.engines.iter().any(|e| e == self.id) {
                    h.engines.push(self.id.to_string());
                }
            }
        }
        out
    }
}

/// A lane's deadline for one batch: the per-wave deadline times the number
/// of waves the batch needs.
///
/// The deadline was fixed at 20 seconds — measured as three times the p50 of
/// a five-query batch — and applied unchanged to enrichment batches of 25
/// and 50 queries. With two harvests sharing DuckDuckGo, both batch sizes ran
/// out of time and every query in them came back empty, which the next steer
/// read as a stalled run (measured 2026-09-24: q81 enrichment rounds 3 and 4
/// lost whole, the run stopped at round 4 with 62 companies and none
/// classified; the same batch sizes cleared the fixed deadline alone that
/// afternoon). `Obscura::scrape` already scales its own budget by waves; the
/// lane deadline now does the same, so a lane still costs only its own
/// results when it hangs.
pub fn lane_budget(per_wave: Duration, queries: usize, concurrency: usize) -> Duration {
    let waves = queries.div_ceil(concurrency.max(1)).max(1) as u32;
    per_wave * waves
}

/// Which backend does the fetching.
#[derive(Debug, Clone)]
pub enum Backend {
    /// The obscura binary, running locally.
    Obscura(Obscura),
    /// Jina's hosted search and reader.
    Jina(Jina),
}

impl Backend {
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Obscura(_) => "obscura",
            Backend::Jina(_) => "jina",
        }
    }

    pub fn concurrency(&self) -> usize {
        match self {
            Backend::Obscura(o) => o.concurrency,
            Backend::Jina(j) => j.concurrency,
        }
    }

    /// The lane this backend can search with on its own, used when the caller
    /// configures no lanes explicitly. Searching is otherwise not the
    /// backend's job — see `SearchLane`.
    pub fn default_lane(&self, deadline: Duration) -> SearchLane {
        match self {
            Backend::Obscura(o) => SearchLane::ddg(o.clone(), deadline),
            Backend::Jina(j) => SearchLane::jina(j.clone(), deadline),
        }
    }

    async fn fetch_many(&self, urls: &[String]) -> Vec<PageContent> {
        match self {
            Backend::Obscura(o) => o.fetch_many(urls).await,
            Backend::Jina(j) => j.fetch_many(urls).await,
        }
    }
}

/// Fetch pages over plain HTTP, falling back to the configured backend.
///
/// Rendering is the expensive part even with parallel workers: a batch of six pages
/// took 30 seconds, where plain HTTP returns most of them in under a second each.
/// Reference sites, documentation, registers and news all ship their content in the
/// initial HTML and need no browser at all.
///
/// The risk is that HTML can look complete and still be missing the data, so this is
/// only a *pre-filter*: anything thin goes to obscura here, and anything that passes
/// but later yields nothing is re-rendered by the caller. Cheap when it works,
/// self-correcting when it does not.
pub struct Fetcher {
    http: reqwest::Client,
    pub backend: Backend,
    /// Search engines to run, concurrently, for every batch of queries.
    /// Defaults to the one lane the backend can serve by itself.
    pub lanes: Vec<SearchLane>,
    /// On-disk cache of lane responses. `None` disables caching entirely.
    pub search_cache: Option<crate::search_cache::SearchCache>,
    /// Lane batches lost to their deadline, over the life of the fetcher.
    ///
    /// Read by the harvest loop to tell a round the web had nothing for from
    /// a round our search layer failed: both look like "nothing new", and only
    /// the first is evidence that the run has levelled off.
    lane_failures: std::sync::atomic::AtomicUsize,
}

impl Fetcher {
    pub fn new(backend: Backend, timeout: Duration) -> Result<Self> {
        let lanes = vec![backend.default_lane(DEFAULT_LANE_DEADLINE)];
        Ok(Self {
            lanes,
            search_cache: None,
            lane_failures: std::sync::atomic::AtomicUsize::new(0),
            http: reqwest::Client::builder()
                .timeout(timeout)
                // Sites serve very different markup to something that looks like a
                // script; presenting as a browser gets the HTML a reader would see.
                .user_agent(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) \
                     Chrome/145.0.0.0 Safari/537.36",
                )
                .build()
                .context("building the fast-path HTTP client")?,
            backend,
        })
    }

    /// Replace the search lanes. An empty list is rejected by the caller, not
    /// here — a `Fetcher` with no lanes searches nothing and says so once.
    pub fn with_lanes(mut self, lanes: Vec<SearchLane>) -> Self {
        self.lanes = lanes;
        self
    }

    pub fn with_search_cache(mut self, cache: Option<crate::search_cache::SearchCache>) -> Self {
        self.search_cache = cache;
        self
    }

    /// Search every enabled lane concurrently and merge the results.
    ///
    /// Searching is not the fetch backend's job: obscura batches DuckDuckGo
    /// result pages into one process and Jina has a purpose-built endpoint,
    /// and running both is strictly better than picking one — a URL two
    /// engines return is a free prior that costs no Jev tokens. Neither
    /// benefits from the plain-HTTP path, which would just be scraping a SERP
    /// badly.
    ///
    /// A lane that fails or blows its deadline is logged and skipped. It must
    /// never take another lane's results with it, which is the whole reason
    /// each lane gets its own deadline rather than sharing one.
    pub async fn search_many(&self, queries: &[String], limit: usize) -> Vec<(String, Vec<Hit>)> {
        if queries.is_empty() {
            return Vec::new();
        }
        if self.lanes.is_empty() {
            tracing::warn!("no search lanes are enabled; returning no results");
            return queries.iter().map(|q| (q.clone(), Vec::new())).collect();
        }

        let per_lane: Vec<Vec<(String, Vec<Hit>)>> = futures::future::join_all(
            self.lanes
                .iter()
                .map(|lane| self.run_lane(lane, queries, limit)),
        )
        .await;

        // Merge per query, in the order the caller asked for them.
        let mut out = Vec::with_capacity(queries.len());
        for query in queries {
            let mut lanes_hits: Vec<Vec<Hit>> = Vec::new();
            for lane_results in &per_lane {
                if let Some((_, hits)) = lane_results.iter().find(|(q, _)| q == query) {
                    lanes_hits.push(hits.clone());
                }
            }
            let merged = merge_hits(lanes_hits, limit);
            if merged.is_empty() {
                tracing::debug!(query = %query, "search returned nothing on every lane");
            }
            out.push((query.clone(), merged));
        }
        out
    }

    /// Run one lane: serve what the cache has, search the rest under the
    /// lane's deadline, store what came back.
    async fn run_lane(
        &self,
        lane: &SearchLane,
        queries: &[String],
        limit: usize,
    ) -> Vec<(String, Vec<Hit>)> {
        use std::collections::HashMap;

        let mut cached: HashMap<&str, Vec<Hit>> = HashMap::new();
        let mut missing: Vec<String> = Vec::new();
        for q in queries {
            if let Some(cache) = &self.search_cache
                && let Some(hits) = cache.get(lane.id, q, limit)
            {
                cached.insert(q.as_str(), hits);
                continue;
            }
            if !missing.iter().any(|m| m == q) {
                missing.push(q.clone());
            }
        }

        let mut fresh: HashMap<String, Vec<Hit>> = HashMap::new();
        if !missing.is_empty() {
            let budget = lane_budget(lane.deadline, missing.len(), lane.concurrency());
            match tokio::time::timeout(budget, lane.search(&missing, limit)).await {
                Ok(results) => {
                    for (q, hits) in results {
                        if let Some(cache) = &self.search_cache {
                            // Successful and non-empty only: an empty result
                            // is what a block looks like, and caching it would
                            // hold the failure for the whole TTL.
                            cache.put(lane.id, &q, limit, &hits);
                        }
                        fresh.insert(q, hits);
                    }
                }
                Err(_) => {
                    self.lane_failures
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    tracing::warn!(
                        lane = lane.id,
                        queries = missing.len(),
                        deadline_s = budget.as_secs(),
                        "search lane exceeded its deadline; continuing with the other lanes"
                    );
                }
            }
        }

        queries
            .iter()
            .map(|q| {
                let hits = cached
                    .get(q.as_str())
                    .cloned()
                    .or_else(|| fresh.get(q).cloned())
                    .unwrap_or_default();
                (q.clone(), hits)
            })
            .collect()
    }

    /// Lane batches lost to their deadline so far. Monotonic; callers diff it
    /// across a round to learn whether that round's searches were starved.
    pub fn lane_failures(&self) -> usize {
        self.lane_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The end-of-run cache line. Called by `main` once the scout is done.
    pub fn log_search_cache_summary(&self) {
        let Some(cache) = &self.search_cache else {
            return;
        };
        let (hits, misses, bytes) = cache.summary();
        if hits == 0 && misses == 0 {
            return;
        }
        tracing::info!(
            hits,
            misses,
            bytes_from_cache = bytes,
            dir = %cache.dir().display(),
            "search cache"
        );
    }

    /// Fetch every URL, using a browser only where one is needed.
    pub async fn fetch_many(&self, urls: &[String]) -> Vec<PageContent> {
        use futures::stream::{self, StreamExt};
        if urls.is_empty() {
            return Vec::new();
        }
        let started = std::time::Instant::now();

        let attempts: Vec<(String, Option<PageContent>)> = stream::iter(urls.to_vec())
            .map(|url| async move {
                let page = self.try_http(&url).await;
                (url, page)
            })
            .buffer_unordered(self.backend.concurrency().max(4))
            .collect()
            .await;

        let mut pages = Vec::new();
        let mut needs_render = Vec::new();
        for (url, page) in attempts {
            match page {
                Some(p) => pages.push(p),
                None => needs_render.push(url),
            }
        }
        // Which URLs went which way, at debug — see the note in
        // `Obscura::fetch_many`; the HTTP/rendered split was already counted,
        // the identities were not.
        tracing::debug!(urls = ?urls, rendering = ?needs_render, "fetch batch split");

        let http_count = pages.len();
        if !needs_render.is_empty() {
            pages.extend(self.backend.fetch_many(&needs_render).await);
        }

        tracing::info!(
            requested = urls.len(),
            http = http_count,
            rendered = pages.len() - http_count,
            ms = started.elapsed().as_millis(),
            "page batch read"
        );
        pages
    }

    /// Render exactly these URLs with the browser, skipping the HTTP path.
    /// Used when a page passed triage over HTTP and then yielded nothing.
    pub async fn render_many(&self, urls: &[String]) -> Vec<PageContent> {
        self.backend.fetch_many(urls).await
    }

    async fn try_http(&self, url: &str) -> Option<PageContent> {
        let requested_url = url.to_string();
        let resp = self.http.get(url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        // A PDF gets the same plain-HTTP fast path HTML does, through the PDF
        // parser rather than the tag stripper (which turns a PDF into
        // convincing-looking garbage). Before this, a PDF had only obscura's
        // raw dump, and when that failed the page was lost: lecture-note and
        // author-hosted copies of the ElGamal paper on drexel.edu, nku.edu and
        // mit.edu failed there on every attempt, though each is a static file
        // any GET returns (measured 2026-09-23). The browser path stays as the
        // fallback for PDFs behind a bot wall, which is what it was built for.
        if ct.contains("application/pdf") || is_pdf_url(url) {
            return self.try_http_pdf(url, resp).await;
        }
        if !ct.contains("text/html") && !ct.contains("application/xhtml") {
            return None;
        }

        let final_url = resp.url().to_string();
        let html = resp.text().await.ok()?;
        let text = html_to_text(&html);

        // Thin text alongside substantial markup is the signature of a JavaScript
        // shell. Err toward rendering: a wrong "this is fine" loses the page.
        if text.len() < 1200 && html.len() > 2_000 {
            return None;
        }

        Some(PageContent {
            title: extract_title(&html),
            links: extract_links(&html, &final_url),
            url: final_url,
            requested_url,
            text,
            rendered: false,
        })
    }

    /// The body of a response already known to be (or to claim to be) a PDF.
    ///
    /// Everything that is not a parseable PDF returns `None`, which hands the
    /// URL to the browser path: an anti-bot challenge served in place of the
    /// document is HTML, fails the `%PDF` magic, and is exactly the case the
    /// browser's stealth session exists for.
    async fn try_http_pdf(&self, url: &str, resp: reqwest::Response) -> Option<PageContent> {
        // Bound memory before reading: a register's scanned annual report can
        // run to hundreds of megabytes, and nothing past the first pages of a
        // document that size is going to be screened anyway.
        if resp
            .content_length()
            .is_some_and(|n| n > MAX_HTTP_PDF_BYTES)
        {
            tracing::debug!(url = %url, "pdf over the plain-HTTP size bound; leaving it to the browser");
            return None;
        }
        let final_url = resp.url().to_string();
        let bytes = resp.bytes().await.ok()?;
        if bytes.len() as u64 > MAX_HTTP_PDF_BYTES
            || !bytes[..bytes.len().min(1024)]
                .windows(4)
                .any(|w| w == b"%PDF")
        {
            return None;
        }
        match pdf_text(&bytes) {
            Ok(text) => {
                tracing::debug!(url = %url, bytes = bytes.len(), chars = text.len(), "pdf extracted over http");
                Some(PageContent {
                    title: pdf_title(&final_url),
                    url: final_url,
                    requested_url: url.to_string(),
                    text,
                    links: Vec::new(),
                    rendered: false,
                })
            }
            Err(reason) => {
                tracing::debug!(url = %url, reason = %reason, "pdf over http could not be read");
                None
            }
        }
    }
}

/// The largest PDF the plain-HTTP path will read into memory. Larger ones
/// fall through to the browser path, which has its own process boundary.
const MAX_HTTP_PDF_BYTES: u64 = 40 * 1024 * 1024;

/// Convert HTML to readable text.
///
/// Deliberately not a parser. Everything downstream consumes prose, so the job is to
/// drop markup and keep the words in a sane order.
pub fn html_to_text(html: &str) -> String {
    let decoded = decode_entities(&strip_tags(html));
    let mut cleaned = String::with_capacity(decoded.len());
    let mut blank = 0;
    for line in decoded.lines() {
        let t = line.trim();
        if t.is_empty() {
            blank += 1;
            if blank == 1 {
                cleaned.push('\n');
            }
            continue;
        }
        blank = 0;
        cleaned.push_str(t);
        cleaned.push('\n');
    }
    cleaned
}

/// Strip tags over `char` boundaries, turning block elements into line breaks.
fn strip_tags(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len() / 3);
    let mut rest = html;
    let mut low = lower.as_str();

    loop {
        // Elements whose contents are code, not prose. Checked at every `<`,
        // not only where one tag follows another: the check used to run only
        // when the previous tag's `>` was immediately followed by `<style`, so
        // a single newline in between let the whole block through as text.
        // Real markup nearly always has that newline — a WordPress page read
        // over HTTP came back 69,000 characters, mostly inline CSS, where the
        // browser saw 2,500 of prose; the bio chunk screened at 0.30 support
        // diluted by stylesheet, and the page had to be re-rendered to be
        // usable at all (measured 2026-09-24).
        //
        // `<head>` is not skipped whole: it holds no prose of its own (its
        // `<style>` and `<script>` are dropped here one by one), and it is
        // where a page declares its structured data — see below.
        let mut skipped = false;
        for tag in ["script", "style", "noscript", "svg"] {
            if !opens_element(low, tag) {
                continue;
            }
            let close = format!("</{tag}>");
            let open_end = low.find('>').map_or(low.len(), |gt| gt + 1);
            let cut = match low.find(&close) {
                Some(end) => {
                    // JSON-LD is data, not code: a schema.org block states a
                    // page's organisation name, official URL, address, email
                    // and phone — the very facts this tool is asked for, and
                    // often stated nowhere in the visible prose. The old leak
                    // let it through by accident along with the stylesheets;
                    // dropping all scripts cost it: "the official website of
                    // Mondragon Corporation" lost its only support, the
                    // homepage's Organization block (0.84 -> 0.27, measured
                    // 2026-09-24). It is kept, compacted and size-bounded.
                    if tag == "script" && low[..open_end].contains("ld+json") {
                        emit_json_ld(&rest[open_end..end], &mut out);
                    }
                    end + close.len()
                }
                // No closing tag: drop just this element's opening tag and
                // read on. Returning here would discard the rest of the page
                // for one malformed block.
                None => open_end,
            };
            rest = &rest[cut..];
            low = &low[cut..];
            skipped = true;
            break;
        }
        if skipped {
            continue;
        }

        match rest.find('<') {
            None => {
                out.push_str(rest);
                return out;
            }
            // Text before the next tag: emit it, then come back round so the
            // skip check above sees the tag itself.
            Some(lt) if lt > 0 => {
                out.push_str(&rest[..lt]);
                rest = &rest[lt..];
                low = &low[lt..];
            }
            Some(lt) => {
                let tail = &low[lt..];
                if [
                    "<p", "<br", "<div", "<li", "<tr", "<h", "</p", "</div", "</li", "</tr",
                    "</table", "</h",
                ]
                .iter()
                .any(|t| tail.starts_with(t))
                    && !out.ends_with('\n')
                {
                    out.push('\n');
                }
                // P8: emit `<img alt="...">` as its alt text inline. Logo
                // walls put every partner name in an alt attribute and
                // nowhere else; stripping the tag before extraction loses
                // exactly the words we need. Only the alt is emitted — src
                // and other attributes stay dropped.
                let is_img = tail.starts_with("<img")
                    && matches!(
                        low.as_bytes().get(lt + 4),
                        Some(b' ')
                            | Some(b'\t')
                            | Some(b'\n')
                            | Some(b'\r')
                            | Some(b'/')
                            | Some(b'>')
                    );
                match rest[lt..].find('>') {
                    None => return out,
                    Some(gt) => {
                        if is_img {
                            let tag_html = &rest[lt..lt + gt + 1];
                            if let Some(alt) = extract_attr(tag_html, "alt") {
                                let a = alt.trim();
                                if !a.is_empty() {
                                    if !out.is_empty()
                                        && !out.ends_with(' ')
                                        && !out.ends_with('\n')
                                    {
                                        out.push(' ');
                                    }
                                    out.push_str(a);
                                    out.push(' ');
                                }
                            }
                        }
                        let cut = lt + gt + 1;
                        rest = &rest[cut..];
                        low = &low[cut..];
                    }
                }
            }
        }
    }
}

/// Emit a JSON-LD block as one compact line of text: whitespace collapsed,
/// JSON's escaped slashes restored so URLs read as URLs, and bounded — an
/// article's structured data can repeat its whole body, which the prose
/// already carries.
fn emit_json_ld(raw: &str, out: &mut String) {
    const MAX_JSON_LD: usize = 2_000;
    let compact: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let compact = compact.replace("\\/", "/");
    if compact.is_empty() {
        return;
    }
    let cut = compact
        .char_indices()
        .nth(MAX_JSON_LD)
        .map_or(compact.len(), |(i, _)| i);
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&compact[..cut]);
    out.push('\n');
}

/// Does `low` (lowercased markup) start with an opening `<tag` for exactly
/// this element? The name must end at whitespace, `>` or `/`: a bare prefix
/// test reads `<header>` as `<head`, and with the skip working that would
/// search for a `</head>` that closed long before and drop the page body.
fn opens_element(low: &str, tag: &str) -> bool {
    low.strip_prefix('<')
        .and_then(|r| r.strip_prefix(tag))
        .and_then(|r| r.chars().next())
        .is_some_and(|c| c.is_ascii_whitespace() || c == '>' || c == '/')
}

/// The named HTML entities that actually appear in body text, as
/// `(entity, replacement)` pairs.
///
/// Accented Latin letters dominate the table for a measured reason
/// (2026-09-21, Spanish professional-association harvest): Spanish,
/// Catalan and French pages spell them by name — `&eacute;`, `&ograve;`,
/// `&Ntilde;`, `&ccedil;` — and `innerText` in the browser decodes them
/// while the plain-HTTP path did not. The same colegio then arrived twice,
/// once as `T&eacute;c` and once as `Téc`, and the two spellings kept two
/// records with two normalised keys.
///
/// `&amp;` is deliberately last: decoding it before the letters would
/// double-unescape `&amp;eacute;` (the literal text "&eacute;") into "é".
const NAMED_ENTITIES: &[(&str, &str)] = &[
    ("&nbsp;", " "),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", "\""),
    ("&#39;", "'"),
    ("&apos;", "'"),
    ("&mdash;", "—"),
    ("&ndash;", "–"),
    ("&hellip;", "…"),
    ("&laquo;", "«"),
    ("&raquo;", "»"),
    ("&middot;", "·"),
    ("&iexcl;", "¡"),
    ("&iquest;", "¿"),
    ("&deg;", "°"),
    ("&sect;", "§"),
    ("&para;", "¶"),
    ("&euro;", "€"),
    ("&pound;", "£"),
    ("&copy;", "©"),
    ("&reg;", "®"),
    ("&trade;", "™"),
    ("&bull;", "•"),
    ("&dagger;", "†"),
    ("&szlig;", "ß"),
    ("&times;", "×"),
    ("&divide;", "÷"),
    ("&plusmn;", "±"),
    ("&agrave;", "à"),
    ("&aacute;", "á"),
    ("&acirc;", "â"),
    ("&atilde;", "ã"),
    ("&auml;", "ä"),
    ("&aring;", "å"),
    ("&ccedil;", "ç"),
    ("&egrave;", "è"),
    ("&eacute;", "é"),
    ("&ecirc;", "ê"),
    ("&euml;", "ë"),
    ("&igrave;", "ì"),
    ("&iacute;", "í"),
    ("&icirc;", "î"),
    ("&iuml;", "ï"),
    ("&ntilde;", "ñ"),
    ("&ograve;", "ò"),
    ("&oacute;", "ó"),
    ("&ocirc;", "ô"),
    ("&otilde;", "õ"),
    ("&ouml;", "ö"),
    ("&ugrave;", "ù"),
    ("&uacute;", "ú"),
    ("&ucirc;", "û"),
    ("&uuml;", "ü"),
    ("&yacute;", "ý"),
    ("&yuml;", "ÿ"),
    ("&Agrave;", "À"),
    ("&Aacute;", "Á"),
    ("&Acirc;", "Â"),
    ("&Atilde;", "Ã"),
    ("&Auml;", "Ä"),
    ("&Aring;", "Å"),
    ("&Ccedil;", "Ç"),
    ("&Egrave;", "È"),
    ("&Eacute;", "É"),
    ("&Ecirc;", "Ê"),
    ("&Euml;", "Ë"),
    ("&Igrave;", "Ì"),
    ("&Iacute;", "Í"),
    ("&Icirc;", "Î"),
    ("&Iuml;", "Ï"),
    ("&Ntilde;", "Ñ"),
    ("&Ograve;", "Ò"),
    ("&Oacute;", "Ó"),
    ("&Ocirc;", "Ô"),
    ("&Otilde;", "Õ"),
    ("&Ouml;", "Ö"),
    ("&Ugrave;", "Ù"),
    ("&Uacute;", "Ú"),
    ("&Ucirc;", "Û"),
    ("&Uuml;", "Ü"),
    ("&Yacute;", "Ý"),
    // Last on purpose: see the note above.
    ("&amp;", "&"),
];

/// Decode the entities that actually appear in body text.
fn decode_entities(s: &str) -> String {
    let mut out = s.to_string();
    for (entity, replacement) in NAMED_ENTITIES {
        // The guard keeps the common case allocation-free per entity: most
        // pages use a handful of these, not all eighty.
        if out.contains(entity) {
            out = out.replace(entity, replacement);
        }
    }
    while let Some(start) = out.find("&#") {
        let Some(end) = out[start..].find(';').map(|e| start + e) else {
            break;
        };
        let body = &out[start + 2..end];
        let ch = body
            .strip_prefix('x')
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .or_else(|| body.parse::<u32>().ok())
            .and_then(char::from_u32);
        match ch {
            Some(c) => out.replace_range(start..=end, &c.to_string()),
            None => out.replace_range(start..=end, ""),
        }
    }
    out
}

fn extract_title(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let Some(a) = lower.find("<title") else {
        return String::new();
    };
    let Some(open_end) = html[a..].find('>').map(|i| a + i + 1) else {
        return String::new();
    };
    let Some(b) = lower[open_end..].find("</title>").map(|i| open_end + i) else {
        return String::new();
    };
    decode_entities(html[open_end..b].trim())
}

/// Minimal PATH lookup, to check for `obscura-worker` beside the binary.
fn which(bin: &str) -> Result<std::path::PathBuf> {
    let p = std::path::Path::new(bin);
    if p.is_absolute() || bin.contains('/') {
        return Ok(p.to_path_buf());
    }
    let path = std::env::var("PATH").unwrap_or_default();
    for dir in path.split(':') {
        let candidate = std::path::Path::new(dir).join(bin);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!("{bin} not found on PATH")
}

/// Unwrap redirects and drop duplicates, preserving rank order.
///
/// Duplicates are judged by `canonical_url`, not by the raw string: the same
/// page arrives as `http`/`https`, with and without `www.`, with a trailing
/// slash, and dressed in whatever campaign parameters the linking site added.
/// Comparing raw URLs let all of those occupy separate result slots.
fn dedupe_hits(hits: Vec<Hit>, limit: usize) -> Vec<Hit> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for mut hit in hits {
        hit.url = unwrap_redirect(&hit.url);
        if hit.url.is_empty() || !seen.insert(canonical_url(&hit.url)) {
            continue;
        }
        out.push(hit);
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    out
}

/// Query parameters that identify a *visit* rather than a page. Dropping them
/// is what lets two engines' copies of the same article merge instead of
/// competing for slots.
///
/// `s` and `t` are DuckDuckGo's own click parameters, `si` Spotify's, and
/// `feature` YouTube's. `lang` is dropped deliberately: a language switch is a
/// translation of the same page, and the language-variant rule elsewhere in
/// this tool already says only the first variant is worth reading.
const TRACKING_PARAMS: &[&str] = &[
    "ref", "fbclid", "gclid", "igshid", "si", "s", "t", "feature", "lang",
];

/// Is this parameter noise rather than content identity?
fn is_tracking_param(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    k.starts_with("utm_") || k.starts_with("ref_") || TRACKING_PARAMS.contains(&k.as_str())
}

/// A stable identity for a page, for comparing hits from different engines.
///
/// Host lowercased with a leading `www.` removed, path with trailing slashes
/// trimmed, query reduced to the parameters that actually select content and
/// sorted, fragment dropped. The scheme is left out on purpose: `http` and
/// `https` of the same page are the same page, and engines disagree about
/// which to report.
///
/// A URL that will not parse is returned lowercased and trimmed, so it still
/// compares equal to itself rather than vanishing.
pub fn canonical_url(url: &str) -> String {
    let raw = url.trim();
    let Ok(parsed) = url::Url::parse(raw) else {
        return raw.to_ascii_lowercase();
    };
    let host = parsed
        .host_str()
        .unwrap_or("")
        .to_ascii_lowercase()
        .trim_start_matches("www.")
        .to_string();
    let path = parsed.path().trim_end_matches('/').to_string();

    let mut params: Vec<(String, String)> = parsed
        .query_pairs()
        .filter(|(k, _)| !is_tracking_param(k))
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    params.sort();

    let mut out = format!("{host}{path}");
    if !params.is_empty() {
        out.push('?');
        let joined: Vec<String> = params
            .into_iter()
            .map(|(k, v)| if v.is_empty() { k } else { format!("{k}={v}") })
            .collect();
        out.push_str(&joined.join("&"));
    }
    out
}

/// Query parameters that are junk in a URL a human is meant to click.
///
/// Deliberately narrower than [`TRACKING_PARAMS`], which serves
/// [`canonical_url`]'s dedup key. That list drops `lang`, `s` and `t`, which
/// is right when deciding whether two URLs are the same page and wrong when
/// printing one: stripping `lang` can send the reader to a different
/// translation than the one that was actually read.
const DISPLAY_JUNK_PARAMS: &[&str] = &[
    "fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "_ga", "ref", "error",
];

/// Is this `code=` value a session token rather than something that selects
/// content? A country or language code is short and wordlike; a session
/// token is long and random. `?code=ES` must survive, `?code=b980f9c2-…`
/// must not.
fn is_session_token(value: &str) -> bool {
    value.len() >= 16
        && value
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-' || c == '_')
}

/// A URL fit to print as a citation: the page that was read, with the
/// session litter removed.
///
/// Cleaning is separate from [`canonical_url`] on purpose. That function
/// builds a comparison key — it lowercases, drops the scheme and sorts the
/// query — which makes it unusable as a link. This one preserves the URL as
/// fetched (scheme, path, ordering, fragment) and removes only parameters
/// that carry no content identity, so the link still resolves to the
/// document the passage came from.
///
/// Measured 2026-09-23 (ElGamal): the answer's one requested deliverable
/// came back as
/// `…/chapter/10.1007/3-540-39568-7_2?error=cookies_not_supported&code=b980f9c2-…`
/// — Springer's cookie-wall redirect, pasted verbatim into the prose as the
/// link to the paper.
///
/// A URL that will not parse is returned trimmed and otherwise untouched: a
/// citation we cannot read is still the provenance we have.
pub fn display_url(url: &str) -> String {
    let raw = url.trim();
    let Ok(parsed) = url::Url::parse(raw) else {
        return raw.to_string();
    };
    let has_error = parsed.query_pairs().any(|(k, _)| k == "error");

    let kept: Vec<(String, String)> = parsed
        .query_pairs()
        .filter(|(k, v)| {
            let key = k.to_ascii_lowercase();
            if key.starts_with("utm_") || key.starts_with("ref_") {
                return false;
            }
            if DISPLAY_JUNK_PARAMS.contains(&key.as_str()) {
                return false;
            }
            // `code` is only junk in the company of an `error`, or when the
            // value is plainly a token.
            if key == "code" && (has_error || is_session_token(v)) {
                return false;
            }
            true
        })
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();

    let mut out = parsed.clone();
    if kept.is_empty() {
        out.set_query(None);
    } else {
        let mut pairs = out.query_pairs_mut();
        pairs.clear();
        for (k, v) in &kept {
            if v.is_empty() {
                pairs.append_key_only(k);
            } else {
                pairs.append_pair(k, v);
            }
        }
    }
    out.to_string()
}

/// A loose identity for a headline, used to keep one syndicated story from
/// taking five slots.
///
/// Lowercased, with the trailing site-name segment after a dash, pipe or colon
/// removed ("… — Reuters", "… | El País"), then the first eight alphanumeric
/// words. Eight words is enough that two genuinely different articles rarely
/// collide while the same wire copy republished under five mastheads does.
pub fn title_key(title: &str) -> String {
    let lower = title.trim().to_lowercase();

    // Strip the last separated segment when there is a head left to keep.
    let separators = [" - ", " – ", " — ", " | ", " : ", ": ", "|", " · "];
    let mut cut = None;
    for sep in separators {
        if let Some(i) = lower.rfind(sep) {
            let head = lower[..i].trim();
            let tail = lower[i + sep.len()..].trim();
            // Only a short trailing segment is a site name; a long one is
            // part of the headline (subtitles after a colon or dash).
            if !head.is_empty()
                && !tail.is_empty()
                && tail.split_whitespace().count() <= 5
                && cut.is_none_or(|c| i > c)
            {
                cut = Some(i);
            }
        }
    }
    let head = match cut {
        Some(i) => &lower[..i],
        None => lower.as_str(),
    };

    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in head.chars() {
        if c.is_alphanumeric() {
            current.push(c);
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
            if words.len() == 8 {
                break;
            }
        }
    }
    if words.len() < 8 && !current.is_empty() {
        words.push(current);
    }
    words.join(" ")
}

/// Merge one query's hits from several lanes into one ranked list.
///
/// Identity is `canonical_url`. When two lanes return the same page the
/// engines are unioned (so `Hit.engines.len()` reads as agreement), the best —
/// lowest — original position wins, the longer snippet is kept, and an empty
/// title is filled from whichever lane had one. A hit whose `title_key` was
/// already claimed by a *different* page is dropped: that is the same story
/// syndicated, and five copies of it is five wasted read slots. Only a
/// headline-length title (≥ 4 words) claims its key: a 2–3 word title is a
/// programme or category name ("NEOTEC 2024", "Ayudas NEOTEC"), and distinct
/// government pages about the same programme legitimately share one.
/// Measured 2026-09-21 (q81): the short-title collision dropped the
/// ministry's resolution page and two other distinct NEOTEC pages as
/// "syndicated copies", 232 drops in one run.
///
/// Ordering is by best original position, ties broken by first appearance, so
/// a result both engines ranked first stays first.
pub fn merge_hits(per_lane: Vec<Vec<Hit>>, limit: usize) -> Vec<Hit> {
    use std::collections::HashMap;

    // (best position, insertion order, hit), keyed by canonical URL.
    let mut merged: Vec<(usize, Hit)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut titles: HashMap<String, usize> = HashMap::new();

    for lane_hits in per_lane {
        for (pos, hit) in lane_hits.into_iter().enumerate() {
            if hit.url.is_empty() {
                continue;
            }
            let key = canonical_url(&hit.url);
            let tkey = title_key(&hit.title);

            if let Some(&slot) = index.get(&key) {
                let (best, existing) = &mut merged[slot];
                *best = (*best).min(pos);
                for e in hit.engines {
                    if !existing.engines.contains(&e) {
                        existing.engines.push(e);
                    }
                }
                if hit.snippet.len() > existing.snippet.len() {
                    existing.snippet = hit.snippet;
                }
                if existing.title.trim().is_empty() && !hit.title.trim().is_empty() {
                    existing.title = hit.title;
                }
                continue;
            }

            // See the doc above: only a headline-length key identifies a
            // story; a two-word programme name is shared by distinct pages.
            if tkey.split_whitespace().count() >= 4 && titles.contains_key(&tkey) {
                tracing::debug!(url = %hit.url, "dropping a duplicate title (syndicated copy)");
                continue;
            }

            let slot = merged.len();
            index.insert(key, slot);
            if !tkey.is_empty() {
                titles.insert(tkey, slot);
            }
            merged.push((pos, hit));
        }
    }

    // Best position first; a tie goes to the hit more engines returned, which
    // is the cheapest relevance prior available. The sort is stable, so an
    // otherwise equal pair keeps first-seen order.
    merged.sort_by_key(|(best, hit)| (*best, usize::MAX - hit.engines.len()));
    let mut out: Vec<Hit> = merged.into_iter().map(|(_, h)| h).collect();
    if limit > 0 && out.len() > limit {
        out.truncate(limit);
    }
    out
}

/// Turn `https://duckduckgo.com/l/?uddg=<encoded>&rut=...` into its destination.
/// Anything that is not such a wrapper passes through unchanged.
pub fn unwrap_redirect(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }
    let normalized = if raw.starts_with("//") {
        format!("https:{raw}")
    } else {
        raw.to_string()
    };

    let parsed = match url::Url::parse(&normalized) {
        Ok(u) => u,
        Err(_) => return String::new(),
    };

    if !parsed.host_str().unwrap_or("").contains("duckduckgo.com") {
        return normalized;
    }
    if let Some((_, target)) = parsed.query_pairs().find(|(k, _)| k == "uddg") {
        return target.into_owned();
    }
    // A duckduckgo.com link with no uddg is internal navigation, not a result.
    String::new()
}

/// Percent-encode a query component. Hand-rolled to keep the dependency list short;
/// the rule is simple enough not to warrant a crate.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Scan `html` for `<a href="...">text</a>` anchors and return resolved links.
///
/// Relative hrefs are resolved against `base` (the page's final URL after any
/// redirects). The following hrefs are skipped: `javascript:`, `mailto:`,
/// `tel:`, and pure fragment-only refs (`#…`). Results are deduplicated by
/// resolved href and capped at 300. Link text has its tags stripped and
/// whitespace collapsed; truncated to 160 characters.
///
/// Regex is intentionally avoided here — the existing byte-walking style in
/// this module extends naturally and keeps the dependency list unchanged.
pub fn extract_links(html: &str, base: &str) -> Vec<Link> {
    let Ok(base_url) = url::Url::parse(base) else {
        return Vec::new();
    };

    let mut links: Vec<Link> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let lower = html.to_ascii_lowercase();
    let mut pos = 0;

    while links.len() < 300 {
        // Locate the next <a that is followed by whitespace or > (not <abbr
        // or <article etc.)
        let Some(rel) = lower[pos..].find("<a") else {
            break;
        };
        let tag_start = pos + rel;
        let after_name = tag_start + 2;

        match lower.as_bytes().get(after_name) {
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') | Some(b'>') => {}
            _ => {
                pos = after_name;
                continue;
            }
        }

        // Find the end of the opening tag.
        let Some(gt_rel) = lower[tag_start..].find('>') else {
            break;
        };
        let tag_end = tag_start + gt_rel + 1;
        let tag_html = &html[tag_start..tag_end];

        // Extract and validate href.
        let Some(href_raw) = extract_attr(tag_html, "href") else {
            pos = tag_end;
            continue;
        };
        let href_raw = decode_entities(href_raw.trim());
        if href_raw.is_empty() {
            pos = tag_end;
            continue;
        }
        let href_lower = href_raw.to_ascii_lowercase();
        if href_lower.starts_with("javascript:")
            || href_lower.starts_with("mailto:")
            || href_lower.starts_with("tel:")
            || href_raw.starts_with('#')
        {
            pos = tag_end;
            continue;
        }

        // Resolve relative URLs.
        let resolved = match base_url.join(&href_raw) {
            Ok(u) => u.to_string(),
            Err(_) => {
                pos = tag_end;
                continue;
            }
        };

        // Deduplicate by resolved href.
        if !seen.insert(resolved.clone()) {
            pos = tag_end;
            continue;
        }

        // Extract inner HTML up to the matching </a>.
        let close = lower[tag_end..].find("</a").map(|i| tag_end + i);
        let inner_end = close.unwrap_or_else(|| (tag_end + 2000).min(html.len()));
        let inner_html = &html[tag_end..inner_end];

        // Strip tags, decode entities, collapse whitespace, cap at 160 chars.
        let raw_text = strip_inner_text(inner_html);
        let collapsed: String = raw_text.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut text: String = collapsed.chars().take(160).collect();

        // Logo-wall recovery: a service-provider directory that renders each
        // partner as a bare `<a><img></a>` yields empty link text through the
        // strip-tags path, and downstream Jev scoring cannot tell one nameless
        // link from another. Fall back to the anchor's own `title`, then its
        // inner `<img alt>`, then `aria-label`, before giving up. This is what
        // keeps a `decidim.org/partners`-shaped page nameable at all.
        if text.trim().is_empty() {
            let alt = extract_attr(tag_html, "title")
                .map(str::to_string)
                .or_else(|| find_img_alt(inner_html))
                .or_else(|| extract_attr(tag_html, "aria-label").map(str::to_string));
            if let Some(a) = alt {
                let decoded = decode_entities(a.trim());
                let collapsed: String = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
                text = collapsed.chars().take(160).collect();
            }
        }

        links.push(Link {
            href: resolved,
            text,
        });

        pos = tag_end;
    }

    links
}

/// Extract links from markdown text. Handles both `[text](url)` links and
/// bare `http(s)://…` URLs. Relative hrefs are resolved against `base`, and
/// unwanted schemes / fragments / duplicates are filtered exactly as in
/// `extract_links`. Result is capped at 300 entries.
///
/// This exists because the Jina reader returns markdown rather than HTML.
/// Even with `X-With-Links-Summary: true` the "Links/Buttons" section is
/// appended as more markdown, so a single markdown-aware extractor covers
/// both cases and needs no bespoke parser for the summary block.
pub fn extract_markdown_links(md: &str, base: &str) -> Vec<Link> {
    let Ok(base_url) = url::Url::parse(base) else {
        return Vec::new();
    };

    let bytes = md.as_bytes();
    let mut links: Vec<Link> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let push = |href_raw: &str,
                text: &str,
                links: &mut Vec<Link>,
                seen: &mut std::collections::HashSet<String>| {
        if links.len() >= 300 {
            return;
        }
        let href_raw = href_raw.trim();
        if href_raw.is_empty() {
            return;
        }
        let hl = href_raw.to_ascii_lowercase();
        if hl.starts_with("javascript:")
            || hl.starts_with("mailto:")
            || hl.starts_with("tel:")
            || href_raw.starts_with('#')
        {
            return;
        }
        let resolved = match base_url.join(href_raw) {
            Ok(u) => u.to_string(),
            Err(_) => return,
        };
        if !seen.insert(resolved.clone()) {
            return;
        }
        let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let text: String = collapsed.chars().take(160).collect();
        links.push(Link {
            href: resolved,
            text,
        });
    };

    // Pass 1: `[text](url)` — walk the string, matching brackets manually so
    // nested `[]` inside text (image alts, etc.) don't confuse us.
    let mut i = 0;
    while i < bytes.len() && links.len() < 300 {
        if bytes[i] == b'[' {
            // Find the matching `]` at depth 0.
            let mut depth = 1;
            let mut j = i + 1;
            while j < bytes.len() {
                match bytes[j] {
                    b'[' => depth += 1,
                    b']' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    b'\\' => j += 1, // skip escaped char
                    _ => {}
                }
                j += 1;
            }
            if depth != 0 || j >= bytes.len() || bytes.get(j + 1) != Some(&b'(') {
                i += 1;
                continue;
            }
            let text_start = i + 1;
            let text_end = j;
            // Find matching ')'.
            let mut k = j + 2;
            let mut pdepth = 1;
            while k < bytes.len() {
                match bytes[k] {
                    b'(' => pdepth += 1,
                    b')' => {
                        pdepth -= 1;
                        if pdepth == 0 {
                            break;
                        }
                    }
                    b'\\' => k += 1,
                    _ => {}
                }
                k += 1;
            }
            if pdepth != 0 || k >= bytes.len() {
                i += 1;
                continue;
            }
            let url_raw = &md[j + 2..k];
            // Markdown allows a title after the URL: `(url "title")`. Strip.
            let url_only = url_raw
                .split_once(|c: char| c.is_whitespace())
                .map(|(u, _)| u)
                .unwrap_or(url_raw);
            let text = &md[text_start..text_end];
            push(url_only, text, &mut links, &mut seen);
            i = k + 1;
        } else {
            i += 1;
        }
    }

    // Pass 2: bare URLs. Scan for `http://` / `https://` occurrences not
    // already emitted, and take the URL up to the next whitespace, `)`, `]`,
    // `>`, or `"`. Trailing punctuation (`.,;:!?`) is stripped.
    let lower = md.to_ascii_lowercase();
    let mut pos = 0;
    while links.len() < 300 {
        let http_at = lower[pos..].find("http://").map(|i| pos + i);
        let https_at = lower[pos..].find("https://").map(|i| pos + i);
        let start = match (http_at, https_at) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let end = md[start..]
            .find(|c: char| c.is_whitespace() || matches!(c, ')' | ']' | '>' | '"' | '<' | '\''))
            .map(|o| start + o)
            .unwrap_or(md.len());
        let mut url = &md[start..end];
        while let Some(last) = url.chars().last() {
            if matches!(last, '.' | ',' | ';' | ':' | '!' | '?') {
                url = &url[..url.len() - last.len_utf8()];
            } else {
                break;
            }
        }
        if !url.is_empty() {
            push(url, "", &mut links, &mut seen);
        }
        pos = end;
    }

    links
}

/// Extract the value of a named HTML attribute from a tag string such as
/// `<a href="/path" class="x">`. Works on both quoted and unquoted values.
///
/// Uses `to_ascii_lowercase` for the attribute name search, then slices the
/// original `tag` bytes — safe because `to_ascii_lowercase` preserves byte
/// lengths for all characters.
fn extract_attr<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let lower = tag.to_ascii_lowercase();
    // Match ` name=` or leading `name=` (for the first attribute with no
    // preceding space, which can't happen but is harmless to handle).
    let pattern = format!("{name}=");
    let pos = lower.find(&pattern)?;
    let after = &tag[pos + pattern.len()..];
    if let Some(rest) = after.strip_prefix('"') {
        rest.find('"').map(|end| &rest[..end])
    } else if let Some(rest) = after.strip_prefix('\'') {
        rest.find('\'').map(|end| &rest[..end])
    } else {
        // Unquoted value ends at the next whitespace or >.
        let end = after
            .find(|c: char| c.is_ascii_whitespace() || c == '>')
            .unwrap_or(after.len());
        Some(&after[..end])
    }
}

/// Find the first `<img alt="...">` value inside a short inner-HTML slice.
/// Used by `extract_links` to recover a name for logo-wall entries whose
/// anchor text is empty — the alt attribute is where the accessible name of
/// a partner's logo lives on nine directory pages out of ten.
fn find_img_alt(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let mut pos = 0;
    while let Some(rel) = lower[pos..].find("<img") {
        let start = pos + rel;
        let gt = lower[start..].find('>')?;
        let tag = &html[start..start + gt + 1];
        if let Some(alt) = extract_attr(tag, "alt") {
            let trimmed = alt.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        pos = start + gt + 1;
    }
    None
}

/// Strip all HTML tags from a short inner-HTML slice and decode entities.
/// Used only for extracting link text, so it does not need `strip_tags`'s
/// block-element newline logic.
fn strip_inner_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        match rest.find('<') {
            None => {
                out.push_str(rest);
                break;
            }
            Some(lt) => {
                out.push_str(&rest[..lt]);
                match rest[lt..].find('>') {
                    None => break,
                    Some(gt) => rest = &rest[lt + gt + 1..],
                }
            }
        }
    }
    decode_entities(&out)
}

/// Break text into paragraph-sized pieces, none longer than `size`.
///
/// Splitting on blank lines alone is not enough, and the gap was a real bug.
/// `innerText` separates blocks with a single `\n`, so a page with no blank lines
/// yields exactly one "paragraph" — the entire document. The size cap then never
/// applies, and a 500,000-character page becomes one chunk that blows past the
/// verification API's per-request token limit.
///
/// So anything still oversized gets hard-split, preferring a line break and then a
/// space near the boundary. Splitting mid-word is the last resort but still beats
/// emitting a chunk no backend will accept.
fn split_paragraphs(text: &str, size: usize) -> Vec<&str> {
    let size = size.max(200);
    let mut out = Vec::new();

    for para in text.split("\n\n") {
        let mut rest = para;
        while rest.len() > size {
            let window = &rest[..floor_char_boundary(rest, size)];
            let cut = window
                .rfind('\n')
                .or_else(|| window.rfind(' '))
                .filter(|&i| i > size / 2)
                .unwrap_or(window.len());
            let cut = floor_char_boundary(rest, cut).max(1);
            out.push(&rest[..cut]);
            rest = &rest[cut..];
        }
        out.push(rest);
    }
    out
}

/// Round `i` down to a character boundary so slicing never panics on UTF-8.
/// Page text is full of multi-byte characters, so this is load-bearing.
fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Split page text into chunks on blank lines, up to roughly `size` characters.
///
/// Paragraph boundaries rather than a fixed byte window, because a chunk that starts
/// mid-sentence asks the model to judge something no author wrote. The cap keeps one
/// enormous page from consuming a whole round's budget.
pub fn chunk(text: &str, size: usize, max_chunks: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for para in split_paragraphs(text, size) {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if !current.is_empty() && current.len() + para.len() > size {
            if current.trim().len() > 80 {
                chunks.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            if max_chunks > 0 && chunks.len() >= max_chunks {
                return chunks;
            }
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if current.trim().len() > 80 {
        chunks.push(current);
    }
    if max_chunks > 0 && chunks.len() > max_chunks {
        chunks.truncate(max_chunks);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- pdf tests -----------------------------------------------------------

    /// A minimal, entirely-ASCII single-page PDF built with correct xref
    /// offsets, so the extraction test carries no binary fixture file.
    fn minimal_pdf(text: &str) -> Vec<u8> {
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_vec(),
            format!("<< /Length {} >>\nstream\nBT /F1 24 Tf 72 720 Td ({text}) Tj ET\nendstream", format!("BT /F1 24 Tf 72 720 Td ({text}) Tj ET").len()).into_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ];
        let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, obj) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(obj);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref_pos = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn pdf_text_extracts_from_a_real_pdf() {
        let bytes = minimal_pdf("NEOTEC 2024 GROWTHROADAI");
        let text = pdf_text(&bytes).expect("minimal pdf parses");
        assert!(text.contains("GROWTHROADAI"), "{text:?}");
    }

    /// The exact shape that crashed a harvest run: a ColorSpace array whose
    /// first element is an indirect reference, where pdf-extract's
    /// `as_name().expect(..)` panics (lib.rs:1459, measured 2026-09-23 on
    /// q102's source register). Built by hand rather than captured so the
    /// fixture stays readable and the xref offsets stay derived.
    fn colorspace_reference_pdf() -> Vec<u8> {
        let content = "/CS0 cs 0 0 0 sc BT /F1 24 Tf 72 720 Td (x) Tj ET";
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> /ColorSpace << /CS0 [6 0 R /DeviceRGB] >> >> >>".to_vec(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()).into_bytes(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /N 3 >>".to_vec(),
        ];
        let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, obj) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(obj);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref_pos = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    /// Serve one HTTP response on 127.0.0.1 and return its URL. Loopback
    /// only: nothing here touches the network.
    async fn serve_once(content_type: &'static str, body: Vec<u8>, path: &str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\n\
                     connection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(&body).await;
                let _ = sock.shutdown().await;
            }
        });
        format!("http://{addr}{path}")
    }

    fn plain_fetcher() -> Fetcher {
        install_crypto();
        let jina = Jina::new("k".into(), 1, Duration::from_secs(2)).unwrap();
        Fetcher::new(Backend::Jina(jina), Duration::from_secs(5)).unwrap()
    }

    #[tokio::test]
    async fn a_static_pdf_is_read_over_plain_http() {
        let url = serve_once(
            "application/pdf",
            minimal_pdf("A Public Key Cryptosystem ElGamal"),
            "/papers/elgamal.pdf",
        )
        .await;
        let page = plain_fetcher()
            .try_http(&url)
            .await
            .expect("a static PDF must not need the browser");
        assert!(page.text.contains("ElGamal"), "{:?}", page.text);
        assert_eq!(page.title, "elgamal.pdf");
        assert!(!page.rendered);
    }

    #[tokio::test]
    async fn a_challenge_page_posing_as_a_pdf_goes_to_the_browser() {
        // A bot wall answers a .pdf URL with HTML: no %PDF magic, so the
        // plain path declines and the stealth browser gets its turn.
        let url = serve_once(
            "application/pdf",
            b"<html><body>Checking your browser...</body></html>".to_vec(),
            "/papers/elgamal.pdf",
        )
        .await;
        assert!(plain_fetcher().try_http(&url).await.is_none());
    }

    #[test]
    fn a_panicking_pdf_parser_is_a_dropped_fetch_not_a_crash() {
        let reason = pdf_text(&colorspace_reference_pdf()).expect_err("must not parse cleanly");
        assert!(reason.contains("panicked"), "{reason}");
    }

    #[test]
    fn a_textless_pdf_is_reported_as_no_text() {
        // A valid document whose stream draws nothing textual: extraction
        // succeeds but yields nothing worth screening.
        let bytes = minimal_pdf("");
        assert_eq!(pdf_text(&bytes).unwrap_err(), "no text");
    }

    #[test]
    fn payload_downcast_reads_both_payload_shapes() {
        let lit: Box<dyn std::any::Any + Send> = Box::new("literal");
        assert_eq!(payload_downcast(&lit), "literal");
        let owned: Box<dyn std::any::Any + Send> = Box::new(String::from("owned"));
        assert_eq!(payload_downcast(&owned), "owned");
        let opaque: Box<dyn std::any::Any + Send> = Box::new(42usize);
        assert_eq!(payload_downcast(&opaque), "non-string panic payload");
    }

    #[test]
    fn is_pdf_url_judges_by_the_path_extension() {
        assert!(is_pdf_url(
            "https://www.cdti.es/sites/default/files/2024-12/resolucion.pdf"
        ));
        // Case-insensitive: registers are inconsistent about their own naming.
        assert!(is_pdf_url("https://example.org/Report.PDF"));
        assert!(!is_pdf_url("https://example.org/resolucion.html"));
        assert!(!is_pdf_url("https://example.org/"));
        // A query string does not make the path a PDF.
        assert!(!is_pdf_url("https://example.org/view?doc=resolucion.pdf"));
        assert!(!is_pdf_url("not a url"));
    }

    #[test]
    fn pdf_title_takes_the_decoded_filename() {
        assert_eq!(
            pdf_title("https://boe.es/boe/dias/2024/04/05/pdfs/BOE-B-2024-12121.pdf"),
            "BOE-B-2024-12121.pdf"
        );
        assert_eq!(
            pdf_title("https://example.org/files/convocatoria%20neotec.pdf"),
            "convocatoria neotec.pdf"
        );
    }

    #[test]
    fn percent_decoding_leaves_a_trailing_partial_escape_alone() {
        assert_eq!(percent_encoding_recover("a%20b"), "a b");
        assert_eq!(percent_encoding_recover("100%"), "100%");
        assert_eq!(percent_encoding_recover("trailing%2"), "trailing%2");
        assert_eq!(percent_encoding_recover("%zz"), "%zz");
    }

    // --- extract_links tests ------------------------------------------------

    #[test]
    fn extract_links_resolves_relative_hrefs() {
        let html = r#"<a href="/page">Abs-path</a> <a href="sub/page.html">Relative</a>"#;
        let links = extract_links(html, "https://example.com/base/");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].href, "https://example.com/page");
        assert_eq!(links[1].href, "https://example.com/base/sub/page.html");
    }

    #[test]
    fn extract_links_deduplicates_by_href() {
        let html = r#"<a href="/page">First</a> <a href="/page">Duplicate</a>"#;
        let links = extract_links(html, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].text, "First");
    }

    #[test]
    fn extract_links_skips_unwanted_schemes_and_fragments() {
        // Use r##"..."## so the href="#section" inside does not close the delimiter.
        let html = r##"
            <a href="javascript:void(0)">JS</a>
            <a href="mailto:info@example.com">Email</a>
            <a href="tel:+34912345678">Phone</a>
            <a href="#section">Anchor-only</a>
            <a href="/valid">Valid</a>
        "##;
        let links = extract_links(html, "https://example.com/");
        assert_eq!(links.len(), 1, "only the /valid href should survive");
        assert_eq!(links[0].href, "https://example.com/valid");
    }

    #[test]
    fn extract_links_caps_at_300() {
        let mut html = String::new();
        for i in 0..400 {
            html.push_str(&format!(r#"<a href="/page{i}">Link {i}</a>"#));
        }
        let links = extract_links(&html, "https://example.com/");
        assert_eq!(links.len(), 300);
    }

    #[test]
    fn extract_links_strips_tags_and_collapses_whitespace_in_text() {
        let html = r#"<a href="/p"><span>Hello</span>  <b>  World  </b></a>"#;
        let links = extract_links(html, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].text, "Hello World");
    }

    #[test]
    fn extract_links_recovers_logo_wall_names_from_alt_and_title_and_aria() {
        // Anchor around a bare logo image: no inner text, alt carries the name.
        let html =
            r#"<a href="https://foo.example/"><img src="/logo.png" alt="Foo Cooperative"></a>"#;
        let links = extract_links(html, "https://directory.example/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].text, "Foo Cooperative");

        // Title on the anchor wins over an inner img alt when text is empty.
        let html = r#"<a href="https://bar.example/" title="Bar SL"><img alt="ignored"></a>"#;
        let links = extract_links(html, "https://directory.example/");
        assert_eq!(links[0].text, "Bar SL");

        // Aria-label fallback when everything else is missing.
        let html = r#"<a href="https://baz.example/" aria-label="Baz Ltd"><img></a>"#;
        let links = extract_links(html, "https://directory.example/");
        assert_eq!(links[0].text, "Baz Ltd");

        // Existing inner text still wins — no regression on normal links.
        let html = r#"<a href="/p" title="ignored">Actual Text</a>"#;
        let links = extract_links(html, "https://example.com/");
        assert_eq!(links[0].text, "Actual Text");
    }

    #[test]
    fn html_to_text_emits_img_alt_inline() {
        // P8: a logo wall's partner names live in `<img alt>`. Emitting them
        // inline means the text-only path still contains the names for Jev
        // triage and downstream extraction.
        let html =
            r#"<div><img alt="Foo Coop" src="/f.png"> <img alt="Bar SL" src="/b.png"></div>"#;
        let text = html_to_text(html);
        assert!(text.contains("Foo Coop"), "text was {text:?}");
        assert!(text.contains("Bar SL"), "text was {text:?}");

        // An img with no alt (or empty alt) contributes nothing but does not
        // break surrounding text.
        let html = r#"<p>Before <img src="/x.png" alt=""> after</p>"#;
        let text = html_to_text(html);
        assert!(text.contains("Before"));
        assert!(text.contains("after"));
    }

    #[test]
    fn extract_links_truncates_text_at_160_chars() {
        let long_text = "x".repeat(300);
        let html = format!(r#"<a href="/p">{long_text}</a>"#);
        let links = extract_links(&html, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].text.chars().count(), 160);
    }

    // --- extract_markdown_links tests --------------------------------------

    #[test]
    fn extract_markdown_links_parses_inline_links_and_resolves_relative() {
        let md = "See [About](/about) or [Home](https://example.com/).";
        let links = extract_markdown_links(md, "https://example.com/base/");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].href, "https://example.com/about");
        assert_eq!(links[0].text, "About");
        assert_eq!(links[1].href, "https://example.com/");
        assert_eq!(links[1].text, "Home");
    }

    #[test]
    fn extract_markdown_links_captures_bare_urls() {
        let md = "raw: https://a.example/x and also http://b.example/y.";
        let links = extract_markdown_links(md, "https://base.example/");
        let hrefs: Vec<&str> = links.iter().map(|l| l.href.as_str()).collect();
        assert!(hrefs.contains(&"https://a.example/x"));
        assert!(hrefs.contains(&"http://b.example/y"));
    }

    #[test]
    fn extract_markdown_links_deduplicates_across_forms() {
        let md = "[X](https://a.example/x) and again: https://a.example/x";
        let links = extract_markdown_links(md, "https://base.example/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].href, "https://a.example/x");
    }

    #[test]
    fn extract_markdown_links_skips_unwanted_schemes_and_fragments() {
        let md = "[m](mailto:a@b.com) [t](tel:+1) [j](javascript:alert(1)) [a](#top) [ok](/ok)";
        let links = extract_markdown_links(md, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].href, "https://example.com/ok");
    }

    #[test]
    fn extract_markdown_links_strips_trailing_punctuation_on_bare_urls() {
        let md = "See https://a.example/x, and https://b.example/y.";
        let links = extract_markdown_links(md, "https://base.example/");
        let hrefs: Vec<&str> = links.iter().map(|l| l.href.as_str()).collect();
        assert!(hrefs.contains(&"https://a.example/x"));
        assert!(hrefs.contains(&"https://b.example/y"));
    }

    #[test]
    fn extract_markdown_links_handles_title_after_url() {
        let md = r#"[Docs](https://example.com/docs "The docs")"#;
        let links = extract_markdown_links(md, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].href, "https://example.com/docs");
    }

    #[test]
    fn extract_markdown_links_caps_at_300() {
        let mut md = String::new();
        for i in 0..400 {
            md.push_str(&format!("[L{i}](/p{i}) "));
        }
        let links = extract_markdown_links(&md, "https://example.com/");
        assert_eq!(links.len(), 300);
    }

    #[test]
    fn extract_markdown_links_parses_links_summary_section() {
        // What r.jina.ai actually appends when X-With-Links-Summary is set:
        // a markdown "Links/Buttons" section. The same extractor handles it.
        let md = "\n\nLinks/Buttons:\n\n- [Contact](/contact)\n- [Register](https://reg.example/)";
        let links = extract_markdown_links(md, "https://ayto.example/");
        let hrefs: Vec<&str> = links.iter().map(|l| l.href.as_str()).collect();
        assert!(hrefs.contains(&"https://ayto.example/contact"));
        assert!(hrefs.contains(&"https://reg.example/"));
    }

    #[test]
    fn extract_markdown_links_truncates_text_at_160_chars() {
        let long = "a".repeat(300);
        let md = format!("[{long}](/p)");
        let links = extract_markdown_links(&md, "https://example.com/");
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].text.chars().count(), 160);
    }

    // --- unwrap_redirect tests ----------------------------------------------

    #[test]
    fn unwraps_duckduckgo_redirects() {
        assert_eq!(
            unwrap_redirect("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdl%2F&rut=abc"),
            "https://go.dev/dl/"
        );
        assert_eq!(
            unwrap_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fx%3Fy%3Dz"),
            "https://a.example/x?y=z"
        );
    }

    #[test]
    fn passes_direct_links_through() {
        assert_eq!(unwrap_redirect("https://go.dev/dl/"), "https://go.dev/dl/");
    }

    #[test]
    fn rejects_internal_duckduckgo_navigation() {
        assert_eq!(unwrap_redirect("https://duckduckgo.com/settings"), "");
        assert_eq!(unwrap_redirect(""), "");
    }

    #[test]
    fn encodes_query_components() {
        assert_eq!(urlencode("a b&c"), "a+b%26c");
        assert_eq!(urlencode("caf\u{e9}"), "caf%C3%A9");
    }

    #[test]
    fn dedupes_and_unwraps_hits() {
        let hits = vec![
            Hit {
                title: "a".into(),
                url: "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fx.org%2F".into(),
                snippet: String::new(),
                ..Default::default()
            },
            Hit {
                title: "b".into(),
                url: "https://x.org/".into(),
                snippet: String::new(),
                ..Default::default()
            },
            Hit {
                title: "c".into(),
                url: "https://y.org/".into(),
                snippet: String::new(),
                ..Default::default()
            },
        ];
        let out = dedupe_hits(hits, 0);
        assert_eq!(
            out.len(),
            2,
            "the wrapped and direct form are the same page"
        );
        assert_eq!(out[0].url, "https://x.org/");
    }

    // --- canonical_url / title_key / merge_hits ------------------------------

    /// reqwest needs a rustls provider chosen before a client is built.
    /// `main` does it at startup; a test binary has to do it itself.
    fn install_crypto() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    fn hit(title: &str, url: &str, snippet: &str, engine: &str) -> Hit {
        Hit {
            title: title.into(),
            url: url.into(),
            snippet: snippet.into(),
            engines: vec![engine.to_string()],
        }
    }

    /// A fully cached lane answers without any network access, which is what
    /// makes this testable at all: nothing here may touch the web.
    #[tokio::test]
    async fn search_many_serves_a_cached_lane_without_calling_out() {
        install_crypto();
        let dir = std::env::temp_dir().join(format!(
            "webscout-lane-test-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let cache = crate::search_cache::SearchCache::new(&dir, Duration::from_secs(600));
        cache.put(
            "jina",
            "decidim partners",
            5,
            &[hit("A", "https://a.example/", "s", "jina")],
        );

        // A Jina client makes no request until it is asked to search, and the
        // cache means it never is. The key is a placeholder for that reason.
        let jina = Jina::new("test-key".into(), 2, Duration::from_secs(1)).unwrap();
        let fetcher = Fetcher::new(Backend::Jina(jina.clone()), Duration::from_secs(1))
            .unwrap()
            .with_lanes(vec![SearchLane::jina(jina, Duration::from_millis(50))])
            .with_search_cache(Some(cache));

        let out = fetcher
            .search_many(&["Decidim   Partners".to_string()], 5)
            .await;
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].0, "Decidim   Partners",
            "keyed by the query as asked"
        );
        assert_eq!(out[0].1.len(), 1);
        assert_eq!(out[0].1[0].url, "https://a.example/");
        assert_eq!(out[0].1[0].engines, vec!["jina".to_string()]);

        let (hits, misses, bytes) = fetcher.search_cache.as_ref().unwrap().summary();
        assert_eq!((hits, misses), (1, 0));
        assert!(bytes > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn search_many_with_no_lanes_returns_empty_results_per_query() {
        install_crypto();
        let jina = Jina::new("k".into(), 1, Duration::from_secs(1)).unwrap();
        let fetcher = Fetcher::new(Backend::Jina(jina), Duration::from_secs(1))
            .unwrap()
            .with_lanes(Vec::new());
        let out = fetcher.search_many(&["q".to_string()], 3).await;
        assert_eq!(out.len(), 1);
        assert!(out[0].1.is_empty());
    }

    /// A batch gets one deadline per wave it needs, never less than one.
    #[test]
    fn lane_budget_scales_with_the_waves_a_batch_needs() {
        let d = Duration::from_secs(20);
        assert_eq!(lane_budget(d, 5, 12), d, "a normal round: one wave");
        assert_eq!(lane_budget(d, 25, 12), d * 3, "q81's enrichment batch");
        assert_eq!(lane_budget(d, 50, 12), d * 5);
        assert_eq!(lane_budget(d, 0, 12), d, "never zero");
        assert_eq!(
            lane_budget(d, 3, 0),
            d * 3,
            "a zero concurrency reads as one"
        );
    }

    #[test]
    fn lane_ids_are_stable() {
        install_crypto();
        // The id is part of the cache key and of `Hit.engines`; changing one
        // silently invalidates cached entries and breaks engine reporting.
        let jina = Jina::new("k".into(), 1, Duration::from_secs(1)).unwrap();
        assert_eq!(SearchLane::jina(jina, DEFAULT_LANE_DEADLINE).id, "jina");
        assert_eq!(DEFAULT_LANE_DEADLINE, Duration::from_secs(20));
    }

    #[test]
    fn canonical_url_folds_scheme_www_and_trailing_slash() {
        let want = "example.com/a/b";
        for u in [
            "https://example.com/a/b",
            "http://example.com/a/b",
            "https://www.example.com/a/b/",
            "https://WWW.Example.COM/a/b///",
            "https://example.com/a/b#section",
        ] {
            assert_eq!(canonical_url(u), want, "{u}");
        }
    }

    #[test]
    fn canonical_url_drops_tracking_and_sorts_the_rest() {
        assert_eq!(
            canonical_url(
                "https://example.com/p?utm_source=x&utm_medium=y&id=7&fbclid=z&page=2&ref_src=q"
            ),
            "example.com/p?id=7&page=2"
        );
        // Every named tracking parameter goes, and an empty query leaves no `?`.
        assert_eq!(
            canonical_url(
                "https://example.com/p?ref=a&gclid=b&igshid=c&si=d&s=e&t=f&feature=g&lang=es"
            ),
            "example.com/p"
        );
        // Parameter order does not change the key.
        assert_eq!(
            canonical_url("https://example.com/p?b=2&a=1"),
            canonical_url("https://example.com/p?a=1&b=2")
        );
    }

    #[test]
    fn canonical_url_keeps_content_selecting_parameters() {
        // A query parameter that selects *what* the page shows must survive,
        // or two different pages collapse into one.
        assert_ne!(
            canonical_url("https://example.com/search?q=alpha"),
            canonical_url("https://example.com/search?q=beta")
        );
        assert_eq!(
            canonical_url("https://example.com/wiki?title=Decidim"),
            "example.com/wiki?title=Decidim"
        );
    }

    #[test]
    fn display_url_strips_the_cookie_wall_litter_but_keeps_the_document() {
        // The measured case: Springer's cookie-wall redirect, pasted into an
        // answer as "the link to the paper".
        assert_eq!(
            display_url(
                "https://link.springer.com/chapter/10.1007/3-540-39568-7_2\
                 ?error=cookies_not_supported&code=b980f9c2-b2af-4653-9fc6-128d080de6aa"
            ),
            "https://link.springer.com/chapter/10.1007/3-540-39568-7_2"
        );
        // Campaign litter goes; the scheme, host case-folding by `Url`, path
        // and fragment all stay, so the link still opens what was read.
        assert_eq!(
            display_url("https://example.org/a/b?utm_source=x&id=7&fbclid=y#sec3"),
            "https://example.org/a/b?id=7#sec3"
        );
    }

    #[test]
    fn display_url_keeps_what_selects_the_content() {
        // A short `code` selects content — a country, a language, a plan.
        assert_eq!(
            display_url("https://example.org/p?code=ES"),
            "https://example.org/p?code=ES"
        );
        // `lang` is dropped by canonical_url for dedup and must survive here:
        // stripping it sends the reader to a different translation.
        assert_eq!(
            display_url("https://example.org/p?lang=ca"),
            "https://example.org/p?lang=ca"
        );
        // Ordinary query pages are untouched.
        assert_eq!(
            display_url("https://example.org/search?q=elgamal&page=2"),
            "https://example.org/search?q=elgamal&page=2"
        );
        // A long random `code` is a session token even without an `error`.
        assert_eq!(
            display_url("https://example.org/p?code=b980f9c2b2af46539fc6128d080de6aa"),
            "https://example.org/p"
        );
    }

    #[test]
    fn display_url_returns_something_citable_for_garbage() {
        // Provenance we cannot parse is still the provenance we have.
        assert_eq!(display_url("  not a url  "), "not a url");
        assert_eq!(display_url(""), "");
    }

    #[test]
    fn canonical_url_survives_garbage() {
        assert_eq!(canonical_url("  NOT a url "), "not a url");
        assert_eq!(canonical_url(""), "");
    }

    #[test]
    fn title_key_strips_site_names_and_caps_at_eight_words() {
        assert_eq!(
            title_key("Decidim launches new release — Reuters"),
            "decidim launches new release"
        );
        assert_eq!(
            title_key("Decidim launches new release | El País"),
            "decidim launches new release"
        );
        assert_eq!(
            title_key("Decidim launches new release: BBC"),
            "decidim launches new release"
        );
        // Eight alphanumeric words, punctuation ignored, lowercased.
        assert_eq!(
            title_key("One, two; three four five six seven eight nine ten"),
            "one two three four five six seven eight"
        );
        // A long trailing segment is part of the headline, not a masthead.
        let k = title_key("Barcelona: how the city built its participation platform");
        assert!(k.starts_with("barcelona how the city"), "got {k}");
    }

    #[test]
    fn title_key_matches_syndicated_copies() {
        assert_eq!(
            title_key("EU adopts the AI Act - Reuters"),
            title_key("EU adopts the AI Act | Politico")
        );
        assert_ne!(
            title_key("EU adopts the AI Act"),
            title_key("EU delays the AI Act")
        );
        assert_eq!(title_key("   "), "");
    }

    #[test]
    fn merge_hits_unions_engines_on_agreement() {
        let ddg = vec![
            hit("A", "https://www.a.example/x/", "short", "ddg"),
            hit("B", "https://b.example/", "b snippet", "ddg"),
        ];
        let jina = vec![hit(
            "A",
            "http://a.example/x?utm_source=news",
            "a much longer snippet",
            "jina",
        )];

        let merged = merge_hits(vec![ddg, jina], 0);
        assert_eq!(merged.len(), 2, "the two forms of A are one page");
        let a = &merged[0];
        assert_eq!(a.engines, vec!["ddg".to_string(), "jina".to_string()]);
        assert_eq!(a.snippet, "a much longer snippet", "longer snippet wins");
        assert_eq!(a.url, "https://www.a.example/x/", "first form is kept");
    }

    #[test]
    fn merge_hits_keeps_the_best_position_and_fills_empty_titles() {
        // A is third on ddg and first on jina, so it should outrank B.
        let ddg = vec![
            hit("B", "https://b.example/", "", "ddg"),
            hit("C", "https://c.example/", "", "ddg"),
            hit("", "https://a.example/", "", "ddg"),
        ];
        let jina = vec![hit("A title", "https://a.example/", "", "jina")];

        let merged = merge_hits(vec![ddg, jina], 0);
        // A was third on ddg and first on jina, and two engines returned it:
        // best position plus agreement puts it ahead of B, which only ddg
        // ranked first.
        assert_eq!(merged[0].url, "https://a.example/");
        assert_eq!(merged[0].title, "A title", "an empty title is filled in");
        assert_eq!(merged[0].engines.len(), 2);
        assert_eq!(merged[1].url, "https://b.example/");
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn merge_hits_drops_syndicated_duplicate_titles() {
        let ddg = vec![
            hit(
                "EU adopts the AI Act — Reuters",
                "https://r.example/1",
                "",
                "ddg",
            ),
            hit(
                "EU adopts the AI Act | Politico",
                "https://p.example/2",
                "",
                "ddg",
            ),
            hit("Something else entirely", "https://x.example/3", "", "ddg"),
        ];
        let merged = merge_hits(vec![ddg], 0);
        assert_eq!(merged.len(), 2, "one story must not take two slots");
        assert_eq!(merged[0].url, "https://r.example/1");
        assert_eq!(merged[1].url, "https://x.example/3");
    }

    /// A two-word title is a programme name, not a story identity: the
    /// funder's own page, a grants tracker and a ministry announcement are
    /// distinct pages that legitimately share it (measured 2026-09-21, q81:
    /// 232 "syndicated copy" drops, including the authoritative resolution
    /// list the run was looking for).
    #[test]
    fn merge_hits_keeps_distinct_pages_sharing_a_short_title() {
        let ddg = vec![
            hit(
                "NEOTEC 2024",
                "https://www.cdti.es/ayudas/ayudas-neotec-2024",
                "",
                "ddg",
            ),
            hit(
                "NEOTEC 2024",
                "https://subvencionespublicas.com/neotec-2024/",
                "",
                "ddg",
            ),
            hit(
                "NEOTEC 2024",
                "https://www.ciencia.gob.es/Noticias/enero/neotec-2024-convocatoria",
                "",
                "ddg",
            ),
        ];
        let merged = merge_hits(vec![ddg], 0);
        assert_eq!(
            merged.len(),
            3,
            "a programme name shared by distinct pages is not syndication"
        );
    }

    #[test]
    fn merge_hits_honours_the_limit_and_ignores_empty_urls() {
        let lane = vec![
            hit("A", "https://a.example/", "", "ddg"),
            hit("B", "", "", "ddg"),
            hit("C", "https://c.example/", "", "ddg"),
            hit("D", "https://d.example/", "", "ddg"),
        ];
        let merged = merge_hits(vec![lane], 2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].url, "https://a.example/");
        assert_eq!(merged[1].url, "https://c.example/");
    }

    #[test]
    fn merge_hits_of_nothing_is_nothing() {
        assert!(merge_hits(Vec::new(), 5).is_empty());
        assert!(merge_hits(vec![Vec::new(), Vec::new()], 5).is_empty());
    }

    #[test]
    fn tracking_params_are_recognised_by_prefix_and_name() {
        for k in [
            "utm_source",
            "UTM_Medium",
            "ref",
            "ref_src",
            "fbclid",
            "si",
            "s",
            "t",
            "lang",
        ] {
            assert!(is_tracking_param(k), "{k} should be tracking noise");
        }
        for k in ["q", "id", "page", "title", "search"] {
            assert!(!is_tracking_param(k), "{k} selects content");
        }
    }

    #[test]
    fn dedupe_hits_uses_the_canonical_key() {
        // Same page, three disguises: a redirect wrapper, a www. form, and a
        // campaign parameter. The raw-string comparison kept all three.
        let hits = vec![
            hit(
                "a",
                "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fx.org%2Fa%2F",
                "",
                "ddg",
            ),
            hit("b", "https://www.x.org/a", "", "ddg"),
            hit("c", "https://x.org/a?utm_source=nl", "", "ddg"),
            hit("d", "https://y.org/", "", "ddg"),
        ];
        let out = dedupe_hits(hits, 0);
        assert_eq!(
            out.len(),
            2,
            "got {:?}",
            out.iter().map(|h| &h.url).collect::<Vec<_>>()
        );
        assert_eq!(out[0].url, "https://x.org/a/");
        assert_eq!(out[1].url, "https://y.org/");
    }

    #[test]
    fn chunks_on_paragraph_boundaries() {
        let para = "x".repeat(300);
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = chunk(&text, 400, 0);
        assert!(chunks.len() >= 2, "expected a split, got {}", chunks.len());
        for c in &chunks {
            assert!(c.len() >= 80);
        }
    }

    #[test]
    fn drops_fragments_too_short_to_judge() {
        // Nav items and headings make up most of a page and can answer nothing.
        assert!(chunk("# Title\n\nHome\n\nAbout", 1000, 0).is_empty());
    }

    /// Regression: `innerText` separates blocks with a single newline, so a page with
    /// no blank lines was treated as one paragraph and the size cap never applied. A
    /// 500,000-character page became a single chunk, which the verification API then
    /// rejected for exceeding its per-request token limit.
    #[test]
    fn hard_splits_text_that_has_no_blank_lines() {
        let text = "word ".repeat(100_000);
        let chunks = chunk(&text, 4000, 0);
        assert!(
            chunks.len() > 100,
            "expected many chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(
                c.len() <= 4200,
                "chunk of {} chars exceeds the cap",
                c.len()
            );
        }
    }

    #[test]
    fn hard_split_never_panics_on_multibyte_text() {
        // Page text is full of accented and CJK characters; slicing on a byte index
        // that is not a character boundary would panic.
        let text = "café… 日本語テキスト ".repeat(20_000);
        let chunks = chunk(&text, 1000, 0);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.len() <= 1200);
        }
    }

    /// Regression: an earlier hand-rolled byte walk landed mid-character on a real
    /// Wikipedia page and panicked at its `»`. Anything touching scraped HTML has to
    /// survive multi-byte text.
    #[test]
    fn html_helpers_survive_multibyte_markup() {
        let html = concat!(
            "<html><head><title>Café » Paris</title></head><body>",
            "<p>Le café coûte 3€ — vraiment.</p>",
            "<script>var x = 1;</script>",
            "<div>日本語のテキスト</div></body></html>"
        );
        let text = html_to_text(html);
        assert!(text.contains("Le café coûte 3€"), "got: {text}");
        assert!(text.contains("日本語のテキスト"));
        assert!(!text.contains("var x"), "script contents must be dropped");
        assert!(!text.contains('<'), "markup must be gone");
        assert_eq!(extract_title(html), "Café » Paris");
    }

    /// The measured leak: a `<style>` or `<script>` block preceded by any text
    /// — a newline is enough — came through as page text. Real markup nearly
    /// always has that newline; the old fixture above only passed because it
    /// wrote `</p><script>` with nothing between.
    #[test]
    fn code_blocks_are_dropped_even_after_text() {
        let html = "<html>\n<head>\n<title>About Sid</title>\n\
                    <style>:root{--wp--preset--color:#000}</style>\n</head>\n<body>\n\
                    <p>Sid Sijbrandij co-founded GitLab.</p>\n\
                    <script>window.__DATA__ = {\"x\": 1};</script>\n\
                    <noscript>enable js</noscript>\n\
                    <svg><path d=\"M0 0L10 10\"/></svg>\n\
                    <p>He served as CEO from 2012 to 2024.</p>\n</body></html>";
        let text = html_to_text(html);
        assert!(
            text.contains("Sid Sijbrandij co-founded GitLab."),
            "{text:?}"
        );
        assert!(text.contains("CEO from 2012 to 2024"), "{text:?}");
        for junk in [
            "--wp--preset",
            "__DATA__",
            "enable js",
            "M0 0L10",
            "<title>",
        ] {
            assert!(!text.contains(junk), "{junk} leaked into {text:?}");
        }
    }

    /// Structured data is content: a schema.org block is often the only place
    /// a homepage states its own official URL, and it was the sole support for
    /// "the official website of Mondragon Corporation". Code around it goes.
    #[test]
    fn json_ld_structured_data_is_kept_as_text() {
        let html = "<html>\n<head>\n<script>var tracker = 1;</script>\n\
                    <script type=\"application/ld+json\">\n{\"@type\": \"Organization\",\n\
                    \"name\": \"MONDRAGON CORPORATION\",\n\
                    \"url\": \"https:\\/\\/www.mondragon-corporation.com\\/en\\/\"}\n</script>\n\
                    </head>\n<body>\n<p>Humanity at work.</p>\n</body></html>";
        let text = html_to_text(html);
        assert!(text.contains("MONDRAGON CORPORATION"), "{text:?}");
        assert!(
            text.contains("https://www.mondragon-corporation.com/en/"),
            "escaped slashes restored: {text:?}"
        );
        assert!(text.contains("Humanity at work."), "{text:?}");
        assert!(
            !text.contains("tracker"),
            "an ordinary script still goes: {text:?}"
        );

        // Bounded: an article's structured data can repeat its whole body.
        let long = format!(
            "<script type=\"application/ld+json\">{{\"articleBody\": \"{}\"}}</script>",
            "word ".repeat(5_000)
        );
        let text = html_to_text(&long);
        assert!(
            text.len() < 2_100,
            "json-ld not bounded: {} chars",
            text.len()
        );
    }

    /// With the skip working, a prefix test would read `<header>` as `<head`,
    /// look for a `</head>` that closed earlier, and drop the page body.
    #[test]
    fn a_header_element_is_not_the_head() {
        let html = "<html><head><title>t</title></head>\n<body>\n\
                    <header>Site navigation</header>\n<p>The article itself.</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Site navigation"), "{text:?}");
        assert!(text.contains("The article itself."), "{text:?}");
    }

    /// A code block with no closing tag costs that element, not the rest of
    /// the page.
    #[test]
    fn an_unclosed_code_block_does_not_truncate_the_page() {
        let html = "<p>Before.</p>\n<script src=\"a.js\">\n<p>After the broken script.</p>";
        let text = html_to_text(html);
        assert!(text.contains("Before."), "{text:?}");
        assert!(text.contains("After the broken script."), "{text:?}");
    }

    #[test]
    fn decodes_latin_named_entities() {
        // Measured 2026-09-21 on a Spanish colegios harvest: the HTTP path
        // emitted `T&eacute;c` while the browser path emitted `Téc`, and the
        // same colegio survived as two records under two keys.
        let text = html_to_text(
            "<p>Ingenieros T&eacute;cnicos, Agr&ograve;noms, &Ntilde;u&ntilde;ez, &Ccedil;AT</p>",
        );
        assert!(text.contains("Técnicos"), "got {text}");
        assert!(text.contains("Agrònoms"), "got {text}");
        assert!(text.contains("Ñuñez"), "got {text}");
        assert!(text.contains("ÇAT"), "got {text}");
    }

    #[test]
    fn double_escaped_entities_decode_once() {
        // `&amp;eacute;` is the literal text "&eacute;", not "é". Decoding
        // `&amp;` before the letter entities unescapes twice; it runs last.
        let text = html_to_text("<p>&amp;eacute; and &amp;amp;</p>");
        assert!(text.contains("&eacute;"), "got {text}");
        assert!(text.contains("&amp;"), "got {text}");
    }

    #[test]
    fn decodes_entities_including_numeric() {
        let text = html_to_text("<p>a&nbsp;&amp;&nbsp;b &#8212; &#x27;q&#x27; &hellip;</p>");
        assert!(text.contains("a & b"), "got: {text}");
        assert!(
            text.contains('—') && text.contains('\u{2026}'),
            "got: {text}"
        );
    }

    #[test]
    fn honors_the_chunk_cap() {
        let para = "y".repeat(200);
        let text = vec![para.as_str(); 40].join("\n\n");
        assert!(chunk(&text, 250, 3).len() <= 3);
    }
}
