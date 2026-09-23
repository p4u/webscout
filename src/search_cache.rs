//! On-disk cache for search-lane responses.
//!
//! A research run re-issues the same query far more often than it looks: the
//! research plan, the round planner and the enrich stage all converge on
//! `{anchor} {entity_type}`-shaped strings, and a re-run of the same mission
//! repeats nearly all of them. A search is the cheapest thing in the pipeline
//! to cache and one of the slowest to redo — a DuckDuckGo scrape is a full
//! browser render.
//!
//! Only *successful, non-empty* responses are stored. An empty response is
//! usually a transient block or a markup change, and caching it would freeze
//! the failure in for the TTL.
//!
//! Everything here is injectable: `SearchCache::new` takes the directory and
//! the TTL, so tests point at a temp dir and never touch the real one.
//! `SearchCache::discover` is the production constructor.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::types::Hit;

/// Default time a cached search stays usable. Six hours: long enough that a
/// re-run or a second mission on the same subject is free, short enough that
/// "what changed this morning" is still answerable.
pub const DEFAULT_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// One cache file: the answer plus the identity of the question, so a hash
/// collision cannot silently serve someone else's results.
#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    lane: String,
    query: String,
    limit: usize,
    /// Unix seconds when this was written.
    stored_at: u64,
    hits: Vec<Hit>,
}

/// Counters for the end-of-run line. Shared by reference, so a `&SearchCache`
/// behind a `Fetcher` can still account for itself.
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub bytes: AtomicU64,
}

/// A directory of JSON search responses, keyed by lane + query + limit.
#[derive(Debug)]
pub struct SearchCache {
    dir: PathBuf,
    ttl: Duration,
    pub stats: CacheStats,
}

impl SearchCache {
    /// Explicit directory and TTL. Nothing is created until the first write,
    /// so constructing one is free and cannot fail.
    pub fn new(dir: impl Into<PathBuf>, ttl: Duration) -> Self {
        Self {
            dir: dir.into(),
            ttl,
            stats: CacheStats::default(),
        }
    }

    /// The production location: `$XDG_CACHE_HOME/webscout/search`, falling
    /// back to `$HOME/.cache/webscout/search` and then to the system temp
    /// directory. The temp fallback keeps the cache working on a machine with
    /// no home directory (a container) rather than disabling it.
    pub fn discover(ttl: Duration) -> Self {
        Self::new(default_cache_dir(), ttl)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    fn path_for(&self, lane: &str, query: &str, limit: usize) -> PathBuf {
        self.dir
            .join(format!("{}.json", cache_key(lane, query, limit)))
    }

    /// Look up a stored response. An expired or unparsable entry is a miss;
    /// the next `put` overwrites it, so nothing has to be cleaned up here.
    pub fn get(&self, lane: &str, query: &str, limit: usize) -> Option<Vec<Hit>> {
        let path = self.path_for(lane, query, limit);
        let raw = match std::fs::read(&path) {
            Ok(r) => r,
            Err(_) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(lane, query, "search cache miss");
                return None;
            }
        };
        let entry: Entry = match serde_json::from_slice(&raw) {
            Ok(e) => e,
            Err(e) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(lane, query, error = %e, "search cache entry unreadable; treating as a miss");
                return None;
            }
        };
        if entry.lane != lane || entry.limit != limit || entry.query != normalize_query(query) {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                lane,
                query,
                "search cache key collision; treating as a miss"
            );
            return None;
        }
        if age_secs(entry.stored_at) > self.ttl.as_secs() {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(lane, query, "search cache entry expired");
            return None;
        }
        if entry.hits.is_empty() {
            // Should never have been written; treat as a miss rather than
            // serving an empty round from disk.
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes
            .fetch_add(raw.len() as u64, Ordering::Relaxed);
        tracing::debug!(
            lane,
            query,
            hits = entry.hits.len(),
            bytes = raw.len(),
            "search cache hit"
        );
        Some(entry.hits)
    }

    /// Store a response. Empty responses are never written: an empty result is
    /// how a block or a markup change looks, and freezing that in for the TTL
    /// would turn a transient failure into a lasting one.
    pub fn put(&self, lane: &str, query: &str, limit: usize, hits: &[Hit]) {
        if hits.is_empty() {
            return;
        }
        if let Err(e) = std::fs::create_dir_all(&self.dir) {
            tracing::debug!(dir = %self.dir.display(), error = %e, "could not create the search cache directory");
            return;
        }
        let entry = Entry {
            lane: lane.to_string(),
            query: normalize_query(query),
            limit,
            stored_at: now_secs(),
            hits: hits.to_vec(),
        };
        let Ok(body) = serde_json::to_vec(&entry) else {
            return;
        };
        let path = self.path_for(lane, query, limit);
        // Write-then-rename, so a crash mid-write cannot leave a truncated
        // file that every later run has to parse and reject.
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &body).is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// `(hits, misses, bytes served)` for the end-of-run line.
    pub fn summary(&self) -> (usize, usize, u64) {
        (
            self.stats.hits.load(Ordering::Relaxed),
            self.stats.misses.load(Ordering::Relaxed),
            self.stats.bytes.load(Ordering::Relaxed),
        )
    }
}

/// Where the cache lives when nobody says otherwise.
pub fn default_cache_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
        let p = PathBuf::from(x);
        if !p.as_os_str().is_empty() {
            return p.join("webscout").join("search");
        }
    }
    if let Some(h) = std::env::var_os("HOME") {
        let p = PathBuf::from(h);
        if !p.as_os_str().is_empty() {
            return p.join(".cache").join("webscout").join("search");
        }
    }
    std::env::temp_dir().join("webscout").join("search")
}

/// Queries differing only in case or spacing are the same search.
pub fn normalize_query(q: &str) -> String {
    q.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Filename for a (lane, query, limit) triple.
///
/// FNV-1a rather than a cryptographic hash: this names a cache file, it does
/// not authenticate anything, and the entry records the original key so a
/// collision is detected on read rather than served.
pub fn cache_key(lane: &str, query: &str, limit: usize) -> String {
    let material = format!("{lane}\u{1}{}\u{1}{limit}", normalize_query(query));
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in material.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Age in seconds, saturating: a clock that went backwards yields 0 (fresh)
/// rather than a huge number that would invalidate the whole cache.
fn age_secs(stored_at: u64) -> u64 {
    now_secs().saturating_sub(stored_at)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory that removes itself. Avoids a dev-dependency
    /// on tempfile for four tests.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let pid = std::process::id();
            let p = std::env::temp_dir().join(format!("webscout-cache-test-{tag}-{pid}-{nanos}"));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn hit(url: &str) -> Hit {
        Hit {
            title: "t".into(),
            url: url.into(),
            snippet: "s".into(),
            engines: vec!["ddg".into()],
        }
    }

    #[test]
    fn miss_then_hit() {
        let dir = TempDir::new("miss-then-hit");
        let cache = SearchCache::new(dir.path(), DEFAULT_TTL);
        assert!(cache.get("ddg", "decidim partners", 8).is_none());
        cache.put("ddg", "decidim partners", 8, &[hit("https://a.example/")]);
        let got = cache.get("ddg", "decidim partners", 8).expect("hit");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].url, "https://a.example/");
        let (hits, misses, bytes) = cache.summary();
        assert_eq!((hits, misses), (1, 1));
        assert!(bytes > 0, "a hit must account for the bytes it served");
    }

    #[test]
    fn normalizes_the_query_and_keys_on_lane_and_limit() {
        let dir = TempDir::new("keys");
        let cache = SearchCache::new(dir.path(), DEFAULT_TTL);
        cache.put("ddg", "Decidim   Partners", 8, &[hit("https://a.example/")]);
        // Case and spacing differences are the same search.
        assert!(cache.get("ddg", "decidim partners", 8).is_some());
        // A different lane or limit is a different search.
        assert!(cache.get("jina", "decidim partners", 8).is_none());
        assert!(cache.get("ddg", "decidim partners", 20).is_none());
    }

    #[test]
    fn expired_entry_is_a_miss_and_is_overwritten() {
        let dir = TempDir::new("expiry");
        let cache = SearchCache::new(dir.path(), Duration::from_secs(1));
        cache.put("ddg", "q", 8, &[hit("https://old.example/")]);
        // Backdate the stored entry rather than sleeping.
        let path = dir
            .path()
            .join(format!("{}.json", cache_key("ddg", "q", 8)));
        let raw = std::fs::read(&path).unwrap();
        let mut entry: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        entry["stored_at"] = serde_json::json!(now_secs() - 3600);
        std::fs::write(&path, serde_json::to_vec(&entry).unwrap()).unwrap();

        assert!(cache.get("ddg", "q", 8).is_none(), "expired must miss");

        cache.put("ddg", "q", 8, &[hit("https://new.example/")]);
        let got = cache
            .get("ddg", "q", 8)
            .expect("fresh entry after overwrite");
        assert_eq!(got[0].url, "https://new.example/");
    }

    #[test]
    fn empty_responses_are_never_cached() {
        let dir = TempDir::new("empty");
        let cache = SearchCache::new(dir.path(), DEFAULT_TTL);
        cache.put("ddg", "q", 8, &[]);
        assert!(cache.get("ddg", "q", 8).is_none());
        assert!(
            !dir.path()
                .join(format!("{}.json", cache_key("ddg", "q", 8)))
                .exists(),
            "an empty response must not create a file"
        );
    }

    #[test]
    fn corrupt_file_is_treated_as_a_miss() {
        let dir = TempDir::new("corrupt");
        let cache = SearchCache::new(dir.path(), DEFAULT_TTL);
        let path = dir
            .path()
            .join(format!("{}.json", cache_key("ddg", "q", 8)));
        std::fs::write(&path, b"{not json").unwrap();
        assert!(cache.get("ddg", "q", 8).is_none());
        // And a later write still succeeds over the corrupt file.
        cache.put("ddg", "q", 8, &[hit("https://a.example/")]);
        assert!(cache.get("ddg", "q", 8).is_some());
    }

    #[test]
    fn default_cache_dir_prefers_xdg_then_home() {
        // Read-only inspection of the two shapes; no env mutation, which
        // would race with other tests in the same process.
        let dir = default_cache_dir();
        assert!(
            dir.ends_with("webscout/search"),
            "unexpected cache dir {}",
            dir.display()
        );
    }

    #[test]
    fn cache_key_is_stable_and_distinguishes_inputs() {
        assert_eq!(cache_key("ddg", "a b", 8), cache_key("ddg", " A  B ", 8));
        assert_ne!(cache_key("ddg", "a b", 8), cache_key("ddg", "a b", 9));
        assert_ne!(cache_key("ddg", "a b", 8), cache_key("jina", "a b", 8));
        assert_eq!(cache_key("ddg", "a b", 8).len(), 16);
    }
}
