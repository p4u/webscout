//! The run log behind `GET /api/stats`: one JSON line per finished search.
//!
//! **An append-only file, not a database.** A run is written once, when it
//! ends, and never changed; the whole log is read once, at startup. JSON Lines
//! is the smallest format that does that, survives a crash mid-write (the torn
//! last line is skipped on the next load, everything before it is intact) and
//! can be read with `jq` when someone wants a number the dashboard does not
//! show. At a few hundred bytes a run, a busy year is a few megabytes in memory.
//!
//! **Statistics never cost a run.** A log that cannot be written is a warning,
//! not a failure: the person waiting for an answer gets it either way.
//!
//! **Aggregation is pure.** `aggregate` takes the records and `now` and returns
//! the response, so bucket alignment, zero-filling and percentiles are tested
//! without a clock, a server or a file.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::api::{UsageSnapshot, rfc3339};
use crate::types::{MissionKind, ScoutReport};

/// Log file name inside the data directory.
const LOG_FILE: &str = "runs.jsonl";

/// Runs listed under `recent`.
const RECENT: usize = 50;

/// Widest range `/api/stats` serves, in days.
pub const MAX_DAYS: u32 = 365;

/// Widest range served with hourly buckets: 7 × 24 = 168 points is already
/// more than a chart can show one bar each for.
pub const MAX_HOURLY_DAYS: u32 = 7;

/// One finished (or failed, or cancelled) search.
///
/// `#[serde(default)]` so a log written by an older build, missing a field
/// added since, still loads instead of being skipped line by line.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunRecord {
    pub id: String,
    /// Unix seconds.
    pub started_at: u64,
    pub elapsed_ms: u64,
    pub query: String,
    /// `answer`, `harvest`, or `unknown` for a run that ended before its
    /// mission was parsed into a report.
    pub kind: String,
    /// `ui` or `mcp`.
    pub source: String,
    pub preset: String,
    /// A report outcome, or `failed` / `cancelled`.
    pub outcome: String,
    pub records: usize,
    pub pages: usize,
    pub rounds: usize,
    pub jev_requests: usize,
    pub jev_input_tokens: usize,
    pub jev_cost_usd: f64,
    pub llm_requests: usize,
    pub llm_prompt_tokens: usize,
    pub llm_completion_tokens: usize,
    /// `None` when the writer's endpoint reports no price.
    pub llm_cost_usd: Option<f64>,
    pub planner_requests: usize,
    pub planner_prompt_tokens: usize,
    pub planner_completion_tokens: usize,
    pub planner_cost_usd: Option<f64>,
    pub error: Option<String>,
}

impl RunRecord {
    /// What is known about a run when it starts. The outcome is `unknown`
    /// until one of `finished`, `failed` or `cancelled` says otherwise.
    pub fn begin(id: &str, query: &str, source: &str, preset: &str) -> Self {
        Self {
            id: id.to_string(),
            started_at: crate::api::unix_now(),
            query: query.to_string(),
            kind: "unknown".into(),
            source: source.to_string(),
            preset: preset.to_string(),
            outcome: "unknown".into(),
            ..Default::default()
        }
    }

    fn with_usage(mut self, u: &UsageSnapshot) -> Self {
        self.jev_requests = u.jev_requests;
        self.jev_input_tokens = u.jev_input_tokens;
        self.jev_cost_usd = u.jev_cost_usd;
        self.llm_requests = u.llm.requests;
        self.llm_prompt_tokens = u.llm.prompt_tokens;
        self.llm_completion_tokens = u.llm.completion_tokens;
        self.llm_cost_usd = u.llm.cost_usd;
        self.planner_requests = u.planner.requests;
        self.planner_prompt_tokens = u.planner.prompt_tokens;
        self.planner_completion_tokens = u.planner.completion_tokens;
        self.planner_cost_usd = u.planner.cost_usd;
        self
    }

    /// A run that produced a report, whatever its outcome.
    pub fn finished(self, report: &ScoutReport) -> Self {
        let st = &report.stats;
        let mut r = self.with_usage(&UsageSnapshot::from_stats(st));
        r.kind = match report.mission.kind {
            MissionKind::Answer => "answer",
            MissionKind::Harvest => "harvest",
        }
        .into();
        r.outcome = report.outcome.as_str().into();
        r.elapsed_ms = (st.elapsed_secs * 1000.0).round() as u64;
        r.records = report.records.len();
        r.pages = st.pages_fetched;
        r.rounds = st.rounds;
        r
    }

    /// A run that ended in an error. The spend is a sample of the clients'
    /// counters: a failed run still cost money.
    pub fn failed(self, error: &str, elapsed_ms: u64, usage: &UsageSnapshot) -> Self {
        let mut r = self.with_usage(usage);
        r.outcome = "failed".into();
        r.error = Some(error.to_string());
        r.elapsed_ms = elapsed_ms;
        r
    }

    /// A run stopped before it ended: a closed browser tab or `cancel_search`.
    pub fn cancelled(self, elapsed_ms: u64, usage: &UsageSnapshot) -> Self {
        let mut r = self.with_usage(usage);
        r.outcome = "cancelled".into();
        r.elapsed_ms = elapsed_ms;
        r
    }

    /// Jev plus whatever the writer and planner reported. An endpoint that
    /// reports no price adds nothing rather than making the total unknown: a
    /// dashboard sum that turns `null` on one run would be useless.
    pub fn cost_usd(&self) -> f64 {
        self.jev_cost_usd + self.llm_cost_usd.unwrap_or(0.0) + self.planner_cost_usd.unwrap_or(0.0)
    }

    fn elapsed_s(&self) -> f64 {
        self.elapsed_ms as f64 / 1000.0
    }
}

// ------------------------------------------------------------------ store --

/// Every run this server has logged, in memory, mirrored to `runs.jsonl`.
#[derive(Default)]
pub struct StatsStore {
    records: Mutex<Vec<RunRecord>>,
    /// `None` keeps the log in memory only (tests, or no data directory).
    path: Option<PathBuf>,
}

impl StatsStore {
    /// Load `<dir>/runs.jsonl` if it exists. Unparseable lines — a torn last
    /// write, a hand edit — are skipped with a warning rather than failing
    /// startup over a dashboard.
    pub fn open(dir: &Path) -> Self {
        let path = dir.join(LOG_FILE);
        let records = match std::fs::read_to_string(&path) {
            Ok(text) => parse_log(&text, &path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "could not read the run log; starting empty");
                Vec::new()
            }
        };
        tracing::info!(path = %path.display(), runs = records.len(), "run log loaded");
        Self {
            records: Mutex::new(records),
            path: Some(path),
        }
    }

    /// Log one run: in memory, then one line appended to the file.
    ///
    /// The file is written under the same lock as the push, so two runs ending
    /// together cannot interleave their lines.
    pub fn append(&self, rec: RunRecord) {
        let line = match serde_json::to_string(&rec) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "could not serialise a run record");
                return;
            }
        };
        let Ok(mut records) = self.records.lock() else {
            return;
        };
        records.push(rec);
        if let Some(path) = &self.path
            && let Err(e) = append_line(path, &line)
        {
            tracing::warn!(path = %path.display(), error = %e, "could not write the run log; the run is kept in memory only");
        }
    }

    /// Run `f` over the log without copying it.
    pub fn with_records<T>(&self, f: impl FnOnce(&[RunRecord]) -> T) -> T {
        match self.records.lock() {
            Ok(r) => f(&r),
            Err(_) => f(&[]),
        }
    }
}

fn parse_log(text: &str, path: &Path) -> Vec<RunRecord> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<RunRecord>(line) {
            Ok(r) => out.push(r),
            Err(e) => {
                tracing::warn!(path = %path.display(), line = i + 1, error = %e, "skipping an unreadable run log line")
            }
        }
    }
    out
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    // One write per line: with O_APPEND a single write lands whole.
    f.write_all(format!("{line}\n").as_bytes())
}

/// `$XDG_DATA_HOME/webscout`, else `$HOME/.local/share/webscout`.
pub fn default_data_dir() -> Option<PathBuf> {
    let nonempty = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
    if let Some(x) = nonempty("XDG_DATA_HOME") {
        return Some(PathBuf::from(x).join("webscout"));
    }
    nonempty("HOME").map(|h| PathBuf::from(h).join(".local/share/webscout"))
}

// ------------------------------------------------------------- aggregation --

/// Width of one point of the `series`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    Hour,
    Day,
    /// ISO week: Monday 00:00 UTC to the next Monday.
    Week,
}

impl Bucket {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "hour" => Some(Bucket::Hour),
            "day" => Some(Bucket::Day),
            "week" => Some(Bucket::Week),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::Hour => "hour",
            Bucket::Day => "day",
            Bucket::Week => "week",
        }
    }

    fn secs(self) -> u64 {
        match self {
            Bucket::Hour => 3_600,
            Bucket::Day => 86_400,
            Bucket::Week => 7 * 86_400,
        }
    }

    /// The start of the bucket containing `t`, in UTC.
    pub fn align(self, t: u64) -> u64 {
        match self {
            Bucket::Hour | Bucket::Day => t - t % self.secs(),
            Bucket::Week => {
                // 1970-01-01 was a Thursday: three days after a Monday.
                let day = t / 86_400;
                (day - (day + 3) % 7) * 86_400
            }
        }
    }
}

/// Clamp a requested range: 1..=365 days, and at most 7 for hourly buckets.
pub fn clamp_days(bucket: Bucket, days: i64) -> u32 {
    let max = if bucket == Bucket::Hour {
        MAX_HOURLY_DAYS
    } else {
        MAX_DAYS
    };
    days.clamp(1, i64::from(max)) as u32
}

/// Start of the first bucket of a `days`-long range ending at `now`.
///
/// By day the range is the last `days` calendar days, today included: 30 days
/// is exactly 30 points. By week it is the same days grouped into the ISO weeks
/// they fall in, so the week view and the day view cover the same runs apart
/// from the partial first week. By hour it is the last `days` × 24 hours, the
/// current hour included.
pub fn range_start(bucket: Bucket, days: u32, now: u64) -> u64 {
    let days = u64::from(days.max(1));
    match bucket {
        Bucket::Hour => Bucket::Hour
            .align(now)
            .saturating_sub((days * 24 - 1) * 3_600),
        Bucket::Day | Bucket::Week => {
            bucket.align(Bucket::Day.align(now).saturating_sub((days - 1) * 86_400))
        }
    }
}

fn round_to(x: f64, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (x * f).round() / f
}

fn usd(x: f64) -> f64 {
    round_to(x, 6)
}

fn secs(x: f64) -> f64 {
    round_to(x, 3)
}

fn mean(sum: f64, n: usize) -> f64 {
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Nearest-rank percentile of an ascending slice: the smallest value with at
/// least `p` percent of the values at or below it. 0 for no values.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Share of runs that came back `complete`; 0 for no runs.
fn success_rate<'a>(runs: impl IntoIterator<Item = &'a RunRecord>) -> f64 {
    let (mut n, mut ok) = (0usize, 0usize);
    for r in runs {
        n += 1;
        ok += usize::from(r.outcome == "complete");
    }
    round_to(mean(ok as f64, n), 4)
}

const OUTCOMES: [&str; 6] = [
    "complete",
    "partial",
    "truncated",
    "empty",
    "failed",
    "cancelled",
];

fn count(runs: &[&RunRecord], pred: impl Fn(&RunRecord) -> bool) -> usize {
    runs.iter().filter(|r| pred(r)).count()
}

/// The `/api/stats` response, minus `server`, which only the handler knows.
pub fn aggregate(all: &[RunRecord], now: u64, bucket: Bucket, days: u32) -> Value {
    let from = range_start(bucket, days, now);
    let runs: Vec<&RunRecord> = all
        .iter()
        .filter(|r| r.started_at >= from && r.started_at <= now)
        .collect();

    let mut totals = serde_json::Map::new();
    totals.insert("runs".into(), json!(runs.len()));
    for o in OUTCOMES {
        totals.insert(o.into(), json!(count(&runs, |r| r.outcome == o)));
    }
    for k in ["answer", "harvest"] {
        totals.insert(k.into(), json!(count(&runs, |r| r.kind == k)));
    }
    for s in ["ui", "mcp"] {
        totals.insert(s.into(), json!(count(&runs, |r| r.source == s)));
    }
    let cost: f64 = runs.iter().map(|r| r.cost_usd()).sum();
    let mut elapsed: Vec<f64> = runs.iter().map(|r| r.elapsed_s()).collect();
    elapsed.sort_by(f64::total_cmp);
    totals.insert(
        "records".into(),
        json!(runs.iter().map(|r| r.records).sum::<usize>()),
    );
    totals.insert(
        "pages".into(),
        json!(runs.iter().map(|r| r.pages).sum::<usize>()),
    );
    totals.insert("cost_usd".into(), json!(usd(cost)));
    totals.insert("avg_cost_usd".into(), json!(usd(mean(cost, runs.len()))));
    totals.insert(
        "avg_elapsed_s".into(),
        json!(secs(mean(elapsed.iter().sum(), elapsed.len()))),
    );
    totals.insert(
        "p50_elapsed_s".into(),
        json!(secs(percentile(&elapsed, 50.0))),
    );
    totals.insert(
        "p90_elapsed_s".into(),
        json!(secs(percentile(&elapsed, 90.0))),
    );
    totals.insert(
        "success_rate".into(),
        json!(success_rate(runs.iter().copied())),
    );

    let jev_usd: f64 = runs.iter().map(|r| r.jev_cost_usd).sum();
    let writer_usd: f64 = runs.iter().filter_map(|r| r.llm_cost_usd).sum();
    let planner_usd: f64 = runs.iter().filter_map(|r| r.planner_cost_usd).sum();
    let tok = |f: fn(&RunRecord) -> usize| runs.iter().map(|r| f(r)).sum::<usize>();

    // Contiguous buckets, zeros included, oldest first. Each run lands in
    // exactly one: its own aligned start, found by offset from `from`.
    let width = bucket.secs();
    let mut starts = Vec::new();
    let mut t = from;
    while t <= now {
        starts.push(t);
        t = bucket.align(t + width);
    }
    let mut per: Vec<Vec<&RunRecord>> = vec![Vec::new(); starts.len()];
    for r in &runs {
        let at = bucket.align(r.started_at);
        if let Ok(i) = starts.binary_search(&at) {
            per[i].push(r);
        }
    }
    let series: Vec<Value> = starts
        .iter()
        .zip(&per)
        .map(|(t, b)| {
            let mut p = serde_json::Map::new();
            p.insert("t".into(), json!(rfc3339(*t)));
            p.insert("runs".into(), json!(b.len()));
            for o in OUTCOMES {
                p.insert(o.into(), json!(count(b, |r| r.outcome == o)));
            }
            p.insert(
                "cost_usd".into(),
                json!(usd(b.iter().map(|r| r.cost_usd()).sum())),
            );
            p.insert(
                "avg_elapsed_s".into(),
                json!(secs(mean(b.iter().map(|r| r.elapsed_s()).sum(), b.len()))),
            );
            Value::Object(p)
        })
        .collect();

    let kind_summary = |k: &str| {
        let of: Vec<&RunRecord> = runs.iter().copied().filter(|r| r.kind == k).collect();
        json!({
            "runs": of.len(),
            "avg_elapsed_s": secs(mean(of.iter().map(|r| r.elapsed_s()).sum(), of.len())),
            "avg_cost_usd": usd(mean(of.iter().map(|r| r.cost_usd()).sum(), of.len())),
            "success_rate": success_rate(of.iter().copied()),
        })
    };

    let mut newest: Vec<&RunRecord> = runs.clone();
    // Stable, so runs started in the same second keep their log order, newest
    // (last logged) first after the reverse.
    newest.sort_by_key(|r| r.started_at);
    newest.reverse();
    let recent: Vec<Value> = newest
        .iter()
        .take(RECENT)
        .map(|r| {
            json!({
                "id": r.id,
                "started_at": rfc3339(r.started_at),
                "query": r.query,
                "kind": r.kind,
                "source": r.source,
                "outcome": r.outcome,
                "elapsed_s": secs(r.elapsed_s()),
                "cost_usd": usd(r.cost_usd()),
                "records": r.records,
                "pages": r.pages,
            })
        })
        .collect();

    json!({
        "generated_at": rfc3339(now),
        "range": {
            "from": rfc3339(from),
            "to": rfc3339(now),
            "bucket": bucket.as_str(),
            "days": days,
        },
        "totals": totals,
        "cost": {
            "jev_usd": usd(jev_usd),
            "writer_usd": usd(writer_usd),
            "planner_usd": usd(planner_usd),
            "total_usd": usd(jev_usd + writer_usd + planner_usd),
        },
        "tokens": {
            "jev_input": tok(|r| r.jev_input_tokens),
            "writer_prompt": tok(|r| r.llm_prompt_tokens),
            "writer_completion": tok(|r| r.llm_completion_tokens),
            "planner_prompt": tok(|r| r.planner_prompt_tokens),
            "planner_completion": tok(|r| r.planner_completion_tokens),
        },
        "series": series,
        "by_kind": {
            "answer": kind_summary("answer"),
            "harvest": kind_summary("harvest"),
        },
        "by_preset": {
            "quick": count(&runs, |r| r.preset == "quick"),
            "standard": count(&runs, |r| r.preset == "standard"),
            "thorough": count(&runs, |r| r.preset == "thorough"),
        },
        "recent": recent,
        "all_time": {
            "runs": all.len(),
            "cost_usd": usd(all.iter().map(|r| r.cost_usd()).sum()),
            "first_run_at": all.iter().map(|r| r.started_at).min().map(rfc3339),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-24T10:30:00Z, a Thursday.
    const NOW: u64 = 1_790_245_800;
    const H: u64 = 3_600;
    const D: u64 = 86_400;

    fn run(started_at: u64, outcome: &str, kind: &str, elapsed_ms: u64, jev: f64) -> RunRecord {
        RunRecord {
            id: format!("r{started_at}"),
            started_at,
            elapsed_ms,
            query: "q".into(),
            kind: kind.into(),
            source: "ui".into(),
            preset: "standard".into(),
            outcome: outcome.into(),
            jev_cost_usd: jev,
            ..Default::default()
        }
    }

    #[test]
    fn the_test_clock_is_what_it_says() {
        assert_eq!(rfc3339(NOW), "2026-09-24T10:30:00Z");
    }

    #[test]
    fn buckets_align_to_utc_hours_days_and_iso_weeks() {
        assert_eq!(rfc3339(Bucket::Hour.align(NOW)), "2026-09-24T10:00:00Z");
        assert_eq!(rfc3339(Bucket::Day.align(NOW)), "2026-09-24T00:00:00Z");
        // Thursday 24 September → Monday 21 September.
        assert_eq!(rfc3339(Bucket::Week.align(NOW)), "2026-09-21T00:00:00Z");
        // A Monday is its own week start; a Sunday belongs to the Monday before.
        let monday = Bucket::Week.align(NOW);
        assert_eq!(Bucket::Week.align(monday), monday);
        assert_eq!(Bucket::Week.align(monday - 1), monday - 7 * D);
        // The epoch (a Thursday) belongs to the week of Monday 1969-12-29,
        // which a u64 cannot hold; nothing logged is that old, but day 4 is.
        assert_eq!(rfc3339(Bucket::Week.align(5 * D)), "1970-01-05T00:00:00Z");
    }

    #[test]
    fn days_are_clamped_and_hourly_ranges_capped_at_a_week() {
        assert_eq!(clamp_days(Bucket::Day, 0), 1);
        assert_eq!(clamp_days(Bucket::Day, -4), 1);
        assert_eq!(clamp_days(Bucket::Day, 9_999), 365);
        assert_eq!(clamp_days(Bucket::Hour, 30), 7);
        assert_eq!(clamp_days(Bucket::Week, 30), 30);
    }

    #[test]
    fn the_series_is_contiguous_zero_filled_and_oldest_first() {
        let v = aggregate(&[], NOW, Bucket::Day, 30);
        let s = v["series"].as_array().unwrap();
        assert_eq!(s.len(), 30);
        assert_eq!(s[0]["t"], "2026-08-26T00:00:00Z");
        assert_eq!(s[29]["t"], "2026-09-24T00:00:00Z");
        assert!(s.iter().all(|p| p["runs"] == 0 && p["cost_usd"] == 0.0));
        assert_eq!(v["range"]["from"], "2026-08-26T00:00:00Z");
        assert_eq!(v["range"]["to"], "2026-09-24T10:30:00Z");
        assert_eq!(v["range"]["days"], 30);

        let v = aggregate(&[], NOW, Bucket::Hour, 1);
        let s = v["series"].as_array().unwrap();
        assert_eq!(s.len(), 24);
        assert_eq!(s[23]["t"], "2026-09-24T10:00:00Z");

        // Seven days back from Thursday 24 September starts on Friday the
        // 18th, in the week of Monday the 14th: two weeks.
        let v = aggregate(&[], NOW, Bucket::Week, 7);
        let s = v["series"].as_array().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0]["t"], "2026-09-14T00:00:00Z");
        let v = aggregate(&[], NOW, Bucket::Week, 30);
        let s = v["series"].as_array().unwrap();
        // 26 August (a Wednesday) → Monday 24 August; five Mondays to 21 September.
        assert_eq!(s[0]["t"], "2026-08-24T00:00:00Z");
        assert_eq!(s.len(), 5);
        assert!(
            s.windows(2)
                .all(|w| { w[0]["t"].as_str().unwrap() < w[1]["t"].as_str().unwrap() })
        );
    }

    #[test]
    fn runs_land_in_their_bucket_and_the_range_filters() {
        let runs = vec![
            run(NOW - 10, "complete", "answer", 10_000, 0.01),
            run(NOW - 2 * H, "failed", "unknown", 3_000, 0.002),
            run(NOW - D, "partial", "harvest", 60_000, 0.1),
            run(NOW - 40 * D, "complete", "answer", 1_000, 1.0), // outside 30 days
            run(NOW + 60, "complete", "answer", 1_000, 1.0),     // in the future
        ];
        let v = aggregate(&runs, NOW, Bucket::Day, 30);
        let t = &v["totals"];
        assert_eq!(t["runs"], 3);
        assert_eq!(t["complete"], 1);
        assert_eq!(t["failed"], 1);
        assert_eq!(t["partial"], 1);
        assert_eq!(t["answer"], 1);
        assert_eq!(t["harvest"], 1);
        assert_eq!(t["ui"], 3);
        assert_eq!(t["cost_usd"], 0.112);
        assert_eq!(t["success_rate"], 0.3333);
        let s = v["series"].as_array().unwrap();
        assert_eq!(s[29]["runs"], 2, "today: the complete and the failed run");
        assert_eq!(s[28]["runs"], 1, "yesterday: the partial run");
        assert_eq!(s[28]["partial"], 1);
        assert_eq!(s[28]["avg_elapsed_s"], 60.0);

        // Hourly over one day keeps yesterday's run out.
        let v = aggregate(&runs, NOW, Bucket::Hour, 1);
        assert_eq!(v["totals"]["runs"], 2);
        let s = v["series"].as_array().unwrap();
        assert_eq!(s[23]["complete"], 1);
        assert_eq!(s[21]["failed"], 1);

        // all_time ignores the range.
        assert_eq!(v["all_time"]["runs"], 5);
        assert_eq!(v["all_time"]["cost_usd"], 2.112);
        assert_eq!(v["all_time"]["first_run_at"], rfc3339(NOW - 40 * D));

        // recent: newest first, range only.
        let recent = v["recent"].as_array().unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0]["outcome"], "complete");
        assert_eq!(recent[0]["elapsed_s"], 10.0);
        assert_eq!(recent[1]["outcome"], "failed");
    }

    #[test]
    fn by_kind_and_by_preset_split_the_range() {
        let mut quick = run(NOW - 5, "empty", "answer", 4_000, 0.0);
        quick.preset = "quick".into();
        let runs = vec![
            run(NOW - 1, "complete", "answer", 2_000, 0.02),
            quick,
            run(NOW - 9, "complete", "harvest", 100_000, 0.5),
        ];
        let v = aggregate(&runs, NOW, Bucket::Day, 1);
        assert_eq!(v["by_kind"]["answer"]["runs"], 2);
        assert_eq!(v["by_kind"]["answer"]["success_rate"], 0.5);
        assert_eq!(v["by_kind"]["answer"]["avg_elapsed_s"], 3.0);
        assert_eq!(v["by_kind"]["answer"]["avg_cost_usd"], 0.01);
        assert_eq!(v["by_kind"]["harvest"]["success_rate"], 1.0);
        assert_eq!(v["by_preset"]["quick"], 1);
        assert_eq!(v["by_preset"]["standard"], 2);
        assert_eq!(v["by_preset"]["thorough"], 0);
    }

    #[test]
    fn empty_ranges_report_zeros_not_nans() {
        let v = aggregate(&[], NOW, Bucket::Day, 7);
        let t = &v["totals"];
        for k in [
            "avg_cost_usd",
            "avg_elapsed_s",
            "p50_elapsed_s",
            "p90_elapsed_s",
            "success_rate",
        ] {
            assert_eq!(t[k], 0.0, "{k}");
        }
        assert!(v["all_time"]["first_run_at"].is_null());
        assert_eq!(v["by_kind"]["harvest"]["success_rate"], 0.0);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let xs: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(percentile(&xs, 50.0), 5.0);
        assert_eq!(percentile(&xs, 90.0), 9.0);
        assert_eq!(percentile(&[7.0], 90.0), 7.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 50.0), 2.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 90.0), 3.0);
        assert_eq!(percentile(&[], 50.0), 0.0);

        let runs: Vec<RunRecord> = (1..=10)
            .map(|i| run(NOW - i, "complete", "answer", i * 1_000, 0.0))
            .collect();
        let v = aggregate(&runs, NOW, Bucket::Day, 1);
        assert_eq!(v["totals"]["p50_elapsed_s"], 5.0);
        assert_eq!(v["totals"]["p90_elapsed_s"], 9.0);
        assert_eq!(v["totals"]["avg_elapsed_s"], 5.5);
    }

    #[test]
    fn cost_counts_unreported_prices_as_nothing() {
        let mut r = run(NOW, "complete", "answer", 0, 0.01);
        r.llm_cost_usd = Some(0.02);
        assert!((r.cost_usd() - 0.03).abs() < 1e-12);
        r.planner_cost_usd = Some(0.5);
        assert!((r.cost_usd() - 0.53).abs() < 1e-12);
        let v = aggregate(&[r], NOW, Bucket::Day, 1);
        assert_eq!(v["cost"]["writer_usd"], 0.02);
        assert_eq!(v["cost"]["planner_usd"], 0.5);
        assert_eq!(v["cost"]["total_usd"], 0.53);
    }

    #[test]
    fn the_log_round_trips_and_skips_a_corrupt_line() {
        let dir = std::env::temp_dir().join(format!("ws-stats-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = StatsStore::open(&dir.join("nested"));
        store.append(run(NOW - 1, "complete", "answer", 1_000, 0.01));
        let mut failed = run(NOW, "failed", "unknown", 2_000, 0.0);
        failed.error = Some("boom".into());
        failed.llm_cost_usd = None;
        store.append(failed.clone());

        // A torn write and a blank line between good records.
        let path = dir.join("nested").join(LOG_FILE);
        let mut text = std::fs::read_to_string(&path).unwrap();
        text.push_str("{\"id\":\"torn\",\"started_\n\n");
        text.push_str(&serde_json::to_string(&run(NOW, "empty", "harvest", 5, 0.0)).unwrap());
        text.push('\n');
        std::fs::write(&path, text).unwrap();

        let loaded = StatsStore::open(&dir.join("nested"));
        let n = loaded.with_records(|r| {
            assert_eq!(r[1], failed);
            assert_eq!(r[2].outcome, "empty");
            r.len()
        });
        assert_eq!(n, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_old_line_missing_new_fields_still_loads() {
        let recs = parse_log(
            r#"{"id":"a","started_at":5,"outcome":"complete"}"#,
            Path::new("x"),
        );
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].llm_cost_usd, None);
    }

    #[test]
    fn an_unwritable_log_never_fails_the_run() {
        // A file where the directory should be: create_dir_all fails.
        let blocker = std::env::temp_dir().join(format!("ws-stats-block-{}", std::process::id()));
        std::fs::write(&blocker, "x").unwrap();
        let store = StatsStore::open(&blocker.join("sub"));
        store.append(run(NOW, "complete", "answer", 1, 0.0));
        assert_eq!(store.with_records(|r| r.len()), 1);
        let _ = std::fs::remove_file(&blocker);
    }

    #[test]
    fn the_record_serialises_with_the_contract_field_names() {
        let v = serde_json::to_value(run(NOW, "complete", "answer", 1, 0.0)).unwrap();
        for k in [
            "id",
            "started_at",
            "elapsed_ms",
            "query",
            "kind",
            "source",
            "preset",
            "outcome",
            "records",
            "pages",
            "rounds",
            "jev_requests",
            "jev_input_tokens",
            "jev_cost_usd",
            "llm_requests",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_cost_usd",
            "planner_requests",
            "planner_prompt_tokens",
            "planner_completion_tokens",
            "planner_cost_usd",
            "error",
        ] {
            assert!(v.get(k).is_some(), "missing {k}");
        }
        assert_eq!(v.as_object().unwrap().len(), 23);
        assert!(v["llm_cost_usd"].is_null());
    }
}
