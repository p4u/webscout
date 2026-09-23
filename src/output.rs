//! Rendering a report.
//!
//! The formats exist for different readers, and the split matters because this
//! tool is meant to be driven by other agents as well as by people. Machine
//! formats go to stdout unadorned — no colour, no progress, nothing that has to be
//! stripped before parsing. Logs go to stderr, always, so a caller can redirect
//! stdout into a file or a pipe and get exactly the payload and nothing else.

use anyhow::Result;
use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::types::{MissionKind, Outcome, ScoutReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "lower")]
pub enum Format {
    /// One JSON document with everything, including per-record grounding scores.
    Json,
    /// One JSON object per line: records for a harvest, evidence for an answer.
    /// Streams into `jq` and line-oriented tooling without loading the whole thing.
    Jsonl,
    /// Markdown suitable for a document or a pull request.
    Markdown,
    /// Human-readable text for a terminal.
    Terminal,
    /// Records only, as a spreadsheet.
    Csv,
}

pub fn render(report: &ScoutReport, format: Format) -> Result<String> {
    match format {
        Format::Json => Ok(serde_json::to_string_pretty(report)? + "\n"),
        Format::Jsonl => render_jsonl(report),
        Format::Markdown => Ok(render_markdown(report)),
        Format::Terminal => Ok(render_terminal(report)),
        Format::Csv => render_csv(report),
    }
}

fn render_jsonl(report: &ScoutReport) -> Result<String> {
    let mut out = String::new();
    if report.mission.kind == MissionKind::Harvest {
        for r in &report.records {
            out.push_str(&serde_json::to_string(r)?);
            out.push('\n');
        }
    } else {
        if let Some(answer) = &report.answer {
            out.push_str(&serde_json::to_string(
                &serde_json::json!({"type": "answer", "text": answer}),
            )?);
            out.push('\n');
        }
        for p in &report.evidence {
            out.push_str(&serde_json::to_string(p)?);
            out.push('\n');
        }
    }
    // A trailing summary line so a streaming consumer learns the outcome without
    // having to count what it received.
    out.push_str(&serde_json::to_string(&serde_json::json!({
        "type": "summary",
        "outcome": report.outcome.as_str(),
        "records": report.records.len(),
        "sources": report.sources.len(),
        "stats": report.stats,
    }))?);
    out.push('\n');
    Ok(out)
}

/// Column order for tabular output: the mission's own field order, then anything
/// else that turned up, so the columns match what was asked for.
fn columns(report: &ScoutReport) -> Vec<String> {
    let mut cols: Vec<String> = report.mission.fields.clone();
    let mut extra = BTreeSet::new();
    for r in &report.records {
        for k in r.fields.keys() {
            if !cols.contains(k) {
                extra.insert(k.clone());
            }
        }
    }
    cols.extend(extra);
    cols
}

/// Fields whose provenance source URL differs from the record's `source_url` in
/// at least one record.  These get extra `{field}_source_url` columns in CSV.
fn provenance_extra_cols(report: &ScoutReport, cols: &[String]) -> Vec<String> {
    let mut extra: Vec<String> = Vec::new();
    for col in cols {
        let has_different = report.records.iter().any(|r| {
            r.provenance
                .get(col)
                .map(|fs| fs.source_url != r.source_url)
                .unwrap_or(false)
        });
        if has_different {
            extra.push(format!("{col}_source_url"));
        }
    }
    extra
}

fn render_csv(report: &ScoutReport) -> Result<String> {
    let cols = columns(report);
    let extra_prov = provenance_extra_cols(report, &cols);

    let mut wtr = csv::Writer::from_writer(vec![]);

    let mut header: Vec<String> = cols.clone();
    header.push("source_url".into());
    header.push("grounding".into());
    header.push("constraint_support".into());
    for ep in &extra_prov {
        header.push(ep.clone());
    }
    wtr.write_record(&header)?;

    for r in &report.records {
        let mut row: Vec<String> = cols.iter().map(|c| r.get(c).to_string()).collect();
        row.push(r.source_url.clone());
        row.push(format!("{:.2}", r.grounding));
        row.push(format!("{:.2}", r.constraint_support));
        for ep in &extra_prov {
            // Strip the trailing "_source_url" suffix to recover the field name.
            let field = ep.trim_end_matches("_source_url");
            let url = r
                .provenance
                .get(field)
                .map(|fs| fs.source_url.as_str())
                .unwrap_or("");
            row.push(url.to_string());
        }
        wtr.write_record(&row)?;
    }
    Ok(String::from_utf8(wtr.into_inner()?)?)
}

fn render_markdown(report: &ScoutReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# {}\n", report.query);
    let _ = writeln!(
        s,
        "**{}** — {}\n",
        report.outcome.as_str().to_uppercase(),
        gloss(report)
    );

    if report.mission.kind == MissionKind::Harvest {
        if !report.records.is_empty() {
            let cols = columns(report);
            let _ = writeln!(s, "| # | {} | source | grounding |", cols.join(" | "));
            // The delimiter row must have exactly as many cells as the header —
            // `#`, one per field, `source`, `grounding` — or GFM does not treat
            // the block as a table at all and every renderer prints the raw
            // pipes as a paragraph. Measured 2026-09-21 with marked 15: the
            // previous line emitted `|---|---|---||---|---|` for two fields, a
            // 6-cell row with a doubled pipe against a 5-cell header, and no
            // table appeared in the UI.
            let _ = writeln!(s, "|{}", "---|".repeat(cols.len() + 3));
            for (i, r) in report.records.iter().enumerate() {
                let values: Vec<String> = cols
                    .iter()
                    .map(|c| {
                        let val = escape_pipes(r.get(c));
                        // Show per-field source only when it differs from the row source_url.
                        let prov_note = r.provenance.get(c).and_then(|fs| {
                            if fs.source_url != r.source_url {
                                Some(format!(" ← {}", fs.source_url))
                            } else {
                                None
                            }
                        });
                        match prov_note {
                            Some(note) => format!("{val}{}", escape_pipes(&note)),
                            None => val,
                        }
                    })
                    .collect();
                let _ = writeln!(
                    s,
                    "| {} | {} | [link]({}) | {:.2} |",
                    i + 1,
                    values.join(" | "),
                    r.source_url,
                    r.grounding
                );
            }
            let _ = writeln!(s);
        }
    } else {
        if let Some(answer) = &report.answer {
            let _ = writeln!(s, "{answer}\n");
        }
        if !report.evidence.is_empty() {
            let _ = writeln!(s, "## Sources\n");
            for (i, p) in report.evidence.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "{}. [{}]({}) — supports {:.2}",
                    i + 1,
                    p.title,
                    p.url,
                    p.supports
                );
            }
            let _ = writeln!(s);
        }
    }

    if !report.quarantined_sources.is_empty() {
        let _ = writeln!(s, "## Quarantined\n");
        let _ = writeln!(
            s,
            "These pages contained text addressed to an AI reader. They were kept away \
             from the generative model and contributed nothing.\n"
        );
        for u in &report.quarantined_sources {
            let _ = writeln!(s, "- <{u}>");
        }
        let _ = writeln!(s);
    }

    if !report.notes.is_empty() {
        let _ = writeln!(s, "## Notes\n");
        for n in &report.notes {
            let _ = writeln!(s, "- {n}");
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "---\n");
    let _ = writeln!(s, "{}", stats_line(report));
    s
}

fn render_terminal(report: &ScoutReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\n{}", "=".repeat(78));
    let _ = writeln!(s, "{}", report.query);
    let _ = writeln!(s, "{}", "=".repeat(78));
    let _ = writeln!(
        s,
        "{}: {}\n",
        report.outcome.as_str().to_uppercase(),
        gloss(report)
    );

    if report.mission.kind == MissionKind::Harvest {
        let cols = columns(report);
        for (i, r) in report.records.iter().enumerate() {
            let primary: Vec<String> = cols
                .iter()
                .map(|c| {
                    let v = r.get(c);
                    if v.is_empty() {
                        return String::new();
                    }
                    // Show per-field source only when it differs from the row source_url.
                    let prov_note = r.provenance.get(c).and_then(|fs| {
                        if fs.source_url != r.source_url {
                            Some(format!(" ← {}", fs.source_url))
                        } else {
                            None
                        }
                    });
                    match prov_note {
                        Some(note) => format!("{}: {}{}", c, v, note),
                        None => format!("{}: {}", c, v),
                    }
                })
                .filter(|p| !p.is_empty())
                .collect();
            let _ = writeln!(s, "{:>4}. {}", i + 1, primary.join("  |  "));
            let _ = writeln!(s, "      {} (grounding {:.2})", r.source_url, r.grounding);
        }
        if !report.records.is_empty() {
            let _ = writeln!(
                s,
                "\n{} record(s) from {} source(s).",
                report.records.len(),
                report.sources.len()
            );
        }
    } else {
        if let Some(answer) = &report.answer {
            let _ = writeln!(s, "{answer}\n");
        }
        if !report.evidence.is_empty() {
            let _ = writeln!(s, "Sources:");
            for (i, p) in report.evidence.iter().enumerate() {
                let _ = writeln!(s, "  [{}] {}", i + 1, p.url);
            }
        }
    }

    if !report.quarantined_sources.is_empty() {
        let _ = writeln!(
            s,
            "\n!! {} page(s) tried to address the model and were quarantined:",
            report.quarantined_sources.len()
        );
        for u in &report.quarantined_sources {
            let _ = writeln!(s, "   {u}");
        }
    }

    if !report.notes.is_empty() {
        let _ = writeln!(s, "\nNotes:");
        for n in &report.notes {
            let _ = writeln!(s, "  - {n}");
        }
    }

    let _ = writeln!(s, "\n{}", "-".repeat(78));
    let _ = writeln!(s, "{}", stats_line(report));
    s
}

fn gloss(report: &ScoutReport) -> String {
    let harvest = report.mission.kind == MissionKind::Harvest;
    match report.outcome {
        Outcome::Complete if harvest => {
            format!("collected {} verified records", report.records.len())
        }
        Outcome::Complete => "the evidence answers the question".into(),
        Outcome::Partial if harvest => format!(
            "found {} of the {} requested; the reachable web ran out first",
            report.records.len(),
            report.mission.target_count.unwrap_or(0)
        ),
        Outcome::Partial => "part of the question is answered, with gaps remaining".into(),
        Outcome::Truncated => format!(
            "stopped at the round ceiling with {} records and more still appearing",
            report.records.len()
        ),
        Outcome::Empty if harvest => "nothing verifiable was found".into(),
        Outcome::Empty => {
            "nothing found. Treat this as absence of evidence, not evidence of absence".into()
        }
    }
}

fn stats_line(report: &ScoutReport) -> String {
    let st = &report.stats;
    let mut s = format!(
        "{} round(s), {} queries, {} pages, {} chunks in {:.1}s. \
         Jev: {} requests, {} tokens, ${:.4}. LLM: {} requests, {} prompt + {} completion tokens.",
        st.rounds,
        st.queries_issued,
        st.pages_fetched,
        st.chunks_examined,
        st.elapsed_secs,
        st.jev_requests,
        st.jev_input_tokens,
        st.jev_cost_usd,
        st.llm_requests,
        st.llm_prompt_tokens,
        st.llm_completion_tokens,
    );
    if st.rejected_ungrounded > 0 || st.quarantined > 0 {
        let _ = write!(
            s,
            " Rejected {} ungrounded record(s); quarantined {} passage(s).",
            st.rejected_ungrounded, st.quarantined
        );
    }
    // The answer-path guard: reported only when it fired, like the two above.
    if st.unsupported_claims > 0 {
        let _ = write!(
            s,
            " {} of {} answer claim(s) unsupported by the evidence and marked in the text.",
            st.unsupported_claims, st.claims_checked
        );
    }
    // Show enrichment and link-following stats when non-zero (Package B populates these).
    if st.entities_discovered > 0 || st.entities_enriched > 0 || st.links_followed > 0 {
        let _ = write!(s, " Entities: {} discovered", st.entities_discovered);
        if st.entities_enriched > 0 {
            let _ = write!(s, ", {} enriched", st.entities_enriched);
        }
        if st.links_followed > 0 {
            let _ = write!(s, ". Links followed: {}", st.links_followed);
        }
        if st.enrich_searches > 0 {
            let _ = write!(s, ". Enrich searches: {}", st.enrich_searches);
        }
        s.push('.');
    }
    s
}

fn escape_pipes(s: &str) -> String {
    s.replace('|', r"\|").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use std::collections::BTreeMap;

    fn sample() -> ScoutReport {
        let mut fields = BTreeMap::new();
        fields.insert("name".to_string(), "Coop Alpha".to_string());
        fields.insert("email".to_string(), "info@alpha.coop".to_string());

        ScoutReport {
            query: "cooperatives with emails".into(),
            mission: Mission {
                query: "cooperatives with emails".into(),
                kind: MissionKind::Harvest,
                target_count: Some(2),
                fields: vec!["name".into(), "email".into()],
                determination_fields: vec![],
                topic: "cooperatives".into(),
                constraints: vec![],
                constraint_glosses: vec![],
                unlistable_constraints: vec![],
                listing_core: String::new(),
                simple: false,
                entity_field: "name".into(),
                anchors: vec![],
                scope: String::new(),
                entity_type: String::new(),
                time_sensitive: false,
            },
            outcome: Outcome::Partial,
            records: vec![Record {
                fields,
                source_url: "https://example.org/list".into(),
                source_title: Some("List".into()),
                grounding: 0.93,
                provenance: BTreeMap::new(),
                constraint_support: 1.0,
                constraint_status: Vec::new(),
                entity_binding: EntityBinding::Unresolved,
            }],
            answer: None,
            evidence: vec![],
            sources: vec!["https://example.org/list".into()],
            quarantined_sources: vec![],
            notes: vec![],
            stats: Stats::default(),
        }
    }

    /// Build a sample report where the email field was sourced from a different page.
    fn sample_with_provenance() -> ScoutReport {
        let mut r = sample();
        r.records[0].provenance.insert(
            "email".to_string(),
            FieldSource {
                source_url: "https://example.org/contact".into(),
                grounding: 0.88,
            },
        );
        r
    }

    #[test]
    fn csv_has_header_and_row() {
        let out = render(&sample(), Format::Csv).unwrap();
        let mut lines = out.lines();
        assert_eq!(
            lines.next().unwrap(),
            "name,email,source_url,grounding,constraint_support"
        );
        assert!(
            lines
                .next()
                .unwrap()
                .starts_with("Coop Alpha,info@alpha.coop,")
        );
    }

    #[test]
    fn csv_adds_provenance_source_url_column() {
        let out = render(&sample_with_provenance(), Format::Csv).unwrap();
        let mut lines = out.lines();
        let header = lines.next().unwrap();
        assert!(
            header.contains("email_source_url"),
            "header should contain email_source_url: {header}"
        );
        let row = lines.next().unwrap();
        assert!(
            row.contains("https://example.org/contact"),
            "row should contain the provenance URL: {row}"
        );
    }

    #[test]
    fn csv_no_extra_col_when_provenance_matches_source_url() {
        let mut r = sample();
        // Provenance URL == source_url → no extra column.
        r.records[0].provenance.insert(
            "email".to_string(),
            FieldSource {
                source_url: "https://example.org/list".into(),
                grounding: 0.93,
            },
        );
        let out = render(&r, Format::Csv).unwrap();
        assert!(
            !out.contains("email_source_url"),
            "should not add column when provenance matches source_url: {out}"
        );
    }

    #[test]
    fn jsonl_emits_one_record_then_a_summary() {
        let out = render(&sample(), Format::Jsonl).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2, "one record plus one summary");
        let summary: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(summary["type"], "summary");
        assert_eq!(summary["outcome"], "partial");
    }

    #[test]
    fn json_round_trips() {
        let out = render(&sample(), Format::Json).unwrap();
        let back: ScoutReport = serde_json::from_str(&out).unwrap();
        assert_eq!(back.records.len(), 1);
        assert_eq!(back.outcome, Outcome::Partial);
    }

    #[test]
    fn markdown_columns_follow_the_mission_field_order() {
        // Records are BTreeMaps, so "email" sorts before "name"; the table must
        // still present the columns the user asked for, in that order.
        let out = render(&sample(), Format::Markdown).unwrap();
        assert!(
            out.contains("| # | name | email | source | grounding |"),
            "got:\n{out}"
        );
    }

    /// GFM only recognises a table when the delimiter row has exactly as many
    /// cells as the header. An earlier version emitted one cell too many and a
    /// doubled pipe, so marked rendered every harvest table as a paragraph of
    /// raw pipes and nobody noticed, because the old test only checked the
    /// header. Count the cells of every row instead.
    #[test]
    fn markdown_table_rows_all_have_the_same_cell_count() {
        let out = render(&sample(), Format::Markdown).unwrap();
        let rows: Vec<&str> = out
            .lines()
            .filter(|l| l.trim_start().starts_with('|'))
            .collect();
        assert!(rows.len() >= 3, "expected a table, got:\n{out}");
        let cells = |row: &str| row.trim().trim_matches('|').split('|').count();
        let header = cells(rows[0]);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                cells(row),
                header,
                "row {i} has {} cells against a {header}-cell header:\n{row}",
                cells(row)
            );
        }
        assert!(
            !out.contains("||"),
            "a doubled pipe breaks the table:\n{out}"
        );
    }

    #[test]
    fn markdown_escapes_pipes_in_values() {
        let mut r = sample();
        r.records[0].fields.insert("name".into(), "A | B".into());
        let out = render(&r, Format::Markdown).unwrap();
        assert!(out.contains(r"A \| B"));
    }

    #[test]
    fn terminal_shows_per_field_provenance_when_different() {
        let out = render(&sample_with_provenance(), Format::Terminal).unwrap();
        assert!(
            out.contains("← https://example.org/contact"),
            "terminal should show per-field provenance: {out}"
        );
    }

    #[test]
    fn stats_line_shows_enrichment_when_nonzero() {
        let mut r = sample();
        r.stats.entities_discovered = 10;
        r.stats.entities_enriched = 5;
        r.stats.links_followed = 3;
        let out = render(&r, Format::Terminal).unwrap();
        assert!(out.contains("10 discovered"), "got: {out}");
        assert!(out.contains("5 enriched"), "got: {out}");
        assert!(out.contains("Links followed: 3"), "got: {out}");
    }

    /// An answer report whose second sentence was marked by the per-claim check.
    fn sample_marked_answer() -> ScoutReport {
        let mut r = sample();
        r.mission.kind = MissionKind::Answer;
        r.mission.fields.clear();
        r.records.clear();
        r.outcome = Outcome::Partial;
        r.answer = Some(
            "Three seasons have been released [2]. The fourth season will premiere in \
             March 2027 on HBO Max. [unsupported]"
                .into(),
        );
        r.stats.claims_checked = 2;
        r.stats.unsupported_claims = 1;
        r
    }

    #[test]
    fn terminal_shows_the_unsupported_marker_in_the_answer_text() {
        let out = render(&sample_marked_answer(), Format::Terminal).unwrap();
        assert!(
            out.contains("March 2027 on HBO Max. [unsupported]"),
            "the marker is part of the answer and must render: {out}"
        );
    }

    #[test]
    fn markdown_shows_the_unsupported_marker_in_the_answer_text() {
        let out = render(&sample_marked_answer(), Format::Markdown).unwrap();
        assert!(out.contains("[unsupported]"), "got: {out}");
    }

    #[test]
    fn stats_line_reports_unsupported_claims_when_nonzero() {
        let out = render(&sample_marked_answer(), Format::Terminal).unwrap();
        assert!(
            out.contains("1 of 2 answer claim(s) unsupported"),
            "got: {out}"
        );
    }

    #[test]
    fn json_carries_the_claim_stats_and_the_marked_answer() {
        let out = render(&sample_marked_answer(), Format::Json).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["stats"]["claims_checked"], 2);
        assert_eq!(v["stats"]["unsupported_claims"], 1);
        assert!(
            v["answer"].as_str().unwrap().contains("[unsupported]"),
            "the marked answer must survive JSON rendering"
        );
        // And it round-trips, so a stored report keeps the counts.
        let back: ScoutReport = serde_json::from_str(&out).unwrap();
        assert_eq!(back.stats.unsupported_claims, 1);
        assert_eq!(back.stats.claims_checked, 2);
    }

    /// A report written before these fields existed must still load, reading as
    /// "not checked" rather than as "checked and clean".
    #[test]
    fn stats_claim_fields_default_to_zero_when_absent() {
        let mut v = serde_json::to_value(Stats::default()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("claims_checked");
        obj.remove("unsupported_claims");
        let st: Stats = serde_json::from_value(v).expect("legacy stats parse");
        assert_eq!(st.claims_checked, 0);
        assert_eq!(st.unsupported_claims, 0);
    }

    #[test]
    fn stats_line_silent_about_claims_when_none_were_unsupported() {
        let mut r = sample_marked_answer();
        r.stats.unsupported_claims = 0;
        let out = render(&r, Format::Terminal).unwrap();
        assert!(!out.contains("unsupported by the evidence"), "got: {out}");
    }

    #[test]
    fn stats_line_silent_when_enrichment_zero() {
        let out = render(&sample(), Format::Terminal).unwrap();
        assert!(!out.contains("discovered"), "got: {out}");
        assert!(!out.contains("enriched"), "got: {out}");
    }
}
