//! Recall-tuned value extraction for regex-able contact fields.
//!
//! Scout uses this module to pre-screen passages before spending Jev questions
//! on them, and to supply candidate values for the `choice`-style enrichment
//! path. Adding it now (Package A2) lets Package B wire it in without touching
//! this file.

// Package B now consumes this module; the crate-level allow is no longer
// necessary. Individual items that stay unused stay flagged.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Compiled regexes (lazy, thread-safe)
// ---------------------------------------------------------------------------

fn email_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Standard email pattern; obfuscation is normalised before this runs.
        Regex::new(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}").unwrap()
    })
}

fn url_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Matches explicit-scheme URLs and bare www. forms.
        Regex::new(r#"(?:https?://|www\.)[^\s<>"'(){}\[\],;]{3,}"#).unwrap()
    })
}

fn phone_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // 7+ digit sequences separated by spaces, dots, or dashes, with an
        // optional leading +. Digit count is verified after the match.
        Regex::new(r"\+?[0-9][0-9 .\-]{5,}[0-9]").unwrap()
    })
}

// ---------------------------------------------------------------------------
// Obfuscation normalisation
// ---------------------------------------------------------------------------

/// Replace common email-obfuscation patterns with their canonical characters.
///
/// Operates on text that has already been lowercased, so comparisons are
/// ASCII-safe. Patterns handled:
/// - `(at)`, `[at]`, ` at ` → `@`
/// - ` arroba ` → `@` (Spanish for @)
/// - `(dot)`, `[dot]` → `.`
fn normalize_obfuscation(s: &str) -> String {
    // Lowercase first so callers do not need to pre-process.
    let s = s.to_lowercase();
    let s = s.replace(" arroba ", "@");
    let s = s.replace("(at)", "@");
    let s = s.replace("[at]", "@");
    let s = s.replace("(dot)", ".");
    let s = s.replace("[dot]", ".");
    // " at " only when it sits between a local part and a dotted domain that
    // is not a web address: "info at ayto-x . es" is an obfuscated email,
    // "available at www.x.es" and "meet at the office" are not. A plain global
    // replace turned the former phrase into the candidate "available@www.x.es".
    at_word_re()
        .replace_all(&s, |c: &regex::Captures| {
            let domain = &c[2];
            if domain.starts_with("www.") || domain.starts_with("http") {
                c[0].to_string()
            } else {
                format!("{}@{}", &c[1], domain.replace(' ', ""))
            }
        })
        .into_owned()
}

fn at_word_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"([a-z0-9._%+\-]+) at ((?:[a-z0-9\-]+ ?\. ?)+[a-z]{2,})").unwrap()
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Broad category of a structured contact or entity field value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Email,
    Url,
    Phone,
}

/// Map a field name to a broad [`Kind`] using case-insensitive substring
/// matching across English, Spanish, Catalan, French, German, Italian, and
/// Portuguese field-name conventions.
///
/// Returns `None` when no recognisable pattern is found.
pub fn kind_for_field(field: &str) -> Option<Kind> {
    let lower = field.to_ascii_lowercase();

    // Email keywords are checked before URL so a field called `contact_email`
    // resolves to Email rather than being shadowed by a hypothetical `url` in
    // it (there is none, but ordering is explicit).
    // "mailing_address" is a postal address, not an email, so the bare "mail"
    // match is fenced off from it.
    if lower.contains("email")
        || lower.contains("e-mail")
        || lower.contains("correo")
        || (lower.contains("mail") && !lower.contains("mailing"))
    {
        return Some(Kind::Email);
    }

    // URL / website keywords.
    if lower.contains("website")
        || lower.contains("sitio")
        || lower.contains("site")
        || lower.contains("web")
        || lower.contains("url")
    {
        return Some(Kind::Url);
    }

    // Phone keywords — `tel` is last so `telephone` and `telefono` already
    // matched; the short form catches bare `tel` and `téléphone`.
    if lower.contains("phone")
        || lower.contains("telephone")
        || lower.contains("teléfono")
        || lower.contains("telefono")
        || lower.contains("telefon")
        || lower.contains("tel")
    {
        return Some(Kind::Phone);
    }

    None
}

/// Find all values of the given Kind in `text`.
///
/// Returns deduplicated, normalised values in order of first occurrence.
/// Trailing punctuation is trimmed. Email obfuscations are normalised to
/// their canonical `user@domain.tld` form before matching, so only the
/// normalised form is returned.
pub fn find(kind: Kind, text: &str) -> Vec<String> {
    match kind {
        Kind::Email => find_emails(text),
        Kind::Url => find_urls(text),
        Kind::Phone => find_phones(text),
    }
}

/// Case-insensitive, whitespace-normalised substring check.
///
/// Returns `true` if `value` appears in `text` when both are lowercased and
/// whitespace-collapsed, or if the obfuscated form of `value` appears in
/// `text` (handles emails written as `info (at) domain (dot) tld`).
pub fn appears_in(value: &str, text: &str) -> bool {
    let v_norm: String = value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        // `find_urls` unescapes `\/`, so the value side is already clean;
        // the text side may still carry the JSON escape and would fail the
        // literal comparison for no reason.
        .replace("\\/", "/");
    let t_norm: String = text
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("\\/", "/");

    if t_norm.contains(&v_norm) {
        return true;
    }

    // Also check with obfuscation normalisation.  This lets the caller pass a
    // canonical email address and still match pages that wrote it with (at)
    // or [dot] substitutions.
    let v_deobf = normalize_obfuscation(&v_norm);
    let t_deobf = normalize_obfuscation(&t_norm);
    t_deobf.contains(&v_deobf)
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

fn find_emails(text: &str) -> Vec<String> {
    // Normalise obfuscation first so the standard regex catches all forms.
    let normalised = normalize_obfuscation(text);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in email_re().find_iter(&normalised) {
        let s = trim_punct(m.as_str());
        if s.is_empty() || !is_plausible_email(&s) {
            continue;
        }
        if seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

/// Reject junk email candidates the regex is happy to accept:
/// - Cloudflare's "[email protected]" hover artefact, which lands in the
///   normalised text as `email@…` and any address whose local part contains
///   the word `protected`.
/// - Placeholder domains (`example.com`, `email.com`, etc.) that appear in
///   templates and boilerplate.
///
/// The rule is deliberately narrow: real addresses with `email` inside a
/// longer local part (`emailing@`, `email.support@`) still pass. Only the
/// exact local part `email` is dropped.
pub(crate) fn is_plausible_email(candidate: &str) -> bool {
    let Some((local, domain)) = candidate.rsplit_once('@') else {
        return false;
    };
    let local_l = local.to_lowercase();
    if local_l == "email" {
        return false;
    }
    if local_l.contains("protected") {
        return false;
    }
    let domain_l = domain.to_lowercase();
    !matches!(
        domain_l.as_str(),
        "example.com" | "example.org" | "domain.com" | "email.com" | "test.com"
    )
}

fn find_urls(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in url_re().find_iter(text) {
        let s = trim_punct(m.as_str());
        // Bare www. forms get the scheme prepended for usability.
        let s = if s.starts_with("www.") {
            format!("https://{s}")
        } else {
            s
        };
        // Pages that embed JSON (API payloads, script data) escape every
        // separator: `https:\/\/example.com\/`. The regex allows the
        // backslash through, so unescape here or the stored value is not a
        // usable URL and the escaped and plain spellings of one address
        // dedupe as two (measured 2026-09-21, q82: `official_website`
        // values full of `\/`).
        let s = s.replace("\\/", "/");
        if !s.is_empty() && seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

fn find_phones(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in phone_re().find_iter(text) {
        let s = m.as_str().trim().to_string();
        // The regex allows spaces/dots/dashes in the middle; require at least
        // 7 actual digits to exclude things like IP fragments or year ranges.
        let digit_count = s.chars().filter(|c| c.is_ascii_digit()).count();
        if digit_count < 7 {
            continue;
        }
        let s = trim_punct(&s);
        if !s.is_empty() && seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

/// Trim trailing punctuation that belongs to surrounding prose, not the value.
fn trim_punct(s: &str) -> String {
    s.trim_end_matches(['.', ',', ':', ';', '!', '?', ')', ']'])
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- kind_for_field -------------------------------------------------------

    #[test]
    fn recognises_email_field_names() {
        assert_eq!(kind_for_field("email"), Some(Kind::Email));
        assert_eq!(kind_for_field("EMAIL"), Some(Kind::Email));
        assert_eq!(kind_for_field("e-mail"), Some(Kind::Email));
        assert_eq!(kind_for_field("contact_email"), Some(Kind::Email));
        assert_eq!(kind_for_field("email_address"), Some(Kind::Email));
        assert_eq!(kind_for_field("correo"), Some(Kind::Email));
        assert_eq!(kind_for_field("correo_electronico"), Some(Kind::Email));
    }

    #[test]
    fn recognises_url_field_names() {
        assert_eq!(kind_for_field("website"), Some(Kind::Url));
        assert_eq!(kind_for_field("url"), Some(Kind::Url));
        assert_eq!(kind_for_field("web"), Some(Kind::Url));
        assert_eq!(kind_for_field("sitio_web"), Some(Kind::Url));
        assert_eq!(kind_for_field("site"), Some(Kind::Url));
    }

    #[test]
    fn recognises_phone_field_names() {
        assert_eq!(kind_for_field("phone"), Some(Kind::Phone));
        assert_eq!(kind_for_field("telephone"), Some(Kind::Phone));
        assert_eq!(kind_for_field("tel"), Some(Kind::Phone));
        assert_eq!(kind_for_field("telefono"), Some(Kind::Phone));
        assert_eq!(kind_for_field("teléfono"), Some(Kind::Phone));
        assert_eq!(kind_for_field("telefon"), Some(Kind::Phone));
    }

    #[test]
    fn returns_none_for_unknown_field() {
        assert_eq!(kind_for_field("name"), None);
        assert_eq!(kind_for_field("address"), None);
        assert_eq!(kind_for_field("population"), None);
    }

    // -- find(Kind::Email) ---------------------------------------------------

    #[test]
    fn finds_standard_email() {
        let emails = find(Kind::Email, "Contact us at info@ayto.es for details.");
        assert_eq!(emails, vec!["info@ayto.es"]);
    }

    #[test]
    fn normalises_at_bracket_obfuscation() {
        let emails = find(Kind::Email, "Write to info[at]ayto[dot]es today.");
        assert_eq!(emails, vec!["info@ayto.es"]);
    }

    #[test]
    fn normalises_at_paren_obfuscation() {
        let emails = find(Kind::Email, "Send to info(at)example.es for help.");
        assert_eq!(emails, vec!["info@example.es"]);
    }

    #[test]
    fn normalises_spaced_at_obfuscation() {
        let emails = find(Kind::Email, "Reach us: hello at domain.org");
        assert_eq!(emails, vec!["hello@domain.org"]);
    }

    #[test]
    fn normalises_arroba_obfuscation() {
        let emails = find(Kind::Email, "Escríbenos: info arroba ayuntamiento.es");
        assert_eq!(emails, vec!["info@ayuntamiento.es"]);
    }

    #[test]
    fn deduplicates_emails() {
        let emails = find(Kind::Email, "Email info@ayto.es or info@ayto.es again.");
        assert_eq!(emails.len(), 1);
    }

    #[test]
    fn finds_multiple_emails() {
        let emails = find(
            Kind::Email,
            "Contact a@x.com and b@y.org for more information.",
        );
        assert_eq!(emails, vec!["a@x.com", "b@y.org"]);
    }

    // -- find(Kind::Url) -----------------------------------------------------

    #[test]
    fn finds_https_url() {
        let urls = find(Kind::Url, "Visit https://www.example.com/path for details.");
        assert_eq!(urls, vec!["https://www.example.com/path"]);
    }

    #[test]
    fn finds_bare_www_url_and_prepends_scheme() {
        let urls = find(Kind::Url, "See www.example.com for more.");
        assert_eq!(urls, vec!["https://www.example.com"]);
    }

    #[test]
    fn deduplicates_urls() {
        let urls = find(
            Kind::Url,
            "Visit https://x.com and then https://x.com again.",
        );
        assert_eq!(urls.len(), 1);
    }

    /// JSON-embedded pages escape every separator; the stored value must be
    /// a usable URL and must fold with the plain spelling (measured
    /// 2026-09-21, q82: `official_website` values full of `\/`).
    #[test]
    fn unescapes_json_separated_urls_and_folds_the_plain_spelling() {
        let urls = find(Kind::Url, r#"{"site": "https:\/\/www.aeropolis.es\/"}"#);
        assert_eq!(urls, vec!["https://www.aeropolis.es/"]);
        // The escaped and plain spellings of one address are one candidate.
        let urls = find(Kind::Url, r#"see https:\/\/x.org/a or https://x.org/a"#);
        assert_eq!(urls, vec!["https://x.org/a"]);
    }

    /// The code-level gate must keep matching once `find_urls` unescapes the
    /// value but the passage still carries the JSON escape.
    #[test]
    fn appears_in_matches_json_escaped_text() {
        assert!(appears_in(
            "https://www.aeropolis.es/",
            r#"{"site": "https:\/\/www.aeropolis.es\/"}"#
        ));
    }

    // -- find(Kind::Phone) ---------------------------------------------------

    #[test]
    fn finds_phone_with_spaces() {
        let phones = find(Kind::Phone, "Call us on 91 234 56 78 for enquiries.");
        assert_eq!(phones.len(), 1);
        let digit_count = phones[0].chars().filter(|c| c.is_ascii_digit()).count();
        assert!(digit_count >= 7, "expected ≥7 digits, got {}", phones[0]);
    }

    #[test]
    fn finds_international_phone() {
        let phones = find(Kind::Phone, "Tel: +34 912 345 678");
        assert_eq!(phones.len(), 1);
        assert!(phones[0].starts_with('+'));
    }

    #[test]
    fn ignores_short_digit_sequences() {
        // Year or small number — fewer than 7 digits should not match.
        let phones = find(Kind::Phone, "In 2024 there were 123 cases.");
        assert!(phones.is_empty(), "got unexpected phones: {:?}", phones);
    }

    // -- appears_in ----------------------------------------------------------

    #[test]
    fn appears_in_case_insensitive() {
        assert!(appears_in("Barcelona", "Located in barcelona, Spain."));
    }

    #[test]
    fn appears_in_whitespace_normalised() {
        assert!(appears_in("info@x.es", "  info@x.es  "));
        assert!(appears_in("hello world", "some hello   world text"));
    }

    #[test]
    fn appears_in_obfuscated_email_in_text() {
        // Canonical value, obfuscated text.
        assert!(appears_in(
            "info@ayto.es",
            "Write to info(at)ayto.es for help."
        ));
        assert!(appears_in(
            "info@ayto.es",
            "Write to info[at]ayto[dot]es for help."
        ));
    }

    #[test]
    fn at_obfuscation_does_not_invent_emails_from_prose() {
        assert!(find(Kind::Email, "The form is available at www.getafe.es today.").is_empty());
        assert!(find(Kind::Email, "We meet at the office.").is_empty());
        assert_eq!(
            find(Kind::Email, "Escriba a info at ayto-getafe . org"),
            vec!["info@ayto-getafe.org"]
        );
    }

    #[test]
    fn mailing_address_is_not_an_email_field() {
        assert_eq!(kind_for_field("mailing_address"), None);
        assert_eq!(kind_for_field("contact_email"), Some(Kind::Email));
    }

    #[test]
    fn appears_in_returns_false_when_absent() {
        assert!(!appears_in("missing@example.com", "No email here at all."));
    }

    // -- H3 email plausibility filter -----------------------------------

    #[test]
    fn rejects_cloudflare_email_protected_artefact() {
        // Cloudflare's "[email protected]" hover text lands as the literal
        // string "email@…" after our normaliser mangles it.
        let text = "Contact us at [email protected] for details.";
        // The regex matches "email@example.com"; the filter drops it.
        assert!(find(Kind::Email, text).is_empty());
    }

    #[test]
    fn rejects_local_part_email_exactly() {
        assert!(find(Kind::Email, "Write to email@ayuntamiento.es today.").is_empty());
    }

    #[test]
    fn keeps_email_prefix_that_is_not_the_full_local_part() {
        assert_eq!(
            find(Kind::Email, "Write emailing@ayuntamiento.es today."),
            vec!["emailing@ayuntamiento.es"]
        );
    }

    #[test]
    fn rejects_placeholder_domains() {
        for junk in [
            "someone@example.com",
            "someone@example.org",
            "someone@domain.com",
            "hello@email.com",
            "user@test.com",
        ] {
            let sample = format!("Reach us at {junk} today.");
            assert!(
                find(Kind::Email, &sample).is_empty(),
                "should reject {junk}"
            );
        }
    }

    #[test]
    fn keeps_ordinary_email_addresses() {
        assert_eq!(
            find(Kind::Email, "Reach us at info@ayto-orrius.cat."),
            vec!["info@ayto-orrius.cat"]
        );
    }

    #[test]
    fn rejects_local_part_containing_protected_word() {
        assert!(find(Kind::Email, "Write to protected@somewhere.com").is_empty());
        assert!(find(Kind::Email, "Write to my.protected.address@somewhere.com").is_empty());
    }
}
