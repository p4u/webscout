//! Types carried between stages.
//!
//! A running theme: anything a model inferred is stored next to the decision the
//! code made about it, never in place of it. A surprising result should always be
//! explicable after the fact from what is in the report.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What the user is actually asking for.
///
/// The split matters because the two shapes need different machinery. A question
/// wants one synthesized answer from a few good sources; an enumeration wants many
/// verified records and is not finished until it has enough of them. Treating the
/// second as the first is why naive research agents return "here are some
/// cooperatives" when you asked for a hundred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum MissionKind {
    /// Produce an answer supported by cited evidence.
    #[default]
    Answer,
    /// Collect as many distinct verified records as asked for.
    Harvest,
}

/// The parsed intent of the user's query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mission {
    pub query: String,
    pub kind: MissionKind,
    /// How many records an enumeration wants. `None` means "as many as exist".
    pub target_count: Option<usize>,
    /// Field names each harvested record should carry, e.g. `["name", "email"]`.
    #[serde(default)]
    pub fields: Vec<String>,
    /// The subject, stripped of instruction phrasing — used to build search queries.
    #[serde(default)]
    pub topic: String,
    /// Constraints a record must satisfy to count, e.g. "located in Catalonia".
    #[serde(default)]
    pub constraints: Vec<String>,

    /// Optional one-sentence plain-English glosses for each entry in
    /// `constraints`, positionally parallel to it. Each gloss lists equivalent
    /// wordings a page might use to satisfy the condition (e.g. for
    /// "integrators of Decidim.org": "companies or cooperatives that implement,
    /// host, customise or provide services around the Decidim platform; sites
    /// may call them partners, service providers, or implementers").
    ///
    /// Populated by `parse_mission`. Load-bearing for `ground_records`: the
    /// gloss is inlined into the per-record constraint noul so Jev judges the
    /// meaning of the condition, not a literal word match. Measured on the
    /// Decidim partners page (2026-09-18): "integrators" appears nowhere in
    /// the text (the page uses "service providers ... collaborate with
    /// Decidim"); a bare-word constraint scored 0.22-0.26 and every record
    /// was rejected. Reworded with a gloss and context-aware criteria the
    /// same records score 0.91-0.92.
    ///
    /// If the LLM omits or mismatches lengths, callers fall back to an empty
    /// gloss and the constraint check reduces to the bare wording.
    #[serde(default)]
    pub constraint_glosses: Vec<String>,

    /// Fields whose value is a determination (is it SaaS, does the provider
    /// offer an API), not a copyable fact. Populated once at harvest start
    /// from the enrich templates; discovery extraction never asks for them
    /// (a listing page cannot state them, and measured noise passes
    /// grounding: q62, 2026-09-23) — enrichment's Jev determination is the
    /// only writer.
    #[serde(default)]
    pub determination_fields: Vec<String>,

    /// Per-constraint flag, positionally parallel to `constraints`: `true`
    /// means a page that LISTS many such entities (a directory, register, or
    /// association member list) would not state this property per item — it
    /// is a fact of each entity's own page, fetched during enrichment, so
    /// stage goals must not demand it. Decided by one batched Jev ask at
    /// harvest start (`judge_listability`); the code heuristics in
    /// `listable_constraints` cover the shapes measured before the ask
    /// existed. Empty when no judgment ran (answer missions, or the ask
    /// failed) — then only the heuristics apply and every constraint stays
    /// listable, which is the pre-ask behaviour.
    ///
    /// Measured 2026-09-21 (q84, "AEI Cyber Security member companies with
    /// more than one office in Spain"): "has more than one office in Spain"
    /// mirrors no field by the token rule ("count" is absent from the
    /// wording) and no listing states office counts per member, so triage
    /// scored the association's own member-list pages 0.07–0.14 and 13
    /// rounds all reported `no_sources` while reading those very pages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unlistable_constraints: Vec<bool>,

    /// The name of the list a directory, register, or member page would
    /// publish for this mission — the topic stripped of any conditions on
    /// the items ("organizations using Decidim" rather than "organizations
    /// using Decidim that ran a vote in 2025 and the organization
    /// responsible for managing that vote"). Selected by one Jev `choice`
    /// over code-built clause prefixes at harvest start (the judge selects,
    /// it does not generate), in the same request as the listability nouls.
    ///
    /// Stage goals render through this (`goal_topic`); query building and
    /// steer keep the full `topic`, so searches stay specific while the
    /// accept-side goal names a list that can actually exist. Empty when no
    /// choice ran or Jev kept the full phrase — callers fall back to
    /// `topic`, which is the pre-mechanism behaviour.
    ///
    /// Measured 2026-09-21 (q63): with the composite topic in every goal,
    /// triage scored decidim.org/installations/ 0.19–0.23 for nine rounds
    /// (the same page scores 0.91 under a clean goal) and the run returned
    /// 1 record. The parse prompt alone did not prevent the classifier from
    /// folding the request's conditions into the topic.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub listing_core: String,

    /// Proper nouns / product names / identifiers copied verbatim from the
    /// request that DEFINE which items count (e.g. `["Decidim"]`). Distinct
    /// from `constraints`, which are conditions like "located in Catalonia":
    /// an anchor is the noun the constraint modifies. Load-bearing because
    /// the query planner is told never to invent product names — without this
    /// list the generated queries would omit the very anchor that identifies
    /// the set. Populated by `parse_mission`; may be empty for open-ended
    /// requests that name no specific product or organisation.
    #[serde(default)]
    pub anchors: Vec<String>,

    /// Geographic/temporal scope of the request ("worldwide", "Spain", "2024").
    /// Free-form phrase used by the research planner; empty when the request
    /// puts no scope on the items.
    #[serde(default)]
    pub scope: String,

    /// A single-fact lookup: one short answer, and the whole web agrees on it.
    ///
    /// "Who is the CEO of X" is not the same job as "compare the GDPR and the AI
    /// Act", and treating them identically is why a trivial question took thirty
    /// seconds: five planned queries, ten pages fetched, eighty-five chunks screened
    /// and a reasoning-heavy synthesis, to establish something the first search
    /// snippet already said. This flag lets the pipeline scale its effort to the
    /// question.
    #[serde(default)]
    pub simple: bool,

    /// The field that names the entity being harvested (e.g. "name", "municipio").
    /// Populated by `pick_entity_field` after parsing; Package B uses it for
    /// stage-specific goals and association grounding.
    #[serde(default)]
    pub entity_field: String,

    /// What kind of thing each requested item is, plain phrase — "companies
    /// and cooperatives", "cities", "academic papers". Populated by the
    /// mission parse (P1). Used by the research planner (P2) to frame the
    /// "where on the web would a complete list of X be published" question
    /// and to build the seed query "{anchor} {entity_type}". Empty when the
    /// parse didn't populate it; callers fall back to `topic` in that case.
    #[serde(default)]
    pub entity_type: String,

    /// The answer depends on when the question is asked.
    ///
    /// True for "the latest", "the current", "the next", "the newest", "upcoming"
    /// — anything whose correct answer changed at some point and will change
    /// again. Measured cause: asked "how many seasons of The White Lotus are
    /// available and when does the next one come out?" with no evidence and no
    /// date, `google/gemma-4-31b-it` answers "two seasons ... the third is
    /// expected in 2025", which was true when its weights were frozen and is not
    /// true now.
    ///
    /// Set by `parse_mission`. Load-bearing on the answer path: it switches on the
    /// per-chunk `cur{slot}` currency noul in `screen_passages`, which costs one
    /// extra question per chunk and so is not paid for timeless questions.
    #[serde(default)]
    pub time_sensitive: bool,
}

impl Mission {
    pub fn is_harvest(&self) -> bool {
        self.kind == MissionKind::Harvest
    }

    /// Whether enough has been collected to stop.
    pub fn satisfied_by(&self, count: usize) -> bool {
        match self.target_count {
            Some(target) => count >= target,
            // With no explicit target, only exhaustion stops a harvest.
            None => false,
        }
    }

    /// Pick the field that names the entity from a list of field names.
    ///
    /// Prefers fields whose lowercase name is a well-known entity identifier (name,
    /// organisation, municipality, etc.), then falls back to the first field, then
    /// to the literal string "name" when the list is empty.
    /// Field names that can carry an entity's identity. Shared by
    /// `pick_entity_field` and `needs_identity_field` so the two can never
    /// drift apart.
    pub const ENTITY_NAMES: &[&str] = &[
        "name",
        "nombre",
        "nom",
        "title",
        "organisation",
        "organization",
        "entity",
        "company",
        "municipality",
        "municipio",
        "ayuntamiento",
    ];

    pub fn pick_entity_field(fields: &[String]) -> String {
        for f in fields {
            if Self::ENTITY_NAMES.contains(&f.to_lowercase().as_str()) {
                return f.clone();
            }
        }
        fields
            .first()
            .cloned()
            .unwrap_or_else(|| "name".to_string())
    }

    /// True when no field can identify an entity on its own. A harvest with
    /// such fields has no identity: `pick_entity_field` would key records by
    /// whatever the first field happened to hold (measured 2026-09-21,
    /// "compare 10 voting providers by pricing, target market, …": the parse
    /// returned the five comparison dimensions and every record came out a
    /// loose page fragment — one per column, duplicated per language).
    pub fn needs_identity_field(fields: &[String]) -> bool {
        !fields
            .iter()
            .any(|f| Self::ENTITY_NAMES.contains(&f.to_lowercase().as_str()))
    }
}

/// One search hit, as observed. Nothing here is inferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    /// Which search engines returned this URL. Two engines agreeing is a
    /// cheap prior that costs no Jev tokens, so triage uses the count as a
    /// tiebreak. Empty on hits from a backend that does not report it.
    #[serde(default)]
    pub engines: Vec<String>,
}

impl Hit {
    pub fn domain(&self) -> String {
        url::Url::parse(&self.url)
            .ok()
            .and_then(|u| {
                u.host_str()
                    .map(|h| h.trim_start_matches("www.").to_string())
            })
            .unwrap_or_default()
    }
}

/// A search hit plus what Jev thought of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub hit: Hit,
    pub relevance: f64,
    pub authority: f64,
    pub slop: f64,
    pub rank: f64,
    pub kept: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop_reason: Option<String>,
}

/// A chunk of a fetched page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passage {
    pub url: String,
    pub title: String,
    pub text: String,
    #[serde(default)]
    pub supports: f64,
    #[serde(default)]
    pub injection: f64,

    /// Jev's probability that this passage describes the state of affairs as it
    /// stands today, rather than one that has since been superseded.
    ///
    /// Only asked for time-sensitive missions (`Mission::time_sensitive`), because
    /// it costs one extra question per chunk; everything else keeps the default
    /// 1.0 and the ranking reduces exactly to `supports`. Used as a ranking
    /// multiplier so current evidence reaches the writer first, and as a floor
    /// (`Tunables::currency_floor`) below which the passage is dropped outright.
    #[serde(default = "default_one")]
    pub currency: f64,
}

/// Where a specific field value was verified: the page it came from and how
/// strongly Jev grounded the value against that page's text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSource {
    pub source_url: String,
    pub grounding: f64,
}

/// One harvested record: a set of fields, where it came from, and how well the
/// source actually backs it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Record {
    /// Field name to value. BTreeMap so serialization order is stable, which
    /// matters for diffing two runs and for CSV column ordering.
    pub fields: BTreeMap<String, String>,
    pub source_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_title: Option<String>,
    /// Jev's probability that this record is genuinely stated in the source text.
    /// A low value here is what a fabricated record looks like.
    pub grounding: f64,

    /// Per-field provenance: where each specific value was verified and how
    /// strongly it was grounded there. Absent when a field was sourced directly
    /// from `source_url` without separate verification.
    /// Populated by Package B enrichment; defaults to empty.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provenance: BTreeMap<String, FieldSource>,

    /// Jev's probability that this record satisfies all mission constraints.
    /// 1.0 when no constraints are defined; populated by Package B.
    ///
    /// Since the constraint check became a five-way `choice` this is the
    /// MINIMUM probability Jev assigned to the `supports` option across the
    /// mission's constraints, so the field keeps its old meaning (higher =
    /// better supported) and existing JSON/CSV consumers and the sort order
    /// are unaffected. The categorical verdict lives in `constraint_status`.
    #[serde(default = "default_one")]
    pub constraint_support: f64,

    /// What the source passage established about each mission constraint,
    /// positionally parallel to `Mission::constraints`.
    ///
    /// The distinction this exists to preserve: a page that is SILENT about a
    /// criterion and a page that CONTRADICTS it used to produce the same low
    /// binary noul and both records were discarded — one live run threw away
    /// 129 records with no way to tell the two apart. `contradicts` now
    /// excludes the record outright; `not_addressed` / `ambiguous` / `mixed`
    /// keep it, unverified, out of the completion count.
    ///
    /// Empty on records produced before this field existed (and on missions
    /// with no constraints); `is_complete` reads an empty vector as "nothing
    /// to check".
    #[serde(default)]
    pub constraint_status: Vec<ConstraintVerdict>,

    /// What relationship the page the record came from has to the entity
    /// named in the record.
    ///
    /// Measured misattributions this catches (live, 2026-09-20): Jumilla was
    /// given `info@vinosdejumilla.org` (the wine regulatory council —
    /// `related` 0.73), Terrassa an insurer's address (`related` 0.53), Elche
    /// the provincial council's (`different` 0.70), while genuine matches
    /// (Bilbao's own town-hall page, Sant Adrià de Besòs' OAC page) score
    /// `same` 1.00. `Related` rows are kept but can never be complete: the
    /// organisation described is not the one that was asked for.
    #[serde(default, skip_serializing_if = "EntityBinding::is_unresolved")]
    pub entity_binding: EntityBinding,
}

/// What a passage establishes about one mission constraint for one record.
///
/// A five-way verdict rather than a probability because the consequences
/// differ in kind, not in degree: silence is recoverable by another page,
/// a contradiction is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintVerdict {
    /// The passage states or clearly implies the entity meets the criterion.
    Supports,
    /// The passage states something that cannot be true if it does.
    Contradicts,
    /// The passage says nothing either way.
    NotAddressed,
    /// The criterion is touched on, but the meaning or the subject is unclear.
    Ambiguous,
    /// Both supporting and conflicting statements are present.
    Mixed,
    /// No check has run yet. The default, so a deserialized record without the
    /// field is never mistaken for a verified one.
    #[default]
    Unchecked,
}

impl ConstraintVerdict {
    /// Only `Supports` counts a constraint as met. A page-level rescue
    /// rewrites the stored verdict to `Supports`, so this stays a single
    /// comparison at every read site.
    pub fn is_satisfied(self) -> bool {
        matches!(self, ConstraintVerdict::Supports)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ConstraintVerdict::Supports => "supports",
            ConstraintVerdict::Contradicts => "contradicts",
            ConstraintVerdict::NotAddressed => "not_addressed",
            ConstraintVerdict::Ambiguous => "ambiguous",
            ConstraintVerdict::Mixed => "mixed",
            ConstraintVerdict::Unchecked => "unchecked",
        }
    }

    /// Read a Jev `choice` label. Anything unrecognised — including the empty
    /// string a failed or missing answer produces — reads as `not_addressed`:
    /// a failed guard is never an open door, and "we could not tell" is
    /// exactly what silence means here.
    pub fn from_choice(label: &str) -> Self {
        match label.trim() {
            "supports" => ConstraintVerdict::Supports,
            "contradicts" => ConstraintVerdict::Contradicts,
            "ambiguous" => ConstraintVerdict::Ambiguous,
            "mixed" => ConstraintVerdict::Mixed,
            _ => ConstraintVerdict::NotAddressed,
        }
    }
}

/// What relationship the organisation a page describes has to the target
/// entity. The answer to the question nobody was asking before: "related but
/// distinct" had no name, so a federation's or an insurer's contact details
/// were attached to a town as if they were the town's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EntityBinding {
    /// The entity itself, including an official department or office of it.
    Same,
    /// A parent, member, subsidiary, federation, supplier or contractor —
    /// related, and not the entity.
    Related,
    /// Unrelated, or a same-named place or body elsewhere.
    Different,
    /// The passage does not establish which organisation it is about. The
    /// default, and the only verdict that behaves as the pre-binding code did.
    #[default]
    Unresolved,
}

impl EntityBinding {
    pub fn is_unresolved(&self) -> bool {
        matches!(self, EntityBinding::Unresolved)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EntityBinding::Same => "same",
            EntityBinding::Related => "related",
            EntityBinding::Different => "different",
            EntityBinding::Unresolved => "unresolved",
        }
    }

    /// Read a Jev `choice` label; an unrecognised or missing label is
    /// `unresolved`, which keeps the caller on its pre-binding behaviour
    /// rather than granting or denying anything on a failed call.
    pub fn from_choice(label: &str) -> Self {
        match label.trim() {
            "same" => EntityBinding::Same,
            "related" => EntityBinding::Related,
            "different" => EntityBinding::Different,
            _ => EntityBinding::Unresolved,
        }
    }

    /// Rank used when two records for the same entity are merged: a page that
    /// resolved the entity outranks one that did not. `Different` never
    /// reaches a merge (those records are discarded at grounding).
    fn rank(self) -> u8 {
        match self {
            EntityBinding::Same => 3,
            EntityBinding::Related => 2,
            EntityBinding::Unresolved => 1,
            EntityBinding::Different => 0,
        }
    }

    /// Merge two bindings for the same entity, keeping the more informative
    /// one. Two pages describing the same town, one of which is unmistakably
    /// its own site, leave the record bound as `same`.
    pub fn merge(self, other: Self) -> Self {
        if other.rank() > self.rank() {
            other
        } else {
            self
        }
    }
}

/// Element-wise merge of two constraint-status vectors for the same entity.
///
/// Both records passed the exclusion gate (neither contradicts), so a
/// `supports` from either page settles the constraint. Lengths may differ
/// when one record predates a constraint list change; the longer vector's
/// tail is carried through.
pub fn merge_constraint_status(
    a: &[ConstraintVerdict],
    b: &[ConstraintVerdict],
) -> Vec<ConstraintVerdict> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| match (a.get(i).copied(), b.get(i).copied()) {
            (Some(x), Some(y)) => {
                if x.is_satisfied() || y.is_satisfied() {
                    ConstraintVerdict::Supports
                } else if x == ConstraintVerdict::Unchecked {
                    y
                } else {
                    x
                }
            }
            (Some(x), None) => x,
            (None, Some(y)) => y,
            (None, None) => ConstraintVerdict::Unchecked,
        })
        .collect()
}

fn default_one() -> f64 {
    1.0
}

impl Record {
    pub fn get(&self, field: &str) -> &str {
        self.fields.get(field).map(String::as_str).unwrap_or("")
    }
}

/// Normalise an entity name for deduplication.
///
/// Steps:
/// 1. Lowercase.
/// 2. Strip accents (á→a, é→e, ñ→n, ç→c, etc.).
/// 3. Strip a single leading municipal/administrative prefix in es/ca/gl/eu/en/it/fr/de.
/// 4. Collapse runs of whitespace and punctuation (excluding alphanumeric) to a
///    single space.
/// 5. Trim.
///
/// "Ayuntamiento de Madrid" and "Madrid" both normalise to "madrid".
/// "Ajuntament d'Olot" and "Olot" both normalise to "olot".
pub fn normalize_entity(s: &str) -> String {
    // Step 1: lowercase.
    let lower = s.to_lowercase();

    // Step 2: strip accents.
    let no_accents = strip_accents(&lower);

    // Step 2b: if a trailing parenthetical is an institutional gloss of the
    // form "(ayuntamiento de X)" / "(ajuntament de X)" / "(concello de X)"
    // etc., use X as the whole name. This is what turns
    // "bilboko udala (ayuntamiento de bilbao)" into "bilbao" — the Basque
    // heading and its Spanish gloss point at the same municipality but
    // would otherwise fail dedup because neither reduces to the other.
    // Otherwise, strip a bare region parenthetical like "(Madrid)".
    let no_parens = match institutional_parenthetical(&no_accents) {
        Some(inner) => inner,
        None => strip_trailing_parenthetical(&no_accents),
    };

    // Step 2c: strip Basque " udala" / " udaletxea" suffix words.
    // "BILBAO UDALA" → "bilbao". Kept deliberately naive (word suffix only);
    // we do not attempt to unwind Basque genitive declension, so e.g.
    // "bilboko udala" reduces to "bilboko" and is expected to collide via
    // its Spanish gloss ("ayuntamiento de bilbao") if the caller supplies
    // one via the institutional-parenthetical rule above.
    let no_basque = strip_basque_udala_suffix(&no_parens);

    // Step 3: strip leading administrative prefix.
    let stripped = strip_municipal_prefix(&no_basque);

    // Step 3b: a GitHub/GitLab slug is `owner/repo`. The owner is account
    // metadata, not part of the project's name, and keeping it in the key
    // left one project twice — once as the slug a repo listing gave
    // (`bottelet/daybydaycrm`) and once as the name an article gave
    // (`DayByDayCRM`) — with the near-collision question reading literally
    // enough to keep them apart (measured 2026-09-21, OSS-CRM harvest:
    // Dolibarr, DayByDayCRM and OroCRM each appeared twice). Stripped only
    // when the shape is exactly one slash with no whitespace anywhere, so
    // ordinary names that merely contain a slash stay whole; a multi-segment
    // path (`a/b/c`) is left for the punctuation collapse.
    let stripped = match stripped.split_once('/') {
        Some((owner, repo))
            if !owner.is_empty()
                && !repo.is_empty()
                && !repo.contains('/')
                && !owner.contains(char::is_whitespace)
                && !repo.contains(char::is_whitespace) =>
        {
            repo
        }
        _ => stripped,
    };

    // Step 4: collapse non-alphanumeric runs (whitespace + punctuation) to one space.
    let mut out = String::with_capacity(stripped.len());
    let mut in_gap = false;
    for ch in stripped.chars() {
        if ch.is_alphanumeric() {
            if in_gap && !out.is_empty() {
                out.push(' ');
            }
            in_gap = false;
            out.push(ch);
        } else {
            in_gap = true;
        }
    }

    out
}

/// If `s` ends with a parenthetical whose contents match a known
/// institutional prefix (`ayuntamiento de X` / `ajuntament de X` /
/// `concello de X` / etc.), return `X` (the whole target name), stripped.
/// Otherwise return `None`.
///
/// The comparison operates on lowercase, accent-stripped text, which is
/// what `normalize_entity` produces by the time it calls in here.
fn institutional_parenthetical(s: &str) -> Option<String> {
    let trimmed = s.trim_end();
    if !trimmed.ends_with(')') {
        return None;
    }
    let open = trimmed.rfind('(')?;
    let inner = trimmed[open + 1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return None;
    }
    // Only accept the parenthetical if it clearly is an institutional gloss.
    let inner_stripped = strip_municipal_prefix(inner);
    if inner_stripped != inner && !inner_stripped.trim().is_empty() {
        return Some(inner_stripped.trim().to_string());
    }
    None
}

/// Strip a trailing Basque " udala" / " udaletxea" suffix word.
/// Case is already lowercased by the caller.
fn strip_basque_udala_suffix(s: &str) -> String {
    let t = s.trim_end();
    for suf in &[" udaletxea", " udala"] {
        if let Some(rest) = t.strip_suffix(suf) {
            return rest.trim_end().to_string();
        }
    }
    t.to_string()
}

/// Drop a trailing parenthetical (usually a region qualifier: "(Madrid)").
fn strip_trailing_parenthetical(s: &str) -> String {
    let trimmed = s.trim_end();
    if let Some(open) = trimmed.rfind('(')
        && trimmed.ends_with(')')
    {
        return trimmed[..open].trim_end().to_string();
    }
    trimmed.to_string()
}

/// Replace accented and special characters with their ASCII base equivalents.
fn strip_accents(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => out.push('a'),
            'æ' => out.push_str("ae"),
            'ç' => out.push('c'),
            'è' | 'é' | 'ê' | 'ë' => out.push('e'),
            'ì' | 'í' | 'î' | 'ï' => out.push('i'),
            'ñ' => out.push('n'),
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' => out.push('o'),
            'œ' => out.push_str("oe"),
            'ù' | 'ú' | 'û' | 'ü' => out.push('u'),
            'ý' | 'ÿ' => out.push('y'),
            'ß' => out.push_str("ss"),
            'ł' => out.push('l'),
            other => out.push(other),
        }
    }
    out
}

/// Strip a single leading municipal/administrative prefix.
///
/// Prefixes are tried longest-first to avoid partial matches (e.g. "ayto. de"
/// before "ayto.").  The comparison is done on already-lowercased, accent-stripped
/// text.
fn strip_municipal_prefix(s: &str) -> &str {
    // Ordered longest → shortest within each language group.
    const PREFIXES: &[&str] = &[
        // "de" forms first, longest → shortest, so "ayuntamiento de X" wins over
        // the bare "ayuntamiento X" case below.
        "ayuntamiento de ",
        "municipality of ",
        "ajuntament de ",
        "concello de ",
        "concejo de ",
        "municipio de ",
        "udaletxea ",
        "comune di ",
        "mairie de ",
        "ajuntament d'",
        "ville de ",
        "town of ",
        "city of ",
        "udala ",
        "ayto. de ",
        "ayto de ",
        "gemeinde ",
        "stadt ",
        "ayto. ",
        "ayto ",
        // Bare institutional prefixes without "de". Observed live: entities
        // arrived keyed as "ayuntamiento san pedro del pinatar", which the
        // "de"-forms above did not touch, and every dedup fell back to
        // whole-string matching.
        "ayuntamiento ",
        "ajuntament ",
        "concello ",
        "concejo ",
        "municipio ",
        "comune ",
        "mairie ",
    ];
    for prefix in PREFIXES {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest;
        }
    }
    s
}

/// How a run ended. Reported rather than inferred from the record count, because
/// "found everything asked for" and "ran out of leads" are different results even
/// when they produce the same number of rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The target count was reached, or the question was answered.
    Complete,
    /// Real results, but fewer than asked for; the web was exhausted first.
    Partial,
    /// Nothing usable was found.
    Empty,
    /// Stopped at the round ceiling with progress still being made.
    /// Re-running with a higher `--max-rounds` would likely find more.
    Truncated,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Complete => "complete",
            Outcome::Partial => "partial",
            Outcome::Empty => "empty",
            Outcome::Truncated => "truncated",
        }
    }
}

/// Token and request accounting, so a run's cost is never a mystery.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stats {
    pub rounds: usize,
    pub queries_issued: usize,
    pub pages_fetched: usize,
    pub chunks_examined: usize,

    pub jev_requests: usize,
    pub jev_input_tokens: usize,
    pub jev_cost_usd: f64,

    pub llm_requests: usize,
    pub llm_prompt_tokens: usize,
    pub llm_completion_tokens: usize,
    /// Of `llm_completion_tokens`, the part spent on chain of thought.
    ///
    /// Reported by providers that send `usage.completion_tokens_details`. A
    /// large share here on extraction calls is the signature of the wrong
    /// `--thinking-control` for the endpoint.
    #[serde(default)]
    pub llm_reasoning_tokens: usize,
    /// What the writer endpoint said this run cost, in USD.
    ///
    /// `None` when the endpoint never reported a cost (anything that is not
    /// OpenRouter, so far). Absent is not zero, and no price table is inferred
    /// from a model name — a guessed invoice is worse than an honest gap.
    #[serde(default)]
    pub llm_cost_usd: Option<f64>,

    /// Planner-LLM accounting, kept separate because the planner is often a
    /// larger reasoning model while the writer stays cheap. When no
    /// `--planner-model` is set both point at the same model and only
    /// `llm_*` above accumulate; a run configured with two models sees the
    /// planner cost broken out here.
    #[serde(default)]
    pub planner_requests: usize,
    #[serde(default)]
    pub planner_prompt_tokens: usize,
    #[serde(default)]
    pub planner_completion_tokens: usize,
    #[serde(default)]
    pub planner_reasoning_tokens: usize,
    /// As `llm_cost_usd`, for the planner endpoint.
    #[serde(default)]
    pub planner_cost_usd: Option<f64>,

    /// Chunks Jev refused to pass to the generative model.
    pub quarantined: usize,
    /// Records the generative model produced that Jev could not find in the source.
    pub rejected_ungrounded: usize,

    /// Claims in the written answer that Jev scored individually against the
    /// evidence. Zero on a harvest run, and on an answer run whose claim check
    /// failed outright — the difference between "checked and clean" and "not
    /// checked" is exactly what this field exists to preserve.
    #[serde(default)]
    pub claims_checked: usize,

    /// Claims the evidence did not state. Each is marked ` [unsupported]` in the
    /// answer text rather than deleted, because removing a sentence from prose
    /// breaks the citations that follow it. The answer-path analogue of
    /// `rejected_ungrounded`.
    #[serde(default)]
    pub unsupported_claims: usize,

    pub elapsed_secs: f64,

    /// Time spent inside each pipeline stage, in milliseconds, with a call count.
    ///
    /// These sum to *more* than `elapsed_secs`, and that is the point: stages
    /// overlap, so the gap between the sum and the wall clock is exactly how much
    /// the concurrency and pipelining are buying. A stage whose time approaches the
    /// wall clock is one that nothing else is hiding behind.
    #[serde(default)]
    pub stage_ms: BTreeMap<String, StageTime>,

    /// Distinct entity names seen during discovery (Package B).
    #[serde(default)]
    pub entities_discovered: usize,

    /// Entities that had at least one field filled by the enrichment phase (Package B).
    #[serde(default)]
    pub entities_enriched: usize,

    /// Pages added to the crawl queue via link-following (Package B).
    #[serde(default)]
    pub links_followed: usize,

    /// Search queries issued specifically for enrichment (Package B).
    #[serde(default)]
    pub enrich_searches: usize,

    /// Records excluded because the source passage CONTRADICTED a mission
    /// constraint. Separate from `rejected_ungrounded`: those records were
    /// not in the text, these were in the text and the text said no. A
    /// page-level rescue never overrides a contradiction.
    #[serde(default)]
    pub excluded_contradicted: usize,

    /// (record, constraint) pairs kept but not verified — the passage said
    /// nothing either way, or was ambiguous, and no page-level rescue
    /// applied. These rows are reported but never counted as complete.
    #[serde(default)]
    pub constraints_unverified: usize,

    /// Values and records rejected because the page's organisation was
    /// `related` or `different` from the target entity, rather than the
    /// entity itself. Counts both the discovery-grounding discards and the
    /// enrichment post-pick rejections.
    #[serde(default)]
    pub wrong_entity_rejected: usize,
    /// Records dropped because a complete enumeration on the mission's
    /// anchor domain never mentioned their entity — third-party pages
    /// inventing members of a list the authoritative page completes.
    pub anchor_unmentioned: usize,
    /// Records dropped because their name is only the mission's own category
    /// ("crm" for a mission about "open-source CRM projects"). Such a key is a
    /// hub: the containment prefilter matches it against every name that
    /// contains the word, and transitive merges then swallow unrelated real
    /// entities through it. Measured 2026-09-21 (q8): one bare "crm" record
    /// chained EspoCRM, CiviCRM, ChurchCRM, DayByDayCRM and more into a single
    /// group, and CiviCRM vanished from the output.
    pub category_echoes_dropped: usize,
}

/// Accumulated time and call count for one stage.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StageTime {
    pub ms: u64,
    pub calls: u32,
}

/// Everything a run produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutReport {
    pub query: String,
    pub mission: Mission,
    pub outcome: Outcome,

    /// Populated for `Harvest` missions. Always serialized, even when empty:
    /// a consumer indexing `result["records"]` must not crash on a run that
    /// found nothing, and "found nothing" is a normal outcome here.
    #[serde(default)]
    pub records: Vec<Record>,

    /// Populated for `Answer` missions: prose written by the generative model from
    /// passages Jev cleared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    /// Passages backing the answer, quoted verbatim.
    #[serde(default)]
    pub evidence: Vec<Passage>,

    /// Distinct pages that contributed at least one kept item.
    #[serde(default)]
    pub sources: Vec<String>,

    /// Pages that tried to address the model reading them.
    #[serde(default)]
    pub quarantined_sources: Vec<String>,

    #[serde(default)]
    pub notes: Vec<String>,

    pub stats: Stats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_entity_strips_repo_owner_slugs() {
        // One slash, no spaces: a repo slug. The owner is not the project's
        // name, so the key is the repo alone and two spellings of one project
        // collide exactly instead of relying on a fuzzy merge.
        assert_eq!(normalize_entity("Dolibarr/dolibarr"), "dolibarr");
        assert_eq!(normalize_entity("bottelet/DaybydayCRM"), "daybydaycrm");
        assert_eq!(normalize_entity("oroinc/crm"), "crm");
    }

    #[test]
    fn normalize_entity_keeps_slashes_that_are_not_slugs() {
        // Whitespace around the slash, or more than one slash: not a slug.
        assert_eq!(normalize_entity("ACME / Consulting"), "acme consulting");
        assert_eq!(normalize_entity("a/b/c"), "a b c");
    }

    #[test]
    fn normalize_entity_strips_accents_and_prefix() {
        assert_eq!(normalize_entity("Ayuntamiento de Madrid"), "madrid");
        assert_eq!(normalize_entity("AYUNTAMIENTO DE MÁLAGA"), "malaga");
        assert_eq!(normalize_entity("Ajuntament d'Olot"), "olot");
        assert_eq!(normalize_entity("Ajuntament de Girona"), "girona");
        assert_eq!(normalize_entity("Concello de Vigo"), "vigo");
        assert_eq!(normalize_entity("Stadt München"), "munchen");
        assert_eq!(normalize_entity("Gemeinde Köln"), "koln");
        assert_eq!(normalize_entity("City of London"), "london");
        assert_eq!(normalize_entity("Commune di Roma"), "commune di roma"); // 'commune' not listed
        assert_eq!(normalize_entity("Comune di Roma"), "roma");
        assert_eq!(normalize_entity("Mairie de Paris"), "paris");
        assert_eq!(normalize_entity("Ville de Lyon"), "lyon");
        assert_eq!(normalize_entity("Udala Bilbao"), "bilbao");
        assert_eq!(normalize_entity("Udaletxea Donostia"), "donostia");
        assert_eq!(normalize_entity("  Barcelona  "), "barcelona");
        assert_eq!(normalize_entity("Añó"), "ano");
    }

    #[test]
    fn normalize_entity_strips_bare_institutional_prefixes() {
        // Observed live: keys arrived without the "de" article joining the
        // institutional prefix to the place name.
        assert_eq!(
            normalize_entity("Ayuntamiento San Pedro del Pinatar"),
            "san pedro del pinatar"
        );
        assert_eq!(normalize_entity("Ajuntament Barcelona"), "barcelona");
        assert_eq!(normalize_entity("Municipio Alcala"), "alcala");
    }

    #[test]
    fn normalize_entity_strips_trailing_region_parenthetical() {
        assert_eq!(
            normalize_entity("San Pedro del Pinatar (Murcia)"),
            "san pedro del pinatar"
        );
        assert_eq!(
            normalize_entity("Ayuntamiento de Alcalá (Madrid)"),
            "alcala"
        );
    }

    #[test]
    fn normalize_entity_collapses_punctuation() {
        assert_eq!(normalize_entity("foo, bar - baz"), "foo bar baz");
        assert_eq!(normalize_entity("A & B"), "a b");
    }

    #[test]
    fn normalize_entity_institutional_parenthetical() {
        // Basque heading + Spanish gloss reduces to the target municipality.
        assert_eq!(
            normalize_entity("Bilboko Udala (Ayuntamiento de Bilbao)"),
            "bilbao"
        );
        assert_eq!(
            normalize_entity("Donostiako Udala (Ayuntamiento de San Sebastián)"),
            "san sebastian"
        );
        assert_eq!(
            normalize_entity("Ourensala (Concello de Ourense)"),
            "ourense"
        );
    }

    #[test]
    fn normalize_entity_bare_region_parenthetical_still_strips() {
        // A parenthetical that is NOT an institutional gloss is dropped, not
        // used as the name.
        assert_eq!(normalize_entity("San Pedro (Murcia)"), "san pedro");
    }

    #[test]
    fn normalize_entity_strips_trailing_basque_udala() {
        assert_eq!(normalize_entity("BILBAO UDALA"), "bilbao");
        assert_eq!(normalize_entity("Getxo Udaletxea"), "getxo");
        // Non-institutional word that happens to end similarly is untouched.
        assert_eq!(normalize_entity("Sudalar"), "sudalar");
    }

    #[test]
    fn normalize_entity_accentless_name_unchanged() {
        assert_eq!(normalize_entity("Barcelona"), "barcelona");
        assert_eq!(normalize_entity("Madrid"), "madrid");
    }

    #[test]
    fn constraint_verdict_round_trips_through_its_labels() {
        for (label, v) in [
            ("supports", ConstraintVerdict::Supports),
            ("contradicts", ConstraintVerdict::Contradicts),
            ("not_addressed", ConstraintVerdict::NotAddressed),
            ("ambiguous", ConstraintVerdict::Ambiguous),
            ("mixed", ConstraintVerdict::Mixed),
        ] {
            assert_eq!(ConstraintVerdict::from_choice(label), v);
            assert_eq!(v.as_str(), label);
            // The serde name must match the Jev option key, or the JSON
            // output and the question would disagree.
            assert_eq!(serde_json::to_string(&v).unwrap(), format!("\"{label}\""));
        }
    }

    /// A missing, empty or unrecognised answer reads as `not_addressed` —
    /// never as support. A failed guard is never an open door.
    #[test]
    fn unknown_constraint_choice_is_not_addressed() {
        for label in ["", "   ", "yes", "SUPPORTS", "unchecked"] {
            assert_eq!(
                ConstraintVerdict::from_choice(label),
                ConstraintVerdict::NotAddressed,
                "{label:?}"
            );
        }
        assert!(ConstraintVerdict::Supports.is_satisfied());
        for v in [
            ConstraintVerdict::Contradicts,
            ConstraintVerdict::NotAddressed,
            ConstraintVerdict::Ambiguous,
            ConstraintVerdict::Mixed,
            ConstraintVerdict::Unchecked,
        ] {
            assert!(!v.is_satisfied(), "{v:?}");
        }
        // The default must be the un-verified one.
        assert_eq!(ConstraintVerdict::default(), ConstraintVerdict::Unchecked);
    }

    #[test]
    fn entity_binding_labels_and_unknown_is_unresolved() {
        for (label, b) in [
            ("same", EntityBinding::Same),
            ("related", EntityBinding::Related),
            ("different", EntityBinding::Different),
            ("unresolved", EntityBinding::Unresolved),
        ] {
            assert_eq!(EntityBinding::from_choice(label), b);
            assert_eq!(b.as_str(), label);
            assert_eq!(serde_json::to_string(&b).unwrap(), format!("\"{label}\""));
        }
        assert_eq!(EntityBinding::from_choice("?"), EntityBinding::Unresolved);
        assert_eq!(EntityBinding::default(), EntityBinding::Unresolved);
        assert!(EntityBinding::Unresolved.is_unresolved());
        assert!(!EntityBinding::Related.is_unresolved());
    }

    /// Merging two views of one entity: the page that resolved the
    /// organisation wins, and `related` is not laundered into `same` by an
    /// unresolved second opinion.
    #[test]
    fn entity_binding_merge_keeps_the_more_informative_verdict() {
        assert_eq!(
            EntityBinding::Unresolved.merge(EntityBinding::Same),
            EntityBinding::Same
        );
        assert_eq!(
            EntityBinding::Same.merge(EntityBinding::Unresolved),
            EntityBinding::Same
        );
        assert_eq!(
            EntityBinding::Related.merge(EntityBinding::Unresolved),
            EntityBinding::Related
        );
        assert_eq!(
            EntityBinding::Related.merge(EntityBinding::Same),
            EntityBinding::Same
        );
    }

    #[test]
    fn merge_constraint_status_takes_support_from_either_page() {
        let a = [ConstraintVerdict::NotAddressed, ConstraintVerdict::Mixed];
        let b = [ConstraintVerdict::Supports, ConstraintVerdict::Ambiguous];
        assert_eq!(
            merge_constraint_status(&a, &b),
            vec![ConstraintVerdict::Supports, ConstraintVerdict::Mixed]
        );
        // Unchecked defers to whatever the other page found.
        assert_eq!(
            merge_constraint_status(&[ConstraintVerdict::Unchecked], &[ConstraintVerdict::Mixed]),
            vec![ConstraintVerdict::Mixed]
        );
        // Differing lengths keep the longer vector's tail.
        assert_eq!(
            merge_constraint_status(&[], &[ConstraintVerdict::Supports]),
            vec![ConstraintVerdict::Supports]
        );
        assert_eq!(
            merge_constraint_status(&[ConstraintVerdict::Ambiguous], &[]),
            vec![ConstraintVerdict::Ambiguous]
        );
    }

    /// A record serialized before these fields existed must still load, and
    /// must not come back looking verified.
    #[test]
    fn record_without_the_new_fields_deserializes_unverified() {
        let json = r#"{
            "fields": {"name": "Acme"},
            "source_url": "https://example.org",
            "grounding": 0.9
        }"#;
        let r: Record = serde_json::from_str(json).unwrap();
        assert!(r.constraint_status.is_empty());
        assert_eq!(r.entity_binding, EntityBinding::Unresolved);
        // `constraint_support` keeps its 1.0 default, so existing consumers
        // and the sort order are unaffected.
        assert_eq!(r.constraint_support, 1.0);
        // An unresolved binding is not written out, so JSON stays as it was
        // for every record the binding question could not resolve.
        let out = serde_json::to_string(&r).unwrap();
        assert!(!out.contains("entity_binding"), "{out}");
        let bound = Record {
            entity_binding: EntityBinding::Related,
            constraint_status: vec![ConstraintVerdict::Mixed],
            ..r
        };
        let out = serde_json::to_string(&bound).unwrap();
        assert!(out.contains("\"entity_binding\":\"related\""), "{out}");
        assert!(out.contains("\"constraint_status\":[\"mixed\"]"), "{out}");
    }

    #[test]
    fn pick_entity_field_prefers_known_names() {
        let fields: Vec<String> = ["email", "name", "website"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(Mission::pick_entity_field(&fields), "name");
    }

    #[test]
    fn pick_entity_field_falls_back_to_first() {
        let fields: Vec<String> = ["correo", "id"].iter().map(|s| s.to_string()).collect();
        assert_eq!(Mission::pick_entity_field(&fields), "correo");
    }

    #[test]
    fn needs_identity_field_detects_missing_identity() {
        let dims: Vec<String> = ["pricing", "target_market", "hosting_model"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(Mission::needs_identity_field(&dims));
        let with_name: Vec<String> = ["pricing", "name"].iter().map(|s| s.to_string()).collect();
        assert!(!Mission::needs_identity_field(&with_name));
        // Regional entity names count as identity too.
        let ayto: Vec<String> = ["email", "ayuntamiento"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(!Mission::needs_identity_field(&ayto));
    }

    #[test]
    fn pick_entity_field_empty_returns_name() {
        assert_eq!(Mission::pick_entity_field(&[]), "name");
    }

    #[test]
    fn pick_entity_field_ayuntamiento() {
        let fields: Vec<String> = ["ayuntamiento", "email"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(Mission::pick_entity_field(&fields), "ayuntamiento");
    }
}
