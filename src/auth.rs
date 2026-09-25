//! The web UI's login: one shared password, server-side sessions, a cookie.
//!
//! This replaced HTTP Basic auth. Basic auth has no logout (a browser keeps
//! resending the credentials until it is closed), shows the browser's own
//! unstyled dialog, and sends the password with every request. A session
//! cookie fixes all three for the price of a small in-memory table.
//!
//! Three rules shape it.
//!
//! **No password, no login.** Unset or blank means the server is open, which is
//! right for a laptop or a private network. A public deployment sets
//! `WEBSCOUT_PASSWORD`; searches are billed to the server's keys.
//!
//! **Sessions live in memory.** A restart logs everyone out, which for a
//! single-password tool is a feature rather than a cost: there is no session
//! file to leak and nothing to migrate.
//!
//! **Guessing is slow and bounded.** Every wrong password costs the caller
//! `FAILURE_DELAY`, and once `MAX_FAILURES` wrong passwords have landed in a
//! minute — counted process-wide, not per address, because behind a platform
//! proxy the address is the proxy's — every further attempt gets 429 until the
//! window slides. A legitimate user waits a minute; a guesser gets ten tries a minute.

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::http::{HeaderMap, header};

/// The session cookie's name.
pub const COOKIE_NAME: &str = "webscout_session";

/// How long a session lasts: thirty days, the cookie's `Max-Age`.
pub const SESSION_TTL: Duration = Duration::from_secs(30 * 24 * 3600);

/// Wrong passwords tolerated inside `FAILURE_WINDOW`, across all callers. The
/// attempt after the tenth failure is refused with 429 whatever its password,
/// so a guesser cannot confirm a hit by timing the lockout.
pub const MAX_FAILURES: usize = 10;

/// The sliding window `MAX_FAILURES` is counted over.
pub const FAILURE_WINDOW: Duration = Duration::from_secs(60);

/// What a wrong password costs the caller before the 401 arrives.
pub const FAILURE_DELAY: Duration = Duration::from_millis(700);

/// Random bytes per session token. 256 bits: unguessable, and the hex form is
/// 64 characters, which is what the contract with the UI states.
const TOKEN_BYTES: usize = 32;

/// Outcome of a login attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum Login {
    /// Right password: here is the new session token.
    Ok(String),
    /// No password configured: nothing to do, no cookie to set.
    Open,
    Wrong,
    /// Too many wrong passwords in the last minute.
    Throttled,
}

/// The password, the live sessions and the recent failures.
pub struct Auth {
    password: Option<String>,
    /// Token → expiry. Pruned lazily on every lookup and insert: the table is
    /// small (one entry per browser that logged in within thirty days), so a
    /// full sweep costs less than a timer task would.
    sessions: Mutex<HashMap<String, Instant>>,
    failures: Mutex<VecDeque<Instant>>,
    /// `FAILURE_DELAY`, except in tests, which set it to zero.
    pub(crate) failure_delay: Duration,
}

impl Default for Auth {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Auth {
    /// `password` goes through `unquote`, like every other configured value;
    /// blank is the same as unset.
    pub fn new(password: Option<String>) -> Self {
        let password = password
            .map(|p| crate::config::unquote(&p).to_string())
            .filter(|p| !p.is_empty());
        Self {
            password,
            sessions: Mutex::new(HashMap::new()),
            failures: Mutex::new(VecDeque::new()),
            failure_delay: FAILURE_DELAY,
        }
    }

    pub fn required(&self) -> bool {
        self.password.is_some()
    }

    /// Check a password and, if right, open a session.
    ///
    /// The delay for a wrong password is the caller's to apply (see
    /// `failure_delay`), so no lock is held while a guesser waits.
    pub fn login(&self, given: &str) -> std::io::Result<Login> {
        let Some(expected) = &self.password else {
            return Ok(Login::Open);
        };
        let now = Instant::now();
        {
            let Ok(mut f) = self.failures.lock() else {
                return Ok(Login::Throttled);
            };
            while f
                .front()
                .is_some_and(|t| now.duration_since(*t) >= FAILURE_WINDOW)
            {
                f.pop_front();
            }
            if f.len() >= MAX_FAILURES {
                return Ok(Login::Throttled);
            }
            if !secret_eq(given.as_bytes(), expected.as_bytes()) {
                f.push_back(now);
                return Ok(Login::Wrong);
            }
        }
        let token = new_token()?;
        if let Ok(mut s) = self.sessions.lock() {
            s.retain(|_, exp| *exp > now);
            s.insert(token.clone(), now + SESSION_TTL);
        }
        Ok(Login::Ok(token))
    }

    /// Whether `token` names a live session.
    pub fn valid(&self, token: &str) -> bool {
        let now = Instant::now();
        let Ok(mut s) = self.sessions.lock() else {
            return false;
        };
        s.retain(|_, exp| *exp > now);
        s.contains_key(token)
    }

    /// Whether any session cookie on the request names a live session.
    pub fn authenticated(&self, headers: &HeaderMap) -> bool {
        session_cookies(headers).any(|t| self.valid(t))
    }

    /// Close every session a request's cookies name.
    pub fn logout(&self, headers: &HeaderMap) {
        let tokens: Vec<&str> = session_cookies(headers).collect();
        if let Ok(mut s) = self.sessions.lock() {
            for t in tokens {
                s.remove(t);
            }
        }
    }
}

/// Compare a guess against the secret without an early exit on the first
/// differing byte *or* on a length mismatch: the loop runs over the secret's
/// length whatever was sent, so timing says nothing about how long the real
/// password is or how much of a guess was right.
fn secret_eq(given: &[u8], secret: &[u8]) -> bool {
    let mut acc = u8::from(given.len() != secret.len());
    for (i, s) in secret.iter().enumerate() {
        acc |= s ^ given.get(i).copied().unwrap_or(0);
    }
    acc == 0
}

/// A fresh session token: `TOKEN_BYTES` from the kernel's CSPRNG, as hex.
///
/// Read straight from `/dev/urandom` rather than through a `rand` crate: one
/// read at login is all this needs, and the image only ever runs on Linux.
fn new_token() -> std::io::Result<String> {
    let mut buf = [0u8; TOKEN_BYTES];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

/// Every `webscout_session` value on a request.
///
/// A browser folds all its cookies into one `Cookie` header separated by `; `,
/// and an HTTP/2 client may split them over several headers; both are read.
/// Every match is returned because a stale cookie for another path or an old
/// deploy can sit next to the live one.
pub fn session_cookies(headers: &HeaderMap) -> impl Iterator<Item = &str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|pair| {
            let (name, value) = pair.trim().split_once('=')?;
            let value = value.trim().trim_matches('"');
            (name.trim() == COOKIE_NAME && !value.is_empty()).then_some(value)
        })
}

/// Whether the request reached the platform's proxy over TLS.
///
/// Railway and DigitalOcean terminate TLS and forward plain HTTP with
/// `X-Forwarded-Proto: https`. `Secure` is only set then: a cookie marked
/// `Secure` on a plain-HTTP localhost would never be sent back, and the login
/// would appear to do nothing.
pub fn over_https(headers: &HeaderMap) -> bool {
    headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .is_some_and(|p| p.trim().eq_ignore_ascii_case("https"))
}

/// The `Set-Cookie` value that opens a session.
pub fn session_cookie(token: &str, secure: bool) -> String {
    format!(
        "{COOKIE_NAME}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={}{}",
        SESSION_TTL.as_secs(),
        if secure { "; Secure" } else { "" }
    )
}

/// The `Set-Cookie` value that removes the session cookie.
pub fn clear_cookie() -> String {
    format!("{COOKIE_NAME}=; Max-Age=0; Path=/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cookies(values: &[&str]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for v in values {
            h.append(header::COOKIE, v.parse().unwrap());
        }
        h
    }

    #[test]
    fn secret_eq_compares_whole_values_of_any_length() {
        assert!(secret_eq(b"hunter2", b"hunter2"));
        assert!(!secret_eq(b"hunter3", b"hunter2"));
        assert!(!secret_eq(b"hunter", b"hunter2"));
        assert!(!secret_eq(b"hunter22", b"hunter2"));
        assert!(!secret_eq(b"", b"hunter2"));
    }

    #[test]
    fn tokens_are_64_hex_characters_and_distinct() {
        let a = new_token().unwrap();
        let b = new_token().unwrap();
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn the_session_cookie_is_found_among_others_and_across_headers() {
        let h = cookies(&["theme=dark; webscout_session=abc; lang=en"]);
        assert_eq!(session_cookies(&h).collect::<Vec<_>>(), ["abc"]);
        let h = cookies(&["theme=dark", "webscout_session=def"]);
        assert_eq!(session_cookies(&h).collect::<Vec<_>>(), ["def"]);
        let h = cookies(&["webscout_session=; x=1", "webscout_sessionx=zzz"]);
        assert_eq!(session_cookies(&h).count(), 0);
        let h = cookies(&["webscout_session=old; webscout_session=new"]);
        assert_eq!(session_cookies(&h).collect::<Vec<_>>(), ["old", "new"]);
    }

    #[test]
    fn secure_follows_the_forwarded_protocol() {
        let mut h = HeaderMap::new();
        assert!(!over_https(&h));
        h.insert("x-forwarded-proto", "HTTPS, http".parse().unwrap());
        assert!(over_https(&h));
        h.insert("x-forwarded-proto", "http".parse().unwrap());
        assert!(!over_https(&h));
        assert!(session_cookie("t", true).ends_with("; Secure"));
        assert_eq!(
            session_cookie("t", false),
            "webscout_session=t; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000"
        );
    }

    #[test]
    fn a_blank_or_quoted_password_is_read_like_other_config() {
        assert!(!Auth::new(None).required());
        assert!(!Auth::new(Some("  ".into())).required());
        assert!(!Auth::new(Some("''".into())).required());
        let a = Auth::new(Some("'pw'".into()));
        assert_eq!(
            a.login("pw").map(|l| matches!(l, Login::Ok(_))).ok(),
            Some(true)
        );
        assert_eq!(Auth::new(None).login("x").unwrap(), Login::Open);
    }

    #[test]
    fn sessions_open_validate_and_close() {
        let a = Auth::new(Some("pw".into()));
        let Login::Ok(token) = a.login("pw").unwrap() else {
            panic!("login failed")
        };
        assert!(a.valid(&token));
        assert!(!a.valid("nope"));
        let h = cookies(&[&format!("webscout_session={token}")]);
        assert!(a.authenticated(&h));
        a.logout(&h);
        assert!(!a.valid(&token));
    }

    #[test]
    fn the_eleventh_attempt_in_a_minute_is_refused_even_if_right() {
        let a = Auth::new(Some("pw".into()));
        for _ in 0..MAX_FAILURES {
            assert_eq!(a.login("wrong").unwrap(), Login::Wrong);
        }
        assert_eq!(a.login("pw").unwrap(), Login::Throttled);
    }
}
