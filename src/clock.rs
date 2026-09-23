//! What day it is, with no calendar crate.
//!
//! Every generative call in this pipeline is answering a question asked *today*,
//! and until this module existed none of them were told so. Measured cause: asked
//! "How many seasons of The White Lotus are available and when does the next one
//! come out?" with no evidence and no date, `google/gemma-4-31b-it` replies "Two
//! seasons ... and the third season is expected to be released in 2025." The model
//! is answering from its training data because nothing in the request contradicts
//! it. The only `now()` calls in the crate were search-cache expiry and retry
//! jitter, so "latest", "next" and "current" were resolved against whenever the
//! weights were frozen.
//!
//! One `SystemTime::now()` per run, formatted once, threaded through the stages.
//! Calling the clock inside a loop would be both wasteful and capable of straddling
//! midnight mid-run, which would put two different dates into one report.
//!
//! Adding `chrono` or `time` for four lines of integer arithmetic is not worth a
//! dependency, so the civil-calendar conversion is inlined below.

use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds in a day. UTC has no leap seconds in `SystemTime`'s model, so this is
/// exact for the purpose of turning a Unix timestamp into a day number.
const SECS_PER_DAY: u64 = 86_400;

/// Today's date in UTC as `YYYY-MM-DD`.
///
/// UTC rather than local time: the sources being read are worldwide, the model is
/// remote, and a run that crosses midnight in one timezone should not disagree with
/// itself about what day it is. A clock set before the epoch (or unreadable) falls
/// back to day 0, which prints `1970-01-01` — obviously wrong rather than subtly
/// wrong, which is the failure mode to prefer here.
pub fn today_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_civil(civil_from_days((secs / SECS_PER_DAY) as i64))
}

/// Render a `(year, month, day)` triple as `YYYY-MM-DD`.
///
/// Split out from `today_utc` so the formatting is testable without the clock.
pub(crate) fn format_civil((y, m, d): (i64, u32, u32)) -> String {
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert a count of days since 1970-01-01 into a civil `(year, month, day)`.
///
/// Howard Hinnant's `civil_from_days`, the standard shift-the-epoch-to-March
/// algorithm: by treating March as the first month of the year, the leap day lands
/// at the end and the month-length sequence becomes the exactly linear
/// `(153 * mp + 2) / 5`. Valid across the whole proleptic Gregorian calendar,
/// including negative day numbers, so it is exact rather than approximately right
/// for the next few decades.
///
/// Reference: <https://howardhinnant.github.io/date_algorithms.html#civil_from_days>
pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the epoch from 1970-01-01 to 0000-03-01, which is 719_468 days earlier.
    let z = days + 719_468;
    // An era is 400 years = 146_097 days, the Gregorian cycle length. Floor-divide,
    // so dates before the shifted epoch land in the right (negative) era.
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    // Day of era, 0..=146_096.
    let doe = z - era * 146_097;
    // Year of era, 0..=399. The correction terms remove the 4-, 100- and 400-year
    // leap rules in that order.
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    // Day of year, 0..=365, counting from March 1.
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    // Month of the March-based year, 0..=11.
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    // Undo the March shift: months 0..=9 are March..December, 10 and 11 are the
    // January and February that belong to the *next* civil year.
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + i64::from(m <= 2), m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_day_zero_is_the_unix_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(format_civil(civil_from_days(0)), "1970-01-01");
    }

    #[test]
    fn known_epoch_2024_new_year() {
        // 54 years × 365 = 19_710 days, plus 13 leap days (1972..=2020).
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(format_civil(civil_from_days(19_723)), "2024-01-01");
    }

    #[test]
    fn leap_day_2024_02_29_exists_and_is_followed_by_march() {
        // 19_723 (2024-01-01) + 31 (January) + 28 = 19_782.
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(19_783), (2024, 3, 1));
        // And the day before is the 28th, so February 2024 has exactly 29 days.
        assert_eq!(civil_from_days(19_781), (2024, 2, 28));
    }

    #[test]
    fn century_case_2000_is_a_leap_year() {
        // 2000 is divisible by 400, so the 100-year rule does not apply.
        // 30 years × 365 = 10_950 + 7 leap days (1972..=1996) = 10_957 → 2000-01-01.
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(10_957 + 31 + 28), (2000, 2, 29));
        assert_eq!(civil_from_days(11_017), (2000, 3, 1));
    }

    #[test]
    fn century_case_1900_is_not_a_leap_year() {
        // The other half of the century rule, on the negative side of the epoch.
        // 1900-01-01 is 25_567 days before the epoch, so 1900-02-28 is -25_509.
        // The next day is March 1: there is no 1900-02-29, because the 100-year
        // rule applies and the 400-year exemption does not.
        assert_eq!(civil_from_days(-25_567), (1900, 1, 1));
        assert_eq!(civil_from_days(-25_509), (1900, 2, 28));
        assert_eq!(civil_from_days(-25_508), (1900, 3, 1));
    }

    #[test]
    fn year_boundaries_round_trip_both_ways() {
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(19_722), (2023, 12, 31));
        // 2024 is a leap year, so 2025-01-01 is 366 days after 2024-01-01.
        assert_eq!(civil_from_days(20_088), (2024, 12, 31));
        assert_eq!(civil_from_days(19_723 + 366), (2025, 1, 1));
    }

    #[test]
    fn format_pads_every_component() {
        assert_eq!(format_civil((2024, 1, 5)), "2024-01-05");
        assert_eq!(format_civil((999, 12, 31)), "0999-12-31");
    }

    #[test]
    fn today_utc_has_the_iso_shape_and_a_plausible_year() {
        let s = today_utc();
        assert_eq!(s.len(), 10, "expected YYYY-MM-DD, got {s}");
        let parts: Vec<&str> = s.split('-').collect();
        assert_eq!(parts.len(), 3);
        let y: i64 = parts[0].parse().expect("year parses");
        let m: u32 = parts[1].parse().expect("month parses");
        let d: u32 = parts[2].parse().expect("day parses");
        // Not a pinned date (the test would rot); a sanity window that still
        // catches an off-by-an-era arithmetic slip.
        assert!((2020..2200).contains(&y), "implausible year {y}");
        assert!((1..=12).contains(&m), "bad month {m}");
        assert!((1..=31).contains(&d), "bad day {d}");
    }
}
