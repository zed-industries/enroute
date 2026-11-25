//! Line hunks and line sketches of a diff's blobs.
//!
//! Both are API output, so their definitions are fixed: a sketch changes only
//! with a new version.

use std::ops::Range;

use imara_diff::{Algorithm, Diff, InternedInput, sources::byte_lines};
use xxhash_rust::xxh3::xxh3_64;

/// The only sketch definition so far.
pub(crate) const SKETCH_VERSION: u32 = 1;

/// Hashes one sketch keeps at most.
pub(crate) const MAX_SKETCH_HASHES: usize = 1024;

/// Lines with fewer non-whitespace bytes nearly every file shares.
const MIN_LINE_CONTENT_BYTES: usize = 3;

/// A bottom-k `MinHash` sketch of a text's distinct lines, which estimates the
/// Jaccard index of two texts' line sets.
///
/// See <https://en.wikipedia.org/wiki/MinHash>.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LineSketch {
    pub smallest_hashes: Vec<u64>,
    pub distinct_lines: u32,
}

/// Bytes `old` of the old text became bytes `new` of the new text. Both are
/// whole lines, and either can be empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hunk {
    pub old: Range<u64>,
    pub new: Range<u64>,
}

/// Whether `bytes` are text: UTF-8 without NUL. Hunks and sketches skip the rest.
pub(crate) fn is_text(bytes: &[u8]) -> bool {
    !bytes.contains(&0) && std::str::from_utf8(bytes).is_ok()
}

/// Sketches `text` by version 1, keeping at most `k` hashes.
pub(crate) fn line_sketch(text: &[u8], k: usize) -> LineSketch {
    let mut hashes = text
        .split(|byte| *byte == b'\n')
        .map(<[u8]>::trim_ascii)
        .filter(|line| {
            line.iter()
                .filter(|byte| !byte.is_ascii_whitespace())
                .count()
                >= MIN_LINE_CONTENT_BYTES
        })
        .map(xxh3_64)
        .collect::<Vec<_>>();
    hashes.sort_unstable();
    hashes.dedup();
    let distinct_lines = u32::try_from(hashes.len()).unwrap_or(u32::MAX);
    hashes.truncate(k);
    LineSketch {
        smallest_hashes: hashes,
        distinct_lines,
    }
}

/// Myers line hunks from `before` to `after`, as byte ranges of whole lines.
pub(crate) fn line_hunks(before: &[u8], after: &[u8]) -> Vec<Hunk> {
    let (before_offsets, after_offsets) = (line_offsets(before), line_offsets(after));
    let (before_lines, after_lines) =
        (lines(before, &before_offsets), lines(after, &after_offsets));
    let mut tokens = InternedInput::new(&[][..], &[][..]);
    tokens.update_before(before_lines.into_iter());
    tokens.update_after(after_lines.into_iter());
    let at = |offsets: &[usize], token: u32| {
        let offset = usize::try_from(token)
            .ok()
            .and_then(|token| offsets.get(token).copied())
            .unwrap_or_default();
        u64::try_from(offset).unwrap_or(u64::MAX)
    };
    Diff::compute(Algorithm::Myers, &tokens)
        .hunks()
        .map(|hunk| Hunk {
            old: at(&before_offsets, hunk.before.start)..at(&before_offsets, hunk.before.end),
            new: at(&after_offsets, hunk.after.start)..at(&after_offsets, hunk.after.end),
        })
        .collect()
}

/// Each line of `bytes`, by the offsets [`line_offsets`] found.
fn lines<'a>(bytes: &'a [u8], offsets: &[usize]) -> Vec<&'a [u8]> {
    offsets
        .windows(2)
        .map(|line| match line {
            [start, end] => bytes.get(*start..*end).unwrap_or_default(),
            _ => &[],
        })
        .collect()
}

/// The start of each line, then the end of the text.
fn line_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut offsets = vec![0];
    let mut offset = 0;
    for line in byte_lines(bytes) {
        offset += line.len();
        offsets.push(offset);
    }
    offsets
}
