//! Read queries over a repository snapshot, and commits made without a
//! push, for an API in front of Enroute.
//!
//! Each function keeps the results, order, limits and truncation flags that
//! the contract promises. Commit graph walks run on the blocking pool. Object
//! reads go through machines, so no thread waits on the object store.
//!
//! - In a repository created by a push, `HEAD` moves to another branch when
//!   its branch is deleted.
//! - Merge bases from one push are ordered by object id, not by the order
//!   the push numbered them in (bases from different pushes keep push order).

mod commit;
mod diff_content;
mod refs;

use std::collections::BTreeSet;
use std::ops::Bound;

use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::fmt;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use crate::compute::{Compute, is_not_found};
use crate::git::object::{Signature, parse_commit, tree_entries};
use crate::git::oid::{Kind, ObjectFormat, Oid};
use crate::repo::graph::{first_parents, paint_down_to_common, reaches};
use crate::repo::machine::peel as read_peeled;
use crate::repo::serve::{FetchArgs, Plan, Sink, plan, write_pack};
use crate::repo::view::{DenseId, Snapshot};

pub(crate) use self::commit::{Expected, FileMode, NewCommit, PathChange, PathEdit, create_commit};
pub(crate) use self::refs::RefView;

/// Commits in one history page, at most.
pub(crate) const COMMIT_PAGE_LIMIT: u32 = 100;

/// Entries in one tree listing, at most.
const TREE_ENTRY_LIMIT: usize = 50_000;

/// Changed paths in one diff, at most.
pub(crate) const DIFF_CHANGE_LIMIT: usize = 50_000;

/// Trees one diff reads before it stops and reports what it has.
const DIFF_TREE_READ_LIMIT: usize = 10_000;

/// Bytes of one blob a diff reads to hunk or sketch it, at most.
const DIFF_BLOB_LIMIT: u64 = 16 * 1024 * 1024;

/// Blob bytes one diff reads to hunk or sketch, at most. Past it, changes
/// are still listed, with their content skipped.
const DIFF_CONTENT_READ_LIMIT: u64 = 256 * 1024 * 1024;

/// Blob bytes a diff holds at once to hunk or sketch them. The largest pair
/// one change can read fits.
const DIFF_CONTENT_CHUNK_BYTES: u64 = 2 * DIFF_BLOB_LIMIT;

/// Commits one merge-base walk reads before it stops.
const MERGE_BASE_MAX_COMMITS: u64 = 250_000;

/// Why a query, a ref update or a commit has no answer.
#[derive(Debug)]
pub enum Error {
    /// The repository is not there.
    NotFound,
    /// The repository has no object with this id.
    Missing(Oid),
    /// The object is not of a kind the call takes.
    WrongKind {
        /// The object the caller named.
        oid: Oid,
        /// What it is.
        kind: Kind,
        /// What the call takes in its place, such as "a diff is of a commit".
        role: &'static str,
    },
    /// The request does not describe something git takes.
    Invalid(String),
    /// The branch is not where the request expected it.
    Moved(String),
    /// Other writers kept moving what the call changes: "the branch" or
    /// "the refs".
    Contended(&'static str),
    /// Storage or the ledger failed, or disagreed with itself.
    Failed(anyhow::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound => f.write_str("no such repository"),
            Error::Missing(oid) => write!(f, "no object {oid}"),
            Error::WrongKind { oid, kind, role } => {
                write!(f, "{oid} is a {}, and {role}", kind.name())
            }
            Error::Invalid(why) | Error::Moved(why) => f.write_str(why),
            Error::Contended(what) => write!(f, "other writers kept moving {what}"),
            Error::Failed(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Self {
        if is_not_found(&e) {
            Error::NotFound
        } else {
            Error::Failed(e)
        }
    }
}

/// `repo` as of every write committed before the call.
///
/// # Errors
///
/// [`Error::NotFound`] if the repository is not there.
pub(crate) async fn snapshot(compute: &Compute, repo: &str) -> Result<Arc<Snapshot>, Error> {
    Ok(compute
        .repo_for_read(repo, true)
        .await?
        .ok_or(Error::NotFound)?
        .snapshot())
}

/// Whether `descendant` is `ancestor` or has it in its history; `false` when
/// either is not a commit the repository stores.
///
/// # Errors
///
/// The commit graph disagrees with the object tables.
pub(crate) async fn is_ancestor(
    snap: &Arc<Snapshot>,
    ancestor: Oid,
    descendant: Oid,
) -> Result<bool> {
    blocking(snap, move |s| {
        let (Some(a), Some(d)) = (
            s.lookup_kind(&ancestor, Kind::Commit),
            s.lookup_kind(&descendant, Kind::Commit),
        ) else {
            return Ok(false);
        };
        reaches(s, [d], a)
    })
    .await
}

/// The best common ancestors of two commits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MergeBases {
    /// Newest first; empty when the histories share nothing.
    pub bases: Vec<Oid>,
    /// The walk stopped at its budget, so `bases` may be incomplete.
    pub exhausted: bool,
}

/// `git merge-base --all` of `a` and `b`, reading at most `max_commits`
/// commits; nothing when either is not a stored commit.
///
/// # Errors
///
/// The commit graph disagrees with the object tables.
pub(crate) async fn merge_bases(snap: &Arc<Snapshot>, a: Oid, b: Oid) -> Result<MergeBases> {
    blocking(snap, move |s| {
        let (Some(a), Some(b)) = (
            s.lookup_kind(&a, Kind::Commit),
            s.lookup_kind(&b, Kind::Commit),
        ) else {
            return Ok(MergeBases::default());
        };
        let budget = usize::try_from(MERGE_BASE_MAX_COMMITS).unwrap_or(usize::MAX);
        let (mut found, exhausted) = paint_down_to_common(s, a, b, budget)?;
        // Ids are given in push order, so newer commits sort first.
        found.sort_unstable_by(|x, y| y.cmp(x));
        Ok(MergeBases {
            bases: found.into_iter().map(|d| s.oid(d)).collect(),
            exhausted,
        })
    })
    .await
}

/// Someone a commit names, as git stored it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Identity {
    /// Raw bytes: git does not require UTF-8.
    pub name: Vec<u8>,
    /// Raw bytes, as `name`.
    pub email: Vec<u8>,
    /// Seconds since the epoch.
    pub seconds: i64,
    /// The time zone, in seconds east of UTC.
    pub offset_seconds: i32,
}

impl Identity {
    /// The identity as a commit writes it.
    fn signature(&self) -> Signature<'_> {
        Signature {
            name: &self.name,
            email: &self.email,
            seconds: self.seconds,
            offset_seconds: self.offset_seconds,
        }
    }
}

impl From<Signature<'_>> for Identity {
    fn from(who: Signature<'_>) -> Self {
        Identity {
            name: who.name.to_vec(),
            email: who.email.to_vec(),
            seconds: who.seconds,
            offset_seconds: who.offset_seconds,
        }
    }
}

/// A commit, parsed for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommitDetails {
    /// The commit's own id.
    pub id: Oid,
    /// Its root tree.
    pub tree: Oid,
    /// In the order the commit gives them.
    pub parents: Vec<Oid>,
    /// Who wrote the change.
    pub author: Identity,
    /// Who applied it.
    pub committer: Identity,
    /// The first line of the message.
    pub summary: Vec<u8>,
    /// The rest of the message, less one blank line after the summary.
    pub body: Vec<u8>,
}

/// One page of first-parent history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommitPage {
    /// Newest first.
    pub commits: Vec<CommitDetails>,
    /// Where the next page starts; `None` at the end of history.
    pub next: Option<Oid>,
}

/// The page size for a request that asked for `asked`: zero means `max`.
#[must_use]
pub(crate) fn page_limit(asked: u32, max: u32) -> u32 {
    if asked == 0 { max } else { max.min(asked) }
}

/// `tip` and its first parents, at most `limit` (see [`page_limit`]) commits.
///
/// # Errors
///
/// `tip` is not a stored commit, or a read failed.
pub(crate) async fn list_commits(
    snap: &Arc<Snapshot>,
    tip: Oid,
    limit: u32,
) -> Result<CommitPage, Error> {
    let limit = usize::try_from(page_limit(limit, COMMIT_PAGE_LIMIT)).unwrap_or(usize::MAX);
    let walked = blocking(snap, move |s| match s.lookup_kind(&tip, Kind::Commit) {
        Some(d) => first_parents(s, d, limit.saturating_add(1)),
        None => Ok(Vec::new()),
    })
    .await?;
    if walked.is_empty() {
        return Err(not_a_commit(snap, tip, "a history starts at a commit"));
    }
    let page = walked.get(..limit).unwrap_or(walked.as_slice());
    let read = snap.load_many(page.iter().copied()).await?;
    let mut commits = Vec::with_capacity(page.len());
    for d in page {
        let data = read.get(d).context("commit not read")?;
        commits.push(commit_details(snap.oid(*d), data)?);
    }
    Ok(CommitPage {
        commits,
        next: walked.get(limit).map(|d| snap.oid(*d)),
    })
}

/// Why `oid`, which the call takes as `role`, cannot start a walk over
/// commits.
fn not_a_commit(snap: &Snapshot, oid: Oid, role: &'static str) -> Error {
    match snap.lookup(&oid) {
        None => Error::Missing(oid),
        Some(d) => match snap.kind(d) {
            Kind::Commit => Error::Failed(anyhow!("commit {oid} has no graph node")),
            kind => Error::WrongKind { oid, kind, role },
        },
    }
}

/// An object's kind and content.
///
/// # Errors
///
/// The object is not stored, or a read failed.
pub(crate) async fn get_object(
    snap: &Arc<Snapshot>,
    oid: Oid,
) -> Result<(Kind, Arc<Vec<u8>>), Error> {
    let d = snap.lookup(&oid).ok_or(Error::Missing(oid))?;
    Ok((snap.kind(d), snap.load(d).await?))
}

/// One path a tree listing names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TreeItem {
    /// From the listing's root, `/`-joined, in git's raw bytes.
    pub path: Vec<u8>,
    /// What the entry names.
    pub oid: Oid,
    /// A directory, not a file.
    pub is_tree: bool,
}

/// One page of the paths under a tree, one level per depth.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TreeListing {
    /// Breadth first, each level sorted by path.
    pub levels: Vec<Vec<TreeItem>>,
    /// The page stopped at its entry limit with more to name.
    pub truncated: bool,
}

/// Paths under `start` (a tree, or a commit's root tree) that come after
/// `start_after`, breadth first, at most [`TREE_ENTRY_LIMIT`] of them;
/// gitlinks are left out.
///
/// A depth's entries are sorted by their path as the API sends it, so a
/// listing is in order of depth, then path, and the last path of one page is
/// where the next starts. A page does not end between paths that read the
/// same, which a lossy decoding can make of different names. Each page walks
/// again the levels above its first entry, so it holds no state between
/// calls.
///
/// # Errors
///
/// `start` is not a stored tree or commit, or a read failed.
pub(crate) async fn list_tree(
    snap: &Arc<Snapshot>,
    start: Oid,
    start_after: &str,
) -> Result<TreeListing, Error> {
    let root = root_tree(snap, start).await?;
    // An empty `start_after` starts before every path, at depth 0.
    let start_depth = start_after.matches('/').count();
    let mut trees: HashMap<Oid, Vec<Child>> = HashMap::new();
    let mut walk = TreeListing::default();
    let mut named = 0usize;
    let mut directories: Vec<(Vec<u8>, Oid)> = vec![(Vec::new(), root)];
    for depth in 0.. {
        if directories.is_empty() {
            break;
        }
        let unread: Vec<Oid> = directories
            .iter()
            .map(|(_, oid)| *oid)
            .filter(|oid| !trees.contains_key(oid))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        read_trees(snap, &unread, &mut trees).await?;
        let mut entries: Vec<(String, TreeItem)> = directories
            .iter()
            .filter_map(|(prefix, oid)| Some((prefix, trees.get(oid)?)))
            .flat_map(|(prefix, children)| {
                children.iter().filter(|c| !c.is_commit).map(|c| {
                    let path = join(prefix, &c.name);
                    let item = TreeItem {
                        path,
                        oid: c.oid,
                        is_tree: c.is_tree,
                    };
                    (String::from_utf8_lossy(&item.path).into_owned(), item)
                })
            })
            .collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));
        directories = entries
            .iter()
            .filter(|(_, item)| item.is_tree)
            .map(|(_, item)| (item.path.clone(), item.oid))
            .collect();
        let listed = entries.into_iter().filter(|(key, _)| {
            depth > start_depth || depth == start_depth && key.as_str() > start_after
        });
        let mut level = Vec::new();
        let mut last: Option<String> = None;
        for (key, item) in listed {
            if named >= TREE_ENTRY_LIMIT && last.as_ref() != Some(&key) {
                walk.levels.push(level);
                walk.truncated = true;
                return Ok(walk);
            }
            named += 1;
            level.push(item);
            last = Some(key);
        }
        if !level.is_empty() {
            walk.levels.push(level);
        }
    }
    Ok(walk)
}

/// The tree a listing starts at.
async fn root_tree(snap: &Arc<Snapshot>, start: Oid) -> Result<Oid, Error> {
    let d = snap.lookup(&start).ok_or(Error::Missing(start))?;
    match snap.kind(d) {
        Kind::Tree => Ok(start),
        Kind::Commit => Ok(parse_commit(&snap.load(d).await?)?.tree),
        kind => Err(Error::WrongKind {
            oid: start,
            kind,
            role: "a listing starts at a commit or a tree",
        }),
    }
}

/// One file a diff found changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FileChange {
    /// From the root, `/`-joined, in git's raw bytes.
    pub path: Vec<u8>,
    /// The blob before; `None` when the path is new.
    pub old: Option<ChangedBlob>,
    /// The blob after; `None` when the path is gone.
    pub new: Option<ChangedBlob>,
    /// With [`DiffOptions::hunks`].
    pub hunks: Vec<Hunk>,
    /// Hunks or a sketch were asked for and not computed.
    pub content_skipped: bool,
    /// With [`DiffOptions::sketch_hashes`], for an add or a delete.
    pub sketch: Option<LineSketch>,
}

/// One side of a [`FileChange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChangedBlob {
    pub oid: Oid,
    pub size: u64,
}

pub(crate) use diff_content::{Hunk, LineSketch, SKETCH_VERSION};

/// What a diff reports beyond the changed paths.
#[derive(Debug, Clone, Default)]
pub(crate) struct DiffOptions {
    /// Only these paths, and only the trees that hold them.
    pub paths: Option<PathFilter>,
    /// Line hunks, skipping blobs past this many bytes.
    pub hunks: Option<u64>,
    /// Version 1 line sketches of added and deleted files, with this many hashes.
    pub sketch_hashes: Option<usize>,
}

impl DiffOptions {
    /// Clamps the caller's sketch size to the server's limit.
    pub(crate) fn sketch_size(k: u32) -> usize {
        usize::try_from(k)
            .unwrap_or(usize::MAX)
            .clamp(1, diff_content::MAX_SKETCH_HASHES)
    }
}

/// A set of file paths, and the directories that hold them.
#[derive(Debug, Clone, Default)]
pub(crate) struct PathFilter(BTreeSet<Vec<u8>>);

impl PathFilter {
    pub(crate) fn new(paths: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self(paths.into_iter().collect())
    }

    fn has_file(&self, path: &[u8]) -> bool {
        self.0.contains(path)
    }

    fn has_under(&self, directory: &[u8]) -> bool {
        // Siblings that continue `directory` with a byte below `/`, such as
        // `directory.rs`, sort between it and the paths under it.
        self.0
            .range::<[u8], _>((Bound::Included(directory), Bound::Unbounded))
            .take_while(|path| path.starts_with(directory))
            .find_map(|path| match path.get(directory.len()) {
                Some(b'/') => Some(true),
                Some(byte) if *byte > b'/' => Some(false),
                _ => None,
            })
            .unwrap_or(false)
    }
}

/// The files that differ between two commits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CommitDiff {
    /// Sorted by path, as UTF-8 with invalid bytes replaced.
    pub changes: Vec<FileChange>,
    /// The diff stopped at a limit, so `changes` is a prefix of the answer.
    pub truncated: bool,
}

/// The files that differ between `commit` and `base`, or its first parent
/// when there is no base (a root commit is diffed against nothing).
///
/// No renames, gitlinks or mode-only changes; a path that turns from file to
/// directory, or back, is both a removal and an addition.
///
/// # Errors
///
/// `commit` or `base` is not a stored commit, or a read failed.
pub(crate) async fn diff_commit(
    snap: &Arc<Snapshot>,
    commit: Oid,
    base: Option<Oid>,
    options: DiffOptions,
) -> Result<CommitDiff, Error> {
    const OF: &str = "a diff is of a commit";
    let older = if let Some(base) = base {
        for (oid, role) in [(commit, OF), (base, "a diff is against a commit")] {
            if snap.lookup_kind(&oid, Kind::Commit).is_none() {
                return Err(not_a_commit(snap, oid, role));
            }
        }
        Some(base)
    } else {
        let walked = blocking(snap, move |s| match s.lookup_kind(&commit, Kind::Commit) {
            Some(d) => first_parents(s, d, 2),
            None => Ok(Vec::new()),
        })
        .await?;
        if walked.is_empty() {
            return Err(not_a_commit(snap, commit, OF));
        }
        walked.get(1).map(|d| snap.oid(*d))
    };
    let new_root = commit_tree(snap, commit).await?;
    let old_root = match older {
        Some(o) => Some(commit_tree(snap, o).await?),
        None => None,
    };
    let filter = options.paths.clone().map(Arc::new);
    let diff = diff_trees(snap, old_root, new_root, filter.clone()).await?;
    let (paths, truncated) = changed_files(diff, filter.as_deref());
    let changes = add_content(snap, paths, &options).await?;
    Ok(CommitDiff { changes, truncated })
}

/// Reads each path's blob sizes, and hunks and sketches when `options` asks
/// for them.
async fn add_content(
    snap: &Arc<Snapshot>,
    paths: Vec<ChangedPath>,
    options: &DiffOptions,
) -> Result<Vec<FileChange>> {
    let blob_limit = options
        .hunks
        .filter(|limit| *limit > 0)
        .map_or(DIFF_BLOB_LIMIT, |limit| limit.min(DIFF_BLOB_LIMIT));
    let want_hunks = options.hunks.is_some();
    let mut budget = DIFF_CONTENT_READ_LIMIT;
    // TODO: Report a blob that a tree names but the snapshot lacks as an
    // error. It now reads as empty, with size zero.
    let stored = |oid: Oid| {
        let stored = snap.lookup_kind(&oid, Kind::Blob);
        let size = stored.map_or(0, |d| snap.size(d));
        (ChangedBlob { oid, size }, stored)
    };
    let mut changes = Vec::with_capacity(paths.len());
    let mut chunk = Vec::new();
    let mut chunk_bytes = 0;
    for ChangedPath { path, old, new } in paths {
        let (old, new) = (old.map(stored), new.map(stored));
        let mut change = FileChange {
            path,
            old: old.map(|(blob, _)| blob),
            new: new.map(|(blob, _)| blob),
            ..FileChange::default()
        };
        let (old, new) = (old.and_then(|(_, d)| d), new.and_then(|(_, d)| d));
        let old_size = change.old.map_or(0, |blob| blob.size);
        let new_size = change.new.map_or(0, |blob| blob.size);
        let sketch = options
            .sketch_hashes
            .filter(|_| change.old.is_none() || change.new.is_none());
        let total = old_size + new_size;
        let wanted = want_hunks || sketch.is_some();
        if wanted && (old_size > blob_limit || new_size > blob_limit || total > budget) {
            change.content_skipped = true;
        } else if wanted {
            budget -= total;
            if chunk_bytes + total > DIFF_CONTENT_CHUNK_BYTES {
                read_content(snap, &mut changes, std::mem::take(&mut chunk), want_hunks).await?;
                chunk_bytes = 0;
            }
            chunk_bytes += total;
            chunk.push(ContentRead {
                change: changes.len(),
                old,
                new,
                sketch,
            });
        }
        changes.push(change);
    }
    read_content(snap, &mut changes, chunk, want_hunks).await?;
    Ok(changes)
}

/// One change whose blobs [`read_content`] reads.
struct ContentRead {
    /// Its index in the changes.
    change: usize,
    old: Option<DenseId>,
    new: Option<DenseId>,
    sketch: Option<usize>,
}

/// Loads a chunk's blobs together, then computes their hunks and sketches on
/// one blocking thread, so a diff of many files waits on neither one load nor
/// one task per file.
async fn read_content(
    snap: &Arc<Snapshot>,
    changes: &mut [FileChange],
    chunk: Vec<ContentRead>,
    want_hunks: bool,
) -> Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }
    let blobs = snap
        .load_many(
            chunk
                .iter()
                .flat_map(|read| read.old.into_iter().chain(read.new)),
        )
        .await?;
    let computed = tokio::task::spawn_blocking(move || {
        let empty = Arc::new(Vec::new());
        let data = |d: Option<DenseId>| match d {
            Some(d) => blobs.get(&d).cloned().context("blob not read"),
            None => Ok(empty.clone()),
        };
        chunk
            .into_iter()
            .map(|read| {
                let (before, after) = (data(read.old)?, data(read.new)?);
                Ok((
                    read.change,
                    content(&before, &after, want_hunks, read.sketch),
                ))
            })
            .collect::<Result<Vec<_>>>()
    })
    .await
    .context("diff content task")??;
    for (index, computed) in computed {
        let Some(change) = changes.get_mut(index) else {
            continue;
        };
        match computed {
            Some((hunks, sketch)) => {
                change.hunks = hunks;
                change.sketch = sketch;
            }
            None => change.content_skipped = true,
        }
    }
    Ok(())
}

/// The hunks and sketch of one change's blobs, or `None` when either is not
/// text. An empty side is an added or deleted file.
fn content(
    before: &[u8],
    after: &[u8],
    want_hunks: bool,
    sketch: Option<usize>,
) -> Option<(Vec<Hunk>, Option<LineSketch>)> {
    if !diff_content::is_text(before) || !diff_content::is_text(after) {
        return None;
    }
    let hunks = if want_hunks {
        diff_content::line_hunks(before, after)
    } else {
        Vec::new()
    };
    let sketched = if after.is_empty() { before } else { after };
    Some((
        hunks,
        sketch.map(|k| diff_content::line_sketch(sketched, k)),
    ))
}

/// The root tree of a stored commit.
async fn commit_tree(snap: &Arc<Snapshot>, commit: Oid) -> Result<Oid> {
    let d = snap
        .lookup_kind(&commit, Kind::Commit)
        .with_context(|| format!("commit {commit} missing"))?;
    Ok(parse_commit(&snap.load(d).await?)?.tree)
}

/// A tree entry, as diffs and listings use it.
#[derive(Debug, Clone)]
struct Child {
    oid: Oid,
    is_tree: bool,
    is_commit: bool,
    name: Vec<u8>,
}

/// The entries of a tree of a `format` repository; none for one that does
/// not parse, as before.
fn children(format: ObjectFormat, data: &[u8]) -> Vec<Child> {
    tree_entries(format, data)
        .map(|e| {
            e.map(|e| Child {
                oid: e.oid,
                is_tree: e.mode.is_tree(),
                is_commit: e.mode.is_gitlink(),
                name: e.name.to_vec(),
            })
        })
        .collect::<Result<Vec<_>>>()
        .unwrap_or_default()
}

/// Read `oids` (trees) into `into`, loading what is not local.
async fn read_trees(
    snap: &Arc<Snapshot>,
    oids: &[Oid],
    into: &mut HashMap<Oid, Vec<Child>>,
) -> Result<()> {
    for (oid, data) in read_tree_data(snap, oids).await? {
        into.insert(oid, children(snap.format(), &data));
    }
    Ok(())
}

/// The data of the trees `oids`, each once, loading what is not local.
async fn read_tree_data(snap: &Arc<Snapshot>, oids: &[Oid]) -> Result<HashMap<Oid, Arc<Vec<u8>>>> {
    let ds = oids
        .iter()
        .map(|o| {
            snap.lookup_kind(o, Kind::Tree)
                .with_context(|| format!("tree {o} is missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let read = snap.load_many(ds).await?;
    Ok(read
        .into_iter()
        .map(|(d, data)| (snap.oid(d), data))
        .collect())
}

/// One pass of a diff over the trees read so far.
#[derive(Debug, Default)]
struct TreeDiff {
    /// Paths whose content differs: (path, is a tree, old, new).
    changes: Vec<(Vec<u8>, bool, Option<Oid>, Oid)>,
    /// Paths that are gone: (path, was a tree, old).
    removals: Vec<(Vec<u8>, bool, Oid)>,
    /// Trees the pass needed and did not have, each once.
    unreadable: Vec<Oid>,
}

/// Diff two trees, reading what each pass lacked, until a pass needs nothing
/// more or [`DIFF_TREE_READ_LIMIT`] trees are read; the rounds decide where a
/// diff stops.
async fn diff_trees(
    snap: &Arc<Snapshot>,
    old_root: Option<Oid>,
    new_root: Oid,
    filter: Option<Arc<PathFilter>>,
) -> Result<TreeDiff> {
    let mut read: HashMap<Oid, Vec<Child>> = HashMap::new();
    loop {
        let filter = filter.clone();
        let (diff, back) = tokio::task::spawn_blocking(move || {
            let diff = diff_pass(&read, old_root, new_root, filter.as_deref());
            (diff, read)
        })
        .await?;
        read = back;
        if diff.unreadable.is_empty() || read.len() >= DIFF_TREE_READ_LIMIT {
            return Ok(diff);
        }
        read_trees(snap, &diff.unreadable, &mut read).await?;
    }
}

/// One pass: every path that differs under the trees in `read`, removals
/// included, and the trees it could not descend into.
fn diff_pass(
    read: &HashMap<Oid, Vec<Child>>,
    old_root: Option<Oid>,
    new_root: Oid,
    filter: Option<&PathFilter>,
) -> TreeDiff {
    let mut out = TreeDiff::default();
    if old_root == Some(new_root) {
        return out;
    }
    let mut stack: Vec<(Vec<u8>, Option<Oid>, Option<Oid>)> =
        vec![(Vec::new(), old_root, Some(new_root))];
    while let Some((prefix, old, new)) = stack.pop() {
        let mut fetch = |oid: Option<Oid>| match oid {
            None => Some(&[][..]),
            Some(o) => {
                let got = read.get(&o).map(Vec::as_slice);
                if got.is_none() {
                    out.unreadable.push(o);
                }
                got
            }
        };
        let old_entries = fetch(old);
        let new_entries = fetch(new);
        let (Some(old_entries), Some(new_entries)) = (old_entries, new_entries) else {
            continue;
        };
        let mut previous: HashMap<&[u8], &Child> = old_entries
            .iter()
            .filter(|e| !e.is_commit)
            .map(|e| (e.name.as_slice(), e))
            .collect();
        for entry in new_entries.iter().filter(|e| !e.is_commit) {
            let was = previous.remove(entry.name.as_slice());
            if was.is_some_and(|w| w.oid == entry.oid && w.is_tree == entry.is_tree) {
                continue;
            }
            let path = join(&prefix, &entry.name);
            let old_oid = was.filter(|w| w.is_tree == entry.is_tree).map(|w| w.oid);
            if let Some(gone) = was.filter(|w| w.is_tree != entry.is_tree) {
                removed(&mut out, &mut stack, path.clone(), gone, filter);
            }
            if entry.is_tree && filter.is_none_or(|f| f.has_under(&path)) {
                stack.push((path.clone(), old_oid, Some(entry.oid)));
            }
            out.changes.push((path, entry.is_tree, old_oid, entry.oid));
        }
        let mut gone: Vec<&Child> = previous.into_values().collect();
        gone.sort_by(|a, b| a.name.cmp(&b.name));
        for entry in gone {
            removed(
                &mut out,
                &mut stack,
                join(&prefix, &entry.name),
                entry,
                filter,
            );
        }
    }
    let mut named = HashSet::new();
    out.unreadable.retain(|o| named.insert(*o));
    out
}

/// Record a removed path, and queue a removed tree so its paths are named.
fn removed(
    out: &mut TreeDiff,
    stack: &mut Vec<(Vec<u8>, Option<Oid>, Option<Oid>)>,
    path: Vec<u8>,
    entry: &Child,
    filter: Option<&PathFilter>,
) {
    if entry.is_tree && filter.is_none_or(|f| f.has_under(&path)) {
        stack.push((path.clone(), Some(entry.oid), None));
    }
    out.removals.push((path, entry.is_tree, entry.oid));
}

/// A file a diff found changed, before its blobs are read.
struct ChangedPath {
    path: Vec<u8>,
    old: Option<Oid>,
    new: Option<Oid>,
}

/// The files of a diff that `filter` keeps, sorted, then cut at the change
/// limit, and whether the cut or an unread tree left some out.
fn changed_files(diff: TreeDiff, filter: Option<&PathFilter>) -> (Vec<ChangedPath>, bool) {
    let complete = diff.unreadable.is_empty();
    let added = diff
        .changes
        .into_iter()
        .filter(|(_, is_tree, _, _)| !is_tree)
        .map(|(path, _, old, new)| ChangedPath {
            path,
            old,
            new: Some(new),
        });
    let gone = diff
        .removals
        .into_iter()
        .filter(|(_, is_tree, _)| !is_tree)
        .map(|(path, _, old)| ChangedPath {
            path,
            old: Some(old),
            new: None,
        });
    let mut changes: Vec<ChangedPath> = added
        .chain(gone)
        .filter(|change| filter.is_none_or(|f| f.has_file(&change.path)))
        .collect();
    // The API sends paths as strings, and sorted them as strings.
    changes.sort_by_cached_key(|c| String::from_utf8_lossy(&c.path).into_owned());
    let truncated = changes.len() > DIFF_CHANGE_LIMIT || !complete;
    changes.truncate(DIFF_CHANGE_LIMIT);
    (changes, truncated)
}

/// `name` under `prefix`.
fn join(prefix: &[u8], name: &[u8]) -> Vec<u8> {
    let mut path = Vec::with_capacity(prefix.len() + 1 + name.len());
    if !prefix.is_empty() {
        path.extend_from_slice(prefix);
        path.push(b'/');
    }
    path.extend_from_slice(name);
    path
}

/// A commit, with its message split into a summary and a body.
fn commit_details(id: Oid, data: &[u8]) -> Result<CommitDetails> {
    let commit = parse_commit(data)?;
    let message = commit.message;
    let end = message
        .iter()
        .position(|b| *b == b'\n')
        .unwrap_or(message.len());
    let rest = message.get(end + 1..).unwrap_or_default();
    Ok(CommitDetails {
        id,
        tree: commit.tree,
        parents: commit.parents,
        author: commit.author.into(),
        committer: commit.committer.into(),
        summary: message.get(..end).unwrap_or_default().to_vec(),
        body: rest.strip_prefix(b"\n").unwrap_or(rest).to_vec(),
    })
}

/// `items` without repeats, in order.
fn unique<T: Copy + Eq + std::hash::Hash>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut seen = HashSet::new();
    items.into_iter().filter(|x| seen.insert(*x)).collect()
}

/// Run `f` on the blocking pool: graph walks are CPU work.
async fn blocking<T: Send + 'static>(
    snap: &Arc<Snapshot>,
    f: impl FnOnce(&Snapshot) -> Result<T> + Send + 'static,
) -> Result<T> {
    let snap = snap.clone();
    tokio::task::spawn_blocking(move || f(&snap)).await?
}

/// A pack for a push to a remote, planned and ready to write.
pub struct PushPack {
    snap: Arc<Snapshot>,
    plan: Plan,
}

impl fmt::Debug for PushPack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PushPack")
            .field("objects", &self.objects())
            .finish_non_exhaustive()
    }
}

impl PushPack {
    /// The number of objects in the pack.
    #[must_use]
    pub fn objects(&self) -> usize {
        self.plan.objects.len()
    }

    /// Write the whole pack (header, entries, trailer) to `sink`.
    ///
    /// # Errors
    ///
    /// Storage failed, or `sink` did.
    pub async fn write(self, sink: Sink) -> Result<()> {
        write_pack(self.snap, self.plan, sink).await
    }
}

/// The pack a git `receive-pack` needs to move a remote's refs to `wants`,
/// when the remote holds `haves`: every object reachable from `wants` and
/// not from `haves`, as `git send-pack` sends it.
///
/// Haves this repository does not hold are ignored, and tags among them
/// count for what they point at. The pack is never thin: every delta has its
/// base in the pack. Deltas name their bases by object id, as every
/// receiver takes. A push of deletions only sends no pack: that is the
/// caller's choice, as this always plans one.
///
/// # Errors
///
/// [`Error::Missing`] if a want is not in the repository.
pub async fn pack_for_push(
    snap: &Arc<Snapshot>,
    wants: &[Oid],
    haves: &[Oid],
) -> Result<PushPack, Error> {
    if let Some(w) = wants.iter().find(|w| snap.lookup(w).is_none()) {
        return Err(Error::Missing(*w));
    }
    let held: Vec<DenseId> = haves.iter().filter_map(|h| snap.lookup(h)).collect();
    let mut have_oids: Vec<Oid> = haves.to_vec();
    for p in read_peeled(snap, held).await? {
        have_oids.extend(p.end.map(|d| snap.oid(d)));
    }
    let have_set: HashSet<Oid> = have_oids.iter().copied().collect();
    let args = FetchArgs {
        // A want the remote has needs nothing sent, as with git.
        wants: unique(wants.iter().copied().filter(|w| !have_set.contains(w))),
        haves: have_oids,
        ..Default::default()
    };
    let plan = if args.wants.is_empty() {
        Plan::default()
    } else {
        plan(snap, &args).await?
    };
    Ok(PushPack {
        snap: snap.clone(),
        plan,
    })
}
