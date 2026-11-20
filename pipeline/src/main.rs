//! Shopping Sessions: the fast pipeline. Every rule cited here is from rules.md.
//!
//! Plan:
//!   1. mmap every input file and cut all of them into line-aligned chunks.
//!   2. Links: union-find over the IDs; each person is named by its smallest ID.
//!   3. Events: parse + validate every copy in parallel, scattered by hash(event_id);
//!      dedup each shard (smallest ingest_ts wins, then the kept copy's validity
//!      decides), then scatter survivors by hash(person).
//!   4. Clicks: parse + validate, scattered by hash(person). No dedup is needed:
//!      copies are identical, so they cannot change a credit or an assist.
//!   5. Per person shard: sessionize, credit, assists, aggregate.
//!   6. Write Parquet.

use memchr::{memchr, memchr_iter};
use memmap2::Mmap;
use parquet::data_type::{ByteArray, ByteArrayType, DataType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
use parquet::schema::parser::parse_message_type;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::hash_map::Entry;
use std::fs::File;
use std::hash::BuildHasher;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const NB: usize = 256; // shards for both shuffles

// ------------------------------------------------------------------ hashing

#[inline]
fn hash_bytes(b: &[u8]) -> u64 {
    FxBuildHasher.hash_one(b)
}

/// Shard from the middle bits, so it does not correlate with the bits the
/// hash maps inside each shard use (low bits for position, top 7 for tags).
#[inline]
fn shard_of(b: &[u8]) -> usize {
    ((hash_bytes(b) >> 32) as usize) & (NB - 1)
}

fn new_map<K, V>(cap: usize) -> FxHashMap<K, V> {
    FxHashMap::with_capacity_and_hasher(cap, FxBuildHasher)
}

// ---------------------------------------------------------------------- CSV

/// Split one line (no '\n') into exactly N fields under the dialect:
/// quotes only around a whole field, never inside; empty field = null (empty slice).
#[inline]
fn split<'a, const N: usize>(line: &'a [u8], out: &mut [&'a [u8]; N]) -> bool {
    if line.is_empty() {
        return false; // an empty line is malformed
    }
    let mut start = 0usize;
    let mut k = 0usize;
    for pos in memchr_iter(b',', line) {
        if k + 1 >= N {
            return false; // too many fields
        }
        out[k] = &line[start..pos];
        k += 1;
        start = pos + 1;
    }
    if k + 1 != N {
        return false; // too few fields
    }
    out[k] = &line[start..];
    for f in out.iter_mut() {
        let s = *f;
        if let Some(q) = memchr(b'"', s) {
            let n = s.len();
            if q != 0 || n < 2 || s[n - 1] != b'"' || memchr(b'"', &s[1..n - 1]).is_some() {
                return false;
            }
            *f = &s[1..n - 1];
        }
    }
    true
}

/// Call `f` on every line of `data`. Only '\n' ends a line. Empty segments are
/// passed through too (split() rejects them as malformed).
#[inline]
fn for_each_line<'a, F: FnMut(&'a [u8])>(data: &'a [u8], skip_first: bool, mut f: F) {
    let mut start = 0usize;
    let mut skip = skip_first;
    for nl in memchr_iter(b'\n', data) {
        if skip {
            skip = false;
        } else {
            f(&data[start..nl]);
        }
        start = nl + 1;
    }
    if start < data.len() && !skip {
        f(&data[start..]);
    }
}

#[derive(Clone, Copy)]
struct Chunk<'a> {
    data: &'a [u8],
    skip_first: bool, // the chunk starts its file: its first line is the header
}

/// Cut files into chunks of about `target` bytes, each ending just after a '\n'.
fn make_chunks<'a>(bufs: &[&'a [u8]], target: usize) -> Vec<Chunk<'a>> {
    let mut out = Vec::new();
    for &buf in bufs {
        let mut start = 0usize;
        while start < buf.len() {
            let mut end = (start + target).min(buf.len());
            if end < buf.len() {
                end = match memchr(b'\n', &buf[end..]) {
                    Some(i) => end + i + 1,
                    None => buf.len(),
                };
            }
            out.push(Chunk { data: &buf[start..end], skip_first: start == 0 });
            start = end;
        }
    }
    out
}

// --------------------------------------------------------------- Timestamps

#[inline]
fn d2(s: &[u8], i: usize) -> Option<i64> {
    let a = s[i].wrapping_sub(b'0');
    let b = s[i + 1].wrapping_sub(b'0');
    if a < 10 && b < 10 { Some((a * 10 + b) as i64) } else { None }
}

#[inline]
fn all_digits(s: &[u8]) -> bool {
    !s.is_empty() && s.iter().all(|c| c.is_ascii_digit())
}

/// Digits -> value, saturating at i64::MAX (too long is "outside the window", not malformed).
#[inline]
fn digits_sat(s: &[u8]) -> i64 {
    let mut v: i64 = 0;
    for &c in s {
        v = v.saturating_mul(10).saturating_add((c - b'0') as i64);
    }
    v
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline]
fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 29 } else { 28 }
        }
    }
}

/// F1 / F2 / F3 -> epoch ms. None for any other form. A huge F3 saturates.
#[inline]
fn parse_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if n != 20 && n != 25 {
        return if all_digits(s) { Some(digits_sat(s)) } else { None };
    }
    if s[4] != b'-' || s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        // 20 or 25 characters can still be an F3 value.
        return if all_digits(s) { Some(digits_sat(s)) } else { None };
    }
    let y = d2(s, 0)? * 100 + d2(s, 2)?;
    let mo = d2(s, 5)?;
    let d = d2(s, 8)?;
    let h = d2(s, 11)?;
    let mi = d2(s, 14)?;
    let se = d2(s, 17)?;
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let off = if n == 20 {
        if s[19] != b'Z' {
            return None;
        }
        0
    } else {
        let sign = s[19];
        if (sign != b'+' && sign != b'-') || s[22] != b':' {
            return None;
        }
        let oh = d2(s, 20)?;
        let om = d2(s, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let o = (oh * 60 + om) * 60_000;
        if sign == b'-' { -o } else { o }
    };
    let ms = ((days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60_000 + se * 1000;
    Some(ms - off)
}

#[inline]
fn parse_price(s: &[u8]) -> Option<u32> {
    if !all_digits(s) {
        return None;
    }
    let v = digits_sat(s);
    if (1..=10_000_000).contains(&v) { Some(v as u32) } else { None }
}

// ------------------------------------------------------------------ records

/// One parsed event copy, before dedup.
#[derive(Clone, Copy)]
struct ERec<'a> {
    id: &'a [u8],
    name: &'a [u8], // the person; only set when valid
    ingest: i64,
    ts: i64,
    price: u32, // purchase price, 0 for view/cart
    cat: u32,   // category rank
    valid: bool,
}

/// A surviving event, keyed by person.
#[derive(Clone, Copy)]
struct PRec<'a> {
    name: &'a [u8],
    ts: i64,
    price: u32,
    cat: u32,
}

/// A valid click, keyed by person.
#[derive(Clone, Copy)]
struct CRec<'a> {
    name: &'a [u8],
    id: &'a [u8],
    ts: i64,
    camp: u32,
}

#[derive(Clone, Copy, Default)]
struct Ev {
    ts: i64,
    price: u32,
    cat: u32,
}

#[derive(Clone, Copy)]
struct Cl<'a> {
    ts: i64,
    id: &'a [u8],
    camp: u32,
}

struct URow<'a> {
    name: &'a [u8],
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u32>,
}

// ---------------------------------------------------------------- utilities

fn log(t0: Instant, what: &str) {
    eprintln!("[{:>8.3}s] {what}", t0.elapsed().as_secs_f64());
}

fn json_str<'a>(text: &'a str, key: &str) -> &'a str {
    let k = format!("\"{key}\"");
    let at = text.find(&k).unwrap_or_else(|| panic!("manifest has no {key}")) + k.len();
    let rest = &text[at..];
    let colon = rest.find(':').expect("manifest colon");
    let rest = &rest[colon + 1..];
    let q1 = rest.find('"').expect("manifest quote");
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"').expect("manifest quote");
    &rest[..q2]
}

fn map_dir(dir: &Path) -> Vec<Mmap> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "csv"))
        .collect();
    paths.sort();
    paths
        .iter()
        .filter_map(|p| {
            let f = File::open(p).expect("open input");
            if f.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
                return None;
            }
            Some(unsafe { Mmap::map(&f) }.expect("mmap input"))
        })
        .collect()
}

fn small_csv<'a, const N: usize>(buf: &'a [u8]) -> Vec<[&'a [u8]; N]> {
    let mut rows = Vec::new();
    for_each_line(buf, true, |line| {
        let mut f: [&[u8]; N] = [&[]; N];
        if split(line, &mut f) {
            rows.push(f);
        }
    });
    rows
}

// --------------------------------------------------------------------- main

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let (data, out) = if args.len() >= 3 {
        (PathBuf::from(&args[1]), PathBuf::from(&args[2]))
    } else {
        (
            PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into())),
            PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into())),
        )
    };
    let threads = std::env::var("NCPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8));
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();

    let manifest = std::fs::read_to_string(data.join("manifest.json")).expect("manifest.json");
    let ws = parse_ts(json_str(&manifest, "window_start").as_bytes()).expect("window_start");
    let we = parse_ts(json_str(&manifest, "window_end").as_bytes()).expect("window_end");

    let products_buf = std::fs::read(data.join("products.csv")).expect("products.csv");
    let campaigns_buf = std::fs::read(data.join("campaigns.csv")).expect("campaigns.csv");
    let ev_maps = map_dir(&data.join("events"));
    let cl_maps = map_dir(&data.join("clicks"));
    let ln_maps = map_dir(&data.join("links"));
    log(t0, &format!("mapped inputs, {threads} threads"));

    let ev_bufs: Vec<&[u8]> = ev_maps.iter().map(|m| &m[..]).collect();
    let cl_bufs: Vec<&[u8]> = cl_maps.iter().map(|m| &m[..]).collect();
    let ln_bufs: Vec<&[u8]> = ln_maps.iter().map(|m| &m[..]).collect();

    run(t0, &out, ws, we, &products_buf, &campaigns_buf, &ev_bufs, &cl_bufs, &ln_bufs, threads);
    log(t0, "done");
}

#[allow(clippy::too_many_arguments)]
fn run<'a>(
    t0: Instant,
    out: &Path,
    ws: i64,
    we: i64,
    products_buf: &'a [u8],
    campaigns_buf: &'a [u8],
    ev_bufs: &[&'a [u8]],
    cl_bufs: &[&'a [u8]],
    ln_bufs: &[&'a [u8]],
    threads: usize,
) {
    // ---- Products -> category rank. Ranks follow byte order of the names,
    // so "smallest name wins a tie" becomes "smallest rank wins".
    let prod_rows = small_csv::<2>(products_buf);
    let mut cat_names: Vec<&[u8]> = prod_rows.iter().map(|r| r[1]).collect();
    cat_names.push(b"unknown");
    cat_names.sort_unstable();
    cat_names.dedup();
    let cat_rank: FxHashMap<&[u8], u32> =
        cat_names.iter().enumerate().map(|(i, c)| (*c, i as u32)).collect();
    let unknown = cat_rank[&b"unknown"[..]];
    let products: FxHashMap<&[u8], u32> =
        prod_rows.iter().map(|r| (r[0], cat_rank[r[1]])).collect();
    let ncat = cat_names.len();

    // ---- Campaigns.
    let camp_rows = small_csv::<2>(campaigns_buf);
    let campaigns: FxHashMap<&[u8], u32> =
        camp_rows.iter().enumerate().map(|(i, r)| (r[0], i as u32)).collect();
    let ncamp = camp_rows.len();
    let organic = ncamp; // index of "organic"

    let total: usize = ev_bufs.iter().chain(cl_bufs).map(|b| b.len()).sum();
    let target = (total / (threads * 16)).clamp(256 << 10, 64 << 20);

    // ---- Links: union-find, person named by the byte-wise smallest ID.
    let ln_chunks = make_chunks(ln_bufs, target);
    let pairs: Vec<Vec<(&[u8], &[u8])>> = ln_chunks
        .par_iter()
        .map(|c| {
            let mut v = Vec::new();
            for_each_line(c.data, c.skip_first, |line| {
                let mut f: [&[u8]; 2] = [&[]; 2];
                if split(line, &mut f) && !f[0].is_empty() && !f[1].is_empty() {
                    v.push((f[0], f[1]));
                }
            });
            v
        })
        .collect();
    let link_map: FxHashMap<&[u8], &[u8]> = {
        let n: usize = pairs.iter().map(|v| v.len()).sum();
        let mut ids: FxHashMap<&[u8], u32> = new_map(n);
        let mut parent: Vec<u32> = Vec::with_capacity(n);
        let mut keys: Vec<&[u8]> = Vec::with_capacity(n);
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                let g = parent[parent[x as usize] as usize];
                parent[x as usize] = g;
                x = g;
            }
            x
        }
        for v in &pairs {
            for &(a, b) in v {
                let mut intern = |s: &'a [u8]| -> u32 {
                    *ids.entry(s).or_insert_with(|| {
                        let i = parent.len() as u32;
                        parent.push(i);
                        keys.push(s);
                        i
                    })
                };
                let ia = intern(a);
                let ib = intern(b);
                let (ra, rb) = (find(&mut parent, ia), find(&mut parent, ib));
                if ra != rb {
                    let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                    parent[hi as usize] = lo;
                }
            }
        }
        let mut name: Vec<&[u8]> = vec![&[]; keys.len()];
        let mut has: Vec<bool> = vec![false; keys.len()];
        let roots: Vec<u32> = (0..keys.len() as u32).map(|i| find(&mut parent, i)).collect();
        for (i, &r) in roots.iter().enumerate() {
            let r = r as usize;
            if !has[r] || keys[i] < name[r] {
                name[r] = keys[i];
                has[r] = true;
            }
        }
        let mut m = new_map(keys.len());
        for (i, &r) in roots.iter().enumerate() {
            let nm = name[r as usize];
            if nm != keys[i] {
                m.insert(keys[i], nm);
            }
        }
        m
    };
    drop(pairs);
    log(t0, &format!("links: {} renamed ids", link_map.len()));
    let person_of = |uid: &'a [u8]| -> &'a [u8] { link_map.get(uid).copied().unwrap_or(uid) };

    // ---- Events: parse every copy, scatter by hash(event_id).
    let ev_chunks = make_chunks(ev_bufs, target);
    let parsed: Vec<Vec<Vec<ERec>>> = ev_chunks
        .par_iter()
        .map(|c| {
            let cap = c.data.len() / 70 / NB + 4;
            let mut out: Vec<Vec<ERec>> = (0..NB).map(|_| Vec::with_capacity(cap)).collect();
            for_each_line(c.data, c.skip_first, |line| {
                let mut f: [&[u8]; 7] = [&[]; 7];
                if !split(line, &mut f) {
                    return;
                }
                let id = f[0];
                if id.is_empty() {
                    return;
                }
                let ingest = parse_ts(f[3]).unwrap_or(i64::MAX);
                let mut rec = ERec { id, name: &[], ingest, ts: 0, price: 0, cat: 0, valid: false };
                'v: {
                    let uid = f[1];
                    if uid.is_empty() || f[5].is_empty() {
                        break 'v;
                    }
                    let Some(ts) = parse_ts(f[2]) else { break 'v };
                    if ts < ws || ts >= we {
                        break 'v;
                    }
                    let price = match f[4] {
                        b"view" | b"cart" => 0,
                        b"purchase" => match parse_price(f[6]) {
                            Some(p) => p,
                            None => break 'v,
                        },
                        _ => break 'v,
                    };
                    if ingest.saturating_sub(ts) > DAY {
                        break 'v; // late
                    }
                    rec.valid = true;
                    rec.ts = ts;
                    rec.price = price;
                    rec.cat = products.get(f[5]).copied().unwrap_or(unknown);
                    rec.name = person_of(uid);
                }
                out[shard_of(id)].push(rec);
            });
            out
        })
        .collect();
    log(t0, &format!("events parsed ({} chunks)", ev_chunks.len()));

    // ---- Dedup each event_id shard; scatter survivors by hash(person).
    let dedup: Vec<Vec<Vec<PRec>>> = (0..NB)
        .into_par_iter()
        .map(|b| {
            let n: usize = parsed.iter().map(|c| c[b].len()).sum();
            let mut best: FxHashMap<&[u8], ERec> = new_map(n);
            for c in &parsed {
                for r in &c[b] {
                    match best.entry(r.id) {
                        Entry::Occupied(mut e) => {
                            if r.ingest < e.get().ingest {
                                *e.get_mut() = *r;
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert(*r);
                        }
                    }
                }
            }
            let mut out: Vec<Vec<PRec>> = (0..NB).map(|_| Vec::with_capacity(n / NB + 4)).collect();
            for r in best.values() {
                if r.valid {
                    out[shard_of(r.name)].push(PRec { name: r.name, ts: r.ts, price: r.price, cat: r.cat });
                }
            }
            out
        })
        .collect();
    drop(parsed);
    log(t0, "events deduped");

    // ---- Clicks: parse + validate, scatter by hash(person).
    let cl_chunks = make_chunks(cl_bufs, target);
    let clicks: Vec<Vec<Vec<CRec>>> = cl_chunks
        .par_iter()
        .map(|c| {
            let cap = c.data.len() / 45 / NB + 4;
            let mut out: Vec<Vec<CRec>> = (0..NB).map(|_| Vec::with_capacity(cap)).collect();
            for_each_line(c.data, c.skip_first, |line| {
                let mut f: [&[u8]; 4] = [&[]; 4];
                if !split(line, &mut f) || f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                let Some(&camp) = campaigns.get(f[2]) else { return };
                let Some(ts) = parse_ts(f[3]) else { return };
                if ts < ws - WEEK || ts >= we {
                    return;
                }
                let name = person_of(f[1]);
                out[shard_of(name)].push(CRec { name, id: f[0], ts, camp });
            });
            out
        })
        .collect();
    log(t0, "clicks parsed");

    // ---- Per person shard: sessions, credit, assists, aggregates.
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;
    let size = (ncamp + 1) * ndays;
    let results: Vec<(Vec<URow>, Vec<[i64; 4]>)> = (0..NB)
        .into_par_iter()
        .fold(
            || (Vec::new(), vec![[0i64; 4]; size]),
            |(mut urows, mut camp), p| {
                process_shard(p, &dedup, &clicks, day0, ndays, ncat, organic, &mut urows, &mut camp);
                (urows, camp)
            },
        )
        .collect();
    drop(dedup);
    drop(clicks);
    log(t0, "sessions done");

    // ---- Write.
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    std::fs::create_dir_all(&ud).expect("mkdir user_daily");
    std::fs::create_dir_all(&cd).expect("mkdir campaign_daily");

    let mut camp_tot = vec![[0i64; 4]; size];
    for (_, c) in &results {
        for (t, x) in camp_tot.iter_mut().zip(c) {
            for k in 0..4 {
                t[k] += x[k];
            }
        }
    }
    // Pack the per-split row lists into at most `threads` files of similar size.
    let nrows: usize = results.iter().map(|(u, _)| u.len()).sum();
    let per_file = nrows.div_ceil(threads).max(1);
    let mut groups: Vec<Vec<&[URow]>> = vec![Vec::new()];
    let mut in_group = 0usize;
    for (u, _) in &results {
        let mut rest: &[URow] = u;
        while !rest.is_empty() {
            if in_group == per_file {
                groups.push(Vec::new());
                in_group = 0;
            }
            let take = rest.len().min(per_file - in_group);
            groups.last_mut().unwrap().push(&rest[..take]);
            in_group += take;
            rest = &rest[take..];
        }
    }
    rayon::join(
        || {
            groups.par_iter().enumerate().for_each(|(i, parts)| {
                write_user(&ud.join(format!("part-{i:05}.parquet")), parts, &cat_names);
            })
        },
        || write_campaign(&cd.join("part-00000.parquet"), &camp_tot, &camp_rows, ncamp, ndays, day0),
    );
    log(t0, &format!("wrote {nrows} user rows in {} files", groups.len()));
}

#[allow(clippy::too_many_arguments)]
fn process_shard<'a>(
    p: usize,
    dedup: &[Vec<Vec<PRec<'a>>>],
    clicks: &[Vec<Vec<CRec<'a>>>],
    day0: i64,
    ndays: usize,
    ncat: usize,
    organic: usize,
    urows: &mut Vec<URow<'a>>,
    camp: &mut [[i64; 4]],
) {
    // Group events by person (counting sort into one flat array).
    let n_ev: usize = dedup.iter().map(|d| d[p].len()).sum();
    if n_ev == 0 {
        return;
    }
    let mut index: FxHashMap<&[u8], u32> = new_map(n_ev / 8 + 16);
    let mut names: Vec<&[u8]> = Vec::new();
    let mut cnt: Vec<u32> = Vec::new();
    let mut pid_of: Vec<u32> = Vec::with_capacity(n_ev);
    for d in dedup {
        for r in &d[p] {
            let id = *index.entry(r.name).or_insert_with(|| {
                names.push(r.name);
                cnt.push(0);
                (names.len() - 1) as u32
            });
            cnt[id as usize] += 1;
            pid_of.push(id);
        }
    }
    let np = names.len();
    let mut ev_off = vec![0usize; np + 1];
    for i in 0..np {
        ev_off[i + 1] = ev_off[i] + cnt[i] as usize;
    }
    let mut evs = vec![Ev::default(); n_ev];
    {
        let mut cur = ev_off.clone();
        let mut k = 0;
        for d in dedup {
            for r in &d[p] {
                let id = pid_of[k] as usize;
                k += 1;
                evs[cur[id]] = Ev { ts: r.ts, price: r.price, cat: r.cat };
                cur[id] += 1;
            }
        }
    }
    drop(pid_of);

    // Group clicks the same way; clicks of people with no session are irrelevant.
    let mut ccnt = vec![0u32; np];
    let mut cpid: Vec<u32> = Vec::new();
    for c in clicks {
        for r in &c[p] {
            match index.get(r.name) {
                Some(&id) => {
                    ccnt[id as usize] += 1;
                    cpid.push(id);
                }
                None => cpid.push(u32::MAX),
            }
        }
    }
    let mut cl_off = vec![0usize; np + 1];
    for i in 0..np {
        cl_off[i + 1] = cl_off[i] + ccnt[i] as usize;
    }
    let mut cls: Vec<Cl> = vec![Cl { ts: 0, id: &[], camp: 0 }; cl_off[np]];
    {
        let mut cur = cl_off.clone();
        let mut k = 0;
        for c in clicks {
            for r in &c[p] {
                let id = cpid[k];
                k += 1;
                if id != u32::MAX {
                    let id = id as usize;
                    cls[cur[id]] = Cl { ts: r.ts, id: r.id, camp: r.camp };
                    cur[id] += 1;
                }
            }
        }
    }
    drop(cpid);

    let mut catsum = vec![0u64; ncat];
    let mut touched: Vec<u32> = Vec::new();
    let mut starts: Vec<i64> = Vec::new();
    let mut sdays: Vec<u32> = Vec::new();
    let mut run_end: Vec<u32> = Vec::new();
    let mut pairs: Vec<(u32, i64)> = Vec::new();

    for pid in 0..np {
        let ev = &mut evs[ev_off[pid]..ev_off[pid + 1]];
        ev.sort_unstable_by_key(|e| e.ts);
        let cl = &mut cls[cl_off[pid]..cl_off[pid + 1]];
        cl.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| a.id.cmp(b.id)));
        let n = ev.len();
        let m = cl.len();
        starts.clear();
        sdays.clear();

        let mut cur_day = u32::MAX;
        let (mut u_sess, mut u_ev, mut u_rev) = (0i64, 0i64, 0i64);
        let flush = |day: u32, s: i64, e: i64, r: i64, catsum: &mut [u64], touched: &mut Vec<u32>, urows: &mut Vec<URow<'a>>| {
            let mut top: Option<u32> = None;
            let mut best = 0u64;
            for &c in touched.iter() {
                let v = catsum[c as usize];
                if top.is_none() || v > best || (v == best && c < top.unwrap()) {
                    top = Some(c);
                    best = v;
                }
                catsum[c as usize] = 0;
            }
            touched.clear();
            urows.push(URow { name: names[pid], day: (day0 + day as i64) as i32, sessions: s, events: e, revenue: r, top });
        };

        let (mut hi, mut run_start) = (0usize, 0usize);
        let mut i = 0usize;
        while i < n {
            let s = ev[i].ts;
            let mut j = i + 1;
            while j < n && ev[j].ts - ev[j - 1].ts <= GAP {
                j += 1;
            }
            let day = (s.div_euclid(DAY) - day0) as u32;
            let mut rev = 0i64;
            let mut purchased = false;
            if day != cur_day {
                if cur_day != u32::MAX {
                    flush(cur_day, u_sess, u_ev, u_rev, &mut catsum, &mut touched, urows);
                }
                cur_day = day;
                u_sess = 0;
                u_ev = 0;
                u_rev = 0;
            }
            for e in &ev[i..j] {
                if e.price > 0 {
                    purchased = true;
                    rev += e.price as i64;
                    let c = e.cat as usize;
                    if catsum[c] == 0 {
                        touched.push(e.cat);
                    }
                    catsum[c] += e.price as u64;
                }
            }
            u_sess += 1;
            u_ev += (j - i) as i64;
            u_rev += rev;

            // Credit: latest click with ts <= start and start - ts <= 7 days; tie -> smallest click_id.
            while hi < m && cl[hi].ts <= s {
                if hi == 0 || cl[hi].ts != cl[hi - 1].ts {
                    run_start = hi;
                }
                hi += 1;
            }
            let c = if hi > 0 && cl[hi - 1].ts >= s - WEEK { cl[run_start].camp as usize } else { organic };
            let slot = &mut camp[c * ndays + day as usize];
            slot[0] += 1;
            slot[1] += purchased as i64;
            slot[2] += rev;

            starts.push(s);
            sdays.push(day);
            i = j;
        }
        if cur_day != u32::MAX {
            flush(cur_day, u_sess, u_ev, u_rev, &mut catsum, &mut touched, urows);
        }

        // Assists: campaign c assists session S iff some click t of c has t <= S <= t + 7d.
        // Merge each campaign's [t, t+7d] intervals, then count sessions per day inside them.
        if m > 0 {
            let ns = starts.len();
            run_end.clear();
            run_end.resize(ns, 0);
            for k in (0..ns).rev() {
                run_end[k] = if k + 1 < ns && sdays[k + 1] == sdays[k] { run_end[k + 1] } else { (k + 1) as u32 };
            }
            pairs.clear();
            pairs.extend(cl.iter().map(|c| (c.camp, c.ts)));
            pairs.sort_unstable();
            let mut add = |c: u32, a: i64, b: i64| {
                let lo = starts.partition_point(|&x| x < a);
                let hi = starts.partition_point(|&x| x <= b);
                let mut k = lo;
                while k < hi {
                    let e = (run_end[k] as usize).min(hi);
                    camp[c as usize * ndays + sdays[k] as usize][3] += (e - k) as i64;
                    k = e;
                }
            };
            let mut q = 0;
            while q < pairs.len() {
                let c = pairs[q].0;
                let mut a = pairs[q].1;
                let mut b = a + WEEK;
                q += 1;
                while q < pairs.len() && pairs[q].0 == c {
                    let t = pairs[q].1;
                    if t <= b {
                        b = t + WEEK;
                    } else {
                        add(c, a, b);
                        a = t;
                        b = t + WEEK;
                    }
                    q += 1;
                }
                add(c, a, b);
            }
        }
    }
}

// ------------------------------------------------------------------ Parquet

fn put<T: DataType, W: Write + Send>(rg: &mut SerializedRowGroupWriter<'_, W>, vals: &[T::T], def: Option<&[i16]>) {
    let mut col = rg.next_column().expect("next column").expect("column exists");
    col.typed::<T>().write_batch(vals, def, None).expect("write column");
    col.close().expect("close column");
}

fn bytes(v: &[u8]) -> ByteArray {
    ByteArray::from(v.to_vec())
}

fn write_user(path: &Path, parts: &[&[URow]], cat_names: &[&[u8]]) {
    let rows: Vec<&URow> = parts.iter().flat_map(|p| p.iter()).collect();
    let schema = parse_message_type(
        "message user_daily {
            REQUIRED BYTE_ARRAY user_id (STRING);
            REQUIRED INT32 day (DATE);
            REQUIRED INT64 sessions;
            REQUIRED INT64 events;
            REQUIRED INT64 revenue_cents;
            OPTIONAL BYTE_ARRAY top_category (STRING);
        }",
    )
    .unwrap();
    let props = WriterProperties::builder().build();
    let file = BufWriter::with_capacity(1 << 20, File::create(path).expect("create parquet"));
    let mut w = SerializedFileWriter::new(file, Arc::new(schema), Arc::new(props)).unwrap();
    if !rows.is_empty() {
        let mut rg = w.next_row_group().unwrap();
        put::<ByteArrayType, _>(&mut rg, &rows.iter().map(|r| bytes(r.name)).collect::<Vec<_>>(), None);
        put::<Int32Type, _>(&mut rg, &rows.iter().map(|r| r.day).collect::<Vec<_>>(), None);
        put::<Int64Type, _>(&mut rg, &rows.iter().map(|r| r.sessions).collect::<Vec<_>>(), None);
        put::<Int64Type, _>(&mut rg, &rows.iter().map(|r| r.events).collect::<Vec<_>>(), None);
        put::<Int64Type, _>(&mut rg, &rows.iter().map(|r| r.revenue).collect::<Vec<_>>(), None);
        let defs: Vec<i16> = rows.iter().map(|r| r.top.is_some() as i16).collect();
        let tops: Vec<ByteArray> = rows.iter().filter_map(|r| r.top).map(|c| bytes(cat_names[c as usize])).collect();
        put::<ByteArrayType, _>(&mut rg, &tops, Some(&defs));
        rg.close().unwrap();
    }
    w.close().unwrap();
}

fn write_campaign(path: &Path, tot: &[[i64; 4]], camp_rows: &[[&[u8]; 2]], ncamp: usize, ndays: usize, day0: i64) {
    let schema = parse_message_type(
        "message campaign_daily {
            REQUIRED BYTE_ARRAY campaign_id (STRING);
            REQUIRED BYTE_ARRAY channel (STRING);
            REQUIRED INT32 day (DATE);
            REQUIRED INT64 sessions;
            REQUIRED INT64 purchasing_sessions;
            REQUIRED INT64 revenue_cents;
            REQUIRED INT64 assisted_sessions;
        }",
    )
    .unwrap();
    let (mut ids, mut chans, mut days) = (Vec::new(), Vec::new(), Vec::new());
    let (mut s, mut ps, mut rev, mut asst) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..=ncamp {
        for d in 0..ndays {
            let t = tot[c * ndays + d];
            if t[0] == 0 && t[3] == 0 {
                continue;
            }
            let (id, ch): (&[u8], &[u8]) = if c == ncamp { (b"organic", b"organic") } else { (camp_rows[c][0], camp_rows[c][1]) };
            ids.push(bytes(id));
            chans.push(bytes(ch));
            days.push((day0 + d as i64) as i32);
            s.push(t[0]);
            ps.push(t[1]);
            rev.push(t[2]);
            asst.push(t[3]);
        }
    }
    let props = WriterProperties::builder().build();
    let file = BufWriter::new(File::create(path).expect("create parquet"));
    let mut w = SerializedFileWriter::new(file, Arc::new(schema), Arc::new(props)).unwrap();
    if !ids.is_empty() {
        let mut rg = w.next_row_group().unwrap();
        put::<ByteArrayType, _>(&mut rg, &ids, None);
        put::<ByteArrayType, _>(&mut rg, &chans, None);
        put::<Int32Type, _>(&mut rg, &days, None);
        put::<Int64Type, _>(&mut rg, &s, None);
        put::<Int64Type, _>(&mut rg, &ps, None);
        put::<Int64Type, _>(&mut rg, &rev, None);
        put::<Int64Type, _>(&mut rg, &asst, None);
        rg.close().unwrap();
    }
    w.close().unwrap();
}
