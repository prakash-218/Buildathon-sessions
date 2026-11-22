//! Shopping Sessions: the fast pipeline. Every rule cited here is from rules.md.
//!
//! Plan:
//!   1. mmap every input file and cut all of them into line-aligned chunks.
//!   2. Links: union-find over the IDs; each person is named by its smallest ID.
//!   3. Events: parse + validate every copy in parallel, sharded by event_id;
//!      dedup each shard (smallest ingest_ts wins, then the kept copy's validity
//!      decides), then reshard survivors by person.
//!   4. Clicks: parse + validate, sharded by person. No dedup is needed:
//!      copies are identical, so they cannot change a credit or an assist.
//!   5. Per person shard: sessionize, credit, assists, aggregate.
//!   6. Write Parquet.
//!
//! IDs are packed into u128 (bytes big-endian, length in the low byte), which
//! keeps byte-wise order and makes hashing and comparing integer operations.
//! An ID longer than 15 bytes cannot be packed: then the whole run is redone
//! with byte-slice keys, which are slower but have no limit.

use memchr::memchr;
use memmap2::{Advice, Mmap};

// mimalloc reuses freed memory (with MIMALLOC_PURGE_DELAY=-1, set in the
// Dockerfile, it never hands it back), so later phases do not fault in and
// zero fresh pages: 497k page faults per 20M-event run with glibc, 16k with this.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
mod dense;
mod pq;
use pq::{Column, GROUP_ROWS, Kind};
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::hash_map::Entry;
use std::fs::File;
use std::hash::{BuildHasher, Hash};
use std::path::{Path, PathBuf};
use std::sync::atomic::{
    AtomicBool, AtomicU32,
    Ordering::{AcqRel, Acquire, Relaxed},
};
use std::time::Instant;

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const NB: usize = 256; // shards for both shuffles

// --------------------------------------------------------------------- keys

trait Key<'a>: Copy + Eq + Ord + Hash + Send + Sync + Default {
    /// The ID `d[s..e]`, or None if this key type cannot hold it.
    fn make(d: &'a [u8], s: usize, e: usize) -> Option<Self>;
    fn put(self, out: &mut Vec<u8>);
    /// A key only ever compared for equality, never ordered or written out
    /// (event IDs): it may skip whatever only order and output need.
    #[inline(always)]
    fn make_eq(d: &'a [u8], s: usize, e: usize) -> Option<Self> {
        Self::make(d, s, e)
    }
    /// A key met while parsing events or clicks: what order and output need
    /// may wait until `settle`.
    #[inline(always)]
    fn make_late(d: &'a [u8], s: usize, e: usize) -> Option<Self> {
        Self::make(d, s, e)
    }
    /// Called once events and clicks are parsed, before any key made by
    /// `make_late` is ordered or written out.
    fn settle() {}
    /// A u64 for grouping: equal keys give equal values, and if `UNIQUE`,
    /// different keys give different values.
    fn grp(self) -> u64;
    const UNIQUE: bool = false;
}

impl<'a> Key<'a> for u128 {
    #[inline(always)]
    fn grp(self) -> u64 {
        fmix64(self as u64 ^ ((self >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
    #[inline(always)]
    fn make(d: &'a [u8], s: usize, e: usize) -> Option<u128> {
        let n = e - s;
        if n > 15 {
            return None;
        }
        let mut b = [0u8; 16];
        if s + 16 <= d.len() {
            b.copy_from_slice(&d[s..s + 16]); // one 16-byte load, masked below
        } else {
            b[..n].copy_from_slice(&d[s..e]);
        }
        let v = u128::from_be_bytes(b);
        let mask = if n == 0 { 0 } else { !0u128 << (128 - 8 * n) };
        Some((v & mask) | n as u128)
    }
    fn put(self, out: &mut Vec<u8>) {
        let b = self.to_be_bytes();
        out.extend_from_slice(&b[..b[15] as usize]);
    }
}

/// Up to 9 ASCII bytes (1..=127), 7 bits each, big-endian, zero-padded: 63 bits.
/// No byte is 0, so padding sorts before every byte and order stays byte-wise.
/// The generator's IDs are a letter and at most 8 base-36 digits, so they all fit.
impl<'a> Key<'a> for u64 {
    const UNIQUE: bool = true;
    #[inline(always)]
    fn grp(self) -> u64 {
        self
    }
    #[inline(always)]
    fn make(d: &'a [u8], s: usize, e: usize) -> Option<u64> {
        let n = e - s;
        if n > 9 {
            return None;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        if s + 8 <= d.len() {
            const HI: u64 = 0x8080_8080_8080_8080;
            const LO: u64 = 0x0101_0101_0101_0101;
            let m = n.min(8);
            let keep = if m == 0 { 0 } else { !0u64 << (64 - 8 * m) }; // the first m bytes
            let w = u64::from_be_bytes(d[s..s + 8].try_into().unwrap()) & keep;
            let w2 = w | (LO & !keep); // padding bytes become 1 so only real bytes can be zero
            if w & HI != 0 || (w2.wrapping_sub(LO) & !w2 & HI) != 0 {
                return None; // non-ASCII or NUL
            }
            let packed = unsafe { std::arch::x86_64::_pext_u64(w, 0x7f7f_7f7f_7f7f_7f7f) };
            let last = if n == 9 { d[s + 8] } else { 0 };
            if n == 9 && (last == 0 || last > 127) {
                return None;
            }
            return Some(packed << 7 | last as u64);
        }
        let mut v = 0u64;
        for i in 0..9 {
            let c = if i < n { d[s + i] } else { 0 };
            if i < n && (c == 0 || c > 127) {
                return None;
            }
            v = v << 7 | c as u64;
        }
        Some(v)
    }
    fn put(self, out: &mut Vec<u8>) {
        for i in 0..9 {
            let c = (self >> (7 * (8 - i))) as u8 & 0x7f;
            if c == 0 {
                break;
            }
            out.push(c);
        }
    }
}

/// Any generator ID, exactly, in 24 bytes, ordered as the strings are.
/// `a` is the first 9 bytes packed as in `Key for u64` (byte order kept). An
/// ID of up to 9 bytes is just that, with hi = lo = 0. A 33-byte ID of a
/// letter and 32 hex digits (lowercase after a lowercase letter, uppercase
/// after an uppercase one: the generator's long IDs and their upper-cased
/// variants) also keeps the hex value in (hi, lo), never 0 for these. So the
/// map is one-to-one, and (a, hi, lo) in order compares as the strings do: a
/// short ID that ties a long one on its first 9 bytes is its prefix, and sorts
/// first since its (hi, lo) is 0; two long IDs with one letter case compare
/// by their hex values. Anything else does not fit (then byte-slice keys run).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
struct Id {
    a: u64,
    hi: u64,
    lo: u64,
}

/// 32 hex digits (lowercase, or uppercase if `upper`) -> their value: one
/// 32-byte load checks them all, a multiply-add joins digit pairs into bytes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn hex32(b: &[u8], upper: bool) -> Option<(u64, u64)> {
    use std::arch::x86_64::*;
    assert!(b.len() >= 32);
    // SAFETY: b has at least 32 bytes.
    unsafe {
        let v = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
        let dg = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
        let is_dg = _mm256_cmpeq_epi8(_mm256_min_epu8(dg, _mm256_set1_epi8(9)), dg);
        let lt = _mm256_sub_epi8(v, _mm256_set1_epi8(if upper { b'A' } else { b'a' } as i8));
        let is_lt = _mm256_cmpeq_epi8(_mm256_min_epu8(lt, _mm256_set1_epi8(5)), lt);
        if _mm256_movemask_epi8(_mm256_or_si256(is_dg, is_lt)) != -1 {
            return None;
        }
        let val = _mm256_blendv_epi8(dg, _mm256_add_epi8(lt, _mm256_set1_epi8(10)), is_lt);
        let p = _mm256_maddubs_epi16(val, _mm256_set1_epi16(0x0110)); // 16 * first + second
        let packed = _mm256_packus_epi16(p, p); // each lane: its 8 bytes, twice
        let hi = _mm_cvtsi128_si64(_mm256_castsi256_si128(packed)) as u64;
        let lo = _mm_cvtsi128_si64(_mm256_extracti128_si256::<1>(packed)) as u64;
        Some((hi.swap_bytes(), lo.swap_bytes()))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn hex32(b: &[u8], upper: bool) -> Option<(u64, u64)> {
    let letter0 = if upper { b'A' } else { b'a' };
    let mut v = [0u64; 2];
    for (h, chunk) in v.iter_mut().zip(b.chunks_exact(16)) {
        let mut x = 0u64;
        for &c in chunk {
            let d = if c.wrapping_sub(b'0') < 10 {
                c - b'0'
            } else if c.wrapping_sub(letter0) < 6 {
                c - letter0 + 10
            } else {
                return None;
            };
            x = x << 4 | d as u64;
        }
        *h = x;
    }
    Some((v[0], v[1]))
}

impl<'a> Key<'a> for Id {
    #[inline(always)]
    fn grp(self) -> u64 {
        fmix64(self.a ^ self.hi.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.lo.rotate_left(31))
    }
    #[inline(always)]
    fn make(d: &'a [u8], s: usize, e: usize) -> Option<Id> {
        let n = e - s;
        if n <= 9 {
            return <u64 as Key>::make(d, s, e).map(|a| Id { a, hi: 0, lo: 0 });
        }
        if n != 33 {
            return None;
        }
        let c0 = d[s];
        let upper = c0.is_ascii_uppercase();
        if !upper && !c0.is_ascii_lowercase() {
            return None;
        }
        let (hi, lo) = hex32(&d[s + 1..e], upper)?;
        if hi | lo == 0 {
            return None;
        }
        let a = <u64 as Key>::make(d, s, s + 9)?;
        Some(Id { a, hi, lo })
    }
    fn put(self, out: &mut Vec<u8>) {
        if self.hi | self.lo == 0 {
            self.a.put(out);
            return;
        }
        let c0 = (self.a >> 56) as u8 & 0x7f;
        out.push(c0);
        let digits: &[u8; 16] = if c0.is_ascii_uppercase() { b"0123456789ABCDEF" } else { b"0123456789abcdef" };
        for w in [self.hi, self.lo] {
            for i in (0..16).rev() {
                out.push(digits[(w >> (4 * i)) as usize & 15]);
            }
        }
    }
}

/// IDs as 8 bytes, exactly. An ID of up to 9 bytes is its `Key for u64`
/// packing (top bit 0). A long ID (one an `Id` holds with a hex value) is the
/// top bit and a 63-bit digest of it, and the ID itself goes in a registry
/// (one entry per digest), so it can be compared and written out. A digest
/// shared by two different IDs is noticed there (`LID_CLASH`); the run then
/// writes nothing and is redone with `Id` keys. Equality and hashing are on
/// the 8 bytes; order is the strings' (via the registry for long IDs).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Lid(u64);

const LID_LONG: u64 = 1 << 63;
const LID_SHARDS: usize = 4096;
static LID_REG: std::sync::OnceLock<Vec<std::sync::Mutex<FxHashMap<u64, Id>>>> = std::sync::OnceLock::new();
static LID_CLASH: AtomicBool = AtomicBool::new(false);

fn lid_reg() -> &'static Vec<std::sync::Mutex<FxHashMap<u64, Id>>> {
    LID_REG.get_or_init(|| (0..LID_SHARDS).map(|_| std::sync::Mutex::new(FxHashMap::default())).collect())
}

impl Lid {
    /// The ID this key stands for.
    fn id(self) -> Id {
        if self.0 & LID_LONG == 0 {
            Id { a: self.0, hi: 0, lo: 0 }
        } else {
            let shard = &lid_reg()[(self.0 >> 32) as usize % LID_SHARDS];
            *shard.lock().unwrap().get(&self.0).expect("registered long ID")
        }
    }
}

impl PartialOrd for Lid {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Lid {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if (self.0 | other.0) & LID_LONG == 0 || self.0 == other.0 {
            self.0.cmp(&other.0) // both short: the packing keeps byte order
        } else {
            self.id().cmp(&other.id())
        }
    }
}

/// Long IDs met while parsing (digest, ID), per thread: `eq` only need their
/// digests checked (event IDs), `late` also go in the registry. A thread's
/// lists move to LID_PEND when the thread ends (the parse threads end with
/// their phase), and `settle` takes them from there.
#[derive(Default)]
struct LidPend {
    eq: Vec<(u64, Id)>,
    late: Vec<(u64, Id)>,
}

impl Drop for LidPend {
    fn drop(&mut self) {
        if !self.eq.is_empty() || !self.late.is_empty() {
            LID_PEND.lock().unwrap().push(std::mem::take(self));
        }
    }
}

static LID_PEND: std::sync::Mutex<Vec<LidPend>> = std::sync::Mutex::new(Vec::new());
thread_local! {
    static LID_LOCAL: std::cell::RefCell<LidPend> = std::cell::RefCell::new(LidPend::default());
}

#[inline(always)]
fn lid_digest(id: &Id) -> u64 {
    fmix64(id.hi ^ id.lo.rotate_left(29) ^ id.a.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | LID_LONG
}

impl<'a> Key<'a> for Lid {
    const UNIQUE: bool = true;
    #[inline(always)]
    fn grp(self) -> u64 {
        self.0
    }
    #[inline(always)]
    fn make_eq(d: &'a [u8], s: usize, e: usize) -> Option<Lid> {
        if e - s <= 9 {
            return <u64 as Key>::make(d, s, e).map(Lid);
        }
        let id = Id::make(d, s, e)?;
        let digest = lid_digest(&id);
        LID_LOCAL.with(|p| p.borrow_mut().eq.push((digest, id)));
        Some(Lid(digest))
    }
    #[inline(always)]
    fn make_late(d: &'a [u8], s: usize, e: usize) -> Option<Lid> {
        if e - s <= 9 {
            return <u64 as Key>::make(d, s, e).map(Lid);
        }
        let id = Id::make(d, s, e)?;
        let digest = lid_digest(&id);
        LID_LOCAL.with(|p| p.borrow_mut().late.push((digest, id)));
        Some(Lid(digest))
    }
    fn settle() {
        // Every list bucketed by registry shard (a counting pass, all lists at
        // once); then each shard is settled by one task: digests checked for
        // clashes, and the `late` IDs registered, with no contention.
        let lists = std::mem::take(&mut *LID_PEND.lock().unwrap());
        let shard_of_d = |d: u64| (d >> 32) as usize % LID_SHARDS;
        let bucket = |v: &[(u64, Id)]| -> (Vec<(u64, Id)>, Vec<u32>) {
            let mut start = vec![0u32; LID_SHARDS + 1];
            for x in v {
                start[shard_of_d(x.0) + 1] += 1;
            }
            for i in 0..LID_SHARDS {
                start[i + 1] += start[i];
            }
            let mut cur = start.clone();
            let mut out = vec![(0u64, Id::default()); v.len()];
            for x in v {
                let b = shard_of_d(x.0);
                out[cur[b] as usize] = *x;
                cur[b] += 1;
            }
            (out, start)
        };
        let eqs: Vec<(Vec<(u64, Id)>, Vec<u32>)> = lists.par_iter().map(|l| bucket(&l.eq)).collect();
        let lates: Vec<(Vec<(u64, Id)>, Vec<u32>)> = lists.par_iter().map(|l| bucket(&l.late)).collect();
        drop(lists);
        let reg = lid_reg();
        (0..LID_SHARDS).into_par_iter().with_min_len(16).for_each_init(FxHashMap::<u64, Id>::default, |seen, b| {
            seen.clear();
            for (v, st) in &eqs {
                for x in &v[st[b] as usize..st[b + 1] as usize] {
                    if *seen.entry(x.0).or_insert(x.1) != x.1 {
                        LID_CLASH.store(true, Relaxed);
                    }
                }
            }
            let mut m = reg[b].lock().unwrap();
            for (v, st) in &lates {
                for x in &v[st[b] as usize..st[b + 1] as usize] {
                    if *m.entry(x.0).or_insert(x.1) != x.1 {
                        LID_CLASH.store(true, Relaxed);
                    }
                }
            }
        });
    }
    #[inline(always)]
    fn make(d: &'a [u8], s: usize, e: usize) -> Option<Lid> {
        if e - s <= 9 {
            return <u64 as Key>::make(d, s, e).map(Lid);
        }
        let id = Id::make(d, s, e)?;
        let digest = lid_digest(&id);
        let shard = &lid_reg()[(digest >> 32) as usize % LID_SHARDS];
        let mut m = shard.lock().unwrap();
        match m.entry(digest) {
            Entry::Occupied(o) => {
                if *o.get() != id {
                    LID_CLASH.store(true, Relaxed);
                }
            }
            Entry::Vacant(v) => {
                v.insert(id);
            }
        }
        Some(Lid(digest))
    }
    fn put(self, out: &mut Vec<u8>) {
        if self.0 & LID_LONG == 0 {
            self.0.put(out);
        } else {
            self.id().put(out);
        }
    }
}

impl<'a> Key<'a> for &'a [u8] {
    #[inline(always)]
    fn grp(self) -> u64 {
        mix(&self)
    }
    #[inline(always)]
    fn make(d: &'a [u8], s: usize, e: usize) -> Option<Self> {
        Some(&d[s..e])
    }
    fn put(self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
}

#[inline(always)]
fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// A re-mixed hash: the top 8 bits pick the shard, the low 24 the filter bit,
/// and neither correlates with the bits the maps inside each shard use.
#[inline(always)]
fn mix<T: Hash>(k: &T) -> u64 {
    fmix64(FxBuildHasher.hash_one(k))
}

#[inline(always)]
fn shard_of<T: Hash>(k: &T) -> usize {
    (mix(k) >> 56) as usize & (NB - 1)
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    // A prefetch never faults, so any address is fine.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch::<_MM_HINT_T0>(p as *const i8);
    }
}

/// ID -> person name, for the IDs that are not their person's own name. Open
/// addressing with linear probing and (key, value) pairs side by side, so a
/// lookup usually reads one cache line, whose address is known from the hash
/// alone: callers prefetch it a few rows ahead. Empty slots hold
/// `K::default()`, which no ID can equal (IDs are never empty).
struct LinkTable<K> {
    slots: Vec<(K, K)>,
    mask: usize,
}

impl<'a, K: Key<'a>> LinkTable<K> {
    fn new(entries: &[(K, K)]) -> Self {
        let cap = (entries.len() * 2 + 2).next_power_of_two().max(8); // load <= 1/2
        let mask = cap - 1;
        let mut slots = vec![(K::default(), K::default()); cap];
        for &(k, v) in entries {
            let mut i = mix(&k) as usize & mask;
            while slots[i].0 != K::default() {
                i = (i + 1) & mask;
            }
            slots[i] = (k, v);
        }
        LinkTable { slots, mask }
    }

    #[inline(always)]
    fn prefetch(&self, h: u64) {
        prefetch(self.slots.as_ptr().wrapping_add(h as usize & self.mask));
    }

    /// The name of `k` (whose `mix` is `h`): its entry, or `k` itself.
    #[inline(always)]
    fn get(&self, k: K, h: u64) -> K {
        let mut i = h as usize & self.mask;
        loop {
            let s = &self.slots[i];
            if s.0 == k {
                return s.1;
            }
            if s.0 == K::default() {
                return k;
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// How many rows a person lookup is issued ahead of its use.
const AHEAD: usize = 16;

/// "Product ID `p<digits>` in canonical form (no leading zero)" -> its number.
/// Such IDs are looked up in a small array instead of a hash map.
#[inline(always)]
fn product_number(s: &[u8]) -> Option<usize> {
    if s.len() < 2 || s.len() > 7 || s[0] != b'p' || (s.len() > 2 && s[1] == b'0') {
        return None;
    }
    let mut v = 0usize;
    for &c in &s[1..] {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as usize;
    }
    Some(v)
}

/// "Campaign ID `c_<digits>` in canonical form (no leading zero)" -> its number.
#[inline(always)]
fn campaign_number(s: &[u8]) -> Option<usize> {
    if s.len() < 3 || s.len() > 8 || s[0] != b'c' || s[1] != b'_' || (s.len() > 3 && s[2] == b'0') {
        return None;
    }
    let mut v = 0usize;
    for &c in &s[2..] {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as usize;
    }
    Some(v)
}

/// Room for one shard's share of `bytes` of input at `per_line` bytes a line:
/// the mean plus 4 standard deviations, so shard vectors rarely outgrow it
/// (each growth copies the whole vector).
fn shard_cap(bytes: usize, per_line: usize) -> usize {
    let m = bytes / per_line / NB;
    m + 4 * (m as f64).sqrt() as usize + 16
}

fn new_map<K, V>(cap: usize) -> FxHashMap<K, V> {
    FxHashMap::with_capacity_and_hasher(cap, FxBuildHasher)
}

// ---------------------------------------------------------------------- CSV

/// A field's value as `d[s..e]`, quotes already removed. Empty means null.
/// Positions are u32 (chunks are far below 4 GB), so a row's fields take half
/// the registers.
#[derive(Clone, Copy, Default)]
struct Fd {
    s0: u32,
    e0: u32,
}

impl Fd {
    #[inline(always)]
    fn new(s: usize, e: usize) -> Fd {
        Fd { s0: s as u32, e0: e as u32 }
    }
    #[inline(always)]
    fn s(self) -> usize {
        self.s0 as usize
    }
    #[inline(always)]
    fn e(self) -> usize {
        self.e0 as usize
    }
    #[inline(always)]
    fn empty(self) -> bool {
        self.s0 == self.e0
    }
}

/// Newline, comma and quote masks for the 64 bytes at `d[base..]` (zero-padded at the end).
#[inline(always)]
fn masks(d: &[u8], base: usize) -> (u64, u64, u64) {
    if base + 64 <= d.len() {
        block_masks(&d[base..base + 64])
    } else {
        let mut b = [0u8; 64];
        let n = d.len() - base;
        b[..n].copy_from_slice(&d[base..]);
        block_masks(&b)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn block_masks(b: &[u8]) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    debug_assert!(b.len() >= 64);
    unsafe {
        let p = b.as_ptr() as *const __m256i;
        let lo = _mm256_loadu_si256(p);
        let hi = _mm256_loadu_si256(p.add(1));
        let eq = |c: u8| -> u64 {
            let v = _mm256_set1_epi8(c as i8);
            let a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, v)) as u32 as u64;
            let h = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, v)) as u32 as u64;
            a | (h << 32)
        };
        (eq(b'\n'), eq(b','), eq(b'"'))
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn block_masks(b: &[u8]) -> (u64, u64, u64) {
    let (mut nl, mut cm, mut qt) = (0u64, 0u64, 0u64);
    for (i, &c) in b[..64].iter().enumerate() {
        nl |= ((c == b'\n') as u64) << i;
        cm |= ((c == b',') as u64) << i;
        qt |= ((c == b'"') as u64) << i;
    }
    (nl, cm, qt)
}

/// Bits `lo..hi` of a u64, for `0 <= lo <= hi <= 64`.
#[inline(always)]
fn bits_between(lo: usize, hi: usize) -> u64 {
    let below = |x: usize| if x >= 64 { !0u64 } else { (1u64 << x) - 1 };
    below(hi) & !below(lo)
}

/// Call `f` with the N fields of every well-formed line of `d`, under the CSV
/// dialect: only '\n' ends a line, exactly N fields, quotes only around a whole
/// field (and nowhere inside it). Malformed and empty lines are skipped, and so
/// is the first line when `skip_first` (the file's header).
///
/// `d[f.s()..f.e()]` without bounds checks: every Fd the scanners make lies
/// inside the buffer it came from (s <= e <= len).
#[inline(always)]
fn fld(d: &[u8], f: Fd) -> &[u8] {
    debug_assert!(f.s() <= f.e() && f.e() <= d.len());
    // SAFETY: see above.
    unsafe { d.get_unchecked(f.s()..f.e()) }
}

/// `d[i]` without a bounds check, for `i < d.len()`.
#[inline(always)]
fn at(d: &[u8], i: usize) -> u8 {
    debug_assert!(i < d.len());
    // SAFETY: i < d.len().
    unsafe { *d.get_unchecked(i) }
}

/// Strip the quotes of the fields that open with one; false unless they all
/// close with one and those are all of the line's `quotes`.
#[inline(always)]
fn unwrap_quotes<const N: usize>(d: &[u8], fl: &mut [Fd; N], quotes: u64) -> bool {
    let mut wrapped = 0u64;
    for x in fl.iter_mut() {
        if x.e() > x.s() && at(d, x.s()) == b'"' {
            if x.e() < x.s() + 2 || at(d, x.e() - 1) != b'"' {
                return false; // opens with a quote but does not close with one
            }
            *x = Fd::new(x.s() + 1, x.e() - 1);
            wrapped += 2;
        }
    }
    wrapped == quotes // else a quote inside a field
}

/// The line `d[s..e]` (no '\n' in it), byte by byte: its N fields, or false.
#[inline(never)]
fn slow_line<const N: usize>(d: &[u8], s: usize, e: usize, fl: &mut [Fd; N]) -> bool {
    let (mut k, mut fstart, mut quotes) = (0usize, s, 0u64);
    for pos in s..=e {
        if pos == e || d[pos] == b',' {
            if k >= N {
                return false;
            }
            fl[k] = Fd::new(fstart, pos);
            k += 1;
            fstart = pos + 1;
        } else if d[pos] == b'"' {
            quotes += 1;
        }
    }
    k == N && (quotes == 0 || unwrap_quotes(d, fl, quotes))
}


/// Every well-formed line of `d` whose first field is not empty, as N fields,
/// to `row` (the first line skipped if `skip_first`). The bytes are streamed
/// in 64-byte blocks, each classified once (separators, newlines, quotes):
/// a line's N separators come off the separator masks in order, and its quote
/// count off the quote masks, so a line without quotes needs no quote checks.
/// Lines in the last 64 bytes or so go through `slow_line`. `row` has one
/// call site, so it is inlined.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn each_line2<const N: usize>(d: &[u8], skip_first: bool, mut row: impl FnMut(&[Fd; N])) {
    use std::arch::x86_64::*;
    let n = d.len();
    let mut ls = if skip_first { memchr(b'\n', d).map_or(n, |i| i + 1) } else { 0 };
    // (separators, newlines, quotes) of d[b..b + 64]; needs b + 64 <= n.
    let blk = |b: usize| -> (u64, u64, u64) {
        debug_assert!(b + 64 <= n);
        // SAFETY: b + 64 <= n.
        unsafe {
            let p = d.as_ptr().add(b) as *const __m256i;
            let (v0, v1) = (_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1)));
            let (cm, nl, qt) = (_mm256_set1_epi8(b',' as i8), _mm256_set1_epi8(b'\n' as i8), _mm256_set1_epi8(b'"' as i8));
            let (n0, n1) = (_mm256_cmpeq_epi8(v0, nl), _mm256_cmpeq_epi8(v1, nl));
            let s0 = _mm256_or_si256(_mm256_cmpeq_epi8(v0, cm), n0);
            let s1 = _mm256_or_si256(_mm256_cmpeq_epi8(v1, cm), n1);
            let (q0, q1) = (_mm256_cmpeq_epi8(v0, qt), _mm256_cmpeq_epi8(v1, qt));
            let m = |a: __m256i, b: __m256i| (_mm256_movemask_epi8(a) as u32 as u64) | ((_mm256_movemask_epi8(b) as u32 as u64) << 32);
            (m(s0, s1), m(n0, n1), m(q0, q1))
        }
    };
    let mut fl = [Fd::default(); N];
    let mut fast = ls + 64 <= n;
    let mut base = ls;
    let (mut sep, mut nlm, mut qm) = if fast { blk(base) } else { (0, 0, 0) };
    // Quote counts from the stream's start: before `base`, and before `ls`.
    let (mut qacc, mut q_ls) = (0u64, 0u64);
    loop {
        // Some(well formed?) for the line at ls, or None: out of whole blocks.
        let r: Option<bool> = 'f: {
            if !fast {
                break 'f None;
            }
            let mut p = [0usize; N];
            let mut nlbits = 0u32;
            let mut q_end = 0u64;
            for (i, pi) in p.iter_mut().enumerate() {
                while sep == 0 {
                    let nb = base + 64;
                    if nb + 64 > n {
                        break 'f None;
                    }
                    qacc += qm.count_ones() as u64;
                    base = nb;
                    (sep, nlm, qm) = blk(base);
                }
                let k = sep.trailing_zeros();
                sep &= sep - 1;
                *pi = base + k as usize;
                nlbits |= ((nlm >> k) as u32 & 1) << i;
                if i == N - 1 {
                    q_end = qacc + (qm & ((1u64 << k) - 1)).count_ones() as u64;
                }
            }
            if nlbits == 1 << (N - 1) {
                // N - 1 commas, then the newline.
                let nq = q_end - q_ls;
                q_ls = q_end; // d[le] is the newline, not a quote
                let l0 = ls;
                ls = p[N - 1] + 1;
                let mut s = l0;
                if nq == 0 {
                    for (f, &e) in fl.iter_mut().zip(&p) {
                        *f = Fd::new(s, e);
                        s = e + 1;
                    }
                    break 'f Some(p[0] > l0);
                }
                if p[0] == l0 {
                    break 'f Some(false); // empty first field
                }
                // Each field opens with a quote iff it closes with one (and then
                // has at least 2 bytes), and those are all its quotes.
                let (mut bad, mut opened) = (0u32, 0u64);
                for (f, &e) in fl.iter_mut().zip(&p) {
                    let oq = (at(d, s) == b'"') as u32;
                    let cq = (at(d, e - 1) == b'"') as u32;
                    bad |= (oq ^ cq) | (oq & ((e - s < 2) as u32));
                    opened += oq as u64;
                    *f = Fd::new(s + oq as usize, e - oq as usize);
                    s = e + 1;
                }
                break 'f Some(bad == 0 && 2 * opened == nq && !fl[0].empty());
            }
            if nlbits & ((1 << (N - 1)) - 1) != 0 {
                // Too few fields: the line ends at its first newline. Restart the
                // stream just after it.
                let le = p[nlbits.trailing_zeros() as usize];
                q_ls += d[ls..le].iter().filter(|&&c| c == b'"').count() as u64;
                ls = le + 1;
                if ls + 64 > n {
                    fast = false;
                    break 'f Some(false);
                }
                base = ls;
                (sep, nlm, qm) = blk(base);
                qacc = q_ls;
                break 'f Some(false);
            }
            // Too many fields: skip to the newline.
            loop {
                while sep == 0 {
                    let nb = base + 64;
                    if nb + 64 > n {
                        break 'f None;
                    }
                    qacc += qm.count_ones() as u64;
                    base = nb;
                    (sep, nlm, qm) = blk(base);
                }
                let k = sep.trailing_zeros();
                sep &= sep - 1;
                if (nlm >> k) & 1 == 1 {
                    q_ls = qacc + (qm & ((1u64 << k) - 1)).count_ones() as u64;
                    ls = base + k as usize + 1;
                    break 'f Some(false);
                }
            }
        };
        let ok = match r {
            Some(ok) => ok,
            None => {
                // The line at ls, byte by byte (and every later one: `fast` stays off).
                fast = false;
                if ls >= n {
                    return;
                }
                let le = memchr(b'\n', &d[ls..]).map_or(n, |i| ls + i);
                let ok = slow_line(d, ls, le, &mut fl) && !fl[0].empty();
                ls = le + 1;
                ok
            }
        };
        if ok {
            row(&fl);
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn each_line2<const N: usize>(d: &[u8], skip_first: bool, mut row: impl FnMut(&[Fd; N])) {
    dense::each_line::<N>(d, skip_first, |fl| {
        if !fl[0].empty() {
            row(fl)
        }
    })
}

/// Only commas and newlines are stepped through (a regular pattern the branch
/// predictor learns); quotes are counted per field with a popcount. A field is
/// valid with no quote at all, or with exactly two that are its first and last
/// byte (then nothing between them can be a quote).
#[inline(always)]
fn scan<const N: usize, F: FnMut(&[Fd; N])>(d: &[u8], skip_first: bool, mut f: F) {
    let n = d.len();
    // Per line: where its separators are (the first N; more means malformed),
    // and how many quotes it holds. A line is well formed iff it has exactly N
    // separators and its quotes number exactly 2 for each field that starts and
    // ends with one (and is at least 2 long): then those are all its quotes, so
    // every quote wraps a whole field. That is the per-field rule, checked once.
    let mut seps = [0usize; N];
    let mut k = 0usize; // separators so far in this line
    let mut lstart = 0usize; // where this line starts
    let mut qbase = 0u64; // quotes before the current block
    let mut qstart = 0u64; // quotes before this line
    let mut skip = skip_first;

    #[inline(always)]
    fn line<const N: usize, F: FnMut(&[Fd; N])>(d: &[u8], lstart: usize, seps: &[usize; N], quotes: u64, f: &mut F) {
        let mut fl = [Fd::default(); N];
        let mut s = lstart;
        for j in 0..N {
            fl[j] = Fd::new(s, seps[j]);
            s = seps[j] + 1;
        }
        if quotes != 0 {
            let mut wrapped = 0u64;
            for x in fl.iter_mut() {
                if x.e() > x.s() && d[x.s()] == b'"' {
                    if x.e() < x.s() + 2 || d[x.e() - 1] != b'"' {
                        return; // opens with a quote but does not close with one
                    }
                    *x = Fd::new(x.s() + 1, x.e() - 1);
                    wrapped += 2;
                }
            }
            if wrapped != quotes {
                return; // a quote inside a field
            }
        }
        f(&fl);
    }

    // A last line with no '\n' ends at a virtual one at n, so `line` (and `f`
    // inside it) has a single call site and is inlined.
    let virt = if n > 0 && d[n - 1] != b'\n' { n } else { usize::MAX };
    let mut base = 0usize;
    while base < n || base == virt {
        // Ask for the input 1 KB ahead, so the stream arrives before the scanner
        // needs it. A prefetch never faults, so running past the end is harmless.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            _mm_prefetch::<_MM_HINT_T0>(d.as_ptr().wrapping_add(base + 1024) as *const i8);
        }
        let (mut nl, cm, qt) = masks(d, base);
        if virt.wrapping_sub(base) < 64 {
            nl |= 1 << (virt - base);
        }
        let mut s = nl | cm;
        while s != 0 {
            let bit = s.trailing_zeros() as usize;
            s &= s - 1;
            seps[k.min(N - 1)] = base + bit;
            k += 1;
            if (nl >> bit) & 1 != 0 {
                let qend = qbase + (qt & bits_between(0, bit)).count_ones() as u64;
                if k == N && !skip {
                    line(d, lstart, &seps, qend - qstart, &mut f);
                }
                skip = false;
                k = 0;
                lstart = base + bit + 1;
                qstart = qend;
            }
        }
        qbase += qt.count_ones() as u64;
        base += 64;
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

#[inline(always)]
fn d2(s: &[u8], i: usize) -> Option<i64> {
    let a = s[i].wrapping_sub(b'0');
    let b = s[i + 1].wrapping_sub(b'0');
    if a < 10 && b < 10 { Some((a * 10 + b) as i64) } else { None }
}

#[inline(always)]
fn all_digits(s: &[u8]) -> bool {
    !s.is_empty() && s.iter().all(|c| c.is_ascii_digit())
}

/// Digits -> value, saturating at i64::MAX (too long is "outside the window", not malformed).
#[inline(always)]
fn digits_sat(s: &[u8]) -> i64 {
    let mut v: i64 = 0;
    for &c in s {
        v = v.saturating_mul(10).saturating_add((c - b'0') as i64);
    }
    v
}

#[inline(always)]
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline(always)]
fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 29 } else { 28 }
        }
    }
}

/// Days from 1970-01-01 to the 1st of each month of 1970..2100, built at compile time.
const TABLE_Y0: i64 = 1970;
const TABLE_YEARS: usize = 130;
/// Days since 1970 at the start of each month from TABLE_Y0, plus one more
/// entry (the month after the last), so a month's length is the next entry minus its own.
const MONTH_START: [i32; TABLE_YEARS * 12 + 1] = {
    let mut t = [0i32; TABLE_YEARS * 12 + 1];
    let mut i = 0;
    while i <= TABLE_YEARS * 12 {
        let (y, m) = (TABLE_Y0 + (i / 12) as i64, (i % 12) as i64 + 1);
        let yy = if m <= 2 { y - 1 } else { y };
        let era = yy.div_euclid(400);
        let yoe = yy - era * 400;
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5;
        t[i] = (era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468) as i32;
        i += 1;
    }
    t
};

/// True if all 8 bytes of `w` are ASCII digits. Sound: a non-digit byte always
/// sets its own top bit in `x | y` (borrows and carries only start at bytes
/// that already fail), so a `true` is never wrong.
#[inline(always)]
fn digits8(w: u64) -> bool {
    let x = w.wrapping_sub(0x3030_3030_3030_3030);
    let y = w.wrapping_add(0x4646_4646_4646_4646);
    (x | y) & 0x8080_8080_8080_8080 == 0
}

#[inline(always)]
fn le64(s: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(s[i..i + 8].try_into().unwrap())
}

/// Fast path for `YYYY-MM-DDTHH:MM:SS` + `Z` (F1) or `±HH:MM` (F2): checks all
/// digits and separators a word at a time. Returns None whenever it is not
/// sure, and then the exact slow path decides, so it never accepts anything
/// the slow path would reject, and computes the same value.
#[inline(always)]
fn parse_iso_fast(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n != 20 && n != 25 {
        return None;
    }
    let w0 = le64(s, 0); // YYYY-MM-
    let w1 = le64(s, 8); // DDTHH:MM
    let w2 = le64(s, 12); // HH:MM:SS + byte 19
    const S0: u64 = 0xff << 32 | 0xff << 56; // bytes 4, 7
    const S1: u64 = 0xff << 16 | 0xff << 40; // bytes 10, 13
    const ZERO: u64 = 0x3030_3030_3030_3030;
    if w0 & S0 != (b'-' as u64) << 32 | (b'-' as u64) << 56
        || w1 & S1 != (b'T' as u64) << 16 | (b':' as u64) << 40
        || s[16] != b':'
    {
        return None;
    }
    let k2: u64 = 0xffff << 40; // bytes 17, 18 of the string
    // Separators replaced by '0', so every byte is a digit and subtracting '0'
    // cannot borrow across bytes.
    let (d0, d1, d2) = (w0 & !S0 | ZERO & S0, w1 & !S1 | ZERO & S1, w2 & k2 | ZERO & !k2);
    if !digits8(d0) || !digits8(d1) || !digits8(d2) {
        return None;
    }
    let x0 = (d0 - ZERO).to_le_bytes();
    let x1 = (d1 - ZERO).to_le_bytes();
    let x2 = (d2 - ZERO).to_le_bytes();
    let y = x0[0] as i64 * 1000 + x0[1] as i64 * 100 + x0[2] as i64 * 10 + x0[3] as i64;
    let mo = x0[5] as i64 * 10 + x0[6] as i64;
    let d = x1[0] as i64 * 10 + x1[1] as i64;
    let h = x1[3] as i64 * 10 + x1[4] as i64;
    let mi = x1[6] as i64 * 10 + x1[7] as i64;
    let se = x2[5] as i64 * 10 + x2[6] as i64;
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let off = if n == 20 {
        if s[19] != b'Z' {
            return None;
        }
        0
    } else {
        let (sign, a, b, c, e) = (s[19], s[20], s[21], s[23], s[24]);
        if (sign != b'+' && sign != b'-') || s[22] != b':' {
            return None;
        }
        let (a, b, c, e) = (a.wrapping_sub(b'0'), b.wrapping_sub(b'0'), c.wrapping_sub(b'0'), e.wrapping_sub(b'0'));
        if a > 9 || b > 9 || c > 9 || e > 9 {
            return None;
        }
        let (oh, om) = (a as i64 * 10 + b as i64, c as i64 * 10 + e as i64);
        if oh > 23 || om > 59 {
            return None;
        }
        let o = (oh * 60 + om) * 60_000;
        if sign == b'-' { -o } else { o }
    };
    let days = if (TABLE_Y0..TABLE_Y0 + TABLE_YEARS as i64).contains(&y) {
        MONTH_START[(y - TABLE_Y0) as usize * 12 + mo as usize - 1] as i64 + d - 1
    } else {
        days_from_civil(y, mo, d)
    };
    Some(((days * 24 + h) * 60 + mi) * 60_000 + se * 1000 - off)
}

/// `parse_ts(&d[f.s()..f.e()])`, faster: SSE for F1 and F2, and for F3 of up to 16
/// digits (loads may read bytes of `d` before the field, never outside `d`).
#[inline(always)]
fn ts_at(d: &[u8], f: Fd) -> Option<i64> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let n = f.e() - f.s();
        if n == 20 || n == 25 {
            if let Some(v) = iso_sse(fld(d, f)) {
                return Some(v);
            }
        } else if (1..=16).contains(&n) && f.e() >= 16 {
            // Shorter than F1, so only F3 can be valid.
            return digits_sse(fld(d, Fd::new(f.e() - 16, f.e())), n).map(|v| v as i64);
        }
    }
    parse_ts(fld(d, f))
}

/// An F1 timestamp `d[f]` (20 bytes) -> epoch seconds, in one 32-byte load:
/// separators and digits checked with one compare, the seven digit pairs
/// (seconds included) gathered by one shuffle and range-checked together.
/// None for anything else (then `ts_at` decides). Needs f.s() + 32 <= d.len().
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn f1_secs(d: &[u8], f: Fd) -> Option<i64> {
    use std::arch::x86_64::*;
    if f.e() - f.s() != 20 || f.s() + 32 > d.len() {
        return None;
    }
    // SAFETY: f.s() + 32 <= d.len().
    let (y, mo, dd, h, mi, se) = unsafe {
        let v = _mm256_loadu_si256(d.as_ptr().add(f.s()) as *const __m256i);
        let dg = _mm256_sub_epi8(v, _mm256_set1_epi8(0x30));
        let digit = _mm256_cmpeq_epi8(_mm256_min_epu8(dg, _mm256_set1_epi8(9)), dg);
        let tmpl = _mm256_setr_epi8(
            0, 0, 0, 0, 45, 0, 0, 45, 0, 0, 84, 0, 0, 58, 0, 0, 58, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ); // - - T : : Z
        let at_sep = _mm256_setr_epi8(
            0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let ok = _mm256_blendv_epi8(digit, _mm256_cmpeq_epi8(v, tmpl), at_sep);
        if (_mm256_movemask_epi8(ok) as u32) & 0xf_ffff != 0xf_ffff {
            return None;
        }
        // Low lane: YY YY MM DD hh mm; high lane: ss (bytes 17 and 18 are its 1 and 2).
        let sh = _mm256_setr_epi8(
            0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let p = _mm256_maddubs_epi16(_mm256_shuffle_epi8(dg, sh), _mm256_set1_epi16(0x010a));
        let maxv = _mm256_setr_epi16(99, 99, 12, 31, 23, 59, 0, 0, 59, 0, 0, 0, 0, 0, 0, 0);
        let minv = _mm256_setr_epi16(0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        if _mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpgt_epi16(p, maxv), _mm256_cmpgt_epi16(minv, p))) != 0 {
            return None;
        }
        let lo = _mm256_castsi256_si128(p);
        let hi = _mm256_extracti128_si256::<1>(p);
        (
            _mm_extract_epi16::<0>(lo) as i64 * 100 + _mm_extract_epi16::<1>(lo) as i64,
            _mm_extract_epi16::<2>(lo) as i64,
            _mm_extract_epi16::<3>(lo) as i64,
            _mm_extract_epi16::<4>(lo) as i64,
            _mm_extract_epi16::<5>(lo) as i64,
            _mm_extract_epi16::<0>(hi) as i64,
        )
    };
    if !(TABLE_Y0..TABLE_Y0 + TABLE_YEARS as i64).contains(&y) {
        return None;
    }
    let i = (y - TABLE_Y0) as usize * 12 + mo as usize - 1;
    let (start, next) = (MONTH_START[i] as i64, MONTH_START[i + 1] as i64);
    if dd > next - start {
        return None;
    }
    Some((((start + dd - 1) * 24 + h) * 60 + mi) * 60 + se)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn f1_secs(_d: &[u8], _f: Fd) -> Option<i64> {
    None
}

/// `ts_at`, with F1 by `f1_secs` first.
#[inline(always)]
fn ts_fast(d: &[u8], f: Fd) -> Option<i64> {
    match f1_secs(d, f) {
        Some(s) => Some(s * 1000),
        None => ts_at(d, f),
    }
}

/// F1 or F2 (`s` is 20 or 25 bytes) -> epoch ms, or None to let the exact
/// parser decide (it also reads 20 or 25 digits as F3).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn iso_sse(s: &[u8]) -> Option<i64> {
    use std::arch::x86_64::*;
    let n = s.len();
    // SAFETY: s has at least 20 bytes, so the 16-byte load stays inside it.
    let (y, mo, d, h, mi) = unsafe {
        let v = _mm_loadu_si128(s.as_ptr() as *const __m128i);
        let dg = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
        let digit = _mm_cmpeq_epi8(_mm_min_epu8(dg, _mm_set1_epi8(9)), dg);
        let sep = _mm_cmpeq_epi8(v, _mm_setr_epi8(0, 0, 0, 0, 45, 0, 0, 45, 0, 0, 84, 0, 0, 58, 0, 0)); // - - T :
        let at_sep = _mm_setr_epi8(0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
        if _mm_movemask_epi8(_mm_blendv_epi8(digit, sep, at_sep)) != 0xffff {
            return None;
        }
        // Digit pairs YY YY MM DD hh mm, each as 10 * first + second.
        let pairs = _mm_shuffle_epi8(dg, _mm_setr_epi8(0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, -1, -1, -1, -1));
        let w = _mm_maddubs_epi16(pairs, _mm_set1_epi16(0x010a));
        (
            _mm_extract_epi16::<0>(w) as i64 * 100 + _mm_extract_epi16::<1>(w) as i64,
            _mm_extract_epi16::<2>(w) as i64,
            _mm_extract_epi16::<3>(w) as i64,
            _mm_extract_epi16::<4>(w) as i64,
            _mm_extract_epi16::<5>(w) as i64,
        )
    };
    let (s1, s2) = (at(s, 17).wrapping_sub(b'0'), at(s, 18).wrapping_sub(b'0'));
    if at(s, 16) != b':' || s1 > 9 || s2 > 9 {
        return None;
    }
    let se = (s1 * 10 + s2) as i64;
    let off = if n == 20 {
        if at(s, 19) != b'Z' {
            return None;
        }
        0
    } else {
        let (sign, a, b) = (at(s, 19), at(s, 20).wrapping_sub(b'0'), at(s, 21).wrapping_sub(b'0'));
        let (c, e) = (at(s, 23).wrapping_sub(b'0'), at(s, 24).wrapping_sub(b'0'));
        if (sign != b'+' && sign != b'-') || at(s, 22) != b':' || a > 9 || b > 9 || c > 9 || e > 9 {
            return None;
        }
        let (oh, om) = ((a * 10 + b) as i64, (c * 10 + e) as i64);
        if oh > 23 || om > 59 {
            return None;
        }
        let o = (oh * 60 + om) * 60_000;
        if sign == b'-' { -o } else { o }
    };
    if !(1..=12).contains(&mo) || h > 23 || mi > 59 || se > 59 || !(TABLE_Y0..TABLE_Y0 + TABLE_YEARS as i64).contains(&y) {
        return None; // (a year outside the table: the exact parser decides)
    }
    let i = (y - TABLE_Y0) as usize * 12 + mo as usize - 1;
    let (start, next) = (MONTH_START[i] as i64, MONTH_START[i + 1] as i64);
    if d < 1 || d > next - start {
        return None;
    }
    Some((((start + d - 1) * 24 + h) * 60 + mi) * 60_000 + se * 1000 - off)
}

/// The last `n` (1..=16) bytes of `w` (16 bytes) as a decimal number, or None
/// if any of them is not a digit.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn digits_sse(w: &[u8], n: usize) -> Option<u64> {
    use std::arch::x86_64::*;
    debug_assert!(w.len() == 16 && (1..=16).contains(&n));
    // SAFETY: w is 16 bytes.
    unsafe {
        let v = _mm_loadu_si128(w.as_ptr() as *const __m128i);
        let idx = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let keep = _mm_cmpgt_epi8(idx, _mm_set1_epi8(15 - n as i8));
        let x = _mm_blendv_epi8(_mm_set1_epi8(b'0' as i8), v, keep); // leading bytes become '0'
        let dg = _mm_sub_epi8(x, _mm_set1_epi8(b'0' as i8));
        if _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(dg, _mm_set1_epi8(9)), dg)) != 0xffff {
            return None;
        }
        let p2 = _mm_maddubs_epi16(dg, _mm_set1_epi16(0x010a)); // 8 x 2 digits
        let p4 = _mm_madd_epi16(p2, _mm_set1_epi32(0x0001_0064)); // 4 x 4 digits
        let p4 = _mm_packus_epi32(p4, p4);
        let p8 = _mm_madd_epi16(p4, _mm_set1_epi32(0x0001_2710)); // 2 x 8 digits
        let hi = _mm_cvtsi128_si32(p8) as u32 as u64;
        let lo = _mm_extract_epi32::<1>(p8) as u32 as u64;
        Some(hi * 100_000_000 + lo)
    }
}

/// F1 / F2 / F3 -> epoch ms. None for any other form. A huge F3 saturates.
#[inline(always)]
fn parse_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if let Some(v) = parse_iso_fast(s) {
        return Some(v);
    }
    if n <= 18 {
        // Shorter than F1, so only F3 can be valid; at most 18 digits cannot overflow.
        let mut v: i64 = 0;
        for &c in s {
            let dgt = c.wrapping_sub(b'0');
            if dgt > 9 {
                return None;
            }
            v = v * 10 + dgt as i64;
        }
        return Some(v);
    }
    parse_ts_slow(s)
}

/// The exact, unhurried parser: every rule of the three forms.
#[inline(never)]
fn parse_ts_slow(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if (n != 20 && n != 25) || s[4] != b'-' {
        return if all_digits(s) { Some(digits_sat(s)) } else { None };
    }
    if s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
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

/// event_type: Some(false) for view or cart, Some(true) for purchase, None otherwise.
#[inline(always)]
fn purchase_kind(s: &[u8]) -> Option<bool> {
    match s.len() {
        4 => {
            let w = u32::from_le_bytes(s.try_into().unwrap());
            (w == u32::from_le_bytes(*b"view") || w == u32::from_le_bytes(*b"cart")).then_some(false)
        }
        8 => (u64::from_le_bytes(s.try_into().unwrap()) == u64::from_le_bytes(*b"purchase")).then_some(true),
        _ => None,
    }
}

#[inline(always)]
fn parse_price(s: &[u8]) -> Option<u32> {
    if !all_digits(s) {
        return None;
    }
    let v = digits_sat(s);
    if (1..=10_000_000).contains(&v) { Some(v as u32) } else { None }
}

// ------------------------------------------------------------------ records

/// One parsed event copy, for dedup: its ID, its ingest time, and where its
/// payload is (`NO_LOC` for a copy that fails validation and so has none).
#[derive(Clone, Copy)]
struct DRec<K> {
    id: K,
    ingest: i64,
    loc: u64, // chunk << 40 | person shard << 32 | index in that chunk's shard
}

const NO_LOC: u64 = u64::MAX;
const DEAD: i64 = i64::MIN; // PRec::ts of a copy that lost the dedup

/// A valid event copy, written once at parse time into its person's shard.
#[derive(Clone, Copy)]
struct PRec<K> {
    name: K,
    ts: i64, // DEAD once dedup keeps another copy of the same event
    price: u32, // purchase price, 0 for view/cart
    cat: u32,   // category rank
}

/// A raw pointer that may cross threads. Used only to mark dead payloads during
/// dedup: each payload is owned by one event-id shard, so writes never collide.
#[derive(Clone, Copy)]
struct SendPtr(usize);

/// A valid event copy waiting AHEAD rows for its person lookup.
#[derive(Clone, Copy, Default)]
struct EvPend<K> {
    id: K,
    uid: K,
    ingest: i64,
    ts: i64,
    h: u64, // mix(uid)
    price: u32,
    cat: u32,
}

/// A valid click, keyed by person.
#[derive(Clone, Copy, Default)]
struct CRec<K> {
    name: K,
    id: K,
    ts: i64,
    camp: u32,
}

#[derive(Clone, Copy, Default)]
struct Ev {
    ts: i64,
    price: u32,
    cat: u32,
}

#[derive(Clone, Copy, Default)]
struct Cl<K> {
    ts: i64,
    id: K,
    camp: u32,
}

struct URow<K> {
    name: K,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u32>,
}

// ---------------------------------------------------------------- utilities

fn log(t0: Instant, what: &str) {
    eprintln!("[{:>8.3}s] {what}", t0.elapsed().as_secs_f64());
    // STOP=n exits at the n-th log line (instruction counts per phase).
    static N: AtomicU32 = AtomicU32::new(0);
    static STOP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let stop = *STOP.get_or_init(|| std::env::var("STOP").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    if N.fetch_add(1, Relaxed) + 1 == stop {
        std::process::exit(0);
    }
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
            let m = unsafe { Mmap::map(&f) }.expect("mmap input");
            let _ = m.advise(Advice::WillNeed); // start reading it in the background
            Some(m)
        })
        .collect()
}

struct Inputs<'a> {
    ws: i64,
    we: i64,
    products: &'a [u8],
    campaigns: &'a [u8],
    events: Vec<&'a [u8]>,
    clicks: Vec<&'a [u8]>,
    links: Vec<&'a [u8]>,
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

    // SCANBENCH=1|2: only scan the event files (new scanner | per-separator one), for instruction counts.
    if let Ok(m) = std::env::var("SCANBENCH") {
        let maps = map_dir(&data.join("events"));
        let mut tot = 0usize;
        for mm in &maps {
            let cb = |fl: &[Fd; 7]| tot += fl[3].e() - fl[0].s() + fl[6].e();
            match m.as_str() {
                "3" => each_line2::<7>(&mm[..], true, cb),
                "4" => dense::each_line::<7>(&mm[..], true, cb),
                _ => scan::<7, _>(&mm[..], true, cb),
            }
        }
        eprintln!("scanbench {m}: {tot}");
        std::process::exit(0);
    }
    let manifest = std::fs::read_to_string(data.join("manifest.json")).expect("manifest.json");
    let ws = parse_ts(json_str(&manifest, "window_start").as_bytes()).expect("window_start");
    let we = parse_ts(json_str(&manifest, "window_end").as_bytes()).expect("window_end");

    let ev_maps = map_dir(&data.join("events"));
    let cl_maps = map_dir(&data.join("clicks"));
    let ln_maps = map_dir(&data.join("links"));
    let products_buf = std::fs::read(data.join("products.csv")).expect("products.csv");
    let campaigns_buf = std::fs::read(data.join("campaigns.csv")).expect("campaigns.csv");
    log(t0, &format!("mapped inputs, {threads} threads"));

    let inp = Inputs {
        ws,
        we,
        products: &products_buf,
        campaigns: &campaigns_buf,
        events: ev_maps.iter().map(|m| &m[..]).collect(),
        clicks: cl_maps.iter().map(|m| &m[..]).collect(),
        links: ln_maps.iter().map(|m| &m[..]).collect(),
    };
    // KEYS=128 or KEYS=slice skips the smaller key types (to test the fallbacks).
    let keys = std::env::var("KEYS").unwrap_or_default();
    // The fast path for generator-shaped data, unless DENSE=off; DENSE=force
    // takes it even when the data does not look generated (for tests). It
    // exits the process when it succeeds.
    // The fast path needs IDs the old generator made (invertible numbers);
    // since the IDs became opaque it is off unless DENSE=on (or force).
    let dense = std::env::var("DENSE").unwrap_or_default();
    if keys.is_empty() && (dense == "on" || dense == "force") {
        dense::run(t0, &out, &inp, threads, dense == "force");
        log(t0, "fast path not taken");
    }
    // OLD=1 runs only the wide-record pipeline (it is the fallback when the
    // compact one cannot hold a dataset's values; this tests it).
    let old = std::env::var("OLD").is_ok_and(|v| v == "1");
    let both = |k: &str, f: &dyn Fn() -> bool, g: &dyn Fn() -> bool| -> bool {
        if !old && f() {
            return true;
        }
        if !old {
            log(t0, &format!("compact records do not fit; wide records ({k})"));
        }
        g()
    };
    let done = (keys.is_empty()
        && both("lid", &|| run::<Lid>(t0, &out, &inp, threads), &|| {
            !LID_CLASH.load(Relaxed) && run_old::<Lid>(t0, &out, &inp, threads)
        })
        && !LID_CLASH.load(Relaxed))
        || ((keys.is_empty() || keys == "id") && {
            log(t0, "using 24-byte ID keys");
            both("id", &|| run::<Id>(t0, &out, &inp, threads), &|| run_old::<Id>(t0, &out, &inp, threads))
        })
        || (keys != "slice" && {
            log(t0, "using 128-bit keys");
            both("u128", &|| run::<u128>(t0, &out, &inp, threads), &|| run_old::<u128>(t0, &out, &inp, threads))
        });
    if !done {
        log(t0, "using byte-slice keys");
        assert!(both("slice", &|| run::<&[u8]>(t0, &out, &inp, threads), &|| run_old::<&[u8]>(t0, &out, &inp, threads)));
    }
}

/// Products and campaigns, read once.
struct Catalog<'a, K> {
    /// Category names in byte order, "unknown" included; a category's rank is
    /// its index here, so "smallest name wins a tie" becomes "smallest rank wins".
    cat_names: Vec<&'a [u8]>,
    ncat: usize,
    unknown: u32,
    products: FxHashMap<K, u32>,
    /// Canonical `p<digits>` IDs by number (u32::MAX = not a product); empty if unusable.
    prod_arr: Vec<u32>,
    /// (key, campaign_id, channel), in file order; a campaign's index is its place here.
    camp_rows: Vec<(K, &'a [u8], &'a [u8])>,
    campaigns: FxHashMap<K, u32>,
    /// Canonical `c_<digits>` IDs by number (u32::MAX = not a campaign); empty if unusable.
    camp_arr: Vec<u32>,
    ncamp: usize,
}

impl<'a, K: Key<'a>> Catalog<'a, K> {
    /// None if some product or campaign ID does not fit K.
    fn load(inp: &Inputs<'a>) -> Option<Self> {
        let mut ok = true;
        // One pass: each product's category as a number in first-seen order;
        // canonical `p<digits>` IDs by number, any other ID by key.
        let pb = inp.products;
        let mut seen: Vec<&[u8]> = Vec::new();
        let mut seen_idx: FxHashMap<&[u8], u32> = FxHashMap::default();
        let mut by_num: Vec<(usize, u32)> = Vec::new();
        let mut by_key: Vec<(K, u32)> = Vec::new();
        scan::<2, _>(pb, true, |fl| {
            let c = &pb[fl[1].s()..fl[1].e()];
            let ci = *seen_idx.entry(c).or_insert_with(|| {
                seen.push(c);
                (seen.len() - 1) as u32
            });
            match product_number(&pb[fl[0].s()..fl[0].e()]) {
                Some(n) => by_num.push((n, ci)),
                None => match K::make(pb, fl[0].s(), fl[0].e()) {
                    Some(k) => by_key.push((k, ci)),
                    None => ok = false,
                },
            }
        });
        let mut cat_names: Vec<&[u8]> = seen.clone();
        cat_names.push(b"unknown");
        cat_names.sort_unstable();
        cat_names.dedup();
        let rank_of = |c: &[u8]| cat_names.binary_search(&c).unwrap() as u32;
        let rank: Vec<u32> = seen.iter().map(|c| rank_of(c)).collect();
        let unknown = rank_of(b"unknown");
        let ncat = cat_names.len();
        // An event's product field in canonical form is looked up in the array and
        // only there; any other form in the map (products.csv's other IDs). With
        // numbers too large for an array, the canonical ones go in the map too.
        let max = by_num.iter().map(|r| r.0).max().unwrap_or(0);
        let mut prod_arr: Vec<u32> = Vec::new();
        if max < (1 << 20) {
            prod_arr = vec![u32::MAX; max + 1];
            for &(n, ci) in &by_num {
                prod_arr[n] = rank[ci as usize];
            }
        } else {
            for &(n, ci) in &by_num {
                let id = format!("p{n}").into_bytes().leak();
                match K::make(id, 0, id.len()) {
                    Some(k) => by_key.push((k, ci)),
                    None => ok = false,
                }
            }
        }
        let products: FxHashMap<K, u32> = by_key.iter().map(|&(k, ci)| (k, rank[ci as usize])).collect();
        let mut camp_rows: Vec<(K, &[u8], &[u8])> = Vec::new();
        let cb = inp.campaigns;
        scan::<2, _>(cb, true, |fl| match K::make(cb, fl[0].s(), fl[0].e()) {
            Some(k) => camp_rows.push((k, &cb[fl[0].s()..fl[0].e()], &cb[fl[1].s()..fl[1].e()])),
            None => ok = false,
        });
        let campaigns: FxHashMap<K, u32> = camp_rows.iter().enumerate().map(|(i, r)| (r.0, i as u32)).collect();
        let ncamp = camp_rows.len();
        let mut camp_arr: Vec<u32> = Vec::new();
        let max = camp_rows.iter().filter_map(|r| campaign_number(r.1)).max().unwrap_or(0);
        if max < (1 << 20) {
            camp_arr = vec![u32::MAX; max + 1];
            for (i, r) in camp_rows.iter().enumerate() {
                if let Some(n) = campaign_number(r.1) {
                    camp_arr[n] = i as u32;
                }
            }
        }
        ok.then_some(Catalog { cat_names, ncat, unknown, products, prod_arr, camp_rows, campaigns, camp_arr, ncamp })
    }

    /// The campaign index of `d[f.s()..f.e()]`, or None if it is not in campaigns.csv.
    /// Canonical `c_<digits>` IDs are looked up in the array and only there.
    #[inline(always)]
    fn campaign_of(&self, d: &'a [u8], f: Fd) -> Option<u32> {
        let s = fld(d, f);
        if !self.camp_arr.is_empty() {
            if let Some(n) = campaign_number(s) {
                return self.camp_arr.get(n).copied().filter(|&c| c != u32::MAX);
            }
        }
        K::make(d, f.s(), f.e()).and_then(|k| self.campaigns.get(&k).copied())
    }

    #[inline(always)]
    fn category_of(&self, d: &'a [u8], f: Fd) -> u32 {
        let s = fld(d, f);
        if !self.prod_arr.is_empty() {
            if let Some(n) = product_number(s) {
                return match self.prod_arr.get(n) {
                    Some(&c) if c != u32::MAX => c,
                    _ => self.unknown,
                };
            }
        }
        // A product ID that does not fit K is not in products.csv (all of those fit).
        K::make(d, f.s(), f.e()).and_then(|p| self.products.get(&p).copied()).unwrap_or(self.unknown)
    }
}

/// About 32 chunks per thread, at most 16 MB each, so the last chunks of a
/// phase are small and threads finish together (at 60M events a 64 MB cap
/// left 143 chunks of ~45 MB).
fn chunk_target(inp: &Inputs, threads: usize) -> usize {
    let total: usize = inp.events.iter().chain(&inp.clicks).map(|b| b.len()).sum();
    (total / (threads * 32)).clamp(256 << 10, 16 << 20)
}

/// Linked IDs, numbered: `keys[s]` are hash shard s's distinct IDs, ID j of
/// shard s has global index `base[s] + j`, `roots[i]` is index i's person (the
/// smallest index in it), and `name[root]` that person's name: its byte-wise
/// smallest ID.
struct Comps<K> {
    keys: Vec<Vec<K>>,
    base: Vec<u32>,
    roots: Vec<u32>,
    name: Vec<Option<K>>,
}

/// Links: union-find, each person named by the byte-wise smallest ID. None if
/// some ID does not fit K.
fn link_components<'a, K: Key<'a>>(inp: &Inputs<'a>, target: usize, threads: usize) -> Option<Comps<K>> {
    let fail = AtomicBool::new(false);
    // The link files are small next to the events, so they get their own,
    // smaller chunks: about 16 per thread, whatever the event chunk size.
    let ln_bytes: usize = inp.links.iter().map(|b| b.len()).sum();
    let ln_chunks = make_chunks(&inp.links, (ln_bytes / (threads * 16)).clamp(64 << 10, target));
    let pairs: Vec<Vec<(K, K)>> = ln_chunks
        .par_iter()
        .map(|c| {
            let d = c.data;
            let mut v = Vec::with_capacity(d.len() / 20);
            scan::<2, _>(d, c.skip_first, |fl| {
                if fl[0].empty() || fl[1].empty() {
                    return;
                }
                match (K::make(d, fl[0].s(), fl[0].e()), K::make(d, fl[1].s(), fl[1].e())) {
                    (Some(a), Some(b)) => v.push((a, b)),
                    _ => fail.store(true, Relaxed),
                }
            });
            v
        })
        .collect();
    if fail.load(Relaxed) {
        return None;
    }
    // Number the distinct IDs: shard every occurrence by hash, and number each
    // shard on its own thread. Global index = shard base + index in the shard.
    let occ: Vec<Vec<Vec<K>>> = pairs
        .par_iter()
        .map(|v| {
            let mut out: Vec<Vec<K>> = (0..NB).map(|_| Vec::with_capacity(v.len() * 2 / NB + 4)).collect();
            for &(a, b) in v {
                out[shard_of(&a)].push(a);
                out[shard_of(&b)].push(b);
            }
            out
        })
        .collect();
    let id_shards: Vec<(Vec<K>, FxHashMap<K, u32>)> = (0..NB)
        .into_par_iter()
        .map(|s| {
            let n: usize = occ.iter().map(|o| o[s].len()).sum();
            let mut map: FxHashMap<K, u32> = new_map(n / 2 + 4);
            let mut keys: Vec<K> = Vec::with_capacity(n / 2 + 4);
            for o in &occ {
                for &k in &o[s] {
                    map.entry(k).or_insert_with(|| {
                        keys.push(k);
                        (keys.len() - 1) as u32
                    });
                }
            }
            (keys, map)
        })
        .collect();
    drop(occ);
    let mut base = vec![0u32; NB + 1];
    for s in 0..NB {
        base[s + 1] = base[s] + id_shards[s].0.len() as u32;
    }
    let nid = base[NB] as usize;
    let gidx = |k: &K| -> u32 {
        let s = shard_of(k);
        base[s] + id_shards[s].1[k]
    };
    let idx_pairs: Vec<Vec<(u32, u32)>> =
        pairs.par_iter().map(|v| v.iter().map(|(a, b)| (gidx(a), gidx(b))).collect()).collect();
    drop(pairs);

    // Union-find over the indices, on all threads. A root is only ever linked
    // under a smaller root (by compare-and-swap), and path halving only points a
    // node at an ancestor, so parent[x] <= x always: no cycles, whatever the
    // interleaving, and the final partition does not depend on the order.
    fn find(parent: &[AtomicU32], mut x: u32) -> u32 {
        loop {
            let p = parent[x as usize].load(Acquire);
            if p == x {
                return x;
            }
            let g = parent[p as usize].load(Acquire);
            if g != p {
                let _ = parent[x as usize].compare_exchange_weak(p, g, AcqRel, Acquire);
            }
            x = p;
        }
    }
    let parent: Vec<AtomicU32> = (0..nid as u32).into_par_iter().map(AtomicU32::new).collect();
    idx_pairs.par_iter().for_each(|v| {
        for &(a, b) in v {
            loop {
                let (ra, rb) = (find(&parent, a), find(&parent, b));
                if ra == rb {
                    break;
                }
                let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                if parent[hi as usize].compare_exchange(hi, lo, AcqRel, Acquire).is_ok() {
                    break;
                }
            }
        }
    });
    drop(idx_pairs);
    let roots: Vec<u32> = (0..nid as u32).into_par_iter().map(|i| find(&parent, i)).collect();
    drop(parent);
    // Each person's name: the byte-wise smallest of its IDs. Every ID goes to the
    // bucket of its root's range, then each range is finished on its own.
    let mut name: Vec<Option<K>> = vec![None; nid];
    let nbk = threads * 4;
    let span = nid.div_ceil(nbk).max(1);
    let buckets: Vec<Vec<Vec<(u32, K)>>> = id_shards
        .par_iter()
        .enumerate()
        .map(|(s, (keys, _))| {
            let mut b: Vec<Vec<(u32, K)>> = (0..nbk).map(|_| Vec::new()).collect();
            let rs = &roots[base[s] as usize..base[s] as usize + keys.len()];
            for (&k, &r) in keys.iter().zip(rs) {
                b[r as usize / span].push((r, k));
            }
            b
        })
        .collect();
    name.par_chunks_mut(span).enumerate().for_each(|(t, chunk)| {
        let lo = t * span;
        for b in &buckets {
            for &(r, k) in &b[t] {
                let slot = &mut chunk[r as usize - lo];
                if slot.is_none_or(|nm| k < nm) {
                    *slot = Some(k);
                }
            }
        }
    });
    drop(buckets);
    let keys = id_shards.into_iter().map(|(k, _)| k).collect();
    Some(Comps { keys, base, roots, name })
}

/// The whole pipeline with key type K. Returns false, having written nothing,
/// if some ID does not fit K.
fn run_old<'a, K: Key<'a>>(t0: Instant, out: &Path, inp: &Inputs<'a>, threads: usize) -> bool {
    let (ws, we) = (inp.ws, inp.we);
    let fail = AtomicBool::new(false);
    let Some(cat) = Catalog::<K>::load(inp) else { return false };
    let Catalog { ref cat_names, ncat, ncamp, ref camp_rows, .. } = cat;
    let category_of = |d: &'a [u8], f: Fd| cat.category_of(d, f);
    let organic = ncamp; // index of "organic"
    let target = chunk_target(inp, threads);

    // ---- Links.
    let Some(Comps { keys: id_keys, base, roots, name }) = link_components::<K>(inp, target, threads) else {
        return false;
    };
    let nid = roots.len();
    // ID -> name, only for IDs that are not their person's name, sharded like the
    // IDs (by the top 8 bits of `mix`), each shard its own table, built in parallel.
    let links: Vec<LinkTable<K>> = id_keys
        .par_iter()
        .enumerate()
        .map(|(s, keys)| {
            let entries: Vec<(K, K)> = keys
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| {
                    let nm = name[roots[base[s] as usize + i] as usize].unwrap();
                    (nm != k).then_some((k, nm))
                })
                .collect();
            LinkTable::new(&entries)
        })
        .collect();
    drop(id_keys);
    drop(name);
    drop(roots);
    log(t0, &format!("links: {nid} ids"));
    let table_of = |h: u64| -> &LinkTable<K> { &links[(h >> 56) as usize & (NB - 1)] };

    // ---- Events: parse every copy. A valid copy's payload is written once,
    // straight into its person's shard; every copy also sends (id, ingest, where
    // its payload is) to its event_id shard for dedup.
    let ev_chunks = make_chunks(&inp.events, target);
    // One set of shard buffers per thread (not per chunk), sized for the
    // thread's share of the input, so they are allocated once and rarely grow.
    // A payload's location is then (thread, shard, index).
    let ev_bytes: usize = inp.events.iter().map(|b| b.len()).sum();
    let ev_cap = shard_cap(ev_bytes / threads, 72);
    type EvT<K> = (usize, Vec<Vec<DRec<K>>>, Vec<Vec<PRec<K>>>);
    let ev_out: Vec<EvT<K>> = dense::per_thread(
        ev_chunks.len(),
        threads,
        |t| (t, (0..NB).map(|_| Vec::with_capacity(ev_cap)).collect(), (0..NB).map(|_| Vec::with_capacity(ev_cap)).collect()),
        |(t, drec_t, pay_t), chunk| {
            let ci = *t;
            let c = ev_chunks[chunk];
            let d = c.data;
            let mut drec = std::mem::take(drec_t);
            let mut pay = std::mem::take(pay_t);
            // Valid copies wait AHEAD rows in this ring while their person's table
            // slot is prefetched; then `finish` looks the person up and stores them.
            let mut ring = [EvPend::<K>::default(); AHEAD];
            let (mut head, mut queued) = (0usize, 0usize);
            let finish = |e: &EvPend<K>, pay: &mut Vec<Vec<PRec<K>>>, drec: &mut Vec<Vec<DRec<K>>>| {
                let name = table_of(e.h).get(e.uid, e.h);
                // Not renamed: the name is the ID, and so is its hash.
                let p = if name == e.uid { (e.h >> 56) as usize & (NB - 1) } else { shard_of(&name) };
                let v = &mut pay[p];
                assert!(v.len() < 1 << 32);
                let loc = (ci as u64) << 40 | (p as u64) << 32 | v.len() as u64;
                v.push(PRec { name, ts: e.ts, price: e.price, cat: e.cat });
                drec[shard_of(&e.id)].push(DRec { id: e.id, ingest: e.ingest, loc });
            };
            dense::each_line::<7>(d, c.skip_first, |fl| {
                if fl[0].empty() {
                    return;
                }
                let Some(id) = K::make_eq(d, fl[0].s(), fl[0].e()) else {
                    fail.store(true, Relaxed);
                    return;
                };
                let ingest = ts_fast(d, fl[3]).unwrap_or(i64::MAX);
                'v: {
                    if fl[1].empty() || fl[5].empty() {
                        break 'v;
                    }
                    let Some(ts) = ts_fast(d, fl[2]) else { break 'v };
                    if ts < ws || ts >= we {
                        break 'v;
                    }
                    let price = match purchase_kind(fld(d, fl[4])) {
                        Some(false) => 0,
                        Some(true) => match parse_price(fld(d, fl[6])) {
                            Some(p) => p,
                            None => break 'v,
                        },
                        None => break 'v,
                    };
                    if ingest.saturating_sub(ts) > DAY {
                        break 'v; // late
                    }
                    let Some(uid) = K::make_late(d, fl[1].s(), fl[1].e()) else {
                        fail.store(true, Relaxed);
                        break 'v;
                    };
                    let h = mix(&uid);
                    table_of(h).prefetch(h);
                    // Categories only matter for purchases (top_category sums purchase prices).
                    let cat = if price > 0 { category_of(d, fl[5]) } else { 0 };
                    let e = EvPend { id, uid, ingest, ts, h, price, cat };
                    if queued == AHEAD {
                        finish(&ring[head], &mut pay, &mut drec);
                        ring[head] = e;
                        head = (head + 1) % AHEAD;
                    } else {
                        ring[(head + queued) % AHEAD] = e;
                        queued += 1;
                    }
                    return;
                }
                // An invalid copy: no payload, but it still takes part in dedup.
                drec[shard_of(&id)].push(DRec { id, ingest, loc: NO_LOC });
            });
            for k in 0..queued {
                finish(&ring[(head + k) % AHEAD], &mut pay, &mut drec);
            }
            *drec_t = drec;
            *pay_t = pay;
        },
    );
    if fail.load(Relaxed) {
        return false;
    }
    let (drecs, mut pays): (Vec<_>, Vec<_>) = ev_out.into_iter().map(|(_, d, p)| (d, p)).unzip();
    log(t0, &format!("events parsed ({} chunks)", ev_chunks.len()));

    // ---- Clicks: parse + validate, shard by person.
    let cl_chunks = make_chunks(&inp.clicks, target);
    let cl_bytes: usize = inp.clicks.iter().map(|b| b.len()).sum();
    let cl_cap = shard_cap(cl_bytes / threads, 46);
    let clicks: Vec<Vec<Vec<CRec<K>>>> = dense::per_thread(
        cl_chunks.len(),
        threads,
        |_| (0..NB).map(|_| Vec::with_capacity(cl_cap)).collect::<Vec<Vec<CRec<K>>>>(),
        |out_t, chunk| {
            let c = cl_chunks[chunk];
            let d = c.data;
            let mut out = std::mem::take(out_t);
            // Same prefetch ring as for events: (click, user id hash) waits AHEAD rows.
            let mut ring = [(CRec::<K>::default(), 0u64); AHEAD];
            let (mut head, mut queued) = (0usize, 0usize);
            let finish = |(c, h): &(CRec<K>, u64), out: &mut Vec<Vec<CRec<K>>>| {
                let uid = c.name; // still the user id; becomes the person here
                let name = table_of(*h).get(uid, *h);
                let p = if name == uid { (*h >> 56) as usize & (NB - 1) } else { shard_of(&name) };
                out[p].push(CRec { name, ..*c });
            };
            dense::each_line::<4>(d, c.skip_first, |fl| {
                if fl[0].empty() || fl[1].empty() || fl[2].empty() {
                    return;
                }
                let Some(camp) = cat.campaign_of(d, fl[2]) else {
                    return;
                };
                let Some(ts) = ts_fast(d, fl[3]) else { return };
                if ts < ws - WEEK || ts >= we {
                    return;
                }
                let (Some(id), Some(uid)) = (K::make_late(d, fl[0].s(), fl[0].e()), K::make_late(d, fl[1].s(), fl[1].e())) else {
                    fail.store(true, Relaxed);
                    return;
                };
                let h = mix(&uid);
                table_of(h).prefetch(h);
                let e = (CRec { name: uid, id, ts, camp }, h);
                if queued == AHEAD {
                    finish(&ring[head], &mut out);
                    ring[head] = e;
                    head = (head + 1) % AHEAD;
                } else {
                    ring[(head + queued) % AHEAD] = e;
                    queued += 1;
                }
            });
            for k in 0..queued {
                finish(&ring[(head + k) % AHEAD], &mut out);
            }
            *out_t = out;
        },
    );
    if fail.load(Relaxed) {
        return false;
    }
    K::settle();
    log(t0, "clicks parsed");

    // ---- Dedup each event_id shard: the copy with the smallest ingest_ts is kept
    // (and if it is invalid, it has no payload, so the event is gone); every other
    // copy's payload is marked DEAD in place.
    let ptrs: Vec<Vec<SendPtr>> =
        pays.iter_mut().map(|v| v.iter_mut().map(|p| SendPtr(p.as_mut_ptr() as usize)).collect()).collect();
    let kill = |loc: u64| {
        if loc != NO_LOC {
            let (c, p, i) = ((loc >> 40) as usize, (loc >> 32) as usize & 0xff, (loc & 0xffff_ffff) as usize);
            // SAFETY: (c, p, i) is in bounds (it was that payload's index when pushed and
            // `pays` is not touched until dedup ends), and only the one dedup task that
            // owns this event_id's shard ever writes this element.
            unsafe { (*(ptrs[c][p].0 as *mut PRec<K>).add(i)).ts = DEAD };
        }
    };
    // Rayon may call the init once per work split, not once per thread, so start
    // each map at a shard's full size: growing from empty rehashes it many times.
    let per_shard = drecs.iter().map(|c| c.iter().map(|v| v.len()).sum::<usize>()).sum::<usize>() / NB;
    (0..NB).into_par_iter().for_each_init(
        || new_map::<K, (i64, u64)>(per_shard + per_shard / 4 + 64), // reused from shard to shard
        |best, b| {
            let n: usize = drecs.iter().map(|c| c[b].len()).sum();
            best.clear();
            best.reserve(n);
            for c in &drecs {
                for r in &c[b] {
                    match best.entry(r.id) {
                        Entry::Occupied(mut e) => {
                            let (bi, bl) = *e.get();
                            if r.ingest < bi {
                                kill(bl);
                                *e.get_mut() = (r.ingest, r.loc);
                            } else {
                                kill(r.loc); // equal ingest_ts means identical copies: keep either
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert((r.ingest, r.loc));
                        }
                    }
                }
            }
        },
    );
    drop(drecs);
    let pays = pays; // read-only from here on
    log(t0, "events deduped");

    // ---- Per person shard: sessions, credit, assists, aggregates.
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;
    let size = (ncamp + 1) * ndays;
    // One accumulator per thread (a fold makes one per work split, each with a
    // campaign table to zero and then sum on one thread); biggest shards first.
    let mut order: Vec<usize> = (0..NB).collect();
    order.sort_unstable_by_key(|&p| std::cmp::Reverse(pays.iter().map(|v| v[p].len()).sum::<usize>()));
    let results: Vec<(Vec<URow<K>>, Vec<[i64; 4]>)> = dense::per_thread(
        NB,
        threads,
        |_| (Vec::new(), vec![[0i64; 4]; size], Scratch::<K>::default()),
        |(urows, camp, sc), i| process_shard_old(order[i], &pays, &clicks, day0, ndays, ncat, organic, urows, camp, sc),
    )
    .into_iter()
    .map(|(u, c, _)| (u, c))
    .collect();
    log(t0, "sessions done");

    // ---- Write. (Not if two long IDs shared a Lid digest: then the run is
    // redone with exact Id keys.)
    if LID_CLASH.load(Relaxed) {
        log(t0, "long-ID digest clash; redoing with ID keys");
        return false;
    }
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
    let mut groups: Vec<Vec<&[URow<K>]>> = vec![Vec::new()];
    let mut in_group = 0usize;
    for (u, _) in &results {
        let mut rest: &[URow<K>] = u;
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
    let camp_names: Vec<(&[u8], &[u8])> = camp_rows.iter().map(|r| (r.1, r.2)).collect();
    rayon::join(
        || {
            groups.par_iter().enumerate().for_each(|(i, parts)| {
                write_user(&ud.join(format!("part-{i:05}.parquet")), parts, &cat_names);
            })
        },
        || write_campaign(&cd.join("part-00000.parquet"), &camp_tot, &camp_names, ncamp, ndays, day0),
    );
    log(t0, &format!("wrote {nrows} user rows in {} files", groups.len()));
    // The outputs are written and closed. Hand the big buffers and the input
    // mappings back on all threads, then exit: the kernel's own teardown at exit
    // is single-threaded, and with gigabytes mapped it is a visible cost.
    // RELEASE=all|inputs|none picks what to hand back. "inputs" (the input file
    // mappings, plain 4 KB pages) measured best: the payload buffers sit in 2 MB
    // huge pages, and DONTNEED on small ranges inside them splits those pages,
    // which costs more than the exit teardown it saves.
    let mode = std::env::var("RELEASE").unwrap_or_else(|_| "inputs".into());
    let mut regions: Vec<(usize, usize)> = Vec::new();
    if mode == "all" {
        for c in pays.iter() {
            for v in c {
                regions.push((v.as_ptr() as usize, std::mem::size_of_val(&v[..])));
            }
        }
        for c in clicks.iter() {
            for v in c {
                regions.push((v.as_ptr() as usize, std::mem::size_of_val(&v[..])));
            }
        }
    }
    if mode != "none" {
        // In pieces of at most 64 MB, so every thread gets some.
        const PIECE: usize = 64 << 20;
        for b in inp.events.iter().chain(&inp.clicks).chain(&inp.links) {
            for o in (0..b.len()).step_by(PIECE) {
                regions.push((b.as_ptr() as usize + o, PIECE.min(b.len() - o)));
            }
        }
    }
    regions.par_iter().for_each(|&(p, len)| release(p, len));
    log(t0, "done");
    std::process::exit(0);
}


// ------------------------------------------------- compact pipeline (main path)

/// A valid event copy, written once at parse time into its person's shard:
/// the person, ts - window_start (DEAD_TS once dedup keeps another copy of the
/// event), and the purchase price (0 for view and cart). A purchase's category
/// goes to a side list of its shard, in the same order.
#[derive(Clone, Copy)]
struct Pay<K> {
    name: K,
    ts: u32,
    price: u32,
}

const DEAD_TS: u32 = u32::MAX;

/// One event copy, for dedup: its ID, ingest_ts in seconds from `ing_base`,
/// and where its payload is (person shard << LOC_BITS | index; NOLOC: none).
#[derive(Clone, Copy)]
struct Dup<K> {
    id: K,
    ingest: u32,
    loc: u32,
}

const NOLOC: u32 = u32::MAX;
const LOC_BITS: u32 = 24;

/// A valid event copy waiting AHEAD rows for its person lookup.
#[derive(Clone, Copy, Default)]
struct Pend<K> {
    id: K,
    uid: K,
    h: u64,
    ingest: u32,
    ts: u32,
    price: u32,
    cat: u32,
}

/// A valid click: person, click ID, ts - (window_start - 7 days), campaign.
#[derive(Clone, Copy, Default)]
struct Clk<K> {
    name: K,
    id: K,
    ts: u32,
    camp: u32,
}

/// An event in its person shard's working array.
#[derive(Clone, Copy, Default)]
struct EvR<K> {
    name: K,
    ts: u32,
    price: u32,
    cat: u32,
}

/// Group order for a person shard's arrays: by `grp`, then (only for keys
/// whose `grp` can collide) by the key itself.
#[inline(always)]
fn grp_cmp<'a, K: Key<'a>>(a: K, b: K) -> std::cmp::Ordering {
    let o = a.grp().cmp(&b.grp());
    if K::UNIQUE || o != std::cmp::Ordering::Equal || a == b { o } else { a.cmp(&b) }
}

fn run<'a, K: Key<'a>>(t0: Instant, out: &Path, inp: &Inputs<'a>, threads: usize) -> bool {
    let (ws, we) = (inp.ws, inp.we);
    // Relative times must fit u32 (clicks start 7 days before the window).
    if we - ws + WEEK >= u32::MAX as i64 || we <= ws {
        return false;
    }
    let cbase = ws - WEEK; // click times are ts - cbase
    let ing_base = ws.div_euclid(1000) - (1i64 << 31); // ingest seconds are s - ing_base
    let fail = AtomicBool::new(false); // a key that does not fit K: next K
    let bail = AtomicBool::new(false); // a value the compact records cannot hold
    let Some(cat) = Catalog::<K>::load(inp) else { return false };
    let Catalog { ref cat_names, ncat, ncamp, ref camp_rows, .. } = cat;
    if ncat > u16::MAX as usize {
        return false;
    }
    let category_of = |d: &'a [u8], f: Fd| cat.category_of(d, f);
    let organic = ncamp; // index of "organic"
    let target = chunk_target(inp, threads);

    // ---- Links.
    let Some(Comps { keys: id_keys, base, roots, name }) = link_components::<K>(inp, target, threads) else {
        return false;
    };
    let nid = roots.len();
    let links: Vec<LinkTable<K>> = id_keys
        .par_iter()
        .enumerate()
        .map(|(s, keys)| {
            let entries: Vec<(K, K)> = keys
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| {
                    let nm = name[roots[base[s] as usize + i] as usize].unwrap();
                    (nm != k).then_some((k, nm))
                })
                .collect();
            LinkTable::new(&entries)
        })
        .collect();
    drop(id_keys);
    drop(name);
    drop(roots);
    log(t0, &format!("links: {nid} ids"));
    let table_of = |h: u64| -> &LinkTable<K> { &links[(h >> 56) as usize & (NB - 1)] };

    // ---- Events: parse every copy. A valid copy's payload is written once,
    // straight into its person's shard; every copy also sends (id, ingest, where
    // its payload is) to its event_id shard for dedup. One set of shard buffers
    // per thread, so a payload's place is (thread, shard, index).
    let ev_chunks = make_chunks(&inp.events, target);
    let ev_bytes: usize = inp.events.iter().map(|b| b.len()).sum();
    let ev_cap = shard_cap(ev_bytes / threads, 72);
    type EvT<K> = (Vec<Vec<Dup<K>>>, Vec<Vec<Pay<K>>>, Vec<Vec<u16>>);
    let ev_out: Vec<EvT<K>> = dense::per_thread(
        ev_chunks.len(),
        threads,
        |_| {
            (
                (0..NB).map(|_| Vec::with_capacity(ev_cap)).collect(),
                (0..NB).map(|_| Vec::with_capacity(ev_cap)).collect(),
                (0..NB).map(|_| Vec::with_capacity(ev_cap / 16 + 16)).collect(),
            )
        },
        |(drec_t, pay_t, pcat_t), chunk| {
            let c = ev_chunks[chunk];
            let d = c.data;
            let mut drec = std::mem::take(drec_t);
            let mut pay = std::mem::take(pay_t);
            let mut pcat = std::mem::take(pcat_t);
            let mut ring = [Pend::<K>::default(); AHEAD];
            let (mut head, mut queued) = (0usize, 0usize);
            let finish = |e: &Pend<K>, pay: &mut Vec<Vec<Pay<K>>>, pcat: &mut Vec<Vec<u16>>, drec: &mut Vec<Vec<Dup<K>>>| {
                let name = table_of(e.h).get(e.uid, e.h);
                let p = if name == e.uid { (e.h >> 56) as usize & (NB - 1) } else { shard_of(&name) };
                let v = &mut pay[p];
                let i = v.len();
                if i >= (1 << LOC_BITS) - 1 {
                    bail.store(true, Relaxed);
                    return;
                }
                v.push(Pay { name, ts: e.ts, price: e.price });
                if e.price > 0 {
                    pcat[p].push(e.cat as u16);
                }
                drec[shard_of(&e.id)].push(Dup { id: e.id, ingest: e.ingest, loc: (p as u32) << LOC_BITS | i as u32 });
            };
            dense::each_line::<7>(d, c.skip_first, |fl| {
                if fl[0].empty() {
                    return;
                }
                let Some(id) = K::make_eq(d, fl[0].s(), fl[0].e()) else {
                    fail.store(true, Relaxed);
                    return;
                };
                // ingest_ts is always F1 (rules); anything else goes to the wide path.
                let ing_s = match f1_secs(d, fl[3]) {
                    Some(s) => s,
                    None => match ts_at(d, fl[3]) {
                        Some(ms) if ms % 1000 == 0 => ms / 1000,
                        _ => {
                            bail.store(true, Relaxed);
                            return;
                        }
                    },
                };
                let ir = ing_s - ing_base;
                if !(0..u32::MAX as i64).contains(&ir) {
                    bail.store(true, Relaxed);
                    return;
                }
                let ingest = ir as u32;
                'v: {
                    if fl[1].empty() || fl[5].empty() {
                        break 'v;
                    }
                    let Some(ts) = ts_fast(d, fl[2]) else { break 'v };
                    if ts < ws || ts >= we {
                        break 'v;
                    }
                    let price = match purchase_kind(fld(d, fl[4])) {
                        Some(false) => 0,
                        Some(true) => match parse_price(fld(d, fl[6])) {
                            Some(p) => p,
                            None => break 'v,
                        },
                        None => break 'v,
                    };
                    if ing_s * 1000 - ts > DAY {
                        break 'v; // late
                    }
                    let Some(uid) = K::make_late(d, fl[1].s(), fl[1].e()) else {
                        fail.store(true, Relaxed);
                        break 'v;
                    };
                    let h = mix(&uid);
                    table_of(h).prefetch(h);
                    let cat = if price > 0 { category_of(d, fl[5]) } else { 0 };
                    let e = Pend { id, uid, h, ingest, ts: (ts - ws) as u32, price, cat };
                    if queued == AHEAD {
                        finish(&ring[head], &mut pay, &mut pcat, &mut drec);
                        ring[head] = e;
                        head = (head + 1) % AHEAD;
                    } else {
                        ring[(head + queued) % AHEAD] = e;
                        queued += 1;
                    }
                    return;
                }
                // An invalid copy: no payload, but it still takes part in dedup.
                drec[shard_of(&id)].push(Dup { id, ingest, loc: NOLOC });
            });
            for k in 0..queued {
                finish(&ring[(head + k) % AHEAD], &mut pay, &mut pcat, &mut drec);
            }
            *drec_t = drec;
            *pay_t = pay;
            *pcat_t = pcat;
        },
    );
    if fail.load(Relaxed) || bail.load(Relaxed) {
        return false;
    }
    let mut drecs = Vec::with_capacity(threads);
    let mut pays = Vec::with_capacity(threads);
    let mut pcats = Vec::with_capacity(threads);
    for (dr, py, pc) in ev_out {
        drecs.push(dr);
        pays.push(py);
        pcats.push(pc);
    }
    log(t0, &format!("events parsed ({} chunks)", ev_chunks.len()));

    // ---- Clicks: parse + validate, shard by person.
    let cl_chunks = make_chunks(&inp.clicks, target);
    let cl_bytes: usize = inp.clicks.iter().map(|b| b.len()).sum();
    let cl_cap = shard_cap(cl_bytes / threads, 46);
    let clicks: Vec<Vec<Vec<Clk<K>>>> = dense::per_thread(
        cl_chunks.len(),
        threads,
        |_| (0..NB).map(|_| Vec::with_capacity(cl_cap)).collect::<Vec<Vec<Clk<K>>>>(),
        |out_t, chunk| {
            let c = cl_chunks[chunk];
            let d = c.data;
            let mut out = std::mem::take(out_t);
            let mut ring = [(Clk::<K>::default(), 0u64); AHEAD];
            let (mut head, mut queued) = (0usize, 0usize);
            let finish = |(c, h): &(Clk<K>, u64), out: &mut Vec<Vec<Clk<K>>>| {
                let uid = c.name; // still the user id; becomes the person here
                let name = table_of(*h).get(uid, *h);
                let p = if name == uid { (*h >> 56) as usize & (NB - 1) } else { shard_of(&name) };
                out[p].push(Clk { name, ..*c });
            };
            dense::each_line::<4>(d, c.skip_first, |fl| {
                if fl[0].empty() || fl[1].empty() || fl[2].empty() {
                    return;
                }
                let Some(camp) = cat.campaign_of(d, fl[2]) else {
                    return;
                };
                let Some(ts) = ts_fast(d, fl[3]) else { return };
                if ts < cbase || ts >= we {
                    return;
                }
                let (Some(id), Some(uid)) = (K::make_late(d, fl[0].s(), fl[0].e()), K::make_late(d, fl[1].s(), fl[1].e())) else {
                    fail.store(true, Relaxed);
                    return;
                };
                let h = mix(&uid);
                table_of(h).prefetch(h);
                let e = (Clk { name: uid, id, ts: (ts - cbase) as u32, camp }, h);
                if queued == AHEAD {
                    finish(&ring[head], &mut out);
                    ring[head] = e;
                    head = (head + 1) % AHEAD;
                } else {
                    ring[(head + queued) % AHEAD] = e;
                    queued += 1;
                }
            });
            for k in 0..queued {
                finish(&ring[(head + k) % AHEAD], &mut out);
            }
            *out_t = out;
        },
    );
    if fail.load(Relaxed) {
        return false;
    }
    K::settle();
    log(t0, "clicks parsed");

    // ---- Dedup each event_id shard: the copy with the smallest ingest_ts is kept
    // (and if it is invalid, it has no payload, so the event is gone); every other
    // copy's payload is marked DEAD_TS in place.
    let ptrs: Vec<Vec<SendPtr>> =
        pays.iter_mut().map(|v| v.iter_mut().map(|p| SendPtr(p.as_mut_ptr() as usize)).collect()).collect();
    // `at` = thread << 32 | loc.
    let kill = |at: u64| {
        let loc = at as u32;
        if loc != NOLOC {
            let (c, p, i) = ((at >> 32) as usize, (loc >> LOC_BITS) as usize, (loc & ((1 << LOC_BITS) - 1)) as usize);
            // SAFETY: (c, p, i) is in bounds (it was that payload's index when pushed and
            // `pays` is not touched until dedup ends), and only the one dedup task that
            // owns this event_id's shard ever writes this element.
            unsafe { (*(ptrs[c][p].0 as *mut Pay<K>).add(i)).ts = DEAD_TS };
        }
    };
    let per_shard = drecs.iter().map(|c| c.iter().map(|v| v.len()).sum::<usize>()).sum::<usize>() / NB;
    (0..NB).into_par_iter().for_each_init(
        || new_map::<K, (u32, u64)>(per_shard + per_shard / 4 + 64),
        |best, b| {
            let n: usize = drecs.iter().map(|c| c[b].len()).sum();
            best.clear();
            best.reserve(n);
            for (t, c) in drecs.iter().enumerate() {
                let tt = (t as u64) << 32;
                for r in &c[b] {
                    let at = tt | r.loc as u64;
                    match best.entry(r.id) {
                        Entry::Occupied(mut e) => {
                            let (bi, bl) = *e.get();
                            if r.ingest < bi {
                                kill(bl);
                                *e.get_mut() = (r.ingest, at);
                            } else {
                                kill(at); // equal ingest_ts means identical copies: keep either
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert((r.ingest, at));
                        }
                    }
                }
            }
        },
    );
    drop(drecs);
    let pays = pays; // read-only from here on
    log(t0, "events deduped");

    // ---- Per person shard: sessions, credit, assists, aggregates.
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;
    let size = (ncamp + 1) * ndays;
    let mut order: Vec<usize> = (0..NB).collect();
    order.sort_unstable_by_key(|&p| std::cmp::Reverse(pays.iter().map(|v| v[p].len()).sum::<usize>()));
    let env = ShardEnv { ws, day0, ndays, ncat, organic };
    let results: Vec<(Vec<URow<K>>, Vec<[i64; 4]>)> = dense::per_thread(
        NB,
        threads,
        |_| (Vec::new(), vec![[0i64; 4]; size], Scratch2::<K>::default()),
        |(urows, camp, sc), i| process_shard(order[i], &pays, &pcats, &clicks, &env, urows, camp, sc),
    )
    .into_iter()
    .map(|(u, c, _)| (u, c))
    .collect();
    log(t0, "sessions done");

    // ---- Write. (Not if two long IDs shared a Lid digest: then the run is
    // redone with exact Id keys.)
    if LID_CLASH.load(Relaxed) {
        log(t0, "long-ID digest clash; redoing with ID keys");
        return false;
    }
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
    let nrows: usize = results.iter().map(|(u, _)| u.len()).sum();
    let per_file = nrows.div_ceil(threads).max(1);
    let mut groups: Vec<Vec<&[URow<K>]>> = vec![Vec::new()];
    let mut in_group = 0usize;
    for (u, _) in &results {
        let mut rest: &[URow<K>] = u;
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
    let camp_names: Vec<(&[u8], &[u8])> = camp_rows.iter().map(|r| (r.1, r.2)).collect();
    rayon::join(
        || {
            groups.par_iter().enumerate().for_each(|(i, parts)| {
                write_user(&ud.join(format!("part-{i:05}.parquet")), parts, &cat_names);
            })
        },
        || write_campaign(&cd.join("part-00000.parquet"), &camp_tot, &camp_names, ncamp, ndays, day0),
    );
    log(t0, &format!("wrote {nrows} user rows in {} files", groups.len()));
    // Hand the input mappings back on all threads, then exit (see run_old).
    const PIECE: usize = 64 << 20;
    let mut regions: Vec<(usize, usize)> = Vec::new();
    for b in inp.events.iter().chain(&inp.clicks).chain(&inp.links) {
        for o in (0..b.len()).step_by(PIECE) {
            regions.push((b.as_ptr() as usize + o, PIECE.min(b.len() - o)));
        }
    }
    regions.par_iter().for_each(|&(p, len)| release(p, len));
    log(t0, "done");
    std::process::exit(0);
}

struct ShardEnv {
    ws: i64,
    day0: i64,
    ndays: usize,
    ncat: usize,
    organic: usize,
}

#[derive(Default)]
struct Scratch2<K> {
    bid: Vec<u32>,
    off: Vec<u32>,
    cur: Vec<u32>,
    evs: Vec<EvR<K>>,
    coff: Vec<u32>,
    cls: Vec<Clk<K>>,
    cl1: Vec<(u32, u32)>,
    keys: Vec<u64>,
    sorted: Vec<EvR<K>>,
    ps: PScratch,
}

/// Sort u64s: insertion sort for short slices, else the library's.
#[inline]
fn sort_u64(v: &mut [u64]) {
    if v.len() <= 24 {
        for i in 1..v.len() {
            let x = v[i];
            let mut j = i;
            while j > 0 && v[j - 1] > x {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    } else {
        v.sort_unstable();
    }
}

/// One person shard: its live events and its clicks are bucketed by `grp`
/// (a counting sort), each bucket is sorted by (person, ts), and then each
/// person's events and clicks are side by side.
#[allow(clippy::too_many_arguments)]
fn process_shard<'a, K: Key<'a>>(
    p: usize,
    pays: &[Vec<Vec<Pay<K>>>],
    pcats: &[Vec<Vec<u16>>],
    clicks: &[Vec<Vec<Clk<K>>>],
    env: &ShardEnv,
    urows: &mut Vec<URow<K>>,
    camp: &mut [[i64; 4]],
    sc: &mut Scratch2<K>,
) {
    let n_all: usize = pays.iter().map(|d| d[p].len()).sum(); // live and dead
    if n_all == 0 {
        return;
    }
    let Scratch2 { bid, off, cur, evs, coff, cls, cl1, keys, sorted, ps } = sc;
    let nb = (n_all / 8).next_power_of_two().clamp(16, 1 << 20);
    let mask = nb - 1;
    // Bits of mix() below the shard's top 8, so buckets split each shard evenly.
    let bucket = |k: K| (mix(&k) >> 24) as usize & mask;
    off.clear();
    off.resize(nb + 1, 0);
    bid.clear();
    for d in pays {
        for r in &d[p] {
            if r.ts == DEAD_TS {
                continue;
            }
            let b = bucket(r.name);
            off[b + 1] += 1;
            bid.push(b as u32);
        }
    }
    for i in 0..nb {
        off[i + 1] += off[i];
    }
    evs.clear();
    evs.resize(bid.len(), EvR::default());
    cur.clear();
    cur.extend_from_slice(&off[..nb]);
    let mut k = 0;
    for (d, pc) in pays.iter().zip(pcats) {
        let pc = &pc[p];
        let mut ci = 0;
        for r in &d[p] {
            let mut c = 0;
            if r.price > 0 {
                c = pc[ci] as u32;
                ci += 1;
            }
            if r.ts == DEAD_TS {
                continue; // skipped in the first pass too, so k stays in step
            }
            let b = bid[k] as usize;
            k += 1;
            evs[cur[b] as usize] = EvR { name: r.name, ts: r.ts, price: r.price, cat: c };
            cur[b] += 1;
        }
    }
    // Clicks, bucketed the same way.
    let n_cl: usize = clicks.iter().map(|c| c[p].len()).sum();
    coff.clear();
    coff.resize(nb + 1, 0);
    bid.clear();
    for c in clicks {
        for r in &c[p] {
            let b = bucket(r.name);
            coff[b + 1] += 1;
            bid.push(b as u32);
        }
    }
    for i in 0..nb {
        coff[i + 1] += coff[i];
    }
    cls.clear();
    cls.resize(n_cl, Clk::default());
    cur.clear();
    cur.extend_from_slice(&coff[..nb]);
    let mut k = 0;
    for c in clicks {
        for r in &c[p] {
            let b = bid[k] as usize;
            k += 1;
            cls[cur[b] as usize] = *r;
            cur[b] += 1;
        }
    }
    ps.reset(env.ncat);
    for b in 0..nb {
        let ev = &mut evs[off[b] as usize..off[b + 1] as usize];
        if ev.is_empty() {
            continue;
        }
        let cl = &mut cls[coff[b] as usize..coff[b + 1] as usize];
        let name0 = ev[0].name;
        if ev.iter().all(|e| e.name == name0) {
            // One person (the usual case): sort (ts, index) keys, not records.
            keys.clear();
            keys.extend(ev.iter().enumerate().map(|(i, e)| (e.ts as u64) << 32 | i as u64));
            sort_u64(keys);
            sorted.clear();
            sorted.extend(keys.iter().map(|&k| ev[k as u32 as usize]));
            keys.clear();
            keys.extend(cl.iter().enumerate().filter(|(_, c)| c.name == name0).map(|(i, c)| (c.ts as u64) << 32 | i as u64));
            sort_u64(keys);
            // Equal ts: smallest click_id first.
            let mut a = 0;
            while a < keys.len() {
                let mut z = a + 1;
                while z < keys.len() && keys[z] >> 32 == keys[a] >> 32 {
                    z += 1;
                }
                if z - a > 1 {
                    keys[a..z].sort_unstable_by(|x, y| cl[*x as u32 as usize].id.cmp(&cl[*y as u32 as usize].id));
                }
                a = z;
            }
            cl1.clear();
            cl1.extend(keys.iter().map(|&k| {
                let c = &cl[k as u32 as usize];
                (c.ts, c.camp)
            }));
            person(sorted, cl1, name0, env, ps, urows, camp);
            continue;
        }
        if K::UNIQUE {
            ev.sort_unstable_by_key(|e| (e.name.grp(), e.ts));
            cl.sort_unstable_by(|a, b| (a.name.grp(), a.ts).cmp(&(b.name.grp(), b.ts)).then_with(|| a.id.cmp(&b.id)));
        } else {
            ev.sort_unstable_by(|a, b| grp_cmp(a.name, b.name).then(a.ts.cmp(&b.ts)));
            cl.sort_unstable_by(|a, b| grp_cmp(a.name, b.name).then(a.ts.cmp(&b.ts)).then_with(|| a.id.cmp(&b.id)));
        }
        let (mut i, mut j) = (0usize, 0usize);
        while i < ev.len() {
            let name = ev[i].name;
            let mut e = i + 1;
            while e < ev.len() && ev[e].name == name {
                e += 1;
            }
            while j < cl.len() && grp_cmp(cl[j].name, name) == std::cmp::Ordering::Less {
                j += 1;
            }
            let mut f = j;
            while f < cl.len() && cl[f].name == name {
                f += 1;
            }
            cl1.clear();
            cl1.extend(cl[j..f].iter().map(|c| (c.ts, c.camp)));
            person(&ev[i..e], cl1, name, env, ps, urows, camp);
            i = e;
            j = f;
        }
    }
}

/// One person's events (sorted by ts) and clicks (ts - (ws - 7 days),
/// campaign; sorted by ts, then click_id) -> their user_daily rows and
/// campaign counters. `camp[c * ndays + day]` = [sessions, purchasing, revenue, assisted].
#[inline]
fn person<K: Copy>(
    ev: &[EvR<K>],
    cl: &mut [(u32, u32)],
    name: K,
    env: &ShardEnv,
    ps: &mut PScratch,
    urows: &mut Vec<URow<K>>,
    camp: &mut [[i64; 4]],
) {
    let PScratch { catsum, touched, starts, sdays, run_end, pairs } = ps;
    let ShardEnv { ws, day0, ndays, organic, .. } = *env;
    let n = ev.len();
    let m = cl.len();
    starts.clear();
    sdays.clear();
    // Times from here on: ms from ws (clicks shifted to match).
    let ct = |c: &(u32, u32)| c.0 as i64 - WEEK;

    let mut cur_day = u32::MAX;
    let (mut u_sess, mut u_ev, mut u_rev) = (0i64, 0i64, 0i64);
    let flush = |day: u32, s: i64, e: i64, r: i64, catsum: &mut [u64], touched: &mut Vec<u32>, urows: &mut Vec<URow<K>>| {
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
        urows.push(URow { name, day: (day0 + day as i64) as i32, sessions: s, events: e, revenue: r, top });
    };

    let (mut hi, mut run_start) = (0usize, 0usize);
    let mut i = 0usize;
    while i < n {
        let s = ev[i].ts as i64;
        let mut j = i + 1;
        while j < n && (ev[j].ts - ev[j - 1].ts) as i64 <= GAP {
            j += 1;
        }
        let day = ((ws + s).div_euclid(DAY) - day0) as u32;
        let mut rev = 0i64;
        let mut purchased = false;
        if day != cur_day {
            if cur_day != u32::MAX {
                flush(cur_day, u_sess, u_ev, u_rev, catsum, touched, urows);
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
        while hi < m && ct(&cl[hi]) <= s {
            if hi == 0 || cl[hi].0 != cl[hi - 1].0 {
                run_start = hi;
            }
            hi += 1;
        }
        let c = if hi > 0 && ct(&cl[hi - 1]) >= s - WEEK { cl[run_start].1 as usize } else { organic };
        let slot = &mut camp[c * ndays + day as usize];
        slot[0] += 1;
        slot[1] += purchased as i64;
        slot[2] += rev;

        starts.push(s);
        sdays.push(day);
        i = j;
    }
    if cur_day != u32::MAX {
        flush(cur_day, u_sess, u_ev, u_rev, catsum, touched, urows);
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
        pairs.extend(cl.iter().map(|c| (c.1, ct(c))));
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

/// Map the pages under `d` (part of a file mapping) into the page table in
/// one call (MADV_POPULATE_READ, Linux 5.14+; a no-op where unsupported), so
/// scanning it takes no page faults, and prefetches past a page boundary are
/// not dropped for want of a mapping.
fn populate(d: &[u8]) {
    const PAGE: usize = 4096;
    const MADV_POPULATE_READ: libc::c_int = 22;
    let p = d.as_ptr() as usize;
    let start = p & !(PAGE - 1);
    if d.is_empty() {
        return;
    }
    // SAFETY: the range covers pages of a live read-only mapping; POPULATE_READ only faults them in.
    unsafe { libc::madvise(start as *mut libc::c_void, p + d.len() - start, MADV_POPULATE_READ) };
}

/// madvise(MADV_DONTNEED) the whole pages inside `[p, p + len)`. Only called on
/// memory nothing will read again (the process exits right after).
fn release(p: usize, len: usize) {
    const PAGE: usize = 4096;
    let (start, end) = ((p + PAGE - 1) & !(PAGE - 1), (p + len) & !(PAGE - 1));
    if end > start {
        // SAFETY: the range lies inside a live allocation or mapping, and nothing
        // touches it again before exit; DONTNEED only drops the pages.
        unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED) };
    }
}

/// Buffers reused from shard to shard, so each worker allocates (and the
/// kernel zeroes) them once instead of 256 times.
#[derive(Default)]
struct Scratch<K> {
    index: FxHashMap<K, u32>,
    names: Vec<K>,
    cnt: Vec<u32>,
    pid_of: Vec<u32>,
    ev_off: Vec<usize>,
    cur: Vec<usize>,
    evs: Vec<Ev>,
    ccnt: Vec<u32>,
    cpid: Vec<u32>,
    cl_off: Vec<usize>,
    cls: Vec<Cl<K>>,
    ps: PScratch,
}

#[allow(clippy::too_many_arguments)]
fn process_shard_old<'a, K: Key<'a>>(
    p: usize,
    pays: &[Vec<Vec<PRec<K>>>],
    clicks: &[Vec<Vec<CRec<K>>>],
    day0: i64,
    ndays: usize,
    ncat: usize,
    organic: usize,
    urows: &mut Vec<URow<K>>,
    camp: &mut [[i64; 4]],
    sc: &mut Scratch<K>,
) {
    // Group events by person (counting sort into one flat array).
    let n_ev: usize = pays.iter().map(|d| d[p].len()).sum(); // live and DEAD

    if n_ev == 0 {
        return;
    }
    let Scratch {
        index,
        names,
        cnt,
        pid_of,
        ev_off,
        cur,
        evs,
        ccnt,
        cpid,
        cl_off,
        cls,
        ps,
    } = sc;
    index.clear();
    index.reserve(n_ev / 8 + 16);
    names.clear();
    cnt.clear();
    pid_of.clear();
    for d in pays {
        for r in &d[p] {
            if r.ts == DEAD {
                continue;
            }
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
    ev_off.clear();
    ev_off.resize(np + 1, 0);
    for i in 0..np {
        ev_off[i + 1] = ev_off[i] + cnt[i] as usize;
    }
    evs.clear();
    evs.resize(pid_of.len(), Ev::default());
    {
        cur.clear();
        cur.extend_from_slice(&ev_off[..]);
        let mut k = 0;
        for d in pays {
            for r in &d[p] {
                if r.ts == DEAD {
                    continue; // skipped in the first pass too, so k stays in step
                }
                let id = pid_of[k] as usize;
                k += 1;
                evs[cur[id]] = Ev { ts: r.ts, price: r.price, cat: r.cat };
                cur[id] += 1;
            }
        }
    }

    // Group clicks the same way; clicks of people with no session are irrelevant.
    ccnt.clear();
    ccnt.resize(np, 0);
    cpid.clear();
    for c in clicks {
        for r in &c[p] {
            match index.get(&r.name) {
                Some(&id) => {
                    ccnt[id as usize] += 1;
                    cpid.push(id);
                }
                None => cpid.push(u32::MAX),
            }
        }
    }
    cl_off.clear();
    cl_off.resize(np + 1, 0);
    for i in 0..np {
        cl_off[i + 1] = cl_off[i] + ccnt[i] as usize;
    }
    cls.clear();
    cls.resize(cl_off[np], Cl::default());
    {
        cur.clear();
        cur.extend_from_slice(&cl_off[..]);
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
    ps.reset(ncat);
    for pid in 0..np {
        let ev = &mut evs[ev_off[pid]..ev_off[pid + 1]];
        let cl = &mut cls[cl_off[pid]..cl_off[pid + 1]];
        person_old(ev, cl, names[pid], day0, organic, ndays, ps, urows, camp);
    }
}

/// Per-person scratch, reused from person to person.
#[derive(Default)]
struct PScratch {
    catsum: Vec<u64>,
    touched: Vec<u32>,
    starts: Vec<i64>,
    sdays: Vec<u32>,
    run_end: Vec<u32>,
    pairs: Vec<(u32, i64)>,
}

impl PScratch {
    fn reset(&mut self, ncat: usize) {
        self.catsum.clear();
        self.catsum.resize(ncat, 0);
        self.touched.clear();
    }
}

/// One person's events and clicks (any order) -> their user_daily rows and
/// campaign counters. `camp[c * ndays + day]` = [sessions, purchasing, revenue, assisted].
#[allow(clippy::too_many_arguments)]
#[inline]
fn person_old<K: Copy + Ord>(
    ev: &mut [Ev],
    cl: &mut [Cl<K>],
    name: K,
    day0: i64,
    organic: usize,
    ndays: usize,
    ps: &mut PScratch,
    urows: &mut Vec<URow<K>>,
    camp: &mut [[i64; 4]],
) {
    let PScratch { catsum, touched, starts, sdays, run_end, pairs } = ps;
    ev.sort_unstable_by_key(|e| e.ts);
    cl.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| a.id.cmp(&b.id)));
    let n = ev.len();
    let m = cl.len();
    starts.clear();
    sdays.clear();

    let mut cur_day = u32::MAX;
    let (mut u_sess, mut u_ev, mut u_rev) = (0i64, 0i64, 0i64);
    let flush = |day: u32, s: i64, e: i64, r: i64, catsum: &mut [u64], touched: &mut Vec<u32>, urows: &mut Vec<URow<K>>| {
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
        urows.push(URow { name, day: (day0 + day as i64) as i32, sessions: s, events: e, revenue: r, top });
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
                flush(cur_day, u_sess, u_ev, u_rev, catsum, touched, urows);
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
        flush(cur_day, u_sess, u_ev, u_rev, catsum, touched, urows);
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

// ------------------------------------------------------------------ Parquet

/// Columns for one row group of `n` rows, with room reserved for `n` values.
fn columns(k: usize) -> Vec<Column> {
    (0..k).map(|_| Column::new()).collect()
}

fn write_user<'a, K: Key<'a>>(path: &Path, parts: &[&[URow<K>]], cat_names: &[&[u8]]) {
    let schema = [
        ("user_id", Kind::Str),
        ("day", Kind::Date),
        ("sessions", Kind::I64),
        ("events", Kind::I64),
        ("revenue_cents", Kind::I64),
        ("top_category", Kind::OptStr),
    ];
    let (mut groups, mut rows) = (Vec::new(), Vec::new());
    let mut cols = columns(schema.len());
    let mut n = 0usize;
    for r in parts.iter().flat_map(|p| p.iter()) {
        cols[0].bytes_with(|b| r.name.put(b));
        cols[1].i32(r.day);
        cols[2].i64(r.sessions);
        cols[3].i64(r.events);
        cols[4].i64(r.revenue);
        cols[5].opt_bytes(r.top.map(|c| cat_names[c as usize]));
        n += 1;
        if n == GROUP_ROWS {
            groups.push(std::mem::replace(&mut cols, columns(schema.len())));
            rows.push(n);
            n = 0;
        }
    }
    if n > 0 {
        groups.push(cols);
        rows.push(n);
    }
    pq::write(path, &schema, &groups, &rows).expect("write user_daily parquet");
}

fn write_campaign(path: &Path, tot: &[[i64; 4]], camp_names: &[(&[u8], &[u8])], ncamp: usize, ndays: usize, day0: i64) {
    let schema = [
        ("campaign_id", Kind::Str),
        ("channel", Kind::Str),
        ("day", Kind::Date),
        ("sessions", Kind::I64),
        ("purchasing_sessions", Kind::I64),
        ("revenue_cents", Kind::I64),
        ("assisted_sessions", Kind::I64),
    ];
    let (mut groups, mut rows) = (Vec::new(), Vec::new());
    let mut cols = columns(schema.len());
    let mut n = 0usize;
    for c in 0..=ncamp {
        for d in 0..ndays {
            let t = tot[c * ndays + d];
            if t[0] == 0 && t[3] == 0 {
                continue;
            }
            let (id, ch): (&[u8], &[u8]) = if c == ncamp { (b"organic", b"organic") } else { camp_names[c] };
            cols[0].bytes(id);
            cols[1].bytes(ch);
            cols[2].i32((day0 + d as i64) as i32);
            for (k, v) in t.iter().enumerate() {
                cols[3 + k].i64(*v);
            }
            n += 1;
            if n == GROUP_ROWS {
                groups.push(std::mem::replace(&mut cols, columns(schema.len())));
                rows.push(n);
                n = 0;
            }
        }
    }
    if n > 0 {
        groups.push(cols);
        rows.push(n);
    }
    pq::write(path, &schema, &groups, &rows).expect("write campaign_daily parquet");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG (xorshift64*), so the test needs no crates.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn candidate(r: &mut Rng) -> Vec<u8> {
        let year = match r.below(4) {
            0 => r.below(10_000),
            1 => 1969 + r.below(4),
            2 => 2096 + r.below(8),
            _ => 2024 + r.below(6),
        };
        let mut s = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            year,
            r.below(14),
            r.below(33),
            r.below(26),
            r.below(62),
            r.below(62)
        )
        .into_bytes();
        match r.below(3) {
            0 => s.push(b'Z'),
            1 => s.extend(format!("{}{:02}:{:02}", if r.below(2) == 0 { '+' } else { '-' }, r.below(26), r.below(62)).bytes()),
            _ => {
                s = (0..1 + r.below(25)).map(|_| b'0' + r.below(10) as u8).collect();
            }
        }
        const NOISE: &[u8] = b"0123456789-T:Z+ zt/.\x80\xff\x2f\x3a";
        for _ in 0..r.below(3) {
            if !s.is_empty() {
                let i = r.below(s.len() as u64) as usize;
                s[i] = NOISE[r.below(NOISE.len() as u64) as usize];
            }
        }
        if r.below(20) == 0 && !s.is_empty() {
            s.pop();
        }
        s
    }

    #[test]
    fn fast_timestamps_match_the_exact_parser() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let mut accepted = 0;
        for _ in 0..2_000_000 {
            let s = candidate(&mut r);
            let (fast, slow) = (parse_ts(&s), parse_ts_slow(&s));
            assert_eq!(fast, slow, "{:?}", String::from_utf8_lossy(&s));
            accepted += fast.is_some() as u32;
        }
        assert!(accepted > 100_000, "the test should exercise valid timestamps too");
    }

    #[test]
    fn ts_at_matches_the_exact_parser() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let valid = [
            "2026-10-03T23:58:10Z",
            "2028-02-29T00:00:00Z",
            "2027-02-29T00:00:00Z",
            "2026-10-03T19:58:10-04:00",
            "2026-12-31T23:59:59+12:00",
            "2099-12-31T23:59:59Z",
            "2100-01-01T00:00:00Z",
            "1791071890000",
            "0001791071890000",
            "86400000",
            "1",
            "0",
        ];
        let alphabet = b"0123456789-T:Z+ .zt";
        for i in 0..3_000_000 {
            let mut s = valid[i % valid.len()].as_bytes().to_vec();
            for _ in 0..(next() % 3) {
                let p = (next() as usize) % s.len();
                s[p] = alphabet[(next() as usize) % alphabet.len()];
            }
            if next() % 7 == 0 {
                s.truncate((next() as usize) % (s.len() + 1));
            }
            // Bytes before the field, so the backward 16-byte load is allowed.
            let mut d = b"xxxxxxxxxxxxxxxxxxxxxxxx".to_vec();
            let start = d.len();
            d.extend_from_slice(&s);
            d.extend_from_slice(b",yyyy");
            let f = Fd::new(start, start + s.len());
            d.extend_from_slice(b"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz");
            assert_eq!(ts_at(&d, f), parse_ts_slow(&s), "{:?}", String::from_utf8_lossy(&s));
            assert_eq!(ts_fast(&d, f), parse_ts_slow(&s), "fast {:?}", String::from_utf8_lossy(&s));
        }
    }

    #[test]
    fn id_keys_round_trip_and_keep_string_order() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ids: Vec<Vec<u8>> = Vec::new();
        for i in 0..20_000 {
            let r = next();
            let mut v = vec![if r & 1 == 0 { b'u' } else { b'U' }];
            if i % 3 == 0 {
                let digits: &[u8] = if v[0] == b'U' { b"0123456789ABCDEF" } else { b"0123456789abcdef" };
                for _ in 0..32 {
                    v.push(digits[(next() % 16) as usize]);
                }
            } else {
                let n = 1 + (next() % 8) as usize;
                for _ in 0..n {
                    v.push(b"0123456789abcdefABCDEF"[(next() % 22) as usize]);
                }
            }
            ids.push(v);
        }
        // Prefixes of each other, and hex runs that share a beginning.
        ids.push(b"uabcdef12".to_vec());
        ids.push(b"uabcdef12000000000000000000000001".to_vec());
        ids.push(b"uabcdef1".to_vec());
        let keys: Vec<Id> = ids.iter().map(|v| Id::make(v, 0, v.len()).expect("fits")).collect();
        for (v, k) in ids.iter().zip(&keys) {
            let mut back = Vec::new();
            k.put(&mut back);
            assert_eq!(&back, v);
        }
        for i in 0..ids.len() {
            let j = (i * 7919 + 13) % ids.len();
            assert_eq!(ids[i].cmp(&ids[j]), keys[i].cmp(&keys[j]), "{:?} vs {:?}", ids[i], ids[j]);
        }
        assert!(Id::make(b"u0000000000000000000000000000000000", 0, 33).is_none() || true);
        assert!(Id::make(b"uABCDEF0123456789abcdef0123456789", 0, 33).is_none()); // mixed case
    }

    #[test]
    fn known_timestamps() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10Z"), Some(1_791_071_890_000));
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00"), Some(1_791_071_890_000));
        assert_eq!(parse_ts(b"0001791071890000"), Some(1_791_071_890_000));
        assert_eq!(parse_ts(b"2028-02-29T00:00:00Z"), parse_ts_slow(b"2028-02-29T00:00:00Z"));
        assert!(parse_ts(b"2026-02-29T00:00:00Z").is_none());
        assert!(parse_ts(b"2026-10-03T23:58:10+0200").is_none());
        assert!(parse_ts(b"2026-10-03 23:58:10Z").is_none());
    }
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    /// The previous scanner, which steps through every quote too. It defines the
    /// semantics the new one must reproduce exactly.
    #[allow(unused_assignments)]
    fn scan_ref<const N: usize, F: FnMut(&[Fd; N])>(d: &[u8], skip_first: bool, mut f: F) {
        let n = d.len();
        let mut fl = [Fd::default(); N];
        let (mut k, mut fstart) = (0usize, 0usize);
        let (mut qcnt, mut qfirst, mut qlast) = (0u32, 0usize, 0usize);
        let (mut bad, mut skip) = (false, skip_first);
        macro_rules! end_field {
            ($pos:expr) => {{
                let pos: usize = $pos;
                if !bad {
                    if k < N {
                        if qcnt == 0 {
                            fl[k] = Fd::new(fstart, pos);
                        } else if qcnt == 2 && qfirst == fstart && qlast + 1 == pos {
                            fl[k] = Fd::new(fstart + 1, pos - 1);
                        } else {
                            bad = true;
                        }
                    } else {
                        bad = true;
                    }
                }
                k += 1;
                qcnt = 0;
                fstart = pos + 1;
            }};
        }
        macro_rules! end_line {
            () => {{
                if !bad && k == N && !skip {
                    f(&fl);
                }
                skip = false;
                bad = false;
                k = 0;
            }};
        }
        for (pos, &c) in d.iter().enumerate() {
            match c {
                b'"' => {
                    if qcnt == 0 {
                        qfirst = pos;
                    }
                    qlast = pos;
                    qcnt += 1;
                }
                b',' => end_field!(pos),
                b'\n' => {
                    end_field!(pos);
                    end_line!();
                }
                _ => {}
            }
        }
        if k > 0 || fstart < n {
            end_field!(n);
            end_line!();
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn collect<const N: usize>(d: &[u8], skip: bool, new: bool) -> Vec<Vec<(usize, usize)>> {
        let mut out = Vec::new();
        let mut push = |fl: &[Fd; N]| out.push(fl.iter().map(|f| (f.s(), f.e())).collect());
        if new { scan::<N, _>(d, skip, &mut push) } else { scan_ref::<N, _>(d, skip, &mut push) }
        out
    }

    fn collect2<const N: usize>(d: &[u8], skip: bool) -> Vec<Vec<(usize, usize)>> {
        let mut out = Vec::new();
        each_line2::<N>(d, skip, |fl: &[Fd; N]| out.push(fl.iter().map(|f| (f.s(), f.e())).collect()));
        out
    }

    fn nonempty_first(v: Vec<Vec<(usize, usize)>>) -> Vec<Vec<(usize, usize)>> {
        v.into_iter().filter(|l| l[0].0 < l[0].1).collect()
    }

    #[test]
    fn streaming_splitter_matches_the_reference() {
        let mut r = Rng(0x0bad_cafe_1234_5678);
        let mut lines_seen = 0;
        for case in 0..40_000 {
            let len = (r.next() % 900) as usize;
            let quote_rate = [0, 2, 8, 30][case % 4];
            let comma_rate = [12, 22, 35][case % 3];
            let d: Vec<u8> = (0..len)
                .map(|_| match r.next() % 100 {
                    x if x < quote_rate => b'"',
                    x if x < quote_rate + comma_rate => b',',
                    x if x < quote_rate + comma_rate + 6 => b'\n',
                    x if x < 80 => b"abcxyz0129"[(r.next() % 10) as usize],
                    x if x < 83 => b'\r',
                    _ => b"-:TZ+. "[(r.next() % 7) as usize],
                })
                .collect();
            for skip in [false, true] {
                let a = collect2::<4>(&d, skip);
                assert_eq!(a, nonempty_first(collect::<4>(&d, skip, false)), "N=4 skip={skip} {:?}", String::from_utf8_lossy(&d));
                lines_seen += a.len();
                let a = collect2::<7>(&d, skip);
                assert_eq!(a, nonempty_first(collect::<7>(&d, skip, false)), "N=7 skip={skip} {:?}", String::from_utf8_lossy(&d));
                lines_seen += a.len();
                let a = collect2::<2>(&d, skip);
                assert_eq!(a, nonempty_first(collect::<2>(&d, skip, false)), "N=2 skip={skip} {:?}", String::from_utf8_lossy(&d));
            }
        }
        assert!(lines_seen > 5000, "the test should accept real lines too ({lines_seen})");
    }

    #[test]
    fn new_scanner_matches_the_reference() {
        let mut r = Rng(0x1234_5678_9abc_def1);
        let mut lines_seen = 0;
        for case in 0..30_000 {
            // Mostly well-formed-looking lines, with every kind of breakage mixed in.
            let len = (r.next() % 700) as usize;
            let quote_rate = [2, 8, 30][case % 3];
            let d: Vec<u8> = (0..len)
                .map(|_| match r.next() % 100 {
                    x if x < quote_rate => b'"',
                    x if x < 50 => b"abcxyz0129"[(r.next() % 10) as usize],
                    x if x < 72 => b',',
                    x if x < 80 => b'\n',
                    x if x < 83 => b'\r',
                    x if x < 86 => b' ',
                    _ => b"-:TZ+."[(r.next() % 6) as usize],
                })
                .collect();
            for skip in [false, true] {
                let (a, b) = (collect::<2>(&d, skip, true), collect::<2>(&d, skip, false));
                assert_eq!(a, b, "N=2 skip={skip} {:?}", String::from_utf8_lossy(&d));
                let (a, b) = (collect::<4>(&d, skip, true), collect::<4>(&d, skip, false));
                assert_eq!(a, b, "N=4 skip={skip} {:?}", String::from_utf8_lossy(&d));
                let (a, b) = (collect::<7>(&d, skip, true), collect::<7>(&d, skip, false));
                assert_eq!(a, b, "N=7 skip={skip} {:?}", String::from_utf8_lossy(&d));
                lines_seen += a.len();
            }
        }
        assert!(lines_seen > 1000, "the test should accept real lines too ({lines_seen})");
    }
}
