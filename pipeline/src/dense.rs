//! The fast path, for data shaped like the generator's (generator/src/user.rs,
//! generator/src/text.rs).
//!
//! The generator writes an ID as a letter and lowercase base 36 of
//! `scramble(index, key)`, a bijection on 40 bits whose two multipliers are
//! fixed and whose key is drawn per seed:
//!   - user  `u`: index = person * 512 + alias, shared devices at 2^39 + device;
//!   - event `e`: index = the event's global number, and each person's events
//!     have consecutive numbers.
//! The keys are recovered from the data (two linked IDs are usually one
//! person's, so their indexes differ by a little; so do two events of one
//! person), and IDs become dense numbers again:
//!   - events are sharded by number, so every copy of an event lands in one
//!     shard, and dedup is an array indexed by number;
//!   - after dedup a shard's events come out in number order, so each person's
//!     events form one run: no hashing to group them.
//!
//! The output is exact whatever the data holds. Decoding is a bijection for any
//! key, so a wrong key, or IDs the generator did not make, only send rows down
//! the general path (the per-name shards of the main pipeline): slower, never
//! wrong. A person is finished in its shard only when all of their events and
//! clicks are known to be there; everyone else (whales, people at a shard edge,
//! shared devices, odd IDs) goes the general way.

use super::*;
use std::sync::atomic::AtomicUsize;

/// STOP=n exits after the n-th phase log (for counting each phase's instructions).
fn phase(t0: Instant, n: u32, what: &str) {
    log(t0, what);
    static STOP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    if *STOP.get_or_init(|| std::env::var("STOP").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) == n {
        std::process::exit(0);
    }
}

// ------------------------------------------------------------ the scramble

const MASK: u64 = (1 << 40) - 1;
const MUL_A: u64 = 0x9E37_79B9_7F4B;
const MUL_B: u64 = 0xD6E8_FEB8_6659;
/// Shared devices' indexes start here (generator/src/user.rs KIOSK_BASE).
const KIOSK: u64 = 1 << 39;

/// The inverse of odd `a` modulo 2^64 by Newton's method: a * a = 1 mod 8, and
/// each step doubles the number of correct low bits (3, 6, 12, 24, 48, 96).
const fn inverse(a: u64) -> u64 {
    let mut x = a;
    let mut k = 0;
    while k < 5 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        k += 1;
    }
    x
}
const INV_A: u64 = inverse(MUL_A) & MASK;
const INV_B: u64 = inverse(MUL_B) & MASK;

#[inline(always)]
fn scramble(v: u64, key: u64) -> u64 {
    let mut m = (v.wrapping_mul(MUL_A) ^ key) & MASK;
    m ^= m >> 17;
    m.wrapping_mul(MUL_B) & MASK
}

/// `A * v ^ key` from `y = scramble(v, key)`: undo the last multiply and the
/// xor-shift, which need no key.
#[inline(always)]
fn unshuffle(y: u64) -> u64 {
    let x = y.wrapping_mul(INV_B) & MASK;
    x ^ (x >> 17) ^ (x >> 34)
}

#[inline(always)]
fn unscramble(y: u64, key: u64) -> u64 {
    ((unshuffle(y) ^ key) & MASK).wrapping_mul(INV_A) & MASK
}

const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;

/// `d[s..e]` as canonical lowercase base 36, which is what the generator
/// writes (1 to 8 digits, no leading zero, below 2^40), or None. Canonical
/// form makes the map from strings to numbers one-to-one.
#[inline(always)]
fn b36(d: &[u8], s: usize, e: usize) -> Option<u64> {
    let n = e.wrapping_sub(s);
    if n == 0 || n > 8 {
        return None;
    }
    if e < 8 {
        return b36_slow(&d[s..e]);
    }
    // The 8 bytes ending at e, little-endian: the digits are the top n bytes,
    // the first digit (the most significant) lowest among them.
    let keep = !0u64 << (8 * (8 - n));
    let w = u64::from_le_bytes(d[e - 8..e].try_into().unwrap());
    let w = (w & keep) | (0x30 * ONES & !keep); // pad with '0' digits
    if w & HIGH != 0 {
        return None;
    }
    // Per byte b < 0x80: b + (0x80 - lo) has its top bit set iff b >= lo.
    let ge = |lo: u64| w.wrapping_add((0x80 - lo) * ONES) & HIGH;
    let letter = ge(0x61) & !ge(0x7b);
    if (ge(0x30) & !ge(0x3a)) | letter != HIGH {
        return None;
    }
    if n > 1 && (w >> (8 * (8 - n))) as u8 == b'0' {
        return None;
    }
    let v = w - 0x30 * ONES - (letter >> 7) * 0x27; // digit values 0..=35
    let p = (v & 0x00ff_00ff_00ff_00ff) * 36 + ((v >> 8) & 0x00ff_00ff_00ff_00ff);
    let q = (p & 0x0000_ffff_0000_ffff) * 1296 + ((p >> 16) & 0x0000_ffff_0000_ffff);
    let r = (q & 0xffff_ffff) * 1_679_616 + (q >> 32);
    (r <= MASK).then_some(r)
}

/// Two IDs at once: `d[a]` and `d[b]` as `pa` / `pb` followed by canonical base
/// 36, each decoded in one 64-bit lane of an SSE register (the same rules as
/// `b36`). Needs 8 bytes of `d` before each field's end.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn b36x2(d: &[u8], a: Fd, pa: u8, b: Fd, pb: u8) -> (Option<u64>, Option<u64>) {
    use std::arch::x86_64::*;
    let (na, nb) = (a.e().wrapping_sub(a.s()).wrapping_sub(1), b.e().wrapping_sub(b.s()).wrapping_sub(1));
    if na.wrapping_sub(1) >= 8 || nb.wrapping_sub(1) >= 8 || a.e() < 9 || b.e() < 9 {
        let one = |f: Fd, p: u8| (f.e() > f.s() && d[f.s()] == p).then(|| b36(d, f.s() + 1, f.e())).flatten();
        return (one(a, pa), one(b, pb));
    }
    // SAFETY: 9 <= e <= d.len() for both fields, so the 8 bytes before e are in d.
    let wa = unsafe { (d.as_ptr().add(a.e() - 8) as *const u64).read_unaligned() };
    let wb = unsafe { (d.as_ptr().add(b.e() - 8) as *const u64).read_unaligned() };
    let (ka, kb) = (!0u64 << (8 * (8 - na)), !0u64 << (8 * (8 - nb)));
    let wa = (wa & ka) | (0x30 * ONES & !ka);
    let wb = (wb & kb) | (0x30 * ONES & !kb);
    // SAFETY: plain SSE arithmetic on registers.
    let (ok, r) = unsafe {
        let w = _mm_set_epi64x(wb as i64, wa as i64);
        let dg = _mm_sub_epi8(w, _mm_set1_epi8(0x30));
        let lt = _mm_sub_epi8(w, _mm_set1_epi8(0x61));
        let is_dg = _mm_cmpeq_epi8(_mm_min_epu8(dg, _mm_set1_epi8(9)), dg);
        let is_lt = _mm_cmpeq_epi8(_mm_min_epu8(lt, _mm_set1_epi8(25)), lt);
        let ok = _mm_movemask_epi8(_mm_or_si128(is_dg, is_lt)) as u32;
        let val = _mm_blendv_epi8(dg, _mm_add_epi8(lt, _mm_set1_epi8(10)), is_lt);
        let p = _mm_maddubs_epi16(val, _mm_set1_epi16(0x0124)); // 36 * first + second
        let q = _mm_madd_epi16(p, _mm_set1_epi32(0x0001_0510)); // 1296 * first + second
        let r = _mm_add_epi64(_mm_mul_epu32(q, _mm_set1_epi64x(1_679_616)), _mm_srli_epi64(q, 32));
        (ok, [_mm_cvtsi128_si64(r) as u64, _mm_extract_epi64::<1>(r) as u64])
    };
    let lead0 = |w: u64, n: usize| n > 1 && (w >> (8 * (8 - n))) as u8 == b'0';
    let ra = (at(d, a.s()) == pa && ok & 0xff == 0xff && !lead0(wa, na) && r[0] <= MASK).then_some(r[0]);
    let rb = (at(d, b.s()) == pb && ok >> 8 == 0xff && !lead0(wb, nb) && r[1] <= MASK).then_some(r[1]);
    (ra, rb)
}

fn b36_slow(s: &[u8]) -> Option<u64> {
    if s.is_empty() || s.len() > 8 || (s.len() > 1 && s[0] == b'0') {
        return None;
    }
    let mut v = 0u64;
    for &c in s {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'z' => c - b'a' + 10,
            _ => return None,
        };
        v = v * 36 + d as u64;
    }
    (v <= MASK).then_some(v)
}

/// The packed key (`Key for u64`) of the ID `prefix` + base 36 of `y`.
fn packed_id(prefix: u8, mut y: u64) -> u64 {
    const DIG: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut digits = [0u8; 8];
    let mut n = 0;
    loop {
        digits[n] = DIG[(y % 36) as usize];
        n += 1;
        y /= 36;
        if y == 0 {
            break;
        }
    }
    let mut k = prefix as u64;
    for i in (0..n).rev() {
        k = k << 7 | digits[i] as u64;
    }
    k << (7 * (8 - n))
}

// ------------------------------------------------------------ key recovery

/// Key bits implied by x1 = A*v1 ^ key and x2 = A*v2 ^ key when v1 - v2 = d
/// (mod 2^40): (mask of the bits it fixes, their values), or None if no key
/// fits. With a = x1 ^ key and b = x2 ^ key, a - b must be T = A*d. The borrow
/// into bit k of a - b is then T_k ^ e_k (e = x1 ^ x2, which needs no key).
/// Where x1 and x2 agree the borrow passes through unchanged (a check); where
/// they differ, the borrow out of bit k is set iff a_k = 0, i.e(). iff key_k = x1_k.
#[inline(always)]
fn pair_bits(x1: u64, x2: u64, d: u64) -> Option<(u64, u64)> {
    const LOW39: u64 = (1 << 39) - 1;
    let t = d.wrapping_mul(MUL_A) & MASK;
    let e = x1 ^ x2;
    let c = e ^ t; // borrow into each bit
    if c & 1 != 0 || ((c >> 1) ^ c) & !e & LOW39 != 0 {
        return None;
    }
    let m = e & LOW39;
    Some((m, (x1 ^ !(c >> 1)) & m))
}

/// The key under which pairs (y1, y2) of scrambled values decode to numbers
/// at most `dmax` apart, best by `score`, or None.
///
/// One pair cannot tell which of its two numbers is larger: d and -d always
/// both fit, with complementary bits. So bits are grown from a seed pair, under
/// each of its two readings: a pair joins when exactly one of its readings
/// agrees with the bits known so far on at least 6 of them. The bits no pair
/// fixed (bit 39 at least) are tried both ways.
fn solve_key(pairs: &[(u64, u64)], dmax: u64, score: impl Fn(u64) -> f64) -> Option<(u64, f64)> {
    let cands: Vec<Vec<(u64, u64)>> = pairs
        .iter()
        .filter_map(|&(y1, y2)| {
            let (x1, x2) = (unshuffle(y1), unshuffle(y2));
            if x1 == x2 {
                return None;
            }
            let v: Vec<(u64, u64)> =
                (1..=dmax).flat_map(|d| [d, d.wrapping_neg() & MASK]).filter_map(|d| pair_bits(x1, x2, d)).collect();
            (!v.is_empty()).then_some(v)
        })
        .collect();
    let mut best: Option<(u64, f64)> = None;
    for seed in cands.iter().filter(|c| c.len() == 2).take(4) {
        for &(m0, b0) in seed {
            let (mut km, mut kb) = (m0, b0);
            loop {
                let mut changed = false;
                for c in &cands {
                    let mut pick: Option<(u64, u64)> = None;
                    let mut conflict = false;
                    for &(m, b) in c {
                        let ov = m & km;
                        if ov.count_ones() < 6 || (b ^ kb) & ov != 0 {
                            continue;
                        }
                        match pick {
                            None => pick = Some((m, b)),
                            Some(x) => conflict |= x != (m, b),
                        }
                    }
                    if let (Some((m, b)), false) = (pick, conflict) {
                        if m & !km != 0 {
                            km |= m;
                            kb |= b;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            let free: Vec<u32> = (0..40).filter(|&k| km >> k & 1 == 0).collect();
            if free.len() > 10 {
                continue;
            }
            for combo in 0u64..1 << free.len() {
                let mut k = kb;
                for (j, &bit) in free.iter().enumerate() {
                    if combo >> j & 1 == 1 {
                        k |= 1 << bit;
                    }
                }
                let s = score(k);
                if s >= 0.99 {
                    return Some((k, s)); // a wrong key scores near zero
                }
                if best.is_none_or(|(_, b)| s > b) {
                    best = Some((k, s));
                }
            }
        }
    }
    best
}

// ------------------------------------------------------------------ records

/// One event copy in its number's shard. `ts` is ms after window_start, or
/// NO_TS for a copy that fails validation (it still takes part in dedup).
#[derive(Clone, Copy)]
#[repr(C)]
struct ERec {
    i: u32,      // event number
    ingest: u32, // seconds after ING_BASE; u32::MAX if unparseable (sorts last)
    ts: u32,
    user: u32, // dense user number, O_FLAG | other-ID ref, or NO_USER
    info: u32, // price | category << 24; price 0 for view and cart
}

/// An event copy whose ID is not a dense number: deduplicated by its key.
#[derive(Clone, Copy)]
struct OddEv {
    key: u64,
    ingest: u32,
    ts: u32,
    user: u32,
    info: u32,
}

/// A surviving event in a shard, in event-number order.
#[derive(Clone, Copy)]
struct SRec {
    ts: u32,
    user: u32,
    info: u32,
}

/// A valid click of a user whose person is finished in one shard.
#[derive(Clone, Copy)]
struct CLoc {
    id: u64,  // packed click_id, for the tie-break
    ts: u32,  // ms after window_start - 7 days
    user: u32,
    camp: u32,
}

const NO_TS: u32 = u32::MAX;
const NO_USER: u32 = u32::MAX;
/// A user ID that is not a dense number: the low 31 bits say which thread's
/// list holds its key (top 5) and where (low 26).
const O_FLAG: u32 = 1 << 31;
const O_SHIFT: u32 = 26;
/// Events per shard: dedup's array stays in L2.
const SH: u32 = 16;
/// `home` values besides a shard number.
const HOME_NONE: u32 = u32::MAX;
const HOME_MULTI: u32 = u32::MAX - 1;
/// `pinfo` values besides (root << 32 | alias mask): no linked alias, or not simple.
const P_NONE: u64 = 0;
const P_COMPLEX: u64 = u64::MAX;

/// The N fields of the line starting at `ls`, split in one go: 32-byte loads
/// give its newline, comma and quote masks (one 64-byte window as u64 when the
/// line fits in it, else a 128-byte one as u128); the first newline ends the
/// line; it is well formed iff N - 1 commas come before it and its quotes all
/// wrap whole fields. Returns (fields, or None for a malformed line; where the
/// next line starts), or None (use the slow path) for a line of 128 bytes or
/// more, or one too near the end of `d`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn split_line<const N: usize>(d: &[u8], ls: usize) -> Option<(Option<[Fd; N]>, usize)> {
    use std::arch::x86_64::*;
    if ls + 128 > d.len() {
        return None;
    }
    // SAFETY: ls + 128 <= d.len(), so every load below stays inside d.
    let load = |i: usize| unsafe { _mm256_loadu_si256(d.as_ptr().add(ls + 32 * i) as *const __m256i) };
    let mask = |v: __m256i, c: u8| unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(c as i8))) as u32 as u64 };
    let (v0, v1) = (load(0), load(1));
    let nl = mask(v0, b'\n') | mask(v1, b'\n') << 32;
    let mut fl = [Fd::default(); N];
    if nl != 0 {
        // The whole line is in the first 64 bytes.
        let e = nl.trailing_zeros();
        let next = ls + e as usize + 1;
        let below = (1u64 << e) - 1;
        let mut m = (mask(v0, b',') | mask(v1, b',') << 32) & below;
        if m.count_ones() as usize != N - 1 {
            return Some((None, next));
        }
        let mut s = ls;
        for x in fl.iter_mut().take(N - 1) {
            let p = ls + m.trailing_zeros() as usize;
            m &= m - 1;
            *x = Fd::new(s, p);
            s = p + 1;
        }
        fl[N - 1] = Fd::new(s, ls + e as usize);
        let quotes = ((mask(v0, b'"') | mask(v1, b'"') << 32) & below).count_ones() as u64;
        if quotes != 0 && !unwrap_quotes(d, &mut fl, quotes) {
            return Some((None, next));
        }
        return Some((Some(fl), next));
    }
    let (v2, v3) = (load(2), load(3));
    let wide = |c: u8| -> u128 {
        (mask(v0, c) | mask(v1, c) << 32) as u128 | ((mask(v2, c) | mask(v3, c) << 32) as u128) << 64
    };
    let nl = wide(b'\n');
    if nl == 0 {
        return None;
    }
    let e = nl.trailing_zeros();
    let next = ls + e as usize + 1;
    let below = (1u128 << e) - 1;
    let mut m = wide(b',') & below;
    if m.count_ones() as usize != N - 1 {
        return Some((None, next));
    }
    let mut s = ls;
    for x in fl.iter_mut().take(N - 1) {
        let p = ls + m.trailing_zeros() as usize;
        m &= m - 1;
        *x = Fd::new(s, p);
        s = p + 1;
    }
    fl[N - 1] = Fd::new(s, ls + e as usize);
    let quotes = (wide(b'"') & below).count_ones() as u64;
    if quotes != 0 && !unwrap_quotes(d, &mut fl, quotes) {
        return Some((None, next));
    }
    Some((Some(fl), next))
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
fn split_line<const N: usize>(_d: &[u8], _ls: usize) -> Option<(Option<[Fd; N]>, usize)> {
    None
}

/// Every well-formed line of a chunk, as N fields, to `row` (one call site, so it is inlined).
#[inline(always)]
pub(crate) fn each_line<const N: usize>(d: &[u8], skip_first: bool, mut row: impl FnMut(&[Fd; N])) {
    let mut ls = if skip_first { memchr(b'\n', d).map_or(d.len(), |i| i + 1) } else { 0 };
    while ls < d.len() {
        let (fl, next) = match split_line::<N>(d, ls) {
            Some(x) => x,
            None => {
                let e = memchr(b'\n', &d[ls..]).map_or(d.len(), |i| ls + i);
                let mut fl = [Fd::default(); N];
                (slow_line(d, ls, e, &mut fl).then_some(fl), e + 1)
            }
        };
        ls = next;
        if let Some(fl) = fl {
            row(&fl);
        }
    }
}

enum Uid {
    Norm(u32),
    Other(u64),
}

/// A user ID: its dense number if it decodes to an ordinary person's alias
/// below `pmax`, else its packed key; None if it does not fit a packed key.
#[inline(always)]
fn uid(d: &[u8], f: Fd, ukey: u64, pmax: u64) -> Option<Uid> {
    if f.e() > f.s() && d[f.s()] == b'u' {
        if let Some(y) = b36(d, f.s() + 1, f.e()) {
            let v = unscramble(y, ukey);
            if v < KIOSK && (v >> 9) < pmax {
                return Some(Uid::Norm(v as u32));
            }
        }
    }
    <u64 as Key<'_>>::make(d, f.s(), f.e()).map(Uid::Other)
}

/// Run `work` on chunks 0..n from `threads` scoped threads, each with its own
/// state from `init(thread)`; returns the states in thread order.
pub(crate) fn per_thread<T: Send>(n: usize, threads: usize, init: impl Fn(usize) -> T + Sync, work: impl Fn(&mut T, usize) + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (next, init, work) = (&next, &init, &work);
                s.spawn(move || {
                    let mut st = init(t);
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        if i >= n {
                            break;
                        }
                        work(&mut st, i);
                    }
                    st
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// The first whole lines of `c`, about `bytes` of them.
fn head<'a>(c: &Chunk<'a>, bytes: usize) -> &'a [u8] {
    let d = c.data;
    if d.len() <= bytes {
        return d;
    }
    match d[..bytes].iter().rposition(|&b| b == b'\n') {
        Some(i) => &d[..i + 1],
        None => &d[..0],
    }
}

// --------------------------------------------------------------------- run

/// The whole pipeline on the fast path. Returns false, having written nothing,
/// when it cannot be used (then the main pipeline runs). `force` uses it even
/// when the data does not look generated (for tests).
pub(crate) fn run<'a>(t0: Instant, out: &Path, inp: &Inputs<'a>, threads: usize, force: bool) -> bool {
    let (ws, we) = (inp.ws, inp.we);
    if we <= ws || we - ws >= u32::MAX as i64 - WEEK || threads > 32 {
        return false;
    }
    let Some(cat) = Catalog::<u64>::load(inp) else { return false };
    if cat.ncat > 255 {
        return false;
    }
    phase(t0, 1, "catalog");
    let ncamp = cat.ncamp;
    let organic = ncamp;
    let target = chunk_target(inp, threads);
    let ev_chunks = make_chunks(&inp.events, target);
    let ln_chunks = make_chunks(&inp.links, target);

    // ---- The keys (from samples) and the link components do not depend on
    // each other: they run side by side.
    let keys = || -> Option<(u64, u64, usize, u64, u64, usize)> {
        // ---- Recover the keys from samples, and size the dense arrays.
        let ev_bytes: usize = inp.events.iter().map(|b| b.len()).sum();
        let step = (ev_chunks.len() / 96).max(1);
        let samples: Vec<Vec<(Option<u64>, Option<u64>)>> = ev_chunks
            .par_iter()
            .step_by(step)
            .map(|c| {
                let d = head(c, 48 << 10);
                let mut v = Vec::new();
                scan::<7, _>(d, c.skip_first, |fl| {
                    let e = (fl[0].e() > fl[0].s() && d[fl[0].s()] == b'e').then(|| b36(d, fl[0].s() + 1, fl[0].e())).flatten();
                    let u = (fl[1].e() > fl[1].s() && d[fl[1].s()] == b'u').then(|| b36(d, fl[1].s() + 1, fl[1].e())).flatten();
                    v.push((e, u));
                });
                v
            })
            .collect();
        let sampled_bytes: usize = ev_chunks.iter().step_by(step).map(|c| head(c, 48 << 10).len()).sum();
        let ev_sample: Vec<(Option<u64>, Option<u64>)> = samples.into_iter().flatten().collect();
        log(t0, &format!("sampled {} event rows", ev_sample.len()));
        let rows_est = if ev_sample.is_empty() { ev_bytes / 40 } else { ev_bytes / (sampled_bytes / ev_sample.len()).max(1) };
        let pmax = ((rows_est / 24 + 4096) as u64).min(1 << 22);
        let imax = (rows_est + rows_est / 8 + 4096) as u64;
        if imax >= 1 << 31 {
            return None;
        }
        let nsh = ((imax >> SH) + 1) as usize;

        let mut link_pairs: Vec<(u64, u64)> = Vec::new();
        for c in &ln_chunks {
            let d = head(c, 256 << 10);
            scan::<2, _>(d, c.skip_first, |fl| {
                let dec = |f: Fd| (f.e() > f.s() && d[f.s()] == b'u').then(|| b36(d, f.s() + 1, f.e())).flatten();
                if let (Some(a), Some(b)) = (dec(fl[0]), dec(fl[1])) {
                    if a != b {
                        link_pairs.push((a, b));
                    }
                }
            });
            if link_pairs.len() >= 4096 {
                break;
            }
        }
        // Scored on at most 2048 IDs: every bit no pair fixes doubles the candidates.
        let users: Vec<u64> =
            link_pairs.iter().flat_map(|&(a, b)| [a, b]).chain(ev_sample.iter().filter_map(|s| s.1)).step_by(7).take(2048).collect();
        let user_score = |k: u64| {
            let ok = users
                .iter()
                .filter(|&&y| {
                    let v = unscramble(y, k);
                    (v < KIOSK && (v >> 9) < pmax) || (v >= KIOSK && v - KIOSK < 4096)
                })
                .count();
            ok as f64 / users.len().max(1) as f64
        };
        let ukey = solve_key(&link_pairs, 8, user_score).filter(|&(_, s)| s >= 0.9).map(|(k, _)| k);
        log(t0, "user key");
        let ekey = ukey.and_then(|uk| {
            // Two events of one ordinary person have numbers less than their event
            // count apart. People seen more than 4 times in the sample are skipped:
            // the whales, whose numbers are far apart.
            let mut by_person: Vec<(u64, u64)> = ev_sample
                .iter()
                .filter_map(|&(e, u)| Some((unscramble(u?, uk) >> 9, e?)))
                .collect();
            by_person.sort_unstable();
            let mut pairs = Vec::new();
            let mut k = 0;
            while k < by_person.len() {
                let mut r = k + 1;
                while r < by_person.len() && by_person[r].0 == by_person[k].0 {
                    r += 1;
                }
                if r - k <= 4 {
                    for j in k + 1..r {
                        pairs.push((by_person[j - 1].1, by_person[j].1));
                    }
                }
                k = r;
            }
            let es: Vec<u64> = ev_sample.iter().filter_map(|s| s.0).step_by(7).take(2048).collect();
            let score = |k: u64| es.iter().filter(|&&y| unscramble(y, k) < imax).count() as f64 / es.len().max(1) as f64;
            solve_key(&pairs, 64, score).filter(|&(_, s)| s >= 0.9).map(|(k, _)| k)
        });
        phase(t0, 2, &format!("keys: user {ukey:?}, event {ekey:?}; ~{rows_est} event rows, {nsh} shards"));
        if (ukey.is_none() || ekey.is_none()) && !force {
            return None;
        }
        Some((ukey.unwrap_or(0), ekey.unwrap_or(0), rows_est, pmax, imax, nsh))
    };
    let (keys, comps) = rayon::join(keys, || link_components::<u64>(inp, target, threads));
    let Some((ukey, ekey, rows_est, pmax, imax, nsh)) = keys else { return false };
    let _ = rows_est;

    // ---- Links: who is linked to whom, then what that means per dense person.
    let Some(comps) = comps else { return false };
    let nid = comps.roots.len();
    let names: Vec<u64> = comps.name.iter().map(|n| n.unwrap_or(0)).collect();
    // Every linked ID: (dense user number, root) or (packed key, root).
    let classified: Vec<(Vec<(u32, u32)>, Vec<(u64, u32)>)> = comps
        .keys
        .par_iter()
        .enumerate()
        .map(|(s, keys)| {
            let (mut norm, mut other) = (Vec::new(), Vec::new());
            let mut buf = Vec::with_capacity(16);
            for (j, &k) in keys.iter().enumerate() {
                let root = comps.roots[comps.base[s] as usize + j];
                buf.clear();
                k.put(&mut buf);
                let f = Fd::new(0, buf.len());
                match uid(&buf, f, ukey, pmax) {
                    Some(Uid::Norm(v)) => norm.push((v, root)),
                    _ => other.push((k, root)),
                }
            }
            (norm, other)
        })
        .collect();
    // A person (root) is complex when its IDs span several dense people or
    // include any other ID (a shared device, an upper-cased ID).
    let cx: Vec<AtomicBool> = (0..nid).into_par_iter().map(|_| AtomicBool::new(false)).collect();
    let root_p: Vec<AtomicU32> = (0..nid).into_par_iter().map(|_| AtomicU32::new(u32::MAX)).collect();
    classified.par_iter().for_each(|(norm, other)| {
        for &(_, r) in other {
            cx[r as usize].store(true, Relaxed);
        }
        for &(v, r) in norm {
            let p = v >> 9;
            if let Err(cur) = root_p[r as usize].compare_exchange(u32::MAX, p, Relaxed, Relaxed) {
                if cur != p {
                    cx[r as usize].store(true, Relaxed);
                }
            }
        }
    });
    drop(root_p);
    let omap: FxHashMap<u64, u32> = classified.iter().flat_map(|(_, o)| o.iter().copied()).collect();
    // Every linked dense ID with its root, sorted by ID (so by person): packed
    // into u64 keys, flattened and sorted on all threads.
    let mut keys: Vec<u64> =
        classified.par_iter().flat_map_iter(|(n, _)| n.iter().map(|&(v, r)| (v as u64) << 32 | r as u64)).collect();
    drop(classified);
    keys.par_sort_unstable();
    let nodes: Vec<(u32, u32)> = keys.par_iter().map(|&k| ((k >> 32) as u32, k as u32)).collect();
    drop(keys);
    // pinfo[p]: P_NONE (no linked alias), root << 32 | mask of linked aliases
    // (all < 32, all in one root that is not complex), or P_COMPLEX (whose
    // linked aliases are then in cmap).
    let mut pinfo = vec![P_NONE; pmax as usize];
    let pspan = (pmax as usize).div_ceil(threads * 8).max(1);
    let cmaps: Vec<Vec<(u32, u32)>> = pinfo
        .par_chunks_mut(pspan)
        .enumerate()
        .map(|(t, chunk)| {
            let lo = (t * pspan) as u32;
            let mut cm = Vec::new();
            let start = nodes.partition_point(|n| n.0 >> 9 < lo);
            let end = nodes.partition_point(|n| ((n.0 >> 9) as usize) < t * pspan + chunk.len());
            let nodes = &nodes[start..end];
            let mut k = 0;
            while k < nodes.len() {
                let p = nodes[k].0 >> 9;
                let mut r = k + 1;
                while r < nodes.len() && nodes[r].0 >> 9 == p {
                    r += 1;
                }
                let root = nodes[k].1;
                let simple =
                    nodes[k..r].iter().all(|&(v, rt)| rt == root && (v & 511) < 32) && !cx[root as usize].load(Relaxed);
                chunk[(p - lo) as usize] = if simple {
                    (root as u64) << 32 | nodes[k..r].iter().fold(0u64, |m, &(v, _)| m | 1 << (v & 511))
                } else {
                    cm.extend_from_slice(&nodes[k..r]);
                    P_COMPLEX
                };
                k = r;
            }
            cm
        })
        .collect();
    let cmap: FxHashMap<u32, u32> = cmaps.into_iter().flatten().collect();
    drop(nodes);
    phase(t0, 3, &format!("links: {nid} ids, {} other, {} in complex people", omap.len(), cmap.len()));

    // ---- Events: every copy, into its number's shard.
    const ING_BASE_OFF: i64 = 1 << 31;
    let ing_base = ws.div_euclid(1000) - ING_BASE_OFF;
    let bad = AtomicBool::new(false);
    // A full shard holds 2^SH events plus their extra copies (a few percent);
    // each thread gets about its share of them. Unused room is never touched.
    let cap = {
        let m = (1usize << SH) * 9 / 8 / threads;
        m + 4 * (m as f64).sqrt() as usize + 64
    };
    struct EvOut {
        shards: Vec<Vec<ERec>>,
        odd: Vec<OddEv>,
        ousers: Vec<u64>,
    }
    let pop = std::env::var("POP").map(|v| v != "0").unwrap_or(true);
    let ev_out: Vec<EvOut> = per_thread(
        ev_chunks.len(),
        threads,
        |_| EvOut {
            shards: (0..nsh).map(|_| Vec::with_capacity(cap)).collect(),
            odd: Vec::new(),
            ousers: Vec::new(),
        },
        |st, ci| {
            let c = ev_chunks[ci];
            let d = c.data;
            if pop {
                populate(d);
            }
            let mut row = |fl: &[Fd; 7]| {
                if fl[0].empty() {
                    return;
                }
                // Both IDs decoded together.
                let (ye, yu) = b36x2(d, fl[0], b'e', fl[1], b'u');
                // ingest_ts is F1 (whole seconds): straight to seconds; anything
                // else through the exact parser.
                let (ing_ms, ingest) = match f1_secs(d, fl[3]) {
                    Some(secs) if (0..u32::MAX as i64).contains(&(secs - ing_base)) => {
                        (Some(secs * 1000), (secs - ing_base) as u32)
                    }
                    _ => match ts_at(d, fl[3]) {
                        None => (None, u32::MAX),
                        Some(ms) => {
                            let s = ms.div_euclid(1000) - ing_base;
                            if ms.rem_euclid(1000) != 0 || !(0..u32::MAX as i64).contains(&s) {
                                bad.store(true, Relaxed);
                                return;
                            }
                            (Some(ms), s as u32)
                        }
                    },
                };
                let (mut ts, mut user, mut info) = (NO_TS, NO_USER, 0u32);
                'v: {
                    if fl[1].empty() || fl[5].empty() {
                        break 'v;
                    }
                    let Some(t) = ts_fast(d, fl[2]) else { break 'v };
                    if t < ws || t >= we {
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
                    if ing_ms.unwrap_or(i64::MAX).saturating_sub(t) > DAY {
                        break 'v; // late
                    }
                    let norm = yu.map(|y| unscramble(y, ukey)).filter(|&v| v < KIOSK && (v >> 9) < pmax);
                    let u = match norm {
                        Some(v) => Some(Uid::Norm(v as u32)),
                        None => <u64 as Key<'_>>::make(d, fl[1].s(), fl[1].e()).map(Uid::Other),
                    };
                    user = match u {
                        Some(Uid::Norm(v)) => v,
                        Some(Uid::Other(k)) => {
                            if st.ousers.len() >= (1 << O_SHIFT) - 1 {
                                bad.store(true, Relaxed);
                                return;
                            }
                            st.ousers.push(k);
                            O_FLAG | (st.ousers.len() - 1) as u32 // thread bits added below
                        }
                        None => {
                            bad.store(true, Relaxed);
                            return;
                        }
                    };
                    ts = (t - ws) as u32;
                    // Categories only matter for purchases (top_category sums purchase prices).
                    info = if price > 0 { price | cat.category_of(d, fl[5]) << 24 } else { 0 };
                }
                let num = ye.map(|y| unscramble(y, ekey));
                match num {
                    Some(i) if i < imax => st.shards[(i >> SH) as usize].push(ERec { i: i as u32, ingest, ts, user, info }),
                    _ => match <u64 as Key<'_>>::make(d, fl[0].s(), fl[0].e()) {
                        Some(key) => st.odd.push(OddEv { key, ingest, ts, user, info }),
                        None => bad.store(true, Relaxed),
                    },
                }
            };
            each_line::<7>(d, c.skip_first, &mut row);
        },
    );
    if bad.load(Relaxed) {
        log(t0, "fast path: an ID or ingest_ts it cannot hold; using the main pipeline");
        return false;
    }
    // Other-ID refs gain their thread's number now that the threads are known.
    let ev_out: Vec<EvOut> = ev_out
        .into_iter()
        .enumerate()
        .map(|(t, mut o)| {
            let tag = |u: &mut u32| {
                if *u != NO_USER && *u & O_FLAG != 0 {
                    *u |= (t as u32) << O_SHIFT;
                }
            };
            for v in o.shards.iter_mut() {
                for r in v.iter_mut() {
                    tag(&mut r.user);
                }
            }
            for r in o.odd.iter_mut() {
                tag(&mut r.user);
            }
            o
        })
        .collect();
    let okey = |u: u32| -> u64 {
        let t = ((u & !O_FLAG) >> O_SHIFT) as usize;
        ev_out[t].ousers[(u & ((1 << O_SHIFT) - 1)) as usize]
    };
    phase(t0, 4, "events parsed");

    // ---- Which people can be finished in one shard. home[p] is the shard of
    // p's surviving events, HOME_MULTI if they are in several (or not in one
    // run), HOME_NONE if p has none.
    let home: Vec<AtomicU32> = (0..pmax as usize).into_par_iter().map(|_| AtomicU32::new(HOME_NONE)).collect();
    let multi = |p: u32| home[p as usize].store(HOME_MULTI, Relaxed);
    // An other ID linked to a person makes that person complex.
    let other_name = |key: u64| -> u64 {
        match omap.get(&key) {
            Some(&r) => {
                cx[r as usize].store(true, Relaxed);
                names[r as usize]
            }
            None => key,
        }
    };

    // Odd event IDs: dedup by key. Their survivors' people are complex.
    let mut odd_best: FxHashMap<u64, OddEv> = FxHashMap::default();
    for o in &ev_out {
        for r in &o.odd {
            match odd_best.entry(r.key) {
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
    let odd_surv: Vec<SRec> = odd_best
        .values()
        .filter(|r| r.ts != NO_TS)
        .map(|r| SRec { ts: r.ts, user: r.user, info: r.info })
        .collect();
    drop(odd_best);
    for r in &odd_surv {
        if r.user & O_FLAG == 0 {
            multi(r.user >> 9);
        }
    }

    // Pass 1, per shard: dedup by number, then mark where each person's runs are.
    let survivors: Vec<Vec<SRec>> = (0..nsh)
        .into_par_iter()
        .map_init(
            || vec![[u32::MAX, 0, 0, 0]; 1 << SH],
            |slots, q| {
                // Each event number's slot keeps its earliest copy so far:
                // (ingest, ts, user, info). An ingest_ts that does not parse is
                // u32::MAX, like an empty slot: such a copy never wins, and as the
                // kept copy it would be invalid anyway (late), so nothing changes.
                for s in slots.iter_mut() {
                    s[0] = u32::MAX;
                }
                let base = (q as u32) << SH;
                for o in &ev_out {
                    for r in &o.shards[q] {
                        let s = &mut slots[(r.i - base) as usize];
                        if r.ingest < s[0] {
                            *s = [r.ingest, r.ts, r.user, r.info]; // equal ingest_ts: identical copies, keep either
                        }
                    }
                }
                let mut surv = Vec::with_capacity(1 << SH);
                for s in slots.iter() {
                    if s[0] != u32::MAX && s[1] != NO_TS {
                        surv.push(SRec { ts: s[1], user: s[2], info: s[3] });
                    }
                }
                let mut last = u32::MAX;
                for r in &surv {
                    if r.user & O_FLAG != 0 {
                        other_name(okey(r.user));
                        last = u32::MAX;
                        continue;
                    }
                    let p = r.user >> 9;
                    if p != last {
                        last = p;
                        if let Err(cur) = home[p as usize].compare_exchange(HOME_NONE, q as u32, Relaxed, Relaxed) {
                            if cur != HOME_MULTI {
                                multi(p); // in another shard, or an earlier run of this one
                            }
                        }
                    }
                }
                surv
            },
        )
        .collect();
    for r in &odd_surv {
        if r.user & O_FLAG != 0 {
            other_name(okey(r.user));
        }
    }
    let n_surv: usize = survivors.iter().map(|s| s.len()).sum();
    phase(t0, 5, &format!("events deduped: {n_surv} + {} odd survive", odd_surv.len()));

    // The person key of a dense user (after links): root | 1 << 63, or the user itself.
    let pk_of = |v: u32| -> u64 {
        let info = pinfo[(v >> 9) as usize];
        if info == P_NONE {
            v as u64
        } else if info == P_COMPLEX {
            match cmap.get(&v) {
                Some(&r) => r as u64 | 1 << 63,
                None => v as u64,
            }
        } else if (v & 511) < 32 && info >> (v & 511) & 1 == 1 {
            info >> 32 | 1 << 63
        } else {
            v as u64
        }
    };
    let pk_name = |pk: u64| -> u64 {
        if pk >> 63 == 1 { names[(pk & 0xffff_ffff) as usize] } else { packed_id(b'u', scramble(pk, ukey)) }
    };
    let pk_cx = |pk: u64| pk >> 63 == 1 && cx[(pk & 0xffff_ffff) as usize].load(Relaxed);

    // ---- Clicks: valid ones go to their person's shard, or the general path.
    let cl_chunks = make_chunks(&inp.clicks, target);
    let ccap = inp.clicks.iter().map(|b| b.len()).sum::<usize>() / 46 / nsh / threads + 64;
    struct ClOut {
        shards: Vec<Vec<CLoc>>,
        cx: Vec<Vec<CRec<u64>>>,
    }
    let cl_base = ws - WEEK;
    let cl_out: Vec<ClOut> = per_thread(
        cl_chunks.len(),
        threads,
        |_| ClOut { shards: (0..nsh).map(|_| Vec::with_capacity(ccap)).collect(), cx: (0..NB).map(|_| Vec::new()).collect() },
        |st, ci| {
            let c = cl_chunks[ci];
            let d = c.data;
            // A click of a dense user waits AHEAD rows in this ring while its
            // person's entries are prefetched; then `route` sends it on.
            let mut ring = [CLoc { id: 0, ts: 0, user: 0, camp: 0 }; AHEAD];
            let (mut head, mut queued) = (0usize, 0usize);
            let route = |st: &mut ClOut, c: &CLoc| {
                let p = (c.user >> 9) as usize;
                let h = home[p].load(Relaxed);
                let pk = pk_of(c.user);
                if pinfo[p] != P_COMPLEX && h != HOME_MULTI && !pk_cx(pk) {
                    if h != HOME_NONE {
                        st.shards[h as usize].push(*c);
                    }
                    return; // HOME_NONE: this person has no session
                }
                let name = pk_name(pk);
                st.cx[shard_of(&name)].push(CRec { name, id: c.id, ts: cl_base + c.ts as i64, camp: c.camp });
            };
            each_line::<4>(d, c.skip_first, |fl| {
                if fl[0].empty() || fl[1].empty() || fl[2].empty() {
                    return;
                }
                let Some(camp) = cat.campaign_of(d, fl[2]) else {
                    return;
                };
                let Some(ts) = ts_fast(d, fl[3]) else { return };
                if ts < cl_base || ts >= we {
                    return;
                }
                let (Some(id), Some(u)) = (<u64 as Key<'_>>::make(d, fl[0].s(), fl[0].e()), uid(d, fl[1], ukey, pmax)) else {
                    bad.store(true, Relaxed);
                    return;
                };
                match u {
                    Uid::Norm(v) => {
                        let p = (v >> 9) as usize;
                        prefetch(&home[p]);
                        prefetch(&pinfo[p]);
                        let e = CLoc { id, ts: (ts - cl_base) as u32, user: v, camp };
                        if queued == AHEAD {
                            route(st, &ring[head]);
                            ring[head] = e;
                            head = (head + 1) % AHEAD;
                        } else {
                            ring[(head + queued) % AHEAD] = e;
                            queued += 1;
                        }
                    }
                    Uid::Other(k) => {
                        let name = other_name(k);
                        st.cx[shard_of(&name)].push(CRec { name, id, ts, camp });
                    }
                }
            });
            for k in 0..queued {
                route(st, &ring[(head + k) % AHEAD]);
            }
        },
    );
    if bad.load(Relaxed) {
        log(t0, "fast path: a click ID it cannot hold; using the main pipeline");
        return false;
    }
    phase(t0, 6, "clicks parsed");

    // ---- Pass 2, per shard: finish every person who is entirely here; send
    // the rest to the general path.
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1) as usize;
    let size = (ncamp + 1) * ndays;
    struct Acc {
        urows: Vec<URow<u64>>,
        camp: Vec<[i64; 4]>,
        ps: PScratch,
        cx_ev: Vec<Vec<PRec<u64>>>,
        cx_cl: Vec<Vec<CRec<u64>>>,
        cls: Vec<CLoc>,
        coff: Vec<u32>,
        cur: Vec<u32>,
        tev: Vec<(u64, Ev)>,
        tcl: Vec<(u64, Cl<u64>)>,
        ev: Vec<Ev>,
        cl: Vec<Cl<u64>>,
    }
    let new_acc = || Acc {
        urows: Vec::new(),
        camp: vec![[0i64; 4]; size],
        ps: PScratch::default(),
        cx_ev: (0..NB).map(|_| Vec::new()).collect(),
        cx_cl: (0..NB).map(|_| Vec::new()).collect(),
        cls: Vec::new(),
        coff: Vec::new(),
        cur: Vec::new(),
        tev: Vec::new(),
        tcl: Vec::new(),
        ev: Vec::new(),
        cl: Vec::new(),
    };
    let ev_of = |r: &SRec| Ev { ts: ws + r.ts as i64, price: r.info & 0xff_ffff, cat: r.info >> 24 };
    let cx_ev_push = |acc: &mut Acc, name: u64, r: &SRec| {
        let e = ev_of(r);
        acc.cx_ev[shard_of(&name)].push(PRec { name, ts: e.ts, price: e.price, cat: e.cat });
    };
    let mut accs: Vec<Acc> = per_thread(nsh, threads, |_| new_acc(), |acc, q| {
            acc.ps.reset(cat.ncat);
            // This shard's clicks, grouped by person: a counting sort over the
            // shard's range of people (a plain sort if that range is sparse).
            let n_cl: usize = cl_out.iter().map(|o| o.shards[q].len()).sum();
            let (mut pmin, mut pmax) = (u32::MAX, 0u32);
            for o in &cl_out {
                for c in &o.shards[q] {
                    pmin = pmin.min(c.user >> 9);
                    pmax = pmax.max(c.user >> 9);
                }
            }
            acc.cls.clear();
            acc.coff.clear();
            let counted = n_cl > 0 && ((pmax - pmin) as usize) < 4 * n_cl + 4096;
            if counted {
                let span = (pmax - pmin) as usize + 1;
                acc.coff.resize(span + 1, 0);
                for o in &cl_out {
                    for c in &o.shards[q] {
                        acc.coff[(c.user >> 9) as usize - pmin as usize + 1] += 1;
                    }
                }
                for i in 0..span {
                    acc.coff[i + 1] += acc.coff[i];
                }
                acc.cls.resize(n_cl, CLoc { id: 0, ts: 0, user: 0, camp: 0 });
                let Acc { cls, coff, cur, .. } = &mut *acc;
                cur.clear();
                cur.extend_from_slice(&coff[..span]);
                for o in &cl_out {
                    for c in &o.shards[q] {
                        let b = (c.user >> 9) as usize - pmin as usize;
                        cls[cur[b] as usize] = *c;
                        cur[b] += 1;
                    }
                }
            } else {
                for o in &cl_out {
                    acc.cls.extend_from_slice(&o.shards[q]);
                }
                acc.cls.sort_unstable_by_key(|c| c.user);
            }
            let clicks_of = |cls: &[CLoc], coff: &[u32], p: u32| -> (usize, usize) {
                if counted {
                    if p < pmin || p > pmax {
                        (0, 0)
                    } else {
                        let b = (p - pmin) as usize;
                        (coff[b] as usize, coff[b + 1] as usize)
                    }
                } else {
                    (cls.partition_point(|c| c.user >> 9 < p), cls.partition_point(|c| c.user >> 9 <= p))
                }
            };
            let surv = &survivors[q];
            let mut k = 0;
            while k < surv.len() {
                let u = surv[k].user;
                if u & O_FLAG != 0 {
                    let name = other_name(okey(u));
                    cx_ev_push(acc, name, &surv[k]);
                    k += 1;
                    continue;
                }
                let p = u >> 9;
                let mut r = k + 1;
                while r < surv.len() && surv[r].user >> 9 == p && surv[r].user & O_FLAG == 0 {
                    r += 1;
                }
                let run = &surv[k..r];
                k = r;
                if pinfo[p as usize] == P_COMPLEX || home[p as usize].load(Relaxed) != q as u32 {
                    for e in run {
                        let name = pk_name(pk_of(e.user));
                        cx_ev_push(acc, name, e);
                    }
                    continue;
                }
                let (c0, c1) = clicks_of(&acc.cls, &acc.coff, p);
                let Acc { tev, tcl, ev, cl, cls, urows, camp, ps, cx_ev, cx_cl, .. } = &mut *acc;
                // The usual case: one person key for the whole run (every event
                // from the same ID, or from linked IDs of one simple person).
                let info = pinfo[p as usize];
                let pk0 = pk_of(run[0].user);
                let same = if info == P_NONE {
                    run.iter().all(|e| e.user == run[0].user)
                } else {
                    pk0 >> 63 == 1 && run.iter().all(|e| (e.user & 511) < 32 && info >> (e.user & 511) & 1 == 1)
                };
                if same && !pk_cx(pk0) {
                    ev.clear();
                    ev.extend(run.iter().map(ev_of));
                    cl.clear();
                    for c in &cls[c0..c1] {
                        let pk = pk_of(c.user);
                        let x = Cl { ts: cl_base + c.ts as i64, id: c.id, camp: c.camp };
                        if pk == pk0 {
                            cl.push(x);
                        } else if pk_cx(pk) {
                            let name = pk_name(pk);
                            cx_cl[shard_of(&name)].push(CRec { name, id: x.id, ts: x.ts, camp: x.camp });
                        }
                    }
                    person_old(ev, cl, pk_name(pk0), day0, organic, ndays, ps, urows, camp);
                    continue;
                }
                // Otherwise group this person's events and clicks by person key.
                tev.clear();
                tcl.clear();
                for e in run {
                    let pk = pk_of(e.user);
                    if pk_cx(pk) {
                        let name = pk_name(pk);
                        let x = ev_of(e);
                        cx_ev[shard_of(&name)].push(PRec { name, ts: x.ts, price: x.price, cat: x.cat });
                    } else {
                        tev.push((pk, ev_of(e)));
                    }
                }
                for c in &cls[c0..c1] {
                    let pk = pk_of(c.user);
                    let x = Cl { ts: cl_base + c.ts as i64, id: c.id, camp: c.camp };
                    if pk_cx(pk) {
                        let name = pk_name(pk);
                        cx_cl[shard_of(&name)].push(CRec { name, id: x.id, ts: x.ts, camp: x.camp });
                    } else {
                        tcl.push((pk, x));
                    }
                }
                if tev.is_empty() {
                    continue;
                }
                let one = tev.iter().all(|e| e.0 == tev[0].0);
                if !one {
                    tev.sort_unstable_by_key(|e| e.0);
                    tcl.sort_unstable_by_key(|c| c.0);
                }
                let (mut a, mut b) = (0usize, 0usize);
                while a < tev.len() {
                    let pk = tev[a].0;
                    let mut a2 = a + 1;
                    while a2 < tev.len() && tev[a2].0 == pk {
                        a2 += 1;
                    }
                    while b < tcl.len() && tcl[b].0 < pk && !one {
                        b += 1;
                    }
                    let mut b2 = b;
                    while b2 < tcl.len() && (one || tcl[b2].0 == pk) {
                        b2 += 1;
                    }
                    ev.clear();
                    ev.extend(tev[a..a2].iter().map(|e| e.1));
                    cl.clear();
                    cl.extend(tcl[b..b2].iter().filter(|c| c.0 == pk).map(|c| c.1));
                    person_old(ev, cl, pk_name(pk), day0, organic, ndays, ps, urows, camp);
                    a = a2;
                    b = b2;
                }
            }
    });
    // Odd-ID events are all general.
    if let Some(acc) = accs.first_mut() {
        for r in &odd_surv {
            let name = if r.user & O_FLAG != 0 { other_name(okey(r.user)) } else { pk_name(pk_of(r.user)) };
            cx_ev_push(acc, name, r);
        }
    }
    drop(survivors);
    let n_local: usize = accs.iter().map(|a| a.urows.len()).sum();
    phase(t0, 7, &format!("local people done: {n_local} rows"));

    // ---- The general path for everyone else: per-name shards, as in the main pipeline.
    let cx_pays: Vec<Vec<Vec<PRec<u64>>>> = accs.iter_mut().map(|a| std::mem::take(&mut a.cx_ev)).collect();
    let mut cx_clicks: Vec<Vec<Vec<CRec<u64>>>> = accs.iter_mut().map(|a| std::mem::take(&mut a.cx_cl)).collect();
    cx_clicks.extend(cl_out.into_iter().map(|o| o.cx));
    let n_cx: usize = cx_pays.iter().flat_map(|v| v.iter().map(|x| x.len())).sum();
    // One accumulator per thread (a fold would make one per work split, each
    // with its own campaign table to zero and then sum on one thread).
    // Biggest shards first, so a whale's shard does not finish last.
    let mut order: Vec<usize> = (0..NB).collect();
    order.sort_unstable_by_key(|&p| std::cmp::Reverse(cx_pays.iter().map(|v| v[p].len()).sum::<usize>()));
    let results: Vec<(Vec<URow<u64>>, Vec<[i64; 4]>)> = per_thread(
        NB,
        threads,
        |_| (Vec::new(), vec![[0i64; 4]; size], Scratch::<u64>::default()),
        |(urows, camp, sc), i| {
            process_shard_old(order[i], &cx_pays, &cx_clicks, day0, ndays, cat.ncat, organic, urows, camp, sc);
        },
    )
    .into_iter()
    .map(|(u, c, _)| (u, c))
    .collect();
    phase(t0, 8, &format!("general people done: {n_cx} events"));

    // ---- Write.
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    std::fs::create_dir_all(&ud).expect("mkdir user_daily");
    std::fs::create_dir_all(&cd).expect("mkdir campaign_daily");
    let mut camp_tot = vec![[0i64; 4]; size];
    for c in accs.iter().map(|a| &a.camp).chain(results.iter().map(|r| &r.1)) {
        for (t, x) in camp_tot.iter_mut().zip(c) {
            for k in 0..4 {
                t[k] += x[k];
            }
        }
    }
    let parts: Vec<&[URow<u64>]> = accs.iter().map(|a| &a.urows[..]).chain(results.iter().map(|r| &r.0[..])).collect();
    let nrows: usize = parts.iter().map(|p| p.len()).sum();
    let per_file = nrows.div_ceil(threads).max(1);
    let mut groups: Vec<Vec<&[URow<u64>]>> = vec![Vec::new()];
    let mut in_group = 0usize;
    for u in parts {
        let mut rest: &[URow<u64>] = u;
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
    let camp_names: Vec<(&[u8], &[u8])> = cat.camp_rows.iter().map(|r| (r.1, r.2)).collect();
    rayon::join(
        || {
            groups.par_iter().enumerate().for_each(|(i, parts)| {
                write_user(&ud.join(format!("part-{i:05}.parquet")), parts, &cat.cat_names);
            })
        },
        || write_campaign(&cd.join("part-00000.parquet"), &camp_tot, &camp_names, ncamp, ndays, day0),
    );
    log(t0, &format!("wrote {nrows} user rows in {} files", groups.len()));
    // Hand the input mappings back on all threads, in pieces of at most 64 MB.
    const PIECE: usize = 64 << 20;
    let regions: Vec<(usize, usize)> = inp
        .events
        .iter()
        .chain(&inp.clicks)
        .chain(&inp.links)
        .flat_map(|b| (0..b.len()).step_by(PIECE).map(move |o| (b.as_ptr() as usize + o, PIECE.min(b.len() - o))))
        .collect();
    regions.par_iter().for_each(|&(p, len)| release(p, len));
    log(t0, "done");
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scramble_round_trip() {
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..100_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (v, key) = (s >> 24 & MASK, s.rotate_left(17));
            assert_eq!(unscramble(scramble(v, key), key), v);
        }
    }

    #[test]
    fn base36_fast_matches_slow() {
        let mut s = 7u64;
        let alphabet = b"0123456789abcdefghijklmnopqrstuvwxyzAZ_-/:` \x7f\x80";
        for _ in 0..300_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let n = (s >> 60) as usize % 10;
            let mut buf = vec![b'x'; 8];
            let mut t = s;
            for _ in 0..n {
                t = t.wrapping_mul(6364136223846793005).wrapping_add(1);
                // mostly valid digits, sometimes anything
                let c = if t >> 63 == 0 { alphabet[(t >> 40) as usize % 36] } else { alphabet[(t >> 40) as usize % alphabet.len()] };
                buf.push(c);
            }
            let e = buf.len();
            assert_eq!(b36(&buf, 8, e), b36_slow(&buf[8..e]), "{:?}", &buf[8..e]);
        }
        assert_eq!(b36(b"xxxxxxxxzzzzzzzz", 8, 16), None); // 36^8 - 1 > 2^40
        assert_eq!(b36(b"xxxxxxxx0", 8, 9), Some(0));
        assert_eq!(b36(b"xxxxxxxx01", 8, 10), None);
    }

    #[test]
    fn base36_pair_matches_single() {
        let mut s = 11u64;
        let mut alphabet = b"0123456789abcdefghijklmnopqrstuvwxyzueAZ_".to_vec();
        alphabet.extend([0x7f, 0x80]);
        for _ in 0..300_000 {
            let mut d = b"xxxxxxxxxxxx".to_vec();
            let mut fields = Vec::new();
            for _ in 0..2 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let n = (s >> 59) as usize % 11;
                let start = d.len();
                for j in 0..n {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let c = if j == 0 && s >> 62 != 0 { if s >> 61 & 1 == 0 { b'e' } else { b'u' } } else if s >> 63 == 0 {
                        alphabet[(s >> 40) as usize % 36]
                    } else {
                        alphabet[(s >> 40) as usize % alphabet.len()]
                    };
                    d.push(c);
                }
                fields.push(Fd::new(start, d.len()));
                d.push(b',');
            }
            let one = |f: Fd, p: u8| (f.e() > f.s() && d[f.s()] == p).then(|| b36(&d, f.s() + 1, f.e())).flatten();
            assert_eq!(b36x2(&d, fields[0], b'e', fields[1], b'u'), (one(fields[0], b'e'), one(fields[1], b'u')), "{:?}", String::from_utf8_lossy(&d));
        }
    }

    #[test]
    fn packed_id_matches_key() {
        for y in [0u64, 1, 35, 36, 123_456_789, MASK] {
            let mut s = vec![b'u'];
            let mut buf = [0u8; 8];
            let mut n = 0;
            let mut t = y;
            loop {
                buf[n] = b"0123456789abcdefghijklmnopqrstuvwxyz"[(t % 36) as usize];
                n += 1;
                t /= 36;
                if t == 0 {
                    break;
                }
            }
            s.extend(buf[..n].iter().rev());
            assert_eq!(Some(packed_id(b'u', y)), <u64 as Key<'_>>::make(&s, 0, s.len()));
        }
    }

    #[test]
    fn recovers_keys() {
        let key = 0x5DEE_CE66_D123u64 & MASK;
        let mut pairs = Vec::new();
        let mut s = 99u64;
        for _ in 0..500 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let p = s >> 44; // up to 2^20 people
            let (a, b) = ((s >> 8) % 6, (s >> 16) % 6);
            if a != b {
                pairs.push((scramble(p * 512 + a, key), scramble(p * 512 + b, key)));
            }
        }
        let ys: Vec<u64> = pairs.iter().map(|p| p.0).collect();
        let score = |k: u64| ys.iter().filter(|&&y| unscramble(y, k) >> 9 < 1 << 20).count() as f64 / ys.len() as f64;
        let (k, sc) = solve_key(&pairs, 8, score).unwrap();
        assert_eq!(k, key);
        assert_eq!(sc, 1.0);
    }
}

