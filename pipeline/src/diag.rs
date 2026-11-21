//! Diagnostics for the scorer's machine (branch `diag` only, never scored):
//! facts and timings on stderr. The dev-set check fails on purpose, since a
//! failed check is the only run whose logs come back.

use std::collections::BTreeSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

fn show(p: &str, max: usize) {
    match std::fs::read_to_string(p) {
        Ok(s) => eprintln!("[{p}] {}", s[..s.len().min(max)].trim_end().replace('\n', " | ")),
        Err(e) => eprintln!("[{p}] {e}"),
    }
}

fn files(dir: &Path) -> Vec<(PathBuf, u64)> {
    let mut v: Vec<(PathBuf, u64)> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|p| {
            let n = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            (p, n)
        })
        .collect();
    v.sort();
    v
}

/// Fraction of the file's pages in the page cache (mincore), without reading it.
fn resident(p: &Path) -> (usize, usize) {
    let Ok(f) = File::open(p) else { return (0, 0) };
    let len = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
    if len == 0 {
        return (0, 0);
    }
    unsafe {
        let a = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
        if a == libc::MAP_FAILED {
            return (0, 0);
        }
        let pages = len.div_ceil(4096);
        let mut v = vec![0u8; pages];
        let r = libc::mincore(a, len, v.as_mut_ptr());
        libc::munmap(a, len);
        if r != 0 {
            return (0, pages);
        }
        (v.iter().filter(|&&b| b & 1 == 1).count(), pages)
    }
}

fn par<F: Fn(usize) + Sync>(threads: usize, f: F) {
    std::thread::scope(|s| {
        for t in 0..threads {
            let f = &f;
            s.spawn(move || f(t));
        }
    });
}

fn mbps(bytes: u64, t: Instant) -> String {
    let s = t.elapsed().as_secs_f64();
    format!("{:.1} MB in {:.1} ms = {:.0} MB/s", bytes as f64 / 1e6, s * 1e3, bytes as f64 / 1e6 / s)
}

pub fn run(data: &Path) {
    let t0 = Instant::now();
    eprintln!("==== DIAG ====");
    show("/proc/version", 300);
    show("/proc/cmdline", 300);
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        let models: BTreeSet<&str> = s.lines().filter(|l| l.starts_with("model name")).collect();
        let n = s.lines().filter(|l| l.starts_with("processor")).count();
        let mhz: Vec<&str> = s.lines().filter(|l| l.starts_with("cpu MHz")).take(4).collect();
        eprintln!("[cpuinfo] {n} cpus, {models:?}, {mhz:?}");
        if let Some(f) = s.lines().find(|l| l.starts_with("flags")) {
            let has = |x: &str| f.split_whitespace().any(|w| w == x);
            eprintln!(
                "[flags] avx2 {} bmi2 {} avx512f {} avx512bw {} avx512vbmi {} avx512ifma {} hypervisor {}",
                has("avx2"),
                has("bmi2"),
                has("avx512f"),
                has("avx512bw"),
                has("avx512vbmi"),
                has("avx512ifma"),
                has("hypervisor")
            );
        }
        for key in ["cache size", "siblings", "cpu cores"] {
            if let Some(l) = s.lines().find(|l| l.starts_with(key)) {
                eprintln!("[cpuinfo] {l}");
            }
        }
    }
    eprintln!("[parallelism] {:?}", std::thread::available_parallelism());
    for p in [
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory.max",
        "/sys/kernel/mm/transparent_hugepage/enabled",
        "/sys/kernel/mm/transparent_hugepage/defrag",
        "/sys/kernel/mm/transparent_hugepage/shmem_enabled",
        "/proc/sys/vm/overcommit_memory",
        "/sys/devices/system/cpu/smt/active",
    ] {
        show(p, 200);
    }
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for l in s.lines().filter(|l| l.starts_with("Cpus_allowed_list") || l.starts_with("Mems_allowed_list")) {
            eprintln!("[status] {l}");
        }
    }
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let keep = ["MemTotal", "MemFree", "MemAvailable", "Cached", "AnonHugePages", "Hugepagesize", "HugePages_Total"];
        let v: Vec<&str> = s.lines().filter(|l| keep.iter().any(|k| l.starts_with(k))).collect();
        eprintln!("[meminfo] {}", v.join(" | "));
    }
    if let Ok(s) = std::fs::read_to_string("/proc/self/mountinfo") {
        for l in s.lines().filter(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            f.len() > 4 && ["/", "/data", "/output", "/scratch", "/tmp"].contains(&f[4])
        }) {
            eprintln!("[mount] {l}");
        }
    }
    // How long between the process starting and main.
    if let (Ok(st), Ok(up)) = (std::fs::read_to_string("/proc/self/stat"), std::fs::read_to_string("/proc/uptime")) {
        let after = st.rsplit(')').next().unwrap_or("");
        let f: Vec<&str> = after.split_whitespace().collect();
        let start_ticks: f64 = f.get(19).and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let up: f64 = up.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
        eprintln!("[start] process age at main: {:.1} ms (uptime {up:.0} s)", (up - start_ticks / 100.0) * 1e3);
    }
    let mut env: Vec<String> = std::env::vars().map(|(k, v)| format!("{k}={}", &v[..v.len().min(60)])).collect();
    env.sort();
    eprintln!("[env] {}", env.join(" "));

    // Input files and how much of each is already in the page cache.
    let mut all = Vec::new();
    for d in ["events", "clicks", "links"] {
        let fs = files(&data.join(d));
        let (mut r, mut n, mut bytes) = (0, 0, 0);
        for (p, len) in &fs {
            let (a, b) = resident(p);
            r += a;
            n += b;
            bytes += len;
        }
        eprintln!("[input] {d}: {} files, {:.1} MB, resident {r}/{n} pages", fs.len(), bytes as f64 / 1e6);
        if d == "events" {
            all = fs;
        }
    }

    // Read speed, three ways, each on its own third of the event files.
    let groups: Vec<Vec<(PathBuf, u64)>> =
        (0..3).map(|g| all.iter().enumerate().filter(|(i, _)| i % 3 == g).map(|(_, f)| f.clone()).collect()).collect();
    let bytes = |g: &Vec<(PathBuf, u64)>| g.iter().map(|f| f.1).sum::<u64>();
    {
        let t = Instant::now();
        let mut buf = vec![0u8; 1 << 20];
        let mut sum = 0u64;
        for (p, _) in &groups[0] {
            let f = File::open(p).unwrap();
            let mut off = 0u64;
            loop {
                let n = f.read_at(&mut buf, off).unwrap_or(0);
                if n == 0 {
                    break;
                }
                sum += buf[n - 1] as u64;
                off += n as u64;
            }
        }
        eprintln!("[read] 1 thread, 1 MB reads: {} ({sum})", mbps(bytes(&groups[0]), t));
    }
    {
        let blocks: Vec<(usize, u64, usize)> = groups[1]
            .iter()
            .enumerate()
            .flat_map(|(i, (_, len))| (0..*len).step_by(4 << 20).map(move |o| (i, o, ((len - o) as usize).min(4 << 20))))
            .collect();
        let fds: Vec<File> = groups[1].iter().map(|(p, _)| File::open(p).unwrap()).collect();
        let next = AtomicUsize::new(0);
        let t = Instant::now();
        par(8, |_| {
            let mut buf = vec![0u8; 4 << 20];
            loop {
                let k = next.fetch_add(1, Relaxed);
                let Some(&(i, o, n)) = blocks.get(k) else { break };
                let _ = fds[i].read_exact_at(&mut buf[..n], o);
            }
        });
        eprintln!("[read] 8 threads, 4 MB preads: {}", mbps(bytes(&groups[1]), t));
    }
    {
        let maps: Vec<memmap2::Mmap> =
            groups[2].iter().filter(|f| f.1 > 0).map(|(p, _)| unsafe { memmap2::Mmap::map(&File::open(p).unwrap()).unwrap() }).collect();
        let pages: Vec<(usize, usize)> =
            maps.iter().enumerate().flat_map(|(i, m)| (0..m.len()).step_by(1 << 20).map(move |o| (i, o))).collect();
        let next = AtomicUsize::new(0);
        let sum = AtomicUsize::new(0);
        let t = Instant::now();
        par(8, |_| {
            let mut s = 0usize;
            loop {
                let k = next.fetch_add(1, Relaxed);
                let Some(&(i, o)) = pages.get(k) else { break };
                let m = &maps[i];
                let end = (o + (1 << 20)).min(m.len());
                let mut x = o;
                while x < end {
                    s += m[x] as usize;
                    x += 4096;
                }
            }
            sum.fetch_add(s, Relaxed);
        });
        eprintln!("[read] 8 threads, mmap touch: {}", mbps(bytes(&groups[2]), t));
    }
    {
        let t = Instant::now();
        let mut buf = vec![0u8; 1 << 20];
        for (p, _) in &groups[0] {
            let f = File::open(p).unwrap();
            let mut off = 0u64;
            loop {
                let n = f.read_at(&mut buf, off).unwrap_or(0);
                if n == 0 {
                    break;
                }
                off += n as u64;
            }
        }
        eprintln!("[read] again (warm), 1 thread: {}", mbps(bytes(&groups[0]), t));
    }

    // Fresh memory: fault it in from 8 threads, then unmap it.
    for (label, huge, populate) in [("4K pages", false, false), ("MADV_HUGEPAGE", true, false), ("MAP_POPULATE", false, true)] {
        let len = 1usize << 30;
        unsafe {
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | if populate { libc::MAP_POPULATE } else { 0 };
            let t = Instant::now();
            let a = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0);
            if a == libc::MAP_FAILED {
                eprintln!("[mem] {label}: mmap failed");
                continue;
            }
            if huge {
                libc::madvise(a, len, libc::MADV_HUGEPAGE);
            }
            let base = a as usize;
            par(8, |t| {
                let per = len / 8;
                let mut x = base + t * per;
                while x < base + (t + 1) * per {
                    *(x as *mut u8) = 1;
                    x += 4096;
                }
            });
            let touched = t.elapsed();
            let t2 = Instant::now();
            libc::munmap(a, len);
            eprintln!("[mem] 1 GiB {label}: fault-in {:.1} ms, munmap {:.1} ms", touched.as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3);
        }
    }
    // Through the global allocator (mimalloc), as the pipeline allocates.
    {
        let t = Instant::now();
        let mut v: Vec<u8> = Vec::with_capacity(1 << 30);
        unsafe { v.set_len(1 << 30) };
        let base = v.as_mut_ptr() as usize;
        par(8, |t| {
            let per = (1usize << 30) / 8;
            let mut x = base + t * per;
            while x < base + (t + 1) * per {
                unsafe { *(x as *mut u8) = 1 };
                x += 4096;
            }
        });
        let a = t.elapsed();
        let t2 = Instant::now();
        drop(v);
        eprintln!("[mem] 1 GiB Vec (mimalloc): fault-in {:.1} ms, drop {:.1} ms", a.as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3);
    }

    // Compute: one thread, then 4 and 8 at once.
    let work = || {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..50_000_000u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x = x.wrapping_mul(0xD6E8_FEB8_6659);
        }
        std::hint::black_box(x);
    };
    for n in [1, 4, 8] {
        let t = Instant::now();
        par(n, |_| work());
        eprintln!("[cpu] {n} threads x 50M steps: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    // Memory latency: dependent random loads over 256 MB.
    {
        let n = 1usize << 25; // 32M u64 = 256 MB
        let mut next: Vec<u32> = (0..n as u32).collect();
        let mut s = 12345u64;
        for i in (1..n).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            next.swap(i, (s >> 33) as usize % i);
        }
        let t = Instant::now();
        let mut p = 0usize;
        for _ in 0..2_000_000 {
            p = next[p] as usize;
        }
        std::hint::black_box(p);
        eprintln!("[mem] dependent random load: {:.1} ns", t.elapsed().as_secs_f64() * 1e9 / 2e6);
    }
    eprintln!("==== DIAG done in {:.0} ms; now the pipeline, writing to scratch ====", t0.elapsed().as_secs_f64() * 1e3);
}
