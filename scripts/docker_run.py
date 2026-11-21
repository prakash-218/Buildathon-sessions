#!/usr/bin/env python3
"""Run the image the way the scorer does: docker_run.py DATA_DIR OUTPUT_DIR

Same flags as the starter's docker_run.sh, in Python so it also runs on Windows.
Resource limits come from CPUS (default 8) and MEM (default 24g); a laptop with
less memory for Docker can pass MEM=12g, and CPUSET=0-7 pins the CPUs as the
scorer does. KEYS=128 or KEYS=slice forces the pipeline's fallback key types.
Prints the wall time to stderr.

DATA_DIR or OUTPUT_DIR may be `vol:<volume>/<path>` to use a Docker volume
instead of a host folder. On Docker Desktop (Windows, macOS) host folders go
through a slow file-sharing layer, so time runs on volumes.
"""
import os
import subprocess
import sys
import tempfile
import time

HELPER = os.environ.get("HELPER_IMAGE", "rust:1.85")  # any image with a root shell


def mount(spec, target, readonly):
    if spec.startswith("vol:"):
        volume, _, sub = spec[4:].partition("/")
        if not readonly:  # the subpath must exist and be writable by uid 1000
            subprocess.check_call(["docker", "run", "--rm", "-v", f"{volume}:/v", HELPER, "sh", "-c",
                                   f"rm -rf '/v/{sub}' && mkdir -p '/v/{sub}' && chmod 777 '/v/{sub}'"])
        opts = f"type=volume,src={volume},dst={target},volume-subpath={sub}"
        return ["--mount", opts + (",readonly" if readonly else "")]
    path = os.path.abspath(spec)
    if not readonly:
        os.makedirs(path, exist_ok=True)
    return ["-v", f"{path}:{target}" + (":ro" if readonly else "")]


def main():
    data_mount = mount(sys.argv[1], "/data", True)
    out_mount = mount(sys.argv[2], "/output", False)
    image = os.environ.get("IMAGE", "shopping-sessions:latest")
    cpus = os.environ.get("CPUS", "8")
    mem = os.environ.get("MEM", "24g")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        cmd = [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "4096", "--memory", mem, "--memory-swap", mem,
            # The scorer pins 8 hyperthreads; CPUSET=0-7 does the same (on a hybrid
            # Intel laptop that keeps the run off the slow efficiency cores).
            *(["--cpuset-cpus", os.environ["CPUSET"]] if os.environ.get("CPUSET") else ["--cpus", cpus]),
            "--user", "1000:1000",
            *data_mount, *out_mount, "-v", f"{scratch}:/scratch",
            "-e", "DATA_DIR=/data", "-e", "OUTPUT_DIR=/output", "-e", "SCRATCH_DIR=/scratch",
            "-e", "HOME=/scratch", "-e", f"NCPU={cpus.split('.')[0]}", "-e", f"MEM_GB={mem.rstrip('g')}",
            *(["-e", f"KEYS={os.environ['KEYS']}"] if os.environ.get("KEYS") else []),
            image,
        ]
        start = time.monotonic()
        code = subprocess.call(cmd)
        print(f"wall {time.monotonic() - start:.2f}s, exit {code}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
