#!/usr/bin/env python3
"""Aperture — unit tests + benchmark harness.

Runs pure-logic unit tests and (optionally) load/generation benchmarks,
then writes a timestamped report.

Usage:
    python tests.py                 # unit tests + benchmarks (loads model)
    python tests.py --no-bench      # unit tests only, no model load
    python tests.py --bench-only    # skip unit tests, benchmark only
    python tests.py --runs 3        # generation benchmark iterations
    python tests.py --out report.txt
"""
import os, sys, io, time, json, platform, hashlib, secrets, argparse, traceback, shutil, subprocess
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_DIR = ROOT / ".venv"

def _venv_python():
    import os
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# re-exec into the venv before importing main (which pulls in torch etc.)
if sys.prefix != str(VENV_DIR.resolve()) and _venv_python().exists():
    import os
    os.execv(str(_venv_python()),
             [str(_venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]])

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# import the module under test
import main as ap


# ───────────────────────────── unit tests ─────────────────────────────
class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.log = []

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            self.log.append(f"  PASS  {name}")
        else:
            self.failed += 1
            self.log.append(f"  FAIL  {name}  {detail}")


def run_unit_tests():
    r = Results()
    base = dict(ap.DEFAULT_CONFIG)
    defaults = {"size": "832x1216", "seed": "1234", "guidance": "5.0",
                "steps": "30", "scheduler": "dpmpp2m", "q": "1",
                "sketchmode": False, "neg": ap.DEFAULT_NEG, "filename": None}

    # --- parse_inline: plain prompt ---
    p, c, ctl, prompt = ap.parse_inline("a woman by a window", defaults)
    r.check("plain prompt parsed", prompt == "a woman by a window")
    r.check("plain prompt no control", ctl == [] and c == [])

    # --- parse_inline: leading params split from prompt ---
    p, c, ctl, prompt = ap.parse_inline("--size=1024x1024 --seed=7 street photo", defaults)
    r.check("size param", p["size"] == "1024x1024")
    r.check("seed param", p["seed"] == "7")
    r.check("prompt after params", prompt == "street photo")

    # --- parse_inline: bare boolean flag ---
    p, c, ctl, prompt = ap.parse_inline("--sketchmode quick test", defaults)
    r.check("sketchmode bool true", p["sketchmode"] is True)

    # --- parse_inline: conf ops ---
    p, c, ctl, prompt = ap.parse_inline("--conf.hf_token=hf_abc123", defaults)
    r.check("conf set parsed", ("hf_token", "hf_abc123") in c)
    p, c, ctl, prompt = ap.parse_inline("--conf.list", defaults)
    r.check("conf query parsed", ("list", None) in c)

    # --- parse_inline: control tokens ---
    p, c, ctl, prompt = ap.parse_inline("--jobs", defaults)
    r.check("jobs control", "--jobs" in ctl)
    p, c, ctl, prompt = ap.parse_inline("--reload", defaults)
    r.check("reload control", "--reload" in ctl)
    p, c, ctl, prompt = ap.parse_inline("--jobs.clear", defaults)
    r.check("jobs.clear control", "--jobs.clear" in ctl)

    # --- parse_inline: params only, no prompt ---
    p, c, ctl, prompt = ap.parse_inline("--steps=40", defaults)
    r.check("params only empty prompt", prompt == "")

    # --- make_scheduler: valid + invalid ---
    try:
        # build_scheduler needs a real config; just check the lookup/raise path
        ap.make_scheduler("not_a_sched", {})
        r.check("invalid scheduler raises", False, "no exception")
    except ValueError:
        r.check("invalid scheduler raises", True)
    except Exception as e:
        # wrong config type can raise other errors for valid names; ignore
        r.check("invalid scheduler raises", isinstance(e, ValueError), str(e))

    # --- resolve_spec: sketchmode caps ---
    sk = dict(defaults); sk["sketchmode"] = True; sk["steps"] = "30"; sk["guidance"] = "5.0"
    spec = ap.resolve_spec(sk, "x", "/tmp")
    r.check("sketch caps steps<=12", spec["steps"] <= 12, f"got {spec['steps']}")
    r.check("sketch caps cfg<=3", spec["guidance"] <= 3.0, f"got {spec['guidance']}")

    # --- resolve_spec: size parse ---
    spec = ap.resolve_spec(defaults, "x", "/tmp")
    r.check("size width", spec["w"] == 832)
    r.check("size height", spec["h"] == 1216)

    # --- resolve_spec: explicit filename gets .png ---
    fn = dict(defaults); fn["filename"] = "myshot"
    spec = ap.resolve_spec(fn, "x", "/tmp")
    r.check("filename appends png", spec["filename"] == "myshot.png")
    fn["filename"] = "myshot.png"
    spec = ap.resolve_spec(fn, "x", "/tmp")
    r.check("filename keeps png", spec["filename"] == "myshot.png")

    # --- resolve_spec: bad size raises ---
    bad = dict(defaults); bad["size"] = "garbage"
    try:
        ap.resolve_spec(bad, "x", "/tmp")
        r.check("bad size raises", False, "no exception")
    except ValueError:
        r.check("bad size raises", True)

    # --- config round-trip (isolated temp config) ---
    real_dir, real_path = ap.CONFIG_DIR, ap.CONFIG_PATH
    tmp = ROOT / "_test_configs"
    try:
        ap.CONFIG_DIR = tmp
        ap.CONFIG_PATH = tmp / "config.txt"
        if tmp.exists():
            shutil.rmtree(tmp)
        cfg = ap.load_config()
        r.check("load creates config file", ap.CONFIG_PATH.exists())
        ap.set_config(cfg, "steps", "42")
        cfg2 = ap.load_config()
        r.check("config persists set value", cfg2["steps"] == "42")
        msg = ap.set_config(cfg, "bogus_key", "x")
        r.check("unknown config key rejected", "unknown" in msg.lower())
        ap.set_config(cfg, "hf_token", "hf_secret")
        raw = ap.CONFIG_PATH.read_text()
        r.check("token written to file", "hf_secret" in raw)  # plain-text by design
    finally:
        ap.CONFIG_DIR, ap.CONFIG_PATH = real_dir, real_path
        if tmp.exists():
            shutil.rmtree(tmp)

    # --- slice_prompt ---
    r.check("slice short unchanged", ap.slice_prompt("hi", 22) == "hi")
    r.check("slice long truncated", len(ap.slice_prompt("x" * 50, 22)) == 22)

    return r


# ──────────────────────────── system info ─────────────────────────────
def system_info():
    info = {}
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["processor"] = platform.processor() or "n/a"
    info["python"] = platform.python_version()

    # CPU / chip detail (macOS)
    def sysctl(key):
        try:
            return subprocess.check_output(["sysctl", "-n", key],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    chip = sysctl("machdep.cpu.brand_string")
    if chip:
        info["cpu"] = chip
    ncpu = sysctl("hw.ncpu")
    if ncpu:
        info["cpu_threads"] = ncpu
    membytes = sysctl("hw.memsize")
    if membytes:
        try:
            info["ram_gb"] = round(int(membytes) / (1024**3), 1)
        except ValueError:
            pass

    # GPU / MPS
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps_available"] = bool(getattr(torch.backends, "mps", None)
                                     and torch.backends.mps.is_available())
        info["cuda_available"] = torch.cuda.is_available()
    except Exception as e:
        info["torch"] = f"import failed: {e}"

    # Network: connectivity + rough latency, no external identifying calls
    import socket
    def ping_host(host, port, timeout=2.0):
        t = time.time()
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            return round((time.time() - t) * 1000, 1)
        except Exception:
            return None
    info["net_hf_ms"] = ping_host("huggingface.co", 443)
    info["net_pypi_ms"] = ping_host("pypi.org", 443)
    info["net_online"] = info["net_hf_ms"] is not None or info["net_pypi_ms"] is not None

    return info


def anon_run_id():
    """Salted one-way id. timestamp+device are hashed with a random per-run
    salt that is NOT stored, so the digest can't be brute-forced back to its
    inputs. Returns (digest, iso_timestamp_for_human_report)."""
    ts = time.time()
    # a device fingerprint; node name + machine, not externally meaningful
    device = f"{platform.node()}--{platform.machine()}"
    salt = secrets.token_bytes(16)               # random, discarded after use
    payload = f"{ts}--{device}".encode()
    digest = hashlib.sha256(salt + payload).hexdigest()
    # human-readable timestamp is fine to show; the *link* to device is what's hidden
    return digest, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


# ───────────────────────────── benchmarks ─────────────────────────────
def bench_load(cfg, logs):
    import torch  # noqa
    t = time.time()
    pipe, bsc, sched = ap.build_pipeline(cfg)
    elapsed = time.time() - t
    return pipe, bsc, sched, elapsed


def bench_generate(pipe, sched, runs, logs):
    import torch
    timings = []
    params = {"size": "832x1216", "seed": "1234", "guidance": "5.0",
              "steps": "20", "scheduler": sched, "q": "1",
              "sketchmode": False, "neg": ap.DEFAULT_NEG, "filename": None}
    out = ROOT / "_bench_out"
    out.mkdir(exist_ok=True)
    try:
        for i in range(runs):
            spec = ap.resolve_spec(params, "benchmark portrait, soft light", str(out))
            spec["seed"] = 1234 + i
            spec["outpath"] = str(out / f"bench_{i}.png")
            steps = spec["steps"]
            g = torch.Generator(device="mps").manual_seed(spec["seed"])
            t = time.time()
            img = pipe(prompt=spec["prompt"], negative_prompt=spec["neg"],
                       num_inference_steps=steps, guidance_scale=spec["guidance"],
                       width=spec["w"], height=spec["h"], generator=g).images[0]
            img.save(spec["outpath"])
            dt = time.time() - t
            timings.append(dt)
            logs.append(f"  gen run {i+1}/{runs}: {dt:.2f}s ({steps} steps)")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return timings


# ─────────────────────────────── report ───────────────────────────────
def write_report(path, run_id, ts_human, sysinfo, unit, bench, logs, errors):
    L = []
    L.append("=" * 60)
    L.append("APERTURE — TEST & BENCHMARK REPORT")
    L.append("=" * 60)
    L.append(f"run id (salted sha256, non-reversible): {run_id}")
    L.append(f"generated: {ts_human}")
    L.append("")

    L.append("── system ─────────────────────────────────────────────")
    for k, v in sysinfo.items():
        L.append(f"  {k:18} {v}")
    L.append("")

    L.append("── unit tests ─────────────────────────────────────────")
    if unit is None:
        L.append("  (skipped)")
    else:
        L.extend(unit.log)
        L.append("")
        L.append(f"  total: {unit.passed} passed, {unit.failed} failed")
    L.append("")

    L.append("── benchmarks ─────────────────────────────────────────")
    if bench is None:
        L.append("  (skipped)")
    else:
        if "load_s" in bench:
            L.append(f"  pipeline load:        {bench['load_s']:.2f}s")
        if bench.get("gen_timings"):
            g = bench["gen_timings"]
            L.append(f"  generation runs:      {len(g)}")
            L.append(f"  gen mean:             {sum(g)/len(g):.2f}s")
            L.append(f"  gen min / max:        {min(g):.2f}s / {max(g):.2f}s")
            L.append(f"  steps per run:        20")
    L.append("")

    L.append("── logs ───────────────────────────────────────────────")
    for line in logs:
        L.append(line)
    L.append("")

    L.append("── errors ─────────────────────────────────────────────")
    if not errors:
        L.append("  (none)")
    else:
        for e in errors:
            L.append(e)
    L.append("")
    L.append("=" * 60)

    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap_arg = argparse.ArgumentParser(description="Aperture tests + benchmarks")
    ap_arg.add_argument("--out", default=None,
                        help="report path (default: ./bench/<run_id>.txt)")
    ap_arg.add_argument("--runs", type=int, default=3,
                        help="generation benchmark iterations")
    ap_arg.add_argument("--no-bench", action="store_true",
                        help="unit tests only, don't load the model")
    ap_arg.add_argument("--bench-only", action="store_true",
                        help="skip unit tests")
    a = ap_arg.parse_args()

    run_id, ts_human = anon_run_id()
    logs, errors = [], []

    if a.out is None:
        bench_dir = ROOT / "bench"
        bench_dir.mkdir(exist_ok=True)
        a.out = str(bench_dir / f"{run_id}.txt")

    print("collecting system info ...")
    sysinfo = system_info()

    unit = None
    if not a.bench_only:
        print("running unit tests ...")
        try:
            unit = run_unit_tests()
            print(f"  {unit.passed} passed, {unit.failed} failed")
        except Exception:
            errors.append("unit test harness crashed:\n" + traceback.format_exc())

    bench = None
    if not a.no_bench:
        print("running benchmarks (loads the model) ...")
        bench = {}
        try:
            cfg = ap.load_config()
            if cfg.get("hf_token"):
                os.environ["HF_TOKEN"] = cfg["hf_token"]
                os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg["hf_token"]
            pipe, bsc, sched, load_s = bench_load(cfg, logs)
            bench["load_s"] = load_s
            logs.append(f"  pipeline loaded in {load_s:.2f}s")
            try:
                timings = bench_generate(pipe, sched, a.runs, logs)
                bench["gen_timings"] = timings
            except Exception:
                errors.append("generation benchmark failed:\n" + traceback.format_exc())
        except Exception:
            errors.append("pipeline load failed:\n" + traceback.format_exc())

    write_report(a.out, run_id, ts_human, sysinfo, unit, bench, logs, errors)
    print(f"\nreport written to {a.out}")

    # non-zero exit if any unit test failed, handy for CI
    if unit and unit.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()