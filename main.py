#!/usr/bin/env python3
"""Aperture — interactive photoreal SDXL shell for Apple Silicon (MPS).

Self-bootstrapping: first run creates a venv, installs deps, writes
./configs/config.txt, then launches the shell.
    git clone <repo>; cd <repo>; python main.py
"""
import os, sys, subprocess
from pathlib import Path

# ─────────────────── self-bootstrap (runs before heavy imports) ───────────────────
VENV_DIR = Path(__file__).parent / ".venv"
REQUIREMENTS = ["torch", "torchvision", "diffusers", "transformers",
                "accelerate", "safetensors"]


def _venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _check_python():
    if sys.version_info[:2] >= (3, 13):
        print("warning: Python 3.13 has rough edges with the ML stack "
              "(and a common _lzma issue). 3.12 is recommended.")
    try:
        import lzma  # noqa: F401
    except ModuleNotFoundError:
        print("error: your Python is missing _lzma. On macOS:\n"
              "  brew install xz   then rebuild python (e.g. pyenv install 3.12.8)")
        sys.exit(1)


def _ensure_env():
    if sys.prefix == str(VENV_DIR.resolve()):
        return
    vpy = _venv_python()
    if not vpy.exists():
        _check_python()
        print("first run — creating virtual environment in .venv ...")
        import venv
        venv.create(VENV_DIR, with_pip=True)
        print("installing dependencies (a few minutes the first time) ...")
        subprocess.check_call([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
        req_file = Path(__file__).parent / "requirements.txt"
        if req_file.exists():
            subprocess.check_call([str(vpy), "-m", "pip", "install",
                                   "-r", str(req_file)])
        else:
            subprocess.check_call([str(vpy), "-m", "pip", "install", *REQUIREMENTS])
        print("environment ready.\n")
    os.execv(str(vpy), [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_env()

# ─────────────────── heavy imports (only inside the venv) ───────────────────
import time, re, threading, queue, itertools, gc
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)

# ─────────────────────────────── config ───────────────────────────────
CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIG_DIR / "config.txt"

DEFAULT_CONFIG = {
    "hf_token": "",
    "checkpoint": "stabilityai/stable-diffusion-xl-base-1.0",
    "outdir": "./output",
    "seed": "1234",
    "steps": "30",
    "scheduler": "dpmpp2m",
    "guidance": "5.0",
    "default_size": "832x1216",
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        for raw in CONFIG_PATH.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    else:
        save_config(cfg)
        print(f"created config at {CONFIG_PATH}")
    return cfg


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Aperture config — plain key=value, one per line"]
    for k in DEFAULT_CONFIG:
        lines.append(f"{k}={cfg.get(k, DEFAULT_CONFIG[k])}")
    CONFIG_PATH.write_text("\n".join(lines) + "\n")


def set_config(cfg, key, value):
    if key not in DEFAULT_CONFIG:
        return f"unknown config key '{key}'. valid: {', '.join(DEFAULT_CONFIG)}"
    cfg[key] = value
    save_config(cfg)
    shown = "***" if key == "hf_token" and value else value
    return f"config: {key} = {shown}"


# ─────────────────────────── generation setup ──────────────────────────
DEFAULT_NEG = ("cgi, 3d render, airbrushed, plastic skin, smooth skin, waxy, "
               "oversaturated, illustration, painting, drawing, deformed, "
               "extra fingers, bad anatomy, lowres, blurry")

SCHEDULERS = {
    "dpmpp2m":  (DPMSolverMultistepScheduler, {"use_karras_sigmas": True,
                 "algorithm_type": "dpmsolver++"}),
    "dpmppsde": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
    "euler":    (EulerDiscreteScheduler, {}),
    "eulera":   (EulerAncestralDiscreteScheduler, {}),
    "unipc":    (UniPCMultistepScheduler, {}),
}

HELP_TEXT = """\
Aperture — commands & inline params
─────────────────────────────────────────────────
Type a prompt and press Enter to generate. Inline params go at the
START of the line (before the prompt text), space-separated.

Generation params:
  --size=WxH        output dimensions      (default 832x1216)
  --seed=N          base seed              (default from config)
  --steps=N         denoise steps         (default 30)
  --scheduler=NAME  sampler: dpmpp2m, dpmppsde, euler, eulera, unipc
  --guidance=F      CFG scale             (default 5.0)
  --q=N             how many images       (seed increments per image)
  --filename=NAME   output name (.png optional; _1.._N if q>1)
  --neg="..."       per-shot negative prompt override
  --sketchmode      fast draft: caps steps@12, cfg@3.0 (resets after)
  --upscale=SPEC    post-resize: x2 (multiplier) or WxH (target size).
                    WxH preserves aspect (cover + center-crop, no stretch).
                    add :refine for an img2img detail pass that regenerates
                    texture — e.g. --upscale=x2:refine, --upscale=1664x2432:refine

Background jobs:
  --jobs            toggle background mode on/off. When ON, prompts are
                    queued and generated by a worker thread while you keep
                    typing. The job list shows above the input line:
                    [#id | prompt... | filename | %/state]
                    States: pending, NN%, done, error
  --jobs.clear      remove finished/errored jobs from the list

Maintenance:
  --reload          reload config from disk and rebuild the pipeline
                    (applies --conf.checkpoint without restarting; waits
                    for active jobs to finish first)

Config (persisted to ./configs/config.txt):
  --conf.KEY=VALUE  set a config value, e.g. --conf.hf_token=hf_xxxx
  --conf.list       show current config (token masked)
  --conf.path       print config file location

Shell commands:
  --help, --h       show this manual
  (empty line)      ignored (refreshes job list in jobs mode)
  Ctrl+C            exit (waits for the running job to finish)

Examples:
  >>> --jobs
  >>> --q=4 woman by a cafe window, 35mm, natural skin
  >>> --conf.checkpoint=Juggernaut_XL_v9.safetensors
  >>> --reload
  >>> --scheduler=eulera --steps=40 close-up portrait, freckles
"""


def make_scheduler(name, base_config):
    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler '{name}', "
                         f"choose from {', '.join(SCHEDULERS)}")
    cls, kw = SCHEDULERS[name]
    return cls.from_config(base_config, **kw)

# pixel budget above which attention slicing is worth it on a 16GB machine.
# 832x1216 ≈ 1.01M px sits just under; 1024x1024 ≈ 1.05M just over.
SLICE_PIXEL_THRESHOLD = 1_040_000
def _apply_slicing(pipe, size_str, q):
    """Enable/disable slicing based on render size and batch count.

    - attention slicing: only above the pixel threshold (large renders)
    - vae slicing: only for batches (q>1); pointless overhead for single images
    """
    try:
        w, h = (int(x) for x in str(size_str).lower().split("x"))
        px = w * h
    except ValueError:
        px = SLICE_PIXEL_THRESHOLD + 1  # unknown -> be safe, slice

    if px > SLICE_PIXEL_THRESHOLD:
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    if q > 1:
        pipe.enable_vae_slicing()
    else:
        pipe.disable_vae_slicing()

def build_pipeline(cfg):
    """Construct pipeline + scheduler from current config."""
    checkpoint = cfg["checkpoint"]
    print(f"loading {checkpoint} ...")
    t0 = time.time()
    if str(checkpoint).endswith(".safetensors"):
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            checkpoint, torch_dtype=torch.float16, use_safetensors=True)
        
    # (1) fp16-fix VAE: stays in fp16 (faster decode than fp32) while avoiding
    # the overflow that produces black images on MPS.
    try:
        fixed_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe.vae = fixed_vae
        print("  using sdxl-vae-fp16-fix")
    except Exception as e:
        # fall back to the safe fp32 path if the download isn't available
        pipe.vae = pipe.vae.to(torch.float32)
        print(f"  fp16-fix VAE unavailable ({e}); using fp32 VAE fallback")

    pipe = pipe.to("mps")

    # (5)(6) slicing trades speed for lower peak memory — only enable it when
    # the workload actually needs it. Decide from the default render size;
    # per-shot overrides are handled in apply_slicing() below.
    _apply_slicing(pipe, cfg["default_size"], q=1)

    base_sched_config = pipe.scheduler.config
    pipe.scheduler = make_scheduler(cfg["scheduler"], base_sched_config)

    # (3) verify we're actually on the GPU — a silent CPU fallback is ~10x slower
    dev = str(pipe.unet.device)
    if "mps" not in dev:
        print(f"  WARNING: pipeline is on '{dev}', not mps — generation will be "
              f"very slow. Check torch MPS availability.")
    else:
        print(f"  device: {dev}")

    # (4) warmup: first generation compiles Metal kernels lazily; do a tiny
    # throwaway run now so the user's first real image isn't penalized.
    try:
        t_w = time.time()
        _ = pipe(prompt="warmup", num_inference_steps=1,
                 width=256, height=256,
                 generator=torch.Generator(device="mps").manual_seed(0)).images[0]
        print(f"  warmup done in {time.time()-t_w:.1f}s")
    except Exception as e:
        print(f"  warmup skipped: {e}")

    # img2img pipeline for --upscale refine; shares components (no extra load)
    refiner = StableDiffusionXLImg2ImgPipeline(**pipe.components)

    print(f"ready in {time.time()-t0:.1f}s")
    return pipe, refiner, base_sched_config, cfg["scheduler"]

def refine_image(refiner, img, prompt, neg, steps, guidance):
    """Low-denoise img2img pass to add real detail after upscaling.
    strength=0.3 keeps composition/identity intact and only sharpens texture."""
    w, h = img.size
    g = torch.Generator(device="mps").manual_seed(secrets.randbelow(SEED_MAX + 1))
    out = refiner(
        prompt=prompt,
        negative_prompt=neg,
        image=img,
        strength=0.3,            # low: preserves the image, adds detail
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images[0]
    return out

def parse_upscale(value, src_w, src_h):
    """Resolve --upscale to (target_w, target_h, mode, refine).

    Append ':refine' to run an img2img detail pass after resizing:
      --upscale=x2:refine   --upscale=1664x2432:refine
    """
    v = str(value).lower().strip()
    refine = False
    if v.endswith(":refine"):
        refine = True
        v = v[:-len(":refine")]

    m = re.fullmatch(r"(\d+)x(\d+)", v)
    if m:
        return int(m.group(1)), int(m.group(2)), "cover", refine

    m = re.fullmatch(r"x?(\d+(?:\.\d+)?)x?", v)
    if m:
        factor = float(m.group(1))
        if factor <= 0:
            raise ValueError("upscale factor must be positive")
        return int(round(src_w * factor)), int(round(src_h * factor)), "scale", refine

    raise ValueError(f"bad --upscale='{value}' (use x2, WxH, optional :refine)")

from PIL import Image
def upscale_image(img, target_w, target_h, mode):
    """Resize a PIL image. 'scale' = direct resample to target.
    'cover' = preserve aspect ratio, fill the target, center-crop the
    overflow (never stretches)."""
    if mode == "scale":
        return img.resize((target_w, target_h), Image.LANCZOS)

    # cover: scale so the image fully covers the target, then center-crop
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))

def parse_inline(line, defaults):
    """Pull leading --key=value, --conf.*, control tokens, and bare flags.

    Returns (params, conf_ops, control, prompt).
    """
    params = dict(defaults)
    conf_ops = []
    control = []
    tokens = line.split()
    i = 0
    while i < len(tokens) and tokens[i].startswith("--"):
        tok = tokens[i]
        if tok == "--jobs" or tok.startswith("--jobs.") or tok == "--reload":
            control.append(tok)
            i += 1
            continue
        if tok.startswith("--conf."):
            rest = tok[len("--conf."):]
            if "=" in rest:
                k, v = rest.split("=", 1)
                conf_ops.append((k.strip(), v))
            else:
                conf_ops.append((rest.strip(), None))
            i += 1
            continue
        m = re.match(r"--([\w]+)=(.+)", tok)
        if m:
            params[m.group(1)] = m.group(2)
        else:
            params[tok[2:]] = True
        i += 1
    prompt = " ".join(tokens[i:]).strip()
    return params, conf_ops, control, prompt


def apply_conf_ops(cfg, conf_ops, defaults, state):
    live = {"seed": "seed", "steps": "steps", "scheduler": "scheduler",
            "guidance": "guidance", "default_size": "size"}
    for key, val in conf_ops:
        if val is None:
            if key == "list":
                print(f"  config: {CONFIG_PATH}")
                for k in DEFAULT_CONFIG:
                    v = cfg.get(k, "")
                    if k == "hf_token":
                        v = "*** (set)" if v else "(empty)"
                    print(f"    {k} = {v}")
            elif key == "path":
                print(f"  {CONFIG_PATH}")
            else:
                print(f"  unknown --conf.{key} (use list or path, "
                      f"or --conf.KEY=VALUE)")
            continue
        print(f"  {set_config(cfg, key, val)}")
        if key in live:
            defaults[live[key]] = val
        if key == "outdir":
            state["outdir"] = val
            os.makedirs(val, exist_ok=True)
        if key == "hf_token" and val:
            os.environ["HF_TOKEN"] = val
            os.environ["HUGGING_FACE_HUB_TOKEN"] = val


# ─────────────────────────────── jobs ──────────────────────────────────
class Job:
    __slots__ = ("id", "prompt", "filename", "spec", "state", "pct")

    def __init__(self, jid, prompt, filename, spec):
        self.id = jid
        self.prompt = prompt
        self.filename = filename
        self.spec = spec
        self.state = "pending"
        self.pct = 0


def slice_prompt(p, width=22):
    p = p.replace("\n", " ")
    return p if len(p) <= width else p[:width - 1] + "…"


def render_jobs(jobs, lock):
    with lock:
        if not jobs:
            return
        print("  ── jobs ───────────────────────────────────────────")
        for j in jobs:
            if j.state == "running":
                st = f"{j.pct:3d}%"
            elif j.state == "done":
                st = "done"
            elif j.state == "error":
                st = "error"
            else:
                st = "pending"
            print(f"  [#{j.id:<3} | {slice_prompt(j.prompt):<22} | "
                  f"{j.filename:<24} | {st}]")
        print("  ───────────────────────────────────────────────────")


def worker_loop(engine, jobq, jobs, lock, sched_state):
    """Background thread: pull jobs, generate, update state in place.

    Reads the live pipeline from `engine` each job so a --reload swap is
    picked up automatically (reload only happens once the queue is drained).
    """
    while True:
        job = jobq.get()
        if job is None:
            jobq.task_done()
            return
        try:
            with lock:
                job.state = "running"
            spec = job.spec
            pipe = engine["pipe"]
            base_sched_config = engine["base_sched_config"]

            if spec["scheduler"] != sched_state["current"]:
                pipe.scheduler = make_scheduler(spec["scheduler"],
                                                base_sched_config)
                sched_state["current"] = spec["scheduler"]

            _apply_slicing(pipe, f"{spec['w']}x{spec['h']}", 1)  # (5)(6)

            steps = spec["steps"]

            def cb(pipe, step, ts, kw):
                with lock:
                    job.pct = int((step + 1) / steps * 100)
                return kw

            g = torch.Generator(device="mps").manual_seed(spec["seed"])
            img = pipe(
                prompt=spec["prompt"],
                negative_prompt=spec["neg"],
                num_inference_steps=steps,
                guidance_scale=spec["guidance"],
                width=spec["w"], height=spec["h"],
                generator=g,
                callback_on_step_end=cb,
            ).images[0]

            # upscale / refine (same as foreground path)
            if spec.get("upscale"):
                try:
                    tw, th, mode, refine = parse_upscale(
                        spec["upscale"], spec["w"], spec["h"])
                    img = upscale_image(img, tw, th, mode)
                    if refine:
                        img = refine_image(
                            engine["refiner"], img, spec["prompt"], spec["neg"],
                            spec["steps"], spec["guidance"])
                except ValueError as e:
                    print(f"\n  job #{job.id} upscale skipped: {e}")

            img.save(spec["outpath"])
            with lock:
                job.pct = 100
                job.state = "done"
        except Exception as e:
            with lock:
                job.state = "error"
            print(f"\n  job #{job.id} failed: {e}")
        finally:
            jobq.task_done()


def resolve_spec(params, prompt, outdir):
    """Validate params -> generation spec dict. Raises ValueError on bad input."""
    w, h = (int(x) for x in str(params["size"]).lower().split("x"))
    seed = make_seed(params.get("seed"))
    steps = int(params["steps"])
    guidance = float(params["guidance"])
    sketch = bool(params["sketchmode"])
    if sketch:
        steps = min(steps, 12)
        guidance = min(guidance, 3.0)
    if params["scheduler"].lower() not in SCHEDULERS:
        raise ValueError(f"unknown scheduler '{params['scheduler']}'")
    base = params["filename"]
    if base:
        stem = base[:-4] if base.endswith(".png") else base
        fname = f"{stem}.png"
    else:
        fname = f"{int(time.time()*1000)}_{seed}.png"
    return {
        "prompt": prompt, "neg": params["neg"], "w": w, "h": h,
        "seed": seed, "steps": steps, "guidance": guidance,
        "scheduler": params["scheduler"].lower(),
        "filename": fname, "outpath": os.path.join(outdir, fname),
        "upscale": params.get("upscale"),
    }

import secrets

SEED_MAX = 2**32 - 1  # 4294967295

def make_seed(value=None):
    """Return a usable seed. If value is None/'random'/'rand', generate one;
    otherwise coerce the provided value into the valid range."""
    if value is None or str(value).lower() in ("random", "rand", "-1"):
        return secrets.randbelow(SEED_MAX + 1)
    return int(value) % (SEED_MAX + 1)

# ─────────────────────────────── main ──────────────────────────────────
def main():
    cfg = load_config()
    if cfg.get("hf_token"):
        os.environ["HF_TOKEN"] = cfg["hf_token"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg["hf_token"]

    state = {"outdir": cfg["outdir"]}
    os.makedirs(state["outdir"], exist_ok=True)

    pipe, refiner, base_sched_config, current_sched = build_pipeline(cfg)
    print("type a prompt, --help for manual, Ctrl+C to exit\n")

    defaults = {"size": cfg["default_size"], "seed": None,
                "guidance": cfg["guidance"], "steps": cfg["steps"],
                "scheduler": cfg["scheduler"], "q": "1", "sketchmode": False,
                "neg": DEFAULT_NEG, "filename": None, "upscale": None}

    # shared live engine (so --reload can swap the pipeline under the closures)
    engine = {"pipe": pipe, "refiner": refiner, "base_sched_config": base_sched_config}
    sched_state = {"current": current_sched}

    jobs = []
    lock = threading.Lock()
    jobq = queue.Queue()
    jobs_mode = False
    worker = None
    id_counter = itertools.count(1)

    def ensure_worker():
        nonlocal worker
        if worker is None:
            worker = threading.Thread(
                target=worker_loop,
                args=(engine, jobq, jobs, lock, sched_state),
                daemon=True)
            worker.start()

    def run_foreground(params, prompt):
        p = engine["pipe"]
        bsc = engine["base_sched_config"]
        try:
            q = max(1, int(params["q"]))
        except ValueError:
            print("  bad --q"); return
        try:
            spec0 = resolve_spec(params, prompt, state["outdir"])
        except ValueError as e:
            print(f"  {e}"); return

        if spec0["scheduler"] != sched_state["current"]:
            p.scheduler = make_scheduler(spec0["scheduler"], bsc)
            sched_state["current"] = spec0["scheduler"]
        
        _apply_slicing(p, f"{spec0['w']}x{spec0['h']}", q)   # (5)(6)

        print(f"  prompt: {slice_prompt(prompt, 60)}")
        print(f"  {spec0['w']}x{spec0['h']}  seed={spec0['seed']}  "
              f"steps={spec0['steps']}  cfg={spec0['guidance']}  "
              f"sched={sched_state['current']}  q={q}")
        for n in range(q):
            spec = dict(spec0)
            spec["seed"] = spec0["seed"] + n
            if params["filename"]:
                stem = params["filename"]
                stem = stem[:-4] if stem.endswith(".png") else stem
                spec["filename"] = f"{stem}.png" if q == 1 else f"{stem}_{n+1}.png"
            else:
                spec["filename"] = f"{int(time.time()*1000)}_{spec['seed']}.png"
            spec["outpath"] = os.path.join(state["outdir"], spec["filename"])
            steps = spec["steps"]

            def cb(pipe, step, ts, kw):
                pct = int((step + 1) / steps * 100)
                tag = f"{n+1}/{q}" if q > 1 else ""
                print(f"\r  generating {tag} {pct:3d}%  [{step+1}/{steps}]",
                      end="", flush=True)
                return kw

            g = torch.Generator(device="mps").manual_seed(spec["seed"])
            t = time.time()
            img = p(prompt=spec["prompt"], negative_prompt=spec["neg"],
                    num_inference_steps=steps, guidance_scale=spec["guidance"],
                    width=spec["w"], height=spec["h"], generator=g,
                    callback_on_step_end=cb).images[0]
            
            if spec.get("upscale"):
                try:
                    tw, th, mode, refine = parse_upscale(
                        spec["upscale"], spec["w"], spec["h"])
                    img = upscale_image(img, tw, th, mode)
                    if refine:
                        img = refine_image(
                            engine["refiner"], img, spec["prompt"], spec["neg"],
                            spec["steps"], spec["guidance"])
                except ValueError as e:
                    print(f"  upscale skipped: {e}")
            
            img.save(spec["outpath"])
            print(f"\r  saved {spec['outpath']}  ({time.time()-t:.1f}s){' '*12}")
        print()

    def enqueue_jobs(params, prompt):
        try:
            q = max(1, int(params["q"]))
            spec0 = resolve_spec(params, prompt, state["outdir"])
        except ValueError as e:
            print(f"  {e}"); return
        for n in range(q):
            spec = dict(spec0)
            spec["seed"] = spec0["seed"] + n
            if params["filename"]:
                stem = params["filename"]
                stem = stem[:-4] if stem.endswith(".png") else stem
                spec["filename"] = f"{stem}.png" if q == 1 else f"{stem}_{n+1}.png"
            else:
                spec["filename"] = f"{int(time.time()*1000)}_{spec['seed']}.png"
            spec["outpath"] = os.path.join(state["outdir"], spec["filename"])
            jid = next(id_counter)
            job = Job(jid, prompt, spec["filename"], spec)
            with lock:
                jobs.append(job)
            jobq.put(job)
        print(f"  queued {q} job(s)")

    def do_reload():
        with lock:
            active = [j for j in jobs if j.state in ("pending", "running")]
        if active:
            print(f"  waiting for {len(active)} active/queued job(s) "
                  f"before reload ...")
            jobq.join()
        cfg.clear(); cfg.update(load_config())
        if cfg.get("hf_token"):
            os.environ["HF_TOKEN"] = cfg["hf_token"]
            os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg["hf_token"]
        old = engine["pipe"]
        engine["pipe"] = None
        del old
        gc.collect()
        if hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        new_pipe, new_refiner, new_bsc, new_sched = build_pipeline(cfg)
        engine["pipe"] = new_pipe
        engine["refiner"] = new_refiner
        engine["base_sched_config"] = new_bsc
        sched_state["current"] = new_sched
        state["outdir"] = cfg["outdir"]
        os.makedirs(state["outdir"], exist_ok=True)
        defaults.update({
            "size": cfg["default_size"], "seed": None,
            "guidance": cfg["guidance"], "steps": cfg["steps"],
            "scheduler": cfg["scheduler"]})
        print("  reload complete.")

    # ---- interactive loop ----
    while True:
        if jobs_mode:
            render_jobs(jobs, lock)
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nexiting — waiting for active job to finish ...")
            if worker is not None:
                jobq.put(None)
                jobq.join()
            break
        if not line:
            continue
        if line in ("--help", "--h"):
            print(HELP_TEXT)
            continue

        params, conf_ops, control, prompt = parse_inline(line, defaults)

        handled_control = False
        for tok in control:
            if tok == "--jobs":
                jobs_mode = not jobs_mode
                if jobs_mode:
                    ensure_worker()
                    print("  jobs mode ON — prompts now run in the background.")
                else:
                    print("  jobs mode OFF — prompts run in the foreground.")
                handled_control = True
            elif tok == "--jobs.clear":
                with lock:
                    before = len(jobs)
                    jobs[:] = [j for j in jobs if j.state in ("pending", "running")]
                print(f"  cleared {before - len(jobs)} finished job(s)")
                handled_control = True
            elif tok == "--reload":
                do_reload()
                handled_control = True

        if conf_ops:
            apply_conf_ops(cfg, conf_ops, defaults, state)

        if not prompt:
            if not (handled_control or conf_ops):
                defaults.update({k: params[k] for k in defaults if k in params})
                print("  (params only, no prompt — defaults updated)")
            continue

        if jobs_mode:
            enqueue_jobs(params, prompt)
        else:
            run_foreground(params, prompt)

        defaults.update({
            "size": params["size"], "seed": None,
            "guidance": params["guidance"], "steps": params["steps"],
            "scheduler": params["scheduler"], "q": params["q"],
            "sketchmode": False, "neg": params["neg"], "filename": None,
            "upscale": None,  # per-shot; don't carry forward by default
        })


if __name__ == "__main__":
    main()