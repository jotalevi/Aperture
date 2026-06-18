#!/usr/bin/env python3
"""Interactive photoreal SDXL shell for Apple Silicon (MPS) — self-bootstrapping.

First run creates a venv, installs deps, and writes ./configs/config.txt:
    git clone <repo>; cd <repo>; python main.py

Then type prompts at the >>> shell. Type --help for the manual. Ctrl+C exits.
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


_ensure_env()

# ─────────────────── heavy imports (only inside the venv) ───────────────────
import time, re
import torch
from diffusers import (
    StableDiffusionXLPipeline,
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
    """Read ./configs/config.txt (key=value lines); create it on first run."""
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
    lines = ["# SDXL shell config — plain key=value, one per line"]
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
Interactive SDXL shell — commands & inline params
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

Config (persisted to ./configs/config.txt):
  --conf.KEY=VALUE  set a config value, e.g. --conf.hf_token=hf_xxxx
                    keys: hf_token, checkpoint, outdir, seed, steps,
                          scheduler, guidance, default_size
  --conf.list       show current config (token masked)
  --conf.path       print config file location
  (config changes apply to saved settings; some need a restart to reload,
   e.g. checkpoint. seed/steps/scheduler/etc. also update live defaults.)

Shell commands:
  --help, --h       show this manual
  (empty line)      ignored
  Ctrl+C            exit

Examples:
  >>> --conf.hf_token=hf_xxxxxxxxxxxx
  >>> --conf.list
  >>> woman by a cafe window, 35mm, natural skin
  >>> --q=4 --size=1024x1024 candid street photo, overcast light
  >>> --sketchmode --q=6 testing compositions
  >>> --scheduler=eulera --steps=40 --filename=final close-up, freckles
"""


def make_scheduler(name, base_config):
    name = name.lower()
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler '{name}', "
                         f"choose from {', '.join(SCHEDULERS)}")
    cls, kw = SCHEDULERS[name]
    return cls.from_config(base_config, **kw)


def parse_inline(line, defaults):
    """Pull leading --key=value, --conf.* and bare --flag tokens off the line.

    Returns (params, conf_ops, prompt). conf_ops is a list of (key, value|None)
    for config commands; value None means a query command (list/path).
    """
    params = dict(defaults)
    conf_ops = []
    tokens = line.split()
    i = 0
    while i < len(tokens) and tokens[i].startswith("--"):
        tok = tokens[i]
        if tok.startswith("--conf."):
            rest = tok[len("--conf."):]
            if "=" in rest:
                k, v = rest.split("=", 1)
                conf_ops.append((k.strip(), v))
            else:
                conf_ops.append((rest.strip(), None))  # list / path
            i += 1
            continue
        m = re.match(r"--([\w]+)=(.+)", tok)
        if m:
            params[m.group(1)] = m.group(2)
        else:
            params[tok[2:]] = True  # bare boolean flag
        i += 1
    prompt = " ".join(tokens[i:]).strip()
    return params, conf_ops, prompt


def apply_conf_ops(cfg, conf_ops, defaults):
    """Handle --conf.* operations; mirror relevant keys into live defaults."""
    # which config keys map onto live shell defaults
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
        msg = set_config(cfg, key, val)
        print(f"  {msg}")
        # reflect into live defaults so it takes effect this session
        if key in live:
            defaults[live[key]] = val
        if key == "hf_token" and val:
            os.environ["HF_TOKEN"] = val
            os.environ["HUGGING_FACE_HUB_TOKEN"] = val


# ─────────────────────────────── main ──────────────────────────────────
def main():
    cfg = load_config()

    if cfg.get("hf_token"):
        os.environ["HF_TOKEN"] = cfg["hf_token"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg["hf_token"]

    checkpoint = cfg["checkpoint"]
    outdir = cfg["outdir"]
    os.makedirs(outdir, exist_ok=True)

    # ---- load once ----
    print(f"loading {checkpoint} ...")
    t0 = time.time()
    if str(checkpoint).endswith(".safetensors"):
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            checkpoint, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    base_sched_config = pipe.scheduler.config
    pipe.scheduler = make_scheduler(cfg["scheduler"], base_sched_config)
    print(f"ready in {time.time()-t0:.1f}s — type a prompt, --help for manual, "
          f"Ctrl+C to exit\n")

    defaults = {"size": cfg["default_size"], "seed": cfg["seed"],
                "guidance": cfg["guidance"], "steps": cfg["steps"],
                "scheduler": cfg["scheduler"], "q": "1", "sketchmode": False,
                "neg": DEFAULT_NEG, "filename": None}
    current_sched = cfg["scheduler"]

    # ---- interactive loop ----
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nexiting.")
            break
        if not line:
            continue
        if line in ("--help", "--h"):
            print(HELP_TEXT)
            continue

        params, conf_ops, prompt = parse_inline(line, defaults)

        if conf_ops:
            apply_conf_ops(cfg, conf_ops, defaults)
            # outdir change should take effect for new generations
            if any(k == "outdir" for k, _ in conf_ops):
                outdir = cfg["outdir"]
                os.makedirs(outdir, exist_ok=True)
            if not prompt:
                continue  # config-only line

        if not prompt:
            print("  (params only, no prompt — defaults updated)")
            defaults.update({k: params[k] for k in defaults if k in params})
            continue

        try:
            w, h = (int(x) for x in str(params["size"]).lower().split("x"))
        except ValueError:
            print(f"  bad --size={params['size']}, expected WxH")
            continue
        try:
            seed_v = int(params["seed"])
            steps = int(params["steps"])
            guidance = float(params["guidance"])
            q = max(1, int(params["q"]))
        except ValueError as e:
            print(f"  bad numeric param: {e}")
            continue

        sketch = bool(params["sketchmode"])
        neg = params["neg"]
        if sketch:
            steps = min(steps, 12)
            guidance = min(guidance, 3.0)

        if params["scheduler"] != current_sched:
            try:
                pipe.scheduler = make_scheduler(params["scheduler"],
                                                base_sched_config)
                current_sched = params["scheduler"]
            except ValueError as e:
                print(f"  {e}")
                continue

        base_name = params["filename"]
        print(f"  prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"  {w}x{h}  seed={seed_v}  steps={steps}  cfg={guidance}  "
              f"sched={current_sched}  q={q}{'  [SKETCH]' if sketch else ''}")

        for n in range(q):
            shot_seed = seed_v + n
            g = torch.Generator(device="mps").manual_seed(shot_seed)

            def cb(pipe, step, ts, kw):
                pct = int((step + 1) / steps * 100)
                tag = f"{n+1}/{q}" if q > 1 else ""
                print(f"\r  generating {tag} {pct:3d}%  [{step+1}/{steps}]",
                      end="", flush=True)
                return kw

            if base_name:
                stem = base_name[:-4] if base_name.endswith(".png") else base_name
                fname = f"{stem}.png" if q == 1 else f"{stem}_{n+1}.png"
            else:
                fname = f"{int(time.time())}_{shot_seed}.png"
            outpath = os.path.join(outdir, fname)

            t0 = time.time()
            img = pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=w, height=h,
                generator=g,
                callback_on_step_end=cb,
            ).images[0]
            img.save(outpath)
            print(f"\r  saved {outpath}  ({time.time()-t0:.1f}s){' '*12}")

        defaults.update({
            "size": f"{w}x{h}", "seed": str(seed_v), "guidance": str(guidance),
            "steps": str(steps), "scheduler": current_sched, "q": str(q),
            "sketchmode": False, "neg": neg, "filename": None,
        })
        print()


if __name__ == "__main__":
    main()