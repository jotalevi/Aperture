# Aperture

A fast, local, terminal-native image generation REPL for Apple Silicon. Load your model once, iterate at the prompt, full control over every parameter.

Aperture runs SDXL entirely on your Mac via Metal/MPS — nothing leaves the machine. It keeps your checkpoint resident across the session, so after the initial load you generate at the speed of typing.

## Features

- **Local & private** — runs entirely on your Mac, no cloud
- **Load once, generate many** — model stays in memory across the whole session
- **Full parameter control** — size, seed, steps, scheduler, guidance, batch count, all inline
- **Photoreal defaults** — low-CFG + DPM++ Karras tuning out of the box
- **Self-bootstrapping** — first run creates the venv and installs dependencies
- **Plain-text config** — token and defaults in a simple local file

## Requirements

- Apple Silicon Mac (M1 or newer)
- Python 3.12 recommended (3.13 has known issues with the ML stack)
- ~10 GB free disk for dependencies + a checkpoint

## Install & run

```bash
git clone <repo-url>
cd aperture
python main.py
```

First run creates `.venv`, installs dependencies, and writes `configs/config.txt`, then drops you into the shell. Subsequent runs launch straight in.

## Setup

Point Aperture at a photoreal checkpoint for best results — base SDXL looks synthetic. Good choices: **RealVisXL**, **Juggernaut XL**, **epiCRealism XL** (download the `.safetensors` from Hugging Face or Civitai).

From inside the shell:

```
>>> --conf.hf_token=hf_xxxxxxxxxxxxxxxx
>>> --conf.checkpoint=RealVisXL_V5.0.safetensors
>>> --conf.outdir=./output
>>> --conf.list
```

`checkpoint` changes take effect on next launch (the model loads at startup).

## Usage

Type a prompt and press Enter. Inline params go at the **start** of the line:

```
>>> woman by a cafe window, 35mm, natural skin texture, soft window light
>>> --q=4 --size=1024x1024 candid street photo, overcast light
>>> --sketchmode --q=6 testing compositions quickly
>>> --scheduler=eulera --steps=40 --filename=final close-up portrait, freckles
```

### Generation params

| Param | Default | Description |
|-------|---------|-------------|
| `--size=WxH` | `832x1216` | Output dimensions |
| `--seed=N` | from config | Base seed (increments per image in a batch) |
| `--steps=N` | `30` | Denoise steps |
| `--scheduler=NAME` | `dpmpp2m` | `dpmpp2m`, `dpmppsde`, `euler`, `eulera`, `unipc` |
| `--guidance=F` | `5.0` | CFG scale (lower = more natural) |
| `--q=N` | `1` | How many images to generate |
| `--filename=NAME` | timestamp | Output name (`_1.._N` suffix when `q>1`) |
| `--neg="..."` | built-in | Per-shot negative prompt override |
| `--sketchmode` | off | Fast draft: caps steps@12, cfg@3.0 |

Params persist as sticky defaults — you only specify what changes between shots.

### Config commands

| Command | Description |
|---------|-------------|
| `--conf.KEY=VALUE` | Set & persist a config value |
| `--conf.list` | Show current config (token masked) |
| `--conf.path` | Print config file location |

Keys: `hf_token`, `checkpoint`, `outdir`, `seed`, `steps`, `scheduler`, `guidance`, `default_size`

### Shell commands

| Command | Description |
|---------|-------------|
| `--help` / `--h` | Show the manual |
| `Ctrl+C` | Exit |

## Tips for realism

- **Checkpoint matters most** — a photoreal fine-tune beats any amount of param tuning on base SDXL.
- **Keep CFG low** (4–5.5). High guidance creates the oversaturated "AI sheen."
- **Prompt like a photographer** — describe lens, lighting, and film stock rather than "beautiful."
- **Generate at native resolution** (832×1216 for portraits, 1024×1024 square) and upscale after.
- **Use `--sketchmode`** to find a composition fast, then re-run without it for the final.

## Performance (Apple Silicon)

The model loads once and stays resident, so per-image cost is just the diffusion run. Attention and VAE slicing are enabled by default to stay off the swap cliff on 16 GB machines. Expect roughly 25–45 s per 1024-class image on a base M4, faster at lower step counts.

## Config & files

```
aperture/
├── main.py              # the REPL
├── requirements.txt
├── configs/config.txt   # local settings (gitignored — holds your token)
├── output/              # generated images (gitignored)
└── .venv/               # auto-created (gitignored)
```

**Note:** `configs/config.txt` stores your HF token in plain text. It's gitignored by default — keep it that way, and never commit it.

## License

MIT