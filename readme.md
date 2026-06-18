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