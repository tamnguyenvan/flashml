# FlashML

Unified FastAPI service for several inference stacks, without changing their algorithms:

| Route | Model | Input | Output |
| --- | --- | --- | --- |
| `POST /reconstruct` | MoGe-3 | multipart image (`file`) | ZIP (`point_map.npy`, `metadata.json`, optional `output.glb` / debug PNGs) |
| `POST /interactive-segment` | SimpleClick | JSON image + clicks | PNG mask (base64) |
| `POST /segment` | OneFormer ADE20K | JSON image | wall / floor / rug PNG masks |
| `POST /remove` | LaMa | multipart image (`file`) + mask (`mask`) | inpainted PNG |

`POST /predict` is kept as an alias of `/reconstruct` for the existing DreamRoom MoGe client.

The models cannot share one Python environment (different PyTorch / MMCV / Transformers stacks). On Vast.ai this repo therefore runs:

1. Four GPU workers (ports 8001–8004), each in its own Miniconda env
2. One public FastAPI gateway on port **8000** that validates requests and proxies to those workers
3. **Supervisor** to keep the processes alive

Routers stay HTTP-only. Inference lives in `src/flashml/services/`.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the API](#run-the-api)
  - [Run the tests](#run-the-tests)
  - [Launch GPU workers](#launch-gpu-workers)
- [Request contracts](#request-contracts)
  - [Reconstruct](#reconstruct)
  - [Interactive segment](#interactive-segment)
  - [Segment](#segment)
  - [Remove](#remove)
- [Health](#health)
- [Deploying on Vast.ai](#deploying-on-vast-ai)
  - [Vast.ai setup](#vast-ai-setup)
  - [GPU / memory notes](#gpu--memory-notes)

## Requirements

- Python **>= 3.10**
- For GPU inference: an NVIDIA GPU (CUDA). The gateway itself is lightweight and HTTP-only; each model runs in its own Python environment (see [Launch GPU workers](#launch-gpu-workers)).

## Installation

Clone the repository:

```bash
git clone <this-repo> flashml
cd flashml
```

Create a local environment file from the template and adjust as needed:

```bash
cp .env.example .env
```

Install the package (editable) with its dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

For on-instance GPU deployments, `scripts/setup_conda.sh` installs Miniconda if needed, creates five environments (`flashml-api`, `flashml-moge`, `flashml-simpleclick`, `flashml-oneformer`, `flashml-lama`), clones MoGe / SimpleClick, and downloads model weights. The first run is long. See [Deploying on Vast.ai](#deploying-on-vast-ai).

## Usage

### Run the API

From the repo root, the gateway serves on `FLASHML_HOST`/`FLASHML_PORT` (defaults `0.0.0.0:8000`) and exposes interactive docs at `/docs`:

```bash
flashml
# or, with explicit host/port
flashml --host 0.0.0.0 --port 8000
# or via uvicorn
uvicorn flashml.app:app --host 0.0.0.0 --port 8000
```

Configuration is read from `FLASHML_*` environment variables (see `.env.example`) and loaded automatically from a `.env` file when present. Set `FLASHML_ENABLED_ROUTES` to restrict which routes load (`all`, `reconstruct`, `interactive-segment`, `segment`, or `remove`).

### Run the tests

```bash
python -m pytest
```

### Launch GPU workers

After `scripts/setup_conda.sh` has created the environments, start the gateway plus the four workers under Supervisor:

```bash
FLASHML_HOME=$PWD CONDA_ROOT=$HOME/miniconda3 ./scripts/start.sh
```

Supervisor keeps all processes alive. `scripts/vastai_onstart.sh` automates the full Vast.ai boot sequence (env setup, cloning, and start).

## Request contracts

These match the previous Modal endpoints.

### Reconstruct

`POST /reconstruct` — `multipart/form-data`

- `file` (required)
- `include_mesh` default `true`
- `include_debug` default `true`
- `max_size` 64–2048, default `800`
- `resolution_level` 0–9, default `9`
- `num_tokens` optional 1200–3600
- `refine_steps` 0–8, default `3`
- `fov_x` optional
- `edge_threshold` default `0.04`

### Interactive segment

`POST /interactive-segment` — JSON

```json
{
  "image": "<base64 or data URL>",
  "positive_points": [[x, y]],
  "negative_points": [],
  "threshold": 0.49
}
```

### Segment

`POST /segment` — JSON

```json
{ "image": "data:image/png;base64,..." }
```

### Remove

`POST /remove` — `multipart/form-data`

- `file` (required) — RGB image (PNG or JPEG) containing the object to remove
- `mask` (required) — binary mask PNG (white = the region to inpaint away)
- `max_size` 64–4096, default `1024` (longest-side limit; larger images are downscaled)

Returns the inpainted result as `image/png`.

Errors are JSON: `{ "error", "code", "request_id", "details?" }` with `X-Request-ID` on every response.

### Health

`GET /health` is liveness. `GET /ready` is 200 only when enabled backends report ready (after model load).

## Deploying on Vast.ai

### Vast.ai setup

Use the CUDA 12.8.1 driver image, `vastai/base-image:cuda-12.8.1-auto`. It provides the NVIDIA driver; PyTorch wheels bring the CUDA runtime.

```bash
# on the instance
export FLASHML_HOME=/workspace/flashml
export CONDA_ROOT=/workspace/miniconda3
git clone <this-repo> "$FLASHML_HOME"
cp "$FLASHML_HOME/.env.example" "$FLASHML_HOME/.env"
bash "$FLASHML_HOME/scripts/vastai_onstart.sh"
```

Public URL: `http://<instance>:8000/docs`

Open port **8000** on the Vast.ai instance. Workers bind to `127.0.0.1` only.

### GPU / memory notes

VRAM: all checkpoints loaded at once wants a large GPU (MoGe-3 ViT-G + SimpleClick ViT-H + OneFormer Swin-L + LaMa). If you OOM, stop unused Supervisor programs, for example:

```bash
supervisorctl -c "$FLASHML_HOME/conf/supervisord.conf" stop flashml-oneformer
```
