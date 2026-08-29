# FlashML

Unified FastAPI service for three existing inference stacks, without changing their algorithms:

| Route | Model | Input | Output |
| --- | --- | --- | --- |
| `POST /reconstruct` | MoGe-3 | multipart image (`file`) | ZIP (`point_map.npy`, `metadata.json`, optional `output.glb` / debug PNGs) |
| `POST /interactive-segment` | SimpleClick | JSON image + clicks | PNG mask (base64) |
| `POST /segment` | OneFormer ADE20K | JSON image | wall / floor / rug PNG masks |

`POST /predict` is kept as an alias of `/reconstruct` for the existing DreamRoom MoGe client.

The three models cannot share one Python environment (different PyTorch / MMCV / Transformers stacks). On Vast.ai this repo therefore runs:

1. Three GPU workers (ports 8001–8003), each in its own Miniconda env
2. One public FastAPI gateway on port **8000** that validates requests and proxies to those workers
3. **Supervisor** to keep the processes alive

Routers stay HTTP-only. Inference lives in `src/flashml/services/`.

## Vast.ai (CUDA 12.8.1 driver image)

Use `vastai/base-image:cuda-12.8.1-auto`. That image provides the NVIDIA driver; PyTorch wheels bring the CUDA runtime.

```bash
# on the instance
export FLASHML_HOME=/workspace/flashml
export CONDA_ROOT=/workspace/miniconda3
git clone <this-repo> "$FLASHML_HOME"
cp "$FLASHML_HOME/.env.example" "$FLASHML_HOME/.env"
bash "$FLASHML_HOME/scripts/vastai_onstart.sh"
```

`scripts/setup_conda.sh` installs Miniconda if needed, creates four envs (`flashml-api`, `flashml-moge`, `flashml-simpleclick`, `flashml-oneformer`), clones MoGe / SimpleClick, and downloads weights. First run is long.

Public URL: `http://<instance>:8000/docs`

Open port **8000** on the Vast.ai instance. Workers bind to `127.0.0.1` only.

VRAM: all three checkpoints loaded at once wants a large GPU (MoGe-3 ViT-G + SimpleClick ViT-H + OneFormer Swin-L). If you OOM, stop unused Supervisor programs, for example:

```bash
supervisorctl -c "$FLASHML_HOME/conf/supervisord.conf" stop flashml-oneformer
```

## Local commands

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

GPU workers (after `scripts/setup_conda.sh`):

```bash
FLASHML_HOME=$PWD CONDA_ROOT=$HOME/miniconda3 ./scripts/start.sh
```

## Request contracts

These match the previous Modal endpoints.

**Reconstruct** — `multipart/form-data`

- `file` (required)
- `include_mesh` default `true`
- `include_debug` default `true`
- `max_size` 64–2048, default `800`
- `resolution_level` 0–9, default `9`
- `num_tokens` optional 1200–3600
- `refine_steps` 0–8, default `3`
- `fov_x` optional
- `edge_threshold` default `0.04`

**Interactive segment** — JSON

```json
{
  "image": "<base64 or data URL>",
  "positive_points": [[x, y]],
  "negative_points": [],
  "threshold": 0.49
}
```

**Segment** — JSON

```json
{ "image": "data:image/png;base64,..." }
```

Errors are JSON: `{ "error", "code", "request_id", "details?" }` with `X-Request-ID` on every response.

`GET /health` is liveness. `GET /ready` is 200 only when enabled backends report ready (after model load).
