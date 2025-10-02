# Depth Pro Inference Toolkit

This project uses a monocular depth model in a modular Python toolkit. It can download the checkpoint, run inference on single images or batches, and export aligned RGB-D assets including:

- Metric depth maps (`*.png` and `*.npy`)
- Dense RGB point clouds (`*.ply` in ASCII format)

All outputs are generated per input image and stored under an output directory you choose.

## Requirements

- Python 3.10+
- Sufficient disk space for the model checkpoint (~1.9 GB)
- CPU-only inference is supported, but a CUDA-enabled GPU significantly speeds things up (optional).

Install the dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements include the official Depth Pro package directly from Apple’s GitHub repository.

## Running inference

### Single image

```bash
source .venv/bin/activate
python scripts/run_depth.py \
  --image sample_data/test_image.png \
  --output-dir outputs
```

### Batch mode

```bash
source .venv/bin/activate
python scripts/run_depth.py \
  --input-dir path/to/images \
  --batch-size 2 \
  --device cuda:0 \
  --output-dir outputs
```

CLI options:

| Flag | Description |
| --- | --- |
| `-i, --image` | Individual image path (can be repeated) |
| `-d, --input-dir` | Folder to scan recursively for supported image files |
| `-o, --output-dir` | Destination folder for depth maps and point clouds |
| `--cache-dir` | Directory used to cache the downloaded checkpoint (`models/depth_pro` by default) |
| `--device` | Torch device spec (`cpu`, `cuda`, `cuda:0`, …). Defaults to auto-detect. |
| `--precision` | `float32` (default) or `float16` (requires CUDA) |
| `--batch-size` | Number of images to enqueue per batch (processed sequentially today) |
| `--log-level` | Logging verbosity (default `INFO`) |

## Outputs

For each input image `image_name.ext` the toolkit writes:

- `image_name_depth.png` — 16-bit normalized depth visualization (near = bright)
- `image_name_depth_heatmap.png` — Color depth map with a Viridis heat map and legend showing metres
- `image_name_depth.npy` — Raw metric depth array in metres
- `image_name_pointcloud.ply` — ASCII PLY file with XYZRGB values (centre principal point, model-estimated focal length)

All files share the same base name and live in the specified output directory.

## Programmatic usage

```python
from depthpro_runner import DepthProInference

runner = DepthProInference(cache_dir="./models/depth_pro", device="cpu")
result = runner.predict_single("path/to/image.jpg", "./outputs")
print(result.depth_map_path)
```

The returned `InferenceResult` dataclass includes paths to each artifact and the focal length (in pixels) used for projection.

## Notes

- The first run downloads the checkpoint from Hugging Face; subsequent runs reuse the cached copy.
- If EXIF metadata includes a 35mm-equivalent focal length, it is used; otherwise the model’s intrinsic predictor fills in the value.
- ASCII PLY files can be loaded in tools like CloudCompare, MeshLab, or Open3D for inspection.

----

## Author
**Shiven Saini**
