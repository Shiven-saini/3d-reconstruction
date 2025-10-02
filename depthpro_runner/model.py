"""Model download and loading helpers for Apple Depth Pro."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import torch
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:  # pragma: no cover
    from depth_pro import create_model_and_transforms  # type: ignore
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # type: ignore

LOGGER = logging.getLogger(__name__)

MODEL_REPO_ID = "apple/DepthPro"
MODEL_FILENAME = "depth_pro.pt"


def ensure_checkpoint(local_dir: Path) -> Path:
    """Ensure that the Depth Pro checkpoint is available locally.

    Args:
        local_dir: Directory where the model weights should live.

    Returns:
        Path to the downloaded checkpoint.
    """

    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = local_dir / MODEL_FILENAME
    if checkpoint_path.exists():
        LOGGER.info("Using cached Depth Pro checkpoint at %s", checkpoint_path)
        return checkpoint_path

    LOGGER.info("Downloading Depth Pro checkpoint from Hugging Face (%s)...", MODEL_REPO_ID)
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    LOGGER.info("Checkpoint downloaded to %s", downloaded_path)
    return Path(downloaded_path)


def get_depth_pro_model(
    cache_dir: Path | str,
    device: torch.device | str | None = None,
    precision: str = "float32",
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the Depth Pro model and preprocessing transforms.

    Args:
        cache_dir: Directory used to cache the checkpoint.
        device: Torch device or string alias (e.g. "cuda", "cpu"). Defaults to CUDA if available.
        precision: Either "float32" or "float16". Float16 requires CUDA.

    Returns:
        Tuple of (model, transform) ready for inference.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available. Specify device='cpu' or install GPU drivers.")

    if precision not in {"float32", "float16"}:
        raise ValueError("precision must be 'float32' or 'float16'")
    dtype = torch.float16 if precision == "float16" else torch.float32
    if dtype == torch.float16 and device.type == "cpu":
        raise ValueError("float16 precision requires a CUDA-enabled device")

    checkpoint_path = ensure_checkpoint(Path(cache_dir))

    # Import lazily so users who have not installed the dependency yet get a clearer error.
    from depth_pro import create_model_and_transforms  # type: ignore
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # type: ignore

    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint_path))

    model, transform = create_model_and_transforms(
        config=config,
        device=device,
        precision=dtype,
    )
    model.eval()

    return model, transform
