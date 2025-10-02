"""Utility helpers for image loading and saving."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ExifTags

# Mapping for orientation tag lookup once at import time.
_ORIENTATION_TAG = next(
    (key for key, value in ExifTags.TAGS.items() if value == "Orientation"),
    None,
)
_FOCAL_LENGTH_35MM_TAG = next(
    (key for key, value in ExifTags.TAGS.items() if value == "FocalLengthIn35mmFilm"),
    None,
)


def load_rgb_image(path: Path | str) -> Tuple[np.ndarray, Optional[float]]:
    """Load an RGB image and estimate focal length in pixels when metadata is present."""

    path = Path(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        exif = img.getexif()
        if _ORIENTATION_TAG in exif:
            img = ImageOps.exif_transpose(img)
        width, height = img.size
        focal_length_px = None
        if _FOCAL_LENGTH_35MM_TAG in exif:
            focal_35mm = exif.get(_FOCAL_LENGTH_35MM_TAG)
            if isinstance(focal_35mm, tuple):
                numerator, denominator = focal_35mm
                focal_35mm = numerator / max(denominator, 1)
            if isinstance(focal_35mm, (int, float)) and focal_35mm > 0:
                # Convert 35mm equivalent focal length to pixels.
                diagonal_px = math.hypot(width, height)
                diagonal_sensor_mm = math.hypot(36.0, 24.0)
                focal_length_px = focal_35mm * diagonal_px / diagonal_sensor_mm
    array = np.asarray(img).copy()
    return array, focal_length_px


def save_depth_map(depth: np.ndarray, output_path: Path) -> None:
    """Save depth map as 16-bit PNG preserving metric depth."""

    output_path = output_path.with_suffix(".png")
    depth_clipped = np.clip(depth, 0.0, np.percentile(depth, 99.9))
    max_val = float(depth_clipped.max())
    max_val = max(max_val, 1e-6)
    depth_normalized = depth_clipped / max_val
    depth_image = (depth_normalized * (2**16 - 1)).astype(np.uint16)
    Image.fromarray(depth_image).save(output_path)


def save_depth_heatmap(
    depth: np.ndarray,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    """Render a color heat map with legend showing depth in metres."""

    output_path = output_path.with_suffix(".png")
    depth = depth.astype(np.float32)
    depth_clipped = np.clip(depth, 0.0, np.percentile(depth, 99.9))
    vmin = float(depth_clipped.min())
    vmax = float(depth_clipped.max())
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    import matplotlib.pyplot as plt  # type: ignore[import]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(depth_clipped, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.05)
    cbar.set_label("Depth (m)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_depth_numpy(depth: np.ndarray, output_path: Path) -> None:
    """Persist the raw depth array as .npy file."""

    output_path = output_path.with_suffix(".npy")
    np.save(output_path, depth)


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    """Group an iterable into batches of size ``batch_size``."""

    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
