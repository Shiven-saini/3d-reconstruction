"""Inference wrapper for Apple Depth Pro."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from .model import get_depth_pro_model
from .pointcloud import depth_to_point_cloud, save_ply
from .utils import (
    batched,
    load_rgb_image,
    save_depth_heatmap,
    save_depth_map,
    save_depth_numpy,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for the artifacts produced for a single image."""

    image_path: Path
    depth_map_path: Path
    depth_heatmap_path: Path
    depth_numpy_path: Path
    point_cloud_path: Path
    focal_length_px: float


class DepthProInference:
    """Convenience wrapper around the Depth Pro model."""

    def __init__(
        self,
        cache_dir: Path | str = Path("./models/depth_pro"),
        device: Optional[str | torch.device] = None,
        precision: str = "float16",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.device = torch.device(device) if device is not None else None
        self.precision = precision
        self.model, self.transform = get_depth_pro_model(
            cache_dir=self.cache_dir,
            device=self.device,
            precision=self.precision,
        )
        self.device = next(self.model.parameters()).device

    def predict_single(
        self,
        image_path: Path | str,
        output_dir: Path | str,
    ) -> InferenceResult:
        """Run inference on a single image."""

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb, f_px_from_meta = load_rgb_image(image_path)
        tensor = self.transform(rgb)
        tensor = tensor.to(self.device)

        f_px_tensor: Optional[torch.Tensor]
        if f_px_from_meta is not None:
            f_px_tensor = torch.tensor([f_px_from_meta], dtype=tensor.dtype, device=self.device)
        else:
            f_px_tensor = None

        with torch.no_grad():
            prediction = self.model.infer(tensor, f_px=f_px_tensor)
        depth = prediction["depth"].detach().cpu().float().numpy()
        focal_length_px = float(prediction["focallength_px"].detach().cpu().float().item())

        stem = image_path.stem
        depth_map_path = output_dir / f"{stem}_depth.png"
        depth_heatmap_path = output_dir / f"{stem}_depth_heatmap.png"
        depth_numpy_path = output_dir / f"{stem}_depth.npy"
        point_cloud_path = output_dir / f"{stem}_pointcloud.ply"

        save_depth_map(depth, depth_map_path)
        save_depth_heatmap(depth, depth_heatmap_path)
        save_depth_numpy(depth, depth_numpy_path)

        points, colors = depth_to_point_cloud(depth, rgb, focal_length_px)
        save_ply(points, colors, point_cloud_path)

        LOGGER.info("Processed %s -> depth map: %s", image_path, depth_map_path.name)

        return InferenceResult(
            image_path=image_path,
            depth_map_path=depth_map_path,
            depth_heatmap_path=depth_heatmap_path,
            depth_numpy_path=depth_numpy_path,
            point_cloud_path=point_cloud_path,
            focal_length_px=focal_length_px,
        )

    def predict_batch(
        self,
        image_paths: Iterable[Path | str],
        output_dir: Path | str,
        batch_size: int = 1,
    ) -> List[InferenceResult]:
        """Run inference for multiple images.

        Currently processes images sequentially but exposes a batching interface
        for potential future optimisations.
        """

        output_dir = Path(output_dir)
        results: List[InferenceResult] = []
        paths = [Path(p) for p in image_paths]
        for group in batched(paths, max(batch_size, 1)):
            for image_path in group:
                results.append(self.predict_single(image_path, output_dir))
        return results
