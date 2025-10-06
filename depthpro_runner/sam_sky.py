"""SAM-based sky segmentation and removal utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

SAM_MODEL_METADATA = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
    },
}


def ensure_sam_checkpoint(cache_dir: Path, model_type: str) -> Path:
    """Download the SAM checkpoint if it is not already cached locally."""

    if model_type not in SAM_MODEL_METADATA:
        raise ValueError(f"Unsupported SAM model type '{model_type}'.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = SAM_MODEL_METADATA[model_type]
    checkpoint_path = cache_dir / metadata["filename"]

    if checkpoint_path.exists():
        return checkpoint_path

    import urllib.request

    LOGGER.info("Downloading SAM checkpoint (%s)...", metadata["filename"])
    with urllib.request.urlopen(metadata["url"]) as response, checkpoint_path.open("wb") as output:
        output.write(response.read())
    LOGGER.info("SAM checkpoint saved to %s", checkpoint_path)
    return checkpoint_path


@dataclass
class SkyRemovalResult:
    """Result of the sky removal step."""

    masked_image: np.ndarray
    mask: Optional[np.ndarray]
    score: float


class SamSkyRemover:
    """Use Segment Anything to remove sky regions from images."""

    def __init__(
        self,
        cache_dir: Path | str = Path("./models/sam"),
        model_type: str = "vit_b",
        device: torch.device | str | None = None,
    ) -> None:
        cache_dir = Path(cache_dir)
        self.model_type = model_type
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = ensure_sam_checkpoint(cache_dir, model_type)

        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore

        LOGGER.info("Loading SAM (%s) from %s", model_type, checkpoint_path)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            box_nms_thresh=0.7,
        )

    def remove_sky(self, image: np.ndarray) -> SkyRemovalResult:
        """Remove sky regions from an RGB image using SAM."""

        if image.dtype != np.uint8:
            raise ValueError("Expected input image as uint8 array.")

        height, width = image.shape[:2]
        total_pixels = height * width

        masks = self.mask_generator.generate(image)
        if len(masks) == 0:
            return SkyRemovalResult(masked_image=image.copy(), mask=None, score=float("-inf"))

        best_score = float("-inf")
        best_mask: Optional[np.ndarray] = None

        for mask in masks:
            segmentation = mask["segmentation"]
            area = int(segmentation.sum())
            if area < 0.02 * total_pixels:
                continue

            x, y, w, h = mask["bbox"]
            y_center = (y + h / 2) / height
            if y_center > 0.65:
                continue

            mask_pixels = image[segmentation]
            mean_rgb = mask_pixels.mean(axis=0)
            total = float(mean_rgb.sum()) + 1e-6
            blue_ratio = float(mean_rgb[2]) / total
            brightness = float(total) / (3 * 255.0)
            vertical_score = 1.0 - y_center
            coverage = area / total_pixels

            score = 0.55 * vertical_score + 0.25 * blue_ratio + 0.1 * brightness + 0.1 * coverage

            if score > best_score:
                best_score = score
                best_mask = segmentation

        if best_mask is None:
            return SkyRemovalResult(masked_image=image.copy(), mask=None, score=float("-inf"))

        masked_image = image.copy()
        masked_image[best_mask] = 0

        return SkyRemovalResult(masked_image=masked_image, mask=best_mask.astype(bool), score=best_score)
