"""Standalone helpger utility to remove sky regions from an image using SAM."""

import argparse
import logging
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from depthpro_runner.sam_sky import SamSkyRemover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove sky from an image using SAM.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the sky-removed image. Defaults to <prefix>_sky_removed.png next to input.",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        help="Optional path to save the binary sky mask as a PNG.",
    )
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--cache-dir", type=Path, default=Path("./models/sam"))
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device to run SAM on (e.g. cuda, cpu). Defaults to auto-detect.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def save_mask(mask: np.ndarray, path: Path) -> None:
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.image.exists():
        raise SystemExit(f"Input image {args.image} does not exist")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    remover = SamSkyRemover(cache_dir=args.cache_dir, model_type=args.model_type, device=device)

    with Image.open(args.image) as img:
        rgb = img.convert("RGB")
    rgb_array = np.array(rgb, dtype=np.uint8)

    result = remover.remove_sky(rgb_array)
    output_path: Path = args.output or args.image.with_name(f"{args.image.stem}_sky_removed.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.masked_image).save(output_path)
    logging.info("Saved sky-removed image to %s", output_path)

    if result.mask is not None and args.mask_output:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        save_mask(result.mask, args.mask_output)
        logging.info("Saved sky mask to %s", args.mask_output)


if __name__ == "__main__":
    main()
