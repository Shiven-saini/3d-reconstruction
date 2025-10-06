#!/usr/bin/env python3
"""CLI entry-point for running Apple Depth Pro inference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from depthpro_runner.inference import DepthProInference

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(images: List[str] | None, input_dir: Path | None) -> List[Path]:
    paths: List[Path] = []
    if images:
        paths.extend(Path(p) for p in images)
    if input_dir:
        for extension in SUPPORTED_EXTENSIONS:
            paths.extend(sorted(input_dir.glob(f"**/*{extension}")))
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run depth estimation model to create depth maps and point clouds.")
    parser.add_argument("-i", "--image", action="append", dest="images", help="Path to an input image. Can be passed multiple times as a list.")
    parser.add_argument("-d", "--input-dir", type=Path, help="Directory of images to process in batch mode.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory where outputs will be written.")
    parser.add_argument("--cache-dir", type=Path, default=Path("./models/depth_pro"), help="Directory for caching the model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run on (e.g. cuda, cpu).")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32", help="Computation precision")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing images.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("depth_pro").setLevel(logging.WARNING)

    images = collect_images(args.images, args.input_dir)
    if not images:
        raise SystemExit("No input images provided. Use --image or --input-dir.")

    runner = DepthProInference(
        cache_dir=args.cache_dir,
        device=args.device,
        precision=args.precision,
    )

    if len(images) == 1:
        result = runner.predict_single(images[0], args.output_dir)
        logging.info("Saved depth results for %s", result.image_path.name)
    else:
        results = runner.predict_batch(images, args.output_dir, batch_size=args.batch_size)
        logging.info("Processed %d images. Last output: %s", len(results), results[-1].depth_map_path)


if __name__ == "__main__":
    main()
