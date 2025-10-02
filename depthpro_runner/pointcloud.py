"""Point cloud generation utilities for RGB-D outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    focal_length_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a depth map into a 3D point cloud.

    Args:
        depth: Depth map in meters of shape (H, W).
        rgb: RGB image aligned with depth, uint8 array of shape (H, W, 3).
        focal_length_px: Focal length in pixels (assumes square pixels).

    Returns:
        Tuple of (points_xyz, colors_rgb) where each is shaped (N, 3).
    """

    if depth.ndim != 2:
        raise ValueError("Depth map must be HxW")
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB image must match depth spatial dimensions")
    if focal_length_px <= 0:
        raise ValueError("Focal length must be positive")

    height, width = depth.shape
    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u_coords, v_coords)

    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    z = depth.astype(np.float32)
    x = (grid_u - cx) * z / focal_length_px
    y = (grid_v - cy) * z / focal_length_px

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    return points, colors


def save_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    """Save a point cloud to ASCII PLY format."""

    if points.shape != colors.shape:
        raise ValueError("Points and colors must have the same shape")

    output_path = output_path.with_suffix(".ply")
    num_points = points.shape[0]
    header = """ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".strip().replace("{num_points}", str(num_points))

    colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for (x, y, z), (r, g, b) in zip(points.astype(np.float32), colors_uint8):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
