"""Projection helpers for calibrated NavLock camera geometry."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def quaternion_to_rotation_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix from a ``[w, x, y, z]`` quaternion."""
    values = [float(value) for value in quaternion]
    if len(values) != 4:
        raise ValueError("quaternion must contain four values")
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def lidar_to_camera_points(points_lidar: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    """Transform lidar-frame points into a calibrated camera frame.

    NavLock stores calibrated camera extrinsics as ``cam2lidar`` pose:
    ``translation`` is the camera origin in lidar coordinates and ``rotation`` is
    the camera-to-lidar quaternion. Projection therefore uses the inverse pose.
    """
    rotation = quaternion_to_rotation_matrix(calibration["rotation"])
    translation = np.array(calibration["translation"], dtype=float)
    points = np.asarray(points_lidar, dtype=float)
    return (rotation.T @ (points - translation).T).T


def camera_ray_to_lidar(
    pixel_xy: Iterable[float],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Back-project an image pixel to a lidar-frame camera ray.

    Returns ``(origin, unit_direction)`` in lidar coordinates. The ray uses the
    same NavLock ``cam2lidar`` convention as :func:`lidar_to_camera_points`.
    """
    intrinsic = np.array(calibration.get("camera_intrinsic") or [], dtype=float)
    if intrinsic.shape != (3, 3):
        return None
    pixel = [float(value) for value in pixel_xy]
    if len(pixel) != 2:
        raise ValueError("pixel_xy must contain two values")
    try:
        inv_intrinsic = np.linalg.inv(intrinsic)
    except np.linalg.LinAlgError:
        return None

    direction_camera = inv_intrinsic @ np.array([pixel[0], pixel[1], 1.0], dtype=float)
    norm = np.linalg.norm(direction_camera)
    if norm <= 0.0:
        return None
    direction_camera = direction_camera / norm
    rotation = quaternion_to_rotation_matrix(calibration["rotation"])
    origin = np.array(calibration["translation"], dtype=float)
    direction = rotation @ direction_camera
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 0.0:
        return None
    return origin, direction / direction_norm


def triangulate_lidar_rays(
    rays: Iterable[tuple[Iterable[float], Iterable[float]]],
) -> tuple[np.ndarray, float] | None:
    """Return least-squares point and mean residual distance for lidar rays."""
    ray_items = [
        (np.array(origin, dtype=float), np.array(direction, dtype=float))
        for origin, direction in rays
    ]
    if len(ray_items) < 2:
        return None

    system = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    identity = np.eye(3, dtype=float)
    for origin, direction in ray_items:
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            return None
        direction = direction / norm
        projector = identity - np.outer(direction, direction)
        system += projector
        rhs += projector @ origin
    try:
        point = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None

    residuals = []
    for origin, direction in ray_items:
        direction = direction / np.linalg.norm(direction)
        residuals.append(np.linalg.norm(np.cross(point - origin, direction)))
    return point, float(sum(residuals) / len(residuals))


def project_lidar_point_to_image(
    point_lidar: Iterable[float],
    calibration: dict[str, Any],
    image_width: float,
    image_height: float,
    *,
    min_depth: float = 0.1,
) -> tuple[float, float] | None:
    """Project one lidar-frame point into image coordinates if it is visible."""
    intrinsic = np.array(calibration.get("camera_intrinsic") or [], dtype=float)
    if intrinsic.shape != (3, 3):
        return None
    point = np.array([list(point_lidar)], dtype=float)
    if point.shape != (1, 3):
        raise ValueError("point_lidar must contain three values")
    point_camera = lidar_to_camera_points(point, calibration)[0]
    if point_camera[2] <= float(min_depth):
        return None
    projected = intrinsic @ point_camera
    x = float(projected[0] / projected[2])
    y = float(projected[1] / projected[2])
    if x < 0.0 or y < 0.0 or x > image_width or y > image_height:
        return None
    return x, y


def box_corners_lidar(box: Iterable[float]) -> np.ndarray:
    """Return 8 lidar-frame corners for ``[x, y, z, dx, dy, dz, yaw]``."""
    values = [float(value) for value in box]
    if len(values) < 7:
        raise ValueError("box must contain at least seven values")
    x, y, z, dx, dy, dz, yaw = values[:7]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                local_x = sx * dx
                local_y = sy * dy
                corners.append(
                    [
                        x + cos_yaw * local_x - sin_yaw * local_y,
                        y + sin_yaw * local_x + cos_yaw * local_y,
                        z + sz * dz,
                    ]
                )
    return np.array(corners, dtype=float)


def project_lidar_box_to_image(
    box: Iterable[float],
    calibration: dict[str, Any],
    image_width: float,
    image_height: float,
    *,
    min_depth: float = 0.1,
) -> tuple[float, float, float, float] | None:
    """Project a lidar 3D box into an image-space 2D box.

    Returns a clipped ``(x1, y1, x2, y2)`` bbox, or ``None`` if all corners are
    behind the camera or outside the image.
    """
    intrinsic = np.array(calibration.get("camera_intrinsic") or [], dtype=float)
    if intrinsic.shape != (3, 3):
        return None

    points_camera = lidar_to_camera_points(box_corners_lidar(box), calibration)
    points_camera = points_camera[points_camera[:, 2] > float(min_depth)]
    if len(points_camera) == 0:
        return None

    projected = (intrinsic @ points_camera.T).T
    image_points = projected[:, :2] / projected[:, 2:3]
    x1, y1 = image_points.min(axis=0)
    x2, y2 = image_points.max(axis=0)
    if x2 < 0.0 or y2 < 0.0 or x1 > image_width or y1 > image_height:
        return None
    return (
        max(0.0, float(x1)),
        max(0.0, float(y1)),
        min(float(image_width), float(x2)),
        min(float(image_height), float(y2)),
    )


def bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0
