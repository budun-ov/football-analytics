"""
Bounding box helpers and kinematic calculations (speed).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Bounding box utilities
# ---------------------------------------------------------------------------

def get_center_of_bbox(bbox: list) -> Tuple[int, int]:
    """Return (cx, cy) — the centre of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list) -> int:
    """Return the width of a bounding box."""
    return int(bbox[2] - bbox[0])


def get_foot_position(bbox: list) -> Tuple[int, int]:
    """Return the bottom-centre point of a bounding box (player's feet)."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1: Tuple, p2: Tuple) -> float:
    """Euclidean distance between two 2-D points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1: Tuple, p2: Tuple) -> Tuple[float, float]:
    """Return (dx, dy) between two points."""
    return p1[0] - p2[0], p1[1] - p2[1]


def get_closest_to_point_bbox(bbox1: list, bbox2: list, point: Tuple) -> list:
    """Return whichever bounding box has its centre closer to *point*."""
    if measure_distance(get_center_of_bbox(bbox1), point) < measure_distance(
        get_center_of_bbox(bbox2), point
    ):
        return bbox1
    return bbox2


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def calculate_kinematics(tracks: dict, fps: float = 25.0) -> dict:
    """
    Compute smoothed player speed (km/h) from ``position_field`` coordinates
    and write it back into *tracks* as the ``"speed"`` key.

    Processing stages:
        1. Collect per-track trajectories from ``position_field``.
        2. Apply a median filter to remove sudden ``teleport`` artefacts
           caused by homography jitter.
        3. Apply a Savitzky–Golay filter for further smoothing.
        4. Compute instantaneous speed from frame-to-frame displacement.
        5. Smooth the speed signal a second time.
        6. Cap at 40 km/h to suppress remaining homography artefacts.

    Args:
        tracks: Tracking dictionary with ``position_field`` populated.
        fps: Video frame rate (used for converting distance/frame to km/h).

    Returns:
        The same *tracks* dict with ``"speed"`` added to each player entry.
    """
    categories = ["players", "goalkeepers", "referees"]

    for cat in categories:
        if cat not in tracks:
            continue

        # ── Collect per-track trajectories ──────────────────────────────
        trajectories: dict = {}

        for frame_idx, frame_data in enumerate(tracks[cat]):
            for track_id, info in frame_data.items():
                if "position_field" not in info:
                    continue

                if track_id not in trajectories:
                    trajectories[track_id] = {"frames": [], "coords": []}

                trajectories[track_id]["frames"].append(frame_idx)
                trajectories[track_id]["coords"].append(info["position_field"])

        # ── Process each track ───────────────────────────────────────────
        for track_id, data in trajectories.items():
            coords = np.array(data["coords"])
            N = len(coords)

            if N < 7:
                # Not enough observations for reliable smoothing
                continue

            # 1. Median filter — suppresses sudden jumps
            coords_med_x = median_filter(coords[:, 0], size=3)
            coords_med_y = median_filter(coords[:, 1], size=3)

            # 2. Savitzky–Golay smoothing — preserves motion shape
            window = min(15, N if N % 2 == 1 else N - 1)
            smooth_x = savgol_filter(coords_med_x, window, polyorder=2)
            smooth_y = savgol_filter(coords_med_y, window, polyorder=2)

            # 3. Speed in km/h  (metres/frame × fps × 3.6)
            dx = np.diff(smooth_x)
            dy = np.diff(smooth_y)
            dist = np.sqrt(dx**2 + dy**2)
            speed_kmh = dist * fps * 3.6

            # 4. Smooth the speed signal itself
            if len(speed_kmh) > 5:
                sp_win = min(9, len(speed_kmh) // 2 * 2 - 1)
                speed_kmh = savgol_filter(speed_kmh, sp_win, polyorder=1)

            # 5. Write results back; cap at 40 km/h
            frames = data["frames"]
            for i, frame_idx in enumerate(frames):
                s = float(speed_kmh[i - 1]) if i > 0 else float(speed_kmh[0])
                tracks[cat][frame_idx][track_id]["speed"] = min(s, 40.0)

    return tracks
