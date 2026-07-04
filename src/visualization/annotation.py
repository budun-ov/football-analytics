"""
Main video annotation — draws ellipses, triangles, ball-possession overlay
and player speed text directly on video frames.
"""

from __future__ import annotations

from typing import Generator, List, Optional

import cv2
import numpy as np

from src.kinematics.speed import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position,
)


# ---------------------------------------------------------------------------
# Primitive drawing helpers
# ---------------------------------------------------------------------------

def draw_ellipse(
    frame: np.ndarray,
    bbox: list,
    color: tuple,
    track_id: Optional[int] = None,
) -> np.ndarray:
    """
    Draw a coloured ellipse below an object (player marker).

    If *track_id* is given, a small rectangle with the ID number is drawn
    beneath the ellipse.
    """
    x_center = get_center_of_bbox(bbox)[0]
    y_bottom = int(bbox[3])
    width = get_bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(x_center, y_bottom),
        axes=(width, int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    if track_id is not None:
        rx1, ry1 = x_center - 20, y_bottom + 5
        rx2, ry2 = x_center + 20, y_bottom + 25

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, cv2.FILLED)

        text_x = rx1 + (2 if track_id > 99 else 12)
        cv2.putText(
            frame,
            str(track_id),
            (text_x, ry1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame


def draw_triangle(
    frame: np.ndarray,
    bbox: list,
    color: tuple,
) -> np.ndarray:
    """Draw a filled triangle above an object (ball / possession indicator)."""
    x_center = get_center_of_bbox(bbox)[0]
    y_top = int(bbox[1])

    triangle_points = np.array(
        [
            [x_center, y_top],
            [x_center - 10, y_top - 20],
            [x_center + 10, y_top - 20],
        ],
        np.int32,
    )

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame


# ---------------------------------------------------------------------------
# Ball possession overlay
# ---------------------------------------------------------------------------

def draw_team_ball_control(
    frame: np.ndarray,
    frame_num: int,
    team_ball_control: np.ndarray,
) -> np.ndarray:
    """
    Draw ball-possession percentages in the bottom-right corner.

    The statistics are calculated from frame 0 up to and including
    *frame_num* (cumulative).
    """
    if team_ball_control is None:
        return frame

    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    control_till_now = team_ball_control[: frame_num + 1]
    if len(control_till_now) == 0:
        return frame

    t1 = np.sum(control_till_now == 0) / len(control_till_now)
    t2 = np.sum(control_till_now == 1) / len(control_till_now)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        f"Team 1 Ball Control: {t1 * 100:.2f}%",
        (1400, 900),
        font,
        1,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        frame,
        f"Team 2 Ball Control: {t2 * 100:.2f}%",
        (1400, 950),
        font,
        1,
        (0, 0, 0),
        3,
    )

    return frame


# ---------------------------------------------------------------------------
# Speed overlay
# ---------------------------------------------------------------------------

def draw_speed(
    frames: List[np.ndarray],
    tracks: dict,
) -> List[np.ndarray]:
    """
    Annotate each frame with per-player speed text (km/h).

    Args:
        frames: List of BGR frames (already annotated with ellipses etc.).
        tracks: Tracking dict containing ``"speed"`` values.

    Returns:
        New list of frames with speed text drawn.
    """
    output: List[np.ndarray] = []

    for frame_num, frame in enumerate(frames):
        frame = frame.copy()

        for obj_type in ["players", "goalkeepers"]:
            if frame_num >= len(tracks[obj_type]):
                continue

            for track_info in tracks[obj_type][frame_num].values():
                if "speed" not in track_info:
                    continue

                speed_kmh = track_info["speed"]
                foot_x, foot_y = get_foot_position(track_info["bbox"])

                cv2.putText(
                    frame,
                    f"{speed_kmh:.1f} km/h",
                    (foot_x - 30, foot_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        output.append(frame)

    return output


# ---------------------------------------------------------------------------
# Main annotation generator
# ---------------------------------------------------------------------------

def draw_annotations(
    frame_generator,
    tracks: dict,
    team_ball_control: Optional[np.ndarray] = None,
    show_ball_possession: bool = True,
) -> Generator[np.ndarray, None, None]:
    """
    Lazily yield annotated frames.

    For each frame:
      - Draws coloured ellipses under every player and referee.
      - Draws a green triangle above the ball.
      - Draws a red triangle above the player who has the ball.
      - Optionally draws the ball-possession overlay.

    Args:
        frame_generator: Iterable of BGR frames.
        tracks: Tracking dict (must contain ``"players"``, ``"referees"``,
                ``"ball"``).
        team_ball_control: Cumulative possession array (may be ``None``).
        show_ball_possession: Whether to draw the possession overlay.

    Yields:
        Annotated BGR frames.
    """
    for frame_num, frame in enumerate(frame_generator):
        frame = frame.copy()

        players = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
        referees = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
        balls = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}

        # Players
        for track_id, player in players.items():
            color = player.get("team_color", (0, 0, 255))
            frame = draw_ellipse(frame, player["bbox"], color, track_id)

            if player.get("has_ball", False):
                frame = draw_triangle(frame, player["bbox"], (0, 0, 255))

        # Referees (cyan)
        for ref in referees.values():
            frame = draw_ellipse(frame, ref["bbox"], (0, 255, 255))

        # Ball (green triangle)
        for ball in balls.values():
            frame = draw_triangle(frame, ball["bbox"], (0, 255, 0))

        # Ball-possession overlay
        if show_ball_possession and team_ball_control is not None:
            frame = draw_team_ball_control(frame, frame_num, team_ball_control)

        yield frame
