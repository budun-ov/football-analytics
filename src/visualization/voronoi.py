"""
Voronoi minimap — top-down pitch view with team-coloured territory regions.

VoronoiMinimap extends RadarMinimap by adding an approximate Voronoi
diagram behind the player dots.  Each point on the canvas is coloured
by the team of the nearest player or goalkeeper.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.visualization.radar import (
    RadarMinimap,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    RADAR_W,
    RADAR_H,
    COLOR_BALL,
    COLOR_REFEREE,
    COLOR_OUTLINE,
    R_PLAYER,
    R_GOALKEEPER,
    R_REFEREE,
    R_BALL,
    _get_frame,
    _team_color,
    _extract_team_colors,
)


class VoronoiMinimap(RadarMinimap):
    """
    Extends RadarMinimap with a fast approximate Voronoi diagram.

    Parameters
    ----------
    alpha : float
        Opacity of the team-coloured regions (0 = fully transparent,
        1 = fully opaque).
    grid_step : int
        Down-sampling step for the Voronoi grid.  Lower values produce
        a finer result but are slower.  Values of 3–5 are typically
        sufficient.
    draw_boundaries : bool
        Whether to draw dark borders between adjacent team regions.
    include_goalkeepers : bool
        Whether to include goalkeepers in the Voronoi region calculation.
    """

    def __init__(
        self,
        template_path: str,
        output_path: str,
        fps: float,
        radar_w: int = RADAR_W,
        radar_h: int = RADAR_H,
        pitch_length: float = PITCH_LENGTH_M,
        pitch_width: float = PITCH_WIDTH_M,
        alpha: float = 0.38,
        grid_step: int = 4,
        draw_boundaries: bool = True,
        include_goalkeepers: bool = True,
        show_id: bool = False,
        show_speed: bool = False,
    ) -> None:
        super().__init__(
            template_path=template_path,
            output_path=output_path,
            fps=fps,
            radar_w=radar_w,
            radar_h=radar_h,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            show_speed=show_speed,
            show_id=show_id,
        )

        self.alpha = float(alpha)
        self.grid_step = max(1, int(grid_step))
        self.draw_boundaries = draw_boundaries
        self.include_goalkeepers = include_goalkeepers

    # ------------------------------------------------------------------
    # Public overrides
    # ------------------------------------------------------------------

    def build_from_tracks(self, tracks: dict) -> str:
        """Render all frames and write the Voronoi MP4."""
        n_frames = max(
            len(tracks.get("players", [])),
            len(tracks.get("goalkeepers", [])),
            len(tracks.get("referees", [])),
            len(tracks.get("ball", [])),
        )

        if n_frames == 0:
            raise ValueError("tracks is empty: no frames available.")

        team_colors = _extract_team_colors(tracks)
        print(f"  [VoronoiMinimap] team colors (BGR): {team_colors}")
        print(f"  [VoronoiMinimap] rendering {n_frames} frames → {self.output_path}")

        self.open_writer()

        try:
            for frame_idx in tqdm(range(n_frames), desc="Voronoi video"):
                frame = self.render_frame(tracks, frame_idx, team_colors)
                self.write_frame(frame)
        finally:
            self.close_writer()

        print(f"  ✓  Voronoi saved: {self.output_path}")
        return self.output_path

    def render_frame(
        self,
        tracks: dict,
        frame_idx: int,
        team_colors: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw one Voronoi frame."""
        if team_colors is None:
            team_colors = _extract_team_colors(tracks)

        canvas = self._blank.copy()

        control_objects = self._collect_control_objects(tracks, frame_idx, team_colors)

        if len(control_objects) >= 2:
            canvas = self._draw_voronoi_regions(canvas, control_objects)

        self._draw_objects_on_top(canvas, tracks, frame_idx, team_colors)

        return canvas

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_control_objects(
        self,
        tracks: dict,
        frame_idx: int,
        team_colors: dict,
    ) -> list:
        """
        Build the list of objects used for Voronoi region construction.

        Referees and the ball are *not* used for regions; they are drawn
        on top afterward.
        """
        categories = ["players"]
        if self.include_goalkeepers:
            categories.append("goalkeepers")

        objects = []

        for category in categories:
            for track_id, info in _get_frame(tracks, category, frame_idx).items():
                pos = info.get("position_field")
                if pos is None:
                    continue

                px, py = self._m_to_px(*pos)
                objects.append(
                    {
                        "track_id": track_id,
                        "category": category,
                        "info": info,
                        "px": px,
                        "py": py,
                        "color": _team_color(info, team_colors),
                    }
                )

        return objects

    def _draw_voronoi_regions(
        self,
        canvas: np.ndarray,
        objects: list,
    ) -> np.ndarray:
        """
        Approximate Voronoi diagram via a downsampled nearest-neighbour grid.

        ``grid_step=4`` gives adequate precision and is much faster than
        computing the exact Voronoi for every pixel.
        """
        h, w = canvas.shape[:2]
        step = self.grid_step

        xs = np.clip(np.arange(0, w, step, dtype=np.float32) + step / 2, 0, w - 1)
        ys = np.clip(np.arange(0, h, step, dtype=np.float32) + step / 2, 0, h - 1)

        grid_x, grid_y = np.meshgrid(xs, ys)

        points = np.array([[obj["px"], obj["py"]] for obj in objects], dtype=np.float32)
        colors = np.array([obj["color"] for obj in objects], dtype=np.uint8)

        dist2 = (
            (grid_x[..., None] - points[:, 0]) ** 2
            + (grid_y[..., None] - points[:, 1]) ** 2
        )
        labels = np.argmin(dist2, axis=2)

        voronoi_small = colors[labels]
        voronoi_full = cv2.resize(
            voronoi_small, (w, h), interpolation=cv2.INTER_NEAREST
        )

        blended = cv2.addWeighted(canvas, 1.0 - self.alpha, voronoi_full, self.alpha, 0)

        if self.draw_boundaries:
            edges = np.zeros(labels.shape, dtype=bool)
            edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            edges[1:, :] |= labels[1:, :] != labels[:-1, :]

            edge_mask = cv2.resize(
                edges.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST
            )
            edge_mask = cv2.dilate(edge_mask, np.ones((2, 2), np.uint8), iterations=1)
            blended[edge_mask > 0] = (35, 35, 35)

        # Restore pitch markings on top of coloured regions
        line_mask = np.all(canvas > 245, axis=2)
        blended[line_mask] = canvas[line_mask]

        return blended

    def _draw_objects_on_top(
        self,
        canvas: np.ndarray,
        tracks: dict,
        frame_idx: int,
        team_colors: dict,
    ) -> None:
        """Draw player dots, referee dots and ball on top of the Voronoi regions."""
        players_d = _get_frame(tracks, "players", frame_idx)
        goalkeepers_d = _get_frame(tracks, "goalkeepers", frame_idx)
        referees_d = _get_frame(tracks, "referees", frame_idx)
        ball_d = _get_frame(tracks, "ball", frame_idx)

        for tid, info in players_d.items():
            self._draw_dot(canvas, info, _team_color(info, team_colors), R_PLAYER, tid)

        for tid, info in goalkeepers_d.items():
            self._draw_dot(
                canvas, info, _team_color(info, team_colors),
                R_GOALKEEPER, tid, outline_thickness=2,
            )

        for tid, info in referees_d.items():
            self._draw_dot(canvas, info, COLOR_REFEREE, R_REFEREE, tid)

        for _, info in ball_d.items():
            self._draw_dot(canvas, info, COLOR_BALL, R_BALL,
                           label=None, outline_color=(80, 80, 80))
