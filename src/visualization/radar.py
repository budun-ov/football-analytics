"""
Radar minimap — 2-D top-down view of player positions on the pitch.

The RadarMinimap class renders player, goalkeeper, referee and ball
positions as coloured dots on a scaled pitch template and writes the
result to an MP4 video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0
RADAR_W: int = 1050
RADAR_H: int = 680

# BGR colours
COLOR_GRASS = (56, 148, 50)
COLOR_LINES = (255, 255, 255)
COLOR_BALL = (255, 255, 255)
COLOR_REFEREE = (0, 165, 255)   # orange
COLOR_OUTLINE = (0, 0, 0)
COLOR_DEFAULT = (180, 180, 180) # player without a team assignment

R_PLAYER = 10
R_GOALKEEPER = 13
R_REFEREE = 8
R_BALL = 6


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_frame(tracks: dict, category: str, frame_idx: int) -> dict:
    """Return the frame dictionary for *category* at *frame_idx*, or ``{}``."""
    lst = tracks.get(category, [])
    return lst[frame_idx] if frame_idx < len(lst) else {}


def _team_color(info: dict, team_colors: dict) -> tuple:
    """Return the team BGR colour for *info*, or the default grey."""
    tid = info.get("team_id")
    if tid is not None and tid in team_colors:
        return team_colors[tid]
    return COLOR_DEFAULT


def _extract_team_colors(tracks: dict) -> dict:
    """
    Collect ``{team_id: color_bgr}`` from player / goalkeeper entries.

    Note: ``team_color`` is stored as RGB by the TeamClassifier, so we
    convert to BGR for OpenCV here.
    """
    colors: dict = {}

    def _process_frame(frame_data: dict) -> None:
        for info in frame_data.values():
            tid = info.get("team_id")
            tc = info.get("team_color")

            if tid is None or tc is None or tid in colors:
                continue

            if isinstance(tc, (tuple, list)) and len(tc) == 3:
                try:
                    r, g, b = map(int, tc)
                    colors[tid] = (b, g, r)   # RGB → BGR
                except (ValueError, TypeError):
                    pass

    for frame_data in tracks.get("players", []):
        _process_frame(frame_data)
        if len(colors) >= 2:
            return colors

    for frame_data in tracks.get("goalkeepers", []):
        _process_frame(frame_data)
        if len(colors) >= 2:
            return colors

    return colors


# ---------------------------------------------------------------------------
# RadarMinimap
# ---------------------------------------------------------------------------

class RadarMinimap:
    """
    Builds a 2-D radar minimap from tracking data and saves it as an MP4.

    Parameters
    ----------
    template_path : str
        Path to ``template_pitch_t.png`` — a white-background image with
        blue pitch-marking lines used to generate the green pitch canvas.
    output_path : str
        Destination MP4 file path.
    fps : float
        Video frame rate.
    radar_w, radar_h : int
        Canvas dimensions in pixels (aspect ratio should be close to 105:68).
    pitch_length, pitch_width : float
        Real pitch dimensions in metres.
    show_speed : bool
        (Currently disabled in the draw loop — available for extension.)
    show_id : bool
        (Currently disabled in the draw loop — available for extension.)
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
        show_speed: bool = True,
        show_id: bool = False,
    ) -> None:
        self.output_path = str(output_path)
        self.fps = fps
        self.radar_w = radar_w
        self.radar_h = radar_h
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.show_speed = show_speed
        self.show_id = show_id

        # Scaling factors: metres → pixels
        self.sx = radar_w / pitch_length
        self.sy = radar_h / pitch_width

        self._blank = self._build_blank_field(template_path)
        self._writer: Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build_from_tracks(self, tracks: dict) -> str:
        """
        Main entry point — render all frames and write the MP4.

        Returns the path of the saved file.
        """
        n_frames = max(
            len(tracks.get("players", [])),
            len(tracks.get("goalkeepers", [])),
            len(tracks.get("referees", [])),
            len(tracks.get("ball", [])),
        )

        team_colors = _extract_team_colors(tracks)
        print(f"  [RadarMinimap] team colors (BGR): {team_colors}")
        print(f"  [RadarMinimap] rendering {n_frames} frames → {self.output_path}")

        self.open_writer()

        for frame_idx in tqdm(range(n_frames), desc="Radar video"):
            frame = self.render_frame(tracks, frame_idx, team_colors)
            self.write_frame(frame)

        self.close_writer()
        print(f"  ✓  Radar saved: {self.output_path}")
        return self.output_path

    def render_frame(
        self,
        tracks: dict,
        frame_idx: int,
        team_colors: Optional[dict] = None,
    ) -> np.ndarray:
        """Draw one radar frame and return it as a BGR ndarray."""
        if team_colors is None:
            team_colors = _extract_team_colors(tracks)

        canvas = self._blank.copy()

        players_d = _get_frame(tracks, "players", frame_idx)
        goalkeepers_d = _get_frame(tracks, "goalkeepers", frame_idx)
        referees_d = _get_frame(tracks, "referees", frame_idx)
        ball_d = _get_frame(tracks, "ball", frame_idx)

        for tid, info in players_d.items():
            self._draw_dot(canvas, info, _team_color(info, team_colors), R_PLAYER, tid)

        for tid, info in goalkeepers_d.items():
            self._draw_dot(canvas, info, _team_color(info, team_colors), R_GOALKEEPER,
                           tid, outline_thickness=2)

        for tid, info in referees_d.items():
            self._draw_dot(canvas, info, COLOR_REFEREE, R_REFEREE, tid)

        for _, info in ball_d.items():
            self._draw_dot(canvas, info, COLOR_BALL, R_BALL,
                           label=None, outline_color=(80, 80, 80))

        return canvas

    def open_writer(self) -> None:
        """Open the VideoWriter. Must be called before :meth:`write_frame`."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.radar_w, self.radar_h)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"VideoWriter failed: {self.output_path}")

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one BGR frame to the open video."""
        assert self._writer is not None, "Call open_writer() first."
        self._writer.write(frame)

    def close_writer(self) -> None:
        """Finalise and close the MP4 file."""
        if self._writer:
            self._writer.release()
            self._writer = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_blank_field(self, template_path: str) -> np.ndarray:
        """
        Build the empty pitch canvas from the template image.

        The template has a white background with blue pitch-marking lines.
        This method:
          1. Detects blue pixels (B channel >> G, R channels).
          2. Creates a green background.
          3. Paints white where blue lines were detected.
        """
        tmpl = cv2.imread(str(template_path))
        if tmpl is None:
            raise FileNotFoundError(
                f"Pitch template not found: {template_path}\n"
                "Place template_pitch_t.png in the 'assets/' folder."
            )

        tmpl = cv2.resize(
            tmpl, (self.radar_w, self.radar_h), interpolation=cv2.INTER_LINEAR
        )

        b = tmpl[:, :, 0].astype(np.int16)
        g = tmpl[:, :, 1].astype(np.int16)
        r = tmpl[:, :, 2].astype(np.int16)

        line_mask = ((b - g > 55) & (b - r > 55) & (b > 80)).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        line_mask = cv2.dilate(line_mask, k)

        blank = np.full((self.radar_h, self.radar_w, 3), COLOR_GRASS, dtype=np.uint8)
        blank[line_mask > 0] = COLOR_LINES

        return blank

    def _m_to_px(self, x_m: float, y_m: float) -> tuple:
        """Convert field coordinates (metres, origin at centre) to canvas pixels."""
        px = int((x_m + self.pitch_length / 2) * self.sx)
        py = int((y_m + self.pitch_width / 2) * self.sy)
        px = max(0, min(self.radar_w - 1, px))
        py = max(0, min(self.radar_h - 1, py))
        return px, py

    def _draw_dot(
        self,
        canvas: np.ndarray,
        info: dict,
        color: tuple,
        radius: int,
        label=None,
        outline_thickness: int = 1,
        outline_color: tuple = COLOR_OUTLINE,
    ) -> None:
        """Draw one circle on *canvas*; skip silently if position is unknown."""
        pos = info.get("position_field")
        if pos is None:
            return

        cx, cy = self._m_to_px(*pos)

        cv2.circle(canvas, (cx, cy), radius + outline_thickness, outline_color, -1)
        cv2.circle(canvas, (cx, cy), radius, color, -1)
