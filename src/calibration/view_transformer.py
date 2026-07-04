"""
ViewTransformer — maps pixel bounding-box positions to real-world field
coordinates (metres) using per-frame homography matrices produced by TVCalib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ViewTransformer:
    """
    Converts pixel-space bounding-box anchors to field-plane coordinates.

    Each frame in the calibration DataFrame must contain a ``homography``
    column (a 3×3 matrix).  The transformer reads these matrices at
    construction time and applies them frame-by-frame when
    :meth:`transform_to_meters` is called.

    Anchor points:
        - Players, goalkeepers, referees → bottom-centre of the bounding box
          (approximate foot position).
        - Ball → centre of the bounding box.

    The result is stored as ``position_field = (x_metres, y_metres)``
    inside each track entry.
    """

    def __init__(self, calibration_df: pd.DataFrame) -> None:
        """
        Args:
            calibration_df: Output of the TVCalib calibration pipeline.
                            Must contain ``image_id`` and ``homography``
                            columns.
        """
        self.homographies: dict[int, np.ndarray] = {}

        for _, row in calibration_df.iterrows():
            try:
                image_id = str(row["image_id"])
                frame_num = int(image_id.split("_")[-1].split(".")[0])
                self.homographies[frame_num] = np.array(row["homography"])
            except (KeyError, ValueError, IndexError):
                continue

        if not self.homographies:
            raise ValueError(
                "No valid homography matrices found in the calibration DataFrame."
            )

        print(f"  ViewTransformer: {len(self.homographies)} frame homographies loaded.")

    def transform_to_meters(self, tracks: dict) -> dict:
        """
        Add ``position_field`` to every object entry that has a
        ``homography`` for its frame.

        Args:
            tracks: Tracking dict with keys ``"players"``, ``"goalkeepers"``,
                    ``"referees"``, ``"ball"``.

        Returns:
            The same *tracks* dict, mutated in place.
        """
        categories = ["players", "goalkeepers", "referees", "ball"]

        for cat in categories:
            if cat not in tracks:
                continue

            for frame_idx, frame_data in enumerate(tracks[cat]):
                h_matrix = self.homographies.get(frame_idx)
                if h_matrix is None:
                    continue

                for track_id, info in frame_data.items():
                    if "bbox" not in info:
                        continue

                    x1, y1, x2, y2 = info["bbox"]

                    # Anchor point selection
                    if cat == "ball":
                        px, py = (x1 + x2) / 2, (y1 + y2) / 2
                    else:
                        px, py = (x1 + x2) / 2, y2  # foot position

                    # Apply homography (perspective division)
                    point = np.array([px, py, 1.0])
                    world = h_matrix @ point

                    if abs(world[2]) > 1e-8:
                        tracks[cat][frame_idx][track_id]["position_field"] = (
                            world[0] / world[2],
                            world[1] / world[2],
                        )

        return tracks
