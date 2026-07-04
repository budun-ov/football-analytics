"""
Object Tracker — YOLO detection + ByteTrack multi-object tracking.

Handles:
  - Player / goalkeeper / referee / ball detection via a YOLO model.
  - Multi-object tracking via supervision ByteTrack.
  - Batch processing to reduce GPU memory pressure.
  - Optional result caching (pickle stub).
  - Ball position interpolation for missing/noisy detections.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.video_io import frame_index_from_image_id


# ---------------------------------------------------------------------------
# Ball interpolation helper (imported where needed)
# ---------------------------------------------------------------------------

def interpolate_ball_positions(ball_tracks: List[dict]) -> List[dict]:
    """
    Linear interpolation of missing or noisy ball bounding boxes.

    Jumps larger than 150 pixels are also treated as missing and
    filled via linear interpolation (up to 30 consecutive frames).
    """
    import pandas as pd  # lazy import — only needed here

    rows = []
    for frame_data in ball_tracks:
        if frame_data:
            bbox = next(iter(frame_data.values()))["bbox"]
            rows.append({"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]})
        else:
            rows.append({"x1": None, "y1": None, "x2": None, "y2": None})

    df = pd.DataFrame(rows)

    # Mark large jumps as missing
    mx = (df["x1"] + df["x2"]) / 2
    my = (df["y1"] + df["y2"]) / 2
    dist = np.sqrt(mx.diff() ** 2 + my.diff() ** 2)
    df.loc[dist > 150, ["x1", "y1", "x2", "y2"]] = np.nan

    df = df.interpolate(method="linear", limit_direction="both", limit=30)
    df = df.bfill()

    return [
        {0: {"bbox": [row.x1, row.y1, row.x2, row.y2]}}
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """
    Wrapper that combines YOLO object detection with ByteTrack tracking.

    Usage::

        tracker = Tracker(model_path="models/detection/best.pt")
        tracks = tracker.get_object_tracks(frame_generator, ...)
    """

    def __init__(
        self,
        model_path: str,
        frame_size: Optional[tuple] = None,
        bytetrack_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model_path: Path to the local YOLO ``.pt`` weights file.
            frame_size: Optional (width, height) tuple (unused by YOLO but
                        kept for future use).
            bytetrack_params: Keyword arguments forwarded to ``sv.ByteTrack``.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Detection model not found: {model_path}\n"
                "Place the YOLO weights file at models/detection/best.pt "
                "(or update 'models.detection' in config.yaml)."
            )

        print(f"  Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        self.frame_size = frame_size

        self.class_names: Dict[int, str] = self.model.names
        self.class_names_inverse: Dict[str, int] = {
            v: k for k, v in self.class_names.items()
        }

        # Commonly used class IDs
        self.player_id = self.class_names_inverse.get("player")
        self.goalkeeper_id = self.class_names_inverse.get("goalkeeper")
        self.referee_id = self.class_names_inverse.get("referee")
        self.ball_id = self.class_names_inverse.get("ball")

        if bytetrack_params:
            self.tracker = sv.ByteTrack(**bytetrack_params)
        else:
            self.tracker = sv.ByteTrack()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _process_track_batch(
        self,
        batch: List[np.ndarray],
        tracks: dict,
        start_frame: int,
        conf: float = 0.3,
        verbose: bool = False,
        half: bool = False,
        stream: bool = False,
    ) -> None:
        """
        Run YOLO + ByteTrack on one batch of frames and append results to
        *tracks*.
        """
        detections_batch = self.model.predict(
            batch,
            conf=conf,
            verbose=verbose,
            half=half,
            stream=stream,
        )

        for i, detection in enumerate(detections_batch):
            cur_frame = start_frame + i

            sv_detections = sv.Detections.from_ultralytics(detection)
            detections_with_tracks = self.tracker.update_with_detections(sv_detections)

            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Players, goalkeepers, referees — store with tracker_id
            for bbox, cls_id, trk_id in zip(
                detections_with_tracks.xyxy,
                detections_with_tracks.class_id,
                detections_with_tracks.tracker_id,
            ):
                if cls_id == self.player_id:
                    tracks["players"][cur_frame][int(trk_id)] = {"bbox": bbox}
                elif cls_id == self.goalkeeper_id:
                    tracks["goalkeepers"][cur_frame][int(trk_id)] = {"bbox": bbox}
                elif cls_id == self.referee_id:
                    tracks["referees"][cur_frame][int(trk_id)] = {"bbox": bbox}

            # Ball — not assigned a persistent tracker_id; stored by detection index
            for j, (bbox, cls_id) in enumerate(
                zip(sv_detections.xyxy, sv_detections.class_id)
            ):
                if cls_id == self.ball_id:
                    tracks["ball"][cur_frame][j] = {"bbox": bbox}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_object_tracks(
        self,
        frame_generator,
        batch_size: int = 20,
        total_frames: int = 0,
        conf: float = 0.3,
        half: bool = False,
        stream: bool = False,
        verbose: bool = False,
        stub_path: Optional[str] = None,
        interpolate_ball: bool = True,
    ) -> dict:
        """
        Run the full detection + tracking pipeline on *frame_generator*.

        Args:
            frame_generator: Iterable that yields BGR frames (e.g. from
                             ``sv.get_video_frames_generator``).
            batch_size: Number of frames to process in one YOLO call.
            total_frames: Total frame count (used only for the progress bar).
            conf: YOLO confidence threshold.
            half: Enable INT8 quantisation (requires compatible hardware).
            stream: Enable YOLO streaming inference.
            verbose: Print per-frame YOLO logs.
            stub_path: If set, load/save results from/to a pickle cache.
            interpolate_ball: Fill in missing ball positions after tracking.

        Returns:
            dict with keys ``"players"``, ``"goalkeepers"``, ``"referees"``,
            ``"ball"``; each is a list of per-frame dicts.
        """
        # ── Try loading from cache ────────────────────────────────────────
        if stub_path and os.path.exists(stub_path):
            if os.path.getsize(stub_path) > 0:
                try:
                    with open(stub_path, "rb") as fh:
                        print(f"  Loading tracking stub: {stub_path}")
                        return pickle.load(fh)
                except (EOFError, pickle.UnpicklingError):
                    print(f"  Stub corrupted — re-running tracking: {stub_path}")
                    os.remove(stub_path)
            else:
                os.remove(stub_path)

        # ── Run tracking ──────────────────────────────────────────────────
        tracks: dict = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": [],
        }

        batch: List[np.ndarray] = []
        frame_num = 0

        with tqdm(
            total=total_frames or None,
            desc="Tracking",
            smoothing=0.1,
            mininterval=0.05,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for frame in frame_generator:
                batch.append(frame)
                frame_num += 1

                if len(batch) == batch_size:
                    self._process_track_batch(
                        batch=batch,
                        tracks=tracks,
                        start_frame=frame_num - batch_size,
                        conf=conf,
                        verbose=verbose,
                        half=half,
                        stream=stream,
                    )
                    batch.clear()

                pbar.update(1)

            # Remaining frames
            if batch:
                self._process_track_batch(
                    batch=batch,
                    tracks=tracks,
                    start_frame=frame_num - len(batch),
                    conf=conf,
                    verbose=verbose,
                    half=half,
                    stream=stream,
                )

        # ── Post-processing ───────────────────────────────────────────────
        if interpolate_ball:
            tracks["ball"] = interpolate_ball_positions(tracks["ball"])

        # ── Save cache ────────────────────────────────────────────────────
        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as fh:
                pickle.dump(tracks, fh)
            print(f"  Tracking stub saved: {stub_path}")

        return tracks
