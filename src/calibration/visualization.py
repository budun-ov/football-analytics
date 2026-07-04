"""
Calibration visualisation — draws TVCalib reprojection overlays on video frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as T
from tqdm import tqdm

from src.video_io import frame_index_from_image_id


def visualize_calibration_frame(
    frame_bgr: np.ndarray,
    sample: pd.Series,
    ctx,   # TVCalibContext — typed loosely to avoid a circular import
) -> np.ndarray:
    """
    Draw the TVCalib reprojection overlay on a single BGR frame.

    Args:
        frame_bgr: Source frame in OpenCV BGR format.
        sample: Row from the calibration DataFrame.
        ctx: Initialised TVCalibContext.

    Returns:
        Annotated frame in BGR format.
    """
    from tvcalib.inference import get_camera_from_per_sample_output
    from tvcalib.utils import visualization_mpl_min as viz

    config = ctx.config

    # BGR → RGB → tensor
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(
        frame_rgb,
        (config.image_width, config.image_height),
        interpolation=cv2.INTER_LINEAR,
    )
    image_tensor = T.functional.to_tensor(frame_rgb)

    cam = get_camera_from_per_sample_output(sample, config.lens_dist)

    fig, ax = viz.init_figure(config.image_width, config.image_height)
    ax = viz.draw_image(ax, image_tensor)
    ax = viz.draw_reprojection(ax, ctx.object3dcpu, cam)
    ax = viz.draw_selected_points(
        ax,
        ctx.object3dcpu,
        sample["points_line"],
        sample["points_circle"],
        kwargs_outer={
            "zorder": 1000,
            "rasterized": False,
            "s": 500,
            "alpha": 0.3,
            "facecolor": "none",
            "linewidths": 3,
        },
        kwargs_inner={
            "zorder": 1000,
            "rasterized": False,
            "s": 50,
            "marker": ".",
            "color": "k",
            "linewidths": 4.0,
        },
    )

    fig.canvas.draw()
    annotated_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


def visualize_calibration_frames_sequential(
    calibration_df: pd.DataFrame,
    video_path: str,
    ctx,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Produce annotated frames for every row in *calibration_df*.

    Reads the video once sequentially for efficiency.

    Args:
        calibration_df: Output of :func:`calibrate_video_iterative`.
        video_path: Path to the source video.
        ctx: Initialised TVCalibContext.
        max_frames: If set, only visualise the first *max_frames* rows.

    Returns:
        List of annotated BGR frames.
    """
    df_vis = (
        calibration_df.head(max_frames).copy()
        if max_frames is not None
        else calibration_df.copy()
    )
    df_vis["frame_index"] = df_vis["image_id"].apply(frame_index_from_image_id)
    df_vis = df_vis.sort_values("frame_index")

    needed_indices = set(df_vis["frame_index"].tolist())
    sample_by_frame = {
        int(row.frame_index): row for _, row in df_vis.iterrows()
    }

    annotated_frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    max_needed = max(needed_indices) if needed_indices else 0

    pbar = tqdm(
        total=len(df_vis),
        desc="Visualising calibration",
        smoothing=0.1,
    )

    current_idx = 0
    while current_idx <= max_needed:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if current_idx in needed_indices:
            sample = sample_by_frame[current_idx]
            annotated = visualize_calibration_frame(frame_bgr, sample, ctx)
            annotated_frames.append(annotated)
            pbar.update(1)

        current_idx += 1

    pbar.close()
    cap.release()

    return annotated_frames
