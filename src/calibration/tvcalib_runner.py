"""
TVCalib runner — wraps the TVCalib library for field-line segmentation
and camera calibration.

Requires the TVCalib repository to be cloned and installed locally.
Run ``python setup_tvcalib.py`` before using this module.

Reference:
    TVCalib — https://github.com/MM4SPA/tvcalib
"""

from __future__ import annotations

import shutil
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm

from src.video_io import (
    iter_video_frame_batches,
    save_frame_batch_to_disk,
    prepare_temp_root,
    frame_index_from_image_id,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TVCalibConfig:
    """All parameters for the TVCalib segmentation + calibration pipeline."""

    video_path: str
    temp_root: str = ".temp_calib"
    checkpoint: str = "models/segmentation/train_59.pt"
    image_width: int = 1920
    image_height: int = 1080
    max_frames: Optional[int] = None
    frame_batch_size: int = 128
    batch_size_seg: int = 8
    batch_size_calib: int = 64
    nworkers: int = 4
    optim_steps: int = 3000
    lens_dist: bool = False
    gpu: bool = True
    line_skeleton_radius: int = 4
    extremities_maxdist: int = 30
    extremities_width: int = 455
    extremities_height: int = 256
    num_points_lines: int = 4
    num_points_circles: int = 8


# ---------------------------------------------------------------------------
# Context — lazily initialises TVCalib components once
# ---------------------------------------------------------------------------

class TVCalibContext:
    """
    Holds all TVCalib models and helper functions.

    Initialised once per run; shared across all video batches to avoid
    repeated model loading.
    """

    def __init__(self, config: TVCalibConfig) -> None:
        self.config = config
        self.device = "cuda" if config.gpu and torch.cuda.is_available() else "cpu"
        print(f"  TVCalib device: {self.device}")

        # Allow safe deserialization of argparse.Namespace (required by TVCalib)
        torch.serialization.add_safe_globals([Namespace])

        # TVCalib imports (available after setup_tvcalib.py is run)
        try:
            from tvcalib.module import TVCalibModule
            from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
            from tvcalib.utils.objects_3d import (
                SoccerPitchLineCircleSegments,
                SoccerPitchSNCircleCentralSplit,
            )
            from tvcalib.inference import InferenceSegmentationModel
            from sn_segmentation.src.custom_extremities import (
                generate_class_synthesis,
                get_line_extremities,
            )
        except ImportError as exc:
            raise ImportError(
                "TVCalib library not found.\n"
                "Please run:  python setup_tvcalib.py\n"
                f"Original error: {exc}"
            ) from exc

        self.object3d = SoccerPitchLineCircleSegments(
            device=self.device,
            base_field=SoccerPitchSNCircleCentralSplit(),
        )
        self.object3dcpu = SoccerPitchLineCircleSegments(
            device="cpu",
            base_field=SoccerPitchSNCircleCentralSplit(),
        )

        self.fn_generate_class_synthesis = partial(
            generate_class_synthesis,
            radius=config.line_skeleton_radius,
        )
        self.fn_get_line_extremities = partial(
            get_line_extremities,
            maxdist=config.extremities_maxdist,
            width=config.extremities_width,
            height=config.extremities_height,
            num_points_lines=config.num_points_lines,
            num_points_circles=config.num_points_circles,
        )

        if not Path(config.checkpoint).exists():
            raise FileNotFoundError(
                f"Segmentation model not found: {config.checkpoint}\n"
                "Place train_59.pt at models/segmentation/train_59.pt "
                "(or update 'models.segmentation' in config.yaml)."
            )

        print(f"  Loading segmentation model: {config.checkpoint}")
        self.model_seg = InferenceSegmentationModel(config.checkpoint, self.device)

        self.model_calib = TVCalibModule(
            self.object3d,
            get_cam_distr(1.96, config.batch_size_calib, 1),
            (
                get_dist_distr(config.batch_size_calib, 1)
                if config.lens_dist
                else None
            ),
            (config.image_height, config.image_width),
            config.optim_steps,
            self.device,
            log_per_step=False,
        )


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def run_segmentation_on_image_dir(
    images_path: str | Path,
    ctx: TVCalibContext,
) -> Tuple[List[str], List[dict]]:
    """
    Run TVCalib segmentation on every image inside *images_path*.

    Returns:
        (image_ids, keypoints_raw) — parallel lists.
    """
    from tvcalib.inference import InferenceDatasetSegmentation
    from tvcalib.sncalib_dataset import custom_list_collate

    config = ctx.config
    images_path = Path(images_path)

    dataset_seg = InferenceDatasetSegmentation(
        images_path,
        config.image_width,
        config.image_height,
    )

    if len(dataset_seg) == 0:
        return [], []

    loader = torch.utils.data.DataLoader(
        dataset_seg,
        batch_size=config.batch_size_seg,
        num_workers=config.nworkers,
        shuffle=False,
        collate_fn=custom_list_collate,
    )

    image_ids: List[str] = []
    keypoints_raw: List[dict] = []

    for batch_dict in loader:
        with torch.no_grad():
            sem_lines = ctx.model_seg.inference(
                batch_dict["image"].to(ctx.device)
            )

        sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

        with Pool(config.nworkers) as pool:
            skeletons_batch = pool.map(ctx.fn_generate_class_synthesis, sem_lines)
            keypoints_raw_batch = pool.map(ctx.fn_get_line_extremities, skeletons_batch)

        image_ids.extend(batch_dict["image_id"])
        keypoints_raw.extend(keypoints_raw_batch)

    return image_ids, keypoints_raw


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def run_calibration_from_keypoints(
    image_ids: List[str],
    keypoints_raw: List[dict],
    ctx: TVCalibContext,
) -> pd.DataFrame:
    """
    Run TVCalib camera calibration from pre-extracted keypoints.

    Returns:
        DataFrame with calibration parameters indexed by image_id.
    """
    from tvcalib.inference import InferenceDatasetCalibration
    from tvcalib.sncalib_dataset import custom_list_collate
    from tvcalib.utils.io import detach_dict, tensor2list

    config = ctx.config

    if not image_ids:
        return pd.DataFrame()

    dataset_calib = InferenceDatasetCalibration(
        keypoints_raw,
        config.image_width,
        config.image_height,
        ctx.object3d,
    )

    loader = torch.utils.data.DataLoader(
        dataset_calib,
        batch_size=config.batch_size_calib,
        collate_fn=custom_list_collate,
    )

    per_sample_output: dict = defaultdict(list)
    per_sample_output["image_id"] = [[img_id] for img_id in image_ids]

    for x_dict in loader:
        batch_sz = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

        per_sample_loss, cam, _ = ctx.model_calib.self_optim_batch(x_dict)

        output_dict = tensor2list(
            detach_dict(
                {
                    **cam.get_parameters(batch_sz),
                    **per_sample_loss,
                }
            )
        )
        output_dict["points_line"] = x_dict["lines__px_projected_selection_shuffled"]
        output_dict["points_circle"] = x_dict["circles__px_projected_selection_shuffled"]

        for key in output_dict:
            per_sample_output[key].extend(output_dict[key])

    df = pd.DataFrame.from_dict(per_sample_output)

    explode_cols = [
        k for k, v in per_sample_output.items()
        if isinstance(v, list) and len(v) > 0
    ]
    df = df.explode(column=explode_cols)
    df.set_index("image_id", inplace=True, drop=False)

    return df


# ---------------------------------------------------------------------------
# Full iterative pipeline
# ---------------------------------------------------------------------------

def calibrate_video_iterative(
    config: TVCalibConfig,
    ctx: Optional[TVCalibContext] = None,
    clean_temp: bool = True,
) -> pd.DataFrame:
    """
    End-to-end calibration pipeline for a video.

    Processing flow for each outer batch:
        1. Read frames from video → save to temporary disk.
        2. Run TVCalib segmentation on the batch directory.
        3. Run TVCalib calibration on the extracted keypoints.
        4. Clean up temporary files.
        5. Accumulate results.

    Returns:
        DataFrame with calibration data for all processed frames, sorted
        by ``frame_index``.

    Raises:
        RuntimeError: If no frames were successfully calibrated.
    """
    if ctx is None:
        ctx = TVCalibContext(config)

    temp_root = prepare_temp_root(Path(config.temp_root), clean=clean_temp)
    calibration_batches: List[pd.DataFrame] = []

    video_batches = iter_video_frame_batches(
        video_path=config.video_path,
        batch_size=config.frame_batch_size,
        start_frame=0,
        max_frames=config.max_frames,
    )

    for batch_idx, (frame_indices, frames_bgr) in enumerate(
        tqdm(video_batches, desc="Calibrating video batches")
    ):
        batch_dir = temp_root / f"batch_{batch_idx:05d}"

        try:
            save_frame_batch_to_disk(frames_bgr, frame_indices, batch_dir)

            image_ids, keypoints_raw = run_segmentation_on_image_dir(batch_dir, ctx)
            df_batch = run_calibration_from_keypoints(image_ids, keypoints_raw, ctx)

            if len(df_batch) > 0:
                calibration_batches.append(df_batch)

        finally:
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not calibration_batches:
        raise RuntimeError("Calibration produced no results.")

    calibration_df = pd.concat(calibration_batches, axis=0)
    calibration_df["frame_index"] = calibration_df["image_id"].apply(
        frame_index_from_image_id
    )
    calibration_df = calibration_df.sort_values("frame_index")
    calibration_df.set_index("image_id", inplace=True, drop=False)

    print(f"  ✓  Calibration done — {len(calibration_df)} frames processed.")
    return calibration_df
