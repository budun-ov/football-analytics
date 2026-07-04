"""
Pipeline runner — ties together all pipeline stages for each run mode.

Modes
-----
1  Full pipeline     → calibration video + annotated video + radar + Voronoi
2  Default mode      → annotated video only
3  Custom mode       → user-selected combination of outputs / annotations
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import supervision as sv

from src.config_loader import load_config
from src.console import print_section
from src.detection.tracker import Tracker
from src.team.classifier import TeamClassifier, TeamClassifierConfig
from src.team.ball_assigner import PlayerBallAssigner
from src.calibration.tvcalib_runner import TVCalibConfig, TVCalibContext, calibrate_video_iterative
from src.calibration.view_transformer import ViewTransformer
from src.calibration.visualization import visualize_calibration_frames_sequential
from src.kinematics.speed import calculate_kinematics
from src.visualization.annotation import draw_annotations, draw_speed
from src.visualization.radar import RadarMinimap
from src.visualization.voronoi import VoronoiMinimap
from src.video_io import (
    load_video_frames,
    get_video_info,
    save_frames_to_video,
    save_video_from_generator,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_bytetrack_params(cfg, fps: float) -> dict:
    """Resolve ByteTrack parameters, substituting FPS where required."""
    bt = cfg.bytetrack
    return {
        "track_activation_threshold": bt.track_activation_threshold,
        "lost_track_buffer": int(bt.lost_track_buffer_seconds * fps),
        "minimum_matching_threshold": bt.minimum_matching_threshold,
        "minimum_consecutive_frames": bt.minimum_consecutive_frames,
        "frame_rate": fps,
    }


def _run_tracking(cfg, video_path: str, fps: float, total_frames: int) -> dict:
    """Initialise the tracker and return the tracks dict."""
    print_section("Stage 1/5 — Object Tracking")

    bytetrack_params = _build_bytetrack_params(cfg, fps)

    tracker = Tracker(
        model_path=cfg.models.detection,
        bytetrack_params=bytetrack_params,
    )

    stub_path = (
        cfg.stubs.stub_path if getattr(cfg.stubs, "use_stub", True) else None
    )

    tracks = tracker.get_object_tracks(
        frame_generator=load_video_frames(video_path),
        batch_size=cfg.detection.batch_size,
        total_frames=total_frames,
        conf=cfg.detection.confidence,
        verbose=cfg.detection.verbose,
        stream=cfg.detection.stream,
        stub_path=stub_path,
    )

    return tracks


def _run_team_assignment(
    cfg,
    video_path: str,
    tracks: dict,
    with_ball_control: bool,
) -> Optional[np.ndarray]:
    """Classify player teams; optionally track ball possession."""
    print_section("Stage 2/5 — Team Assignment")

    tc_cfg = cfg.team_classifier
    team_classifier = TeamClassifier(
        config=TeamClassifierConfig(
            model_name=tc_cfg.model_name,
            batch_size=tc_cfg.batch_size,
            n_clusters=tc_cfg.n_clusters,
            warmup_samples=tc_cfg.warmup_samples,
            buffer_frames=tc_cfg.buffer_frames,
        )
    )

    ball_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_id, frame in enumerate(load_video_frames(video_path)):
        players = tracks["players"][frame_id]
        goalkeepers = tracks["goalkeepers"][frame_id]

        team_classifier.process_frame(frame, players, goalkeepers)

        if with_ball_control:
            ball_bbox_data = tracks["ball"][frame_id]
            if ball_bbox_data:
                ball_bbox = next(iter(ball_bbox_data.values()))["bbox"]
                assigned = ball_assigner.assign_ball_to_player(players, ball_bbox)

                if assigned != -1:
                    tracks["players"][frame_id][assigned]["has_ball"] = True
                    try:
                        team_ball_control.append(
                            tracks["players"][frame_id][assigned]["team_id"]
                        )
                    except KeyError:
                        if team_ball_control:
                            team_ball_control.append(team_ball_control[-1])
                elif team_ball_control:
                    team_ball_control.append(team_ball_control[-1])

    return np.array(team_ball_control) if with_ball_control else None


def _run_calibration(cfg, video_path: str) -> tuple:
    """Run TVCalib calibration and return (calibration_df, ctx)."""
    print_section("Stage 3/5 — Camera Calibration")

    calib_cfg = cfg.calibration
    tvcalib_config = TVCalibConfig(
        video_path=video_path,
        temp_root=calib_cfg.temp_root,
        checkpoint=cfg.models.segmentation,
        image_width=calib_cfg.image_width,
        image_height=calib_cfg.image_height,
        max_frames=calib_cfg.max_frames,
        frame_batch_size=calib_cfg.frame_batch_size,
        batch_size_seg=calib_cfg.batch_size_seg,
        batch_size_calib=calib_cfg.batch_size_calib,
        nworkers=calib_cfg.nworkers,
        optim_steps=calib_cfg.optim_steps,
        lens_dist=calib_cfg.lens_dist,
        gpu=calib_cfg.gpu,
        line_skeleton_radius=calib_cfg.line_skeleton_radius,
        extremities_maxdist=calib_cfg.extremities_maxdist,
        extremities_width=calib_cfg.extremities_width,
        extremities_height=calib_cfg.extremities_height,
        num_points_lines=calib_cfg.num_points_lines,
        num_points_circles=calib_cfg.num_points_circles,
    )

    ctx = TVCalibContext(tvcalib_config)
    calibration_df = calibrate_video_iterative(tvcalib_config, ctx)
    return calibration_df, ctx


def _run_coordinate_transform(cfg, tracks: dict, calibration_df, fps: float) -> dict:
    """Apply homography to get field-plane coordinates and compute speed."""
    print_section("Stage 4/5 — Coordinate Transform & Speed")

    view_transformer = ViewTransformer(calibration_df)
    tracks = view_transformer.transform_to_meters(tracks)
    tracks = calculate_kinematics(tracks, fps)

    return tracks


def _has_speed_data(tracks: dict) -> bool:
    """Check whether any track entry already has a 'speed' value.

    Speed is only populated after coordinate transform + kinematics
    (i.e. after calibration has run). Modes that skip calibration
    (Mode 2, or Mode 3 without calibration/radar/voronoi selected)
    will not have this data.
    """
    for cat in ("players", "goalkeepers"):
        for frame_data in tracks.get(cat, []):
            for info in frame_data.values():
                if "speed" in info:
                    return True
    return False


def _save_annotated_video(
    cfg,
    video_path: str,
    tracks: dict,
    team_ball_control,
    output_dir: str,
    show_ball_possession: bool,
    show_speed: bool,
) -> None:
    """Render and save the main annotated video."""
    output_path = Path(output_dir) / cfg.output.annotated_video

    if show_speed and not _has_speed_data(tracks):
        print(
            "  ⚠  show_speed is enabled but no speed data is available "
            "(calibration was not run in this mode) — skipping speed overlay."
        )
        show_speed = False

    frame_gen = draw_annotations(
        load_video_frames(video_path),
        tracks,
        team_ball_control,
        show_ball_possession=show_ball_possession,
    )

    if show_speed:
        frames = list(frame_gen)
        frames = draw_speed(frames, tracks)
        video_info = get_video_info(video_path)
        save_frames_to_video(frames, output_path, video_info.fps)
    else:
        video_info = get_video_info(video_path)
        save_video_from_generator(frame_gen, output_path, video_info.fps)


def _save_calibration_video(cfg, video_path: str, calibration_df, ctx, output_dir: str) -> None:
    """Render and save the calibration reprojection video."""
    frames = visualize_calibration_frames_sequential(
        calibration_df=calibration_df,
        video_path=video_path,
        ctx=ctx,
    )

    video_info = get_video_info(video_path)
    save_frames_to_video(
        frames,
        Path(output_dir) / cfg.output.calibration_video,
        video_info.fps,
    )


def _save_radar_video(cfg, tracks: dict, video_path: str, output_dir: str) -> None:
    """Render and save the radar minimap video."""
    radar_cfg = cfg.radar
    video_info = get_video_info(video_path)

    radar = RadarMinimap(
        template_path=cfg.tvcalib.template_path,
        output_path=str(Path(output_dir) / cfg.output.radar_video),
        fps=video_info.fps,
        radar_w=radar_cfg.width,
        radar_h=radar_cfg.height,
        pitch_length=radar_cfg.pitch_length,
        pitch_width=radar_cfg.pitch_width,
        show_speed=radar_cfg.show_speed,
        show_id=radar_cfg.show_id,
    )
    radar.build_from_tracks(tracks)


def _save_voronoi_video(cfg, tracks: dict, video_path: str, output_dir: str) -> None:
    """Render and save the Voronoi diagram video."""
    vor_cfg = cfg.voronoi
    video_info = get_video_info(video_path)

    voronoi = VoronoiMinimap(
        template_path=cfg.tvcalib.template_path,
        output_path=str(Path(output_dir) / cfg.output.voronoi_video),
        fps=video_info.fps,
        alpha=vor_cfg.alpha,
        grid_step=vor_cfg.grid_step,
        draw_boundaries=vor_cfg.draw_boundaries,
        include_goalkeepers=vor_cfg.include_goalkeepers,
        show_id=vor_cfg.show_id,
        show_speed=vor_cfg.show_speed,
    )
    voronoi.build_from_tracks(tracks)


# ---------------------------------------------------------------------------
# Public entry points for each mode
# ---------------------------------------------------------------------------

def run_mode_1_full_pipeline(
    cfg,
    video_path: str,
    output_dir: str,
) -> None:
    """
    Mode 1 — Full pipeline.

    Produces:
      - Main annotated video (all annotations)
      - Calibration reprojection video
      - Radar minimap video
      - Voronoi diagram video
    """
    video_info = get_video_info(video_path)
    fps = video_info.fps
    total_frames = video_info.total_frames

    tracks = _run_tracking(cfg, video_path, fps, total_frames)
    team_ball_control = _run_team_assignment(
        cfg, video_path, tracks, with_ball_control=True
    )
    calibration_df, ctx = _run_calibration(cfg, video_path)
    tracks = _run_coordinate_transform(cfg, tracks, calibration_df, fps)

    print_section("Stage 5/5 — Rendering Output Videos")

    _save_annotated_video(
        cfg, video_path, tracks, team_ball_control, output_dir,
        show_ball_possession=True, show_speed=True,
    )
    _save_calibration_video(cfg, video_path, calibration_df, ctx, output_dir)
    _save_radar_video(cfg, tracks, video_path, output_dir)
    _save_voronoi_video(cfg, tracks, video_path, output_dir)


def run_mode_2_default(
    cfg,
    video_path: str,
    output_dir: str,
) -> None:
    """
    Mode 2 — Default mode.

    Produces:
      - Main annotated video with all standard annotations.

    Skips calibration, radar and Voronoi to save time.
    """
    video_info = get_video_info(video_path)
    fps = video_info.fps
    total_frames = video_info.total_frames

    tracks = _run_tracking(cfg, video_path, fps, total_frames)
    team_ball_control = _run_team_assignment(
        cfg, video_path, tracks, with_ball_control=True
    )

    print_section("Stage 3/3 — Rendering Annotated Video")
    _save_annotated_video(
        cfg, video_path, tracks, team_ball_control, output_dir,
        show_ball_possession=cfg.visualization.show_ball_possession,
        show_speed=cfg.visualization.show_speed,
    )


def run_mode_3_custom(
    cfg,
    video_path: str,
    output_dir: str,
    options: SimpleNamespace,
) -> None:
    """
    Mode 3 — Custom mode.

    The *options* namespace (from :func:`src.console.get_custom_options`)
    determines which pipeline stages and output videos are activated.
    """
    video_info = get_video_info(video_path)
    fps = video_info.fps
    total_frames = video_info.total_frames

    # Speed and radar/voronoi both depend on position_field (metres),
    # which is only populated after camera calibration. Calibration video
    # itself is also a direct calibration output.
    need_calibration = (
        options.calibration_video
        or options.radar_video
        or options.voronoi_video
        or options.show_speed
    )

    tracks = _run_tracking(cfg, video_path, fps, total_frames)
    team_ball_control = _run_team_assignment(
        cfg, video_path, tracks,
        with_ball_control=options.show_ball_possession,
    )

    calibration_df = ctx = None

    if need_calibration:
        calibration_df, ctx = _run_calibration(cfg, video_path)
        tracks = _run_coordinate_transform(cfg, tracks, calibration_df, fps)

    print_section("Rendering Selected Output Videos")

    _save_annotated_video(
        cfg, video_path, tracks, team_ball_control, output_dir,
        show_ball_possession=options.show_ball_possession,
        show_speed=options.show_speed,
    )

    if options.calibration_video and calibration_df is not None:
        _save_calibration_video(cfg, video_path, calibration_df, ctx, output_dir)

    if options.radar_video:
        _save_radar_video(cfg, tracks, video_path, output_dir)

    if options.voronoi_video:
        _save_voronoi_video(cfg, tracks, video_path, output_dir)
