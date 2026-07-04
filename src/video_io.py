"""
Video I/O utilities.

Covers:
- Reading frames from a video file (batched or frame-by-frame)
- Writing annotated frames back to disk
- Saving frame batches to temporary disk locations
"""

from __future__ import annotations

import shutil
import re
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def get_video_info(video_path: Union[str, Path]) -> sv.VideoInfo:
    """Return a supervision VideoInfo object for *video_path*."""
    return sv.VideoInfo.from_video_path(str(video_path))


def get_video_fps(video_path: Union[str, Path]) -> float:
    """Return the FPS of *video_path*."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def load_video_frames(video_path: Union[str, Path]) -> Generator:
    """
    Return a supervision generator that yields BGR frames one by one.

    The generator is lazy — it does not load the whole video into RAM.
    """
    return sv.get_video_frames_generator(str(video_path))


def read_first_n_frames(
    video_path: Union[str, Path],
    n: int = 32,
) -> Tuple[List[np.ndarray], float]:
    """Read the first *n* frames of *video_path*."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: List[np.ndarray] = []

    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def iter_video_frame_batches(
    video_path: Union[str, Path],
    batch_size: int = 128,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[List[int], List[np.ndarray]]]:
    """
    Yield ``(frame_indices, frames_bgr)`` tuples of length *batch_size*.

    The final batch may be shorter if the video has fewer remaining frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    indices: List[int] = []
    frames: List[np.ndarray] = []
    current_idx = start_frame
    read_count = 0

    while True:
        if max_frames is not None and read_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        indices.append(current_idx)
        frames.append(frame)
        current_idx += 1
        read_count += 1

        if len(frames) >= batch_size:
            yield indices, frames
            indices = []
            frames = []

    if frames:
        yield indices, frames

    cap.release()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_frames_to_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: float,
    codec: str = "mp4v",
) -> str:
    """
    Write *frames* (BGR) to *output_path* at *fps*.

    Returns the absolute path of the saved file.

    Raises:
        ValueError: If *frames* is empty.
        RuntimeError: If the VideoWriter cannot be opened.
    """
    if not frames:
        raise ValueError("No frames to write.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {output_path}")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    print(f"  ✓  Saved: {output_path}  ({len(frames)} frames, {fps:.1f} fps)")
    return str(output_path.resolve())


def save_video_from_generator(
    frame_gen,
    output_path: Union[str, Path],
    fps: float,
    codec: str = "mp4v",
) -> str:
    """
    Stream frames from *frame_gen* directly to *output_path*.

    Useful when frames are produced lazily to avoid holding all of them
    in memory at once.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    h = w = 0
    count = 0

    for frame in frame_gen:
        if frame is None:
            continue

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            if not writer.isOpened():
                # Fallback to XVID
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {output_path}")

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)
        count += 1

    if writer is not None:
        writer.release()

    if count == 0:
        print(f"  ⚠  No frames written to: {output_path}")
    else:
        print(f"  ✓  Saved: {output_path}  ({count} frames, {fps:.1f} fps)")

    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# Temporary disk helpers (used by calibration pipeline)
# ---------------------------------------------------------------------------

def save_frame_batch_to_disk(
    frames_bgr: List[np.ndarray],
    frame_indices: List[int],
    batch_dir: Union[str, Path],
) -> List[str]:
    """
    Write a list of BGR frames to *batch_dir* using the naming convention
    ``frame_000123.png`` expected by the TVCalib InferenceDataset.

    Returns the list of image filenames (not full paths).
    """
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    image_ids: List[str] = []
    for frame, idx in zip(frames_bgr, frame_indices):
        image_id = f"frame_{idx:06d}.png"
        ok = cv2.imwrite(str(batch_dir / image_id), frame)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {batch_dir / image_id}")
        image_ids.append(image_id)

    return image_ids


def prepare_temp_root(temp_root: Union[str, Path], clean: bool = True) -> Path:
    """
    Create (or recreate) a temporary directory.

    Args:
        temp_root: Path to the temporary directory.
        clean: If *True*, remove the directory first if it already exists.
    """
    temp_root = Path(temp_root)

    if clean and temp_root.exists():
        shutil.rmtree(temp_root)

    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


# ---------------------------------------------------------------------------
# Frame-index helpers
# ---------------------------------------------------------------------------

def frame_index_from_image_id(image_id: str) -> int:
    """
    Extract the integer frame number from a filename like ``frame_000123.png``.

    Raises:
        ValueError: If the format is not recognised.
    """
    match = re.search(r"frame_(\d+)", str(image_id))
    if match is None:
        raise ValueError(f"Cannot parse frame index from: {image_id!r}")
    return int(match.group(1))
