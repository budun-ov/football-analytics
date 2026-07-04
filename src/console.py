"""
Console interface — collects user inputs before the pipeline starts.

All prompts show a default (from config.yaml); pressing Enter keeps it.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# Valid video extensions accepted by the pipeline
_VALID_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prompt_with_default(prompt: str, default: str) -> str:
    """Show *prompt* with *default* and return what the user typed (or default)."""
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer if answer else default


def get_video_path(default: str) -> str:
    """
    Interactively ask for the input video path.

    Loops until the user provides a path to an existing file.
    """
    while True:
        path = prompt_with_default("Enter input video path", default)
        path = path.strip("\"'")

        if not path:
            print("  ✗  Path cannot be empty.")
            continue

        if not os.path.exists(path):
            print(f"  ✗  File not found: {path}")
            continue

        if not os.path.isfile(path):
            print(f"  ✗  Not a file: {path}")
            continue

        ext = Path(path).suffix.lower()
        if ext not in _VALID_VIDEO_EXT:
            print(f"  ⚠  Extension '{ext}' is not a common video format.")
            confirm = input("     Continue anyway? (y/n): ").strip().lower()
            if confirm != "y":
                continue

        print(f"  ✓  Video path: {path}")
        return path


def get_output_dir(default: str) -> str:
    """Ask for the output directory, creating it if it does not exist."""
    path = prompt_with_default("Enter output directory", default)
    path = path.strip("\"'")

    os.makedirs(path, exist_ok=True)
    print(f"  ✓  Output directory: {path}")
    return path


def get_run_mode() -> int:
    """
    Ask the user to pick a run mode.

    Returns:
        1 — Full pipeline
        2 — Default (annotated video only)
        3 — Custom
    """
    print()
    print("╔══════════════════════════════════════════╗")
    print("║              Select Run Mode             ║")
    print("╠══════════════════════════════════════════╣")
    print("║  1  Full pipeline (all outputs)          ║")
    print("║  2  Default mode  (annotated video only) ║")
    print("║  3  Custom mode   (choose outputs)       ║")
    print("╚══════════════════════════════════════════╝")

    while True:
        choice = input("Mode [2]: ").strip() or "2"
        if choice in ("1", "2", "3"):
            return int(choice)
        print("  ✗  Please enter 1, 2, or 3.")


def get_custom_options() -> SimpleNamespace:
    """
    Interactively collect options for custom mode.

    Returns a SimpleNamespace with boolean flags.
    """
    print()
    print("─── Custom Mode — choose what to generate ───")

    def yes_no(prompt: str, default: bool = True) -> bool:
        d = "y" if default else "n"
        answer = input(f"  {prompt} (y/n) [{d}]: ").strip().lower() or d
        return answer == "y"

    return SimpleNamespace(
        calibration_video=yes_no("Generate calibration visualisation video?"),
        radar_video=yes_no("Generate radar (minimap) video?"),
        voronoi_video=yes_no("Generate Voronoi diagram video?"),
        show_ball_possession=yes_no("Show ball possession overlay on main video?"),
        show_speed=yes_no(
            "Show player speed overlay on main video? "
            "(requires camera calibration — will be run automatically if enabled)"
        ),
    )


def print_banner() -> None:
    """Print a startup banner."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        Football Analytics Pipeline v1.0          ║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def print_section(title: str) -> None:
    """Print a section divider."""
    width = 56
    print()
    print("─" * width)
    print(f"  {title}")
    print("─" * width)


def confirm_settings(
    video_path: str,
    output_dir: str,
    mode: int,
    mode_label: str,
) -> bool:
    """Show a summary of chosen settings and ask for confirmation."""
    print()
    print("  Settings summary")
    print(f"    Input video : {video_path}")
    print(f"    Output dir  : {output_dir}")
    print(f"    Mode        : {mode} — {mode_label}")
    answer = input("\n  Start pipeline? (y/n) [y]: ").strip().lower() or "y"
    return answer == "y"
