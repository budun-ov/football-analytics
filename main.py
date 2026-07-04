"""
Football Analytics Pipeline — main entry point.

Usage
-----
    python main.py

The program interactively collects:
  - Input video path
  - Output directory
  - Run mode (1 = Full, 2 = Default, 3 = Custom)
  - Custom mode options (when mode 3 is chosen)

All defaults come from config.yaml; pressing Enter at any prompt
keeps the current default.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so 'src.*' imports work whether
# main.py is run from inside or outside the project directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports (after path fixup)
# ---------------------------------------------------------------------------

from src.config_loader import load_config
from src.console import (
    print_banner,
    print_section,
    get_video_path,
    get_output_dir,
    get_run_mode,
    get_custom_options,
    confirm_settings,
)
from src.pipeline.runner import (
    run_mode_1_full_pipeline,
    run_mode_2_default,
    run_mode_3_custom,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_environment(cfg) -> None:
    """
    Check that required files and directories exist before starting.

    Raises SystemExit with a descriptive message on failure.
    """
    errors = []

    # Detection model
    det_path = Path(cfg.models.detection)
    if not det_path.exists():
        errors.append(
            f"  ✗ Detection model not found: {det_path}\n"
            f"    → Place your YOLO weights at models/detection/best.pt\n"
            f"    → Or update 'models.detection' in config.yaml"
        )

    # Segmentation model (only needed for calibration in modes 1 and 3)
    seg_path = Path(cfg.models.segmentation)
    if not seg_path.exists():
        errors.append(
            f"  ⚠ Segmentation model not found: {seg_path}\n"
            f"    → Required for calibration, radar, and Voronoi outputs.\n"
            f"    → Place train_59.pt at models/segmentation/train_59.pt\n"
            f"    → Or update 'models.segmentation' in config.yaml"
        )

    # TVCalib repo (only needed for calibration)
    tvcalib_dir = Path(cfg.tvcalib.repo_dir)
    if not tvcalib_dir.exists():
        errors.append(
            f"  ⚠ TVCalib repository not found at: {tvcalib_dir}\n"
            f"    → Run:  python setup_tvcalib.py"
        )

    # Pitch template (only needed for radar / Voronoi)
    tmpl_path = Path(cfg.tvcalib.template_path)
    if not tmpl_path.exists():
        errors.append(
            f"  ⚠ Pitch template not found: {tmpl_path}\n"
            f"    → Copy template_pitch_t.png to assets/\n"
            f"    → It is included in the TVCalib repo root after running setup."
        )

    # Hard failure — detection model is always required
    hard_errors = [e for e in errors if "Detection model" in e]
    if hard_errors:
        print("\n[ERROR] Cannot start — missing required files:\n")
        for err in hard_errors:
            print(err)
        sys.exit(1)

    # Soft warnings — calibration-related assets
    soft_errors = [e for e in errors if e not in hard_errors]
    if soft_errors:
        print("\n[WARNING] Some optional assets are missing:\n")
        for err in soft_errors:
            print(err)
        print(
            "\n  Modes that require calibration (Mode 1, and Mode 3 with radar/Voronoi)\n"
            "  will fail if the segmentation model or TVCalib repo is absent.\n"
            "  Mode 2 (default annotated video) does not require them.\n"
        )


# ---------------------------------------------------------------------------
# Mode labels
# ---------------------------------------------------------------------------

_MODE_LABELS = {
    1: "Full pipeline",
    2: "Default — annotated video only",
    3: "Custom",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()

    # ── Load configuration ──────────────────────────────────────────────
    print_section("Loading configuration")
    try:
        cfg = load_config(PROJECT_ROOT / "config.yaml")
        print("  ✓  config.yaml loaded.")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Validate environment ────────────────────────────────────────────
    _validate_environment(cfg)

    # ── Collect user inputs ─────────────────────────────────────────────
    print_section("Input / Output Settings")
    video_path = get_video_path(cfg.video.input_path)
    output_dir = get_output_dir(cfg.video.output_dir)

    mode = get_run_mode()
    custom_options = None

    if mode == 3:
        custom_options = get_custom_options()

    mode_label = _MODE_LABELS[mode]

    if not confirm_settings(video_path, output_dir, mode, mode_label):
        print("\n  Aborted.")
        sys.exit(0)

    # ── Run selected mode ───────────────────────────────────────────────
    print_section(f"Running Mode {mode}: {mode_label}")

    try:
        if mode == 1:
            run_mode_1_full_pipeline(cfg, video_path, output_dir)
        elif mode == 2:
            run_mode_2_default(cfg, video_path, output_dir)
        elif mode == 3:
            run_mode_3_custom(cfg, video_path, output_dir, custom_options)

    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        sys.exit(0)

    # ── Done ────────────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("  Pipeline finished. Output files are in:", output_dir)
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
