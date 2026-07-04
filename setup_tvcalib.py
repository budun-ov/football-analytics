"""
TVCalib Setup Script

Clones the TVCalib repository, installs it as a local package, applies
compatibility patches (Python 3.10+ collections and torch._six removal),
and copies the pitch template to assets/.

Run this script once before using the calibration features:

    python setup_tvcalib.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TVCALIB_REPO = "https://github.com/MM4SPA/tvcalib"
SN_SEGMENTATION_REPO = "https://github.com/jtheiner/sn-calibration-segmentation"
TARGET_DIR = Path("tvcalib")
SN_SEG_DIR = TARGET_DIR / "sn_segmentation"
ASSETS_DIR = Path("assets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], **kwargs) -> None:
    """Run *cmd* and raise on failure."""
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _replace_in_file(path: str | Path, old: str, new: str) -> bool:
    """Replace first occurrence of *old* with *new* in *path*.

    Returns True if a replacement was made.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    if old not in text:
        return False

    p.write_text(text.replace(old, new), encoding="utf-8")
    return True


def _sub_in_file(path: str | Path, pattern: str, replacement: str) -> int:
    """Apply regex substitution in *path*.  Returns number of replacements."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, replacement, text)
    if n > 0:
        p.write_text(new_text, encoding="utf-8")
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 60)
    print("  TVCalib Setup")
    print("=" * 60)

    # ── 1. Clone TVCalib ────────────────────────────────────────────────
    if not TARGET_DIR.exists():
        print(f"\n[1/5] Cloning TVCalib from {TVCALIB_REPO} ...")
        _run(["git", "clone", TVCALIB_REPO, str(TARGET_DIR)])
    else:
        print(f"\n[1/5] TVCalib directory already exists — pulling latest ...")
        _run(["git", "-C", str(TARGET_DIR), "pull"])

    # ── 2. Replace sn_segmentation with the full repo ───────────────────
    if SN_SEG_DIR.exists():
        print(f"\n[2/5] Removing old {SN_SEG_DIR} ...")
        shutil.rmtree(SN_SEG_DIR)

    print(f"\n[2/5] Cloning sn-calibration-segmentation into {SN_SEG_DIR} ...")
    _run(["git", "clone", SN_SEGMENTATION_REPO, str(SN_SEG_DIR)])

    # Remove the nested .git to keep tvcalib as the single repository root
    nested_git = SN_SEG_DIR / ".git"
    if nested_git.exists():
        shutil.rmtree(nested_git)
        print("  Removed nested .git from sn_segmentation.")

    if not (SN_SEG_DIR / "src" / "segmentation").exists():
        print(
            "\n[ERROR] sn_segmentation folder structure is unexpected.\n"
            "        Check that https://github.com/jtheiner/sn-calibration-segmentation\n"
            "        was cloned correctly."
        )
        sys.exit(1)

    # ── 3. Install TVCalib as a local package ────────────────────────────
    print("\n[3/5] Installing TVCalib package ...")
    _run([sys.executable, "-m", "pip", "install", "-e", str(TARGET_DIR)])

    # ── 4. Apply compatibility patches ──────────────────────────────────
    print("\n[4/5] Applying source code patches ...")

    base = TARGET_DIR / "tvcalib"
    patched_any = False

    # Patch 1: torch._six.string_classes (removed in PyTorch ≥ 2.0)
    file_sncalib = base / "sncalib_dataset.py"
    if file_sncalib.exists():
        replaced = _replace_in_file(
            file_sncalib,
            "from torch._six import string_classes",
            "import collections.abc\nstring_classes = str",
        )
        if replaced:
            print("  ✓  torch._six patch applied.")
            patched_any = True

        # Patch 2: collections.Iterable (moved to collections.abc in Python 3.10)
        n = _sub_in_file(
            file_sncalib,
            r"from collections import Iterable",
            "from collections.abc import Iterable",
        )
        if n:
            print(f"  ✓  collections.Iterable patch applied ({n} replacement(s)).")
            patched_any = True

    if not patched_any:
        print("  No patches needed — files are already up to date.")

    # ── 5. Copy pitch template to assets/ ──────────────────────────────
    print("\n[5/5] Copying pitch template ...")
    ASSETS_DIR.mkdir(exist_ok=True)
    src_tmpl = TARGET_DIR / "template_pitch_t.png"
    dst_tmpl = ASSETS_DIR / "template_pitch_t.png"

    if src_tmpl.exists():
        shutil.copy2(src_tmpl, dst_tmpl)
        print(f"  ✓  template_pitch_t.png → {dst_tmpl}")
    else:
        print(
            f"  ⚠  template_pitch_t.png not found in {TARGET_DIR}/\n"
            f"     Place it manually in assets/ before running the radar/Voronoi modes."
        )

    # ── Done ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TVCalib setup complete.")
    print()
    print("  Next steps:")
    print("    1. Place your YOLO model weights:    models/detection/best.pt")
    print("    2. Place the segmentation model:     models/segmentation/train_59.pt")
    print("    3. Run the pipeline:                 python main.py")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
