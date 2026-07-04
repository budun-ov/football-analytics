# Football Analytics Pipeline

A production-ready computer-vision pipeline for automated football (soccer) video analysis.

Given a broadcast video, the pipeline detects players, tracks them across frames, classifies team assignments, calibrates the camera to the pitch plane, computes player kinematics, and renders a rich set of annotated output videos.

---

## Demo

<video controls src="assets/DEMO.MOV" title="Title"></video>

---

## Key Features

- **Object detection** — Custom YOLO model (players, goalkeepers, referees, ball)
- **Multi-object tracking** — ByteTrack via the [supervision](https://github.com/roboflow/supervision) library
- **Team classification** — SigLIP vision embeddings + PCA + K-Means; colour auto-detected from jersey crops
- **Ball possession tracking** — Frame-level assignment of the ball to the nearest player
- **Camera calibration** — TVCalib (field-line segmentation → keypoint extraction → self-optimising calibration)
- **Coordinate transform** — Homography maps pixel bounding boxes to real-world pitch metres
- **Player kinematics** — Smoothed speed estimation (km/h) using Savitzky–Golay filtering
- **Radar minimap** — Top-down 2-D view of player positions rendered on a pitch template
- **Voronoi diagram** — Team territory map with alpha-blended region fill and boundary lines
- **Three run modes** — Full pipeline, fast default mode, or user-customised output selection
- **Console-driven UI** — Interactive prompts with config-file defaults; no code editing required

---

## Pipeline Overview

```
Input video
    │
    ├─ [Detection + ByteTrack]  ─────────────────────────► tracks dict
    │                                                           │
    ├─ [Team Classifier (SigLIP)]  ──────── team_id, team_color added to tracks
    │                                                           │
    ├─ [Ball Possession]  ───────────────── team_ball_control array
    │                                                           │
    ├─ [TVCalib Calibration]  ───────────► calibration_df (homographies per frame)
    │       │                                                   │
    │       └─ Segmentation (train_59.pt) → keypoints          │
    │           → Camera optimisation                          │
    │                                                           │
    ├─ [ViewTransformer]  ──────────────── position_field added (metres)
    │                                                           │
    ├─ [Kinematics]  ───────────────────── speed (km/h) added
    │                                                           │
    └─ [Rendering]
           ├─ annotated.mp4          (main output — always generated)
           ├─ calibration_preview.mp4 (Mode 1 / custom)
           ├─ radar.mp4              (Mode 1 / custom)
           └─ voronoi.mp4            (Mode 1 / custom)
```

---

## Repository Structure

```
football-analytics/
│
├── main.py                   # Entry point — run this
├── setup_tvcalib.py          # One-time TVCalib setup script
├── config.yaml               # All adjustable parameters
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── config_loader.py      # YAML → SimpleNamespace
│   ├── console.py            # Interactive CLI helpers
│   ├── video_io.py           # Frame reading / writing utilities
│   │
│   ├── detection/
│   │   └── tracker.py        # YOLO + ByteTrack
│   │
│   ├── team/
│   │   ├── classifier.py     # SigLIP team classifier
│   │   └── ball_assigner.py  # Ball → nearest player
│   │
│   ├── calibration/
│   │   ├── tvcalib_runner.py # Segmentation + calibration pipeline
│   │   ├── view_transformer.py # Homography → world coords
│   │   └── visualization.py  # Reprojection overlay rendering
│   │
│   ├── kinematics/
│   │   └── speed.py          # Savitzky–Golay speed estimation
│   │
│   ├── visualization/
│   │   ├── annotation.py     # Ellipses, triangles, possession overlay
│   │   ├── radar.py          # RadarMinimap class
│   │   └── voronoi.py        # VoronoiMinimap class
│   │
│   └── pipeline/
│       └── runner.py         # Orchestrates all modes
│
├── models/
│   ├── detection/            # Put downloaded YOLO weights here (ignored by Git)
│   └── segmentation/         # Put downloaded calibration weights here (ignored by Git)
│
├── assets/
│   └── template_pitch_t.png  # Pitch template (copied by setup_tvcalib.py)
│
├── inputs/                   # Drop your input videos here
├── outputs/                  # All generated videos are saved here
├── stubs/                    # Tracking result cache (auto-generated)
└── tvcalib/                  # TVCalib repository (cloned by setup_tvcalib.py)
```

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/budun-ov/football-analytics.git
cd football-analytics
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is listed without a CUDA index; for GPU acceleration install it separately following the [official instructions](https://pytorch.org/get-started/locally/).

### 4. Set up TVCalib

```bash
python setup_tvcalib.py
```

This script:
- Clones the [TVCalib](https://github.com/MM4SPA/tvcalib) repository into `tvcalib/`
- Clones the [sn-calibration-segmentation](https://github.com/jtheiner/sn-calibration-segmentation) sub-repo
- Installs TVCalib as a local editable package
- Applies Python 3.10+ compatibility patches
- Copies `template_pitch_t.png` to `assets/`

---

## Model Weights

Model weights are **not stored in this GitHub repository**. They are hosted separately on Hugging Face to keep the repository lightweight and avoid committing large binary files to Git history.

Required weights:

| Model | Hugging Face repository | Default local path | Purpose |
|------|--------------------------|--------------------|---------|
| Custom YOLO detector | [budun-ov/yolov8_football_analysis](https://huggingface.co/budun-ov/yolov8_football_analysis) | `models/detection/best.pt` | Detects players, goalkeepers, referees and ball |
| Football pitch line calibration model | [budun-ov/football_pitch_lines_calibration](https://huggingface.co/budun-ov/football_pitch_lines_calibration) | `models/segmentation/train_59.pt` | Detects field-line keypoints for camera calibration |

Create the local model directories:

```bash
mkdir -p models/detection models/segmentation
```

Download the weights from Hugging Face and place them as follows:

```text
models/detection/best.pt
models/segmentation/train_59.pt
```

You can also change the default paths in `config.yaml` under the `models:` section.

> Large model files should remain outside Git tracking. The `models/` directory is intended for local runtime files and should be listed in `.gitignore`.

---

## Configuration

The file `config.yaml` in the project root controls all adjustable parameters.

### Where it is

```
football-analytics/config.yaml
```

### Key sections and parameters

| Section | Parameter | Description |
|---------|-----------|-------------|
| `models` | `detection` | Path to the YOLO `.pt` file |
| `models` | `segmentation` | Path to the TVCalib `train_59.pt` file |
| `video` | `input_path` | Default input video (can be overridden at runtime) |
| `video` | `output_dir` | Default output directory |
| `detection` | `confidence` | YOLO confidence threshold (0.0–1.0) |
| `detection` | `batch_size` | Frames per YOLO inference call |
| `bytetrack` | `track_activation_threshold` | Minimum detection score to start a new track |
| `bytetrack` | `lost_track_buffer_seconds` | Seconds to keep a lost track alive |
| `bytetrack` | `minimum_matching_threshold` | IoU matching threshold |
| `bytetrack` | `minimum_consecutive_frames` | Frames needed to confirm a new track |
| `calibration` | `max_frames` | Frames to calibrate (`null` = entire video) |
| `calibration` | `optim_steps` | TVCalib optimisation iterations per batch |
| `calibration` | `batch_size_seg` | Segmentation network batch size |
| `calibration` | `batch_size_calib` | Calibration network batch size |
| `team_classifier` | `warmup_samples` | Crops collected before classifier is fitted |
| `radar` | `width` / `height` | Radar canvas resolution |
| `voronoi` | `alpha` | Region fill transparency (0 = none, 1 = solid) |
| `voronoi` | `grid_step` | Voronoi grid resolution (lower = finer but slower) |
| `visualization` | `show_ball_possession` | Draw ball-possession overlay |
| `visualization` | `show_speed` | Draw player speed labels — requires calibration (see Notes & Limitations) |
| `stubs` | `use_stub` | Cache tracking results to avoid re-running |
| `output` | `annotated_video` | Filename for the main output video |

### Console input vs config.yaml

At runtime, the program prompts for input video path, output directory, and run mode.  Entering a new value overrides the config-file default **for that run only**.  The config.yaml file is not modified.

If you press **Enter** without typing, the config-file default is used.

---

## Usage

```bash
python main.py
```

The program will prompt you for:

1. **Input video path** — defaults to `video.input_path` in config.yaml
2. **Output directory** — defaults to `video.output_dir` in config.yaml
3. **Run mode** — see below

---

## Run Modes

### Mode 1 — Full Pipeline

Runs every stage and produces four output videos:

| Output | Description |
|--------|-------------|
| `annotated.mp4` | Main video with player ellipses, possession overlay, speed labels |
| `calibration_preview.mp4` | Reprojected field lines drawn over source frames |
| `radar.mp4` | Top-down radar minimap |
| `voronoi.mp4` | Top-down Voronoi territory diagram |

### Mode 2 — Default *(fastest)*

Produces only the main annotated video.  Calibration, radar and Voronoi are skipped.

| Output | Description |
|--------|-------------|
| `annotated.mp4` | Players, teams, ball and possession overlay. Speed is not available in this mode because calibration is skipped. |

### Mode 3 — Custom

The program asks which outputs and annotations to enable:

- Calibration visualisation video (y/n)
- Radar minimap video (y/n)
- Voronoi diagram video (y/n)
- Ball possession overlay on main video (y/n)
- Player speed overlay on main video (y/n)

Only the selected stages of the pipeline are executed.

---

## Output Files

All output files are saved to the directory you specify at the run prompt (default: `outputs/`).

| File | Always generated | Description |
|------|-----------------|-------------|
| `annotated.mp4` | ✓ | Main output: players colour-coded by team, ball, possession overlay and, when calibration is enabled, speed labels |
| `calibration_preview.mp4` | Mode 1 / Custom | Reprojected field-line overlay for visual calibration check |
| `radar.mp4` | Mode 1 / Custom | 2-D minimap of player positions |
| `voronoi.mp4` | Mode 1 / Custom | Team territory map with Voronoi regions |

Tracking results are cached in `stubs/track_stubs.pkl` (if `stubs.use_stub: true` in config.yaml) to avoid re-running detection on the same video.

---

## Requirements & Dependencies

See `requirements.txt` for the full list.  Key packages:

- **PyTorch** ≥ 2.0 + **torchvision**
- **Ultralytics** — YOLO inference
- **supervision** 0.26.0 — ByteTrack + detection helpers
- **transformers** — SigLIP team classifier (HuggingFace)
- **scikit-learn** — PCA + K-Means
- **OpenCV** — frame reading / writing / annotation
- **kornia**, **pytorch-lightning**, **SoccerNet** — TVCalib dependencies
- **scipy** — Savitzky–Golay and median filters

---

## Notes & Limitations

- **Resolution** — The calibration pipeline is tuned for 1920×1080 input.  Other resolutions require updating `calibration.image_width` and `calibration.image_height` in config.yaml.
- **Camera angle** — TVCalib works best with broadcast-style camera angles that show the majority of the pitch.  Extreme close-ups or end-zone views may produce poor calibration.
- **Ball possession overlay** — The UI assumes a 1920×1080 display resolution for the possession rectangle coordinates.  Adjust `draw_team_ball_control()` in `src/visualization/annotation.py` for other resolutions.
- **`kaggle_secrets`** — The original notebook used `kaggle_secrets.UserSecretsClient` for API key management in a Kaggle environment.  This has been removed; model paths are now specified locally via config.yaml.
- **Speed requires calibration — no exceptions.** Player speed (km/h) is computed exclusively from `position_field`, which only exists after the pixel-to-metre homography transform (`ViewTransformer`). This transform requires a completed TVCalib calibration pass. There is no pixel-space fallback for speed, because pixel displacement is not a physically meaningful speed (it depends on camera distance, angle and zoom). Consequences:
  - **Mode 2 (Default)** never runs calibration, so it can never show speed — `visualization.show_speed` in config.yaml has no effect in this mode and defaults to `false`.
  - **Mode 3 (Custom)** automatically runs calibration if you answer "yes" to the speed prompt, even if you declined the calibration/radar/Voronoi videos — calibration is a prerequisite, not an optional extra, whenever speed is requested.
  - **Mode 1 (Full)** always includes calibration, so speed is always available.
- **Speed accuracy** — Speed estimates depend on calibration quality. Frames where calibration fails (no homography for that frame) will not have speed data for objects in that frame.

---

## Acknowledgements & Credits

### TVCalib

This project uses [TVCalib](https://github.com/MM4SPA/tvcalib) for camera calibration from broadcast video.

TVCalib is a self-calibrating system that fits camera parameters to field-line keypoints detected by a segmentation network.  It was developed by the [MM4SPA](https://github.com/MM4SPA) group and is available under the MIT licence.

```
https://github.com/MM4SPA/tvcalib
```

The segmentation component uses the [sn-calibration-segmentation](https://github.com/jtheiner/sn-calibration-segmentation) sub-repository by [@jtheiner](https://github.com/jtheiner).

---

## License

This project is released for research and educational use.  Please respect the individual licences of all external components (TVCalib, SigLIP, Ultralytics YOLO, supervision).
