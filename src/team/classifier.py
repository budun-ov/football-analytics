"""
Team Classifier — assigns players to teams using SigLIP embeddings + K-Means.

Pipeline:
  1. Collect player torso crops during a warmup period.
  2. Extract vision embeddings via SigLIP (google/siglip-base-patch16-224).
  3. Reduce to 3 dimensions with PCA.
  4. Cluster with K-Means (n_clusters = 2 teams).
  5. Compute representative team colours from the crops.
  6. Apply team assignments per frame, with a smoothing buffer per track ID.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from transformers import AutoProcessor, SiglipVisionModel


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TeamClassifierConfig:
    model_name: str = "google/siglip-base-patch16-224"
    batch_size: int = 32
    n_clusters: int = 2
    random_state: int = 0
    warmup_samples: int = 150  # Crops collected before fitting
    buffer_frames: int = 10   # Votes before a track ID is locked to a team


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TeamClassifier:
    """
    Assign each tracked player to one of two teams.

    After a warmup phase (collecting ``config.warmup_samples`` crops), the
    PCA + K-Means model is fitted once.  Subsequent frames use the fitted
    model; each track ID accumulates votes over ``config.buffer_frames``
    frames before the team assignment is locked.
    """

    def __init__(
        self,
        config: Optional[TeamClassifierConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or TeamClassifierConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  Loading team classifier: {self.config.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = (
            SiglipVisionModel.from_pretrained(self.config.model_name)
            .to(self.device)
            .eval()
        )

        self.pca = PCA(n_components=3, random_state=self.config.random_state)
        self.kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            n_init=20,
            random_state=self.config.random_state,
        )

        self.is_fitted: bool = False
        self.team_colors: List[Tuple[int, int, int]] = []

        # Warmup buffer
        self._warmup_crops: List[np.ndarray] = []

        # Per-track voting history: {track_id: [team_id, ...]}
        self.track_history: Dict[int, List[int]] = {}

        # Locked assignments: {track_id: team_id}
        self.fixed_teams: Dict[int, int] = {}

        # Pending assignments from warmup frames (applied after fitting)
        self._pending_assignments: List[tuple] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, crops: List[np.ndarray]) -> None:
        """Fit PCA + K-Means on *crops* and compute team colours."""
        if not crops:
            return

        embeddings = self._extract_embeddings(crops)
        projections = self.pca.fit_transform(embeddings)
        labels = self.kmeans.fit_predict(projections)

        self.is_fitted = True

        raw_colors = self._compute_team_colors(crops, labels)
        self.team_colors = [self._get_vibrant_color(c) for c in raw_colors]

    def process_frame(
        self,
        frame: np.ndarray,
        players: Dict[int, dict],
        goalkeepers: Dict[int, dict],
    ) -> None:
        """
        Classify all players (and assign goalkeepers) in one frame.

        Mutates *players* and *goalkeepers* in-place, adding ``team_id``
        and ``team_color`` keys.
        """
        if not self.is_fitted:
            self._collect_warmup(frame, players)
            return

        self._classify_players(frame, players)
        self._classify_goalkeepers(players, goalkeepers)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Return K-Means team predictions (0 or 1) for *crops*."""
        embeddings = self._extract_embeddings(crops)
        projections = self.pca.transform(embeddings)
        return self.kmeans.predict(projections)

    # ------------------------------------------------------------------
    # Internal — warmup
    # ------------------------------------------------------------------

    def _collect_warmup(
        self,
        frame: np.ndarray,
        players: Dict[int, dict],
    ) -> None:
        """Accumulate torso crops during the warmup phase."""
        for p_id, p_data in players.items():
            crop = self._crop_torso(frame, p_data["bbox"])
            if crop.size > 0:
                self._warmup_crops.append(crop)
                self._pending_assignments.append((p_data, p_id, crop))

        if len(self._warmup_crops) >= self.config.warmup_samples:
            self.fit(self._warmup_crops)
            self._apply_pending_assignments()
            self._warmup_crops.clear()

    def _apply_pending_assignments(self) -> None:
        """Back-fill team assignments for frames collected during warmup."""
        if not self._pending_assignments:
            return

        crops = [item[2] for item in self._pending_assignments]
        predicted = self.predict(crops)

        for (p_data, p_id, _), team_id in zip(self._pending_assignments, predicted):
            if p_id not in self.track_history:
                self.track_history[p_id] = []
            self.track_history[p_id].append(int(team_id))
            p_data["team_id"] = int(team_id)
            p_data["team_color"] = self.team_colors[int(team_id)]

        self._pending_assignments.clear()

    # ------------------------------------------------------------------
    # Internal — classification
    # ------------------------------------------------------------------

    def _classify_players(
        self,
        frame: np.ndarray,
        player_dict: Dict[int, dict],
    ) -> None:
        new_crops: List[np.ndarray] = []
        new_ids: List[int] = []

        for t_id, player in player_dict.items():
            if t_id in self.fixed_teams:
                # Already locked — just assign colour
                team_id = self.fixed_teams[t_id]
                player["team_id"] = team_id
                player["team_color"] = self.team_colors[team_id]
            else:
                crop = self._crop_torso(frame, player["bbox"])
                if crop.size > 0:
                    new_crops.append(crop)
                    new_ids.append(t_id)

        if not new_crops:
            return

        predicted = self.predict(new_crops)

        for t_id, team_id in zip(new_ids, predicted):
            if t_id not in self.track_history:
                self.track_history[t_id] = []

            self.track_history[t_id].append(int(team_id))

            # Majority vote for stability
            smooth_team_id = Counter(self.track_history[t_id]).most_common(1)[0][0]
            player_dict[t_id]["team_id"] = smooth_team_id
            player_dict[t_id]["team_color"] = self.team_colors[smooth_team_id]

            # Lock after enough frames
            if len(self.track_history[t_id]) >= self.config.buffer_frames:
                self.fixed_teams[t_id] = smooth_team_id

    def _classify_goalkeepers(
        self,
        players: Dict[int, dict],
        goalkeepers: Dict[int, dict],
    ) -> None:
        """Assign each goalkeeper to the team whose players are nearest to him."""
        if not goalkeepers or not players:
            return

        for g_id, goalie in goalkeepers.items():
            g_bbox = goalie["bbox"]
            g_center = np.array(
                [(g_bbox[0] + g_bbox[2]) / 2, (g_bbox[1] + g_bbox[3]) / 2]
            )

            distances: Dict[int, List[float]] = {0: [], 1: []}

            for p_data in players.values():
                if "team_id" not in p_data:
                    continue
                p_bbox = p_data["bbox"]
                p_center = np.array(
                    [(p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2]
                )
                dist = float(np.linalg.norm(g_center - p_center))
                distances[p_data["team_id"]].append(dist)

            avg_dist = {
                t: np.mean(sorted(dists)[:3]) if dists else float("inf")
                for t, dists in distances.items()
            }

            assigned_team = min(avg_dist, key=avg_dist.get)
            goalie["team_id"] = assigned_team
            goalie["team_color"] = self.team_colors[assigned_team]

    # ------------------------------------------------------------------
    # Internal — embeddings & colours
    # ------------------------------------------------------------------

    def _extract_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        processed = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in crops
        ]

        all_emb = []
        for i in range(0, len(processed), self.config.batch_size):
            batch = processed[i : i + self.config.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_emb.append(emb)

        return normalize(np.concatenate(all_emb))

    def _compute_team_colors(
        self,
        crops: List[np.ndarray],
        labels: np.ndarray,
    ) -> List[Tuple[int, int, int]]:
        colors = []
        for team_id in range(self.config.n_clusters):
            team_crops = [c for c, lbl in zip(crops, labels) if lbl == team_id]

            if not team_crops:
                colors.append((128, 128, 128))
                continue

            pixels = np.concatenate(
                [cv2.resize(c, (16, 16)).reshape(-1, 3) for c in team_crops]
            )
            km = KMeans(n_clusters=1, n_init=5).fit(pixels)
            colors.append(tuple(map(int, km.cluster_centers_[0])))

        return colors

    def _get_vibrant_color(
        self,
        rgb: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """Boost saturation and brightness while preserving hue."""
        r, g, b = [x / 255.0 for x in rgb]
        h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
        new_rgb = colorsys.hsv_to_rgb(h, 0.75, 0.9)
        return tuple(int(x * 255) for x in new_rgb)

    def _crop_torso(
        self,
        frame: np.ndarray,
        bbox: List[float],
    ) -> np.ndarray:
        """Crop the torso region of a player bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        return frame[int(y1 + h * 0.15) : int(y1 + h * 0.5), x1:x2]
