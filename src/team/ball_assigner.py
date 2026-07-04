"""
PlayerBallAssigner — assigns the ball to the nearest player each frame.
"""

from __future__ import annotations

from src.kinematics.speed import measure_distance, get_center_of_bbox


class PlayerBallAssigner:
    """
    Determines which player is in possession of the ball each frame.

    The ball is assigned to the player whose feet (bottom corners of the
    bounding box) are closest to the ball centre, provided the distance
    is within ``max_player_ball_distance`` pixels.
    """

    def __init__(self, max_player_ball_distance: int = 40) -> None:
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(
        self,
        players: dict,
        ball_bbox: list,
    ) -> int:
        """
        Return the *player_id* of the player who has the ball, or ``-1``.

        Args:
            players: Per-frame player dictionary ``{track_id: {"bbox": ...}}``.
            ball_bbox: Ball bounding box ``[x1, y1, x2, y2]``.

        Returns:
            int: Assigned player's track ID, or ``-1`` if no player is close.
        """
        if not players:
            return -1

        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            bbox = player["bbox"]

            dist_left = measure_distance((bbox[0], bbox[3]), ball_position)
            dist_right = measure_distance((bbox[2], bbox[3]), ball_position)
            distance = min(dist_left, dist_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
