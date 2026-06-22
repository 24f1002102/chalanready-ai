from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class WrongSideRule:
    """
    Detects wrong-side / counter-flow driving.

    A vehicle is considered wrong-side if its recent travel vector
    has a strong negative cosine similarity with the legal traffic direction.

    Tuning for real intersections:
    - min_points=12  → requires sustained wrong-way movement, not a quick turn
    - opposite_threshold=-0.65 → only very clearly counter-flow vehicles are flagged
      (was -0.45, which flagged vehicles making legal right-angle turns at intersections)
    - min_movement_px=40 → ignores nearly-stationary vehicles (slow-moving traffic jams)
    """
    allowed_direction: tuple[float, float]
    min_points: int = 12
    opposite_threshold: float = -0.65
    min_movement_px: float = 40.0   # minimum travel distance before checking direction

    def is_wrong_side(self, centers: list[tuple[int, int]]) -> bool:
        if len(centers) < self.min_points:
            return False

        # Use the most recent window to capture current direction of travel
        recent = centers[-self.min_points:]
        start = recent[0]
        end = recent[-1]
        movement = (float(end[0] - start[0]), float(end[1] - start[1]))
        movement_length = hypot(*movement)
        allowed_length = hypot(*self.allowed_direction)

        # Skip vehicles that haven't moved enough — avoids flagging stationary / slow traffic
        if movement_length < self.min_movement_px:
            return False
        if allowed_length == 0:
            return False

        dot = (
            movement[0] * self.allowed_direction[0]
            + movement[1] * self.allowed_direction[1]
        )
        cosine = dot / (movement_length * allowed_length)

        # cosine <= -0.65 means vehicle is moving at >130° from legal direction
        # (i.e., clearly counter-flow, not just turning)
        return cosine <= self.opposite_threshold
