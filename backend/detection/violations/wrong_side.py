from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class WrongSideRule:
    allowed_direction: tuple[float, float]
    min_points: int = 8
    opposite_threshold: float = -0.45

    def is_wrong_side(self, centers: list[tuple[int, int]]) -> bool:
        if len(centers) < self.min_points:
            return False

        start = centers[0]
        end = centers[-1]
        movement = (float(end[0] - start[0]), float(end[1] - start[1]))
        movement_length = hypot(*movement)
        allowed_length = hypot(*self.allowed_direction)

        if movement_length == 0 or allowed_length == 0:
            return False

        dot = (
            movement[0] * self.allowed_direction[0]
            + movement[1] * self.allowed_direction[1]
        )
        cosine = dot / (movement_length * allowed_length)
        return cosine <= self.opposite_threshold
