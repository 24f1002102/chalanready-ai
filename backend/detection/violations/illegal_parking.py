from __future__ import annotations

from dataclasses import dataclass, field


Point = tuple[int, int]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (
            (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


@dataclass
class IllegalParkingRule:
    restricted_zone: list[Point]
    dwell_frames: int
    max_stationary_pixels: float = 12.0
    stationary_counts: dict[int, int] = field(default_factory=dict)
    anchor_centers: dict[int, Point] = field(default_factory=dict)

    def observe(
        self,
        track_id: int,
        centers: list[Point],
    ) -> bool:
        if len(centers) < 2:
            return False

        current = centers[-1]
        previous = centers[-2]
        if not point_in_polygon(current, self.restricted_zone):
            self.stationary_counts.pop(track_id, None)
            self.anchor_centers.pop(track_id, None)
            return False

        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        frame_movement = (dx * dx + dy * dy) ** 0.5

        anchor = self.anchor_centers.get(track_id)
        if anchor is None:
            self.anchor_centers[track_id] = current
            self.stationary_counts[track_id] = 1
            return False

        drift_x = current[0] - anchor[0]
        drift_y = current[1] - anchor[1]
        anchor_drift = (drift_x * drift_x + drift_y * drift_y) ** 0.5

        if frame_movement <= self.max_stationary_pixels and anchor_drift <= self.max_stationary_pixels:
            self.stationary_counts[track_id] = self.stationary_counts.get(track_id, 0) + 1
        else:
            self.anchor_centers[track_id] = current
            self.stationary_counts[track_id] = 1

        return self.stationary_counts[track_id] >= self.dwell_frames
