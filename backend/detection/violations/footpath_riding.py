from __future__ import annotations

from dataclasses import dataclass, field

from .illegal_parking import Point, point_in_polygon


@dataclass
class FootpathRidingRule:
    footpath_zone: list[Point]
    min_inside_frames: int = 5
    allowed_classes: set[str] = field(
        default_factory=lambda: {"bicycle", "motorcycle", "car", "truck", "bus"}
    )
    inside_counts: dict[int, int] = field(default_factory=dict)

    def observe(
        self,
        track_id: int,
        class_name: str,
        center: Point,
    ) -> bool:
        if class_name not in self.allowed_classes:
            return False

        if point_in_polygon(center, self.footpath_zone):
            self.inside_counts[track_id] = self.inside_counts.get(track_id, 0) + 1
        else:
            self.inside_counts.pop(track_id, None)

        return self.inside_counts.get(track_id, 0) >= self.min_inside_frames
