"""
Stop-line and red-light violation detection.
Checks if a vehicle crosses a defined stop-line while the signal is red.
"""
from __future__ import annotations

from dataclasses import dataclass, field


Point = tuple[int, int]


@dataclass
class StopLineRule:
    """
    Detects stop-line violations.

    stop_line_y: the Y pixel coordinate of the stop line (horizontal line).
    approach_direction: 'down' (y increasing) or 'up' (y decreasing).
    red_phase_frames: set of frame indices where signal is red.
                      In production, connect to a signal controller API.
    """
    stop_line_y: int
    approach_direction: str = "down"  # 'down' or 'up'
    min_speed_pixels: float = 3.0
    crossed_tracks: dict[int, bool] = field(default_factory=dict)
    violation_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        centers: list[Point],
        frame_index: int,
        is_red_phase: bool,
    ) -> bool:
        if len(centers) < 2:
            return False
        if track_id in self.violation_tracks:
            return False  # already flagged, don't double-count

        prev_y = centers[-2][1]
        curr_y = centers[-1][1]

        if self.approach_direction == "down":
            crossed = prev_y < self.stop_line_y <= curr_y
        else:
            crossed = prev_y > self.stop_line_y >= curr_y

        if crossed and is_red_phase:
            self.violation_tracks.add(track_id)
            return True
        return False


@dataclass
class HelmetRule:
    """
    Helmet non-compliance detection on two-wheelers.

    Uses a heuristic: the top 35% of a motorcycle bounding box is the rider's head zone.
    If that zone is predominantly skin-tone colored (no helmet), flag it.

    Guards:
    - Minimum bbox area of 1500px² — skips distant/tiny bikes where the crop
      is too small for reliable detection.
    - Minimum track age of 10 frames — avoids false positives on just-detected bikes.
    - Skin-tone threshold of 30% (up from 25%) — reduces false positives on
      light-coloured or white helmets.

    In production: use a dedicated helmet classifier model on the cropped region.
    """
    two_wheeler_classes: set[str] = field(
        default_factory=lambda: {"motorcycle", "bicycle"}
    )
    min_track_age: int = 10
    min_bbox_area: int = 1500   # skip tiny/far-away bikes
    skin_ratio_threshold: float = 0.30  # raised from 0.25 for fewer false positives
    flagged_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        bbox: tuple[int, int, int, int],
        frame,
    ) -> bool:
        """
        frame: numpy array (BGR) — used for pixel analysis of head zone.
        """
        if class_name not in self.two_wheeler_classes:
            return False
        if track_id in self.flagged_tracks:
            return False
        if track_age < self.min_track_age:
            return False

        try:
            import numpy as np
            import cv2

            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = x2 - x1
            if h < 20 or w < 10:
                return False

            # Guard: skip tiny/distant bikes — head zone too small for reliable detection
            bbox_area = h * w
            if bbox_area < self.min_bbox_area:
                return False

            # Crop head zone: top 35% of the bounding box
            head_y2 = y1 + int(h * 0.35)
            head_crop = frame[y1:head_y2, x1:x2]
            if head_crop.size == 0:
                return False

            # Convert to HSV and check for skin tone dominance (no helmet = skin visible)
            hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            skin_lower = np.array([0, 30, 60], dtype=np.uint8)
            skin_upper = np.array([25, 170, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            skin_ratio = np.sum(skin_mask > 0) / max(skin_mask.size, 1)

            if skin_ratio > self.skin_ratio_threshold:
                self.flagged_tracks.add(track_id)
                return True
        except Exception:
            pass

        return False
