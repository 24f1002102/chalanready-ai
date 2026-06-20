"""
Seatbelt non-compliance and triple riding detection stubs.

Production implementation notes:
- Seatbelt: Requires YOLOv8-pose model to detect body keypoints.
  Check if shoulder-diagonal strap keypoints are present over the chest region.
  Needs specialized seatbelt dataset (CCTV angle-specific training).

- Triple Riding: Count distinct body contours within a motorcycle bounding box.
  If ≥3 human heads/torsos detected → triple riding.
  Requires a person/head detection model (e.g., YOLOv8n with 'person' class).

Both are currently stubs pending availability of BTP ASTraM labeled field datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeatbeltRule:
    """
    Detects seatbelt non-compliance in car/truck drivers.

    Production plan:
      1. Detect vehicle bbox.
      2. Crop driver region (left-front seat area: ~30% width, upper-center of bbox).
      3. Run YOLOv8-pose: check if 'left_shoulder'→'right_hip' keypoint line exists.
      4. If diagonal strap absent for ≥10 consecutive frames → flag.

    Dataset needed: BTP dashboard-camera footage with labeled seatbelt/no-seatbelt frames.
    """
    car_classes: set[str] = field(default_factory=lambda: {"car", "truck", "bus"})
    min_track_age: int = 15
    flagged_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        frame,
    ) -> bool:
        """
        Returns True if seatbelt non-compliance is detected.
        Currently a stub — returns False always.
        Integration point: replace with pose-based detection when model is available.
        """
        if class_name not in self.car_classes:
            return False
        if track_id in self.flagged_tracks:
            return False
        # TODO: wire in YOLOv8-pose seatbelt keypoint check
        return False


@dataclass
class TripleRidingRule:
    """
    Detects triple riding (3+ persons on a two-wheeler).

    Production plan:
      1. Detect motorcycle bbox.
      2. Run person/head detector on cropped region.
      3. If ≥3 distinct person instances detected with IoU < 0.3 → flag.
      4. Require ≥5 consecutive frames for confirmation.

    Dataset needed: BTP CCTV footage with labeled 2-person vs 3-person motorcycle riders.
    Alternative: Use MediaPipe pose with multi-person mode on the cropped region.
    """
    two_wheeler_classes: set[str] = field(
        default_factory=lambda: {"motorcycle", "bicycle"}
    )
    min_track_age: int = 10
    min_persons_threshold: int = 3
    flagged_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        frame,
    ) -> bool:
        """
        Returns True if triple riding is detected.
        Currently a stub — returns False always.
        Integration point: replace with multi-person head detection when model available.
        """
        if class_name not in self.two_wheeler_classes:
            return False
        if track_id in self.flagged_tracks:
            return False
        # TODO: wire in person/head count on motorcycle crop
        return False
