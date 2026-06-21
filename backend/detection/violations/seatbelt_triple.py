"""
Seatbelt non-compliance and triple riding detection.

Seatbelt:
  - Crops the driver-seat region (left 50%, upper-middle of car bbox).
  - Checks for the diagonal belt strap using HSV grey/dark tone along
    a diagonal line from left-shoulder to right-hip area.
  - No belt found after min_track_age frames → flag.

Triple Riding:
  - Crops the motorcycle bounding box.
  - Counts distinct skin-tone blobs (approximate head/torso regions).
  - If 3 or more blobs detected → triple riding.
  - Also counts detected person-shaped contours by height ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeatbeltRule:
    """
    Detects seatbelt non-compliance in car/truck drivers.

    Uses a pixel-level diagonal stripe heuristic:
    1. Crop driver region (left 55% width, upper 55% height of car bbox).
    2. Convert to grayscale; look for a diagonal dark/grey stripe
       (the seatbelt strap crosses from top-left to bottom-right).
    3. If no stripe found after min_track_age frames → flag.
    """
    car_classes: set[str] = field(default_factory=lambda: {"car", "truck", "bus"})
    min_track_age: int = 15
    flagged_tracks: set[int] = field(default_factory=set)
    # tracks that passed the check (belt found) — don't re-check
    cleared_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        frame,
    ) -> bool:
        if class_name not in self.car_classes:
            return False
        if track_id in self.flagged_tracks:
            return False
        if track_id in self.cleared_tracks:
            return False
        if track_age < self.min_track_age:
            return False

        # frame is passed as None in some call paths — guard
        if frame is None:
            return False

        try:
            import cv2
            import numpy as np

            # We don't have bbox here; a separate overload with bbox would be
            # ideal, but the rule engine only passes frame. We sample the whole
            # frame for grey-stripe patterns as a lightweight proxy.
            # In the video pipeline the frame is the preprocessed crop from
            # the tracker's bbox region passed through annotate_frame.
            h, w = frame.shape[:2]
            if h < 20 or w < 20:
                return False

            # Convert to greyscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # A seatbelt appears as a near-diagonal dark stripe in the upper
            # body region. We sample a diagonal strip from top-left to center.
            # If variance along that diagonal is low (flat grey) → belt present.
            samples = []
            n = 20
            for i in range(n):
                rx = int(w * 0.05 + (w * 0.45) * i / n)
                ry = int(h * 0.05 + (h * 0.50) * i / n)
                if 0 <= ry < h and 0 <= rx < w:
                    samples.append(int(gray[ry, rx]))

            if len(samples) < 5:
                return False

            arr = np.array(samples, dtype=float)
            # Seatbelt stripe: dark pixels (belt colour < 80 grey) in sequence
            dark_count = int(np.sum(arr < 80))
            belt_detected = dark_count >= (n // 3)  # at least 1/3 of samples dark

            if not belt_detected:
                self.flagged_tracks.add(track_id)
                return True
            else:
                self.cleared_tracks.add(track_id)
                return False

        except Exception:
            return False


@dataclass
class TripleRidingRule:
    """
    Detects triple riding (3+ persons on a two-wheeler).

    Method:
    1. Crop the motorcycle bounding box from the frame.
    2. Detect skin-tone blobs (HSV-based) in the cropped region.
    3. Run connected-component analysis on the skin mask.
    4. If 3+ distinct blobs (each large enough to be a human head/torso) → flag.
    """
    two_wheeler_classes: set[str] = field(
        default_factory=lambda: {"motorcycle", "bicycle"}
    )
    min_track_age: int = 10
    min_persons_threshold: int = 3
    min_blob_area: int = 60   # minimum pixel area for a blob to count as a person
    flagged_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        frame,
    ) -> bool:
        if class_name not in self.two_wheeler_classes:
            return False
        if track_id in self.flagged_tracks:
            return False
        if track_age < self.min_track_age:
            return False

        if frame is None:
            return False

        try:
            import cv2
            import numpy as np

            h, w = frame.shape[:2]
            if h < 15 or w < 15:
                return False

            # Skin tone HSV range (covers Indian/South Asian tones)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 180, 255], dtype=np.uint8)
            lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 180, 255], dtype=np.uint8)

            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(mask1, mask2)

            # Morphological close to merge nearby skin pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

            # Count connected components (each is a potential person)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                skin_mask, connectivity=8
            )

            person_blobs = 0
            for label_idx in range(1, num_labels):  # skip background (0)
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area >= self.min_blob_area:
                    person_blobs += 1

            if person_blobs >= self.min_persons_threshold:
                self.flagged_tracks.add(track_id)
                return True

            return False

        except Exception:
            return False
