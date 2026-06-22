"""
Seatbelt non-compliance and triple riding detection.

Seatbelt:
  - Crops the DRIVER REGION from the vehicle bounding box
    (left 55%, upper-middle 55% of car bbox — where driver sits)
  - Checks for diagonal belt strap using greyscale diagonal sampling
  - No belt pattern found after min_track_age frames → flag

Triple Riding:
  - Crops the motorcycle bounding box
  - Counts distinct skin-tone blobs (approximate head/torso regions)
  - Blobs must be at least 15px apart to avoid counting one person twice
  - If 3+ distinct blobs detected → triple riding
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeatbeltRule:
    """
    Detects seatbelt non-compliance in car/truck drivers.

    Method:
    1. Crop driver region (left 55% width, upper 55% height of car bbox).
    2. Convert to greyscale; look for a diagonal dark/grey stripe
       (the seatbelt strap crosses from shoulder to hip diagonally).
    3. If no belt stripe found after min_track_age frames → flag.
    """
    car_classes: set[str] = field(default_factory=lambda: {"car", "truck", "bus"})
    min_track_age: int = 15
    flagged_tracks: set[int] = field(default_factory=set)
    cleared_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        vehicle_crop: "np.ndarray | None",  # already-cropped vehicle bbox region
    ) -> bool:
        if class_name not in self.car_classes:
            return False
        if track_id in self.flagged_tracks or track_id in self.cleared_tracks:
            return False
        if track_age < self.min_track_age:
            return False
        if vehicle_crop is None:
            return False

        try:
            import cv2
            import numpy as np

            h, w = vehicle_crop.shape[:2]
            if h < 30 or w < 30:
                return False

            # Crop driver region: left 55% width, upper 55% height
            driver_w = int(w * 0.55)
            driver_h = int(h * 0.55)
            driver_crop = vehicle_crop[:driver_h, :driver_w]

            if driver_crop.size == 0:
                return False

            gray = cv2.cvtColor(driver_crop, cv2.COLOR_BGR2GRAY)
            dh, dw = gray.shape[:2]

            # Sample diagonal from top-left (shoulder) to bottom-right (hip)
            # A seatbelt appears as a dark diagonal stripe in this region
            n = 25
            samples = []
            for i in range(n):
                rx = int(dw * 0.05 + (dw * 0.85) * i / n)
                ry = int(dh * 0.05 + (dh * 0.85) * i / n)
                if 0 <= ry < dh and 0 <= rx < dw:
                    samples.append(int(gray[ry, rx]))

            if len(samples) < 8:
                return False

            arr = np.array(samples, dtype=float)
            # Belt is dark (< 80 greyscale) and appears as a consistent stripe
            dark_count = int(np.sum(arr < 80))
            # Need at least 1/3 of samples dark AND some variance
            # (a uniform dark region is shadow, not a belt)
            variance = float(np.std(arr))
            belt_detected = dark_count >= (n // 3) and variance > 5.0

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
    1. Crop the motorcycle bbox.
    2. Detect skin-tone blobs (HSV) in the cropped region.
    3. Run connected-component analysis.
    4. Filter blobs: must be large enough (> min_blob_area) AND
       centroids must be at least min_separation_px apart
       (avoids counting one person's head + shoulder as two people).
    5. If 3+ distinct blobs → flag.
    """
    two_wheeler_classes: set[str] = field(
        default_factory=lambda: {"motorcycle", "bicycle"}
    )
    min_track_age: int = 10
    min_persons_threshold: int = 3
    min_blob_area: int = 80
    min_separation_px: float = 15.0   # blobs closer than this = same person
    flagged_tracks: set[int] = field(default_factory=set)

    def observe(
        self,
        track_id: int,
        class_name: str,
        track_age: int,
        vehicle_crop: "np.ndarray | None",
    ) -> bool:
        if class_name not in self.two_wheeler_classes:
            return False
        if track_id in self.flagged_tracks:
            return False
        if track_age < self.min_track_age:
            return False
        if vehicle_crop is None:
            return False

        try:
            import cv2
            import numpy as np
            from math import dist

            h, w = vehicle_crop.shape[:2]
            if h < 20 or w < 20:
                return False

            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 180, 255], dtype=np.uint8)
            lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 180, 255], dtype=np.uint8)

            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(mask1, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                skin_mask, connectivity=8
            )

            # Collect valid blobs
            valid_centroids: list[tuple[float, float]] = []
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area < self.min_blob_area:
                    continue
                cx, cy = centroids[label_idx]
                valid_centroids.append((cx, cy))

            # Merge blobs that are too close (same person)
            merged: list[tuple[float, float]] = []
            for c in valid_centroids:
                too_close = any(
                    dist(c, m) < self.min_separation_px for m in merged
                )
                if not too_close:
                    merged.append(c)

            if len(merged) >= self.min_persons_threshold:
                self.flagged_tracks.add(track_id)
                return True

            return False

        except Exception:
            return False
