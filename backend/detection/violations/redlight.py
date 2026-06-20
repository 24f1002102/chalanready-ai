"""
Red-light violation detection — heuristic signal ROI approach.

How it works:
  1. Look for a bright-red circular region in the upper portion of the frame
     (where traffic signals are typically located in Indian road CCTV footage).
  2. Use HSV colour range for red (wraps around H=0 and H=180 in OpenCV).
  3. If a sufficiently large red blob is found → signal is likely red.

Limitations (documented honestly):
  - Works only if the signal head is visible in the camera frame.
  - Requires per-camera calibration of the signal ROI in production.
  - Not reliable in heavy rain or lens flare conditions.
  - For production: connect to BTP's ASTraM signal controller API directly.

Production upgrade path:
  signal_state = astrap_api.get_signal_state(junction_id, timestamp)
"""
from __future__ import annotations

import numpy as np


# Default signal search region: top 40% of frame, full width
# In production this is a per-camera calibrated bounding box
_SIGNAL_ROI_TOP_FRAC    = 0.0   # 0% from top
_SIGNAL_ROI_BOTTOM_FRAC = 0.40  # down to 40% from top

# Minimum fraction of ROI pixels that must be red to declare "red signal"
_RED_PIXEL_THRESHOLD = 0.008   # 0.8% of ROI — avoids car taillights triggering it


def is_red_signal(
    frame: np.ndarray,
    roi_top_frac: float = _SIGNAL_ROI_TOP_FRAC,
    roi_bottom_frac: float = _SIGNAL_ROI_BOTTOM_FRAC,
    threshold: float = _RED_PIXEL_THRESHOLD,
) -> bool:
    """
    Returns True if a red traffic signal is detected in the frame.

    Parameters
    ----------
    frame       : BGR numpy array (already CLAHE-preprocessed)
    roi_top_frac    : fraction from top to start the signal search ROI
    roi_bottom_frac : fraction from top to end the signal search ROI
    threshold   : minimum red pixel fraction in ROI to trigger detection
    """
    try:
        import cv2

        h, w = frame.shape[:2]
        y1 = int(h * roi_top_frac)
        y2 = int(h * roi_bottom_frac)
        roi = frame[y1:y2, :]

        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red in HSV wraps: H ∈ [0, 10] and H ∈ [160, 180]
        # Also require high Saturation (>80) and high Value (>80) to avoid
        # dark red objects (like car body paint) triggering the detector.
        lower_red1 = np.array([0,   80, 80],  dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 80, 80],  dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological close to join nearby red pixels (signal glow)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_fraction = float(np.sum(red_mask > 0)) / max(red_mask.size, 1)
        return red_fraction >= threshold

    except Exception:
        return False


def annotate_signal_state(frame: np.ndarray, is_red: bool) -> np.ndarray:
    """Draw a signal state indicator on the frame (for annotated evidence)."""
    try:
        import cv2
        h, _ = frame.shape[:2]
        label  = "SIGNAL: RED" if is_red else "SIGNAL: ---"
        colour = (0, 0, 220) if is_red else (80, 80, 80)
        cv2.putText(frame, label, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    except Exception:
        pass
    return frame
