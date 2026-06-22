"""
License Plate OCR — EasyOCR (production) with honest fallback (demo).

Production mode (requires: pip install easyocr):
  - Crops the BOTTOM 30% of vehicle bbox (where plates are mounted)
  - Applies adaptive thresholding to sharpen plate region
  - Results filtered to Indian plate format (KA01AB1234)
  - source="easyocr" → shown as verified in the UI

Fallback mode (EasyOCR not installed or crops too small):
  - Returns source="ocr_unavailable" with text="PLATE_UNREAD"
  - Confidence=0.0 — explicitly honest, NOT labelled as a real plate
  - UI should display "⚠ OCR N/A" not a fake registration number

Switch: set env var CHALANREADY_OCR_MODE=real to force EasyOCR attempt.

Install EasyOCR:
    pip install easyocr
    (Downloads ~100MB models on first run)
"""
from __future__ import annotations

import os
import re
import importlib.util
from dataclasses import dataclass

import numpy as np


INDIAN_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
)


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    is_valid_indian_format: bool
    source: str  # "easyocr" | "ocr_unavailable"


def _crop_plate_region(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """
    Crop the most likely plate region from a vehicle bounding box.

    Plates are mounted at the front/rear of vehicles.
    In top-down CCTV view: bottom 30% of bbox.
    In front-facing CCTV view: bottom-centre 30% of bbox.
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    if h < 30 or w < 30:
        return None

    # Bottom 30% of vehicle bbox — plate zone
    plate_y1 = y2 - int(h * 0.30)
    plate_crop = frame[plate_y1:y2, x1:x2]

    if plate_crop.size == 0:
        return None

    return plate_crop


def _preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
    """
    Enhance plate crop for better OCR:
    1. Resize to 2x for small plates
    2. Convert to greyscale
    3. Apply CLAHE for contrast
    4. Adaptive threshold to binarize
    """
    import cv2
    h, w = crop.shape[:2]
    if w < 120:
        scale = max(2, 120 // w)
        crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary


def _try_load_easyocr():
    """Try importing EasyOCR — returns reader or None."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return reader
    except (ImportError, Exception):
        return None


_OCR_MODE = os.getenv("CHALANREADY_OCR_MODE", "auto")
_EASYOCR_READER = None
if _OCR_MODE in ("auto", "real"):
    _EASYOCR_READER = _try_load_easyocr()

# Honest fallback result — returned whenever OCR cannot be performed
_OCR_UNAVAILABLE_RESULT = PlateRead(
    text="PLATE_UNREAD",
    confidence=0.0,
    is_valid_indian_format=False,
    source="ocr_unavailable",
)


class PlateReader:
    """
    Two-mode licence plate reader:

    • Production (EasyOCR available):
        - Crops bottom 30% of vehicle bbox (plate zone)
        - Preprocesses crop (CLAHE + adaptive threshold)
        - Runs EasyOCR on the enhanced crop
        - Filters to valid Indian plate format KA01AB1234
        - source="easyocr"

    • Fallback (EasyOCR not installed):
        - Returns text="PLATE_UNREAD", confidence=0.0
        - source="ocr_unavailable" — clearly labelled, never faked
        - UI shows "⚠ OCR N/A" badge — honest to officers and judges

    To upgrade to production OCR:
        pip install easyocr
        (EasyOCR will auto-download ~100MB models on first run)
    """

    def __init__(self) -> None:
        self._reader = _EASYOCR_READER

    def read_for_track(
        self,
        track_id: int,
        frame: np.ndarray | None = None,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> PlateRead:
        # Attempt real EasyOCR on focused plate crop
        if self._reader is not None and frame is not None and bbox is not None:
            try:
                import cv2
                plate_crop = _crop_plate_region(frame, bbox)
                if plate_crop is not None:
                    # Try both colour and binary-threshold crops
                    crops_to_try = [plate_crop]
                    binary = _preprocess_plate_crop(plate_crop)
                    # Convert binary back to BGR so EasyOCR gets 3-channel
                    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    crops_to_try.append(binary_bgr)

                    best_text = None
                    best_conf = 0.0

                    for crop in crops_to_try:
                        results = self._reader.readtext(crop, detail=1)
                        for (_, text, conf) in results:
                            norm = normalize_plate_text(text)
                            if is_valid_indian_plate(norm) and conf > best_conf:
                                best_text = norm
                                best_conf = conf

                    if best_text:
                        return PlateRead(
                            text=best_text,
                            confidence=round(float(best_conf), 3),
                            is_valid_indian_format=True,
                            source="easyocr",
                        )
            except Exception:
                pass  # fall through to honest fallback

        # Honest fallback — plate cannot be read, never fake a number
        return _OCR_UNAVAILABLE_RESULT

    def read(self, _frame) -> PlateRead | None:
        """Legacy compatibility method. Use read_for_track() instead."""
        return None


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_indian_plate(text: str) -> bool:
    return bool(INDIAN_PLATE_PATTERN.match(normalize_plate_text(text)))


def get_ocr_status() -> dict[str, object]:
    """Return OCR readiness for the dashboard status bar."""
    easyocr_installed = importlib.util.find_spec("easyocr") is not None
    return {
        "mode": _OCR_MODE,
        "easyocr_installed": easyocr_installed,
        "reader_loaded": _EASYOCR_READER is not None,
        "available": _EASYOCR_READER is not None,
        "fallback_label": "PLATE_UNREAD",
        "warning": None if _EASYOCR_READER is not None else "EasyOCR unavailable; plates are marked unread instead of faked.",
    }
