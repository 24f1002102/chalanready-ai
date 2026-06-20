"""
Synthetic plate OCR — generates realistic Indian license plates for demo.
In production, swap with EasyOCR or Tesseract.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass


INDIAN_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
)

# Common Karnataka (KA) and other state prefixes for Bengaluru area
_STATE_CODES = ["KA", "KA", "KA", "KA", "TN", "AP", "MH", "KL", "DL"]
_DISTRICTS_KA = ["01", "02", "03", "04", "05", "41", "50", "51", "52", "53"]
_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def _generate_plate_for_track(track_id: int) -> str:
    """
    Deterministic but random-looking plate from track_id.
    Same track always gets same plate — consistent across frames.
    """
    seed = hashlib.md5(f"track_{track_id}".encode()).hexdigest()
    rng = random.Random(seed)
    state = rng.choice(_STATE_CODES)
    district = rng.choice(_DISTRICTS_KA)
    series = "".join(rng.choices(_LETTERS, k=2))
    number = str(rng.randint(1000, 9999))
    return f"{state}{district}{series}{number}"


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    is_valid_indian_format: bool


class PlateReader:
    """
    Demo plate reader — generates realistic Indian plates from track ID.
    Replace with EasyOCR in production:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(plate_crop)
    """

    def read_for_track(self, track_id: int) -> PlateRead:
        plate_text = _generate_plate_for_track(track_id)
        return PlateRead(
            text=plate_text,
            confidence=round(random.uniform(0.78, 0.97), 2),
            is_valid_indian_format=True,
        )

    def read(self, _frame) -> PlateRead | None:
        """Legacy stub for frame-based reading — use read_for_track() instead."""
        return None


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_indian_plate(text: str) -> bool:
    return bool(INDIAN_PLATE_PATTERN.match(normalize_plate_text(text)))
