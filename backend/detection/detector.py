from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


COCO_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
COCO_PERSON_CLASS = "person"
COCO_ALL_CLASSES = COCO_VEHICLE_CLASSES | {COCO_PERSON_CLASS}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    source: str

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class DetectorUnavailableError(RuntimeError):
    """Raised when a requested detector cannot be started."""


class ColorVehicleDetector:
    """
    Deterministic HSV-based detector — ONLY for the synthetic demo video.
    The synthetic video uses specific HSV colors for each vehicle type.
    This detector will NOT work reliably on real CCTV footage.
    Use only when YOLO weights are unavailable.
    """

    name = "color"
    display_name = "Color-HSV Synthetic Demo Detector"
    mode = "synthetic_demo"
    is_demo_mode = True

    def __init__(self, min_area: int = 450) -> None:
        self.min_area = min_area
        self.color_ranges = [
            ("car",        (45,  80,  80), (90,  255, 255)),
            ("motorcycle", (95,  80,  80), (132, 255, 255)),
            ("truck",      (18,  90,  90), (36,  255, 255)),
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for class_name, lower, upper in self.color_ranges:
            mask = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                # Confidence proportional to blob size (not a fixed 0.92)
                conf = min(0.55 + (area / 15000) * 0.30, 0.85)
                detections.append(
                    Detection(
                        bbox=(x, y, x + w, y + h),
                        confidence=round(conf, 3),
                        class_name=class_name,
                        source=self.name,
                    )
                )

        return detections


class YOLOVehicleDetector:
    """
    YOLOv8-based detector — the real AI detector.
    Detects vehicles AND persons for comprehensive violation analysis.
    Returns actual model confidence scores (not hardcoded values).
    """

    name = "yolo"
    display_name = "YOLOv8n Vehicle Detector"
    mode = "ai"
    is_demo_mode = False

    def __init__(self, model_path: str | Path, confidence: float = 0.30) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError(
                "Ultralytics is not installed. "
                "Run: pip install -r requirements-local.txt"
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise DetectorUnavailableError(
                f"YOLO model not found at {model_path}. "
                "Place yolov8n.pt in repo root."
            )

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.model_path = str(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = names.get(class_id, str(class_id))
            if class_name not in COCO_ALL_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            real_conf = float(box.conf.item())  # actual model confidence
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=round(real_conf, 4),
                    class_name=class_name,
                    source=self.name,
                )
            )

        return detections


def find_local_yolo_model() -> Path | None:
    """
    Search for YOLOv8 weights in priority order:
    1. CHALANREADY_YOLO_MODEL env var
    2. Repo root yolov8n.pt  ← where the file is actually committed
    3. backend/weights/yolov8n.pt  (legacy path)
    4. Current working directory
    """
    env_path = os.getenv("CHALANREADY_YOLO_MODEL")
    # This file is at backend/detection/detector.py → parents[2] = repo root
    _repo_root = Path(__file__).resolve().parents[2]

    candidates = [
        Path(env_path) if env_path else None,
        _repo_root / "yolov8n.pt",                          # ← primary location
        _repo_root / "backend" / "weights" / "yolov8n.pt",  # legacy
        Path.cwd() / "yolov8n.pt",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def create_detector(kind: str = "auto") -> Detector:
    """
    Create the best available detector.

    'auto'  → tries YOLOv8 first (real AI), falls back to color-HSV (demo only)
    'yolo'  → YOLOv8 only; raises DetectorUnavailableError if not available
    'color' → HSV color detector (works on synthetic demo video only)
    """
    normalized = kind.lower().strip()

    if normalized == "color":
        return ColorVehicleDetector()

    if normalized == "yolo":
        model_path = find_local_yolo_model()
        if not model_path:
            raise DetectorUnavailableError(
                "No YOLO model found. Place yolov8n.pt in repo root "
                "or set CHALANREADY_YOLO_MODEL env var."
            )
        return YOLOVehicleDetector(model_path)

    if normalized == "auto":
        model_path = find_local_yolo_model()
        if model_path:
            try:
                return YOLOVehicleDetector(model_path)
            except DetectorUnavailableError:
                pass
        import warnings
        warnings.warn(
            "YOLOv8 unavailable — falling back to color-HSV detector "
            "(synthetic demo only, not suitable for real CCTV footage).",
            RuntimeWarning,
            stacklevel=2,
        )
        return ColorVehicleDetector()

    raise ValueError(f"Unsupported detector '{kind}'. Use: auto, yolo, or color.")


def get_detector_status() -> dict[str, object]:
    """Return lightweight detector readiness without instantiating YOLO."""
    model_path = find_local_yolo_model()
    ultralytics_available = importlib.util.find_spec("ultralytics") is not None
    yolo_ready = bool(model_path and ultralytics_available)
    return {
        "ai_mode_ready": yolo_ready,
        "yolo": {
            "available": yolo_ready,
            "ultralytics_installed": ultralytics_available,
            "weights_found": model_path is not None,
            "model_path": str(model_path) if model_path else None,
            "mode_label": "YOLOv8n AI Mode",
        },
        "synthetic_demo": {
            "available": True,
            "mode_label": "Color-HSV Synthetic Demo",
            "warning": "Use only for synthetic validation clips, not real CCTV.",
        },
    }
