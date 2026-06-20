from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


COCO_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}


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
    """Deterministic detector for the generated synthetic demo video."""

    name = "color"

    def __init__(self, min_area: int = 450) -> None:
        self.min_area = min_area
        self.color_ranges = [
            ("car", (45, 80, 80), (90, 255, 255)),
            ("motorcycle", (95, 80, 80), (132, 255, 255)),
            ("truck", (18, 90, 90), (36, 255, 255)),
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
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(
                    Detection(
                        bbox=(x, y, x + w, y + h),
                        confidence=0.92,
                        class_name=class_name,
                        source=self.name,
                    )
                )

        return detections


class YOLOVehicleDetector:
    name = "yolo"

    def __init__(self, model_path: str | Path, confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError(
                "Ultralytics is not installed. Install requirements or use --detector color."
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise DetectorUnavailableError(
                f"YOLO model not found at {model_path}. Place weights locally first."
            )

        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = names.get(class_id, str(class_id))
            if class_name not in COCO_VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(box.conf.item()),
                    class_name=class_name,
                    source=self.name,
                )
            )

        return detections


def find_local_yolo_model() -> Path | None:
    env_path = os.getenv("CHALANREADY_YOLO_MODEL")
    candidates = [
        Path(env_path) if env_path else None,
        Path(__file__).resolve().parents[1] / "weights" / "yolov8n.pt",
        Path.cwd() / "yolov8n.pt",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def create_detector(kind: str = "auto") -> Detector:
    normalized = kind.lower().strip()

    if normalized == "color":
        return ColorVehicleDetector()

    if normalized == "yolo":
        model_path = find_local_yolo_model()
        if not model_path:
            raise DetectorUnavailableError(
                "No local YOLO model found. Set CHALANREADY_YOLO_MODEL or use --detector color."
            )
        return YOLOVehicleDetector(model_path)

    if normalized == "auto":
        model_path = find_local_yolo_model()
        if model_path:
            try:
                return YOLOVehicleDetector(model_path)
            except DetectorUnavailableError:
                pass
        return ColorVehicleDetector()

    raise ValueError(f"Unsupported detector '{kind}'. Use auto, yolo, or color.")
