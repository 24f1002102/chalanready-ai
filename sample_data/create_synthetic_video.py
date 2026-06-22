"""
Improved synthetic video — used as a FALLBACK when no real CCTV footage is available.
The real demo video is:
  sample_data/videos/stock-footage-delhi-india-jul-smooth-traffic-flow-at-intersection-with-green-signal.mov

This synthetic video has 5 vehicles demonstrating all 3 primary violations clearly,
with realistic road markings and CCTV watermark.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


WIDTH = 960
HEIGHT = 540
FPS = 25
FRAMES = 300   # 12 seconds

OUTPUT_PATH = Path(__file__).resolve().parent / "videos" / "synthetic_stage1.mp4"


def draw_scene(frame: np.ndarray, frame_idx: int) -> None:
    """Draw a realistic-looking road intersection scene."""
    # Sky gradient
    for y in range(80):
        t = y / 80
        color = (
            int(140 + t * 60),
            int(160 + t * 50),
            int(180 + t * 40),
        )
        frame[y, :] = color

    # Ground / asphalt
    frame[80:] = (52, 58, 62)

    # Main road surface
    road = np.array([(0, 160), (960, 110), (960, 430), (0, 480)], dtype=np.int32)
    cv2.fillPoly(frame, [road], (62, 68, 74))

    # Footpath (upper strip)
    footpath = np.array([(0, 68), (960, 38), (960, 112), (0, 160)], dtype=np.int32)
    cv2.fillPoly(frame, [footpath], (100, 104, 108))

    # Footpath edge line
    cv2.line(frame, (0, 160), (960, 110), (130, 134, 136), 2, cv2.LINE_AA)

    # Road centre line (dashed)
    for x in range(-30, WIDTH + 50, 90):
        offset = int(np.sin(frame_idx / 60) * 0)  # static
        cv2.line(frame, (x + offset, 295), (x + 55 + offset, 292),
                 (200, 205, 200), 2, cv2.LINE_AA)

    # Stop line (solid white horizontal)
    cv2.line(frame, (0, 268), (960, 260), (210, 215, 210), 3, cv2.LINE_AA)
    cv2.putText(frame, "STOP", (430, 257), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (210, 215, 210), 1, cv2.LINE_AA)

    # No-parking zone (right side, marked with yellow hatching)
    np_zone = [(490, 78), (790, 78), (790, 142), (490, 142)]
    cv2.polylines(frame, [np.array(np_zone, dtype=np.int32)], True, (0, 200, 255), 2)
    cv2.putText(frame, "NO PARKING", (520, 118), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # Traffic signal (top right)
    sig_x, sig_y = 810, 55
    cv2.rectangle(frame, (sig_x, sig_y), (sig_x + 22, sig_y + 58), (30, 30, 30), -1)
    red_on = frame_idx < 120 or frame_idx > 240
    cv2.circle(frame, (sig_x + 11, sig_y + 12), 8,
               (0, 0, 220) if red_on else (20, 20, 60), -1)
    cv2.circle(frame, (sig_x + 11, sig_y + 30), 8, (20, 60, 20), -1)
    cv2.circle(frame, (sig_x + 11, sig_y + 48), 8,
               (20, 20, 60) if red_on else (0, 200, 60), -1)

    # CCTV watermark
    cv2.putText(frame, "CAM-01 | MG ROAD INTERSECTION | BENGALURU",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 225, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, "SYNTHETIC DEMO",
                (790, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 80, 200), 1, cv2.LINE_AA)

    # Footpath label
    cv2.putText(frame, "FOOTPATH ZONE", (650, 86), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (210, 215, 215), 1, cv2.LINE_AA)


def draw_vehicle(
    frame: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    body_color: tuple[int, int, int],
    label: str,
    plate: str = "",
) -> None:
    cx, cy = center
    w, h = size
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)

    # Body
    cv2.rectangle(frame, (x1, y1), (x2, y2), body_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 25, 30), 2)
    # Windscreen
    cv2.rectangle(frame, (x1 + 4, y1 + 3), (x2 - 4, y1 + h // 3), (160, 190, 210), -1)
    # Wheels
    cv2.circle(frame, (x1 + 10, y2), 6, (20, 25, 30), -1)
    cv2.circle(frame, (x2 - 10, y2), 6, (20, 25, 30), -1)
    # Label
    cv2.putText(frame, label, (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 245, 250), 1, cv2.LINE_AA)
    # Plate (bottom of vehicle)
    if plate:
        px, py = cx - 20, y2 + 2
        cv2.rectangle(frame, (px, py), (px + 40, py + 12), (240, 240, 200), -1)
        cv2.putText(frame, plate, (px + 1, py + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (10, 10, 10), 1, cv2.LINE_AA)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video at {OUTPUT_PATH}")

    for idx in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_scene(frame, idx)

        # Vehicle 1: CAR going left-to-right (legal direction) on road
        car_x = 60 + idx * 3
        draw_vehicle(frame, (min(car_x, 980), 370), (82, 42), (70, 130, 200),
                     "car", "KA01AB1234")

        # Vehicle 2: MOTORCYCLE going RIGHT-TO-LEFT = WRONG-SIDE DRIVING
        bike_x = 920 - idx * 4
        draw_vehicle(frame, (max(bike_x, -40), 295), (52, 28), (210, 100, 30),
                     "moto [WRONG SIDE]", "KA05MN7890")

        # Vehicle 3: TRUCK PARKED in no-parking zone (stationary from frame 0)
        parked_x = 640 + int(np.sin(idx / 20) * 1)
        parked_y = 108 + int(np.sin(idx / 15) * 1)
        draw_vehicle(frame, (parked_x, parked_y), (94, 46), (30, 195, 220),
                     "truck [PARKED]", "KA41CD5678")

        # Vehicle 4: MOTORCYCLE on FOOTPATH (enters footpath zone early)
        foot_x = 40 + idx * 3
        foot_y = 98 + int(np.sin(idx / 8) * 3)
        draw_vehicle(frame, (min(foot_x, 960), foot_y), (46, 24), (200, 60, 200),
                     "moto [FOOTPATH]", "TN01ZZ9999")

        # Vehicle 5: CAR running STOP LINE (crosses at frame 80, during red phase)
        if idx < 80:
            stop_car_y = 390 - idx * 2
        else:
            stop_car_y = 230 - (idx - 80) * 1
        draw_vehicle(frame, (500, max(stop_car_y, 60)), (76, 38), (220, 60, 60),
                     "car [STOPLINE]", "MH12XY3456")

        writer.write(frame)

    writer.release()
    print(f"Wrote {OUTPUT_PATH}  ({FRAMES} frames @ {FPS}fps = {FRAMES/FPS:.1f}s)")


if __name__ == "__main__":
    main()
