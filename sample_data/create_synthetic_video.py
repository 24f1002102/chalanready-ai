from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


WIDTH = 960
HEIGHT = 540
FPS = 18
FRAMES = 150

OUTPUT_PATH = Path(__file__).resolve().parent / "videos" / "synthetic_stage1.mp4"


def draw_scene(frame: np.ndarray) -> None:
    frame[:] = (48, 58, 66)

    road = np.array(
        [
            (0, 170),
            (960, 120),
            (960, 410),
            (0, 460),
        ],
        dtype=np.int32,
    )
    footpath = np.array(
        [
            (0, 72),
            (960, 42),
            (960, 122),
            (0, 170),
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(frame, [road], (70, 76, 82))
    cv2.fillPoly(frame, [footpath], (96, 100, 100))
    cv2.line(frame, (0, 315), (960, 265), (180, 184, 186), 2, cv2.LINE_AA)

    for x in range(-40, WIDTH, 110):
        cv2.line(
            frame,
            (x, 248),
            (x + 52, 245),
            (210, 210, 206),
            3,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "SYNTHETIC CCTV DEMO - offline smoke test",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (238, 242, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Pedestrian / footpath zone",
        (670, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (222, 226, 226),
        1,
        cv2.LINE_AA,
    )


def draw_vehicle(
    frame: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    cx, cy = center
    width, height = size
    x1 = int(cx - width / 2)
    y1 = int(cy - height / 2)
    x2 = int(cx + width / 2)
    y2 = int(cy + height / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (18, 24, 28), 2)
    cv2.circle(frame, (x1 + 12, y2), 6, (18, 24, 28), -1)
    cv2.circle(frame, (x2 - 12, y2), 6, (18, 24, 28), -1)
    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 248, 250),
        1,
        cv2.LINE_AA,
    )


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

    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_scene(frame)

        car_x = 80 + index * 5
        bike_x = 860 - index * 4
        bike_y = 118 + int(np.sin(index / 10) * 2)
        parked_y = 112 + int(np.sin(index / 12) * 1)

        draw_vehicle(
            frame,
            (car_x, 350),
            (76, 38),
            (70, 210, 70),
            "car",
        )
        draw_vehicle(
            frame,
            (bike_x, bike_y),
            (54, 28),
            (210, 110, 35),
            "bike",
        )
        draw_vehicle(
            frame,
            (650, parked_y),
            (88, 38),
            (30, 205, 230),
            "parked",
        )

        writer.write(frame)

    writer.release()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
