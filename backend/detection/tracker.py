from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

from .detector import Detection


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    age: int = 1
    missing_frames: int = 0
    centers: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class CentroidTracker:
    """Small, dependency-free tracker for the prototype."""

    def __init__(self, max_distance: float = 85.0, max_missing: int = 12) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_track_id = 1
        self.tracks: dict[int, TrackedObject] = {}

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[TrackedObject]:
        if not self.tracks:
            for detection in detections:
                self._start_track(detection, frame_index)
            return list(self.tracks.values())

        candidate_pairs: list[tuple[float, int, int]] = []
        track_items = list(self.tracks.items())

        for detection_index, detection in enumerate(detections):
            for track_id, track in track_items:
                if track.class_name != detection.class_name:
                    continue
                distance = dist(track.center, detection.center)
                if distance <= self.max_distance:
                    candidate_pairs.append((distance, track_id, detection_index))

        candidate_pairs.sort(key=lambda item: item[0])
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for _, track_id, detection_index in candidate_pairs:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self._update_track(track_id, detections[detection_index], frame_index)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._start_track(detection, frame_index)

        for track_id in list(self.tracks):
            if track_id in matched_tracks:
                continue
            track = self.tracks[track_id]
            if track.last_seen_frame == frame_index:
                continue
            track.missing_frames += 1
            if track.missing_frames > self.max_missing:
                del self.tracks[track_id]

        return [
            track
            for track in self.tracks.values()
            if track.last_seen_frame == frame_index
        ]

    def _start_track(self, detection: Detection, frame_index: int) -> None:
        track = TrackedObject(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            class_name=detection.class_name,
            confidence=detection.confidence,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            centers=[detection.center],
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def _update_track(
        self,
        track_id: int,
        detection: Detection,
        frame_index: int,
    ) -> None:
        track = self.tracks[track_id]
        track.bbox = detection.bbox
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.last_seen_frame = frame_index
        track.age += 1
        track.missing_frames = 0
        track.centers.append(detection.center)
