from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


Point = tuple[int, int]

CONFIG_PATH = Path(__file__).with_name("camera_profiles.json")


@dataclass(frozen=True)
class CameraProfile:
    camera_id: str
    display_name: str
    zone_name: str
    profile_kind: str
    gps_lat: float
    gps_lng: float
    frame_width: int
    frame_height: int
    enabled_rules: frozenset[str]
    road_polygon: list[Point]
    footpath_zone: list[Point]
    restricted_parking_zone: list[Point]
    allowed_direction: tuple[float, float]
    stop_line_y: int
    approach_direction: str
    signal_roi: tuple[float, float]
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CameraProfile":
        geometry = raw.get("geometry", {})
        camera_id = str(raw["camera_id"])
        profile_kind = str(raw.get("profile_kind") or _derive_profile_kind(camera_id))

        def points(key: str) -> list[Point]:
            return [(int(x), int(y)) for x, y in geometry.get(key, [])]

        allowed = geometry.get("allowed_direction", [1.0, 0.0])
        roi = geometry.get("signal_roi", [0.0, 0.40])
        return cls(
            camera_id=camera_id,
            display_name=str(raw.get("display_name", camera_id)),
            zone_name=str(raw.get("zone_name", camera_id)),
            profile_kind=profile_kind,
            gps_lat=float(raw.get("gps_lat", 12.9716)),
            gps_lng=float(raw.get("gps_lng", 77.5946)),
            frame_width=int(raw.get("frame_width", 960)),
            frame_height=int(raw.get("frame_height", 540)),
            enabled_rules=frozenset(str(v) for v in raw.get("enabled_rules", [])),
            road_polygon=points("road_polygon"),
            footpath_zone=points("footpath_zone"),
            restricted_parking_zone=points("restricted_parking_zone"),
            allowed_direction=(float(allowed[0]), float(allowed[1])),
            stop_line_y=int(geometry.get("stop_line_y", 0)),
            approach_direction=str(geometry.get("approach_direction", "down")),
            signal_roi=(float(roi[0]), float(roi[1])),
            notes=str(raw.get("notes", "")),
        )

    def scale_points(self, points: list[Point], width: int, height: int) -> list[Point]:
        sx = width / max(self.frame_width, 1)
        sy = height / max(self.frame_height, 1)
        return [(int(x * sx), int(y * sy)) for x, y in points]

    def scale_y(self, y: int, height: int) -> int:
        return int(y * height / max(self.frame_height, 1))

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "display_name": self.display_name,
            "zone_name": self.zone_name,
            "profile_kind": self.profile_kind,
            "calibration_warning": self.calibration_warning,
            "gps_lat": self.gps_lat,
            "gps_lng": self.gps_lng,
            "lat": self.gps_lat,  # legacy compat
            "lng": self.gps_lng,  # legacy compat
            "enabled_rules": sorted(self.enabled_rules),
            "notes": self.notes,
        }

    @property
    def calibration_warning(self) -> str | None:
        if self.profile_kind == "generic":
            return "Using Bengaluru Generic Profile. Accuracy may improve with camera calibration."
        if self.profile_kind == "synthetic":
            return "Synthetic demo calibration; use only for generated validation clips."
        if self.profile_kind == "demo":
            return "Demo clip calibration; use a real calibrated profile for field footage."
        return None


def _derive_profile_kind(camera_id: str) -> str:
    if camera_id == "bengaluru_generic_upload":
        return "generic"
    if "synthetic" in camera_id:
        return "synthetic"
    if "demo" in camera_id:
        return "demo"
    return "calibrated"


@lru_cache(maxsize=1)
def _load_raw_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_camera_profiles() -> dict[str, CameraProfile]:
    raw = _load_raw_config()
    profiles = [CameraProfile.from_dict(item) for item in raw.get("profiles", [])]
    return {profile.camera_id: profile for profile in profiles}


def get_default_camera_id() -> str:
    raw = _load_raw_config()
    return str(raw.get("default_camera_id", "bengaluru_generic_upload"))


def get_camera_profile(camera_id: str | None = None) -> CameraProfile:
    profiles = get_camera_profiles()
    selected_id = camera_id or get_default_camera_id()
    if selected_id in profiles:
        return profiles[selected_id]
    return profiles[get_default_camera_id()]


def get_zone_summary() -> list[dict[str, Any]]:
    return [profile.to_api_dict() for profile in get_camera_profiles().values()]
