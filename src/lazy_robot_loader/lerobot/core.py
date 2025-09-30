from __future__ import annotations
from typing import TypedDict


class LeRobotDatasetFeature(TypedDict):
    dtype: str
    shape: list[int]
    names: list[str] | None


LeRobotDatasetVideoFeatureInfo = TypedDict(
    "LeRobotDatasetVideoFeatureInfo",
    {
        "video.fps": int,
        "video.codec": str,
        "video.pix_fmt": str,
        "video.is_depth_map": bool,
        "has_audio": bool,
    },
)


class LeRobotDatasetVideoFeature(LeRobotDatasetFeature):
    info: LeRobotDatasetVideoFeatureInfo


class LeRobotDatasetInfo(TypedDict):
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    splits: dict[str, str]
    data_path: str
    video_path: str
    features: dict[str, LeRobotDatasetFeature]
