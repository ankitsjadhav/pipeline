from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SceneType = Literal["A", "B", "C"]
OverlayType = Literal["app_screenshot", "symbol_text", "logo_only", "null"]
TextSize = Literal["large", "medium", "small"]
Position = Literal[
    "top_left",
    "top_center",
    "top_right",
    "center",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]
CameraMovement = Literal[
    "slow_zoom_in",
    "slow_zoom_out",
    "pan_left",
    "pan_right",
    "pan_top_bottom",
    "diagonal_drift",
]
ColorGrade = Literal["warm_golden", "cool_mystical", "dark_cinematic", "soft_warm"]
Transition = Literal["crossfade", "fadeblack", "wipeleft"]


class Overlay(BaseModel):
    type: str | None = None  # allow unknown; validated later
    mockup_file: str | None = None
    screenshot_file: str | None = None
    symbol_file: str | None = None
    symbol_position: str | None = None
    symbol_opacity: float | None = Field(default=0.65, ge=0.0, le=1.0)
    text: str | None = None
    text_language: str | None = None
    text_position: str | None = None
    text_color: str | None = None
    text_size: str | None = None


class Scene(BaseModel):
    id: int
    type: SceneType
    duration_sec: int = Field(ge=1, le=15)
    voiceover_line: str
    flux_prompt: str | None = None
    camera_movement: str
    color_grade: str
    transition_out: str
    overlay: Overlay = Field(default_factory=Overlay)


class Plan(BaseModel):
    app_name: str
    app_niche: str
    video_topic: str
    dominant_mood: str
    voiceover_full: str
    scenes: list[Scene]

