from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

PromptAudioInput = list[tuple[Any, int]] | tuple[Any, int] | None
PromptImageInput = list[Any] | Any | None
PromptVideoInput = list[Any] | Any | None


class OmniServerParams(NamedTuple):
    model: str
    port: int | None = None
    stage_config_path: str | None = None
    server_args: list[str] | None = None
    env_dict: dict[str, str] | None = None
    use_omni: bool = True


@dataclass
class OmniResponse:
    text_content: str | None = None
    audio_data: list[str] | None = None
    audio_content: str | None = None
    audio_format: str | None = None
    audio_bytes: bytes | None = None
    similarity: float | None = None
    e2e_latency: float | None = None
    success: bool = False
    error_message: str | None = None


@dataclass
class DiffusionResponse:
    text_content: str | None = None
    images: list[Any] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None
    e2e_latency: float | None = None
    success: bool = False
    error_message: str | None = None
