"""Shared test utilities for vLLM-Omni.

This package extracts importable helpers from the root conftest.py so that
``conftest`` files contain only pytest fixtures and hooks.
"""

from tests.tools.assertions import (
    assert_audio_diffusion_response,
    assert_audio_speech_response,
    assert_audio_valid,
    assert_diffusion_response,
    assert_image_diffusion_response,
    assert_image_valid,
    assert_omni_response,
    assert_video_diffusion_response,
    assert_video_valid,
    decode_b64_image,
)
from tests.tools.audio_processing import (
    convert_audio_bytes_to_text,
    convert_audio_file_to_text,
    convert_audio_to_text,
)
from tests.tools.client import OpenAIClientHandler
from tests.tools.config_utils import modify_stage_config
from tests.tools.gpu_cleanup import (
    _print_gpu_processes,
    _run_post_test_cleanup,
    _run_pre_test_cleanup,
)
from tests.tools.media_generators import (
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.tools.runner import OmniRunner, OmniRunnerHandler
from tests.tools.server import OmniServer
from tests.tools.text_processing import cosine_similarity_text, preprocess_text
from tests.tools.types import (
    DiffusionResponse,
    OmniResponse,
    OmniServerParams,
    PromptAudioInput,
    PromptImageInput,
    PromptVideoInput,
)
from tests.tools.voice_analysis import (
    _assert_pcm_int16_speech_hnr,
    _assert_preset_voice_gender_from_audio,
    _compute_pcm_hnr_db,
    _estimate_voice_gender_from_audio,
)

__all__ = [
    "OmniServerParams",
    "OmniResponse",
    "DiffusionResponse",
    "PromptAudioInput",
    "PromptImageInput",
    "PromptVideoInput",
    "assert_image_diffusion_response",
    "assert_video_diffusion_response",
    "assert_audio_diffusion_response",
    "assert_image_valid",
    "assert_video_valid",
    "assert_audio_valid",
    "decode_b64_image",
    "assert_omni_response",
    "assert_audio_speech_response",
    "assert_diffusion_response",
    "preprocess_text",
    "cosine_similarity_text",
    "_estimate_voice_gender_from_audio",
    "_assert_preset_voice_gender_from_audio",
    "_compute_pcm_hnr_db",
    "_assert_pcm_int16_speech_hnr",
    "convert_audio_to_text",
    "convert_audio_file_to_text",
    "convert_audio_bytes_to_text",
    "dummy_messages_from_mix_data",
    "generate_synthetic_audio",
    "generate_synthetic_video",
    "generate_synthetic_image",
    "modify_stage_config",
    "_run_pre_test_cleanup",
    "_run_post_test_cleanup",
    "_print_gpu_processes",
    "OmniServer",
    "OpenAIClientHandler",
    "OmniRunner",
    "OmniRunnerHandler",
]
