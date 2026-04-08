from __future__ import annotations

import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import soundfile as sf
from PIL import Image

from tests.tools.text_processing import cosine_similarity_text
from tests.tools.types import DiffusionResponse, OmniResponse
from tests.tools.voice_analysis import _assert_pcm_int16_speech_hnr, _assert_preset_voice_gender_from_audio


def assert_image_diffusion_response(
    response,
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate image diffusion response.

    Expected request_config schema:
        {
            "request_type": "image",
            "extra_body": {
                "num_outputs_per_prompt": 1,
                "width": ...,
                "height": ...,
                ...
            }
        }
    """
    assert response.images is not None, "Image response is None"
    assert len(response.images) > 0, "No images in response"

    extra_body = request_config.get("extra_body") or {}

    num_outputs_per_prompt = extra_body.get("num_outputs_per_prompt")
    if num_outputs_per_prompt is not None:
        assert len(response.images) == num_outputs_per_prompt, (
            f"Expected {num_outputs_per_prompt} images, got {len(response.images)}"
        )

    if run_level == "advanced_model":
        width = extra_body.get("width")
        height = extra_body.get("height")

        if width is not None or height is not None:
            for img in response.images:
                assert_image_valid(img, width=width, height=height)


def assert_video_diffusion_response(
    response,
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate video diffusion response.

    Expected request_config schema:
        {
            "request_type": "video",
            "form_data": {
                "prompt": "...",
                "num_frames": ...,
                "width": ...,
                "height": ...,
                "fps": ...,
                ...
            }
        }
    """
    form_data = request_config.get("form_data", {})

    assert response.videos is not None, "Video response is None"
    assert len(response.videos) > 0, "No videos in response"

    expected_frames = _maybe_int(form_data.get("num_frames"))
    expected_width = _maybe_int(form_data.get("width"))
    expected_height = _maybe_int(form_data.get("height"))
    expected_fps = _maybe_int(form_data.get("fps"))

    for vid_bytes in response.videos:
        assert_video_valid(
            vid_bytes,
            num_frames=expected_frames,
            width=expected_width,
            height=expected_height,
            fps=expected_fps,
        )


def assert_audio_diffusion_response(
    response,
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate audio diffusion response.
    """
    raise NotImplementedError("Audio validation is not implemented yet")
    # consider using assert_audio_valid defined above


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def assert_image_valid(image: Path | Image.Image, *, width: int | None = None, height: int | None = None):
    """Assert the file is a loadable image with optional exact dimensions."""
    if isinstance(image, Path):
        assert image.exists(), f"Image not found: {image}"
        image = Image.open(image)
        image.load()
    assert image.width > 0 and image.height > 0
    if width is not None:
        assert image.width == width, f"Expected width={width}, got {image.width}"
    if height is not None:
        assert image.height == height, f"Expected height={height}, got {image.height}"
    return image


def assert_video_valid(
    video: Path | bytes | BytesIO,
    *,
    num_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> dict[str, int | float]:
    """Assert the MP4 has the expected resolution and exact frame count."""
    temp_path = None
    cap = None
    try:
        # Normalize input to file path
        if isinstance(video, Path):
            if not video.exists():
                raise AssertionError(f"Video file not found: {video}")
            video_path = str(video)
        else:
            # Create temp file for bytes/BytesIO
            suffix = ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
                if isinstance(video, bytes):
                    tmp.write(video)
                elif isinstance(video, BytesIO):
                    tmp.write(video.getvalue())
                else:
                    raise TypeError(f"Unsupported video type: {type(video)}")
                temp_path = Path(tmp.name)
                video_path = str(temp_path)

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise AssertionError(f"Failed to open video: {video_path}")

        # Extract properties
        actual_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        actual_num_frames = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            actual_num_frames += 1

        # Basic validity checks
        if actual_num_frames <= 0:
            raise AssertionError(f"Invalid frame count: {actual_num_frames} (must be > 0)")
        if actual_width <= 0 or actual_height <= 0:
            raise AssertionError(f"Invalid dimensions: {actual_width}x{actual_height} (must be > 0)")
        if actual_fps <= 0:
            raise AssertionError(f"Invalid FPS: {actual_fps} (must be > 0)")

        # Validate against expectations
        if num_frames is not None:
            expected_num_frames = (num_frames // 4) * 4 + 1
            assert actual_num_frames == expected_num_frames, (
                f"Frame count mismatch: expected {num_frames}, got {actual_num_frames}"
            )
        if width is not None:
            assert actual_width == width, f"Width mismatch: expected {width}px, got {actual_width}px"
        if height is not None:
            assert actual_height == height, f"Height mismatch: expected {height}px, got {actual_height}px"
        if fps is not None:
            # Use tolerance for float comparison (codec rounding)
            assert abs(actual_fps - fps) < 0.5, f"FPS mismatch: expected {fps}, got {actual_fps:.2f}"

        return {"num_frames": actual_num_frames, "width": actual_width, "height": actual_height, "fps": actual_fps}

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        raise

    finally:
        # Cleanup resources
        if cap is not None:
            cap.release()
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def assert_audio_valid(path: Path, *, sample_rate: int, channels: int, duration_s: float) -> None:
    """Assert the WAV has the expected sample rate, channel count, and duration."""
    assert path.exists(), f"Audio not found: {path}"
    info = sf.info(str(path))
    assert info.samplerate == sample_rate, f"Expected sample_rate={sample_rate}, got {info.samplerate}"
    assert info.channels == channels, f"Expected {channels} channel(s), got {info.channels}"
    expected_frames = int(duration_s * sample_rate)
    assert info.frames == expected_frames, (
        f"Expected {expected_frames} frames ({duration_s}s @ {sample_rate} Hz), got {info.frames}"
    )


def decode_b64_image(b64: str):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return img


def assert_omni_response(response: OmniResponse, request_config: dict[str, Any], run_level):
    """
    Validate response results.

    Args:
        response: OmniResponse object

    Raises:
        AssertionError: When the response does not meet validation criteria
    """
    assert response.success, "The request failed."
    e2e_latency = response.e2e_latency
    if e2e_latency is not None:
        print(f"the e2e latency is: {e2e_latency}")

    modalities = request_config.get("modalities", ["text", "audio"])

    if run_level == "advanced_model":
        if "audio" in modalities:
            assert response.audio_content is not None, "No audio output is generated"
            print(f"audio content is: {response.audio_content}")
            speaker = request_config.get("speaker")
            if speaker:
                _assert_preset_voice_gender_from_audio(
                    response.audio_bytes,
                    speaker,
                )

        if "text" in modalities:
            assert response.text_content is not None, "No text output is generated"
            print(f"text content is: {response.text_content}")

        # Verify image description
        word_types = ["text", "image", "audio", "video"]
        keywords_dict = request_config.get("key_words", {})
        for word_type in word_types:
            keywords = keywords_dict.get(word_type)
            if "text" in modalities:
                if keywords:
                    text_lower = response.text_content.lower()
                    assert any(str(kw).lower() in text_lower for kw in keywords), (
                        "The output does not contain any of the keywords."
                    )
            else:
                if keywords:
                    audio_lower = response.audio_content.lower()
                    assert any(str(kw).lower() in audio_lower for kw in keywords), (
                        "The output does not contain any of the keywords."
                    )

        # Verify similarity (Whisper transcript vs streamed/detokenized text)
        if "text" in modalities and "audio" in modalities:
            assert response.similarity is not None and response.similarity > 0.9, (
                "The audio content is not same as the text"
            )
            print(f"similarity is: {response.similarity}")


def assert_audio_speech_response(
    response: OmniResponse,
    request_config: dict[str, Any],
    run_level: str,
) -> None:
    """
    Validate /v1/audio/speech response: success, optional format check, transcription similarity
    and gender (non-PCM only for advanced_model), and int16 PCM HNR when response_format is pcm.
    """
    assert response.success, "The request failed."

    req_fmt = request_config.get("response_format")

    if req_fmt == "pcm" and response.audio_bytes:
        _assert_pcm_int16_speech_hnr(response.audio_bytes)
        if response.audio_format:
            assert "pcm" in response.audio_format.lower(), (
                f"Expected audio/pcm content-type, got {response.audio_format!r}"
            )

    elif req_fmt == "wav" and response.audio_format:
        assert req_fmt in response.audio_format, (
            f"The response audio format {response.audio_format} don't match the request audio format {req_fmt}"
        )

    e2e_latency = response.e2e_latency
    if e2e_latency is not None:
        print(f"the avg e2e latency is: {e2e_latency}")

    if run_level == "advanced_model" and req_fmt != "pcm":
        # Text-audio semantic similarity check (skipped for raw PCM: no Whisper transcript).
        expected_text = request_config.get("input")
        if expected_text:
            transcript = (response.audio_content or "").strip()
            print(f"audio content is: {transcript}")
            print(f"input text is: {expected_text}")
            similarity = cosine_similarity_text(transcript.lower(), expected_text.lower())
            print(f"Cosine similarity: {similarity:.3f}")
            assert similarity > 0.9, (
                f"Transcript doesn't match input: similarity={similarity:.2f}, transcript='{transcript}'"
            )

        # Voice gender consistency check (preset names in ``_PRESET_VOICE_GENDER_MAP``).
        # When the estimator returns 'unknown', we treat it as inconclusive and do NOT fail the test.
        _assert_preset_voice_gender_from_audio(
            response.audio_bytes,
            request_config.get("voice"),
        )


def assert_diffusion_response(response: DiffusionResponse, request_config: dict[str, Any], run_level: str = None):
    """
    Validate diffusion response results.

    Dispatcher that routes validation to modality-specific assert functions.

    Args:
        response: DiffusionResponse object.
        request_config: Request configuration dictionary.
        run_level: Test run level (e.g. "core_model", "advanced_model")

    Raises:
        AssertionError: When the response does not meet validation criteria
        KeyError: When the request_config does not contain necessary parameters for validation
    """
    assert response.success, "The request failed."

    e2e_latency = response.e2e_latency
    if e2e_latency is not None:
        print(f"the avg e2e is: {e2e_latency}")

    has_any_content = any(content is not None for content in (response.images, response.videos, response.audios))
    assert has_any_content, "Response contains no images, videos, or audios"

    if response.images is not None:
        assert_image_diffusion_response(
            response=response,
            request_config=request_config,
            run_level=run_level,
        )

    if response.videos is not None:
        assert_video_diffusion_response(
            response=response,
            request_config=request_config,
            run_level=run_level,
        )

    if response.audios is not None:
        assert_audio_diffusion_response(
            response=response,
            request_config=request_config,
            run_level=run_level,
        )
