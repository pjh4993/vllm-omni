from __future__ import annotations

import base64
import datetime
import io
import math
import os
import random
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf
from vllm.logger import init_logger

logger = init_logger(__name__)


def dummy_messages_from_mix_data(
    system_prompt: dict[str, Any] = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""

    if content_text is not None:
        content = [{"type": "text", "text": content_text}]
    else:
        content = []

    media_items = []
    if isinstance(video_data_url, list):
        for video_url in video_data_url:
            media_items.append((video_url, "video"))
    else:
        media_items.append((video_data_url, "video"))

    if isinstance(image_data_url, list):
        for url in image_data_url:
            media_items.append((url, "image"))
    else:
        media_items.append((image_data_url, "image"))

    if isinstance(audio_data_url, list):
        for url in audio_data_url:
            media_items.append((url, "audio"))
    else:
        media_items.append((audio_data_url, "audio"))

    content.extend(
        {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
        for url, media_type in media_items
        if url is not None
    )
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages = [system_prompt] + messages
    return messages


def generate_synthetic_audio(
    duration: int,  # seconds
    num_channels: int,  # 1：Mono，2：Stereo 5：5.1 surround sound
    sample_rate: int = 48000,  # Default use 48000Hz.
    save_to_file: bool = False,
) -> dict[str, Any]:
    """
    Generate TTS speech with pyttsx3 and return base64 string.
    """

    import pyttsx3
    import soundfile as sf

    def _pick_voice(engine: pyttsx3.Engine) -> str | None:
        voices = engine.getProperty("voices")
        if not voices:
            return None

        preferred_tokens = (
            "natural",
            "jenny",
            "sonia",
            "susan",
            "zira",
            "aria",
            "hazel",
            "samantha",
            "ava",
            "allison",
            "female",
            "woman",
            "english-us",
            "en-us",
            "english",
        )
        discouraged_tokens = (
            "espeak",
            "robot",
            "mbrola",
            "microsoft david",
            "male",
            "man",
        )

        best_voice = voices[0]
        best_score = float("-inf")
        for voice in voices:
            voice_text = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            voice_languages = " ".join(
                lang.decode(errors="ignore") if isinstance(lang, bytes) else str(lang)
                for lang in getattr(voice, "languages", [])
            ).lower()
            combined_text = f"{voice_text} {voice_languages}"
            score = 0
            for idx, token in enumerate(preferred_tokens):
                if token in combined_text:
                    score += 20 - idx
            for token in discouraged_tokens:
                if token in combined_text:
                    score -= 10
            if "english" in combined_text or "en_" in combined_text or "en-" in combined_text:
                score += 4
            if "en-us" in combined_text or "english-us" in combined_text:
                score += 4
            if score > best_score:
                best_score = score
                best_voice = voice

        return best_voice.id

    def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or len(audio) == 0:
            return audio.astype(np.float32)

        src_len = audio.shape[0]
        dst_len = max(1, int(round(src_len * float(dst_sr) / float(src_sr))))
        src_idx = np.arange(src_len, dtype=np.float32)
        dst_idx = np.linspace(0, src_len - 1, dst_len, dtype=np.float32)

        resampled_channels: list[np.ndarray] = []
        for ch in range(audio.shape[1]):
            resampled_channels.append(np.interp(dst_idx, src_idx, audio[:, ch]).astype(np.float32))
        return np.stack(resampled_channels, axis=1)

    def _match_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
        current_channels = audio.shape[1]
        if current_channels == target_channels:
            return audio.astype(np.float32)
        if target_channels == 1:
            return np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
        if current_channels == 1:
            return np.repeat(audio, target_channels, axis=1).astype(np.float32)

        collapsed = np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
        return np.repeat(collapsed, target_channels, axis=1).astype(np.float32)

    def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        if len(audio) == 0:
            return audio
        energy = np.max(np.abs(audio), axis=1)
        voiced = np.where(energy > threshold)[0]
        if len(voiced) == 0:
            return audio
        start = max(0, int(voiced[0]) - int(sample_rate * 0.02))
        end = min(len(audio), int(voiced[-1]) + int(sample_rate * 0.04) + 1)
        return audio[start:end]

    def _enhance_speech(audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio.astype(np.float32)
        enhanced = audio.astype(np.float32).copy()
        enhanced -= np.mean(enhanced, axis=0, keepdims=True, dtype=np.float32)
        if len(enhanced) > 1:
            preemphasis = enhanced.copy()
            preemphasis[1:] = enhanced[1:] - 0.94 * enhanced[:-1]
            enhanced = 0.7 * enhanced + 0.3 * preemphasis
        # Mild dynamic-range compression for ASR/TTS robustness.
        enhanced = np.sign(enhanced) * np.sqrt(np.abs(enhanced))
        # Light fade to avoid clicks after trimming/repeating.
        fade = min(len(enhanced) // 4, max(1, int(sample_rate * 0.01)))
        if fade > 1:
            ramp_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            enhanced[:fade] *= ramp_in[:, None]
            enhanced[-fade:] *= ramp_out[:, None]
        peak = float(np.max(np.abs(enhanced)))
        if peak > 1e-8:
            enhanced = enhanced / peak * 0.95
        return enhanced.astype(np.float32)

    phrase_text = "test"
    num_samples = int(sample_rate * max(1, duration))
    audio_data = np.zeros((num_samples, num_channels), dtype=np.float32)

    engine = pyttsx3.init()
    engine.setProperty("rate", 112)
    engine.setProperty("volume", 1.0)
    selected_voice = _pick_voice(engine)
    if selected_voice is not None:
        engine.setProperty("voice", selected_voice)

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()

    try:
        engine.save_to_file(phrase_text, temp_wav.name)
        engine.runAndWait()
        engine.stop()

        ready = False
        for _ in range(50):
            if os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 44:
                ready = True
                break
            time.sleep(0.1)

        if not ready:
            raise RuntimeError("pyttsx3 did not produce a WAV file in time.")

        tts_audio, tts_sr = sf.read(temp_wav.name, dtype="float32", always_2d=True)
    finally:
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)

    if len(tts_audio) == 0:
        raise RuntimeError("pyttsx3 produced an empty WAV file.")

    tts_audio = _resample_audio(tts_audio, tts_sr, sample_rate)
    tts_audio = _match_channels(tts_audio, num_channels)
    tts_audio = _trim_silence(tts_audio, threshold=0.012)
    tts_audio = _enhance_speech(tts_audio)

    lead_silence = min(int(sample_rate * 0.02), num_samples // 8)
    pause_samples = int(sample_rate * 0.18)
    start = lead_silence
    phrase_len = tts_audio.shape[0]

    while start < num_samples:
        take = min(phrase_len, num_samples - start)
        audio_data[start : start + take] = tts_audio[:take]
        start += phrase_len + pause_samples

    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp > 0:
        audio_data = audio_data / max_amp * 0.95

    audio_bytes: bytes | None = None
    output_path: str | None = None
    result: dict[str, Any] = {
        "np_array": audio_data.copy(),
    }

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"audio_{num_channels}ch_{timestamp}.wav"

        try:
            sf.write(output_path, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            print(f"Audio saved: {output_path}")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            print(f"Save failed: {e}")
            save_to_file = False

    # If not saving or save failed, create in memory
    if not save_to_file or audio_bytes is None:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        audio_bytes = buffer.read()

    # Return result
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    result["base64"] = base64_audio
    # Always include file_path to avoid KeyError in callers.
    result["file_path"] = output_path if save_to_file and output_path else None

    return result


def _mux_mp4_bytes_with_synthetic_audio(
    video_mp4_bytes: bytes,
    *,
    num_frames: int,
    fps: float = 30.0,
    sample_rate: int = 48000,
) -> bytes:
    """
    Mux a video-only MP4 with mono TTS audio from :func:`generate_synthetic_audio` (AAC).

    Audio length is at least the video duration in whole seconds (rounded up); ffmpeg
    ``-shortest`` trims to the video when the WAV is longer.

    Uses ffmpeg from ``imageio_ffmpeg`` when available, else ``ffmpeg`` on PATH.
    If TTS or mux fails, returns ``video_mp4_bytes`` unchanged.

    Mux subprocess does **not** use ``capture_output=True``: ffmpeg can block writing
    to a full stderr pipe while :func:`subprocess.run` waits for exit (classic deadlock).
    """
    duration_sec = num_frames / fps if fps > 0 else 0.0
    # generate_synthetic_audio(duration=int) uses at least 1s of buffer internally
    duration_int = max(1, int(math.ceil(duration_sec)))

    try:
        audio_result = generate_synthetic_audio(
            duration=duration_int,
            num_channels=1,
            sample_rate=sample_rate,
            save_to_file=False,
        )
        audio_pcm = audio_result["np_array"]
    except Exception as e:
        logger.warning("Synthetic video: generate_synthetic_audio failed (%s); using video-only MP4.", e)
        return video_mp4_bytes

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="syn_vid_mux_") as tmp:
            vid_path = os.path.join(tmp, "video.mp4")
            wav_path = os.path.join(tmp, "audio.wav")
            out_path = os.path.join(tmp, "out.mp4")
            with open(vid_path, "wb") as f:
                f.write(video_mp4_bytes)
            sf.write(wav_path, audio_pcm, sample_rate, format="WAV", subtype="PCM_16")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                vid_path,
                "-i",
                wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                "-movflags",
                "+faststart",
                out_path,
            ]
            subprocess.run(
                cmd,
                check=True,
                stdin=subprocess.DEVNULL,
                timeout=300,
            )
            with open(out_path, "rb") as f:
                return f.read()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ) as e:
        logger.warning("Synthetic video: audio mux failed (%s); using video-only MP4.", e)
        return video_mp4_bytes


def generate_synthetic_video(
    width: int,
    height: int,
    num_frames: int,
    save_to_file: bool = False,
    *,
    embed_audio: bool = False,
) -> dict[str, Any]:
    """Generate synthetic video with bouncing balls and base64 MP4.

    When ``embed_audio`` is True, muxes mono AAC from :func:`generate_synthetic_audio`
    (TTS + ffmpeg) into the MP4; otherwise returns video-only MP4 (faster when tests do
    not need an audio track).
    """

    import cv2
    import imageio

    # Create random balls
    num_balls = random.randint(3, 8)
    balls = []

    for _ in range(num_balls):
        radius = min(width, height) // 8
        if radius < 1:
            raise ValueError(f"Video dimensions ({width}x{height}) are too small for synthetic video generation")
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)

        speed = random.uniform(3.0, 8.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        # OpenCV uses BGR format, but imageio expects RGB
        # We'll create in BGR first, then convert to RGB later
        color_bgr = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        balls.append({"x": x, "y": y, "vx": vx, "vy": vy, "radius": radius, "color_bgr": color_bgr})

    # Generate video frames
    video_frames = []

    for frame_idx in range(num_frames):
        # Create black background (BGR format)
        frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)

        for ball in balls:
            # Update position
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"]

            # Boundary collision detection
            if ball["x"] - ball["radius"] <= 0 or ball["x"] + ball["radius"] >= width:
                ball["vx"] = -ball["vx"]
                ball["x"] = max(ball["radius"], min(width - ball["radius"], ball["x"]))

            if ball["y"] - ball["radius"] <= 0 or ball["y"] + ball["radius"] >= height:
                ball["vy"] = -ball["vy"]
                ball["y"] = max(ball["radius"], min(height - ball["radius"], ball["y"]))

            # Use cv2 to draw circle
            x, y = int(ball["x"]), int(ball["y"])
            radius = ball["radius"]

            # Draw solid circle (main circle)
            cv2.circle(frame_bgr, (x, y), radius, ball["color_bgr"], -1)

            # Add simple 3D effect: draw a brighter center
            if radius > 3:  # Only add highlight when radius is large enough
                highlight_radius = max(1, radius // 2)
                highlight_x = max(highlight_radius, min(x - radius // 4, width - highlight_radius))
                highlight_y = max(highlight_radius, min(y - radius // 4, height - highlight_radius))

                # Create highlight color (brighter)
                highlight_color = tuple(min(c + 40, 255) for c in ball["color_bgr"])
                cv2.circle(frame_bgr, (highlight_x, highlight_y), highlight_radius, highlight_color, -1)

        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frames.append(frame_rgb)

    video_array = np.array(video_frames)
    result = {
        "np_array": video_array,
    }
    saved_file_path = None

    fps = 30
    buffer = io.BytesIO()
    writer_kwargs = {
        "format": "mp4",
        "fps": fps,
        "codec": "libx264",
        "quality": 7,
        "pixelformat": "yuv420p",
        "macro_block_size": 16,
        "ffmpeg_params": [
            "-preset",
            "medium",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={width}:{height}",
        ],
    }

    try:
        with imageio.get_writer(buffer, **writer_kwargs) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        buffer.seek(0)
        video_only_bytes = buffer.read()
    except Exception as e:
        print(f"Warning: Failed to encode synthetic video: {e}")
        raise

    if embed_audio:
        video_bytes = _mux_mp4_bytes_with_synthetic_audio(video_only_bytes, num_frames=num_frames, fps=float(fps))
    else:
        video_bytes = video_only_bytes

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_{width}x{height}_{timestamp}.mp4"
        try:
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            saved_file_path = output_path
            print(f"Video saved to: {saved_file_path}")
        except Exception as e:
            print(f"Warning: Failed to save video to file {output_path}: {e}")

    base64_video = base64.b64encode(video_bytes).decode("utf-8")

    result["base64"] = base64_video
    if save_to_file and saved_file_path:
        result["file_path"] = saved_file_path

    return result


def generate_synthetic_image(width: int, height: int, save_to_file: bool = False) -> dict[str, Any]:
    """Generate synthetic image with randomly colored squares and return base64 string."""
    from PIL import Image, ImageDraw

    # Create white background
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Generate random number of squares
    num_squares = random.randint(3, 8)

    for _ in range(num_squares):
        # Random square size
        square_size = random.randint(min(width, height) // 8, min(width, height) // 4)

        # Random position
        x = random.randint(0, width - square_size - 1)
        y = random.randint(0, height - square_size - 1)

        # Random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Random border width
        border_width = random.randint(1, 5)

        # Draw square
        draw.rectangle([x, y, x + square_size, y + square_size], fill=color, outline=(0, 0, 0), width=border_width)

    image_array = np.array(image)
    result = {"np_array": image_array.copy()}

    # Handle file saving
    image_bytes = None
    saved_file_path = None

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"image_{width}x{height}_{timestamp}.jpg"

        try:
            # Save image to file
            image.save(output_path, format="JPEG", quality=85, optimize=True)
            saved_file_path = output_path
            print(f"Image saved to: {saved_file_path}")

            # Read file for base64 encoding
            with open(output_path, "rb") as f:
                image_bytes = f.read()

        except Exception as e:
            print(f"Warning: Failed to save image to file {output_path}: {e}")
            save_to_file = False

    # If not saving or save failed, create in memory
    if not save_to_file or image_bytes is None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)
        image_bytes = buffer.read()

    # Generate base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Return result
    result["base64"] = base64_image
    if save_to_file and saved_file_path:
        result["file_path"] = saved_file_path

    return result
