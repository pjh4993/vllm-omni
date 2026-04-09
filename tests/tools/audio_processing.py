from __future__ import annotations

import base64
import concurrent.futures
import gc
import io
import multiprocessing
import uuid

import soundfile as sf

from vllm_omni.platforms import current_omni_platform


def convert_audio_to_text(audio_data):
    """
    Convert base64 encoded audio data to text using speech recognition.
    """
    audio_data = base64.b64decode(audio_data)
    output_path = f"./test_{uuid.uuid4().hex}.wav"
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_data)

    print(f"audio data is saved: {output_path}")
    text = convert_audio_file_to_text(output_path=output_path)
    return text


def _merge_base64_audio_to_segment(base64_list: list[str]):
    """Merge a list of base64-encoded audio chunks into one pydub AudioSegment."""
    from pydub import AudioSegment

    merged = None
    for b64 in base64_list:
        raw = base64.b64decode(b64.split(",", 1)[-1])
        seg = AudioSegment.from_file(io.BytesIO(raw))
        merged = seg if merged is None else merged + seg
    return merged


def _whisper_transcribe_in_current_process(output_path: str) -> str:
    import whisper

    device_index = None
    if current_omni_platform.is_available():
        n = current_omni_platform.get_device_count()
        if n == 1:
            device_index = 0
        elif n > 1:
            device_index = n - 1

    if device_index is not None:
        torch_device = current_omni_platform.get_torch_device(device_index)
        current_omni_platform.set_device(torch_device)
        device = str(torch_device)
        use_accelerator = True
    else:
        use_accelerator = False
        device = "cpu"
    model = whisper.load_model("small", device=device)
    try:
        text = model.transcribe(
            output_path,
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
        )["text"]
    finally:
        del model
        gc.collect()
        if use_accelerator:
            current_omni_platform.synchronize()
            current_omni_platform.empty_cache()

    return text or ""


def convert_audio_file_to_text(output_path: str) -> str:
    """Convert an audio file to text in an isolated subprocess."""
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(_whisper_transcribe_in_current_process, output_path)
        return future.result()


def convert_audio_bytes_to_text(raw_bytes: bytes) -> str:
    """
    Write container audio bytes (WAV, etc.) to a temp WAV file suitable for Whisper/ffmpeg.
    Normalizes with soundfile to PCM_16 WAV when possible to avoid codec issues.
    """
    output_path = f"./test_{uuid.uuid4().hex}.wav"
    data, samplerate = sf.read(io.BytesIO(raw_bytes))
    sf.write(output_path, data, samplerate, format="WAV", subtype="PCM_16")
    text = convert_audio_file_to_text(output_path)
    return text
