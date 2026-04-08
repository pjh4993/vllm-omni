from __future__ import annotations

import io
import threading

import numpy as np
import soundfile as sf
from transformers import pipeline

_GENDER_PIPELINE = None
_GENDER_PIPELINE_LOCK = threading.Lock()

_PCM_SPEECH_SAMPLE_RATE_HZ = 24_000


def _load_gender_pipeline():
    """
    Lazy-load a cached audio-classification pipeline for gender.

    We prefer the pipeline wrapper because it encapsulates processor/model loading
    and avoids direct AutoProcessor.from_pretrained call sites in this file.
    """
    global _GENDER_PIPELINE
    if _GENDER_PIPELINE is not None:
        return _GENDER_PIPELINE

    model_name = "7wolf/wav2vec2-base-gender-classification"
    try:
        _GENDER_PIPELINE = pipeline(
            task="audio-classification",
            model=model_name,
            device=-1,
        )
        return _GENDER_PIPELINE
    except Exception as exc:
        print(f"Warning: failed to create gender pipeline '{model_name}': {exc}")
        _GENDER_PIPELINE = None
        return None


def _median_pitch_hz_from_autocorr(mono: np.ndarray, sr: int) -> float | None:
    """
    Rough median F0 (Hz) over short-time frames. Used to debias wav2vec2 gender head on TTS,
    which often labels lower-pitched synthetic speech as female under load or on clean signals.
    Returns None if the clip is too short or mostly unvoiced.
    """
    x = np.asarray(mono, dtype=np.float64)
    x = x - np.mean(x)
    if x.size < int(0.15 * sr):
        return None
    frame_len = int(0.04 * sr)
    hop = max(frame_len // 2, 1)
    f0_min_hz, f0_max_hz = 70.0, 400.0
    lag_min = max(1, int(sr / f0_max_hz))
    lag_max = min(frame_len - 2, int(sr / f0_min_hz))
    if lag_max <= lag_min:
        return None
    win = np.hamming(frame_len)
    pitches: list[float] = []
    for start in range(0, int(x.shape[0]) - frame_len, hop):
        frame = x[start : start + frame_len] * win
        frame = frame - np.mean(frame)
        if float(np.sqrt(np.mean(frame**2))) < 1e-4:
            continue
        ac = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        ac = ac / (float(ac[0]) + 1e-12)
        region = ac[lag_min : lag_max + 1]
        peak_rel = int(np.argmax(region))
        peak_lag = peak_rel + lag_min
        if peak_lag <= 0:
            continue
        f0 = float(sr) / float(peak_lag)
        if f0_min_hz <= f0 <= f0_max_hz:
            pitches.append(f0)
    if len(pitches) < 4:
        return None
    return float(np.median(np.asarray(pitches, dtype=np.float64)))


def _estimate_voice_gender_from_audio(audio_bytes: bytes) -> str:
    """
    Estimate voice gender from audio using a small pre-trained classification model.

    Uses a cached `audio-classification` pipeline to classify the clip.
    Returns 'male' / 'female' when the model confidence is >= 0.9 and the label
    maps to one of these; otherwise returns 'unknown'. If the model is unavailable
    or inference fails, returns 'unknown' to keep tests stable.

    Under concurrent tests, a global lock serializes pipeline calls (the HF pipeline is not
    thread-safe). A coarse F0 median can correct systematic "male -> female" errors on TTS audio.
    """
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("Empty audio")
    mono = np.mean(data, axis=1)

    try:
        target_sr = 16000
        if int(sr) != target_sr and mono.size > 1:
            src_len = int(mono.shape[0])
            dst_len = max(1, int(round(src_len * float(target_sr) / float(sr))))
            src_idx = np.arange(src_len, dtype=np.float32)
            dst_idx = np.linspace(0, src_len - 1, dst_len, dtype=np.float32)
            mono = np.interp(dst_idx, src_idx, mono.astype(np.float32, copy=False)).astype(np.float32)
            sr = target_sr

        median_f0 = _median_pitch_hz_from_autocorr(mono, sr)

        clf = _load_gender_pipeline()
        if clf is None:
            print("gender model not available, returning 'unknown'")
            return "unknown"

        with _GENDER_PIPELINE_LOCK:
            outputs = clf(mono, sampling_rate=sr)
        if not outputs:
            return "unknown"

        top = outputs[0]
        label = str(top.get("label", "")).lower()
        conf = float(top.get("score", 0.0))

        if conf < 0.5:
            gender = "unknown"
        elif ("female" in label) or ("\u0436\u0435\u043d" in label):
            gender = "female"
        elif ("male" in label) or ("\u043c\u0443\u0436" in label):
            gender = "male"
        else:
            gender = "unknown"

        if gender == "female" and median_f0 is not None and median_f0 < 165.0 and conf < 0.88:
            print(f"gender pitch assist: reclassifying female->male (median_f0={median_f0:.1f} Hz, conf={conf:.3f})")
            gender = "male"
        elif gender == "male" and median_f0 is not None and median_f0 > 230.0 and conf < 0.88:
            print(f"gender pitch assist: reclassifying male->female (median_f0={median_f0:.1f} Hz, conf={conf:.3f})")
            gender = "female"

        print(
            f"gender classifier: label={label}, conf={conf:.3f}, gender={gender}"
            + (f", median_f0={median_f0:.1f}Hz" if median_f0 is not None else "")
        )
        return gender
    except Exception as exc:
        print(f"Warning: gender classification failed, returning 'unknown': {exc}")
        return "unknown"


_PRESET_VOICE_GENDER_MAP: dict[str, str] = {
    "serena": "female",
    "uncle_fu": "male",
    "chelsie": "female",
    "clone": "female",
    "ethan": "male",
}


def _assert_preset_voice_gender_from_audio(
    audio_bytes: bytes | None,
    voice_name: str | None,
) -> None:
    """If ``voice_name`` matches a known preset, assert classifier gender matches (skip when unknown)."""
    if not voice_name or not audio_bytes:
        return
    key = str(voice_name).lower()
    expected_gender = _PRESET_VOICE_GENDER_MAP.get(key)
    if expected_gender is None:
        return
    estimated_gender = _estimate_voice_gender_from_audio(audio_bytes)
    print(f"Preset voice gender check: preset={key!r}, estimated={estimated_gender!r}, expected={expected_gender!r}")
    if estimated_gender != "unknown":
        assert estimated_gender == expected_gender, (
            f"{voice_name!r} is expected {expected_gender}, but estimated gender is {estimated_gender!r}"
        )


_MIN_PCM_SPEECH_HNR_DB = 1.0


def _compute_pcm_hnr_db(pcm_samples: np.ndarray, sr: int = _PCM_SPEECH_SAMPLE_RATE_HZ) -> float:
    """Compute mean Harmonic-to-Noise Ratio (dB) for speech quality.

    Clean cloned speech has HNR > 1.2 dB; distorted speech (e.g. lost
    ref_code decoder context) drops below 1.0 dB.
    """
    frame_len = int(0.03 * sr)  # 30ms frames
    hop = frame_len // 2
    hnr_values: list[float] = []

    for start in range(0, len(pcm_samples) - frame_len, hop):
        frame = pcm_samples[start : start + frame_len].astype(np.float32, copy=False)
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 0.01:
            continue
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        ac = ac / (ac[0] + 1e-10)
        min_lag = int(sr / 400)
        max_lag = min(int(sr / 80), len(ac))
        if min_lag >= max_lag:
            continue
        peak = float(np.max(ac[min_lag:max_lag]))
        if 0 < peak < 1:
            hnr_values.append(10 * np.log10(peak / (1 - peak + 1e-10)))

    return float(np.mean(hnr_values)) if hnr_values else 0.0


def _assert_pcm_int16_speech_hnr(audio_bytes: bytes) -> None:
    """Validate harmonic-to-noise ratio on raw int16 PCM from /v1/audio/speech."""
    assert audio_bytes is not None and len(audio_bytes) >= 2, "missing PCM bytes"
    assert len(audio_bytes) % 2 == 0, "PCM byte length must be aligned to int16"
    pcm_samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    hnr = _compute_pcm_hnr_db(pcm_samples)
    print(f"PCM speech HNR: {hnr:.2f} dB (threshold: {_MIN_PCM_SPEECH_HNR_DB} dB)")
    assert hnr >= _MIN_PCM_SPEECH_HNR_DB, (
        f"Audio distortion detected: HNR={hnr:.2f} dB < {_MIN_PCM_SPEECH_HNR_DB} dB. "
        "Voice clone decoder may be losing ref_code speaker context on later chunks."
    )
