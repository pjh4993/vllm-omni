from __future__ import annotations

import io
from typing import Any

import psutil
import soundfile as sf
import torch
from vllm import TextPrompt
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.logger import init_logger

from tests.tools.assertions import assert_audio_speech_response, assert_omni_response
from tests.tools.gpu_cleanup import _run_post_test_cleanup, _run_pre_test_cleanup
from tests.tools.types import OmniResponse, PromptAudioInput, PromptImageInput, PromptVideoInput
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform  # noqa: F401

logger = init_logger(__name__)


class OmniRunner:
    """
    Offline test runner for Omni models.
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        stage_init_timeout: int = 300,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        shm_threshold_bytes: int = 65536,
        log_stats: bool = False,
        stage_configs_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize an OmniRunner for testing.

        Args:
            model_name: The model name or path
            seed: Random seed for reproducibility
            stage_init_timeout: Timeout for initializing a single stage in seconds
            batch_timeout: Timeout for batching in seconds
            init_timeout: Timeout for initializing stages in seconds
            shm_threshold_bytes: Threshold for using shared memory
            log_stats: Enable detailed statistics logging
            stage_configs_path: Optional path to YAML stage config file
            **kwargs: Additional arguments passed to Omni
        """
        cleanup_dist_env_and_memory()
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        self.model_name = model_name
        self.seed = seed

        self.omni = Omni(
            model=model_name,
            log_stats=log_stats,
            stage_init_timeout=stage_init_timeout,
            batch_timeout=batch_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            stage_configs_path=stage_configs_path,
            **kwargs,
        )

    def _estimate_prompt_len(
        self,
        additional_information: dict[str, Any],
        model_name: str,
        _cache: dict[str, Any] = {},
    ) -> int:
        """Estimate prompt_token_ids placeholder length for the Talker stage.

        The AR Talker replaces all input embeddings via ``preprocess``, so the
        placeholder values are irrelevant but the **length** must match the
        embeddings that ``preprocess`` will produce.
        """
        try:
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            if model_name not in _cache:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
                cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
                _cache[model_name] = (tok, getattr(cfg, "talker_config", None))

            tok, tcfg = _cache[model_name]
            task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]
            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=task_type,
                tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
                codec_language_id=getattr(tcfg, "codec_language_id", None),
                spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
            )
        except Exception as exc:
            logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
            return 2048

    def get_default_sampling_params_list(self) -> list[OmniSamplingParams]:
        """
        Get a list of default sampling parameters for all stages.

        Returns:
            List of SamplingParams with default decoding for each stage
        """
        if not hasattr(self.omni, "default_sampling_params_list"):
            raise AttributeError("Omni.default_sampling_params_list is not available")
        return list(self.omni.default_sampling_params_list)

    def get_omni_inputs(
        self,
        prompts: list[str] | str,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[TextPrompt]:
        """
        Construct Omni input format from prompts and multimodal data.

        Args:
            prompts: Text prompt(s) - either a single string or list of strings
            system_prompt: Optional system prompt (defaults to Qwen system prompt)
            audios: Audio input(s) - tuple of (audio_array, sample_rate) or list of tuples
            images: Image input(s) - PIL Image or list of PIL Images
            videos: Video input(s) - numpy array or list of numpy arrays
            mm_processor_kwargs: Optional processor kwargs (e.g., use_audio_in_video)

        Returns:
            List of prompt dictionaries suitable for Omni.generate()
        """
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                "Group, capable of perceiving auditory and visual inputs, as well as "
                "generating text and speech."
            )

        video_padding_token = "<|VIDEO|>"
        image_padding_token = "<|IMAGE|>"
        audio_padding_token = "<|AUDIO|>"

        if "Qwen3-Omni-30B-A3B-Instruct" in self.model_name:
            video_padding_token = "<|video_pad|>"
            image_padding_token = "<|image_pad|>"
            audio_padding_token = "<|audio_pad|>"

        if isinstance(prompts, str):
            prompts = [prompts]

        # Qwen-TTS: follow examples/offline_inference/qwen3_tts/end2end.py style.
        # Stage 0 expects token placeholders + additional_information (text/speaker/task_type/...),
        # and Talker replaces embeddings in preprocess based on additional_information only.
        is_tts_model = "Qwen3-TTS" in self.model_name or "qwen3_tts" in self.model_name.lower()
        if is_tts_model and modalities == ["audio"]:
            tts_kw = mm_processor_kwargs or {}
            task_type = tts_kw.get("task_type", "CustomVoice")
            speaker = tts_kw.get("speaker", "Vivian")
            language = tts_kw.get("language", "Auto")
            max_new_tokens = int(tts_kw.get("max_new_tokens", 2048))
            ref_audio = tts_kw.get("ref_audio", None)
            ref_text = tts_kw.get("ref_text", None)

            omni_inputs: list[TextPrompt] = []
            for prompt_text in prompts:
                text_str = str(prompt_text).strip() or " "
                additional_information: dict[str, Any] = {
                    "task_type": [task_type],
                    "text": [text_str],
                    "language": [language],
                    "speaker": [speaker],
                    "max_new_tokens": [max_new_tokens],
                }
                if ref_audio is not None:
                    additional_information["ref_audio"] = [ref_audio]
                if ref_text is not None:
                    additional_information["ref_text"] = [ref_text]
                # Use official helper to get correct placeholder length
                plen = self._estimate_prompt_len(additional_information, self.model_name)
                input_dict: TextPrompt = {
                    "prompt_token_ids": [0] * plen,
                    "additional_information": additional_information,
                }
                omni_inputs.append(input_dict)
            return omni_inputs

        def _normalize_mm_input(mm_input, num_prompts):
            if mm_input is None:
                return [None] * num_prompts
            if isinstance(mm_input, list):
                if len(mm_input) != num_prompts:
                    raise ValueError(
                        f"Multimodal input list length ({len(mm_input)}) must match prompts length ({num_prompts})"
                    )
                return mm_input
            return [mm_input] * num_prompts

        num_prompts = len(prompts)
        audios_list = _normalize_mm_input(audios, num_prompts)
        images_list = _normalize_mm_input(images, num_prompts)
        videos_list = _normalize_mm_input(videos, num_prompts)

        omni_inputs = []
        for i, prompt_text in enumerate(prompts):
            user_content = ""
            multi_modal_data = {}

            audio = audios_list[i]
            if audio is not None:
                if isinstance(audio, list):
                    for _ in audio:
                        user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio
                else:
                    user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio

            image = images_list[i]
            if image is not None:
                if isinstance(image, list):
                    for _ in image:
                        user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image
                else:
                    user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image

            video = videos_list[i]
            if video is not None:
                if isinstance(video, list):
                    for _ in video:
                        user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video
                else:
                    user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video

            user_content += prompt_text

            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            input_dict: TextPrompt = {"prompt": full_prompt}
            if multi_modal_data:
                input_dict["multi_modal_data"] = multi_modal_data
            if modalities:
                input_dict["modalities"] = modalities
            if mm_processor_kwargs:
                input_dict["mm_processor_kwargs"] = mm_processor_kwargs

            omni_inputs.append(input_dict)

        return omni_inputs

    def generate(
        self,
        prompts: list[TextPrompt],
        sampling_params_list: list[OmniSamplingParams] | None = None,
    ) -> list[OmniRequestOutput]:
        """
        Generate outputs for the given prompts.

        Args:
            prompts: List of prompt dictionaries with 'prompt' and optionally
                    'multi_modal_data' keys
            sampling_params_list: List of sampling parameters for each stage.
                                 If None, uses default parameters.

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()

        return self.omni.generate(prompts, sampling_params_list)

    def generate_multimodal(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[OmniSamplingParams] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[OmniRequestOutput]:
        """
        Convenience method to generate with multimodal inputs.

        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            images: Image input(s)
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            images=images,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
            modalities=modalities,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages.

        Args:
            profile_prefix: Optional prefix for the trace file names.
            stages: List of stage IDs to profile. If None, profiles all stages.

        Returns:
            List of results from each stage.
        """
        return self.omni.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages.

        Args:
            stages: List of stage IDs to profile. If None, stops all stages.

        Returns:
            List of results from each stage.
        """
        return self.omni.stop_profile(stages=stages)

    def _cleanup_process(self):
        try:
            keywords = ["enginecore"]
            matched = []

            for proc in psutil.process_iter(["pid", "name", "cmdline", "username"]):
                try:
                    cmdline = " ".join(proc.cmdline()).lower() if proc.cmdline() else ""
                    name = proc.name().lower()

                    is_process = any(keyword in cmdline for keyword in keywords) or any(
                        keyword in name for keyword in keywords
                    )

                    if is_process:
                        print(f"Found vllm process: PID={proc.pid}, cmd={cmdline[:100]}")
                        matched.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            for proc in matched:
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            _, still_alive = psutil.wait_procs(matched, timeout=5)
            for proc in still_alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if still_alive:
                _, stubborn = psutil.wait_procs(still_alive, timeout=3)
                if stubborn:
                    print(f"Warning: failed to kill residual vllm pids: {[p.pid for p in stubborn]}")
                else:
                    print(f"Force-killed residual vllm pids: {[p.pid for p in still_alive]}")
            elif matched:
                print(f"Terminated vllm pids: {[p.pid for p in matched]}")

        except Exception as e:
            print(f"Error in psutil vllm cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if hasattr(self.omni, "close"):
            self.omni.close()
        self._cleanup_process()
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        cleanup_dist_env_and_memory()


class OmniRunnerHandler:
    def __init__(self, omni_runner):
        self.runner = omni_runner

    def _process_output(self, outputs: list[Any]) -> OmniResponse:
        result = OmniResponse()
        try:
            text_content = None
            audio_content = None
            for stage_output in outputs:
                if getattr(stage_output, "final_output_type", None) == "text":
                    text_content = stage_output.request_output.outputs[0].text
                if getattr(stage_output, "final_output_type", None) == "audio":
                    audio_content = stage_output.request_output.outputs[0].multimodal_output["audio"]

            result.audio_content = audio_content
            result.text_content = text_content
            result.success = True

        except Exception as e:
            result.error_message = f"Output processing error: {str(e)}"
            result.success = False
            print(f"Error: {result.error_message}")

        return result

    def send_request(self, request_config: dict[str, Any] | None = None) -> OmniResponse:
        if request_config is None:
            request_config = {}
        prompts = request_config.get("prompts")
        videos = request_config.get("videos")
        images = request_config.get("images")
        audios = request_config.get("audios")
        modalities = request_config.get("modalities", ["text", "audio"])
        outputs = self.runner.generate_multimodal(
            prompts=prompts, videos=videos, images=images, audios=audios, modalities=modalities
        )
        response = self._process_output(outputs)
        assert_omni_response(response, request_config, run_level="core_model")
        return response

    def send_audio_speech_request(
        self,
        request_config: dict[str, Any],
    ) -> OmniResponse:
        """
        Offline TTS: text -> audio via generate_multimodal, then validate with assert_audio_speech_response.

        request_config must contain:
          - 'input' or 'prompts': text to synthesize.
        Optional keys:
          - 'voice'       -> speaker (CustomVoice)
          - 'task_type'   -> task_type in additional_information (default: "CustomVoice")
          - 'language'    -> language in additional_information (default: "Auto")
          - 'max_new_tokens' -> max_new_tokens in additional_information (default: 2048)
          - 'response_format' -> desired audio format (used only for assertion)
        """
        input_text = request_config.get("input") or request_config.get("prompts")
        if input_text is None:
            raise ValueError("request_config must contain 'input' or 'prompts' for TTS")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        # Build TTS-specific kwargs passed through to get_omni_inputs for Qwen3-TTS,
        # matching examples/offline_inference/qwen3_tts/end2end.py.
        mm_processor_kwargs: dict[str, Any] = {}
        if "voice" in request_config:
            mm_processor_kwargs["speaker"] = request_config["voice"]
        if "task_type" in request_config:
            mm_processor_kwargs["task_type"] = request_config["task_type"]
        if "ref_audio" in request_config:
            mm_processor_kwargs["ref_audio"] = request_config["ref_audio"]
        if "ref_text" in request_config:
            mm_processor_kwargs["ref_text"] = request_config["ref_text"]
        if "language" in request_config:
            mm_processor_kwargs["language"] = request_config["language"]
        if "max_new_tokens" in request_config:
            mm_processor_kwargs["max_new_tokens"] = request_config["max_new_tokens"]

        outputs = self.runner.generate_multimodal(
            prompts=input_text,
            modalities=["audio"],
            mm_processor_kwargs=mm_processor_kwargs or None,
        )
        mm_out: dict[str, Any] | None = None
        for stage_out in outputs:
            if getattr(stage_out, "final_output_type", None) == "audio":
                mm_out = stage_out.request_output.outputs[0].multimodal_output
                break
        if mm_out is None:
            result = OmniResponse(success=False, error_message="No audio output from pipeline")
            assert result.success, result.error_message
            return result

        audio_data = mm_out.get("audio")
        if audio_data is None:
            result = OmniResponse(success=False, error_message="No audio tensor in multimodal output")
            assert result.success, result.error_message
            return result

        sr_raw = mm_out.get("sr")
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
        wav_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
        wav_buf = io.BytesIO()
        sf.write(
            wav_buf,
            wav_tensor.float().cpu().numpy().reshape(-1),
            samplerate=sr,
            format="WAV",
            subtype="PCM_16",
        )
        result = OmniResponse(success=True, audio_bytes=wav_buf.getvalue(), audio_format="audio/wav")
        assert_audio_speech_response(result, request_config, run_level="core_model")
        return result

    def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages."""
        return self.runner.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages."""
        return self.runner.stop_profile(stages=stages)
