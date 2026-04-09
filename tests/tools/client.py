from __future__ import annotations

import base64
import concurrent.futures
import json
import time
from io import BytesIO
from typing import Any

import requests
from openai import OpenAI, omit
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port

from tests.tools.assertions import (
    assert_audio_speech_response,
    assert_diffusion_response,
    assert_omni_response,
    decode_b64_image,
)
from tests.tools.audio_processing import _merge_base64_audio_to_segment, convert_audio_bytes_to_text
from tests.tools.text_processing import cosine_similarity_text
from tests.tools.types import DiffusionResponse, OmniResponse

logger = init_logger(__name__)


class OpenAIClientHandler:
    """
    OpenAI client handler class, encapsulating both streaming and non-streaming response processing logic.

    This class integrates OpenAI API request sending, response handling, and validation functionality,
    supporting both single request and concurrent request modes.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = get_open_port(), api_key: str = "EMPTY", run_level: str = None
    ):
        """
        Initialize the OpenAI client.

        Args:
            host: vLLM-Omni server host address
            port: vLLM-Omni server port
            api_key: API key (defaults to "EMPTY")
        """
        self.base_url = f"http://{host}:{port}"
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
        self.run_level = run_level

    def _process_stream_omni_response(self, chat_completion) -> OmniResponse:
        """
        Process streaming responses.

        Args:
            chat_completion: OpenAI streaming response object
            request_config: Request configuration dictionary

        Returns:
            OmniResponse: Processed response object
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            text_content = ""
            audio_data = []

            for chunk in chat_completion:
                for choice in chunk.choices:
                    # Get content data
                    if hasattr(choice, "delta"):
                        content = getattr(choice.delta, "content", None)
                    else:
                        content = None

                    # Get modality type
                    modality = getattr(chunk, "modality", None)

                    # Process content based on modality type
                    if modality == "audio" and content:
                        audio_data.append(content)
                    elif modality == "text" and content:
                        text_content += content if content else ""

            # Calculate end-to-end latency
            result.e2e_latency = time.perf_counter() - start_time

            # Process audio and text content
            audio_content = None
            similarity = None

            if audio_data or text_content:
                if audio_data:
                    merged_seg = _merge_base64_audio_to_segment(audio_data)
                    wav_buf = BytesIO()
                    merged_seg.export(wav_buf, format="wav")
                    result.audio_bytes = wav_buf.getvalue()
                    audio_content = convert_audio_bytes_to_text(result.audio_bytes)
                if audio_content and text_content:
                    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())

            # Populate result object
            result.text_content = text_content
            result.audio_data = audio_data
            result.audio_content = audio_content
            result.similarity = similarity
            result.success = True

        except Exception as e:
            result.error_message = f"Stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_non_stream_omni_response(self, chat_completion) -> OmniResponse:
        """
        Process non-streaming responses.

        Args:
            chat_completion: OpenAI non-streaming response object
            request_config: Request configuration dictionary

        Returns:
            OmniResponse: Processed response object
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            audio_data = None
            text_content = None

            # Iterate through all choices
            for choice in chat_completion.choices:
                # Process audio data
                if hasattr(choice.message, "audio") and choice.message.audio is not None:
                    audio_message = choice.message
                    audio_data = audio_message.audio.data

                # Process text content
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    text_content = choice.message.content

            # Calculate end-to-end latency
            result.e2e_latency = time.perf_counter() - start_time

            # Process audio and text content
            audio_content = None
            similarity = None

            if audio_data or text_content:
                if audio_data:
                    result.audio_bytes = base64.b64decode(audio_data)
                    audio_content = convert_audio_bytes_to_text(result.audio_bytes)
                if audio_content and text_content:
                    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())

            # Populate result object
            result.text_content = text_content
            result.audio_content = audio_content
            result.similarity = similarity
            result.success = True

        except Exception as e:
            result.error_message = f"Non-stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_diffusion_response(self, chat_completion) -> DiffusionResponse:
        """
        Process diffusion responses (image generation/editing).

        Args:
            chat_completion: OpenAI response object

        Returns:
            DiffusionResponse: Processed response object
        """
        result = DiffusionResponse()
        start_time = time.perf_counter()

        try:
            images = []
            # [TODO] reading video and audio output from API response for later validation

            for choice in chat_completion.choices:
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    content = choice.message.content
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                image_url = item.get("image_url", {}).get("url")
                            else:
                                image_url_obj = getattr(item, "image_url", None)
                                image_url = hasattr(image_url_obj, "url", None) if image_url_obj else None
                            if image_url and image_url.startswith("data:image"):
                                b64_data = image_url.split(",", 1)[1]
                                img = decode_b64_image(b64_data)
                                images.append(img)

            result.e2e_latency = time.perf_counter() - start_time
            result.images = images if images else None
            result.success = True

        except Exception as e:
            result.error_message = f"Diffusion response processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_stream_audio_speech_response(self, response, *, response_format: str | None = None) -> OmniResponse:
        """
        Process streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_stream_omni_response but operates on low-level
        audio bytes and produces an OmniResponse with audio_content filled
        from Whisper transcription.
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            # Aggregate all audio bytes from the streaming response.
            data = bytearray()

            # Preferred OpenAI helper.
            if hasattr(response, "iter_bytes") and callable(getattr(response, "iter_bytes")):
                for chunk in response.iter_bytes():
                    if chunk:
                        data.extend(chunk)
            else:
                # Generic iterable-of-bytes fallback (e.g., generator or list of chunks).
                try:
                    iterator = iter(response)
                except TypeError:
                    iterator = None

                if iterator is not None:
                    for chunk in iterator:
                        if not chunk:
                            continue
                        if isinstance(chunk, (bytes, bytearray)):
                            data.extend(chunk)
                        elif hasattr(chunk, "data"):
                            data.extend(chunk.data)  # type: ignore[arg-type]
                        elif hasattr(chunk, "content"):
                            data.extend(chunk.content)  # type: ignore[arg-type]
                        else:
                            raise TypeError(f"Unsupported stream chunk type: {type(chunk)}")
                else:
                    raise TypeError(f"Unsupported audio speech streaming response type: {type(response)}")

            raw_bytes = bytes(data)
            if response_format == "pcm":
                transcript = None
            else:
                transcript = convert_audio_bytes_to_text(raw_bytes)

            # Populate OmniResponse.
            result.audio_bytes = raw_bytes
            result.audio_content = transcript
            result.e2e_latency = time.perf_counter() - start_time
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            result.error_message = f"Audio speech stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_non_stream_audio_speech_response(
        self, response, *, response_format: str | None = None
    ) -> OmniResponse:
        """
        Process non-streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_non_stream_omni_response but for the binary
        audio payload returned by audio.speech.create.
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            # OpenAI non-streaming audio.speech.create returns HttpxBinaryResponseContent (.read() or .content)
            if hasattr(response, "read") and callable(getattr(response, "read")):
                raw_bytes = response.read()
            elif hasattr(response, "content"):
                raw_bytes = response.content  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported audio speech response type: {type(response)}")

            if response_format == "pcm":
                transcript = None
            else:
                transcript = convert_audio_bytes_to_text(raw_bytes)

            result.audio_bytes = raw_bytes
            result.audio_content = transcript
            result.e2e_latency = time.perf_counter() - start_time
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            result.error_message = f"Audio speech non-stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def send_omni_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send OpenAI requests.

        Args:
            request_config: Request configuration dictionary containing parameters like model, messages, stream.
                Optional ``use_audio_in_video`` (bool): when true, sets
                ``extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}`` for Qwen-Omni video+audio
                extraction.
                Optional top-level ``speaker`` (str): Qwen3-Omni preset TTS speaker name; sent as
                ``extra_body["speaker"]`` to ``chat.completions.create``.
            request_num: Number of requests, defaults to 1 (single request)

        Returns:
            List[OmniResponse]: List of response objects
        """

        responses = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", ["text", "audio"])

        extra_body: dict[str, Any] = {}
        if "speaker" in request_config:
            extra_body["speaker"] = request_config["speaker"]
        if request_config.get("use_audio_in_video"):
            mm = dict(extra_body.get("mm_processor_kwargs") or {})
            mm["use_audio_in_video"] = True
            extra_body["mm_processor_kwargs"] = mm
        extra_body_arg: dict[str, Any] | None = extra_body if extra_body else None

        create_kwargs: dict[str, Any] = {
            "model": request_config.get("model"),
            "messages": request_config.get("messages"),
            "stream": stream,
            "modalities": modalities,
        }
        if extra_body_arg is not None:
            create_kwargs["extra_body"] = extra_body_arg

        if request_num == 1:
            # Send single request
            chat_completion = self.client.chat.completions.create(**create_kwargs)

            if stream:
                response = self._process_stream_omni_response(chat_completion)
            else:
                response = self._process_non_stream_omni_response(chat_completion)

            assert_omni_response(response, request_config, run_level=self.run_level)
            responses.append(response)

        else:
            # Send concurrent requests: run create + process in worker so e2e_latency includes full round-trip.
            def _one_omni_request():
                start = time.perf_counter()
                worker_kwargs: dict[str, Any] = {
                    "model": request_config.get("model"),
                    "messages": request_config.get("messages"),
                    "modalities": modalities,
                    "stream": stream,
                }
                if extra_body_arg is not None:
                    worker_kwargs["extra_body"] = extra_body_arg
                chat_completion = self.client.chat.completions.create(**worker_kwargs)
                if stream:
                    response = self._process_stream_omni_response(chat_completion)
                else:
                    response = self._process_non_stream_omni_response(chat_completion)
                response.e2e_latency = time.perf_counter() - start
                return response

            with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                futures = [executor.submit(_one_omni_request) for _ in range(request_num)]
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    assert_omni_response(response, request_config, run_level=self.run_level)
                    responses.append(response)

        return responses

    def send_audio_speech_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Call the /v1/audio/speech endpoint using the same configuration-dict
        style as send_omni_request, but via the OpenAI Python client's
        audio.speech APIs.

        Expected keys in request_config:
          - model: model name/path (required)
          - input: text to synthesize (required)
          - response_format: audio format such as "wav" or "pcm" (optional)
          - task_type, ref_text, ref_audio: TTS-specific extras (optional, passed via extra_body)
          - timeout: request timeout in seconds (float, optional, default 120.0)
          - stream: whether to use streaming API (bool, optional, default False)
        """
        timeout = float(request_config.get("timeout", 120.0))

        model = request_config["model"]
        text_input = request_config["input"]
        stream = bool(request_config.get("stream", False))
        voice = request_config.get("voice", None)

        # Standard OpenAI param: use omit when not provided to keep default behavior.
        response_format = request_config.get("response_format", omit)

        # Qwen3-TTS custom fields, forwarded via extra_body.
        extra_body: dict[str, Any] = {}
        # Keep this list aligned with vllm_omni.entrypoints.openai.protocol.audio params.
        for key in ("task_type", "ref_text", "ref_audio", "language", "max_new_tokens"):
            if key in request_config:
                extra_body[key] = request_config[key]

        responses: list[OmniResponse] = []

        speech_fmt: str | None = None if response_format is omit else str(response_format).lower()

        if request_num == 1:
            if stream:
                # Use streaming response helper.
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                ) as resp:
                    omni_resp = self._process_stream_audio_speech_response(resp, response_format=speech_fmt)
            else:
                # Non-streaming response.
                resp = self.client.audio.speech.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                )
                omni_resp = self._process_non_stream_audio_speech_response(resp, response_format=speech_fmt)

            assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
            responses.append(omni_resp)
            return responses
        else:
            # request_num > 1: concurrent requests (use same params as single-request path)

            if stream:

                def _stream_task():
                    with self.client.audio.speech.with_streaming_response.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    ) as resp:
                        return self._process_stream_audio_speech_response(resp, response_format=speech_fmt)

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = [executor.submit(_stream_task) for _ in range(request_num)]
                    for future in concurrent.futures.as_completed(futures):
                        omni_resp = future.result()
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = []
                    for _ in range(request_num):
                        future = executor.submit(
                            self.client.audio.speech.create,
                            model=model,
                            input=text_input,
                            response_format=response_format,
                            extra_body=extra_body or None,
                            timeout=timeout,
                            voice=voice,
                        )
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        resp = future.result()
                        omni_resp = self._process_non_stream_audio_speech_response(resp, response_format=speech_fmt)
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)

        return responses

    def send_diffusion_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send OpenAI requests for diffusion models.

        Args:
            request_config: Request configuration dictionary containing parameters like model, messages
            request_num: Number of requests to send concurrently, defaults to 1 (single request)
        Returns:
            List[OmniResponse]: List of response objects
        """
        responses = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", omit)  # Most diffusion models don't require modalities param
        extra_body = request_config.get("extra_body", None)

        if stream:
            raise NotImplementedError("Streaming is not currently implemented for diffusion model e2e test")

        if request_num == 1:
            # Send single request
            chat_completion = self.client.chat.completions.create(
                model=request_config.get("model"),
                messages=request_config.get("messages"),
                extra_body=extra_body,
                modalities=modalities,
            )

            response = self._process_diffusion_response(chat_completion)
            assert_diffusion_response(response, request_config, run_level=self.run_level)
            responses.append(response)

        else:
            # Send concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                futures = []

                # Submit all request tasks
                for _ in range(request_num):
                    future = executor.submit(
                        self.client.chat.completions.create,
                        model=request_config.get("model"),
                        messages=request_config.get("messages"),
                        modalities=modalities,
                        extra_body=extra_body,
                    )
                    futures.append(future)

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    chat_completion = future.result()
                    response = self._process_diffusion_response(chat_completion)
                    assert_diffusion_response(response, request_config, run_level=self.run_level)
                    responses.append(response)

        return responses

    def send_video_diffusion_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send native /v1/videos requests.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent video diffusion requests are not currently implemented")

        if request_config.get("stream", False):
            raise NotImplementedError("Streaming is not currently implemented for video diffusion e2e test")

        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")

        if not form_data.get("prompt"):
            raise ValueError("Video request_config['form_data'] must contain 'prompt'")

        normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}

        files: dict[str, tuple[str, BytesIO, str]] = {}
        image_reference = request_config.get("image_reference")
        if image_reference:
            if image_reference.startswith("data:image"):
                header, encoded = image_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)

                files["input_reference"] = (
                    f"reference.{extension}",
                    BytesIO(file_data),
                    content_type,
                )
            else:
                normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})

        result = DiffusionResponse()
        start_time = time.perf_counter()

        try:
            create_url = self._build_url("/v1/videos")
            response = requests.post(
                create_url,
                data=normalized_form_data,
                files=files,
                headers={"Accept": "application/json"},
                timeout=60,
            )
            response.raise_for_status()

            job_data = response.json()
            video_id = job_data["id"]

            self._wait_until_video_completed(video_id)

            video_content = self._download_video_content(video_id)

            result.success = True
            result.videos = [video_content]
            result.e2e_latency = time.perf_counter() - start_time

            assert_diffusion_response(result, request_config, run_level=self.run_level)

        except Exception as e:
            result.success = False
            result.error_message = f"Diffusion response processing error: {e}"
            assert False, result.error_message

        return [result]

    def _wait_until_video_completed(
        self,
        video_id: str,
        poll_interval_seconds: int = 2,
        timeout_seconds: int = 300,
    ) -> None:
        status_url = self._build_url(f"/v1/videos/{video_id}")
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            status_resp = requests.get(
                status_url,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            status_resp.raise_for_status()

            status_data = status_resp.json()
            current_status = status_data["status"]

            if current_status == "completed":
                return

            if current_status == "failed":
                error_msg = status_data.get("last_error", "Unknown error")
                raise RuntimeError(f"Job failed: {error_msg}")

            time.sleep(poll_interval_seconds)

        raise TimeoutError(f"Video job {video_id} did not complete within {timeout_seconds}s")

    def _download_video_content(self, video_id: str) -> bytes:
        download_url = self._build_url(f"/v1/videos/{video_id}/content")
        video_resp = requests.get(download_url, stream=True, timeout=60)
        video_resp.raise_for_status()

        video_bytes = BytesIO()
        for chunk in video_resp.iter_content(chunk_size=8192):
            if chunk:
                video_bytes.write(chunk)

        return video_bytes.getvalue()

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
