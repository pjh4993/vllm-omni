import os
import threading
from collections.abc import Generator
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Set CPU device for CI environments without GPU
if "VLLM_TARGET_DEVICE" not in os.environ:
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"

import pytest
import torch
import yaml

# ---------------------------------------------------------------------------
# Re-export every public symbol that test files previously imported from here
# (``from tests.conftest import X``).  The canonical home is now
# ``tests.tools``; these re-exports keep existing imports working.
# ---------------------------------------------------------------------------
from tests.tools import (  # noqa: F401, E402
    OmniRunner,
    OmniRunnerHandler,
    OmniServer,
    OpenAIClientHandler,
    _assert_pcm_int16_speech_hnr,
    _assert_preset_voice_gender_from_audio,
    _compute_pcm_hnr_db,
    _estimate_voice_gender_from_audio,
    _print_gpu_processes,
    _run_post_test_cleanup,
    _run_pre_test_cleanup,
    assert_audio_diffusion_response,
    assert_audio_speech_response,
    assert_audio_valid,
    assert_diffusion_response,
    assert_image_diffusion_response,
    assert_image_valid,
    assert_omni_response,
    assert_video_diffusion_response,
    assert_video_valid,
    convert_audio_bytes_to_text,
    convert_audio_file_to_text,
    convert_audio_to_text,
    cosine_similarity_text,
    decode_b64_image,
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
    preprocess_text,
)
from tests.tools.types import (  # noqa: F401, E402
    DiffusionResponse,
    OmniResponse,
    OmniServerParams,
    PromptAudioInput,
    PromptImageInput,
    PromptVideoInput,
)

# ---------------------------------------------------------------------------
# Fixtures & hooks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_prefix() -> str:
    """Optional model-path prefix from MODEL_PREFIX env var.
    Useful if models are downloaded to non-default local directories.
    """
    prefix = os.environ.get("MODEL_PREFIX", "")
    return f"{prefix.rstrip('/')}/" if prefix else ""


@pytest.fixture(autouse=True)
def default_vllm_config():
    """Set a default VllmConfig for all tests.

    This fixture is auto-used for all tests to ensure that any test
    that directly instantiates vLLM CustomOps (e.g., RMSNorm, LayerNorm)
    or model components has the required VllmConfig context.

    This fixture is required for vLLM 0.14.0+ where CustomOp initialization
    requires a VllmConfig context set via set_current_vllm_config().
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    # Use CPU device if no GPU is available (e.g., in CI environments)
    has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if has_gpu else "cpu"
    device_config = DeviceConfig(device=device)

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


@pytest.fixture(autouse=True)
def clean_gpu_memory_between_tests():
    print("\n=== PRE-TEST GPU CLEANUP ===")
    _run_pre_test_cleanup()
    yield
    _run_post_test_cleanup()


@pytest.fixture(autouse=True)
def log_test_name_before_test(request):
    print(f"--- Running test: {request.node.name}")
    yield


def pytest_addoption(parser):
    parser.addoption(
        "--run-level",
        action="store",
        default="core_model",
        choices=["core_model", "advanced_model"],
        help="Test level to run: L2, L3",
    )


@pytest.fixture(scope="session")
def run_level(request) -> str:
    """A command-line argument that specifies the level of tests to run in this session.
    See https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/CI_5levels/"""
    return request.config.getoption("--run-level")


_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str, model_prefix: str) -> Generator[OmniServer, Any, None]:
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        params: OmniServerParams = request.param
        model = model_prefix + params.model
        port = params.port
        stage_config_path = params.stage_config_path
        if run_level == "advanced_model" and stage_config_path is not None:
            with open(stage_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            stage_ids = [stage["stage_id"] for stage in cfg.get("stage_args", []) if "stage_id" in stage]
            stage_config_path = modify_stage_config(
                stage_config_path,
                deletes={"stage_args": {stage_id: ["engine_args.load_format"] for stage_id in stage_ids}},
            )

        server_args = params.server_args or []
        if params.use_omni:
            server_args = ["--stage-init-timeout", "120", *server_args]
        if stage_config_path is not None:
            server_args += ["--stage-configs-path", stage_config_path]

        with (
            OmniServer(
                model,
                server_args,
                port=port,
                env_dict=params.env_dict,
                use_omni=params.use_omni,
            )
            if port
            else OmniServer(
                model,
                server_args,
                env_dict=params.env_dict,
                use_omni=params.use_omni,
            )
        ) as server:
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


@pytest.fixture
def openai_client(omni_server: OmniServer, run_level: str):
    """Create OpenAIClientHandler fixture to facilitate communication with OmniServer
    with encapsulated request sending, concurrent requests, response handling, and validation."""
    return OpenAIClientHandler(host=omni_server.host, port=omni_server.port, api_key="EMPTY", run_level=run_level)


@pytest.fixture(scope="module")
def omni_runner(request, model_prefix):
    with _omni_server_lock:
        model, stage_config_path = request.param
        model = model_prefix + model
        with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


@pytest.fixture
def omni_runner_handler(omni_runner):
    return OmniRunnerHandler(omni_runner)
