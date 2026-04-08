from __future__ import annotations

import gc
import os
import subprocess

import torch


def _run_pre_test_cleanup(enable_force=False):
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1" and not enable_force:
        print("GPU cleanup disabled")
        return

    print("Pre-test GPU status:")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        try:
            from tests.utils import wait_for_gpu_memory_to_clear

            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.05,
            )
        except Exception as e:
            print(f"Pre-test cleanup note: {e}")


def _run_post_test_cleanup(enable_force=False):
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1" and not enable_force:
        print("GPU cleanup disabled")
        return

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

        print("Post-test GPU status:")
        _print_gpu_processes()


def _print_gpu_processes():
    """Print GPU information including nvidia-smi and system processes"""

    print("\n" + "=" * 80)
    print("NVIDIA GPU Information (nvidia-smi)")
    print("=" * 80)

    try:
        nvidia_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if nvidia_result.returncode == 0:
            lines = nvidia_result.stdout.strip().split("\n")
            for line in lines[:20]:
                print(line)

            if len(lines) > 20:
                print(f"... (showing first 20 of {len(lines)} lines)")
        else:
            print("nvidia-smi command failed")

    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi not available or timed out")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

    print("\n" + "=" * 80)
    print("Detailed GPU Processes (nvidia-smi pmon)")
    print("=" * 80)

    try:
        pmon_result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        if pmon_result.returncode == 0 and pmon_result.stdout.strip():
            print(pmon_result.stdout)
        else:
            print("No active GPU processes found via nvidia-smi pmon")

    except Exception:
        print("nvidia-smi pmon not available")

    print("\n" + "=" * 80)
    print("System Processes with GPU keywords")
    print("=" * 80)
