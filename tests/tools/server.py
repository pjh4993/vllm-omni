from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import psutil
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.utils.network_utils import get_open_port

from tests.tools.gpu_cleanup import _run_post_test_cleanup, _run_pre_test_cleanup


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
        env_dict: dict[str, str] | None = None,
        use_omni: bool = True,
    ) -> None:
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        cleanup_dist_env_and_memory()
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.use_omni = use_omni
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        if port is None:
            self.port = get_open_port()
        else:
            self.port = port

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.use_omni:
            cmd.append("--omni")
        cmd += self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
        )

        # Wait for server to be ready
        max_wait = 1200  # 20 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            # Check for process status
            ret = self.proc.poll()
            if ret is not None:
                raise RuntimeError(f"Server processes exited with code {ret} before becoming ready.")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                if result == 0:
                    print(f"Server ready on {self.host}:{self.port}")
                    return
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def _kill_process_tree(self, pid):
        """kill process and its children with verification"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Get all PIDs first
            all_pids = [pid] + [child.pid for child in children]

            # Terminate children
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for children
            gone, still_alive = psutil.wait_procs(children, timeout=10)

            # Kill remaining children
            for child in still_alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            # Terminate parent
            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

            # VERIFICATION: Check if all processes are gone
            time.sleep(1)  # Give system time
            alive_processes = []
            for check_pid in all_pids:
                if psutil.pid_exists(check_pid):
                    alive_processes.append(check_pid)

            if alive_processes:
                print(f"Warning: Processes still alive: {alive_processes}")
                # Optional: Try system kill
                import subprocess

                for alive_pid in alive_processes:
                    try:
                        subprocess.run(["kill", "-9", str(alive_pid)], timeout=2)
                    except Exception as e:
                        print(f"Cleanup failed: {e}")

        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self._kill_process_tree(self.proc.pid)
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        cleanup_dist_env_and_memory()
