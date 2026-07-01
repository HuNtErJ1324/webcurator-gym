"""Docker-host reachability fixes for Docker Desktop under WSL2."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerHostReachability:
    """Make the host interception server reachable from Docker Desktop's VM.

    Verifiers treats Docker as host-network-local. That is true for a native
    Linux daemon, but Docker Desktop's WSL2 proxy runs containers in a separate
    VM where container localhost is not WSL localhost. Bind the interception
    server to the WSL interface and advertise that address on WSL only.
    """

    _configured = False

    @classmethod
    def configure(cls) -> None:
        if cls._configured or not cls._is_wsl():
            return
        host = os.environ.get("PDC_DOCKER_HOST_IP") or cls._route_address()
        if not host:
            logger.warning(
                "could not determine the WSL host address for Docker interception"
            )
            return

        from verifiers.v1.interception import server
        from verifiers.v1.runtimes import base

        original = base.host_endpoint

        @contextlib.asynccontextmanager
        async def wsl_host_endpoint(port, is_local, labels=None):
            if is_local:
                yield f"http://{host}:{port}"
                return
            async with original(port, is_local, labels) as url:
                yield url

        server._HOST = host
        base.host_endpoint = wsl_host_endpoint
        cls._configured = True
        logger.info("Docker Desktop WSL host endpoint: http://%s", host)

    @staticmethod
    def _is_wsl() -> bool:
        try:
            return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
        except OSError:
            return False

    @staticmethod
    def _route_address() -> str | None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        except OSError:
            return None
        finally:
            sock.close()
