"""Docker training backend for the proxy-student trainer.

This is an OPTIONAL, selectable backend for ``SandboxProxyTrainer`` (the Prime
sandbox backend remains the default and is untouched). It implements the exact
5-method, duck-typed client contract the trainer's lifecycle drives
(``trainer.py``: ``create`` -> handle(.id), ``wait_for_creation``,
``upload_bytes``, ``execute_command``, ``delete``) on top of verifiers' own v1
``DockerRuntime`` (``verifiers.v1.runtimes.make_runtime`` + ``DockerConfig``).

Because ``DockerRuntime`` shells out to the local ``docker`` CLI inheriting
``os.environ``, pointing it at a remote rented host needs nothing more than the
standard Docker mechanism: ``DOCKER_HOST=ssh://user@host`` (plus the TLS vars for
a TLS endpoint). ``DockerRuntimeClient`` will set ``DOCKER_HOST`` from the config
when it is not already present in the environment. Rollouts stay local; only the
GPU training container runs on the remote host.

The trainer never imports this module: ``taskset.py`` injects
``DockerRuntimeClient`` / ``DockerRunRequest`` as the trainer's
``client_factory`` / ``request_factory`` only when ``trainer_backend='docker'``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DockerRunRequest:
    """The docker-backend analogue of ``CreateSandboxRequest``.

    Carries everything needed to build a ``DockerConfig`` and name the container.
    ``timeout_minutes`` is informational here (the trainer derives the command
    timeout from the config itself); Docker has no portable container-lifetime
    limit, and ``disk`` is advisory (Docker has no portable per-container cap).
    """

    name: str
    image: str
    gpu: str | None = None
    cpu: float | None = None
    memory: float | None = None
    disk: float | None = None
    workdir: str = "/workspace"
    timeout_minutes: int = 30


@dataclass
class _ExecResult:
    """Normalized exec result exposing the ``.stdout/.stderr/.exit_code`` the
    trainer's ``_parse_result`` / ``_run_training`` read (the same shape Prime's
    ``CommandResult`` exposes)."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass
class _Handle:
    """The ``create`` handle whose ``.id`` keys every later lifecycle call."""

    id: str


class DockerRuntimeClient:
    """A sandbox-client over verifiers' v1 ``DockerRuntime``.

    One instance owns the runtimes it provisions, keyed by the id returned from
    ``create``. The five methods mirror the Prime ``AsyncSandboxClient`` surface
    the trainer drives, so the trainer lifecycle is reused verbatim.
    """

    def __init__(
        self,
        docker_host: str | None = None,
        tls_verify: bool = False,
        cert_path: str | None = None,
    ) -> None:
        self._docker_host = docker_host
        self._tls_verify = tls_verify
        self._cert_path = cert_path
        self._runtimes: dict[str, Any] = {}
        self._apply_docker_host()

    def _apply_docker_host(self) -> None:
        """Point the docker CLI at the configured (remote) host.

        REVIEWER NOTE: this MUTATES the process-global ``os.environ`` so the
        ``docker`` CLI that ``DockerRuntime`` shells out to targets the remote
        host. It is deliberately conservative — it never overrides an ambient
        ``DOCKER_HOST`` the operator already set — but it is a global side effect.
        """
        if not self._docker_host:
            return  # rely on the ambient DOCKER_HOST / local daemon
        if "DOCKER_HOST" in os.environ:
            ambient = os.environ["DOCKER_HOST"]
            if ambient != self._docker_host:
                # A mis-set ambient host would silently target the WRONG daemon;
                # warn (not info) so the ignored request fails visibly rather than
                # quietly aiming training at some other host. Behavior is unchanged:
                # an ambient DOCKER_HOST is still never overridden.
                logger.warning(
                    "docker backend: ambient DOCKER_HOST (%s) differs from requested "
                    "%s; the requested host is being IGNORED",
                    ambient,
                    self._docker_host,
                )
            else:
                logger.info(
                    "docker backend: DOCKER_HOST already set to the requested host "
                    "(%s); nothing to do",
                    ambient,
                )
            return
        os.environ["DOCKER_HOST"] = self._docker_host
        logger.info("docker backend: set DOCKER_HOST=%s", self._docker_host)
        if self._tls_verify:
            os.environ.setdefault("DOCKER_TLS_VERIFY", "1")
        if self._cert_path:
            os.environ.setdefault("DOCKER_CERT_PATH", self._cert_path)

    async def create(self, request: DockerRunRequest) -> _Handle:
        # Imported here (not at module top) so the docker runtime is only required
        # on the live docker path, mirroring the trainer's lazy prime import.
        from verifiers.v1.runtimes import DockerConfig, make_runtime

        config = DockerConfig(
            image=request.image,
            workdir=request.workdir,
            gpu=request.gpu,
            cpu=request.cpu,
            memory=request.memory,
            disk=request.disk,
        )
        # The trainer passes a CONSTANT container name ('proxy-student-trainer');
        # with max_concurrent_training>1 a second concurrent `docker run --name`
        # would collide (and a retry after a leak would too). Uniquify the docker
        # `--name` per create() here (the trainer stays untouched); the handle id
        # below stays unique + consistent for every later lifecycle call.
        container_name = f"{request.name}-{uuid.uuid4().hex[:8]}"
        runtime = make_runtime(config, name=container_name)
        try:
            await runtime.start()
        except BaseException:
            # start() shells out to `docker run`. The trainer wraps create() in
            # asyncio.wait_for(..., create_timeout_seconds); if that cancels/times
            # out AFTER the daemon created the container but BEFORE we store the
            # handle, nothing else could ever tear it down. Tear the just-created
            # container down here (covering the CancelledError wait_for raises into
            # this coroutine), then re-raise so the trainer still sees the failure.
            await self._stop_and_cleanup(runtime)
            raise
        # The trainer uses this id for every subsequent call; prefer docker's short
        # container id (set after start), fall back to the unique container name.
        runtime_id = runtime.descriptor or runtime.name
        self._runtimes[runtime_id] = runtime
        return _Handle(id=runtime_id)

    async def wait_for_creation(self, sandbox_id: str) -> None:
        # ``DockerRuntime.start`` already blocked until the container was running,
        # so there is nothing left to wait for.
        return None

    async def upload_bytes(
        self, sandbox_id: str, path: str, data: bytes, name: str | None = None
    ) -> None:
        # The trainer uploads absolute ``/workspace/...`` paths; ``DockerRuntime``
        # writes them with ``mkdir -p <parent> && cat > <path>``, so an absolute
        # path lands exactly there regardless of the container workdir.
        await self._runtime(sandbox_id).write(path, data)

    async def execute_command(
        self, sandbox_id: str, command: str, timeout: float | None = None
    ) -> _ExecResult:
        runtime = self._runtime(sandbox_id)
        try:
            result = await asyncio.wait_for(
                runtime.run(["bash", "-lc", command], {}), timeout
            )
        except BaseException:
            # On timeout/cancel/error, tear the container down so a failed or
            # abandoned run never leaks a container on the remote host. The
            # trainer's ``finally`` also calls ``delete``, which then no-ops.
            await self._teardown(sandbox_id)
            raise
        return _ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    async def delete(self, sandbox_id: str) -> None:
        await self._teardown(sandbox_id)

    def _runtime(self, sandbox_id: str) -> Any:
        try:
            return self._runtimes[sandbox_id]
        except KeyError:
            raise KeyError(f"unknown docker runtime id {sandbox_id!r}") from None

    async def _teardown(self, sandbox_id: str) -> None:
        """Stop + remove the container and drop it from the map (idempotent)."""
        runtime = self._runtimes.pop(sandbox_id, None)
        if runtime is None:
            return  # already torn down (e.g. by execute_command on error)
        await self._stop_and_cleanup(runtime)

    @staticmethod
    async def _stop_and_cleanup(runtime: Any) -> None:
        """Stop the container, with the synchronous ``cleanup`` as the backstop.

        ``cleanup`` is the source-of-truth teardown and is idempotent; call it in
        a ``finally`` so it still runs if ``stop`` raised. Shared by ``_teardown``
        (after the handle is stored) and ``create`` (a ``start`` that failed before
        the handle was stored), so both paths tear a container down identically.
        """
        try:
            await runtime.stop()
        finally:
            runtime.cleanup()
