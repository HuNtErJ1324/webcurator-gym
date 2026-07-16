"""Concurrency regressions for the rendered standalone ``self_score.py``."""

from __future__ import annotations

import subprocess
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from pretrain_curation_gym.gpu.self_score import (
    SELF_SCORE_FILENAME,
    render_self_score_script,
)
from pretrain_curation_gym.models import CuratorConfig


def _rendered_helpers(**render_kwargs: Any) -> dict[str, Any]:
    """Execute the rendered script without its ``__main__`` entrypoint."""
    namespace: dict[str, Any] = {
        "__name__": "self_score_concurrency_test",
        "__file__": SELF_SCORE_FILENAME,
    }
    exec(
        compile(
            render_self_score_script(
                CuratorConfig(token_budget=1_000, use_real_trainer=True),
                **render_kwargs,
            ),
            SELF_SCORE_FILENAME,
            "exec",
        ),
        namespace,
    )
    return namespace


def test_score_components_overlap_and_preserve_results():
    helpers = _rendered_helpers()
    train_started = threading.Event()
    decon_started = threading.Event()
    worker_threads: set[int] = set()

    def fake_train(docs, *, max_corpus_chars, max_steps, train_timeout):
        assert docs == ["sample"]
        assert (max_corpus_chars, max_steps, train_timeout) == (123, 7, 11)
        worker_threads.add(threading.get_ident())
        train_started.set()
        assert decon_started.wait(2), "Decon did not overlap proxy training"
        return 2.75, 0.625, "sample_ce"

    def fake_decon(docs):
        assert docs == ["sample"]
        worker_threads.add(threading.get_ident())
        decon_started.set()
        assert train_started.wait(2), "proxy training did not overlap Decon"
        return 0.125, 4

    helpers["train_perf"] = fake_train
    helpers["decon_score"] = fake_decon

    result = helpers["score_components"](
        lambda: fake_train(
            ["sample"],
            max_corpus_chars=123,
            max_steps=7,
            train_timeout=11,
        ),
        lambda: fake_decon(["sample"]),
    )

    assert result == ((2.75, 0.625, "sample_ce"), (0.125, 4))
    assert len(worker_threads) == 2


def test_score_components_preserves_training_exception_precedence():
    """The concurrent join raises the exception the old sequential path saw first."""
    helpers = _rendered_helpers()
    both_running = threading.Barrier(2)
    cleanups: list[str] = []

    class TrainFailure(RuntimeError):
        pass

    class DeconFailure(RuntimeError):
        pass

    def fake_train(docs, *, max_corpus_chars, max_steps, train_timeout):
        both_running.wait(timeout=2)
        raise TrainFailure("train failed first in sequential order")

    def fake_decon(docs):
        both_running.wait(timeout=2)
        raise DeconFailure("decon also failed")

    helpers["train_perf"] = fake_train
    helpers["decon_score"] = fake_decon
    helpers["_cleanup_active_train_proc"] = lambda: cleanups.append("cleanup")

    with pytest.raises(TrainFailure, match="sequential order"):
        helpers["score_components"](
            lambda: fake_train(
                ["sample"],
                max_corpus_chars=None,
                max_steps=None,
                train_timeout=11,
            ),
            lambda: fake_decon(["sample"]),
        )

    assert cleanups == ["cleanup"]


def _process_is_running(pid: int) -> bool:
    """Treat a reparented zombie as dead: it cannot execute or hold resources."""
    try:
        stat_fields = Path("/proc/%d/stat" % pid).read_text(encoding="utf-8")
    except OSError:
        return False
    close = stat_fields.rfind(")")
    return close < 0 or stat_fields[close + 2 :].split()[0] != "Z"


def test_worker_failure_reaps_decon_process_group(tmp_path: Path, monkeypatch):
    """A failed training future cannot leave Decon or its descendant alive."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval.jsonl").write_text('{"text":"eval"}\n', encoding="utf-8")
    pidfile = tmp_path / "decon-pids"
    fake_decon = tmp_path / "decon"
    fake_decon.write_text(
        "#!/usr/bin/env python3\n"
        "import os, time\n"
        "from pathlib import Path\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    while True:\n"
        "        time.sleep(0.1)\n"
        f"Path({str(pidfile)!r}).write_text("
        "    '%d %d' % (os.getpid(), child), encoding='utf-8'"
        ")\n"
        "# Exit the session leader immediately. Cleanup must still signal the\n"
        "# recorded process group and reap the surviving descendant.\n",
        encoding="utf-8",
    )
    fake_decon.chmod(fake_decon.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(
        "PDC_SELF_SCORE_DECON_LOCK", str(tmp_path / "decon.lock")
    )
    helpers = _rendered_helpers(
        decon_binary=str(fake_decon),
        decon_evals_dir=str(evals),
    )
    monkeypatch.setenv("PDC_SELF_SCORE_DECON_TIMEOUT", "5")

    class TrainFailure(RuntimeError):
        pass

    def failing_train(docs, *, max_corpus_chars, max_steps, train_timeout):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if pidfile.exists() and helpers["_ACTIVE_DECON_PROC"] is not None:
                raise TrainFailure("stop concurrent scoring")
            time.sleep(0.01)
        raise AssertionError("Decon child never became active")

    helpers["train_perf"] = failing_train
    with pytest.raises(TrainFailure, match="stop concurrent scoring"):
        helpers["score_components"](
            lambda: failing_train(
                ["sample"],
                max_corpus_chars=None,
                max_steps=None,
                train_timeout=11,
            ),
            lambda: helpers["decon_score"](["sample"]),
        )

    leader_pid, child_pid = map(
        int, pidfile.read_text(encoding="utf-8").split()
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(
        _process_is_running(pid) for pid in (leader_pid, child_pid)
    ):
        time.sleep(0.02)

    assert not _process_is_running(leader_pid)
    assert not _process_is_running(child_pid)
    assert helpers["_ACTIVE_DECON_PROC"] is None
    assert helpers["_ACTIVE_DECON_IDENTITY"] is None


def test_decon_lock_is_cross_process_and_independent_from_trainer(
    tmp_path: Path,
):
    helpers = _rendered_helpers()
    train_lock_path = str(tmp_path / "train.lock")
    decon_lock_path = str(tmp_path / "decon.lock")

    # Separate lock files are independent: one process may run one trainer and
    # one Decon concurrently without admitting a second instance of either.
    train_fh = helpers["_train_lock"](train_lock_path, timeout=0.2)
    decon_fh = helpers["_decon_lock"](decon_lock_path, timeout=0.2)
    rendered = tmp_path / SELF_SCORE_FILENAME
    rendered.write_bytes(
        render_self_score_script(
            CuratorConfig(token_budget=1_000, use_real_trainer=True)
        )
    )
    acquired = tmp_path / "decon-acquired"
    driver = (
        "import time\n"
        f"ns = {{'__name__': 'decon_lock_child', '__file__': {str(rendered)!r}}}\n"
        f"exec(compile(open({str(rendered)!r}).read(), 'self_score.py', 'exec'), ns)\n"
        "started = time.monotonic()\n"
        f"fh = ns['_decon_lock']({decon_lock_path!r}, timeout=3)\n"
        f"open({str(acquired)!r}, 'w').write(str(time.monotonic() - started))\n"
        "ns['_release_decon_lock'](fh)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", driver],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.2)
        assert child.poll() is None, child.stderr.read() if child.stderr else ""
        assert not acquired.exists(), "second Decon entered while lock was held"
        helpers["_release_decon_lock"](decon_fh)
        decon_fh = None
        assert child.wait(timeout=3) == 0, child.stderr.read() if child.stderr else ""
        assert float(acquired.read_text(encoding="utf-8")) >= 0.15
    finally:
        helpers["_release_decon_lock"](decon_fh)
        helpers["_release_train_lock"](train_fh)
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_signal_cleanup_waits_through_decon_registration_window(
    tmp_path: Path, monkeypatch
):
    """A signal in Popen's return window still finds and reaps the new group."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval.jsonl").write_text('{"text":"eval"}\n', encoding="utf-8")
    fake_decon = tmp_path / "decon"
    fake_decon.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_decon.chmod(fake_decon.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv(
        "PDC_SELF_SCORE_DECON_LOCK", str(tmp_path / "decon.lock")
    )
    monkeypatch.setenv("PDC_SELF_SCORE_DECON_TIMEOUT", "3")
    helpers = _rendered_helpers(
        decon_binary=str(fake_decon),
        decon_evals_dir=str(evals),
    )

    real_popen = subprocess.Popen
    child_created = threading.Event()
    allow_registration = threading.Event()
    child_holder: dict[str, subprocess.Popen] = {}

    def delayed_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child_holder["child"] = child
        child_created.set()
        assert allow_registration.wait(2)
        return child

    monkeypatch.setattr(subprocess, "Popen", delayed_popen)
    result: list[tuple[float | None, int | None]] = []
    worker = threading.Thread(
        target=lambda: result.append(helpers["decon_score"](["sample"]))
    )
    worker.start()
    assert child_created.wait(2), "Decon Popen was never reached"
    assert helpers["_DECON_STARTING"] is True
    child = child_holder["child"]

    release = threading.Timer(0.1, allow_registration.set)
    release.start()
    started = time.monotonic()
    helpers["_cleanup_active_train_proc"]()
    cleanup_elapsed = time.monotonic() - started
    worker.join(timeout=3)
    release.join(timeout=1)

    assert not worker.is_alive(), "cleanup left the Decon worker blocked"
    assert cleanup_elapsed >= 0.05, "cleanup skipped the registration window"
    assert not _process_is_running(child.pid)
    assert helpers["_ACTIVE_DECON_PROC"] is None
    assert helpers["_ACTIVE_DECON_LOCK_FH"] is None
    assert helpers["_DECON_STARTING"] is False
    assert result == [(None, None)]


def test_empty_docs_keep_decon_skipped_and_result_shape():
    helpers = _rendered_helpers()
    calls: list[str] = []

    def fake_train(docs, *, max_corpus_chars, max_steps, train_timeout):
        calls.append("train")
        return None, None, None

    def forbidden_decon(docs):
        calls.append("decon")
        raise AssertionError("empty corpus must not launch Decon")

    helpers["train_perf"] = fake_train
    helpers["decon_score"] = forbidden_decon

    result = helpers["score_components"](
        lambda: fake_train(
            [], max_corpus_chars=None, max_steps=None, train_timeout=11
        ),
        None,
    )

    assert result == ((None, None, None), (None, None))
    assert calls == ["train"]
