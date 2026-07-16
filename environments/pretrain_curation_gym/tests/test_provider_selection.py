"""End-to-end regression tests driving the actual 400M A100 launcher.

These reuse the fake-PATH harness from ``test_400m_eval_a100_launcher`` so the
real script runs against recorded fake binaries. The live repo-root
secrets.env is never touched because ``_make_temp_repo`` copies the launcher
into an isolated temp repo with its own throwaway secrets.env.
"""

from __future__ import annotations

from .test_400m_eval_a100_launcher import (
    _AVAIL_MIXED,
    _STATUS_NONROOT,
    _STATUS_ROOT,
    _make_temp_repo,
    _run_script,
)

AVAIL_ONLY_EXCLUDED = {
    "gpu_resources": [
        {
            "cloud_id": "mc-a100",
            "provider": "MassedCompute",
            "stock_status": "available",
            "price_per_hour": "0.90",
            "is_spot": False,
            "gpu_type": "A100_80GB",
        },
        {
            "cloud_id": "cr-a100",
            "provider": "crusoecloud",
            "stock_status": "available",
            "price_per_hour": "1.00",
            "is_spot": False,
            "gpu_type": "A100_80GB",
        },
    ]
}


def _run(tmp_path, *, avail, status, rsync_mode="ok"):
    repo = _make_temp_repo(tmp_path)
    return _run_script(
        tmp_path, repo, avail=avail, status=status, rsync_mode=rsync_mode
    )


def test_cheapest_massedcompute_excluded_selects_datacrunch(tmp_path):
    result, record = _run(tmp_path, avail=_AVAIL_MIXED, status=_STATUS_ROOT)
    picked = record / "picked.txt"
    assert picked.exists(), f"pods create never called; stderr={result.stderr}"
    chosen = picked.read_text().split()
    assert "dc-a100" in chosen, f"expected DataCrunch pick, got {chosen}"
    assert "mc-a100" not in chosen, "MassedCompute (excluded) must not be selected"


def test_only_excluded_providers_yields_no_offer(tmp_path):
    result, record = _run(tmp_path, avail=AVAIL_ONLY_EXCLUDED, status=_STATUS_ROOT)
    assert not (record / "picked.txt").exists(), "must not create a pod when no offer"
    assert result.returncode != 0


def test_non_root_fails_before_sync(tmp_path):
    result, record = _run(tmp_path, avail=_AVAIL_MIXED, status=_STATUS_NONROOT)
    assert not (record / "rsync_count.txt").exists(), "rsync must not run for non-root"
    assert not (record / "chmod.log").exists(), (
        "secrets must not be uploaded for non-root"
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "non-root" in combined or "root" in combined


def test_permanent_rsync_fails_immediately(tmp_path):
    result, record = _run(
        tmp_path, avail=_AVAIL_MIXED, status=_STATUS_ROOT, rsync_mode="permanent"
    )
    count = record / "rsync_count.txt"
    assert count.exists(), f"rsync was never invoked; stderr={result.stderr}"
    # Permanent failures must not be retried (5x). Exactly one attempt.
    assert len(count.read_text().split()) == 1, "permanent failure must not be retried"
    assert result.returncode != 0


def test_secrets_mode_enforced(tmp_path):
    result, record = _run(tmp_path, avail=_AVAIL_MIXED, status=_STATUS_ROOT)
    chmod_log = record / "chmod.log"
    assert chmod_log.exists(), (
        f"remote chmod on secrets.env never ran; stderr={result.stderr}"
    )
    assert "chmod" in chmod_log.read_text()
    assert (record / "scp.log").exists(), "secrets.env upload (scp) must have run"
