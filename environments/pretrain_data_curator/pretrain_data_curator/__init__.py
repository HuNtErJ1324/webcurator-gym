"""Pretraining-data curation environment (verifiers v1).

The package exports its :class:`CuratorTaskset` via ``__all__`` so the v1 loader
can compose it with the default (MCP-backed) harness. ``load_environment`` returns
the native v1 environment directly.
"""

# ── verifiers v1 path bootstrap ───────────────────────────────────────────
# When running inside prime-rl Hosted Training the orchestrator pre-loads
# verifiers==0.0.0 from /app/deps/verifiers/; its sys.modules entry carries a
# stale __path__ pointing there. A newer verifiers (>=0.1.3, with full v1/)
# is then installed into /app/.venv/ by uv, but the cached module's __path__
# doesn't know about it. We patch it here so that `import verifiers.v1` finds
# the real v1 in the newly-installed location.
import importlib
import os
import sys


def _bootstrap_verifiers_v1() -> None:
    """Load the full installed ``verifiers.v1`` instead of the cached stub.

    Prime-rl pre-loads verifiers==0.0.0 whose verifiers.v1 is a stub that lacks
    the clients / trace / task modules. After uv
    installs a newer verifiers (>=0.1.3) the full v1 is on disk but unreachable
    because sys.modules['verifiers'] still has a stale __path__ pointing at the
    old location, and any attempt to `import verifiers.v1` loads the stub
    instead.

    Fix: find the installed package containing the clients package and task.py,
    prepend its parent to the cached ``verifiers.__path__``, evict stale v1
    modules, and import the real package so its public exports (``Environment``,
    ``Taskset``, ``State``, etc.) are initialized.
    """
    try:
        print('[pretrain-data-curator] bootstrap starting', file=sys.stderr, flush=True)
        print(f'[pretrain-data-curator] sys.path: {sys.path[:8]}', file=sys.stderr, flush=True)
        if 'verifiers' in sys.modules:
            _v = sys.modules['verifiers']
            print(f'[pretrain-data-curator] cached verifiers version={getattr(_v, "__version__", "?")} path={list(getattr(_v, "__path__", []))[:3]}', file=sys.stderr, flush=True)
        else:
            print('[pretrain-data-curator] verifiers NOT in sys.modules', file=sys.stderr, flush=True)

        import verifiers

        v1 = sys.modules.get("verifiers.v1")
        if v1 is not None:
            v1_dir = getattr(v1, "__path__", [None])[0] or ""
            has_clients = os.path.isfile(os.path.join(v1_dir, "clients.py")) or (
                os.path.isfile(os.path.join(v1_dir, "clients", "__init__.py"))
            )
            if has_clients and os.path.isfile(os.path.join(v1_dir, "task.py")):
                return

        # Find the full v1 on sys.path.
        full_v1_dir: str | None = None
        for entry in sys.path:
            candidate = os.path.join(entry, "verifiers", "v1")
            has_clients = os.path.isfile(
                os.path.join(candidate, "clients.py")
            ) or os.path.isfile(os.path.join(candidate, "clients", "__init__.py"))
            if has_clients and os.path.isfile(os.path.join(candidate, "task.py")):
                full_v1_dir = candidate
                break

        print(f'[pretrain-data-curator] scan result: full_v1_dir={full_v1_dir}', file=sys.stderr, flush=True)

        if full_v1_dir is None:
            return  # full v1 not found — nothing we can do

        parent_dir = os.path.dirname(full_v1_dir)
        if parent_dir in list(verifiers.__path__):
            verifiers.__path__.remove(parent_dir)
        verifiers.__path__.insert(0, parent_dir)
        importlib.invalidate_caches()

        for name in [
            key
            for key in sys.modules
            if key == "verifiers.v1" or key.startswith("verifiers.v1.")
        ]:
            del sys.modules[name]
        importlib.import_module("verifiers.v1")
        print(f'[pretrain-data-curator] bootstrap SUCCESS, verifiers.__path__={list(verifiers.__path__)[:3]}', file=sys.stderr, flush=True)
    except Exception as _bootstrap_exc:
        import traceback as _tb
        print(f'[pretrain-data-curator] bootstrap EXCEPTION: {_bootstrap_exc}', file=sys.stderr, flush=True)
        print(_tb.format_exc(), file=sys.stderr, flush=True)


_bootstrap_verifiers_v1()
# ─────────────────────────────────────────────────────────────────────────

try:
    from .pretrain_data_curator import load_environment
except Exception as _ie:
    import traceback as _tb2
    print(f'[pretrain-data-curator] IMPORT pretrain_data_curator FAILED: {_ie}', file=sys.stderr, flush=True)
    print(_tb2.format_exc(), file=sys.stderr, flush=True)
    raise

try:
    from .taskset import CuratorTaskset
except Exception as _ie2:
    import traceback as _tb3
    print(f'[pretrain-data-curator] IMPORT taskset FAILED: {_ie2}', file=sys.stderr, flush=True)
    print(_tb3.format_exc(), file=sys.stderr, flush=True)
    raise

__all__ = ["CuratorTaskset", "load_environment"]
