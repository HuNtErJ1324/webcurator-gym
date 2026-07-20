# decon (vendored)

Leakage scoring invokes the [allenai/decon](https://github.com/allenai/decon) CLI as a subprocess.
The binary is baked into `Dockerfile.runtime` for agent self-score, kept in
this source tree for host-side scoring, and force-included into the manylinux
wheel via `scripts/hatch_build.py` (gitignored local build hook).

## Vendored binary

`bin/decon` is a **static musl** Linux x86_64 build so it runs on Prime GPU pods (Ubuntu 22, glibc 2.35) and newer dev machines without a host glibc mismatch. The previous dynamically linked binary required **GLIBC 2.39** and failed on pods with:

```text
decon: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.39' not found
```

## Rebuild

Portable static binary (maintainers):

```bash
./decon/build_static.sh
```

Native glibc build on the current machine (pods, local Ubuntu):

```bash
./decon/build_from_source.sh
```

`scripts/run_400m_eval_a100.sh` smoke-tests `decon --version` during provisioning and falls back to `build_from_source.sh` if the vendored binary is missing or incompatible.
