# Development

## Workspace prerequisites

Work from a Prime Lab workspace:

```bash
prime lab setup --agent codex
prime lab doctor
```

The environment is self-contained under
`environments/pretrain_data_curator/`. Do not move helpers into the workspace
root or introduce a background service requirement.

## Install

Use the Prime environment lifecycle:

```bash
prime env install pretrain-data-curator -p ./environments
```

For direct package tests:

```bash
cd environments/pretrain_data_curator
uv sync --group dev
```

## Test and lint

From the workspace root:

```bash
uv run --project environments/pretrain_data_curator \
  pytest -q environments/pretrain_data_curator/tests

ruff check \
  environments/pretrain_data_curator/pretrain_data_curator \
  environments/pretrain_data_curator/tests
```

The suite covers manifest recovery, metering, filtering/allocation, fetch
single-flight, failure degradation, leakage determinism, state/compat behavior,
student architecture/training, validation windows, and mocked Docker/Modal
trainer lifecycles.

GPU execution is not exercised by the ordinary unit suite. Before publishing a
backend change, run the relevant live smoke path with a small token budget.

## Canonical eval validation

After unit tests:

```bash
prime env install pretrain-data-curator -p ./environments
cd environments/pretrain_data_curator
set -a; source ../../secrets.env; set +a
prime eval run configs/eval/deepseek-v4-flash-smoke.toml
```

Do not use `--skip-upload`; canonical evals should remain visible in the private
Evaluations tab and `prime eval tui`.

A live eval makes Hugging Face and model calls and requires valid authenticated
CLI state plus `HF_TOKEN`. For the local GPU Docker runtime the Docker Engine
with NVIDIA Container Toolkit must be available on the same machine.

## Design rules

- Keep `load_environment` strict and explicit.
- Put structured configuration in Pydantic models with narrow types and bounds.
- Keep agent interaction in the bash/`hf` contract; do not reintroduce a parallel
  mutable manifest tool API.
- Keep per-rollout state serializable. Locks, semaphores, clients, runtimes, and
  paths belong in owning classes or loop-local registries.
- Exercise the async `CorpusBuilder.materialize` path in tests. Do not add a
  second synchronous implementation that can drift from billing/cache behavior.
- Keep the heavy scoring pass single-flight and cached.
- Preserve cancellation semantics; do not convert cancellation into an ordinary
  zero-performance failure.
- Hold model, recipe, tokenizer, validation, and budgets fixed when comparing
  data.

## Adding a filter

1. Add the name to `_SUPPORTED_FILTER_KINDS` in `taskset.py`.
2. Implement it in `DocumentFilter._apply_one_iter` in `corpus.py`.
3. Add parser and materialization tests.
4. Document parameters and order-sensitive behavior in
   [Manifest and filtering](manifest.md).
5. Decide explicitly how malformed parameters behave.

Unknown filter kinds are currently dropped during manifest coercion, while an
unknown `DocumentFilter` kind is a no-op. Keep both layers aligned.

## Changing manifest fields

Update together:

- Pydantic models in `models.py`;
- coercion in `taskset.py`;
- single initial prompt contract in `tasks.TASK_PROMPT`;
- parsing/finalization tests;
- [Manifest and filtering](manifest.md);
- relevant checked-in configs.

Model output is intentionally tolerant, but the internal manifest should remain
strict.

## Changing proxy training

The real training script embeds source definitions from `student_model.py`,
`student_train.py`, and `val_set.plan_val_windows`. Preserve the byte-identity
tests when changing those files.

Test on CPU first. Then validate the generated script parses and run one
small-budget GPU smoke on every affected backend. Check:

- structured `RESULT_JSON`;
- validation source and target count;
- tokens/FLOPs billing;
- timeout and cancellation cleanup;
- stderr preservation on a forced failure.

## Adding a runtime backend

Implement the `ProxyStudentTrainer` protocol and keep runtime selection explicit.
If the backend owns the full rollout runtime:

1. add a v1 runtime config in `load_environment`;
2. validate runtime/backend pairing in taskset setup;
3. pass the live runtime through scoring;
4. use runtime `write`/`run` APIs;
5. implement timeout, cancellation, result parsing, and teardown;
6. add mocked lifecycle tests;
7. document credentials, image contract, networking, and billing duration.

Do not require users to start a service manually before `load_environment`.

## Documentation accuracy checklist

When behavior changes, search docs and configs for the old field or command:

```bash
rg -n 'old_name|old command' \
  environments/pretrain_data_curator \
  configs
```

Verify:

- loader defaults match the configuration table and example TOML;
- manifest examples parse through `parse_manifest`;
- backend commands use the correct local/published environment target;
- metric names match decorated methods;
- image and credential requirements match the selected runtime;
- limitations are stated rather than implied.

## Publish only after validation

After local eval behavior is verified:

```bash
prime env push --path ./environments/pretrain_data_curator
```

Do not publish from an untested worktree.
