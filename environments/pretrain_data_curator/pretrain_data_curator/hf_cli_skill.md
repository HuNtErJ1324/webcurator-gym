# Hugging Face CLI guidance

Use the `hf` executable already provided in this workspace. Do not install, upgrade,
replace, or shadow it, and do not run `hf skills add`; this tracked file is the
project-local reference.

## Discover the available CLI

Treat the installed CLI as current truth:

1. Run `hf --version`.
2. Run `hf --help` to discover command groups.
3. Before using a command, run its subgroup help (for example,
   `hf <group> --help`, then `hf <group> <command> --help`).

Do not assume flags or commands from memory. The workspace `hf` command is metered
and delegates to the real CLI, so keep using it rather than seeking another binary.

## Safety

- Prefer read-only search, listing, and metadata inspection while researching.
- Uploading, creating, updating, deleting, launching jobs or Spaces, and changing
  authentication are external mutations. Perform them only when the task explicitly
  requires that exact mutation; inspect help and the target first.
- Use existing environment-based authentication. Never print, echo, log, commit, or
  place tokens in command arguments, URLs, scripts, or files. Do not reveal token
  environment variables or run an interactive token login.
- Start with narrow queries and inspect results before downloading large artifacts.

Detailed upstream guide: https://huggingface.co/docs/hub/agents-cli
