# First-Party Comment Cleanup Design

## Goal

Remove comments and redundant internal docstrings that obscure the first-party
code without changing executable behavior.

## Scope

The sweep covers tracked first-party source in `src/`, `nix/`, and root project
configuration. It excludes vendored code, generated files, lockfiles,
migrations, example workflows, and user-owned working-tree changes.

## Removal criteria

Remove comments and internal docstrings that:

- narrate code that is already clear;
- read like implementation prompts or instructions to an agent;
- preserve dead or commented-out code;
- describe implementation chronology or previous approaches;
- label sections whose structure is already evident.

Retain comments that:

- explain why a decision or constraint exists;
- state a non-obvious invariant or external-system requirement;
- clarify mathematical conventions that identifiers and types cannot express;
- warn about a concrete failure mode;
- provide required license, formatter, linter, type-checker, or coverage
  directives;
- document a public API contract that is not expressed by its types.

## Execution

Work from the highest-density StyleTTS files outward through all first-party
source. Each edit must be comment-only or docstring-only. Any suspicious dead
code discovered during the sweep is reported separately rather than removed as
part of this change.

## Verification

Review the final diff to confirm that no executable tokens changed. Run the
repository's formatting, static checks, and relevant test commands through
`./nix/run-venv.sh`. Preserve the existing modification to
`src/runner/nodes/training/beetle/config/default.yaml`.
