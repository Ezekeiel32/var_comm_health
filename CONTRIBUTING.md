# Contributing

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Branches

- `main` is always in a working state and is protected: no force pushes, no deletion, and changes arrive through pull requests with CI passing.
- Work happens on short-lived branches cut from `main` and deleted after the merge:
  - `feat/<topic>` for new behavior
  - `fix/<topic>` for bug fixes
  - `docs/<topic>` for documentation
  - `chore/<topic>` for build, CI and repository upkeep
- `main` keeps a linear history: no merge commits.

## Commits

- Subject line in the imperative, 72 characters or fewer.
- One logical change per commit. Explain the why in the body when it is not obvious from the diff.

## Before you open a pull request

Run the same checks CI runs:

```bash
   python -m compileall -q var_comm_health
   pytest -q
```

Add or update a test when you change behavior, and add a line to `CHANGELOG.md` under "Unreleased".

## Releases

Releases are tagged `vMAJOR.MINOR.PATCH` and described in `CHANGELOG.md`.

## Reporting bugs and security issues

Use the issue templates for bugs and feature requests. For security problems, see [SECURITY.md](SECURITY.md).
