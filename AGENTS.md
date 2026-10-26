# Agent Instructions for Suffix Repair Research

This repository is a standalone research project based on vLLM internals. It is
not a contribution branch or PR for the main vLLM project. Prioritize rapid,
iterative experimentation and clear, reproducible research results.

## Development Workflow

- Never use system `python3` or bare `pip`/`pip install`. Use `uv` and
  `.venv/bin/python` for Python environment management and commands.

### Environment setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

For C/C++ or CUDA changes, use incremental builds when they help shorten the
experiment cycle.

### Experiments and tests

- Put project-specific test and experiment scripts in `suffix_repair_test/`.
- Favor small, fast checks that answer the current research question. Run the
  relevant checks when useful; full upstream vLLM test suites and strict CI
  coverage are not required for routine experimentation.
- Record enough command, configuration, and result detail to reproduce
  experiments when results are important to the project.

### Code style

Match the surrounding vLLM code and keep changes focused. Use a line length of
88 characters for Python where practical. Minimize comments; keep necessary
comments and docstrings brief. Use Google-style docstrings (`Args:`, `Returns:`,
`Raises:`), not reStructuredText/Sphinx fields. Assume readers are familiar
with vLLM.

## Domain-Specific Guides

Follow applicable area-specific guides when changing those parts of the code.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
- **Rust frontend (`rust/`)**: [`rust/AGENTS.md`](rust/AGENTS.md)
  — Crate ownership, design principles, Python parity, and testing conventions.
