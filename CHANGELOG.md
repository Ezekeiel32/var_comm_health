# Changelog

Versions follow `MAJOR.MINOR.PATCH`.

## Unreleased

## v0.1.0

First public release.

- LangGraph workflow that scores an email thread or a meeting transcript on six dimensions with parallel LLM nodes.
- Deterministic preprocessing and aggregation, and a written summary with recommendations.
- Docker and LangGraph Studio setup, with sample emails and transcripts in `examples/`.
- Tests for the deterministic parts of the workflow (parsing, preprocessing, aggregation). They need no API key.
