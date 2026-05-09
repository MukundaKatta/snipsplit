# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-09

### Added

- Initial public release.
- Rust core crate `snipsplit-core` wrapping
  [tiktoken-rs](https://crates.io/crates/tiktoken-rs) for cl100k/o200k.
- Sentence-respecting greedy chunker with configurable max-tokens, overlap,
  and minimum chunk size.
- Falls back to mid-sentence token slicing only when a single sentence
  exceeds the budget on its own.
- `split_many(texts, parallel=true)` parallel batch ingestion via rayon.
- Python package `snipsplit` with PyO3 bindings.
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.

[Unreleased]: https://github.com/MukundaKatta/snipsplit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MukundaKatta/snipsplit/releases/tag/v0.1.0
