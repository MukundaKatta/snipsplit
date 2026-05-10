# Changelog

## [0.1.1] - 2026-05-10

### Added
- `ChunkConfig.preserve_paragraphs` (default `false`): when `true`, blank-
  line-separated paragraphs become hard chunk boundaries — content from
  two paragraphs never gets packed into the same chunk even if the token
  budget would allow it. Useful for documents with semantically distinct
  sections (FAQs, news articles, code blocks separated from prose).
- New `split_paragraphs(text)` helper used internally; available via
  module-private API today, may be promoted in a future minor release.

### Tests
- 4 new tests covering paragraph preservation behavior, default behavior
  unchanged, and span offset translation.

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
