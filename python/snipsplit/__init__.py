"""Token-aware text chunker for RAG ingestion.

Sentence-respecting greedy packer with optional overlap. Heavy work runs
in `snipsplit._native` (Rust + tiktoken-rs); this module wraps it in a
typed dataclass-shaped facade.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Final

from snipsplit._native import (
    Chunker as _NativeChunker,
)
from snipsplit._native import (
    SnipsplitError,
)


def _read_version() -> str:
    try:
        return metadata.version("snipsplit")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__: Final[str] = _read_version()

__all__ = ["Chunk", "Chunker", "SnipsplitError", "__version__"]


@dataclass(frozen=True)
class Chunk:
    """One emitted chunk."""

    text: str
    start: int
    end: int
    token_count: int


class Chunker:
    """Token-aware text chunker."""

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        min_tokens: int = 1,
        encoding: str = "cl100k_base",
    ) -> None:
        self._inner = _NativeChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
            encoding=encoding,
        )

    def split(self, text: str) -> list[Chunk]:
        """Split `text` into chunks."""
        return [_chunk_from_native(c) for c in self._inner.split(text)]

    def split_many(self, texts: Sequence[str], *, parallel: bool = False) -> list[list[Chunk]]:
        """Split many texts; with `parallel=True`, distributes across rayon."""
        raw = self._inner.split_many(list(texts), parallel)
        return [[_chunk_from_native(c) for c in lst] for lst in raw]

    def __repr__(self) -> str:
        return repr(self._inner)


def _chunk_from_native(c: object) -> Chunk:
    return Chunk(
        text=c.text,  # type: ignore[attr-defined]
        start=c.start,  # type: ignore[attr-defined]
        end=c.end,  # type: ignore[attr-defined]
        token_count=c.token_count,  # type: ignore[attr-defined]
    )
