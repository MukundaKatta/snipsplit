"""Type stubs for `snipsplit._native`. Hand-written; keep in sync with
`crates/snipsplit-py/src/lib.rs`."""

from __future__ import annotations

__version__: str

class SnipsplitError(Exception):
    """Raised on tiktoken-rs failures inside the native chunker."""

class Chunk:
    @property
    def text(self) -> str: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def token_count(self) -> int: ...
    def __repr__(self) -> str: ...

class Chunker:
    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        min_tokens: int = 1,
        encoding: str = "cl100k_base",
    ) -> None: ...
    def split(self, text: str) -> list[Chunk]: ...
    def split_many(self, texts: list[str], parallel: bool = False) -> list[list[Chunk]]: ...
    def __repr__(self) -> str: ...
