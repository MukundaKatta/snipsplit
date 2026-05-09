# snipsplit

Token-aware text chunker for RAG ingestion. Sentence-respecting,
overlap-friendly. Rust core, Python frontend.

## The problem

Your RAG ingestion pipeline needs to split long documents into chunks
that fit a context budget. Naive token-window chunking cuts mid-sentence
and degrades retrieval. Sentence-only splitting blows the token budget
on legal/medical text. The right thing is to greedy-pack sentences into
a token-budgeted window, with optional overlap, and fall back to
token-level slicing only when a single sentence is genuinely too long.

`snipsplit` does exactly that, in Rust, fast enough that bulk ingestion
of 100k documents runs in seconds rather than minutes.

## Install

```bash
pip install snipsplit
```

## 30-second quickstart

```python
from snipsplit import Chunker

chunker = Chunker(max_tokens=512, overlap_tokens=64, encoding="cl100k_base")

text = open("long_document.txt").read()
for chunk in chunker.split(text):
    print(chunk.token_count, chunk.start, chunk.end, chunk.text[:60])
```

For batch ingestion across many docs:

```python
texts = [open(p).read() for p in paths]
all_chunks = chunker.split_many(texts, parallel=True)  # list[list[Chunk]]
```

## API

```python
class Chunker:
    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        min_tokens: int = 1,
        encoding: str = "cl100k_base",       # or "o200k_base"
    ) -> None: ...

    def split(self, text: str) -> list[Chunk]: ...
    def split_many(self, texts: Sequence[str], *, parallel: bool = False) -> list[list[Chunk]]: ...

class Chunk:
    text: str
    start: int          # byte offset in the original text
    end: int            # byte offset (exclusive)
    token_count: int    # exact BPE token count
```

## Algorithm

1. Split into paragraphs on `\n{2,}`, then sentences on `[.!?]\s+` plus a
   handful of abbreviations (`Mr.`, `Dr.`, `e.g.`, etc.).
2. Greedy-pack sentences into a chunk while the running token count is
   `<= max_tokens`.
3. If a single sentence exceeds `max_tokens` on its own, slice it at
   token boundaries (BPE) instead.
4. Apply `overlap_tokens` by re-prepending the last *N* tokens of each
   chunk to the next.
5. Drop chunks shorter than `min_tokens`.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
