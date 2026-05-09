"""End-to-end tests for the Python facade."""

from __future__ import annotations

import pytest
from snipsplit import Chunk, Chunker, SnipsplitError, __version__


def test_version_present() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_short_text_one_chunk() -> None:
    c = Chunker(max_tokens=100)
    chunks = c.split("hello world")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert isinstance(chunks[0], Chunk)


def test_empty_text_no_chunks() -> None:
    c = Chunker(max_tokens=100)
    assert c.split("") == []


def test_splits_at_sentence_boundary() -> None:
    c = Chunker(max_tokens=8)
    chunks = c.split("Alpha beta gamma. Delta epsilon zeta. Eta theta iota.")
    assert len(chunks) >= 2
    for ch in chunks:
        assert ch.token_count <= 8


def test_long_sentence_falls_back_to_token_slicing() -> None:
    c = Chunker(max_tokens=5)
    chunks = c.split("the quick brown fox jumps over the lazy dog and runs through fields")
    assert len(chunks) > 1
    for ch in chunks:
        assert ch.token_count <= 5


def test_overlap_tokens_re_prepended() -> None:
    c = Chunker(max_tokens=6, overlap_tokens=2)
    chunks = c.split("Alpha beta gamma. Delta epsilon zeta. Eta theta iota.")
    assert len(chunks) >= 2
    for ch in chunks[1:]:
        assert ch.token_count <= 6 + 2


def test_min_tokens_filters() -> None:
    c = Chunker(max_tokens=100, min_tokens=5)
    assert c.split("hi") == []


def test_invalid_zero_max_tokens_rejected() -> None:
    with pytest.raises(ValueError):
        Chunker(max_tokens=0)


def test_invalid_overlap_ge_max_rejected() -> None:
    with pytest.raises(ValueError):
        Chunker(max_tokens=10, overlap_tokens=10)


def test_unknown_encoding_rejected() -> None:
    with pytest.raises(ValueError):
        Chunker(encoding="not_a_thing")


def test_o200k_encoding() -> None:
    c = Chunker(max_tokens=100, encoding="o200k_base")
    chunks = c.split("Hello world.")
    assert len(chunks) == 1


def test_split_many_serial_and_parallel_match() -> None:
    c = Chunker(max_tokens=10)
    texts = ["Alpha beta gamma.", "Delta. Epsilon. Zeta."]
    serial = c.split_many(texts)
    parallel = c.split_many(texts, parallel=True)
    assert len(serial) == len(parallel) == 2
    for s, p in zip(serial, parallel, strict=True):
        assert [(c.start, c.end, c.token_count) for c in s] == [
            (c.start, c.end, c.token_count) for c in p
        ]


def test_chunk_dataclass_immutable() -> None:
    ch = Chunker(max_tokens=100).split("hello")[0]
    with pytest.raises(AttributeError):
        ch.text = "other"  # type: ignore[misc]


def test_split_many_empty_list() -> None:
    c = Chunker(max_tokens=100)
    assert c.split_many([]) == []


def test_unicode_input() -> None:
    c = Chunker(max_tokens=100)
    chunks = c.split("你好世界. Hello world. 🌍 done.")
    assert len(chunks) >= 1
    for ch in chunks:
        assert ch.text != ""


def test_native_error_class_exposed() -> None:
    assert issubclass(SnipsplitError, Exception)


def test_chunk_offsets_are_byte_safe() -> None:
    c = Chunker(max_tokens=20)
    text = "Alpha. Beta. Gamma."
    chunks = c.split(text)
    for ch in chunks:
        assert 0 <= ch.start < ch.end <= len(text)
