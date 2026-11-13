import math
import uuid

import pytest

from mini_dify.llm.embeddings import EMBEDDING_DIM, FakeEmbedding
from mini_dify.rag.extract import UnsupportedFileType, clean_text, extract_text
from mini_dify.rag.retrieval import RetrievedChunk, reciprocal_rank_fusion
from mini_dify.rag.splitter import split_text

# ---------- Splitting ----------


def test_short_text_is_one_chunk():
    assert split_text("Just one short line.", chunk_size=100, chunk_overlap=10) == [
        "Just one short line."
    ]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n  ", chunk_size=100, chunk_overlap=10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(12))
    chunks = split_text(text, chunk_size=100, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    # Each chunk starts with the last sentence of the chunk before it (the overlap).
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.split(".")[0] in previous


def test_paragraph_boundaries_are_preferred():
    text = "First paragraph about cats.\n\nSecond paragraph about dogs."
    assert split_text(text, chunk_size=40, chunk_overlap=0) == [
        "First paragraph about cats.",
        "Second paragraph about dogs.",
    ]


def test_text_without_separators_is_cut_hard():
    chunks = split_text("a" * 250, chunk_size=100, chunk_overlap=0)
    assert [len(chunk) for chunk in chunks] == [100, 100, 50]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("anything", chunk_size=100, chunk_overlap=100)


# ---------- Extraction and cleaning ----------


def test_extract_text_file():
    assert extract_text("notes.md", b"# Title\nhello") == "# Title\nhello"


def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileType):
        extract_text("photo.png", b"...")


def test_clean_text_removes_nul_and_extra_whitespace():
    assert clean_text("a\x00b   c\r\n\n\n\n\nd") == "ab c\n\nd"


# ---------- Fake embeddings ----------


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.mark.anyio
async def test_fake_embedding_shape_and_similarity():
    embedder = FakeEmbedding()
    refund, refund_again, weather = await embedder.embed(
        ["refund policy for orders", "what is the refund policy", "sunny weather today"]
    )
    assert len(refund) == EMBEDDING_DIM
    assert math.isclose(cosine(refund, refund), 1.0)
    assert cosine(refund, refund_again) > cosine(refund, weather)


# ---------- Reciprocal rank fusion ----------


def make_chunk(name: str) -> RetrievedChunk:
    return RetrievedChunk(uuid.uuid5(uuid.NAMESPACE_DNS, name), uuid.uuid4(), "f.txt", 0, name, 0)


def test_rrf_rewards_chunks_found_by_both_searches():
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")
    by_meaning = [a, b]  # a is 1st by meaning
    by_keyword = [c, b]  # c is 1st by keyword
    fused = reciprocal_rank_fusion([by_meaning, by_keyword], top_k=3)

    # b is only 2nd in each list, but appears in both, so it wins overall.
    assert [chunk.content for chunk in fused] == ["b", "a", "c"]
    assert fused[0].score == round(2 / 62, 4)
