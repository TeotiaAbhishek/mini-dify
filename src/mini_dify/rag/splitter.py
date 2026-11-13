# Try to split on the biggest natural boundary first: paragraphs, then lines,
# then sentences, then words, and only as a last resort in the middle of a word.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters.

    Neighbouring chunks share about `chunk_overlap` characters, so a sentence cut
    at a boundary still appears whole in one of them.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    pieces = _split_recursive(text, chunk_size, SEPARATORS)
    return _merge(pieces, chunk_size, chunk_overlap)


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than chunk_size, using the best separator."""
    if len(text) <= chunk_size:
        return [text]

    separator = next(sep for sep in separators if sep == "" or sep in text)
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    # Keep the separator on the end of each part, so joining parts restores the text.
    parts = [part + separator for part in parts[:-1]] + [parts[-1]]

    remaining = separators[separators.index(separator) + 1 :]
    pieces: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, chunk_size, remaining))
    return [piece for piece in pieces if piece]


def _merge(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Glue small pieces back together into chunks, carrying an overlap forward."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        if current and current_len + len(piece) > chunk_size:
            chunks.append("".join(current).strip())
            # Drop pieces from the front until what's left fits the overlap
            # and leaves room for the new piece.
            while current and (
                current_len > chunk_overlap or current_len + len(piece) > chunk_size
            ):
                current_len -= len(current[0])
                current.pop(0)
        current.append(piece)
        current_len += len(piece)

    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]
