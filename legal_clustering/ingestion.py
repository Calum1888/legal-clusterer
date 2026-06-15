"""
Ingestion layer: turn an uploaded archive of files into the
`{doc_id -> text}` dict that `cluster_documents` expects.

Supports .txt, .md, .pdf, and .docx. Anything else is skipped rather than
crashing the run. Files that extract to empty/whitespace are dropped too,
since they carry no text for clustering.

Each extractor stops at MAX_CHARS characters. The clusterers only use the
opening of each document (EmbeddingClusterer truncates to ~2000 chars), so
reading more than that is wasted work — the PDF reader even stops parsing
pages once the budget is hit, avoiding the back half of long contracts.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


# extensions we can read (text, markdown, pdfs, word docs)
SUPPORTED = {".txt", ".md", ".pdf", ".docx"}

# Character budget per document. Kept a little above the clusterer's
# max_chars (2000) so the embedder still sees enough to cluster on, while
# we avoid reading/parsing the rest of long documents.
MAX_CHARS = 2500


class UnsupportedFileError(ValueError):
    """
    Raised when asked to extract a file type we don't handle.
    """


def _extract_txt(data: bytes, max_chars: int = MAX_CHARS) -> str:
    '''
    Extract text from a .txt/.md file's bytes and decode to a string.

    Args:
        data (bytes): Raw file bytes.
        max_chars (int): Stop after roughly this many characters.

    Returns:
        str: Decoded text, capped at max_chars.
    '''
    # try UTF-8 first
    try:
        text = data.decode("utf-8")
    # fall back to a permissive decode rather than erroring: replace
    # characters that can't be read and keep everything else
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return text[:max_chars]


def _extract_pdf(data: bytes, max_chars: int = MAX_CHARS) -> str:
    '''
    Extract text from a PDF's bytes using pypdf, stopping once max_chars is
    reached so we never parse pages the clusterer won't see.

    Args:
        data (bytes): Raw PDF bytes.
        max_chars (int): Stop after roughly this many characters.

    Returns:
        str: Extracted text, capped at max_chars.
    '''
    # PdfReader needs a stream, so wrap the raw bytes from the zip.
    reader = PdfReader(io.BytesIO(data))
    parts, total = [], 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        total += len(text)
        if total >= max_chars:        # stop once we have enough
            break
    return "".join(parts)[:max_chars]


def _extract_docx(data: bytes, max_chars: int = MAX_CHARS) -> str:
    '''
    Extract text from a .docx file's bytes and join paragraphs into a string,
    stopping once max_chars is reached.

    Args:
        data (bytes): Raw .docx bytes.
        max_chars (int): Stop after roughly this many characters.

    Returns:
        str: Extracted text, capped at max_chars.
    '''
    # io.BytesIO treats the raw bytes from the zip as an open file
    doc = DocxDocument(io.BytesIO(data))
    parts, total = [], 0
    for p in doc.paragraphs:
        parts.append(p.text)
        total += len(p.text) + 1      # +1 for the newline we join on
        if total >= max_chars:        # stop once we have enough
            break
    return "\n".join(parts)[:max_chars]


_EXTRACTORS = {
    ".txt": _extract_txt,
    ".md": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}


def extract_text(filename: str, data: bytes, max_chars: int = MAX_CHARS) -> str:
    """
    Extract plain text from a single file's bytes, dispatched by extension.

    Args:
        filename: Name of the file (used only for its extension).
        data: Raw file bytes.
        max_chars: Character budget passed to the extractor.

    Returns:
        Extracted text, capped at max_chars (may be empty if the file had no
        extractable text).

    Raises:
        UnsupportedFileError: If the extension isn't in SUPPORTED.
    """
    # get file extension (.txt, .pdf, .md, .docx)
    ext = Path(filename).suffix.lower()
    # match extension to the extractor functions in _EXTRACTORS
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise UnsupportedFileError(f"unsupported file type: {ext or '(none)'}")
    # run the correct extractor on the data
    return extractor(data, max_chars)


def _is_junk(name: str) -> bool:
    """
    Skip directories, macOS resource forks, and hidden files.
    """
    base = os.path.basename(name)
    return (
        name.endswith("/")
        or name.startswith("__MACOSX")
        or base.startswith(".")
        or not base
    )


def load_documents_from_zip(zip_source) -> dict[str, str]:
    """
    Read a zip archive and return {doc_id -> text} for every supported,
    non-empty file inside it (including files in nested folders).

    Args:
        zip_source: A path, a file-like object, or raw bytes of a .zip.

    Returns:
        Mapping of doc_id -> extracted text. doc_id is the file's base name;
        on a name collision the relative path is used to disambiguate.

    Raises:
        zipfile.BadZipFile: If the archive can't be read as a zip.
    """
    # convert input to a file-like object for zipfile.ZipFile
    if isinstance(zip_source, (bytes, bytearray)):
        zip_source = io.BytesIO(zip_source)

    # storage for accepted / skipped documents
    documents: dict[str, str] = {}
    skipped: list[str] = []

    with zipfile.ZipFile(zip_source) as zf:

        for info in zf.infolist():
            name = info.filename
            # drop folder entries and junk
            if info.is_dir() or _is_junk(name):
                continue
            # skip if the extension isn't supported
            if Path(name).suffix.lower() not in SUPPORTED:
                skipped.append(name)
                continue

            data = zf.read(name)
            try:
                # extract text (capped at MAX_CHARS)
                text = extract_text(name, data).strip()

            # if extraction fails for any reason, skip rather than crash
            except Exception as exc:
                skipped.append(f"{name} (extraction failed: {exc})")
                continue

            # nothing extractable (e.g. an image-only PDF) -> skip
            if not text:
                skipped.append(f"{name} (empty)")
                continue

            # prefer the bare filename as the id; fall back to the full path
            # if two files share a name
            doc_id = os.path.basename(name)
            if doc_id in documents:
                doc_id = name
            documents[doc_id] = text

    if skipped:
        # report how many were skipped and the first 10 names
        print(f"Skipped {len(skipped)} file(s): {skipped[:10]}"
              + (" ..." if len(skipped) > 10 else ""))

    # return accepted files
    return documents