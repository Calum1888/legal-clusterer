"""
Ingestion layer: turn an uploaded archive of files into the
`{doc_id -> text}` dict that `cluster_documents` expects.

Supports .txt, .md, .pdf, and .docx. Anything else (images, spreadsheets,
junk like __MACOSX/ or .DS_Store) is skipped rather than crashing the run.
Files that extract to empty/whitespace are dropped too, since they carry no
clustering signal.

This module handles per-file extraction. Corpus-level guards (too few / too
many documents to cluster) live in the size/quality checks (step 3), not here.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument


# Extensions we know how to read. Everything else is skipped.
SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


class UnsupportedFileError(ValueError):
    """Raised when asked to extract a file type we don't handle."""


def _extract_txt(data: bytes) -> str:
    # Try UTF-8 first; fall back to a permissive decode rather than erroring
    # on the occasional mis-encoded byte.
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _extract_docx(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


_EXTRACTORS = {
    ".txt": _extract_txt,
    ".md": _extract_txt,
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
}


def extract_text(filename: str, data: bytes) -> str:
    """
    Extract plain text from a single file's bytes, dispatched by extension.

    Args:
        filename: Name of the file (used only for its extension).
        data: Raw file bytes.

    Returns:
        Extracted text (may be empty if the file had no extractable text).

    Raises:
        UnsupportedFileError: If the extension isn't in SUPPORTED.
    """
    ext = Path(filename).suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise UnsupportedFileError(f"unsupported file type: {ext or '(none)'}")
    return extractor(data)


def _is_junk(name: str) -> bool:
    """Skip directories, macOS resource forks, and hidden files."""
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
    if isinstance(zip_source, (bytes, bytearray)):
        zip_source = io.BytesIO(zip_source)

    documents: dict[str, str] = {}
    skipped: list[str] = []

    with zipfile.ZipFile(zip_source) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or _is_junk(name):
                continue
            if Path(name).suffix.lower() not in SUPPORTED:
                skipped.append(name)
                continue

            data = zf.read(name)
            try:
                text = extract_text(name, data).strip()
            except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the batch
                skipped.append(f"{name} (extraction failed: {exc})")
                continue

            if not text:
                skipped.append(f"{name} (empty)")
                continue

            # Prefer the bare filename as the id; fall back to the full path
            # if two files share a name.
            doc_id = os.path.basename(name)
            if doc_id in documents:
                doc_id = name
            documents[doc_id] = text

    if skipped:
        print(f"Skipped {len(skipped)} file(s): {skipped[:10]}"
              + (" ..." if len(skipped) > 10 else ""))

    return documents