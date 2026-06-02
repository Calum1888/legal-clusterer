"""
Ingestion layer: turn an uploaded archive of files into the
`{doc_id -> text}` dict that `cluster_documents` expects.

Supports .txt, .md, .pdf, and .docx. Anything else is skipped rather than crashing the run.
Files that extract to empty/whitespace are dropped too, since they carry no text for clustering.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument


# extensions we can read (text, mardown, pdfs, word docs)
SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


class UnsupportedFileError(ValueError):
    """
    Raised when asked to extract a file type we don't handle.
    """


def _extract_txt(data: bytes) -> str:
    '''
    Extracts text from the ingested text files and converts to a string.

    Args:
        data (bytes): data from ingested text files to be decoded.
    
    Returns:
        str: string of text from the ingested text file.
    '''
    # try UTF-8 
    try:
        return data.decode("utf-8")
    # fall back to a permissive decode rather than erroring
    # replace charatcters that can not be read and keep everywhere else  
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> str:
    '''
    Extracts test from ingeted pdf files and converts to a string using 
    pdfplumber.  

    Args:
        data (bytes): data from ingested pdf file.

    Returns:
        str: A string of the pdf text content.
    '''
    parts: list[str] = []
    # opens data as a pdf
    # io.BytesIO treats raw bytes from zip file as an open file 
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            # for each page, extract text from that page
            # add it to storage, parts
            parts.append(page.extract_text() or "")
    # return pdf content with pages joined by new lines
    return "\n".join(parts)


def _extract_docx(data: bytes) -> str:
    '''
    Extract data from ingested docx file and converts to a string.
    
    Args:
        data (bytes): data from docx file
    
    Returns:
        str: A string of the docx's text content.
    '''
    # opens docx 
    # io.BytesIO treats raw bytes from zip file as an open file
    doc = DocxDocument(io.BytesIO(data))
    # for each paragraph, join the raw text together with new lines
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
    # get file extension (.txt, .pdf, .md, .docx)
    ext = Path(filename).suffix.lower()
    # matches extension to the extractor functions in _EXTRACTORS
    extractor = _EXTRACTORS.get(ext)
    if extractor is None: 
        raise UnsupportedFileError(f"unsupported file type: {ext or '(none)'}")
    # runs the correct extractor on the data
    return extractor(data)


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
    # converts input to file-like object to be opened by zipfile.ZipFile
    if isinstance(zip_source, (bytes, bytearray)):
        zip_source = io.BytesIO(zip_source)

    # storgae for accepted/skipped documents 
    documents: dict[str, str] = {}
    skipped: list[str] = []

    with zipfile.ZipFile(zip_source) as zf:
        
        for info in zf.infolist():
            name = info.filename
            # drops folder entries and junk
            if info.is_dir() or _is_junk(name):
                continue
            # skip document if extension is not supported 
            if Path(name).suffix.lower() not in SUPPORTED:
                # add to skipped storage
                skipped.append(name)
                continue

            data = zf.read(name)
            try:
                # extract text
                text = extract_text(name, data).strip()

            # if extraction fails for whatever reason, skip do not crash    
            except Exception as exc: 
                skipped.append(f"{name} (extraction failed: {exc})")
                continue
            
            # if not text then skip, i.e images etc. 
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
        # print amount that were skipped and first 10 files names that are skipped
        print(f"Skipped {len(skipped)} file(s): {skipped[:10]}"
              + (" ..." if len(skipped) > 10 else ""))

    # return accepted files
    return documents