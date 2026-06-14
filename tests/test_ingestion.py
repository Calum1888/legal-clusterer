"""
Tests for load_documents_from_zip.

ASSUMPTION: it takes a path to a .zip and returns a dict {name: text}, skipping
empty files (matching the "Skipped N file(s)" behaviour seen at runtime). If it
returns a list or document objects, adjust the length/identity assertions.
"""
import zipfile

from legal_clustering.ingestion import load_documents_from_zip


def _make_zip(tmp_path, files: dict) -> str:
    path = tmp_path / "docs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return str(path)


def test_loads_text_files(tmp_path):
    z = _make_zip(tmp_path, {
        "a.txt": "first document about cats",
        "b.txt": "second document about finance",
    })
    docs = load_documents_from_zip(z)
    assert len(docs) == 2


def test_skips_empty_files(tmp_path):
    z = _make_zip(tmp_path, {
        "good.txt": "real content here that is not empty",
        "empty.txt": "",
    })
    docs = load_documents_from_zip(z)
    assert len(docs) == 1
