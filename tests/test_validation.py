"""
Tests for the corpus-size guards in validation.py.

ASSUMPTION: validate_corpus(documents, min_documents, max_documents) raises
CorpusError when the corpus is outside [min, max] and returns normally when it
is valid. If your signature differs (different names, keyword-only, a different
guard for "too uniform"), adjust the calls below — the intent stays the same.
"""
import pytest
from legal_clustering.validation import validate_corpus, CorpusError


def test_too_few_documents_raises(corpus):
    one_doc = dict(list(corpus.items())[:1])
    with pytest.raises(CorpusError):
        validate_corpus(one_doc, min_documents=3, max_documents=1000)


def test_too_many_documents_raises(corpus):
    with pytest.raises(CorpusError):
        validate_corpus(corpus, min_documents=1, max_documents=3)


def test_valid_corpus_passes(corpus):
    # should not raise
    validate_corpus(corpus, min_documents=2, max_documents=1000)
