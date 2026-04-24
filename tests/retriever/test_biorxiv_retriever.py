"""Tests for BiorxivRetriever."""

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.retriever.biorxiv_retriever import BiorxivRetriever
from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE


def test_biorxiv_retrieve(config, mock_biorxiv_api, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    # Only latest date + matching category
    assert len(papers) == 1
    assert papers[0].title == "A biorxiv paper"


def test_biorxiv_empty_response(config, monkeypatch):
    import requests
    from types import SimpleNamespace

    empty = {"messages": [{"status": "ok"}], "collection": []}

    def _patched(url, **kw):
        resp = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        resp.json = lambda: empty
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    assert papers == []


def test_biorxiv_convert_to_paper(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    raw = SAMPLE_BIORXIV_API_RESPONSE["collection"][0]
    paper = retriever.convert_to_paper(raw)
    assert paper.title == "A biorxiv paper"
    assert paper.source == "biorxiv"
    assert "biorxiv.org" in paper.pdf_url
    assert paper.authors == ["Smith, J.", "Doe, A.", "Lee, K."]


def test_biorxiv_requires_category(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": None}
    with pytest.raises(ValueError, match="category must be specified"):
        BiorxivRetriever(config)


def test_biorxiv_string_false_debug_does_not_truncate(monkeypatch, config):
    import requests
    from types import SimpleNamespace

    latest_date = "2026-04-24"
    collection = [
        {
            "doi": f"10.1101/2026.04.24.{idx:06d}",
            "title": f"Paper {idx}",
            "authors": "Smith, J.",
            "abstract": "Abstract",
            "date": latest_date,
            "category": "bioinformatics",
            "version": "1",
        }
        for idx in range(12)
    ]

    def _patched(url, **kw):
        resp = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        resp.json = lambda: {"messages": [{"status": "ok"}], "collection": collection}
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    with open_dict(config):
        config.executor.debug = "false"
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}

    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    assert len(papers) == 12
