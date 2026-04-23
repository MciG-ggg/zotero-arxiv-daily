import json

import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.classics import (
    OpenAlexClassicRetriever,
    get_paper_dedup_id,
    load_classic_history,
    save_classic_history,
)


def _make_openalex_work(
    *,
    work_id: str = "https://openalex.org/W123",
    title: str = "Embodied robot learning for manipulation",
    abstract: str = "Embodied robot manipulation with reinforcement learning.",
    year: int = 2020,
    citations: int = 321,
):
    abstract_words = abstract.split()
    return {
        "id": work_id,
        "display_name": title,
        "publication_year": year,
        "cited_by_count": citations,
        "abstract_inverted_index": {
            word: [idx] for idx, word in enumerate(abstract_words)
        },
        "doi": "https://doi.org/10.1234/example",
        "primary_location": {
            "landing_page_url": "https://example.com/paper",
            "pdf_url": "https://example.com/paper.pdf",
        },
        "authorships": [
            {
                "author": {"display_name": "Alice"},
                "institutions": [{"display_name": "MIT"}],
            },
            {
                "author": {"display_name": "Bob"},
                "institutions": [{"display_name": "Stanford"}],
            },
        ],
    }


def test_classics_require_openalex_credentials_when_enabled(config):
    with open_dict(config):
        config.classics.enabled = True
        config.classics.openalex.api_key = None
        config.classics.openalex.mailto = None

    with pytest.raises(ValueError, match="config.classics.enabled=true"):
        OpenAlexClassicRetriever(config)


def test_openalex_retriever_filters_and_normalizes(config, monkeypatch):
    with open_dict(config):
        config.classics.enabled = True
        config.classics.openalex.mailto = "test@example.com"
        config.classics.min_citation_count = 100
        config.classics.max_publication_year = 9999
        config.classics.max_age_years = 10
        config.classics.topic_filter.keywords = ["embodied", "robot"]
        config.classics.topic_filter.required_keywords_any = ["robot"]
        config.classics.openalex.query_terms = ["embodied ai"]

    class StubResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    _make_openalex_work(),
                    _make_openalex_work(
                        work_id="https://openalex.org/W456",
                        title="Unrelated chemistry paper",
                        abstract="A chemistry abstract without any control or laboratory language overlap.",
                    ),
                ]
            }

    monkeypatch.setattr("zotero_arxiv_daily.classics.requests.get", lambda *a, **kw: StubResponse())
    papers = OpenAlexClassicRetriever(config).retrieve_papers()

    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Embodied robot learning for manipulation"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.affiliations == ["MIT", "Stanford"]
    assert paper.published_year == 2020
    assert paper.citation_count == 321
    assert get_paper_dedup_id(paper) == "https://openalex.org/W123"


def test_openalex_retriever_excludes_papers_older_than_max_age(config, monkeypatch):
    with open_dict(config):
        config.classics.enabled = True
        config.classics.openalex.mailto = "test@example.com"
        config.classics.max_publication_year = 9999
        config.classics.max_age_years = 5
        config.classics.topic_filter.keywords = ["robot", "manipulation"]
        config.classics.topic_filter.required_keywords_any = ["robot"]
        config.classics.min_citation_count = 100

    class StubResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    _make_openalex_work(year=2019),
                    _make_openalex_work(work_id="https://openalex.org/W456", year=2024, citations=400),
                ]
            }

    monkeypatch.setattr("zotero_arxiv_daily.classics.requests.get", lambda *a, **kw: StubResponse())
    papers = OpenAlexClassicRetriever(config).retrieve_papers()

    assert len(papers) == 1
    assert papers[0].published_year == 2024


def test_openalex_retriever_excludes_llm_only_papers_without_robotics_terms(config, monkeypatch):
    with open_dict(config):
        config.classics.enabled = True
        config.classics.openalex.mailto = "test@example.com"
        config.classics.max_publication_year = 9999
        config.classics.max_age_years = 10
        config.classics.min_citation_count = 100
        config.classics.topic_filter.keywords = ["embodied", "robot", "manipulation", "humanoid"]
        config.classics.topic_filter.required_keywords_any = ["robot", "manipulation", "humanoid"]

    class StubResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    _make_openalex_work(
                        title="Embodied large language model agents",
                        abstract="Embodied agents powered by large language models collaborate in virtual worlds.",
                        work_id="https://openalex.org/W111",
                        year=2025,
                    ),
                    _make_openalex_work(
                        title="Humanoid robot manipulation with foundation models",
                        abstract="A humanoid robot learns manipulation with a foundation model and visuomotor policy.",
                        work_id="https://openalex.org/W222",
                        year=2025,
                        citations=500,
                    ),
                ]
            }

    monkeypatch.setattr("zotero_arxiv_daily.classics.requests.get", lambda *a, **kw: StubResponse())
    papers = OpenAlexClassicRetriever(config).retrieve_papers()

    assert [paper.title for paper in papers] == ["Humanoid robot manipulation with foundation models"]


def test_classic_history_load_and_save_roundtrip(tmp_path):
    path = tmp_path / "classic_history.json"
    paper = make_sample_paper(
        title="Classic",
        url="https://example.com/classic",
        dedup_id="https://openalex.org/W123",
        published_year=2019,
        citation_count=456,
    )
    updated = save_classic_history(str(path), [paper])
    assert updated is True
    assert load_classic_history(str(path)) == {"https://openalex.org/W123"}

    updated_again = save_classic_history(str(path), [paper])
    assert updated_again is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["recommended"][0]["title"] == "Classic"


def test_load_classic_history_ignores_invalid_payload_shape(tmp_path):
    path = tmp_path / "classic_history.json"
    path.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
    assert load_classic_history(str(path)) == set()
