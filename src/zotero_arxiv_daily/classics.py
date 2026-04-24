from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger
from omegaconf import DictConfig

from .protocol import Paper

REQUEST_TIMEOUT = (10, 60)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def get_paper_dedup_id(paper: Paper) -> str:
    return paper.dedup_id or paper.url


def load_classic_history(history_path: str) -> set[str]:
    path = Path(history_path)
    if not path.exists():
        return set()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse classic history from {path}: {exc}")
        return set()

    if not isinstance(payload, dict):
        logger.warning(f"Classic history at {path} must be a JSON object; ignoring it.")
        return set()

    entries = payload.get("recommended", [])
    if not isinstance(entries, list):
        logger.warning(f"Classic history at {path} has invalid 'recommended' payload; ignoring it.")
        return set()

    dedup_ids = set()
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            dedup_ids.add(entry["id"])
    return dedup_ids


def save_classic_history(history_path: str, papers: list[Paper]) -> bool:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_entries = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing_entries = payload.get("recommended", [])
        except json.JSONDecodeError:
            logger.warning(f"Classic history file {path} is invalid JSON; recreating it.")

    if not isinstance(existing_entries, list):
        logger.warning(f"Classic history file {path} has invalid 'recommended' payload; recreating it.")
        existing_entries = []

    existing_by_id = {
        entry["id"]: entry
        for entry in existing_entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    before_count = len(existing_by_id)

    now = datetime.now(timezone.utc).isoformat()
    for paper in papers:
        dedup_id = get_paper_dedup_id(paper)
        existing_by_id.setdefault(
            dedup_id,
            {
                "id": dedup_id,
                "title": paper.title,
                "url": paper.url,
                "source": paper.source,
                "published_year": paper.published_year,
                "citation_count": paper.citation_count,
                "recommended_at": now,
            },
        )

    payload = {
        "version": 1,
        "recommended": sorted(existing_by_id.values(), key=lambda item: item["id"]),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(existing_by_id) != before_count


def _reconstruct_abstract(abstract_index: dict[str, list[int]] | None) -> str:
    if not abstract_index:
        return ""

    words_by_position: dict[int, str] = {}
    for word, positions in abstract_index.items():
        for position in positions:
            words_by_position[position] = word
    return " ".join(words_by_position[idx] for idx in sorted(words_by_position))


def _normalize_authors(raw_work: dict) -> tuple[list[str], list[str] | None]:
    authors = []
    affiliations = []
    for authorship in raw_work.get("authorships", []):
        author = authorship.get("author") or {}
        author_name = author.get("display_name")
        if author_name:
            authors.append(author_name)
        for institution in authorship.get("institutions", []):
            name = institution.get("display_name")
            if name and name not in affiliations:
                affiliations.append(name)
    return authors, affiliations or None


def _paper_matches_topic_rules(paper: Paper, classics_config: DictConfig) -> bool:
    topic_filter = classics_config.topic_filter
    keywords = [keyword.lower() for keyword in topic_filter.keywords]
    required_keywords = [keyword.lower() for keyword in topic_filter.get("required_keywords_any", [])]
    text = " ".join(part for part in [paper.title, paper.abstract] if part).lower()
    keyword_hits = 0
    for keyword in keywords:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b")
        if pattern.search(text):
            keyword_hits += 1
    if keyword_hits < int(topic_filter.min_keyword_matches):
        return False

    if not required_keywords:
        return True

    return any(re.compile(rf"\b{re.escape(keyword)}\b").search(text) for keyword in required_keywords)


class OpenAlexClassicRetriever:
    def __init__(self, config: DictConfig):
        self.config = config
        self.classics_config = config.classics
        self.openalex_config = config.classics.openalex

        if _as_bool(self.classics_config.enabled) and not (self.openalex_config.api_key or self.openalex_config.mailto):
            raise ValueError(
                "config.classics.enabled=true requires config.classics.openalex.api_key "
                "or config.classics.openalex.mailto."
            )

    def _get_publication_year_bounds(self) -> tuple[int, int]:
        current_year = datetime.now(timezone.utc).year
        max_publication_year = min(int(self.classics_config.max_publication_year), current_year)
        min_publication_year = current_year - int(self.classics_config.max_age_years) + 1
        return min_publication_year, max_publication_year

    def _build_request_params(self, query: str) -> dict[str, str | int]:
        min_publication_year, max_publication_year = self._get_publication_year_bounds()
        params: dict[str, str | int] = {
            "search": query,
            "sort": "cited_by_count:desc",
            "per-page": int(self.classics_config.candidate_pool_size),
            "filter": (
                "has_abstract:true,"
                "is_retracted:false,"
                f"from_publication_date:{min_publication_year}-01-01,"
                f"to_publication_date:{max_publication_year}-12-31"
            ),
        }
        if self.openalex_config.api_key:
            params["api_key"] = self.openalex_config.api_key
        if self.openalex_config.mailto:
            params["mailto"] = self.openalex_config.mailto
        return params

    def _fetch_query_results(self, query: str) -> list[dict]:
        url = f"{self.openalex_config.base_url.rstrip('/')}/works"
        response = requests.get(url, params=self._build_request_params(query), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])

    def _convert_to_paper(self, raw_work: dict) -> Paper | None:
        publication_year = raw_work.get("publication_year")
        cited_by_count = raw_work.get("cited_by_count")
        if publication_year is None or cited_by_count is None:
            return None

        min_publication_year, max_publication_year = self._get_publication_year_bounds()
        if publication_year > max_publication_year:
            return None
        if publication_year < min_publication_year:
            return None
        if cited_by_count < int(self.classics_config.min_citation_count):
            return None

        title = raw_work.get("display_name") or raw_work.get("title")
        if not title:
            return None

        abstract = _reconstruct_abstract(raw_work.get("abstract_inverted_index"))
        if not abstract:
            return None

        authors, affiliations = _normalize_authors(raw_work)
        primary_location = raw_work.get("primary_location") or {}
        best_location = raw_work.get("best_oa_location") or {}
        url = (
            primary_location.get("landing_page_url")
            or best_location.get("landing_page_url")
            or raw_work.get("doi")
            or raw_work.get("id")
        )
        pdf_url = (
            primary_location.get("pdf_url")
            or best_location.get("pdf_url")
            or primary_location.get("landing_page_url")
            or best_location.get("landing_page_url")
            or raw_work.get("doi")
            or raw_work.get("id")
        )
        dedup_id = raw_work.get("id") or raw_work.get("doi") or url
        paper = Paper(
            source="openalex-classics",
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            affiliations=affiliations,
            published_year=publication_year,
            citation_count=cited_by_count,
            dedup_id=dedup_id,
        )
        if not _paper_matches_topic_rules(paper, self.classics_config):
            return None
        return paper

    def retrieve_papers(self) -> list[Paper]:
        query_terms = list(self.openalex_config.query_terms)
        if not query_terms:
            logger.warning("No OpenAlex classic query terms configured; skipping classic retrieval.")
            return []

        merged_by_id: dict[str, Paper] = {}
        for query in query_terms:
            logger.info(f"Retrieving OpenAlex classics for query: {query}")
            for raw_work in self._fetch_query_results(query):
                paper = self._convert_to_paper(raw_work)
                if paper is None:
                    continue
                dedup_id = get_paper_dedup_id(paper)
                current = merged_by_id.get(dedup_id)
                if current is None or (paper.citation_count or 0) > (current.citation_count or 0):
                    merged_by_id[dedup_id] = paper

        papers = list(merged_by_id.values())
        papers.sort(
            key=lambda paper: (
                paper.citation_count or 0,
                paper.published_year or 0,
                paper.title.lower(),
            ),
            reverse=True,
        )
        logger.info(f"Retrieved {len(papers)} classic candidates from OpenAlex")
        return papers
