import random
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .classics import (
    OpenAlexClassicRetriever,
    _as_bool,
    _paper_matches_topic_rules,
    get_paper_dedup_id,
    load_classic_history,
    save_classic_history,
)
from .construct_email import render_email
from .protocol import CorpusPaper, Paper
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils import glob_match, send_email


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
        self.classic_retriever = OpenAlexClassicRetriever(config) if _as_bool(config.classics.enabled) else None

    def _filter_latest_candidates_by_topic_rules(self, papers: list[Paper]) -> list[Paper]:
        if self.classic_retriever is None or not papers:
            return papers

        filtered = [paper for paper in papers if _paper_matches_topic_rules(paper, self.config.classics)]
        logger.info(
            f"Retained {len(filtered)} latest candidates after explicit embodied-topic filtering "
            f"(dropped {len(papers) - len(filtered)})"
        )
        if not filtered:
            logger.warning(
                "Explicit embodied-topic filtering removed all latest-paper candidates. "
                "Check your arXiv categories and topic-filter keywords."
            )
        return filtered

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {collection['key']: collection for collection in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [paper for paper in corpus if paper['data']['abstractNote'] != '']

        def get_collection_path(col_key: str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            return collections[col_key]['data']['name']

        for paper in corpus:
            paper['paths'] = [get_collection_path(col) for col in paper['data']['collections']]
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [
            CorpusPaper(
                title=paper['data']['title'],
                abstract=paper['data']['abstractNote'],
                added_date=datetime.strptime(paper['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
                paths=paper['paths'],
            )
            for paper in corpus
        ]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                paper for paper in corpus
                if any(
                    glob_match(path, pattern)
                    for path in paper.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                paper for paper in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in paper.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            sample_text = '\n'.join([paper.title + ' - ' + '\n'.join(paper.paths) for paper in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{sample_text}\n...")
        return corpus

    def _retrieve_latest_papers(self) -> list[Paper]:
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all latest-paper sources")
        return self._filter_latest_candidates_by_topic_rules(all_papers)

    def _retrieve_classic_papers(self, corpus: list[CorpusPaper]) -> list[Paper]:
        if self.classic_retriever is None:
            return []

        logger.info("Retrieving classic papers...")
        classic_candidates = self.classic_retriever.retrieve_papers()
        if not classic_candidates:
            return []

        history_ids = load_classic_history(self.config.classics.history_path)
        filtered_candidates = [
            paper for paper in classic_candidates if get_paper_dedup_id(paper) not in history_ids
        ]
        logger.info(
            f"Retained {len(filtered_candidates)} classic candidates after removing "
            f"{len(classic_candidates) - len(filtered_candidates)} already-sent classics"
        )
        if not filtered_candidates:
            return []

        reranked = self.reranker.rerank(filtered_candidates, corpus)
        limit = int(self.config.classics.max_paper_num)
        return reranked[:limit]

    def _enrich_papers(self, papers: list[Paper]) -> None:
        if not papers:
            return
        logger.info("Generating TLDR and affiliations...")
        for paper in tqdm(papers):
            paper.generate_tldr(self.openai_client, self.config.llm)
            if paper.affiliations is None:
                paper.generate_affiliations(self.openai_client, self.config.llm)

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return

        latest_candidates = self._retrieve_latest_papers()
        latest_papers = []
        if latest_candidates:
            logger.info("Reranking latest papers...")
            latest_papers = self.reranker.rerank(latest_candidates, corpus)
            latest_papers = latest_papers[:self.config.executor.max_paper_num]

        classic_papers = self._retrieve_classic_papers(corpus)

        if not latest_papers and not classic_papers and not _as_bool(self.config.executor.send_empty):
            logger.info("No new or classic papers found. No email will be sent.")
            return

        self._enrich_papers(latest_papers)
        self._enrich_papers(classic_papers)

        logger.info("Sending email...")
        email_content = render_email(latest_papers, classic_papers)
        send_email(self.config, email_content)

        if classic_papers and _as_bool(self.config.classics.persist_history):
            updated = save_classic_history(self.config.classics.history_path, classic_papers)
            if updated:
                logger.info(f"Updated classic recommendation history at {self.config.classics.history_path}")
            else:
                logger.info("Classic recommendation history already contained all sent classics")

        logger.info("Email sent successfully")
