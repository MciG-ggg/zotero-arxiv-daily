from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    published_year: Optional[int] = None
    citation_count: Optional[int] = None
    dedup_id: Optional[str] = None

    @staticmethod
    def _extract_tldr_segment(text: str) -> str:
        marker_pattern = re.compile(
            r"(?is)(?:^|\n|\s)(?:\*\*)?(?:(?:final|最终)\s*)?(?:tl\s*;?\s*dr|summary|final\s+summary|摘要|总结|一句话总结)(?:\*\*)?\s*[:：-]\s*(.+)"
        )
        matches = list(marker_pattern.finditer(text))
        return matches[-1].group(1).strip() if matches else text

    @staticmethod
    def _strip_tldr_prefix(text: str) -> str:
        prefix_pattern = re.compile(
            r"(?is)^\s*(?:\*\*)?(?:here(?:'s| is)\s+(?:the\s+)?)?(?:(?:final|最终)\s*)?(?:tl\s*;?\s*dr|summary|final\s+summary|摘要|总结|一句话总结|简而言之|总之)"
            r"(?:\*\*)?\s*[:：-]?\s*"
        )
        return prefix_pattern.sub("", text, count=1).strip()

    @staticmethod
    def _split_tldr_sentences(text: str) -> list[str]:
        sentences = re.findall(r"[^.!?。！？]+(?:[.!?。！？]+|$)", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def _is_scaffold_sentence(text: str) -> bool:
        scaffold_pattern = re.compile(
            r"(?is)^\s*(?:"
            r"let me|let's|first[, ]|second[, ]|third[, ]|here(?:'s| is)|below is|the key points|key points|"
            r"i will|we can|to summarize|in summary|overall[, ]|"
            r"让我|先来看|下面|核心要点|关键要点|总结如下|最终优化|先总结|简要总结"
            r")"
        )
        return bool(scaffold_pattern.match(text))

    def _cleanup_tldr(self, raw_tldr: str) -> str:
        text = (raw_tldr or "").strip()
        if not text:
            return text

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
        text = re.sub(r"[*_`#>]", "", text)
        text = self._extract_tldr_segment(text)

        cleaned_lines: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^(?:[-*•]+|\d+[.)]|[A-Za-z][.)])\s*", "", line)
            line = self._strip_tldr_prefix(line)
            if line:
                cleaned_lines.append(line)

        text = " ".join(cleaned_lines)
        text = re.sub(r"\s+", " ", text).strip()

        candidate_sentences = []
        for sentence in self._split_tldr_sentences(text):
            sentence = self._strip_tldr_prefix(sentence)
            sentence = re.sub(r"\s+", " ", sentence).strip(" -–—:;，、")
            if sentence and not self._is_scaffold_sentence(sentence):
                candidate_sentences.append(sentence)

        if not candidate_sentences:
            candidate_sentences = self._split_tldr_sentences(text)

        sentences = []
        for sentence in candidate_sentences:
            if sentence:
                sentences.append(sentence)
            if len(sentences) == 2:
                break

        if sentences:
            return " ".join(sentences)
        return text

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = (
            f"Given the following information of a paper, write an email-ready TLDR in {lang}. "
            "Return only the final TLDR text with no reasoning, bullets, markdown, or labels. "
            "Prefer a single conclusion sentence. Use a second sentence only when needed to preserve the key method or mechanism.\n\n"
        )
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who summarizes scientific papers for email readers. "
                        f"Answer in {lang}. Return only the final TLDR text: at most two sentences, "
                        "with one conclusion sentence preferred and an optional second sentence only for the key method or mechanism. "
                        "Do not include reasoning, analysis, bullets, markdown, prefixes, or labels."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            cleaned_tldr = self._cleanup_tldr(tldr)
            self.tldr = cleaned_tldr
            return cleaned_tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations

    def generate_affiliations(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
