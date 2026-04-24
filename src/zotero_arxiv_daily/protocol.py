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
    def _truncate_prompt(prompt: str, max_tokens: int) -> str:
        enc = tiktoken.encoding_for_model("gpt-4o")
        return enc.decode(enc.encode(prompt)[:max_tokens])

    def _build_tldr_context(self) -> str:
        context = ""
        if self.title:
            context += f"Title:\n{self.title}\n\n"
        if self.abstract:
            context += f"Abstract:\n{self.abstract}\n\n"
        if self.full_text:
            context += f"Preview of main content:\n{self.full_text}\n\n"
        return context

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
            r"(?is)^\s*(?:\*\*)?(?:here(?:'s| is)\s+(?:the\s+)?)?(?:(?:final|最终)\s*)?(?:"
            r"tl\s*;?\s*dr|summary|final\s+summary|摘要|总结|一句话总结|简而言之|总之|"
            r"combined|or\s+shorter|shorter(?:\s+version)?|or\s+more\s+concisely(?:\s+focusing\s+on\s+the\s+conclusion)?|"
            r"sentence\s*[12](?:\s*\([^)]*\))?|main\s+conclusion(?:\s+sentence)?|conclusion(?:\s+sentence)?|"
            r"key\s+method(?:\s*/\s*mechanism)?(?:\s*\([^)]*\))?|method(?:\s*/\s*mechanism)?|"
            r"更简洁地说|更简短地说|更简洁版本|组合版|合并版|结论句|方法句)"
            r"(?:\*\*)?\s*[:：-]?\s*"
        )
        return prefix_pattern.sub("", text, count=1).strip()

    @staticmethod
    def _split_tldr_sentences(text: str) -> list[str]:
        text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
        sentences: list[str] = []
        buffer: list[str] = []

        for index, char in enumerate(text):
            buffer.append(char)
            previous_char = text[index - 1] if index > 0 else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""

            if char in "。！？":
                sentences.append("".join(buffer).strip())
                buffer = []
                continue

            if char in ".!?":
                if previous_char.isdigit() and next_char.isdigit():
                    continue
                sentences.append("".join(buffer).strip())
                buffer = []

        if buffer:
            remainder = "".join(buffer).strip()
            if remainder:
                sentences.append(remainder)

        return [sentence for sentence in sentences if sentence]

    @staticmethod
    def _join_tldr_sentences(sentences: list[str]) -> str:
        if not sentences:
            return ""

        result = sentences[0]
        for sentence in sentences[1:]:
            separator = "" if re.search(r"[。！？]\s*$", result) else " "
            result += f"{separator}{sentence}"
        return result

    @staticmethod
    def _is_scaffold_sentence(text: str) -> bool:
        scaffold_pattern = re.compile(
            r"(?is)^\s*(?:"
            r"let me|let's|first[, ]|second[, ]|third[, ]|here(?:'s| is)|below is|the key points|key points|"
            r"i will|we can|to summarize|in summary|overall[, ]|this captures|the previous draft|"
            r"main conclusion|the main conclusion|key method|the key method|application context|"
            r"the application context|let me check if this meets the requirements|extra analysis|"
            r"supplementary note|supplementary explanation|this is one sentence|this is two sentences|"
            r"or shorter|combined|sentence\s*[12]|"
            r"让我|先来看|下面|核心要点|关键要点|总结如下|最终优化|先总结|简要总结|补充说明|额外分析|"
            r"让我检查|检查是否符合要求"
            r")"
        )
        meta_leak_pattern = re.compile(
            r"(?is)(?:"
            r"meets the requirements|no markdown|no labels|two sentences|one conclusion sentence|"
            r"optional second sentence|return only the final|email-ready tldr|"
            r"符合要求|不要 markdown|不要标签|两句话|一句结论|第二句"
            r")"
        )
        return bool(scaffold_pattern.match(text)) or bool(meta_leak_pattern.search(text))

    @staticmethod
    def _targets_chinese(lang: str) -> bool:
        normalized = (lang or "").strip().lower()
        return "chinese" in normalized or "中文" in normalized or "简体" in normalized

    @staticmethod
    def _sentence_needs_chinese_repair(text: str) -> bool:
        cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
        latin_words = len(re.findall(r"[A-Za-z]{2,}", text))
        return cjk_chars == 0 and latin_words >= 3

    @staticmethod
    def _strip_surrounding_quotes(text: str) -> str:
        return text.strip().strip("\"'“”‘’").strip()

    @staticmethod
    def _prefer_last_alternative_segment(text: str) -> str:
        alternative_patterns = [
            r"(?is)\b(?:or\s+shorter|shorter(?:\s+version)?|or\s+more\s+concisely(?:\s+focusing\s+on\s+the\s+conclusion)?)\s*[:：-]\s*(.+)",
            r"(?is)(?:更简洁地说|更简短地说|更简洁版本)\s*[:：-]\s*(.+)",
        ]
        for pattern in alternative_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                text = matches[-1].group(1).strip()
        return text

    @staticmethod
    def _has_suspicious_numeric_tail(text: str) -> bool:
        return bool(re.search(r"(?:\b(?:and|or)\s+|[与和及]\s*)?\d+\.$", text.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _is_failure_tldr_message(text: str) -> bool:
        return text.strip().startswith("Failed to generate TLDR.")

    def _needs_tldr_repair(self, cleaned_tldr: str, lang: str) -> bool:
        text = (cleaned_tldr or "").strip()
        if not text:
            return True
        if self._is_failure_tldr_message(text):
            return False

        sentences = self._split_tldr_sentences(text) or [text]
        if any(self._is_scaffold_sentence(sentence) for sentence in sentences):
            return True
        if self._has_suspicious_numeric_tail(text):
            return True

        if self._targets_chinese(lang):
            return any(self._sentence_needs_chinese_repair(sentence) for sentence in sentences)
        return False

    def _fallback_tldr_from_source(self) -> str:
        source = self.abstract or self.full_text or self.title or ""
        source = re.sub(r"\s+", " ", source).strip()
        if not source:
            return ""

        sentences = []
        for sentence in self._split_tldr_sentences(source):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
            if len(sentences) == 2:
                break

        if sentences:
            return self._join_tldr_sentences(sentences)
        return source[:280].strip()

    def _repair_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict, raw_tldr: str, cleaned_tldr: str) -> str:
        lang = llm_params.get('language', 'English')
        if self._targets_chinese(lang):
            prompt = (
                "上一个TLDR草稿不合格。请基于下面论文信息重新写成适合邮件直接展示的TLDR。"
                "只输出最终TLDR正文，不要解释、不要分析、不要项目符号、不要Markdown、不要标签。"
                "最多两句，优先一句结论，第二句仅在必须保留关键方法或机制时使用。"
                "请使用简体中文，除模型、方法、数据集或基准的正式名称外尽量不要保留英文。"
                "所有表述必须以论文信息为依据，不要补充 affiliations 或无依据事实。"
            )
        else:
            prompt = (
                f"The previous TLDR draft is invalid. Rewrite it as an email-ready TLDR in {lang}. "
                "Return only the final TLDR text with no reasoning, bullets, markdown, or labels. "
                "Use at most two sentences. Prefer one conclusion sentence, and use a second sentence only if needed to preserve the key method or mechanism. "
                "Keep every claim grounded in the paper information below, and do not add affiliations or unsupported facts. "
            )

        draft = (cleaned_tldr or raw_tldr or "").strip()
        if draft:
            if self._targets_chinese(lang):
                prompt += f"\n\n待修正草稿：\n{draft}\n\n"
            else:
                prompt += f"\n\nDraft to fix:\n{draft}\n\n"

        prompt += self._build_tldr_context()
        prompt = self._truncate_prompt(prompt, 4000)

        system_content = (
            "You repair malformed scientific-paper TLDR drafts. "
            f"Answer in {lang}. Return only the final TLDR text: at most two sentences, "
            "with one conclusion sentence preferred and an optional second sentence only for the key method or mechanism. "
            "Do not include reasoning, analysis, bullets, markdown, prefixes, or labels."
        )
        if self._targets_chinese(lang):
            system_content = (
                "你负责修复不合格的论文TLDR草稿。"
                "只输出最终TLDR正文，使用简体中文；最多两句，优先一句结论，第二句仅在必须保留关键方法或机制时使用。"
                "不要输出推理、分析、项目符号、Markdown、前缀或标签。"
            )

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        return response.choices[0].message.content

    def _cleanup_tldr(self, raw_tldr: str) -> str:
        text = (raw_tldr or "").strip()
        if not text:
            return text

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
        text = re.sub(r"[*_`#>✓✔✗✘]", "", text)
        text = self._extract_tldr_segment(text)
        text = self._prefer_last_alternative_segment(text)
        text = re.sub(r"(?is)\bSentence\s*1(?:\s*\([^)]*\))?\s*[:：-]\s*", "", text)
        text = re.sub(r"(?is)\bSentence\s*2(?:\s*\([^)]*\))?\s*[:：-]\s*", " ", text)
        text = self._strip_surrounding_quotes(text)

        cleaned_lines: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^(?:[-*•]+|\d+[.)]|[A-Za-z][.)])\s*", "", line)
            line = self._strip_tldr_prefix(line)
            line = self._strip_surrounding_quotes(line)
            if line:
                cleaned_lines.append(line)

        text = " ".join(cleaned_lines)
        text = re.sub(r"\s+", " ", text).strip()

        candidate_sentences = []
        for sentence in self._split_tldr_sentences(text):
            sentence = self._strip_tldr_prefix(sentence)
            sentence = self._strip_surrounding_quotes(sentence)
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
            return self._join_tldr_sentences(sentences)
        return text

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get('language', 'English')
        if self._targets_chinese(lang):
            prompt = (
                "请根据下面论文信息写一段适合邮件直接展示的TLDR。"
                "只输出最终TLDR正文，不要解释、不要分析、不要项目符号、不要Markdown、不要标签。"
                "最多两句，优先一句结论，第二句仅在必须保留关键方法或机制时使用。"
                "请使用简体中文，除模型、方法、数据集或基准的正式名称外尽量不要保留英文。\n\n"
            )
        else:
            prompt = (
                f"Given the following information of a paper, write an email-ready TLDR in {lang}. "
                "Return only the final TLDR text with no reasoning, bullets, markdown, or labels. "
                "Prefer a single conclusion sentence. Use a second sentence only when needed to preserve the key method or mechanism.\n\n"
            )
        prompt += self._build_tldr_context()

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        prompt = self._truncate_prompt(prompt, 4000)

        system_content = (
            "You are an assistant who summarizes scientific papers for email readers. "
            f"Answer in {lang}. Return only the final TLDR text: at most two sentences, "
            "with one conclusion sentence preferred and an optional second sentence only for the key method or mechanism. "
            "Do not include reasoning, analysis, bullets, markdown, prefixes, or labels."
        )
        if self._targets_chinese(lang):
            system_content = (
                "你负责为科研论文生成可直接发送邮件的TLDR。"
                "只输出最终TLDR正文，使用简体中文；最多两句，优先一句结论，第二句仅在必须保留关键方法或机制时使用。"
                "不要输出推理、分析、项目符号、Markdown、前缀或标签。"
            )

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            lang = llm_params.get('language', 'English')
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            if self._is_failure_tldr_message(tldr):
                self.tldr = tldr
                return tldr

            cleaned_tldr = self._cleanup_tldr(tldr)
            if self._needs_tldr_repair(cleaned_tldr, lang):
                logger.warning(f"TLDR output needs repair for {self.url}")
                repaired_tldr = self._repair_tldr_with_llm(openai_client, llm_params, tldr, cleaned_tldr)
                repaired_cleaned_tldr = self._cleanup_tldr(repaired_tldr)
                if repaired_cleaned_tldr and not self._needs_tldr_repair(repaired_cleaned_tldr, lang):
                    cleaned_tldr = repaired_cleaned_tldr
                else:
                    logger.warning(f"Falling back to source excerpt after invalid TLDR repair for {self.url}")
                    cleaned_tldr = self._fallback_tldr_from_source()

            if not cleaned_tldr:
                logger.warning(f"Falling back to source excerpt after empty TLDR cleanup for {self.url}")
                cleaned_tldr = self._fallback_tldr_from_source()

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
