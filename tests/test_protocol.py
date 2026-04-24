"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import re
from types import SimpleNamespace

import pytest

from zotero_arxiv_daily.protocol import Paper
from tests.canned_responses import (
    CLEAN_ONE_LINE_TLDR_RESPONSE,
    COMBINED_TLDR_RESPONSE,
    DECIMAL_METRIC_TLDR_RESPONSE,
    DECIMAL_PERCENT_TLDR_RESPONSE,
    EMPTY_TLDR_RESPONSE,
    ENGLISH_ONLY_TLDR_RESPONSE,
    HTML_TAGGED_TLDR_RESPONSE,
    IDEATION_TLDR_RESPONSE,
    MARKDOWN_TLDR_RESPONSE,
    MIXED_TLDR_RESPONSE_CN_WITH_EN_META,
    NOISY_TLDR_RESPONSE_CN,
    NOISY_TLDR_RESPONSE_EN,
    OR_SHORTER_TLDR_RESPONSE,
    REPAIRED_TLDR_RESPONSE_CN,
    SENTENCE_LABEL_TLDR_RESPONSE,
    THREE_SENTENCE_TLDR_RESPONSE,
    make_sample_paper,
    make_stub_openai_client,
    make_stub_openai_client_sequence,
)


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def _sentence_count(text: str) -> int:
    return len(Paper._split_tldr_sentences(text))


def test_tldr_keeps_only_final_summary_for_mixed_chinese_output(llm_params):
    client = make_stub_openai_client(NOISY_TLDR_RESPONSE_CN)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert "该方法显著提升了多模态检索的准确率" in result
    assert "它通过跨模态对齐保留关键机制" in result
    assert "让我先理解" not in result
    assert "分析问题背景" not in result
    assert "TLDR" not in result
    assert "摘要" not in result
    assert _sentence_count(result) <= 2


def test_tldr_keeps_only_final_summary_for_mixed_english_output(llm_params):
    client = make_stub_openai_client(NOISY_TLDR_RESPONSE_EN)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert "The model improves retrieval accuracy on long-context documents." in result
    assert "It does so via a hierarchical memory mechanism." in result
    assert "Let me break this down first" not in result
    assert "Key idea" not in result
    assert "TL;DR" not in result
    assert _sentence_count(result) <= 2


def test_tldr_strips_markdown_wrappers_and_prefixes(llm_params):
    client = make_stub_openai_client(MARKDOWN_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result.startswith("The pipeline cuts hallucinations in agent traces.")
    assert "It adds verifier-gated checkpoints." in result
    assert "**" not in result
    assert "TLDR" not in result
    assert "- " not in result


def test_tldr_caps_summary_at_two_sentences(llm_params):
    client = make_stub_openai_client(THREE_SENTENCE_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result.startswith("The method improves theorem proving accuracy on competition benchmarks.")
    assert "It uses retrieval-augmented search to preserve key premises." in result
    assert "It also reduces inference costs through speculative decoding." not in result
    assert _sentence_count(result) == 2


def test_tldr_preserves_clean_one_line_summary(llm_params):
    client = make_stub_openai_client(CLEAN_ONE_LINE_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result == "A compact summary stays intact after cleanup."
    assert paper.tldr == result


def test_tldr_drops_english_requirement_leak_after_chinese_summary(llm_params):
    client = make_stub_openai_client(MIXED_TLDR_RESPONSE_CN_WITH_EN_META)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == "本研究针对无人机目标导航问题，提出了一种名为 AmelPred 的自预测表征模型，其随机版本 AmelPredSto 与演员-评论家强化学习结合后显著提升了样本效率和导航性能。"
    assert "This captures" not in result
    assert "requirements" not in result


def test_tldr_repairs_english_only_output_into_chinese(llm_params):
    client = make_stub_openai_client_sequence(
        ENGLISH_ONLY_TLDR_RESPONSE,
        REPAIRED_TLDR_RESPONSE_CN,
    )
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == REPAIRED_TLDR_RESPONSE_CN
    assert "The method improves sample efficiency" not in result
    assert _sentence_count(result) == 2


def test_tldr_falls_back_to_source_excerpt_after_empty_generation(llm_params):
    client = make_stub_openai_client_sequence(
        EMPTY_TLDR_RESPONSE,
        EMPTY_TLDR_RESPONSE,
    )
    paper = make_sample_paper(
        abstract=(
            "This paper improves robot planning under sparse rewards. "
            "It aligns intermediate subgoals with visual affordances. "
            "A third sentence should not appear."
        )
    )

    result = paper.generate_tldr(client, llm_params)

    assert result == (
        "This paper improves robot planning under sparse rewards. "
        "It aligns intermediate subgoals with visual affordances."
    )
    assert "A third sentence should not appear." not in result
    assert _sentence_count(result) == 2


def test_tldr_prefers_last_shorter_alternative(llm_params):
    client = make_stub_openai_client(OR_SHORTER_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == (
        "LoHo-Manip通过任务管理VLM预测剩余子任务和视觉轨迹，使执行器VLA能够通过跟随轨迹完成局部控制，"
        "实现了长时序操作任务的稳健执行和错误恢复，在仿真和真实Franka机器人上验证了其有效性。"
    )
    assert '"' not in result
    assert "Or shorter" not in result


def test_tldr_strips_sentence_labels_and_keeps_two_sentences(llm_params):
    client = make_stub_openai_client(SENTENCE_LABEL_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == (
        "ResVLA通过“意图精炼”范式，利用频谱分析将控制解耦为低频意图锚点和高频残差，实现了比标准生成式基线更快的收敛速度和更强的扰动鲁棒性。"
        "该方法通过残差扩散桥接技术在预测的全局意图上锚定生成过程，使模型能够专注于精炼局部动力学而非从噪声开始重建完整动作。"
    )
    assert "Sentence 1" not in result
    assert "Sentence 2" not in result
    assert _sentence_count(result) == 2


def test_tldr_strips_combined_prefix_and_meta_tail(llm_params):
    client = make_stub_openai_client(COMBINED_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == (
        "本研究提出了一种名为AmelPred的自预测表征模型，其随机版本AmelPredSto在与actor-critic强化学习算法结合时，"
        "可显著提升无人机目标导航任务的样本效率。"
    )
    assert "Combined" not in result
    assert "This is one sentence" not in result


def test_tldr_preserves_decimal_metrics_without_sentence_truncation(llm_params):
    client = make_stub_openai_client(DECIMAL_METRIC_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == DECIMAL_METRIC_TLDR_RESPONSE
    assert "2.79倍和2.31倍" in result
    assert _sentence_count(result) == 1


def test_tldr_preserves_decimal_percentages_without_truncation(llm_params):
    client = make_stub_openai_client(DECIMAL_PERCENT_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == DECIMAL_PERCENT_TLDR_RESPONSE
    assert "37.5%" in result
    assert _sentence_count(result) == 2


def test_tldr_strips_html_like_tags_from_model_output(llm_params):
    client = make_stub_openai_client(HTML_TAGGED_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == "该方法显著提升了操作成功率，并增强了跨视角鲁棒性。"
    assert "<plan>" not in result


def test_tldr_strips_chinese_ideation_scaffold(llm_params):
    client = make_stub_openai_client(IDEATION_TLDR_RESPONSE)
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == "研究提出了一种名为AmelPred的自预测表示方法，其随机版本AmelPredSto能显著提升强化学习在无人机目标导航任务中的样本效率。"
    assert "构思" not in result
    assert "进一步精炼表述" not in result


def test_tldr_uses_chinese_native_prompt_when_language_is_chinese(llm_params):
    captured_messages = []

    def create_chat_completion(**kwargs):
        captured_messages.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="中文TLDR。"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_chat_completion)
        )
    )
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == "中文TLDR。"
    assert len(captured_messages) == 1
    system_message = captured_messages[0][0]["content"]
    user_message = captured_messages[0][1]["content"]
    assert "简体中文" in system_message
    assert "简体中文" in user_message
    assert "Given the following information of a paper" not in user_message


def test_tldr_uses_chinese_native_repair_prompt_when_cleanup_detects_invalid_output(llm_params):
    captured_messages = []
    responses = iter([
        ENGLISH_ONLY_TLDR_RESPONSE,
        REPAIRED_TLDR_RESPONSE_CN,
    ])

    def create_chat_completion(**kwargs):
        captured_messages.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_chat_completion)
        )
    )
    paper = make_sample_paper()

    result = paper.generate_tldr(client, {**llm_params, "language": "Chinese"})

    assert result == REPAIRED_TLDR_RESPONSE_CN
    assert len(captured_messages) == 2
    repair_system_message = captured_messages[1][0]["content"]
    repair_user_message = captured_messages[1][1]["content"]
    assert "简体中文" in repair_system_message
    assert "待修正草稿" in repair_user_message


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None
