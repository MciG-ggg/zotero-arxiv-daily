"""Tests for zotero_arxiv_daily.construct_email: render_email, get_stars, get_block_html."""

from zotero_arxiv_daily.construct_email import render_email, get_stars, get_block_html, get_empty_html
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Latest papers" in html
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_render_email_uses_fallback_when_tldr_is_empty():
    paper = make_sample_paper(
        score=7.0,
        tldr="",
        abstract="This paper improves manipulation reliability. It adds view-consistent planning.",
    )
    html = render_email([paper])
    assert "This paper improves manipulation reliability. It adds view-consistent planning." in html


def test_render_email_strips_html_like_tags_from_tldr():
    paper = make_sample_paper(score=7.0, tldr="<plan>Safe summary.</plan>")
    html = render_email([paper])
    assert "Safe summary." in html
    assert "<plan>" not in html


def test_render_email_recleans_stale_meta_tldr():
    paper = make_sample_paper(
        score=7.0,
        tldr="用户要求我根据论文信息重新写一个TLDR摘要。实验表明，该方法在长期任务成功率和鲁棒性方面显著提升，泛化能力强。",
    )
    html = render_email([paper])
    assert "实验表明，该方法在长期任务成功率和鲁棒性方面显著提升，泛化能力强。" in html
    assert "用户要求我" not in html


def test_render_email_falls_back_when_tldr_is_only_meta_scaffold():
    paper = make_sample_paper(
        score=7.0,
        tldr="这是一篇关于机器人后训练的论文。一句话概括即可。",
        abstract="Hi-WM提出了一种可扩展的机器人后训练框架。实验表明，该方法显著提升真实任务成功率。",
    )
    html = render_email([paper])
    assert "Hi-WM提出了一种可扩展的机器人后训练框架。实验表明，该方法显著提升真实任务成功率。" in html
    assert "一句话概括即可" not in html


def test_render_email_falls_back_when_tldr_is_failure_message():
    paper = make_sample_paper(
        score=7.0,
        tldr="Failed to generate TLDR. Neither full text nor abstract is provided",
        abstract="This paper improves manipulation reliability. It adds view-consistent planning.",
    )
    html = render_email([paper])
    assert "This paper improves manipulation reliability. It adds view-consistent planning." in html
    assert "Failed to generate TLDR." not in html


def test_render_email_separates_latest_and_classics():
    latest = [make_sample_paper(title="Latest Paper", score=7.0, tldr="latest")]
    classic = [
        make_sample_paper(
            title="Classic Paper",
            score=8.2,
            tldr="classic",
            published_year=2018,
            citation_count=999,
        )
    ]
    html = render_email(latest, classic)
    assert "Latest papers" in html
    assert "Classic papers" in html
    assert "Latest Paper" in html
    assert "Classic Paper" in html
    assert "Year:</strong> 2018" in html
    assert "Citations:</strong> 999" in html


def test_render_email_skips_empty_classic_section():
    latest = [make_sample_paper(title="Latest Only", score=7.0, tldr="latest")]
    html = render_email(latest, [])
    assert "Latest papers" in html
    assert "Classic papers" not in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "Summary", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html
