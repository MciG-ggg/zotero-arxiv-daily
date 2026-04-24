import math

import re
from html import escape

from .protocol import Paper


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em;
      line-height: 1;
      display: inline-flex;
      align-items: center;
    }
    .half-star {
      display: inline-block;
      width: 0.5em;
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


def get_empty_html():
    return """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str = None,
    details: str | None = None,
):
    detail_html = (
        f"""
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            {details}
        </td>
    </tr>
"""
        if details
        else ""
    )
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>
    {detail_html}
    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=escape(str(title)),
        authors=escape(str(authors)),
        rate=escape(str(rate)),
        tldr=escape(str(tldr)),
        pdf_url=escape(str(pdf_url), quote=True),
        affiliations=escape(str(affiliations)),
        detail_html=detail_html,
    )


def get_stars(score: float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high - low) / 10
        star_num = math.ceil((score - low) / interval)
        full_star_num = int(star_num / 2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">' + full_star * full_star_num + half_star * half_star_num + '</div>'


def _truncate_authors(paper: Paper) -> str:
    author_list = [author for author in paper.authors]
    if len(author_list) <= 5:
        return ', '.join(author_list)
    return ', '.join(author_list[:3] + ['...'] + author_list[-2:])


def _truncate_affiliations(paper: Paper) -> str:
    if paper.affiliations is None:
        return 'Unknown Affiliation'
    affiliations = paper.affiliations[:5]
    text = ', '.join(affiliations)
    if len(paper.affiliations) > 5:
        text += ', ...'
    return text


def _safe_tldr(paper: Paper) -> str:
    tldr = re.sub(r"(?is)<[^>]+>", " ", paper.tldr or "")
    tldr = paper._cleanup_tldr(tldr)
    tldr = re.sub(r"\s+", " ", tldr).strip()
    if tldr and not paper._is_failure_tldr_message(tldr) and not paper._needs_tldr_repair(tldr, "English"):
        return tldr

    fallback = paper._fallback_tldr_from_source()
    if fallback:
        return fallback

    return paper.title


def _render_section(title: str, subtitle: str, papers: list[Paper]) -> str:
    parts = []
    for paper in papers:
        details = []
        if paper.published_year is not None:
            details.append(f"<strong>Year:</strong> {paper.published_year}")
        if paper.citation_count is not None:
            details.append(f"<strong>Citations:</strong> {paper.citation_count}")
        parts.append(
            get_block_html(
                paper.title,
                _truncate_authors(paper),
                round(paper.score, 1) if paper.score is not None else 'Unknown',
                _safe_tldr(paper),
                paper.pdf_url or paper.url,
                _truncate_affiliations(paper),
                " &nbsp;|&nbsp; ".join(details) if details else None,
            )
        )

    return f"""
<div style="margin: 24px 0 12px 0;">
  <h2 style="margin-bottom: 6px;">{title}</h2>
  <p style="margin-top: 0; color: #666;">{subtitle}</p>
</div>
<br>{'</br><br>'.join(parts)}</br>
"""


def render_email(latest_papers: list[Paper], classic_papers: list[Paper] | None = None) -> str:
    classic_papers = classic_papers or []
    if not latest_papers and not classic_papers:
        return framework.replace('__CONTENT__', get_empty_html())

    sections = []
    if latest_papers:
        sections.append(
            _render_section(
                "Latest papers",
                "Fresh papers retrieved from your configured sources and reranked against Zotero.",
                latest_papers,
            )
        )
    if classic_papers:
        sections.append(
            _render_section(
                "Classic papers",
                "High-citation embodied-AI classics filtered by explicit topic rules and reranked against Zotero.",
                classic_papers,
            )
        )
    return framework.replace('__CONTENT__', "".join(sections))
