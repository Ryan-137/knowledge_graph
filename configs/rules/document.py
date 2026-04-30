from __future__ import annotations

# 这些 block tag 决定 HTML 正文抽取时优先保留哪些结构块。
BLOCK_TAGS = (
    "p",
    "li",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "article",
    "section",
)

# 优先尝试的正文容器选择器。
CONTENT_SELECTORS = (
    "article",
    "main",
    "[role='main']",
    ".mw-parser-output",
    ".article-body",
    ".entry-content",
    ".post-content",
    ".content",
    ".main-content",
    "#content",
    "#main-content",
    "#mw-content-text",
)

# 明确应当去掉的噪声节点。
NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "footer",
    "header nav",
    "aside",
    ".navbox",
    ".metadata",
    ".reference",
    ".references",
    ".reflist",
    ".infobox",
    ".toc",
    ".mw-editsection",
    ".sidebar",
    ".advert",
    ".ads",
    ".cookie-banner",
)

# 标签级直接删除的噪声类型。
NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "form",
    "button",
)

# 某些标签即便命中关键词，也要慎删，避免误删正文。
PROTECTED_CONTENT_TAGS = (
    "article",
    "main",
    "section",
)

# 命中后通常视为导航、营销、版权、评论等非正文模块。
NOISE_KEYWORDS = (
    "nav",
    "navbar",
    "menu",
    "breadcrumb",
    "footer",
    "sidebar",
    "advert",
    "cookie",
    "subscribe",
    "newsletter",
    "related",
    "comment",
    "share",
    "social",
    "promo",
    "banner",
)

# 出现这些尾部章节标题时，后续内容通常不再属于正文。
TAIL_SECTION_TITLES = {
    "references",
    "external links",
    "see also",
    "notes",
    "citations",
    "bibliography",
    "further reading",
    "sources",
    "works cited",
    "参考文献",
    "外部链接",
    "延伸阅读",
    "注释",
}

