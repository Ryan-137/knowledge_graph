"""正文抽取器集合。"""

from .html import extract_html_text
from .pdf import extract_pdf_text

__all__ = [
    "extract_html_text",
    "extract_pdf_text",
]
