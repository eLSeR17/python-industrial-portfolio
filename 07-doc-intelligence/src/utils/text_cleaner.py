"""Text cleaning utilities for OCR artifact removal and normalization.

Provides functions to clean raw text extracted from PDFs, DOCX files,
and OCR output so downstream NLP produces more accurate results.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Remove OCR artifacts, normalize whitespace, and strip noise.

    Args:
        text: Raw text potentially containing OCR errors.

    Returns:
        Cleaned text ready for NLP processing.
    """
    if not text:
        return ""
    text = _fix_common_ocr_substitutions(text)
    text = normalize_whitespace(text)
    text = _fix_encoding_artifacts(text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and newlines into single delimiters.

    Args:
        text: Text with inconsistent whitespace.

    Returns:
        Text with normalized whitespace.
    """
    if not text:
        return ""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    return text


def extract_tables_as_text(text: str) -> str:
    """Detect table-like patterns and format them as structured text.

    Identifies pipe-delimited rows and converts them into a
    consistent markdown-style table format.

    Args:
        text: Raw text that may contain table structures.

    Returns:
        Text with tables formatted consistently.
    """
    if not text:
        return ""
    lines = text.split("\n")
    result_lines: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_pipe_row = "|" in stripped and stripped.count("|") >= 2
        is_dash_row = bool(re.match(r"^[|\s\-:]+$", stripped))

        if is_pipe_row or is_dash_row:
            if not in_table:
                in_table = True
            if is_dash_row and not any(c.isalpha() for c in stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            formatted = "| " + " | ".join(cells) + " |"
            result_lines.append(formatted)
        else:
            if in_table:
                in_table = False
            result_lines.append(line)

    return "\n".join(result_lines)


def remove_headers_footers(text: str, page_markers: list[str] | None = None) -> str:
    """Strip common header and footer patterns from document text.

    Args:
        text: Full document text.
        page_markers: Optional list of regex patterns identifying page
            numbers or footer lines to remove.

    Returns:
        Text with headers and footers removed.
    """
    if not text:
        return ""
    if page_markers is None:
        page_markers = [
            r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$",
            r"^\s*\d+\s*/\s*\d+\s*$",
            r"^\s*Confidential\s*$",
            r"^\s*©\s*\d{4}.*$",
            r"^\s*Page\s+\d+\s*$",
        ]
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in page_markers):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _fix_common_ocr_substitutions(text: str) -> str:
    """Fix common character substitutions made by OCR engines.

    Only applies substitutions inside words where context makes the
    substitution unambiguous.

    Args:
        text: Text with potential OCR character errors.

    Returns:
        Text with corrected characters.
    """
    replacements = [
        (r"(?<=\w)l(?=\w)", "1"),
        (r"(?<=\w)O(?=\w)", "0"),
        (r"(?<=\w)I(?=\w)", "1"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def _fix_encoding_artifacts(text: str) -> str:
    """Normalize unicode and strip invisible or problematic characters.

    Args:
        text: Text with potential encoding issues.

    Returns:
        Normalized unicode text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text
