"""
OCR-based PDF parser that converts raw PDFs into markdown documents.
The script prioritises native text extraction and falls back to pytesseract
for image-only pages so that the downstream pipeline can operate without the
vision model API.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # type: ignore
import pdfplumber  # type: ignore
import pytesseract  # type: ignore
from PIL import Image  # type: ignore
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

RAW_PDF_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
PROCESSED_PDF_DIR = Path(os.getenv("PROCESSED_PDF_DIR", DATA_DIR / "parsed"))
LOG_DIR = Path(os.getenv("LOG_DIR", LOGS_DIR / "pdf_parser"))
PAGE_DELIMITER = os.getenv("PAGE_DELIMITER", "\n\n---\n\n")
# OCR_STRATEGY:
#   "auto"  -> only OCR pages with no native text (default)
#   "force" -> always combine native text + OCR
#   "off"   -> never run OCR
OCR_STRATEGY = os.getenv("OCR_STRATEGY", "auto").lower()

for directory in (RAW_PDF_DIR, PROCESSED_PDF_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("pdf_parser_ocr")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / "pdf_parser.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------
def _configure_tesseract() -> Optional[str]:
    """Locate a usable Tesseract executable."""
    candidates = [
        os.getenv("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        resolved = Path(candidate).expanduser()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            pytesseract.pytesseract.tesseract_cmd = str(resolved)
            return str(resolved)
    return None


TESSERACT_PATH = _configure_tesseract()
TESSERACT_AVAILABLE = TESSERACT_PATH is not None

if not TESSERACT_AVAILABLE:
    LOGGER.warning(
        "Tesseract executable not detected; OCR-only PDFs may produce empty output."
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters from extracted text."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    return cleaned.strip()


def _render_page_to_image(page: fitz.Page, zoom: float = 2.0) -> Optional[Image.Image]:
    """Render a PyMuPDF page to a PIL image for OCR."""
    try:
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to render page %s for OCR: %s", page.number + 1, exc)
        return None


def _ocr_page(page: fitz.Page) -> str:
    """Run pytesseract on the rendered page image."""
    if not TESSERACT_AVAILABLE:
        return ""

    image = _render_page_to_image(page)
    if image is None:
        return ""

    try:
        text = pytesseract.image_to_string(image)
        return _clean_text(text)
    except pytesseract.TesseractError as exc:  
        LOGGER.warning("Tesseract failed on page %s: %s", page.number + 1, exc)
        return ""


def _extract_tables(pdf_path: Path) -> Dict[int, List[str]]:
    """Extract tables from each page and convert them to markdown."""
    tables: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                extracted_tables = page.extract_tables()
                markdown_tables = []
                for table in extracted_tables or []:
                    md_table = _table_to_markdown(table)
                    if md_table:
                        markdown_tables.append(md_table)
                if markdown_tables:
                    tables[page_index] = markdown_tables
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Table extraction failed for %s: %s", pdf_path.name, exc)
    return tables


def _table_to_markdown(table: List[List[Optional[str]]]) -> Optional[str]:
    """Convert a raw table (list of rows) to a markdown representation."""
    if not table:
        return None

    normalised_rows: List[List[str]] = []
    for row in table:
        cells = [(cell or "").strip() for cell in row]
        normalised_rows.append(cells)

    header = normalised_rows[0]
    if not header:
        return None

    column_count = len(header)
    body_rows = normalised_rows[1:] or [["" for _ in range(column_count)]]

    header_cells = header + [""] * (column_count - len(header))
    header_line = "| " + " | ".join(header_cells[:column_count]) + " |"
    separator_line = "| " + " | ".join("---" for _ in range(column_count)) + " |"

    body_lines = []
    for row in body_rows:
        padded = row + [""] * (column_count - len(row))
        body_lines.append("| " + " | ".join(padded[:column_count]) + " |")

    lines = [header_line, separator_line, *body_lines]
    return "\n".join(lines)


def _extract_page_text(page: fitz.Page) -> str:
    """Extract combined text for a page using native extraction and OCR."""
    native_text = _clean_text(page.get_text("text", sort=True) or "")

    strategy = OCR_STRATEGY
    if strategy not in {"auto", "force", "off"}:
        strategy = "auto"

    if strategy == "off":
        return native_text

    if strategy == "auto":
        # Only use OCR when native text is absent (ignores images otherwise)
        return native_text or _ocr_page(page)

    # "force" behaviour: combine both native and OCR outputs
    ocr_text = _ocr_page(page)
    if native_text and ocr_text and ocr_text not in native_text:
        return f"{native_text}\n\n{ocr_text}".strip()
    return (native_text or ocr_text).strip()


def _assemble_markdown(
    pdf_path: Path,
    tables_by_page: Dict[int, List[str]],
) -> str:
    """Build the markdown document for the PDF."""
    sections: List[str] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            page_text = _extract_page_text(page)
            page_tables = tables_by_page.get(index, [])

            section_lines: List[str] = [f"## Page {index}"]
            if page_text:
                section_lines.extend(["", page_text])

            for table_index, table_md in enumerate(page_tables, start=1):
                section_lines.extend(
                    ["", f"### Table {table_index}", "", table_md]
                )

            section = "\n".join(line for line in section_lines if line is not None).strip()
            if section:
                sections.append(section)

    markdown = PAGE_DELIMITER.join(sections).strip()
    if not markdown:
        markdown = "# Empty Document\n\nNo extractable text was found."
    return markdown


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def convert_pdf_to_md(pdf_path: str, out_path: Optional[str] = None) -> Path:
    """Convert a PDF file to markdown and write it to disk."""
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if out_path:
        out_path_obj = Path(out_path)
        if out_path_obj.suffix.lower() == ".md":
            output_path = out_path_obj
            output_dir = output_path.parent
        else:
            output_dir = out_path_obj
            output_path = output_dir / f"{pdf_path_obj.stem}.md"
    else:
        output_dir = PROCESSED_PDF_DIR
        output_path = output_dir / f"{pdf_path_obj.stem}.md"

    output_dir.mkdir(parents=True, exist_ok=True)

    tables_by_page = _extract_tables(pdf_path_obj)
    markdown = _assemble_markdown(pdf_path_obj, tables_by_page)

    output_path.write_text(markdown, encoding="utf-8")
    LOGGER.info("Saved markdown to %s", output_path)
    return output_path


def convert_all_pdfs() -> None:
    """Process every PDF in the raw directory."""
    pdf_files = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        LOGGER.info("No PDFs found in %s", RAW_PDF_DIR)
        return

    for pdf_file in pdf_files:
        try:
            convert_pdf_to_md(str(pdf_file))
        except Exception as exc:
            LOGGER.error("Failed to convert %s: %s", pdf_file.name, exc)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw PDF files to markdown using OCR."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Optional path to a single PDF file. If omitted, process all PDFs in RAW_PDF_DIR.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        help="Optional output file or directory. Defaults to PROCESSED_PDF_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pdf_path:
        convert_pdf_to_md(args.pdf_path, args.out_path)
    else:
        convert_all_pdfs()


if __name__ == "__main__":
    main()
