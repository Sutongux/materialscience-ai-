"""
End-to-end pipeline to convert raw PDFs to markdown and produce cleaned outputs.
Steps:
1) Parse all PDFs in data/raw/ to markdown in data/parsed/ (via parser.py).
2) Clean markdown and emit cleaned markdown + metadata JSON in data/cleaned/ (via preprocessing.py).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Any

from parser import convert_all_pdfs
from preprocessing import TextPreprocessor, EXTRACTED_DATA_DIR, CLEANED_DATA_DIR


def _configure_logging() -> None:
    """Configure a basic logger for pipeline status."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_pipeline() -> Dict[str, Any]:
    """Run parsing then preprocessing; returns preprocessing stats."""
    logging.info("Starting PDF parsing stage...")
    convert_all_pdfs()
    logging.info("Parsing stage complete.")

    logging.info("Starting preprocessing stage...")
    preprocessor = TextPreprocessor(
        input_dir=EXTRACTED_DATA_DIR,
        output_dir=CLEANED_DATA_DIR,
    )
    stats = preprocessor.process_all()
    logging.info(
        "Preprocessing complete. Docs: %s Sections: %s Characters: %s",
        stats.get("total_docs"),
        stats.get("total_sections"),
        stats.get("cleaned_chars"),
    )
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF-to-cleaned-markdown pipeline.")
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip the parsing stage and only run preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    if args.skip_parse:
        logging.info("Skipping parsing stage; running preprocessing only.")
        preprocessor = TextPreprocessor(
            input_dir=EXTRACTED_DATA_DIR,
            output_dir=CLEANED_DATA_DIR,
        )
        preprocessor.process_all()
        return

    run_pipeline()


if __name__ == "__main__":
    main()
