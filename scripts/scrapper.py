#!/usr/bin/env python3

"""
scrapper.py

Download PDF files from Google Patents and save them to the data folder.

Examples:

    # Single URL
    python scrapper.py "https://patents.google.com/patent/US1234567A/en"

    # Batch from file (one URL per line)
    python scrapper.py --urls-file patent_urls.txt

    # Batch with concurrency
    python scrapper.py --urls-file patent_urls.txt --workers 8

"""

import argparse
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup


# --- Config / constants -------------------------------------------------------

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 15  # seconds


# --- Core logic ---------------------------------------------------------------

def download_patent_pdf(patent_url: str, out_dir: Path, session: requests.Session | None = None) -> Path:
    """
    Given a Google Patents URL, find and download the PDF file.
    
    Returns the path to the saved PDF file.
    
    Raises:
        ValueError: if URL is invalid or PDF cannot be found/downloaded.
        requests.RequestException: for network-related errors.
    """
    logging.debug("Processing patent URL: %s", patent_url)
    _validate_patent_url(patent_url)

    html = _fetch_html(patent_url, session=session)
    pdf_url = _find_pdf_url(html, patent_url)

    if not pdf_url:
        raise ValueError(f"Could not find PDF download link for: {patent_url}")

    logging.info("Found PDF URL: %s", pdf_url)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = _derive_pdf_filename(patent_url)
    out_path = out_dir / filename

    # Download the PDF
    logging.info("Downloading PDF from %s", pdf_url)
    pdf_content = _download_pdf(pdf_url, session=session)

    logging.info("Saving PDF to %s", out_path)
    with open(out_path, "wb") as f:
        f.write(pdf_content)

    logging.info("Saved PDF to %s", out_path)
    return out_path


# --- Helpers ------------------------------------------------------------------

def _validate_patent_url(patent_url: str) -> None:
    """Basic sanity checks for the input URL."""
    parsed = urlparse(patent_url)
    if not (parsed.scheme and parsed.netloc):
        raise ValueError(f"Invalid URL: {patent_url}")
    if "patents.google.com" not in parsed.netloc:
        logging.warning("URL does not look like a Google Patents URL: %s", patent_url)


def _fetch_html(url: str, session: requests.Session | None = None) -> str:
    """Fetch page HTML."""
    http = session or requests
    resp = http.get(
        url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text


def _find_pdf_url(html: str, page_url: str) -> str | None:
    """
    Find the PDF download URL from the patent page HTML.
    
    Google Patents typically has PDF links in various formats:
    - Direct links to PDF files
    - Download buttons with PDF links
    - Links in the page metadata
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Method 1: Look for direct PDF links
    pdf_links = soup.find_all("a", href=re.compile(r"\.pdf", re.I))
    for link in pdf_links:
        href = link.get("href", "")
        if href:
            full_url = urljoin(page_url, href)
            if full_url.endswith(".pdf") or "pdf" in href.lower():
                return full_url
    
    # Method 2: Look for download buttons/links with "pdf" in text or class
    download_links = soup.find_all("a", string=re.compile(r"pdf|download", re.I))
    for link in download_links:
        href = link.get("href", "")
        if href:
            full_url = urljoin(page_url, href)
            if "pdf" in full_url.lower():
                return full_url
    
    # Method 3: Look for links with download attribute or specific classes
    download_buttons = soup.find_all("a", {"download": True}) + \
                      soup.find_all("a", class_=re.compile(r"download|pdf", re.I))
    for link in download_buttons:
        href = link.get("href", "")
        if href:
            full_url = urljoin(page_url, href)
            if "pdf" in full_url.lower() or full_url.endswith(".pdf"):
                return full_url
    
    # Method 4: Look in meta tags or data attributes
    meta_tags = soup.find_all("meta", property=re.compile(r"pdf|download", re.I))
    for meta in meta_tags:
        content = meta.get("content", "")
        if content and ("pdf" in content.lower() or content.endswith(".pdf")):
            return urljoin(page_url, content)
    
    # Method 5: Try to construct PDF URL from patent number
    # Google Patents PDFs are often at: /patent/{PATENT_NUMBER}/pdf
    parsed_url = urlparse(page_url)
    path_parts = [p for p in parsed_url.path.split("/") if p]
    if len(path_parts) >= 2 and path_parts[0] == "patent":
        patent_number = path_parts[1]
        pdf_url = f"{parsed_url.scheme}://{parsed_url.netloc}/patent/{patent_number}/pdf"
        return pdf_url
    
    # Method 6: Search for any link containing "pdf" in the href
    all_links = soup.find_all("a", href=True)
    for link in all_links:
        href = link.get("href", "")
        if "pdf" in href.lower():
            full_url = urljoin(page_url, href)
            return full_url
    
    return None


def _download_pdf(pdf_url: str, session: requests.Session | None = None) -> bytes:
    """Download PDF file content."""
    http = session or requests
    resp = http.get(
        pdf_url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()
    
    # Verify it's actually a PDF
    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
        # Check first few bytes for PDF magic number
        content = resp.content[:4]
        if content != b"%PDF":
            logging.warning("Downloaded content may not be a PDF (Content-Type: %s)", content_type)
    
    return resp.content


def _derive_pdf_filename(patent_url: str) -> str:
    """
    Derive a reasonable filename for the PDF file based on patent URL.
    """
    parsed_patent = urlparse(patent_url)
    path_parts = [p for p in parsed_patent.path.split("/") if p]
    
    if len(path_parts) >= 2 and path_parts[0] == "patent":
        slug = path_parts[1]
    else:
        slug = Path(parsed_patent.path).name or "patent"
    
    # Clean up the slug to be filesystem-safe
    slug = slug.replace("/", "_").replace("\\", "_")
    return f"{slug}.pdf"


def _read_urls_from_file(path: Path) -> list[str]:
    """
    Read a file containing one URL per line.
    Ignore empty lines and lines starting with '#'.
    """
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    return urls


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDF files from Google Patents and save to data folder."
    )

    parser.add_argument(
        "url",
        nargs="?",
        help="Single Google Patents URL "
             "(e.g., https://patents.google.com/patent/US1234567A/en)",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Path to a text file with one Google Patents URL per line.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save PDF files (default: ./data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent downloads when using --urls-file "
             "(default: 1 = sequential).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Mode selection: single URL or batch
    if args.urls_file:
        _run_batch_mode(args)
    elif args.url:
        _run_single_mode(args)
    else:
        # No URL or urls-file provided
        raise SystemExit("Error: provide either a URL or --urls-file")


def _run_single_mode(args: argparse.Namespace) -> None:
    """Handle the single-URL case."""
    try:
        out_path = download_patent_pdf(args.url, args.out_dir)
        print(f"Saved PDF to: {out_path}")
    except Exception as e:
        logging.error("Failed to download PDF: %s", e)
        raise SystemExit(1)


def _run_batch_mode(args: argparse.Namespace) -> None:
    """Handle the batch (file of URLs) case."""
    urls = _read_urls_from_file(args.urls_file)
    if not urls:
        raise SystemExit(f"No URLs found in {args.urls_file}")

    logging.info("Loaded %d URLs from %s", len(urls), args.urls_file)

    if args.workers <= 1:
        _batch_sequential(urls, args.out_dir)
    else:
        _batch_concurrent(urls, args.out_dir, workers=args.workers)


def _batch_sequential(urls: list[str], out_dir: Path) -> None:
    """Sequential batch download with a shared Session."""
    successes = 0
    failures = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

        for url in urls:
            try:
                logging.info("Processing %s", url)
                download_patent_pdf(url, out_dir, session=session)
                successes += 1
            except Exception as e:
                logging.error("Failed for %s: %s", url, e)
                failures += 1

    logging.info("Batch complete. Success: %d, Failed: %d", successes, failures)


def _batch_concurrent(urls: list[str], out_dir: Path, workers: int) -> None:
    """
    Concurrent batch download using ThreadPoolExecutor.
    Note: each worker uses its own underlying requests calls.
    """
    logging.info("Running batch with %d workers", workers)

    successes = 0
    failures = 0

    def worker(url: str) -> tuple[str, bool]:
        try:
            logging.info("Worker handling %s", url)
            download_patent_pdf(url, out_dir)
            return (url, True)
        except Exception as e:
            logging.error("Failed for %s: %s", url, e)
            return (url, False)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {executor.submit(worker, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                _, ok = future.result()
                if ok:
                    successes += 1
                else:
                    failures += 1
            except Exception as e:
                logging.error("Unexpected error for %s: %s", url, e)
                failures += 1

    logging.info("Batch complete. Success: %d, Failed: %d", successes, failures)


if __name__ == "__main__":
    main()
