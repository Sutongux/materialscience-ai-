"""
Patent-aware semantic chunker for RAG pipelines.

Reads cleaned patent text (.md or .txt), detects sections, paragraphs, claims,
preserves formulas/tables, and emits token-aware chunks suitable for embeddings.
Chunks are written to data/chunked/<filename>.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - fallback
    tiktoken = None


# ---------------------- Tokenization -----------------------------------------

def get_tokenizer():
    """Return a tokenizer; fallback to whitespace split if tiktoken unavailable."""
    if tiktoken:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return tiktoken.get_encoding("gpt2")

    class FallbackTokenizer:
        def encode(self, text: str) -> List[str]:
            return text.split()

    return FallbackTokenizer()


TOKENIZER = get_tokenizer()


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


# ---------------------- Data structures --------------------------------------

@dataclass
class AtomicBlock:
    text: str
    section: str
    para_id: Optional[str] = None
    claim_id: Optional[str] = None
    is_formula_or_table: bool = False

    def tokens(self) -> int:
        return count_tokens(self.text)


# ---------------------- Parsing helpers --------------------------------------

SECTION_PATTERN = re.compile(
    r"(?im)^\s*(?P<header>("
    r"technical field|field of the invention|field|background of the invention|background|summary of the invention|summary|"
    r"brief description of the drawings|brief description|detailed description of the invention|"
    r"detailed description of the preferred embodiments|detailed description|embodiments?|examples?|claims|"
    r"技术领域|背景技术|发明内容|附图说明|具体实施方式|权利要求书|摘要"
    r"))\s*[:：]?\s*$"
)

PARA_PATTERN = re.compile(
    r"(?m)^\s*("
    r"\[\s*(?P<brack>\d{3,5})\s*\]"            # [0001]
    r"|\【\s*(?P<cnbrack>\d{3,5})\s*\】"        # 【0001】
    r"|\(\s*(?P<paren>\d{3,5})\s*\)"           # (0001)
    r"|(?P<plain>\d{3,5})\."                   # 0001.
    r"|(?P<plaincolon>\d{3,5})[:：]"           # 0001: or 0001：
    r"|(?P<plaintextdash>\d{3,5})\s*[—-]"      # 0001 — or -
    r"|(?P<cncomma>\d{3,5})、"                 # 0001、
    r"|(?P<cnfullcomma>\d{3,5})，"             # 0001，
    r")"
)

CLAIM_PATTERN = re.compile(
    r"(?mi)^\s*("
    r"(?P<num>\d+)\.\s*"                  # 1.
    r"|claim\s+(?P<wordnum>\d+)"          # Claim 1
    r"|(?P<cnnum>\d+)[\.、]\s*"           # 1、 or 1.
    r")",
    re.MULTILINE,
)

FORMULA_TABLE_PATTERN = re.compile(
    r"(?im)("
    r"formula\s*\(\s*[ivx0-9]+\s*\)"         # Formula (I)
    r"|table\s*\d+"                          # Table 1
    r"|%\s*w/?w|wt\s*%"                      # % w/w
    r"|°c|degrees?\s*c"                      # temperature
    r"|表\s*\d+"                             # 表1
    r"|实施例\s*\d+"                         # 实施例1
    r"|式\s*\(\s*[ivx0-9]+\s*\)"             # 式(I)
    r"|温度"                                 # temperature keyword
    r")",
)


def normalize_section_name(name: str) -> str:
    return name.strip().upper()


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Split text into (section_name, content) preserving order."""
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("UNKNOWN", text)]

    sections: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        header = normalize_section_name(match.group("header"))
        body = text[start:end].strip()
        sections.append((header, body))
    return sections


def split_paragraphs(section_text: str) -> List[Tuple[Optional[str], str]]:
    """Split section text into paragraphs by numbered markers; include marker text."""
    chunks: List[Tuple[Optional[str], str]] = []
    last_idx = 0
    for match in PARA_PATTERN.finditer(section_text):
        start = match.start()
        if start > last_idx:
            prev = section_text[last_idx:start].strip()
            if prev:
                chunks.append((None, prev))
        para_id = (
            match.group("brack")
            or match.group("cnbrack")
            or match.group("paren")
            or match.group("plain")
            or match.group("plaincolon")
            or match.group("plaintextdash")
            or match.group("cncomma")
            or match.group("cnfullcomma")
        )
        end = PARA_PATTERN.search(section_text, match.end())
        next_start = end.start() if end else len(section_text)
        para_text = section_text[match.start():next_start].strip()
        chunks.append((para_id, para_text))
        last_idx = next_start

    if last_idx < len(section_text):
        tail = section_text[last_idx:].strip()
        if tail:
            chunks.append((None, tail))
    if chunks:
        return chunks

    # Fallback: split on blank lines to avoid one giant block when markers are missing.
    parts = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    return [(None, p) for p in parts] if parts else [(None, section_text.strip())]


def detect_claim_blocks(section_text: str) -> List[Tuple[str, str]]:
    """Detect claims; return list of (claim_id, text)."""
    matches = list(CLAIM_PATTERN.finditer(section_text))
    if not matches:
        return []
    claims: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        claim_id = match.group("num") or match.group("wordnum")
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section_text)
        claim_text = section_text[start:end].strip()
        claims.append((claim_id, claim_text))
    return claims


def label_formula_table(text: str) -> bool:
    return bool(FORMULA_TABLE_PATTERN.search(text))


# ---------------------- Chunk assembly ---------------------------------------

def build_atomic_blocks(section: str, content: str) -> List[AtomicBlock]:
    section_upper = normalize_section_name(section)

    if "CLAIMS" in section_upper or "权利要求书" in section_upper:
        claims = detect_claim_blocks(content)
        if claims:
            return [
                AtomicBlock(
                    text=txt,
                    section=section_upper,
                    para_id=None,
                    claim_id=cid,
                    is_formula_or_table=label_formula_table(txt),
                )
                for cid, txt in claims
            ]

    blocks: List[AtomicBlock] = []
    for para_id, para_text in split_paragraphs(content):
        blocks.append(
            AtomicBlock(
                text=para_text,
                section=section_upper,
                para_id=para_id,
                claim_id=None,
                is_formula_or_table=label_formula_table(para_text),
            )
        )
    return blocks


def merge_atomic_blocks(blocks: List[AtomicBlock], target_tokens: int = 800) -> List[AtomicBlock]:
    merged: List[AtomicBlock] = []
    current: List[AtomicBlock] = []
    current_tokens = 0

    def flush():
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(b.text for b in current)
        section = current[0].section
        para_ids = [b.para_id for b in current if b.para_id]
        claim_ids = [b.claim_id for b in current if b.claim_id]
        merged.append(
            AtomicBlock(
                text=text,
                section=section,
                para_id=para_ids[0] if para_ids else None,
                claim_id=claim_ids[0] if claim_ids else None,
                is_formula_or_table=any(b.is_formula_or_table for b in current),
            )
        )
        current = []
        current_tokens = 0

    for block in blocks:
        block_tokens = block.tokens()
        # If block alone exceeds target, flush current and append as standalone
        if block_tokens >= target_tokens:
            flush()
            merged.append(block)
            continue

        if current_tokens + block_tokens > target_tokens and current:
            flush()

        current.append(block)
        current_tokens += block_tokens

    flush()
    return merged


# ---------------------- Chunker API ------------------------------------------

def chunk_patent(text: str, filename: str) -> List[Dict[str, any]]:
    language = "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"
    sections = split_sections(text)
    atomic_blocks: List[AtomicBlock] = []
    for section, content in sections:
        atomic_blocks.extend(build_atomic_blocks(section, content))

    merged_blocks = merge_atomic_blocks(atomic_blocks, target_tokens=800)

    chunks: List[Dict[str, any]] = []
    for idx, block in enumerate(merged_blocks, start=1):
        tokens = block.tokens()
        para_range = block.para_id or ""
        chunk_id = f"{Path(filename).stem}_{idx:04d}"
        chunks.append(
            {
                "text": block.text,
                "metadata": {
                    "section": block.section,
                    "paragraphs": para_range if para_range else None,
                    "claim": int(block.claim_id) if block.claim_id and block.claim_id.isdigit() else block.claim_id,
                    "source": filename,
                    "tokens": tokens,
                    "chunk_id": chunk_id,
                    "language": language,
                },
            }
        )
    return chunks


def process_file(path: Path, out_dir: Path) -> None:
    text = path.read_text(encoding="utf-8")
    chunks = chunk_patent(text, path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Patent-aware chunker for RAG.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "data" / "cleaned"),
        help="Path to a cleaned patent file (.md or .txt) or a directory containing such files. "
        "Defaults to data/cleaned.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "chunked"),
        help="Output directory for jsonl chunks (default: data/chunked)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    if input_path.is_file():
        process_file(input_path, out_dir)
    elif input_path.is_dir():
        files = list(input_path.glob("*.md")) + list(input_path.glob("*.txt"))
        if not files:
            print(f"No .md or .txt files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        for path in files:
            process_file(path, out_dir)
    else:
        print(f"Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
