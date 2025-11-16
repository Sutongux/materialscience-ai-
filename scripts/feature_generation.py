"""
Feature generation script.
Reads cleaned documents from data/cleaned/, sends them to the Mercury LLM with a fixed
extraction prompt, and writes aggregated JSON rows to data/data.csv.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUT_CSV = DATA_DIR / "data.csv"

MERCURY_API_KEY = os.getenv("MERCURY_API_KEY")
MERCURY_BASE_URL = os.getenv("MERCURY_BASE_URL", "https://api.inceptionlabs.ai/v1/chat/completions")
MERCURY_MODEL = os.getenv("MERCURY_MODEL", "mercury")
MERCURY_MAX_TOKENS = int(os.getenv("MERCURY_MAX_TOKENS", "2000"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "350000"))  # ~87k tokens @4 chars/token
REFINE_ENABLED = os.getenv("REFINE_ENABLED", "true").lower() == "true"

PROMPT_TEMPLATE = """You are an information-extraction engine that reads cleaned markdown text 
and outputs structured data in a strict JSON schema that will later be converted into a CSV.

Your job:
- Parse the input text.
- Identify only information that is explicitly present.
- Do not infer, assume, or guess missing information.
- Never hallucinate content.
- If a field is not found in the input text, return the value null.
- Always return valid JSON, following the schema exactly.

Extraction Target:
Extract all relevant information from the text into the following fields:

{
  "core_invention": null,
  "mechanisms_principles": null,
  "formulations": null,
  "manufacturing_methods": null,
  "equipment_used": null,
  "performance_data": null,
  "comparative_tests": null,
  "failure_points": null,
  "alternative_embodiments": null,

  "components_list": null,
  "concentration_ranges": null,
  "functional_roles": null,
  "process_parameters": null,
  "application_instructions": null,
  "property_tables": null,

  "inventors": null,
  "assignee": null,
  "rd_direction": null,
  "market_pain_points": null,
  "target_markets": null,

  "protected_claims": null,
  "unprotected_opportunities": null,
  "essential_claim_elements": null,
  "expiration_dates": null,

  "economic_value": null,
  "cost_saving_measures": null,
  "target_customer": null,
  "scalability_notes": null,
  "manufacturing_feasibility": null,

  "regulatory_safety": null,
  "stability_safety_data": null,
  "compliance_references": null,
  "failure_modes": null,

  "technology_trends": null,
  "white_spaces": null,
  "citation_network": null
}

Rules:
1. Output must be valid JSON only. No comments, no explanations.
2. Do not remove required fields.
3. Do not change field names.
4. Use strings for text. Use null for missing fields.
5. Ignore markdown headers, bullets, artifacts, page numbers, figure references, and non-textual noise.
6. Extract content only from the text provided in the "input_text" section.

The final output must be a single JSON object and nothing else."""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_cleaned_documents() -> List[Dict[str, Any]]:
    """Load cleaned markdown (preferred) or JSON files from CLEANED_DIR."""
    docs: List[Dict[str, Any]] = []
    if not CLEANED_DIR.exists():
        raise FileNotFoundError(f"Cleaned directory not found: {CLEANED_DIR}")

    md_files = sorted(CLEANED_DIR.glob("*_cleaned.md"))
    json_files = sorted(CLEANED_DIR.glob("*_cleaned.json"))

    # Prefer markdown; if none, fall back to JSON text field.
    if md_files:
        for path in md_files:
            text = path.read_text(encoding="utf-8")
            docs.append({"filename": path.name, "text": text})
    elif json_files:
        for path in json_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = data.get("text", "")
            docs.append({"filename": path.name, "text": text})
    else:
        logging.warning("No cleaned markdown or JSON files found in %s", CLEANED_DIR)
    return docs


def invoke_mercury(prompt: str) -> str:
    """Call the Mercury model and return the raw JSON string."""
    if not MERCURY_API_KEY:
        raise RuntimeError("MERCURY_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {MERCURY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MERCURY_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": MERCURY_MAX_TOKENS,
    }

    resp = requests.post(MERCURY_BASE_URL, headers=headers, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} | Response: {resp.text}") from exc
    data = resp.json()

    # Compatible with OpenAI-style response
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Unexpected Mercury response format: {data}") from exc


def parse_json_response(raw_output: str, context: str) -> Dict[str, Any]:
    """Parse and validate that the model returned a JSON object."""
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON for {context}: {raw_output}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Model returned non-object JSON for {context}: {parsed}")
    return parsed


def chunk_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> List[str]:
    """Split text into chunks under the character budget, preserving paragraph boundaries where possible."""
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        line_len = len(line)
        if current_len + line_len > max_chars and current:
            chunks.append("".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("".join(current))
    return chunks


def merge_records(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two extracted records: prefer existing non-null; if both non-null and different, concatenate."""
    merged: Dict[str, Any] = {}
    for key, base_val in base.items():
        new_val = new.get(key)
        if base_val is None:
            merged[key] = new_val
        elif new_val is None:
            merged[key] = base_val
        elif base_val == new_val:
            merged[key] = base_val
        else:
            merged[key] = f"{base_val}\n\n{new_val}"
    return merged


def ensure_schema(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys with None to match the schema exactly."""
    schema_keys = [
        "core_invention",
        "mechanisms_principles",
        "formulations",
        "manufacturing_methods",
        "equipment_used",
        "performance_data",
        "comparative_tests",
        "failure_points",
        "alternative_embodiments",
        "components_list",
        "concentration_ranges",
        "functional_roles",
        "process_parameters",
        "application_instructions",
        "property_tables",
        "inventors",
        "assignee",
        "rd_direction",
        "market_pain_points",
        "target_markets",
        "protected_claims",
        "unprotected_opportunities",
        "essential_claim_elements",
        "expiration_dates",
        "economic_value",
        "cost_saving_measures",
        "target_customer",
        "scalability_notes",
        "manufacturing_feasibility",
        "regulatory_safety",
        "stability_safety_data",
        "compliance_references",
        "failure_modes",
        "technology_trends",
        "white_spaces",
        "citation_network",
    ]
    return {key: parsed.get(key, None) for key in schema_keys}


def refine_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Optional refinement pass to deduplicate and resolve contradictions."""
    if not REFINE_ENABLED:
        return record

    system_prompt = (
        "You are a JSON record refiner. Given a JSON object that follows a fixed schema, "
        "remove duplicate statements, resolve contradictions conservatively (retain only what is explicitly supported), "
        "and ensure every field uses null when empty. Do not add new fields."
    )
    user_prompt = (
        "Refine the following JSON record. Preserve the same keys and return JSON only:\n\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2)}"
    )

    headers = {
        "Authorization": f"Bearer {MERCURY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MERCURY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": MERCURY_MAX_TOKENS,
    }

    resp = requests.post(MERCURY_BASE_URL, headers=headers, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} | Response: {resp.text}") from exc
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Unexpected Mercury response format during refine: {data}") from exc

    refined_raw = content
    refined = parse_json_response(refined_raw, "refinement")
    refined = ensure_schema(refined)
    return refined


def process_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run extraction on all documents and return a list of rows."""
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        filename = doc["filename"]
        text = doc["text"]
        logging.info("Processing %s", filename)
        text_chunks = chunk_text(text, MAX_INPUT_CHARS)

        combined: Dict[str, Any] | None = None
        for idx, chunk in enumerate(text_chunks, start=1):
            if len(text_chunks) > 1:
                logging.info("  Chunk %d/%d (%d chars)", idx, len(text_chunks), len(chunk))
            user_prompt = f"input_text:\n{chunk}"
            raw_output = invoke_mercury(user_prompt)
            parsed = parse_json_response(raw_output, f"{filename} (chunk {idx})")
            parsed = ensure_schema(parsed)
            combined = parsed if combined is None else merge_records(combined, parsed)

        merged = combined or ensure_schema({})
        refined = refine_record(merged)

        row = {"filename": filename}
        row.update(refined)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]]) -> None:
    """Write rows to OUTPUT_CSV."""
    if not rows:
        logging.warning("No rows to write; skipping CSV creation.")
        return
    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Wrote %d rows to %s", len(rows), OUTPUT_CSV)


def main() -> None:
    configure_logging()
    docs = load_cleaned_documents()
    if not docs:
        logging.warning("No documents loaded; exiting.")
        return
    rows = process_documents(docs)
    write_csv(rows)


if __name__ == "__main__":
    main()
