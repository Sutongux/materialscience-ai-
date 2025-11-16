"""
Embed patent chunks and store them in ChromaDB.
Reads JSONL chunk files from data/chunked, embeds with sentence-transformers,
and writes to a persistent Chroma collection.
"""

from __future__ import annotations

import os 
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHUNK_DIR = DATA_DIR / "chunked"
VECTOR_DB_PATH = DATA_DIR / "vector_db"
COLLECTION_NAME = "patent_chunks"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")

LOGGER = logging.getLogger("embed_chunks")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOGGER.addHandler(handler)


def read_chunks(chunk_dir: Path) -> pd.DataFrame:
    """Load all jsonl chunk files into a DataFrame."""
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    rows: List[Dict[str, Any]] = []
    for path in sorted(chunk_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                meta = record.get("metadata", {})
                meta = sanitize_metadata(meta)
                rows.append(
                    {
                        "id": meta.get("chunk_id") or f"{path.stem}_{len(rows)}",
                        "text": record.get("text", ""),
                        "metadata": meta,
                    }
                )
    if not rows:
        raise ValueError(f"No chunks found in {chunk_dir}")

    df = pd.DataFrame(rows)
    LOGGER.info("Loaded %d chunks from %d files", len(df), len(list(chunk_dir.glob('*.jsonl'))))
    return df


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata contains only Chroma-friendly primitives; drop None values."""
    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def get_collection(client: chromadb.PersistentClient, name: str) -> chromadb.api.models.Collection.Collection:
    """Create or get an existing Chroma collection."""
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        LOGGER.info("Collection '%s' exists; deleting for fresh load.", name)
        client.delete_collection(name)
    return client.create_collection(
        name=name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        ),
    )


def embed_and_store(df: pd.DataFrame, collection) -> None:
    """Embed chunks and store in Chroma."""
    df["metadata"] = df["metadata"].apply(sanitize_metadata)
    LOGGER.info("Embedding %d chunks ...", len(df))
    collection.add(
        ids=df["id"].tolist(),
        documents=df["text"].tolist(),
        metadatas=df["metadata"].tolist(),
    )
    LOGGER.info("Stored %d embeddings in collection '%s'", len(df), collection.name)


def main() -> None:
    LOGGER.info("Starting embedding pipeline")
    df = read_chunks(CHUNK_DIR)

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = get_collection(client, COLLECTION_NAME)

    embed_and_store(df, collection)
    LOGGER.info("Embedding pipeline complete. Vector DB at %s", VECTOR_DB_PATH)


if __name__ == "__main__":
    main()
