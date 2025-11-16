"""
Embedding and ChromaDB storage pipeline.
Reads validated chunks from data/validated/valid_chunks.jsonl
and stores embeddings into data/vector_db/.
"""

import json
import logging
import sys
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
VALIDATED_CHUNKS_PATH = DATA_DIR / "validated" / "valid_chunks.jsonl"
VECTOR_DB_PATH = DATA_DIR / "vector_db"
LOG_DIR = BASE_DIR / "logs" / "vector_store"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_DIR / "store_in_chromadb.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def main():
    logger.info("🚀 Starting embedding and ChromaDB storage pipeline...")

    try:
        if not VALIDATED_CHUNKS_PATH.exists():
            msg = f"Validated chunks file not found: {VALIDATED_CHUNKS_PATH}. Run validate_data.py first."
            logger.error(msg)
            raise FileNotFoundError(msg)

        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

        # --- Load validated chunks ---
        logger.info(f"📥 Loading validated chunks from: {VALIDATED_CHUNKS_PATH}")
        all_chunks = []
        with open(VALIDATED_CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    meta = record.get("metadata", {})
                    all_chunks.append({
                        "text": record.get("text", ""),
                        "company": meta.get("company", "Unknown"),
                        "chunk_id": meta.get("chunk_id", ""),
                        "filename": meta.get("doc_id", ""),
                        "section": meta.get("section_title", ""),
                        "hash": meta.get("hash_64", "")
                    })
                except json.JSONDecodeError:
                    logger.warning("⚠️ Skipping invalid JSON line in valid_chunks.jsonl")

        df = pd.DataFrame(all_chunks)
        logger.info(f"✅ Loaded {len(df)} valid chunks for embedding.")

        if df.empty:
            raise ValueError("No valid chunks found in file. Check validation output.")

        # --- Initialize ChromaDB ---
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        collection_name = "frontshift_handbooks"
        existing_collections = [c.name for c in client.list_collections()]

        # Clean rebuild
        if collection_name in existing_collections:
            logger.warning(f"🧹 Removing existing collection '{collection_name}' for rebuild.")
            client.delete_collection(name=collection_name)

        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

        # --- Prepare data ---
        documents = df["text"].tolist()
        metadatas = df[["filename", "company", "chunk_id", "section"]].to_dict(orient="records")
        ids = [f"chunk_{i}" for i in range(len(df))]

        # --- Sanity checks ---
        if len(documents) == 0:
            logger.warning("No documents to embed. Check valid_chunks.jsonl contents.")
        elif len(documents) < 10:
            logger.warning(f"Only {len(documents)} chunks detected. Possible small dataset.")

        # --- Add to ChromaDB ---
        logger.info(f"🧠 Adding {len(documents)} chunks to ChromaDB collection '{collection_name}'...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

        logger.info(f"💾 Stored {len(documents)} embeddings in collection '{collection_name}'.")
        logger.info(f"📂 Vector DB saved at: {VECTOR_DB_PATH}")

    except Exception as e:
        logger.error(f"❌ Error during embedding/storage stage: {e}", exc_info=True)
        raise

    logger.info("✅ Embedding pipeline completed successfully.")


if __name__ == "__main__":
    main()
