"""
Vectorize chunked collection and push to Qdrant.
Each chunk gets its own vector with rich metadata.
"""
import json
import hashlib
import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ingestion.embeddings import embed_batch, EMBEDDING_DIM
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
COLLECTION_FILE = Path("storage/collection/collection_chunks.json")
COLLECTION_NAME = "notification_chunks"  # Updated collection name
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RECREATE_COLLECTION = True  # Set True only when rebuilding

BATCH_SIZE = 32  # Process embeddings in batches for efficiency

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def generate_chunk_id(chunk: Dict[str, Any]) -> str:
    """Generate a stable ID for a chunk."""
    # Use chunk_id if available, otherwise generate from content
    if "chunk_id" in chunk:
        return hashlib.md5(chunk["chunk_id"].encode("utf-8")).hexdigest()
    
    # Fallback: generate from group_id + chunk_index
    group_id = chunk.get("group_id", "unknown")
    chunk_idx = chunk.get("chunk_index", chunk.get("global_chunk_index", 0))
    unique_str = f"{group_id}__{chunk_idx}__{chunk.get('chunk_text', '')[:100]}"
    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()

def extract_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata payload for Qdrant."""
    # Core metadata
    payload = {
        "chunk_text": chunk.get("chunk_text", ""),
        "chunk_index": chunk.get("chunk_index", chunk.get("global_chunk_index", 0)),
        "total_chunks": chunk.get("total_chunks", 1),
        "chunk_size": chunk.get("chunk_size", len(chunk.get("chunk_text", ""))),
    }
    
    # Document metadata
    if "group_id" in chunk:
        payload["group_id"] = chunk["group_id"]
    if "tax_type" in chunk:
        payload["tax_type"] = chunk["tax_type"]
    if "notification_no" in chunk:
        payload["notification_no"] = chunk["notification_no"]
    if "issued_on" in chunk:
        payload["issued_on"] = chunk["issued_on"]
    if "latest_effective_date" in chunk:
        payload["latest_effective_date"] = chunk["latest_effective_date"]
    if "document_nature" in chunk:
        payload["document_nature"] = chunk["document_nature"]
    
    # Page metadata
    if "page_no" in chunk:
        payload["page_no"] = chunk["page_no"]
    if "is_page_start" in chunk:
        payload["is_page_start"] = chunk["is_page_start"]
    
    # Source file metadata
    if "file_path" in chunk:
        payload["file_path"] = chunk["file_path"]
    if "source_file" in chunk:
        payload["source_file"] = chunk["source_file"]
    if "source_file_path" in chunk:
        payload["source_file_path"] = chunk["source_file_path"]
    if "file_paths" in chunk:
        payload["file_paths"] = chunk["file_paths"]
    
    # Additional metadata
    if "source_type" in chunk:
        payload["source_type"] = chunk["source_type"]
    if "document_index" in chunk:
        payload["document_index"] = chunk["document_index"]
    if "document_section_index" in chunk:
        payload["document_section_index"] = chunk["document_section_index"]
    
    return payload

# --------------------------------------------------
# QDRANT SETUP
# --------------------------------------------------
def ensure_collection(client: QdrantClient):
    """Create or recreate Qdrant collection."""
    collections = {c.name for c in client.get_collections().collections}
    
    if COLLECTION_NAME in collections and RECREATE_COLLECTION:
        logger.info(f"🗑️  Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        collections.remove(COLLECTION_NAME)
        logger.info("✅ Collection deleted")
    
    if COLLECTION_NAME not in collections:
        logger.info(f"📦 Creating collection: {COLLECTION_NAME}")
        logger.info(f"   • Vector dimension: {EMBEDDING_DIM}")
        logger.info(f"   • Distance metric: Cosine")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
            optimizers_config={"indexing_threshold": 1}
        )
        logger.info("✅ Collection created successfully")
    else:
        logger.info(f"ℹ️  Collection already exists: {COLLECTION_NAME}")

# --------------------------------------------------
# MAIN VECTORIZATION
# --------------------------------------------------
def run_vectorization():
    """Load chunks, generate embeddings, and push to Qdrant."""
    start_time = time.time()
    
    if not COLLECTION_FILE.exists():
        raise FileNotFoundError(
            f"Collection file not found: {COLLECTION_FILE}\n"
            "Run: python -m ingestion.collection_builder first"
        )
    
    # Initialize Qdrant client
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL not set in environment variables")
    
    logger.info("🔗 Connecting to Qdrant...")
    logger.info(f"   • URL: {QDRANT_URL}")
    logger.info(f"   • Collection: {COLLECTION_NAME}")
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # Ensure collection exists
    ensure_collection(client)
    
    # Load chunks
    logger.info("")
    logger.info("📖 Loading chunks from collection file...")
    load_start = time.time()
    chunks = json.loads(COLLECTION_FILE.read_text(encoding="utf-8"))
    load_duration = time.time() - load_start
    
    total_chunks = len(chunks)
    logger.info(f"✅ Loaded {total_chunks:,} chunks in {load_duration:.2f} seconds")
    
    if not chunks:
        logger.warning("⚠️  No chunks to vectorize")
        return
    
    # Process in batches
    processed = 0
    failed = 0
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info("")
    logger.info("🔄 Starting vectorization...")
    logger.info(f"   • Batch size: {BATCH_SIZE}")
    logger.info(f"   • Total batches: {total_batches}")
    logger.info(f"   • Embedding dimension: {EMBEDDING_DIM}")
    logger.info("")
    
    embedding_time = 0
    upload_time = 0
    last_log_time = time.time()
    log_interval = 3  # Log every 3 seconds
    
    for batch_num, batch_start in enumerate(range(0, total_chunks, BATCH_SIZE), 1):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch = chunks[batch_start:batch_end]
        
        current_time = time.time()
        
        # Log progress periodically
        if current_time - last_log_time >= log_interval or batch_num == 1 or batch_num == total_batches:
            percentage = (batch_num / total_batches) * 100
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_chunks - processed) / rate if rate > 0 else 0
            
            logger.info(
                f"Progress: Batch [{batch_num}/{total_batches}] ({percentage:.1f}%) | "
                f"Chunks: [{processed}/{total_chunks}] | "
                f"Rate: {rate:.1f} chunks/s | ETA: {eta:.0f}s"
            )
            last_log_time = current_time
        
        try:
            # Extract texts for embedding
            texts = [chunk.get("chunk_text", "").strip() for chunk in batch]
            
            # Filter out empty texts
            valid_indices = [i for i, text in enumerate(texts) if text]
            valid_texts = [texts[i] for i in valid_indices]
            valid_chunks = [batch[i] for i in valid_indices]
            
            if not valid_texts:
                continue
            
            # Generate embeddings in batch
            embed_start = time.time()
            vectors = embed_batch(valid_texts)
            embedding_time += time.time() - embed_start
            
            # Prepare points for Qdrant
            points = []
            for chunk, vector in zip(valid_chunks, vectors):
                point_id = generate_chunk_id(chunk)
                payload = extract_chunk_metadata(chunk)
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            # Upsert to Qdrant
            if points:
                upload_start = time.time()
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                upload_time += time.time() - upload_start
            
            processed += len(points)
            
        except Exception as e:
            logger.error(f"❌ Error processing batch [{batch_num}/{total_batches}]: {e}")
            failed += len(batch)
            continue
    
    total_duration = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Vectorization complete")
    logger.info("=" * 70)
    logger.info(f"📊 Statistics:")
    logger.info(f"   • Total chunks processed: {processed:,}")
    logger.info(f"   • Failed chunks: {failed:,}")
    logger.info(f"   • Success rate: {(processed/(processed+failed)*100) if (processed+failed) > 0 else 0:.1f}%")
    logger.info(f"   • Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info(f"   • Embedding time: {embedding_time:.2f}s ({embedding_time/total_duration*100:.1f}%)")
    logger.info(f"   • Upload time: {upload_time:.2f}s ({upload_time/total_duration*100:.1f}%)")
    logger.info(f"   • Processing rate: {processed/total_duration if total_duration > 0 else 0:.1f} chunks/second")
    logger.info(f"   • Collection: {COLLECTION_NAME}")
    logger.info(f"   • Qdrant URL: {QDRANT_URL}")
    logger.info("=" * 70)

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run_vectorization()