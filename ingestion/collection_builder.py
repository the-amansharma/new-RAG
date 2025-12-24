"""
Builds a collection from text extractor data and doc_identity metadata.
Creates chunked documents with rich metadata for vectorization.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ingestion.chunker import chunk_composite_text, chunk_document_with_pages

# Configure logging
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
REGISTRY_PATH = Path("storage/document_registry.json")
EXTRACTED_TEXT_DIR = Path("storage/extracted_text")
INSTRUMENTS_DIR = Path("storage/instruments")
COLLECTION_OUTPUT_DIR = Path("storage/collection")
COLLECTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def parse_date(date_str: str) -> datetime:
    """Parse ISO date string to datetime."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None

def load_extracted_pages_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load extracted text pages for a PDF."""
    # Handle both absolute and relative paths
    try:
        relative_path = file_path.relative_to(Path("data/notifications"))
    except ValueError:
        # If not relative to data/notifications, try to find it
        if "notifications" in str(file_path):
            parts = Path(file_path).parts
            idx = parts.index("notifications")
            relative_path = Path(*parts[idx+1:])
        else:
            return []
    
    safe_name = (
        relative_path
        .as_posix()
        .replace("/", "__")
        .replace(".pdf", ".json")
    )
    json_path = EXTRACTED_TEXT_DIR / safe_name
    
    if not json_path.exists():
        return []
    
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("pages", [])
    except Exception:
        return []

def build_document_metadata(doc_identity: Dict[str, Any], document_index: int = 0) -> Dict[str, Any]:
    """Build metadata dictionary from document identity."""
    return {
        "group_id": doc_identity.get("group_id"),
        "tax_type": doc_identity.get("tax_type"),
        "notification_no": doc_identity.get("notification_no"),
        "issued_on": doc_identity.get("issued_on"),
        "document_nature": doc_identity.get("document_nature"),
        "source_type": doc_identity.get("source_type", "notification"),
        "identity_source": doc_identity.get("identity_source"),
        "file_path": doc_identity.get("file_path"),
        "is_groupable": doc_identity.get("is_groupable", False),
        "document_index": document_index
    }

# --------------------------------------------------
# MAIN COLLECTION BUILDER
# --------------------------------------------------
def build_collection_from_registry():
    """
    Build collection from document registry.
    Creates chunked documents with metadata for each entry.
    """
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Document registry not found: {REGISTRY_PATH}")
    
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    print(f"📋 Processing {len(registry)} documents from registry\n")
    
    all_chunks = []
    processed_count = 0
    error_count = 0
    
    for idx, doc_identity in enumerate(registry, start=1):
        file_path = Path(doc_identity.get("file_path", ""))
        
        if not file_path.exists():
            print(f"[{idx}/{len(registry)}] ⚠️ File not found: {file_path}")
            error_count += 1
            continue
        
        print(f"[{idx}/{len(registry)}] Processing: {file_path.name}", end="\r", flush=True)
        
        try:
            # Load extracted pages
            pages = load_extracted_pages_from_file(file_path)
            
            if not pages:
                print(f"\n[{idx}/{len(registry)}] ⚠️ No pages extracted: {file_path.name}")
                error_count += 1
                continue
            
            # Build base metadata
            base_metadata = build_document_metadata(doc_identity)
            
            # Chunk document with page awareness
            doc_chunks = chunk_document_with_pages(pages, base_metadata)
            
            # Add unique chunk IDs
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_id = f"{base_metadata['group_id']}__doc_{idx}__chunk_{chunk_idx}"
                chunk["chunk_id"] = chunk_id
                chunk["source_file"] = file_path.name
                chunk["source_file_path"] = str(file_path)
            
            all_chunks.extend(doc_chunks)
            processed_count += 1
            
        except Exception as e:
            print(f"\n[{idx}/{len(registry)}] ❌ Error processing {file_path.name}: {e}")
            error_count += 1
            continue
    
    print(f"\n\n✅ Collection building complete")
    print(f"   ✔ Processed: {processed_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📦 Total chunks: {len(all_chunks)}")
    
    # Save collection
    output_file = COLLECTION_OUTPUT_DIR / "collection_chunks.json"
    output_file.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"   💾 Saved to: {output_file.resolve()}")
    
    return all_chunks

def build_collection_from_instruments():
    """
    Build collection from grouped instruments.
    Uses composite_text from instruments for better context.
    """
    start_time = time.time()
    
    if not INSTRUMENTS_DIR.exists():
        raise FileNotFoundError(f"Instruments directory not found: {INSTRUMENTS_DIR}")
    
    instrument_files = sorted(INSTRUMENTS_DIR.glob("*.json"))
    total_instruments = len(instrument_files)
    
    logger.info(f"📦 Found {total_instruments} instrument files to process")
    logger.info(f"📂 Instruments directory: {INSTRUMENTS_DIR.resolve()}")
    
    if total_instruments == 0:
        logger.warning("⚠️  No instrument files found!")
        return []
    
    all_chunks = []
    processed_count = 0
    error_count = 0
    skipped_count = 0
    total_text_length = 0
    
    # Progress tracking
    last_log_time = time.time()
    log_interval = 5  # Log progress every 5 seconds
    
    for idx, instrument_file in enumerate(instrument_files, start=1):
        current_time = time.time()
        
        # Log progress periodically
        if current_time - last_log_time >= log_interval or idx == 1 or idx == total_instruments:
            percentage = (idx / total_instruments) * 100
            elapsed = current_time - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_instruments - idx) / rate if rate > 0 else 0
            
            logger.info(
                f"Progress: [{idx}/{total_instruments}] ({percentage:.1f}%) | "
                f"Processed: {processed_count} | Errors: {error_count} | "
                f"Chunks: {len(all_chunks)} | ETA: {eta:.0f}s"
            )
            last_log_time = current_time
        
        try:
            data = json.loads(instrument_file.read_text(encoding="utf-8"))
            
            composite_text = data.get("composite_text", "").strip()
            if not composite_text:
                skipped_count += 1
                continue
            
            text_length = len(composite_text)
            total_text_length += text_length
            
            # Build metadata from instrument
            base_metadata = {
                "group_id": data.get("group_id"),
                "tax_type": data.get("tax_type"),
                "notification_no": data.get("notification_no"),
                "latest_effective_date": data.get("latest_effective_date"),
                "file_paths": data.get("file_paths", []),
                "document_count": len(data.get("documents", []))
            }
            
            # Add document details
            documents = data.get("documents", [])
            for doc_idx, doc in enumerate(documents):
                base_metadata[f"doc_{doc_idx}_nature"] = doc.get("document_nature")
                base_metadata[f"doc_{doc_idx}_issued_on"] = doc.get("issued_on")
            
            # Chunk composite text
            doc_chunks = chunk_composite_text(composite_text, base_metadata)
            
            # Add unique chunk IDs and source info
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunk_id = f"{base_metadata['group_id']}__chunk_{chunk_idx}"
                chunk["chunk_id"] = chunk_id
                chunk["instrument_file"] = instrument_file.name
            
            all_chunks.extend(doc_chunks)
            processed_count += 1
            
            # Log detailed progress for every 50th file
            if idx % 50 == 0 or idx == total_instruments:
                logger.debug(
                    f"  [{idx}/{total_instruments}] {instrument_file.name}: "
                    f"{len(doc_chunks)} chunks created"
                )
            
        except Exception as e:
            logger.error(f"  [{idx}/{total_instruments}] ❌ Error processing {instrument_file.name}: {e}")
            error_count += 1
            continue
    
    duration = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Collection building complete")
    logger.info("=" * 70)
    logger.info(f"📊 Statistics:")
    logger.info(f"   • Total instruments: {total_instruments}")
    logger.info(f"   • Successfully processed: {processed_count}")
    logger.info(f"   • Skipped (no text): {skipped_count}")
    logger.info(f"   • Errors: {error_count}")
    logger.info(f"   • Total chunks created: {len(all_chunks)}")
    logger.info(f"   • Average chunks per instrument: {len(all_chunks)/processed_count if processed_count > 0 else 0:.1f}")
    logger.info(f"   • Total text processed: {total_text_length:,} characters")
    logger.info(f"   • Processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"   • Processing rate: {processed_count/duration if duration > 0 else 0:.1f} instruments/sec")
    
    # Save collection
    logger.info("")
    logger.info("💾 Saving collection to disk...")
    output_file = COLLECTION_OUTPUT_DIR / "collection_chunks.json"
    
    save_start = time.time()
    output_file.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    save_duration = time.time() - save_start
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Collection saved successfully")
    logger.info(f"   • File: {output_file.resolve()}")
    logger.info(f"   • Size: {file_size_mb:.2f} MB")
    logger.info(f"   • Save time: {save_duration:.2f} seconds")
    logger.info("=" * 70)
    
    return all_chunks

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "instruments"
    
    if mode == "registry":
        build_collection_from_registry()
    elif mode == "instruments":
        build_collection_from_instruments()
    else:
        print("Usage: python collection_builder.py [registry|instruments]")
        print("  registry: Build from document_registry.json (individual documents)")
        print("  instruments: Build from instruments/*.json (grouped documents) [default]")
