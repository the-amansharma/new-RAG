"""
Create required indexes in Qdrant for better search performance.
Run this script once to create indexes for chunk_text, group_id, and other fields.
"""
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "notification_chunks"

def create_indexes():
    """Create indexes for better search performance."""
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL not set in environment variables")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Verify collection exists
    collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        raise ValueError(f"Collection '{COLLECTION_NAME}' not found. Run ingestion pipeline first.")
    
    logger.info(f"📦 Creating indexes for collection: {COLLECTION_NAME}")
    logger.info("=" * 70)
    
    # Indexes to create
    indexes = [
        {
            "field_name": "chunk_text",
            "field_schema": PayloadSchemaType.TEXT,
            "description": "Text index for keyword search in chunk_text"
        },
        {
            "field_name": "group_id",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "Keyword index for filtering by group_id"
        },
        {
            "field_name": "tax_type",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "Keyword index for filtering by tax_type"
        },
        {
            "field_name": "notification_no",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "Keyword index for filtering by notification_no"
        },
    ]
    
    created_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx_info in indexes:
        field_name = idx_info["field_name"]
        field_schema = idx_info["field_schema"]
        description = idx_info["description"]
        
        try:
            logger.info(f"\n📝 Creating {field_schema.value} index for '{field_name}'...")
            logger.info(f"   Purpose: {description}")
            
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema
            )
            
            logger.info(f"   ✅ Index created successfully")
            created_count += 1
            
        except Exception as e:
            error_msg = str(e)
            # Check if index already exists
            if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                logger.info(f"   ⚠️  Index already exists, skipping")
                skipped_count += 1
            else:
                logger.error(f"   ❌ Error creating index: {e}")
                error_count += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 Index Creation Summary:")
    logger.info(f"   ✅ Created: {created_count}")
    logger.info(f"   ⚠️  Skipped (already exist): {skipped_count}")
    logger.info(f"   ❌ Errors: {error_count}")
    logger.info("=" * 70)
    
    if created_count > 0 or skipped_count > 0:
        logger.info("\n✅ Index setup complete!")
        logger.info("   The search system will now work more efficiently.")
        logger.info("   Note: Indexes are optional - the system works without them using in-memory filtering.")
    else:
        logger.warning("\n⚠️  No indexes were created. Check errors above.")

if __name__ == "__main__":
    try:
        create_indexes()
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}", exc_info=True)
        exit(1)


