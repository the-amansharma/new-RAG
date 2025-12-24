# RAG Ingestion Pipeline - Optimized Version

## Overview

This optimized ingestion pipeline creates a proper vector collection for RAG (Retrieval-Augmented Generation) with:
- **Intelligent chunking** with overlap and page boundary awareness
- **Better embedding models** (BGE-base-en-v1.5, 768 dimensions)
- **Rich metadata** attached to each chunk
- **Efficient batch processing** for embeddings

## Architecture

### Pipeline Steps

1. **Text Extraction** (`text_extractor.py`)
   - Extracts text from PDFs page by page
   - Saves to `storage/extracted_text/`

2. **Document Identity** (`doc_identity.py`)
   - Extracts metadata: tax_type, notification_no, issued_on, document_nature
   - Creates document registry: `storage/document_registry.json`

3. **Instrument Grouping** (`instrument_grouper.py`)
   - Groups related documents by group_id
   - Creates composite text: `storage/instruments/*.json`

4. **Collection Building** (`collection_builder.py`) ⭐ NEW
   - Chunks documents intelligently with overlap
   - Preserves page boundaries and document structure
   - Combines text with metadata
   - Output: `storage/collection/collection_chunks.json`

5. **Vectorization** (`instrument_vectorizer.py`) ⭐ UPDATED
   - Generates embeddings using BGE model
   - Pushes chunked data with metadata to Qdrant
   - Each chunk is a separate vector with rich metadata

## Key Improvements

### 1. Better Embedding Model
- **Old**: `all-MiniLM-L6-v2` (384 dim, general purpose)
- **New**: `BAAI/bge-base-en-v1.5` (768 dim, optimized for retrieval)
- **Fallback**: HuggingFace Hub API if local model unavailable
- **Batch processing**: More efficient embedding generation

### 2. Intelligent Chunking
- **Chunk size**: ~1000 characters (~200-250 tokens)
- **Overlap**: 200 characters between chunks
- **Sentence-aware**: Respects sentence boundaries
- **Page-aware**: Preserves page information
- **Document-aware**: Maintains document structure

### 3. Rich Metadata
Each chunk includes:
- `chunk_text`: The actual text content
- `chunk_index`: Position within document
- `group_id`: Document group identifier
- `tax_type`: Tax type (Central Tax, Integrated Tax, etc.)
- `notification_no`: Notification number
- `issued_on`: Issue date
- `page_no`: Page number
- `file_path`: Source file path
- And more...

## Usage

### Full Pipeline
```bash
python -m ingestion.pipeline
```

### Quick Update (Collection + Vectorization only)
```bash
python -m ingestion.pipeline --quick
```

### Individual Steps
```bash
# Step 1: Extract text
python -m ingestion.text_extractor

# Step 2: Extract identity
python -m ingestion.doc_identity

# Step 3: Group instruments
python -m ingestion.instrument_grouper

# Step 4: Build collection
python -m ingestion.collection_builder instruments

# Step 5: Vectorize
python -m ingestion.instrument_vectorizer
```

## Configuration

### Environment Variables
Create a `.env` file:
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key
HF_TOKEN=your_huggingface_token  # Optional, for API fallback
```

### Chunking Parameters
Edit `ingestion/chunker.py`:
```python
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
MIN_CHUNK_SIZE = 100   # Minimum chunk size
```

### Embedding Model
Edit `ingestion/embeddings.py`:
```python
MODEL_ID = "BAAI/bge-base-en-v1.5"  # Change model here
EMBEDDING_DIM = 768                 # Update dimension
```

**Alternative models:**
- `BAAI/bge-small-en-v1.5` (384 dim, faster)
- `BAAI/bge-large-en-v1.5` (1024 dim, best quality)
- `intfloat/multilingual-e5-large` (1024 dim, multilingual)

## Output Structure

### Collection Chunks (`storage/collection/collection_chunks.json`)
```json
[
  {
    "chunk_id": "Central Tax__1_2018__chunk_0",
    "chunk_text": "...",
    "chunk_index": 0,
    "global_chunk_index": 0,
    "total_chunks": 15,
    "chunk_size": 987,
    "group_id": "Central Tax::1/2018",
    "tax_type": "Central Tax",
    "notification_no": "1/2018",
    "issued_on": "2018-01-01",
    "page_no": 1,
    "file_path": "data/notifications/...",
    ...
  }
]
```

### Qdrant Collection
- **Collection name**: `notification_chunks`
- **Vector dimension**: 768
- **Distance metric**: Cosine
- **Points**: One per chunk with full metadata

## Performance

- **Batch embedding**: Processes 32 chunks at a time
- **Local model**: Faster, no API limits (if available)
- **Overlap**: Ensures context preservation across chunks
- **Metadata filtering**: Enables efficient filtering in Qdrant

## Troubleshooting

### Model Download
First run will download the embedding model (~400MB). Ensure internet connection.

### Memory Issues
- Use `bge-small-en-v1.5` for lower memory
- Reduce `BATCH_SIZE` in `instrument_vectorizer.py`

### Qdrant Connection
- Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
- Check Qdrant service is running/accessible

## Next Steps

1. **Query Optimization**: Update search endpoints to use chunked collection
2. **Reranking**: Add cross-encoder reranking for better results
3. **Hybrid Search**: Combine vector search with keyword search
4. **Metadata Filtering**: Use Qdrant filters for tax_type, date ranges, etc.


