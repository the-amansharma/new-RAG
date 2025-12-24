# Hybrid Search System for Indian Tax Notifications

A sophisticated hybrid search system that combines semantic similarity with legal accuracy for precise retrieval of Indian government tax notifications.

## Features

### 🔍 Hybrid Search Approach
- **Semantic Search**: Uses BGE embeddings for semantic similarity matching
- **Keyword Search**: BM25-based keyword matching for exact legal term matching
- **Legal Accuracy Reranking**: Boosts results based on legal relevance factors

### 📊 Query Processing
- **Legal Term Extraction**: Identifies tax types, notification numbers, years
- **Query Expansion**: Expands queries with legal terminology synonyms
- **Entity Extraction**: Extracts key entities (notification numbers, dates, tax types)

### 🎯 Legal Accuracy Features
- **Notification Number Matching**: Boosts exact notification number matches
- **Tax Type Filtering**: Prioritizes relevant tax types
- **Document Authority**: Prefers original documents over amendments/corrigendums
- **Metadata Boosting**: Considers page numbers, issue dates, and document structure

## Architecture

```
User Query
    ↓
Query Processor
    ├─ Extract Entities (notification_no, tax_type, year)
    ├─ Expand Legal Terms
    └─ Extract Keywords
    ↓
Hybrid Search
    ├─ Semantic Search (Vector Similarity)
    └─ Keyword Search (BM25)
    ↓
Result Combination
    ├─ Score Normalization
    ├─ Weighted Combination
    └─ Deduplication
    ↓
Legal Accuracy Reranking
    ├─ Notification Number Boost
    ├─ Tax Type Boost
    ├─ Document Authority Boost
    └─ Metadata Boost
    ↓
Final Results
```

## Usage

### Interactive CLI

```bash
python search_api.py
```

Then enter your query interactively. Example queries:
- "input tax credit conditions"
- "notification 1/2018"
- "GST rate for services 2022"
- "reverse charge mechanism"

### Command Line

```bash
# Basic search
python search_api.py "input tax credit"

# Save results to file
python search_api.py "GST rate" --output results.json
```

### Python API

```python
from search.hybrid_search import HybridSearch

# Initialize search system
search = HybridSearch()

# Perform search
result = search.search(
    query="input tax credit conditions",
    limit=10,
    tax_type="Central Tax",  # Optional filter
    notification_no="1/2018",  # Optional filter
    year="2022"  # Optional filter
)

# Access results
for item in result["results"]:
    print(f"Notification: {item['notification_no']}")
    print(f"Score: {item['scores']['final']}")
    print(f"Text: {item['chunk_text'][:200]}...")
```

## Search Components

### 1. Query Processor (`query_processor.py`)
- Extracts notification numbers, years, tax types
- Expands queries with legal terminology
- Identifies important keywords

### 2. Hybrid Search (`hybrid_search.py`)
- Combines semantic and keyword search
- Normalizes and weights scores
- Performs legal accuracy reranking

### 3. Keyword Scorer (`keyword_scorer.py`)
- BM25-like scoring for keyword matching
- Handles term frequency and document length normalization

## Scoring System

### Score Components

1. **Semantic Score** (0-1): Cosine similarity from vector search
2. **Keyword Score** (0-1): BM25-based keyword matching
3. **Combined Score**: Weighted combination (60% semantic, 40% keyword)
4. **Legal Score** (0-1): Legal accuracy boost based on:
   - Exact notification number match: +0.3
   - Tax type match: +0.2
   - Year match: +0.1
   - Original document: +0.1
   - Page start: +0.05
5. **Final Score**: Combined score × (1 + legal_score)

### Result Ranking

Results are ranked by final score, with additional boosts for:
- Documents found in both semantic and keyword search
- Exact phrase matches in keyword search
- Original documents over amendments

## Configuration

### Search Parameters

Edit `search/hybrid_search.py`:

```python
DEFAULT_SEMANTIC_LIMIT = 50  # Candidates for semantic search
DEFAULT_KEYWORD_LIMIT = 50   # Candidates for keyword search
FINAL_RESULT_LIMIT = 10      # Final results returned

SEMANTIC_WEIGHT = 0.6        # Weight for semantic score
KEYWORD_WEIGHT = 0.4         # Weight for keyword score
```

### BM25 Parameters

Edit `search/keyword_scorer.py`:

```python
k1 = 1.5  # Term frequency saturation
b = 0.75  # Length normalization
```

## Legal Terminology

The system includes a comprehensive dictionary of Indian tax legal terms with synonyms:
- Tax types (GST, CGST, SGST, IGST, UTGST, Cess)
- Common terms (ITC, RCM, LUT, Refund, Assessment)
- Notification types (Notification, Circular, Corrigendum, Amendment)
- Rate-related terms (HSN, SAC, Tax Rate)

## Requirements

- Python 3.8+
- qdrant-client
- sentence-transformers (or HuggingFace API)
- dotenv

## Environment Variables

Ensure `.env` file contains:
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_token  # Optional, for API fallback
```

## Performance

- **Semantic Search**: ~100-200ms per query
- **Keyword Search**: ~50-150ms per query
- **Total Search Time**: ~200-400ms for hybrid search
- **Reranking**: ~10-50ms

## Tips for Better Results

1. **Be Specific**: Include notification numbers, tax types, or years when known
2. **Use Legal Terms**: Use standard legal terminology (e.g., "ITC" instead of "tax credit")
3. **Combine Terms**: Use multiple relevant terms (e.g., "input tax credit conditions LUT")
4. **Filter When Possible**: Use tax_type, notification_no, or year filters for precision

## Future Enhancements

- [ ] Cross-encoder reranking for better precision
- [ ] Query expansion using LLMs
- [ ] Multi-lingual support (Hindi legal terms)
- [ ] Citation and reference extraction
- [ ] Temporal relevance (recent notifications prioritized)
- [ ] Related notifications discovery

