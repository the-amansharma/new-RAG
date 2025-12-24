import os
from typing import List, Union
from dotenv import load_dotenv
import time
load_dotenv()
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
# Using BGE-base-en-v1.5 for better retrieval performance
# Alternative: "BAAI/bge-small-en-v1.5" (384 dim, faster) or "BAAI/bge-large-en-v1.5" (1024 dim, best quality)
# For Indian legal text, multilingual models like "BAAI/bge-m3" or "intfloat/multilingual-e5-large" work better
MODEL_ID = "BAAI/bge-base-en-v1.5"  # 768 dimensions, good balance
EMBEDDING_DIM = 768

# Use HuggingFace Hub API exclusively (lightweight server configuration)
# Set USE_LOCAL_MODEL=true in .env to use local model instead
USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

if USE_LOCAL:
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(MODEL_ID)
        logger.info(f"✅ Using local SentenceTransformer model: {MODEL_ID}")
    except Exception as e:
        logger.warning(f"⚠️ Local model not available, falling back to HuggingFace API: {e}")
        USE_LOCAL = False

if not USE_LOCAL:
    from huggingface_hub import InferenceClient
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError(
            "❌ HF_TOKEN or HUGGINGFACE_API_TOKEN is required in environment variables. "
            "Get your token from https://huggingface.co/settings/tokens"
        )
    _hf_client = InferenceClient(token=hf_token, model=MODEL_ID)
    logger.info(f"✅ Using HuggingFace Hub API for embeddings: {MODEL_ID}")

def embed_text(text: str) -> List[float]:
    """
    Generate embedding for text.
    Uses local SentenceTransformer if available, otherwise HuggingFace Hub API.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    text = text.strip()
    
    if USE_LOCAL:
        try:
            # Local model - encode returns numpy array
            vector = _local_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            # Convert to list
            if hasattr(vector, "tolist"):
                return vector.tolist()
            return list(vector)
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise e
    else:
        # HuggingFace Hub API
        try:
            response = _hf_client.feature_extraction(text, model=MODEL_ID)
            
            # Safe cast to list
            if hasattr(response, "tolist"):
                vector = response.tolist()
            else:
                vector = response

            # Handle nesting if the API returns [[...]] for a single input
            if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                return vector[0]
                
            return vector
        except Exception as e:
            logger.error(f"⚠️ HuggingFace Client Error: {e}")
            raise e

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts (more efficient).
    """
    if not texts:
        return []
    
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return []
    
    if USE_LOCAL:
        try:
            vectors = _local_model.encode(
                texts, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                batch_size=32
            )
            if hasattr(vectors, "tolist"):
                return vectors.tolist()
            return [list(v) for v in vectors]
        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")
            raise e
    else:
        # HuggingFace Hub API - supports batch processing
        try:
            # Process in smaller batches to avoid rate limits
            # HuggingFace API typically handles 10-50 texts per request
            batch_size = int(os.getenv("HF_BATCH_SIZE", "10"))
            all_vectors = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # Try batch processing - API accepts list of texts
                    response = _hf_client.feature_extraction(batch_texts, model=MODEL_ID)
                    
                    # Handle response format - API returns list of embeddings for batch input
                    if hasattr(response, "tolist"):
                        vectors = response.tolist()
                    else:
                        vectors = response
                    
                    # Normalize response format
                    if not isinstance(vectors, list):
                        # Unexpected format, try to convert
                        vectors = [vectors] if len(batch_texts) == 1 else list(vectors)
                    
                    # Ensure we have the right number of vectors
                    if len(vectors) != len(batch_texts):
                        # Response format mismatch, process individually
                        logger.warning(f"Batch response length mismatch, processing individually")
                        for text in batch_texts:
                            vector = embed_text(text)
                            all_vectors.append(vector)
                            time.sleep(0.05)
                    else:
                        # Process each vector in the batch response
                        for vec in vectors:
                            # Ensure vector is a list
                            if hasattr(vec, "tolist"):
                                vec = vec.tolist()
                            elif not isinstance(vec, list):
                                vec = list(vec)
                            
                            # Handle nested format [[...]] -> [...]
                            if isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
                                vec = vec[0]
                            
                            all_vectors.append(vec)
                    
                    # Small delay to respect rate limits (free tier: ~30 requests/min)
                    if i + batch_size < len(texts):
                        time.sleep(0.2)  # ~5 requests per second
                        
                except Exception as batch_error:
                    # If batch fails, fall back to individual calls
                    logger.warning(f"Batch embedding failed for batch {i//batch_size + 1}, using individual calls: {batch_error}")
                    for text in batch_texts:
                        try:
                            vector = embed_text(text)
                            all_vectors.append(vector)
                            time.sleep(0.1)  # Slower for individual calls
                        except Exception as e:
                            logger.error(f"Failed to embed text: {e}")
                            # Return zero vector as fallback (better than crashing)
                            all_vectors.append([0.0] * EMBEDDING_DIM)
            
            if len(all_vectors) != len(texts):
                logger.warning(f"Expected {len(texts)} vectors, got {len(all_vectors)}")
            
            return all_vectors
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Final fallback to individual calls
            logger.info("Falling back to individual embedding calls...")
            return [embed_text(text) for text in texts]
