"""
Intelligent text chunking for legal documents.
Respects page boundaries and uses overlap for context preservation.
"""
import re
from typing import List, Dict, Any
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CHUNK_SIZE = 1000  # Characters (approximately 200-250 tokens)
CHUNK_OVERLAP = 200  # Characters overlap between chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep
MAX_CHUNK_SIZE = 2000  # Maximum chunk size (safety limit)

# Sentence ending patterns for legal text
SENTENCE_ENDINGS = r'[.!?]\s+|\.\s*\n|\n\n+'

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving structure."""
    # Split by sentence endings
    sentences = re.split(SENTENCE_ENDINGS, text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def merge_sentences(sentences: List[str], target_size: int) -> List[str]:
    """Merge sentences into chunks of approximately target_size."""
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed target, finalize current chunk
        if current_size + sentence_size > target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap (last few sentences)
            overlap_sentences = []
            overlap_size = 0
            # Take last 2-3 sentences for overlap
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= CHUNK_OVERLAP:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for space
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunk_text(text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Chunk text intelligently with overlap and metadata.
    
    Args:
        text: Text to chunk
        metadata: Metadata to attach to each chunk
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    # Merge sentences into chunks
    text_chunks = merge_sentences(sentences, CHUNK_SIZE)
    
    # Filter and validate chunks
    valid_chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        chunk_text = chunk_text.strip()
        
        # Skip too small chunks (unless it's the only chunk)
        if len(chunk_text) < MIN_CHUNK_SIZE and len(text_chunks) > 1:
            continue
        
        # Truncate if too large (shouldn't happen with our logic, but safety check)
        if len(chunk_text) > MAX_CHUNK_SIZE:
            chunk_text = chunk_text[:MAX_CHUNK_SIZE]
        
        chunk_data = {
            "chunk_text": chunk_text,
            "chunk_index": idx,
            "chunk_size": len(chunk_text),
            "total_chunks": len(text_chunks)
        }
        
        # Add metadata if provided
        if metadata:
            chunk_data.update(metadata)
        
        valid_chunks.append(chunk_data)
    
    return valid_chunks

def chunk_document_with_pages(pages: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Chunk a document while respecting page boundaries.
    Each page is chunked separately, but metadata includes page info.
    
    Args:
        pages: List of page dicts with 'page_no' and 'text'
        metadata: Base metadata for the document
    
    Returns:
        List of chunk dictionaries with text, page info, and metadata
    """
    all_chunks = []
    
    for page in pages:
        page_no = page.get("page_no", 0)
        page_text = page.get("text", "").strip()
        
        if not page_text:
            continue
        
        # Create page-specific metadata
        page_metadata = metadata.copy() if metadata else {}
        page_metadata.update({
            "page_no": page_no,
            "page_text_length": len(page_text)
        })
        
        # Chunk this page
        page_chunks = chunk_text(page_text, page_metadata)
        
        # Add page context to each chunk
        for chunk in page_chunks:
            chunk["page_no"] = page_no
            chunk["is_page_start"] = (chunk["chunk_index"] == 0)
        
        all_chunks.extend(page_chunks)
    
    # Re-index chunks globally
    for idx, chunk in enumerate(all_chunks):
        chunk["global_chunk_index"] = idx
        chunk["total_chunks"] = len(all_chunks)
    
    return all_chunks

def chunk_composite_text(composite_text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Chunk composite text that may contain multiple documents.
    Tries to preserve document boundaries marked by headers.
    
    Args:
        composite_text: Composite text with document markers
        metadata: Base metadata
    
    Returns:
        List of chunk dictionaries
    """
    if not composite_text or not composite_text.strip():
        return []
    
    # Split by document markers (e.g., [ORIGINAL | Issued: ...])
    doc_pattern = r'\[(ORIGINAL|CORRIGENDUM|AMENDMENT)\s*\|[^\]]+\]'
    doc_sections = re.split(doc_pattern, composite_text)
    
    all_chunks = []
    doc_index = 0
    
    for i, section in enumerate(doc_sections):
        section = section.strip()
        if not section:
            continue
        
        # Check if this section is a document marker
        if re.match(doc_pattern, section):
            doc_index += 1
            continue
        
        # Chunk this section
        section_metadata = metadata.copy() if metadata else {}
        section_metadata["document_section_index"] = doc_index
        
        section_chunks = chunk_text(section, section_metadata)
        all_chunks.extend(section_chunks)
    
    # If no document markers found, chunk as single document
    if not all_chunks:
        all_chunks = chunk_text(composite_text, metadata)
    
    # Re-index globally
    for idx, chunk in enumerate(all_chunks):
        chunk["global_chunk_index"] = idx
        chunk["total_chunks"] = len(all_chunks)
    
    return all_chunks


