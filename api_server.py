"""
Production-ready Flask API server for search functionality.
Returns top 3 results with target chunk ±1 chunks.
Each result includes the target chunk plus the chunks before and after it (±1),
combined into a single text output, along with accurate page number and tax type.
"""
import os
import json
import re
import codecs
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any, List, Optional
from search.hybrid_search import HybridSearch

# Configure logging for production
# Render.com logs to stdout/stderr, so we use basic format
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if os.getenv("FLASK_ENV") == "production":
    log_level = "WARNING"

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Ensure logs go to stdout for Render
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration from environment
# Render.com provides PORT automatically, use 5000 as fallback
DEBUG_MODE = os.getenv("FLASK_DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")  # 0.0.0.0 for Render.com

# Initialize search system
search_system = None
try:
    search_system = HybridSearch()
    logger.info("✅ Search system initialized!")
except Exception as e:
    logger.error(f"Failed to initialize search system: {e}")
    search_system = None


def format_page_content(chunks: List[Dict[str, Any]]) -> str:
    """Combine all chunks from a page into a single text."""
    if not chunks:
        return ""
    
    texts = []
    for chunk in sorted(chunks, key=lambda x: x.get("chunk_index", 0)):
        chunk_text = chunk.get("payload", {}).get("chunk_text", "")
        if chunk_text:
            texts.append(chunk_text)
    
    return "\n\n".join(texts)


def get_page_text_from_extracted_data(
    tax_type: str = None,
    notification_no: str = None,
    page_no: int = None,
    file_path: str = None
) -> str:
    """
    Get full page text from extracted data files in storage/extracted_text.
    
    Priority order:
    1. Use file_path to construct exact JSON filename (most reliable)
    2. Fall back to tax_type + notification_no + page_no matching
    
    Args:
        tax_type: Tax type (e.g., "Central Tax" or "Central Tax (Rate)")
        notification_no: Notification number (e.g., "05/2018")
        page_no: Page number to retrieve (1-indexed)
        file_path: Original PDF file path (e.g., "data\\notifications\\Central Tax (Rate)\\2017\\file.pdf")
    
    Returns:
        Full page text content, or empty string if not found
    """
    if page_no is None:
        return ""
    
    try:
        extracted_text_dir = Path("storage/extracted_text")
        if not extracted_text_dir.exists():
            logger.warning(f"Extracted text directory not found: {extracted_text_dir}")
            return ""
        
        json_file = None
        
        # Method 1: Use file_path if available (most reliable)
        if file_path:
            try:
                file_path_obj = Path(file_path)
                # Extract relative path from data/notifications
                if "notifications" in str(file_path_obj):
                    parts = file_path_obj.parts
                    try:
                        idx = parts.index("notifications")
                        relative_path = Path(*parts[idx+1:])
                    except ValueError:
                        # Fallback: try to extract from path string
                        path_str = str(file_path_obj).replace("\\", "/")
                        if "notifications/" in path_str:
                            relative_path = Path(path_str.split("notifications/")[-1])
                        else:
                            relative_path = None
                    
                    if relative_path:
                        # Convert to safe JSON filename: replace / with __, .pdf with .json
                        safe_name = relative_path.as_posix().replace("/", "__").replace(".pdf", ".json")
                        json_file = extracted_text_dir / safe_name
                        
                        if json_file.exists():
                            logger.debug(f"Found extracted text file using file_path: {json_file.name}")
            except Exception as e:
                logger.debug(f"Error constructing path from file_path {file_path}: {e}")
        
        # Method 2: Fall back to searching by tax_type, notification_no, and year
        if not json_file or not json_file.exists():
            if not tax_type or not notification_no:
                logger.warning(f"Insufficient parameters: tax_type={tax_type}, notification_no={notification_no}")
                return ""
            
            # Extract year from notification_no (e.g., "05/2018" -> "2018")
            year = None
            if "/" in notification_no:
                year = notification_no.split("/")[-1]
            else:
                logger.warning(f"Invalid notification_no format: {notification_no}")
                return ""
            
            # Try tax_type variations to find matching file
            tax_type_variations = [
                tax_type,
                tax_type.replace(" ", "_"),
                tax_type.replace(" ", "_").replace("(", "").replace(")", "").replace(" ", ""),
            ]
            
            # Also handle "Central Tax (Rate)" vs "Central Tax (Rate)" naming
            if "(Rate)" in tax_type:
                tax_type_variations.append(tax_type.replace(" ", "_").replace("(", "").replace(")", ""))
            
            pattern = f"*__{year}__*.json"
            matching_files = list(extracted_text_dir.glob(pattern))
            
            for json_candidate in matching_files:
                # Check if filename starts with any tax_type variation
                filename = json_candidate.name
                matches_tax_type = any(
                    filename.startswith(f"{tax_var}__{year}__") or 
                    filename.startswith(f"{tax_var.replace('_', ' ')}__{year}__")
                    for tax_var in tax_type_variations
                )
                
                if matches_tax_type:
                    try:
                        data = json.loads(json_candidate.read_text(encoding="utf-8"))
                        pages = data.get("pages", [])
                        if not pages:
                            continue
                        
                        # Check if this file contains the notification number
                        first_page_text = pages[0].get("text", "")
                        notification_variations = [
                            notification_no,
                            notification_no.replace("/", "-"),
                            f"Notification No {notification_no}",
                            f"Notification No. {notification_no}",
                            f"No {notification_no}",
                            f"No. {notification_no}"
                        ]
                        
                        matches_notification = any(nv in first_page_text for nv in notification_variations)
                        
                        if matches_notification:
                            json_file = json_candidate
                            logger.debug(f"Found extracted text file by search: {json_file.name}")
                            break
                    except Exception as e:
                        logger.debug(f"Error reading {json_candidate}: {e}")
                        continue
        
        # Load and return the requested page
        if json_file and json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                pages = data.get("pages", [])
                
                # Find the requested page
                for page in pages:
                    if page.get("page_no") == page_no:
                        page_text = page.get("text", "")
                        logger.info(f"Retrieved page {page_no} from {json_file.name}")
                        return page_text
                
                logger.warning(f"Page {page_no} not found in {json_file.name} (file has {len(pages)} pages)")
                return ""
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")
                return ""
        else:
            logger.warning(f"Could not find extracted text file for tax_type={tax_type}, notification_no={notification_no}, page_no={page_no}")
            return ""
        
    except Exception as e:
        logger.error(f"Error getting page text from extracted data: {e}")
        return ""


def get_total_pages_for_document(group_id: str) -> int:
    """
    Get total number of pages for a document.
    
    Args:
        group_id: Document group ID
    
    Returns:
        Total number of pages, or 0 if error
    """
    if not group_id:
        return 0
    
    try:
        # Get all chunks for the document
        all_chunks = search_system.get_document_chunks(group_id)
        
        if not all_chunks:
            return 0
        
        # Get unique page numbers
        page_numbers = set()
        for chunk in all_chunks:
            page_no = chunk.get("payload", {}).get("page_no")
            if page_no is not None:
                page_numbers.add(page_no)
        
        return len(page_numbers)
        
    except Exception as e:
        logger.error(f"Error getting total pages for group_id {group_id}: {e}")
        return 0


def get_chunks_by_index_range(
    group_id: str,
    matched_chunk_index: int,
    context_range: int = 2
) -> List[Dict[str, Any]]:
    """
    Get chunks within ±context_range of the matched chunk index.
    
    Args:
        group_id: Document group ID
        matched_chunk_index: The chunk index of the matched chunk
        context_range: Number of chunks before and after to include (default: 2)
    
    Returns:
        List of chunks with their text and chunk_index, sorted by chunk_index
    """
    if not group_id or matched_chunk_index is None:
        return []
    
    try:
        # Get all chunks for the document
        all_chunks = search_system.get_document_chunks(group_id)
        
        if not all_chunks:
            return []
        
        # Filter chunks within the range
        start_index = max(0, matched_chunk_index - context_range)
        end_index = matched_chunk_index + context_range + 1
        
        context_chunks = []
        for chunk in all_chunks:
            chunk_payload = chunk.get("payload", {})
            chunk_idx = chunk_payload.get("chunk_index")
            if chunk_idx is None:
                chunk_idx = chunk_payload.get("global_chunk_index")
            
            if chunk_idx is not None and start_index <= chunk_idx < end_index:
                context_chunks.append({
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk_payload.get("chunk_text", "")
                })
        
        # Sort by chunk_index
        context_chunks.sort(key=lambda x: x["chunk_index"])
        return context_chunks
        
    except Exception as e:
        logger.error(f"Error fetching chunks by index range for group_id {group_id}, chunk_index {matched_chunk_index}: {e}")
        return []


def get_target_chunk_plus_minus_one(
    group_id: str,
    page_no: int,
    matched_chunk_text: str = None,
    target_chunk_index: int = None
) -> str:
    """
    Get target chunk ±1 chunks from the same page and combine them.
    
    Args:
        group_id: Document group ID
        page_no: Page number where the target chunk is located
        matched_chunk_text: Text of the matched chunk (used to find the target chunk on the page)
        target_chunk_index: Optional chunk_index hint (global or page-specific)
    
    Returns:
        Combined text of chunks at index-1, index, and index+1 from the same page, or empty string if not found
    """
    if not group_id or page_no is None:
        return ""
    
    try:
        # Get all chunks for the specific page, sorted by chunk_index
        page_chunks = search_system.get_page_chunks(group_id, page_no)
        
        if not page_chunks:
            logger.warning(f"No chunks found for page {page_no} in group_id {group_id}")
            return ""
        
        # Sort chunks by chunk_index to maintain order
        sorted_page_chunks = sorted(page_chunks, key=lambda x: x.get("chunk_index", 0))
        
        # Find the target chunk on this page
        target_position = None
        
        # Method 1: Try to find by chunk_index first (most reliable if available)
        if target_chunk_index is not None:
            for idx, chunk in enumerate(sorted_page_chunks):
                chunk_payload = chunk.get("payload", {})
                chunk_idx = chunk_payload.get("chunk_index")
                if chunk_idx is None:
                    chunk_idx = chunk_payload.get("global_chunk_index")
                
                if chunk_idx == target_chunk_index:
                    target_position = idx
                    logger.debug(f"Found target chunk at position {idx} on page {page_no} by chunk_index {target_chunk_index}")
                    break
        
        # Method 2: If chunk_index didn't work, try text matching (more flexible)
        if target_position is None and matched_chunk_text:
            matched_text_clean = matched_chunk_text.strip().lower()
            
            # Try exact substring match first
            for idx, chunk in enumerate(sorted_page_chunks):
                chunk_text = chunk.get("payload", {}).get("chunk_text", "").strip().lower()
                if matched_text_clean in chunk_text or chunk_text in matched_text_clean:
                    target_position = idx
                    logger.debug(f"Found target chunk at position {idx} on page {page_no} by exact text matching")
                    break
            
            # If still not found, try fuzzy matching with first 100 chars
            if target_position is None:
                matched_snippet = matched_text_clean[:100]
                for idx, chunk in enumerate(sorted_page_chunks):
                    chunk_text = chunk.get("payload", {}).get("chunk_text", "").strip().lower()
                    chunk_snippet = chunk_text[:100]
                    if matched_snippet in chunk_snippet or chunk_snippet in matched_snippet:
                        target_position = idx
                        logger.debug(f"Found target chunk at position {idx} on page {page_no} by fuzzy text matching")
                        break
        
        # If still not found, use the first chunk as fallback (most likely the matched chunk)
        if target_position is None:
            if sorted_page_chunks:
                target_position = 0
                logger.warning(f"Could not find target chunk reliably, using first chunk at position 0 on page {page_no}")
            else:
                return ""
        
        # Get chunks at position-1, position, and position+1
        target_chunks = []
        positions_to_get = [target_position - 1, target_position, target_position + 1]
        
        for pos in positions_to_get:
            if 0 <= pos < len(sorted_page_chunks):
                chunk = sorted_page_chunks[pos]
                chunk_text = chunk.get("payload", {}).get("chunk_text", "")
                if chunk_text:
                    target_chunks.append({
                        "position": pos,
                        "chunk_index": chunk.get("chunk_index", 0),
                        "chunk_text": chunk_text
                    })
        
        if not target_chunks:
            logger.warning(f"No chunks found around position {target_position} on page {page_no}")
            return ""
        
        # Sort by position to maintain order
        target_chunks.sort(key=lambda x: x["position"])
        
        # Combine the chunks with clear separators
        combined_text = "\n\n".join([chunk["chunk_text"] for chunk in target_chunks])
        
        logger.info(f"Retrieved {len(target_chunks)} chunks (positions: {[c['position'] for c in target_chunks]}) for page {page_no}")
        return combined_text
        
    except Exception as e:
        logger.error(f"Error fetching target chunk ±1 for group_id {group_id}, page {page_no}: {e}", exc_info=True)
        return ""


def format_score_for_user(final_score: float) -> str:
    """
    Convert technical score to user-friendly format.
    
    Args:
        final_score: The final combined score (0.0 to ~2.1)
    
    Returns:
        User-friendly score string (e.g., "95%", "High", "Medium", "Low")
    """
    if final_score >= 1.5:
        return "Very High"
    elif final_score >= 1.2:
        return "High"
    elif final_score >= 0.8:
        return "Medium"
    elif final_score >= 0.5:
        return "Low"
    else:
        return "Very Low"


def clean_text_for_output(text: str) -> str:
    """
    Clean text to remove newlines and Unicode escape sequences.
    
    Args:
        text: Input text that may contain \\n and \\u escape sequences
    
    Returns:
        Cleaned text with newlines replaced by spaces and Unicode escapes decoded
    """
    if not text:
        return ""
    
    # Replace all newline characters (\n, \r\n, \r) with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Decode Unicode escape sequences (like \u201c, \u2019, \u00A0, etc.)
    # Check for both \u and \\u patterns
    if '\\u' in text:
        # First try codecs.decode approach (most reliable)
        try:
            # Encode to latin-1 to preserve all byte values (0-255)
            # Then decode with unicode_escape which handles \uXXXX sequences
            decoded = text.encode('latin-1').decode('unicode_escape')
            text = decoded
        except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
            # If encoding fails, use regex-based approach
            # This handles cases where text contains characters outside latin-1 range
            def decode_unicode_escape(match):
                """Decode a single Unicode escape sequence."""
                try:
                    hex_str = match.group(1)
                    code_point = int(hex_str, 16)
                    if 0 <= code_point <= 0x10FFFF:
                        return chr(code_point)
                    return match.group(0)
                except (ValueError, OverflowError):
                    return match.group(0)
            
            # Match \u followed by exactly 4 hex digits (case insensitive)
            # Process multiple times to handle nested or multiple sequences
            prev_text = ""
            iterations = 0
            while text != prev_text and iterations < 5:
                prev_text = text
                text = re.sub(r'\\u([0-9a-fA-F]{4})', decode_unicode_escape, text, flags=re.IGNORECASE)
                iterations += 1
    
    # Replace multiple consecutive spaces with a single space
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    # Strip leading/trailing whitespace
    return text.strip()


def format_document_content(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format all chunks from a document organized by page.
    
    Returns:
        Dict with:
        - full_content: Combined text of all pages
        - pages: Dict mapping page_no to page content
        - total_pages: Number of pages
        - total_chunks: Total number of chunks
    """
    if not chunks:
        return {
            "full_content": "",
            "pages": {},
            "total_pages": 0,
            "total_chunks": 0,
            "chunks_by_page": {}
        }
    
    # Organize chunks by page
    pages_dict = {}
    chunks_by_page = {}
    
    for chunk in chunks:
        page_no = chunk.get("page_no", 0)
        chunk_text = chunk.get("payload", {}).get("chunk_text", "")
        chunk_idx = chunk.get("chunk_index", 0)
        
        if page_no not in pages_dict:
            pages_dict[page_no] = []
            chunks_by_page[page_no] = []
        
        pages_dict[page_no].append(chunk)
        chunks_by_page[page_no].append({
            "chunk_text": chunk_text,
            "chunk_index": chunk_idx,
            "id": str(chunk.get("id", ""))
        })
    
    # Format each page
    formatted_pages = {}
    all_texts = []
    
    for page_no in sorted(pages_dict.keys()):
        page_chunks = pages_dict[page_no]
        page_content = format_page_content(page_chunks)
        formatted_pages[page_no] = {
            "page_no": page_no,
            "content": page_content,
            "chunks": chunks_by_page[page_no],
            "chunk_count": len(page_chunks)
        }
        all_texts.append(f"[PAGE {page_no}]\n{page_content}")
    
    full_content = "\n\n".join(all_texts)
    
    return {
        "full_content": full_content,
        "pages": formatted_pages,
        "total_pages": len(formatted_pages),
        "total_chunks": len(chunks),
        "chunks_by_page": chunks_by_page
    }


def get_page_content_for_result(
    group_id: str,
    current_page_no: int,
    matched_chunk_text: str = None
) -> Dict[str, Any]:
    """
    Get previous, current, and next page content for a search result.
    
    Args:
        group_id: Document group ID
        current_page_no: Current page number
        matched_chunk_text: Optional text of the matched chunk to identify it
    
    Returns:
        Dict with previous_page, current_page, and next_page content
    """
    if not group_id or current_page_no is None:
        return {
            "previous_page": {
                "page_no": None,
                "content": "",
                "chunks": []
            },
            "current_page": {
                "page_no": current_page_no,
                "content": matched_chunk_text or "",
                "chunks": []
            },
            "next_page": {
                "page_no": None,
                "content": "",
                "chunks": []
            }
        }
    
    # Get chunks for previous, current, and next pages
    previous_page_no = current_page_no - 1 if current_page_no > 0 else None
    next_page_no = current_page_no + 1
    
    previous_chunks = []
    current_chunks = []
    next_chunks = []
    
    if previous_page_no is not None:
        previous_chunks = search_system.get_page_chunks(group_id, previous_page_no)
    
    current_chunks = search_system.get_page_chunks(group_id, current_page_no)
    next_chunks = search_system.get_page_chunks(group_id, next_page_no)
    
    # Identify matched chunk in current page by text similarity
    def identify_matched_chunk(chunks, matched_text):
        if not matched_text:
            return None
        matched_text_lower = matched_text.lower().strip()[:200]  # First 200 chars for matching
        for chunk in chunks:
            chunk_text = chunk.get("payload", {}).get("chunk_text", "")
            if matched_text_lower in chunk_text.lower()[:200]:
                return str(chunk.get("id", ""))
        return None
    
    matched_chunk_id = identify_matched_chunk(current_chunks, matched_chunk_text) if matched_chunk_text else None
    
    # Format page content
    return {
        "previous_page": {
            "page_no": previous_page_no,
            "content": format_page_content(previous_chunks),
            "chunks": [
                {
                    "chunk_text": chunk.get("payload", {}).get("chunk_text", ""),
                    "chunk_index": chunk.get("chunk_index", 0)
                }
                for chunk in previous_chunks
            ]
        },
        "current_page": {
            "page_no": current_page_no,
            "content": format_page_content(current_chunks),
            "chunks": [
                {
                    "chunk_text": chunk.get("payload", {}).get("chunk_text", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "is_matched": str(chunk.get("id", "")) == matched_chunk_id if matched_chunk_id else False
                }
                for chunk in current_chunks
            ]
        },
        "next_page": {
            "page_no": next_page_no if next_chunks else None,
            "content": format_page_content(next_chunks),
            "chunks": [
                {
                    "chunk_text": chunk.get("payload", {}).get("chunk_text", ""),
                    "chunk_index": chunk.get("chunk_index", 0)
                }
                for chunk in next_chunks
            ]
        }
    }


@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    Search API endpoint.
    
    GET/POST parameters:
        query (required): Search query string
    
    Returns:
        JSON with top 3 results. Each result includes:
        - tax_type: Tax type (e.g., "Central Tax" or "Central Tax (Rate)")
        - notification_no: Notification number (e.g., "05/2018")
        - page_no: Page number where the matched chunk was found (e.g., 5)
        - text: Combined content of target chunk ±1 chunks (chunks at index-1, index, and index+1)
    """
    if search_system is None:
        logger.error("Search system not initialized")
        return jsonify({
            "error": "Search system not initialized",
            "status": "service_unavailable"
        }), 503
    
    # Get query from request
    try:
        if request.method == 'GET':
            query = request.args.get('query', '').strip()
        else:
            data = request.get_json() or {}
            query = data.get('query', '').strip()
    except Exception as e:
        logger.warning(f"Error parsing request: {e}")
        return jsonify({
            "error": "Invalid request format",
            "status": "bad_request"
        }), 400
    
    if not query:
        return jsonify({
            "error": "Query parameter is required",
            "status": "bad_request"
        }), 400
    
    if len(query) > 1000:  # Reasonable limit
        return jsonify({
            "error": "Query too long (max 1000 characters)",
            "status": "bad_request"
        }), 400
    
    try:
        # Perform search with limit=3 to get top 3 results
        logger.info(f"Search request: query='{query[:50]}...'")
        search_result = search_system.search(query=query, limit=3)
        
        # Format results with page context
        formatted_results = []
        
        for idx, result_data in enumerate(search_result.get("results", [])):
            # Extract metadata from the formatted result
            # These values come directly from the matched chunk's payload
            group_id = result_data.get("group_id")
            page_no = result_data.get("page_no")  # Page number from matched chunk
            notification_no = result_data.get("notification_no")
            tax_type = result_data.get("tax_type")
            matched_chunk_text = result_data.get("matched_chunk_text", "")
            
            # Get the matched chunk index and page_no from context_metadata
            context_metadata = result_data.get("context_metadata", [])
            matched_chunk_index = None
            page_no_from_context = None
            for ctx in context_metadata:
                if ctx.get("is_matched", False):
                    matched_chunk_index = ctx.get("chunk_index")
                    page_no_from_context = ctx.get("page_no")
                    break
            
            # Use page_no from context_metadata if not in result_data
            if page_no is None and page_no_from_context is not None:
                page_no = page_no_from_context
                logger.debug(f"Got page_no {page_no} from context_metadata for result {idx}")
            
            # Log for debugging
            if page_no is not None:
                logger.debug(f"Result {idx}: {tax_type} {notification_no}, page {page_no}, chunk_index {matched_chunk_index}")
            else:
                logger.warning(f"Result {idx}: Missing page_no for {tax_type} {notification_no}")
            
            # Get chunks ±2 from matched chunk index
            context_chunks_list = []
            if matched_chunk_index is not None and group_id:
                context_chunks_list = get_chunks_by_index_range(
                    group_id=group_id,
                    matched_chunk_index=matched_chunk_index,
                    context_range=2
                )
            
            # Get page content (previous, current, next) - still needed for page_no
            page_content = get_page_content_for_result(
                group_id=group_id,
                current_page_no=page_no,
                matched_chunk_text=matched_chunk_text
            )
            
            # Ensure we have page_no - prioritize from result_data (matched chunk), then page_content
            final_page_no = page_no
            if final_page_no is None:
                final_page_no = page_content["current_page"].get("page_no")
            
            # Determine actual tax_type - check file_path to see if it's "Central Tax (Rate)" vs "Central Tax"
            tax_type_from_result = result_data.get("tax_type")
            file_path_from_result = result_data.get("file_path") or result_data.get("source_file_path") or ""
            
            # If file_path contains "(Rate)" and tax_type doesn't, update tax_type
            if file_path_from_result and tax_type_from_result and "(Rate)" not in tax_type_from_result:
                if tax_type_from_result == "Central Tax" and "(Rate)" in file_path_from_result:
                    tax_type_from_result = "Central Tax (Rate)"
                elif tax_type_from_result == "Integrated Tax" and "(Rate)" in file_path_from_result:
                    tax_type_from_result = "Integrated Tax (Rate)"
                elif tax_type_from_result == "Union Territory Tax" and "(Rate)" in file_path_from_result:
                    tax_type_from_result = "Union Territory Tax (Rate)"
            
            # Build result
            result = {
                "metadata": {
                    "notification_no": result_data.get("notification_no"),
                    "tax_type": tax_type_from_result,  # Preserve original tax_type (with Rate if applicable)
                    "issued_on": result_data.get("issued_on"),
                    "page_no": final_page_no,  # Page number from matched chunk
                    "file_path": file_path_from_result or result_data.get("file_path") or result_data.get("source_file_path"),
                    "group_id": result_data.get("group_id"),
                    "matched_chunk_index": matched_chunk_index  # Store for later use
                },
                "scores": result_data.get("scores", {}),
                "search_types": result_data.get("search_types", []),
                "matched_chunk_text": matched_chunk_text,
                "context_chunks": context_chunks_list,  # Chunks ±2 from matched
                "previous_page": page_content["previous_page"],
                "current_page": page_content["current_page"],
                "next_page": page_content["next_page"]
            }
            
            formatted_results.append(result)
        
        # Simplified response with only the page content that the chunk comes from
        simplified_results = []
        
        for idx, result in enumerate(formatted_results):
            # Get metadata and ensure page_no is accurate
            metadata = result["metadata"]
            scores = result.get("scores", {})
            context_chunks = result.get("context_chunks", [])
            current_page_data = result.get("current_page", {})
            group_id = metadata.get("group_id")
            matched_chunk_text = result.get("matched_chunk_text", "")
            matched_chunk_index = metadata.get("matched_chunk_index")
            
            # Get page_no - prioritize from metadata (which comes from matched chunk), then current_page
            page_no = metadata.get("page_no")
            if page_no is None:
                page_no = current_page_data.get("page_no")
            
            # If still None, log warning
            if page_no is None:
                logger.warning(f"Page number is None for result {idx}: {metadata.get('notification_no')}, {metadata.get('tax_type')}")
            
            # Get tax_type and notification_no from metadata (preserves original tax_type)
            tax_type = metadata.get("tax_type")
            notification_no = metadata.get("notification_no")
            
            # Get target chunk ±1 chunks from the SAME PAGE ONLY
            # Important: get_target_chunk_plus_minus_one uses search_system.get_page_chunks() 
            # which filters by both group_id AND page_no, ensuring we ONLY get chunks 
            # from that specific page, never from multiple pages
            context_text = ""
            if page_no is not None and group_id:
                context_text = get_target_chunk_plus_minus_one(
                    group_id=group_id,
                    page_no=page_no,
                    matched_chunk_text=matched_chunk_text,
                    target_chunk_index=matched_chunk_index
                )
            
            # Fallback: use matched chunk text only if we couldn't get context
            if not context_text:
                context_text = matched_chunk_text or ""
                if not group_id:
                    logger.warning(f"Could not get target chunk ±1: missing group_id")
                else:
                    logger.warning(f"Could not get target chunk ±1 for page {page_no}, using matched chunk text only")
            
            # Clean the text to remove \n and \u escape sequences
            cleaned_text = clean_text_for_output(context_text)
            
            simplified_results.append({
                "tax_type": tax_type,  # Accurate tax_type (includes "(Rate)" if applicable)
                "notification_no": notification_no,
                "page_no": page_no,  # Accurate page number
                "text": cleaned_text  # Target chunk ±1 chunks from the same page ONLY (cleaned)
            })
        
        # Ensure only top 3 results are returned
        limited_results = simplified_results[:3]
        
        response_data = {
            "query": query,
            "results": limited_results
        }
        
        logger.info(f"Search completed: {len(limited_results)} results")
        return jsonify(response_data)
        
    except ValueError as e:
        logger.warning(f"Invalid query: {e}")
        return jsonify({
            "error": str(e),
            "status": "bad_request"
        }), 400
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "internal_error",
            "message": str(e) if DEBUG_MODE else "An error occurred while processing your request"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = "healthy" if search_system is not None else "unhealthy"
    return jsonify({
        "status": status,
        "search_system_initialized": search_system is not None,
        "service": "search-api"
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "status": "not_found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "status": "internal_error"
    }), 500


if __name__ == '__main__':
    # Development server (use gunicorn for production)
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG_MODE,
        threaded=True
    )
