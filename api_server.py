"""
Production-ready Flask API server for search functionality.
Returns top 3 results with full page context.
For the top result, includes the complete document content.
"""
import os
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
        JSON with top 3 results. The top result includes:
        - full_document: Complete document content (all pages)
        - previous_page, current_page, next_page: Page context
        - metadata: Document metadata
        - scores: Search scores
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
            group_id = result_data.get("group_id")
            page_no = result_data.get("page_no")
            matched_chunk_text = result_data.get("matched_chunk_text", "")
            
            # Get page content (previous, current, next)
            page_content = get_page_content_for_result(
                group_id=group_id,
                current_page_no=page_no,
                matched_chunk_text=matched_chunk_text
            )
            
            # Build result
            result = {
                "metadata": {
                    "notification_no": result_data.get("notification_no"),
                    "tax_type": result_data.get("tax_type"),
                    "issued_on": result_data.get("issued_on"),
                    "page_no": result_data.get("page_no"),
                    "file_path": result_data.get("file_path"),
                    "group_id": result_data.get("group_id")
                },
                "scores": result_data.get("scores", {}),
                "search_types": result_data.get("search_types", []),
                "matched_chunk_text": matched_chunk_text,
                "previous_page": page_content["previous_page"],
                "current_page": page_content["current_page"],
                "next_page": page_content["next_page"]
            }
            
            # For the top result (index 0), add full document content
            if idx == 0 and group_id:
                try:
                    logger.info(f"Fetching full document for top result: {group_id}")
                    all_document_chunks = search_system.get_document_chunks(group_id)
                    document_content = format_document_content(all_document_chunks)
                    result["full_document"] = document_content
                    logger.info(f"Full document retrieved: {document_content['total_pages']} pages, {document_content['total_chunks']} chunks")
                except Exception as e:
                    logger.error(f"Error fetching full document: {e}")
                    result["full_document"] = {
                        "full_content": "",
                        "pages": {},
                        "total_pages": 0,
                        "total_chunks": 0,
                        "error": "Failed to retrieve full document"
                    }
            
            formatted_results.append(result)
        
        # Simplified response with only main text content
        simplified_results = []
        
        for idx, result in enumerate(formatted_results):
            # For top result, include full document content
            if idx == 0 and result.get("full_document"):
                full_doc = result["full_document"]
                simplified_results.append({
                    "text": full_doc.get("full_content", ""),
                    "notification_no": result["metadata"].get("notification_no"),
                    "tax_type": result["metadata"].get("tax_type"),
                    "page_no": result["metadata"].get("page_no")
                })
            else:
                # For other results, combine current page content
                current_page_content = result.get("current_page", {}).get("content", "")
                simplified_results.append({
                    "text": current_page_content,
                    "notification_no": result["metadata"].get("notification_no"),
                    "tax_type": result["metadata"].get("tax_type"),
                    "page_no": result["metadata"].get("page_no")
                })
        
        response_data = {
            "query": query,
            "results": simplified_results
        }
        
        logger.info(f"Search completed: {len(simplified_results)} results")
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
