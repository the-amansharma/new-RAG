"""
Hybrid Search System combining semantic and keyword search for legal accuracy.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchText, MatchAny,
    Distance, VectorParams
)
from dotenv import load_dotenv

from ingestion.embeddings import embed_text
from search.query_processor import QueryProcessor
from search.keyword_scorer import BM25Scorer
from search.query_expander import QueryExpander

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "notification_chunks"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Search parameters
DEFAULT_SEMANTIC_LIMIT = 50  # Get more candidates for reranking
DEFAULT_KEYWORD_LIMIT = 50
FINAL_RESULT_LIMIT = 10  # Final results after reranking

# Score weights for hybrid search (default)
DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_KEYWORD_WEIGHT = 0.4

# Dynamic weights for deep queries (no notification number)
DEEP_QUERY_SEMANTIC_WEIGHT = 0.75  # More weight on semantic for deep queries
DEEP_QUERY_KEYWORD_WEIGHT = 0.25


class HybridSearch:
    """Hybrid search combining semantic similarity and keyword matching."""
    
    def __init__(self):
        """Initialize search system with Qdrant client."""
        if not QDRANT_URL:
            raise ValueError("QDRANT_URL not set in environment variables")
        
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        self.query_processor = QueryProcessor()
        self.bm25_scorer = BM25Scorer()
        self.query_expander = QueryExpander()
        
        # Verify collection exists
        collections = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME not in collections:
            raise ValueError(f"Collection '{COLLECTION_NAME}' not found in Qdrant. Run ingestion pipeline first.")
        
        logger.info(f"✅ Hybrid Search initialized with collection: {COLLECTION_NAME}")
    
    def _build_metadata_filter(self, entities: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter from extracted entities."""
        conditions = []
        
        # Filter by tax type
        if entities.get("tax_types"):
            conditions.append(
                FieldCondition(
                    key="tax_type",
                    match=MatchAny(any=entities["tax_types"])
                )
            )
        
        # Filter by notification number
        if entities.get("notification_numbers"):
            conditions.append(
                FieldCondition(
                    key="notification_no",
                    match=MatchAny(any=entities["notification_numbers"])
                )
            )
        
        # Filter by year (from issued_on)
        if entities.get("years"):
            # Note: This is a simplified filter. For exact date filtering, 
            # you'd need to parse issued_on and filter properly
            # For now, we'll use keyword search for year matching
            pass
        
        if not conditions:
            return None
        
        # Combine with AND logic
        if len(conditions) == 1:
            return Filter(must=conditions)
        else:
            return Filter(must=conditions)
    
    def semantic_search(
        self, 
        query: str, 
        limit: int = DEFAULT_SEMANTIC_LIMIT,
        filter_condition: Optional[Filter] = None,
        enhanced_queries: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        Can use multiple enhanced queries for better coverage.
        """
        try:
            # Use enhanced queries if provided, otherwise use original
            queries_to_search = enhanced_queries if enhanced_queries else [query]
            
            all_results = {}
            
            for search_query in queries_to_search:
                # Generate query embedding
                query_vector = embed_text(search_query)
                
                # Use query_points() - the current API for qdrant-client >= 1.9.0
                try:
                    search_results = self.client.query_points(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_vector,
                        query_filter=filter_condition,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # query_points returns QueryResponse with points attribute
                    if hasattr(search_results, 'points'):
                        results_list = search_results.points
                    else:
                        # If it's already a list, use it directly
                        results_list = search_results
                        
                except Exception as e:
                    logger.error(f"Error in query_points: {e}")
                    continue
                
                # Combine results (take best score for each point)
                for result in results_list:
                    point_id = str(result.id)
                    score = result.score
                    
                    if point_id not in all_results:
                        all_results[point_id] = {
                            "id": result.id,
                            "score": score,
                            "payload": result.payload,
                            "search_type": "semantic",
                            "query_count": 1
                        }
                    else:
                        # Take the best score from multiple queries
                        if score > all_results[point_id]["score"]:
                            all_results[point_id]["score"] = score
                        all_results[point_id]["query_count"] += 1
            
            # Convert to list and sort by score
            results = list(all_results.values())
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Boost results found by multiple queries
            for result in results:
                if result["query_count"] > 1:
                    result["score"] *= (1 + 0.1 * (result["query_count"] - 1))
            
            # Re-sort after boosting
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(
        self,
        keywords: List[str],
        limit: int = DEFAULT_KEYWORD_LIMIT,
        filter_condition: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using Qdrant's text matching."""
        if not keywords:
            return []
        
        try:
            # Skip text index filters - use in-memory filtering instead
            # This avoids requiring text indexes on chunk_text
            results = []
            candidate_docs = []
            
            # Try to use metadata filter only (if it doesn't require text index)
            metadata_only_filter = None
            if filter_condition:
                # Check if filter only uses metadata fields (not chunk_text)
                try:
                    # Try to use metadata filter
                    metadata_only_filter = filter_condition
                except:
                    metadata_only_filter = None
            
            # Scroll to get candidate documents
            try:
                scroll_result = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=metadata_only_filter,
                    limit=limit * 10,  # Get more for in-memory filtering
                    with_payload=True,
                    with_vectors=False
                )
            except Exception as e:
                # If filter fails, scroll without filter
                logger.debug(f"Scroll with filter failed, using no filter: {e}")
                try:
                    scroll_result = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=limit * 10,
                        with_payload=True,
                        with_vectors=False
                    )
                except Exception as e2:
                    logger.error(f"Scroll failed: {e2}")
                    return []
            
            # Filter in memory by keywords and metadata
            for point in scroll_result[0]:
                chunk_text = point.payload.get("chunk_text", "").lower()
                
                # Check if any keyword matches in text
                if not any(kw.lower() in chunk_text for kw in keywords):
                    continue
                
                # Apply metadata filter if exists
                if filter_condition:
                    tax_type = point.payload.get("tax_type")
                    notification_no = point.payload.get("notification_no")
                    
                    # Check if matches filter
                    matches_filter = True
                    if hasattr(filter_condition, 'must'):
                        for condition in filter_condition.must:
                            if hasattr(condition, 'key'):
                                if condition.key == "tax_type" and tax_type:
                                    if hasattr(condition.match, 'any'):
                                        if tax_type not in condition.match.any:
                                            matches_filter = False
                                            break
                                elif condition.key == "notification_no" and notification_no:
                                    if hasattr(condition.match, 'any'):
                                        if notification_no not in condition.match.any:
                                            matches_filter = False
                                            break
                    
                    if not matches_filter:
                        continue
                
                candidate_docs.append({
                    "id": point.id,
                    "chunk_text": point.payload.get("chunk_text", ""),
                    "payload": point.payload
                })
            
            if not candidate_docs:
                return []
            
            # Score with BM25
            scored_docs = self.bm25_scorer.score_batch(
                candidate_docs,
                keywords,
                text_key="chunk_text"
            )
            
            # Normalize BM25 scores to 0-1 range
            if scored_docs:
                max_score = max(doc["bm25_score"] for doc in scored_docs)
                min_score = min(doc["bm25_score"] for doc in scored_docs)
                score_range = max_score - min_score if max_score != min_score else 1
                
                for doc in scored_docs:
                    if doc["bm25_score"] > 0:  # Only include documents with matches
                        normalized_score = (doc["bm25_score"] - min_score) / score_range if score_range > 0 else doc["bm25_score"]
                        
                        # Boost for exact phrase matches
                        chunk_text_lower = doc["chunk_text"].lower()
                        query_phrase = " ".join(keywords).lower()
                        if query_phrase in chunk_text_lower:
                            normalized_score = min(1.0, normalized_score * 1.3)
                        
                        results.append({
                            "id": doc["id"],
                            "score": normalized_score,
                            "payload": doc["payload"],
                            "search_type": "keyword",
                            "bm25_score": doc["bm25_score"]
                        })
            
            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        is_deep_query: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Combine semantic and keyword results with deduplication."""
        combined = {}
        
        # Normalize semantic scores (cosine similarity is 0-1, but we'll normalize to 0-1)
        if semantic_results:
            max_semantic = max(r["score"] for r in semantic_results)
            min_semantic = min(r["score"] for r in semantic_results)
            semantic_range = max_semantic - min_semantic if max_semantic != min_semantic else 1
        
        # Normalize keyword scores (already 0-1, but ensure consistency)
        if keyword_results:
            max_keyword = max(r["score"] for r in keyword_results)
            min_keyword = min(r["score"] for r in keyword_results)
            keyword_range = max_keyword - min_keyword if max_keyword != min_keyword else 1
        
        # Add semantic results
        for result in semantic_results:
            point_id = str(result["id"])
            normalized_score = (result["score"] - min_semantic) / semantic_range if semantic_range > 0 else result["score"]
            
            if point_id not in combined:
                combined[point_id] = {
                    "id": point_id,
                    "payload": result["payload"],
                    "semantic_score": normalized_score,
                    "keyword_score": 0.0,
                    "combined_score": 0.0,
                    "search_types": []
                }
            
            combined[point_id]["semantic_score"] = normalized_score
            combined[point_id]["search_types"].append("semantic")
        
        # Add keyword results
        for result in keyword_results:
            point_id = str(result["id"])
            normalized_score = (result["score"] - min_keyword) / keyword_range if keyword_range > 0 else result["score"]
            
            if point_id not in combined:
                combined[point_id] = {
                    "id": point_id,
                    "payload": result["payload"],
                    "semantic_score": 0.0,
                    "keyword_score": normalized_score,
                    "combined_score": 0.0,
                    "search_types": []
                }
            
            combined[point_id]["keyword_score"] = max(combined[point_id]["keyword_score"], normalized_score)
            combined[point_id]["search_types"].append("keyword")
        
        # Calculate combined scores with dynamic weights
        # Use dynamic weights based on query type
        semantic_weight = DEEP_QUERY_SEMANTIC_WEIGHT if is_deep_query else DEFAULT_SEMANTIC_WEIGHT
        keyword_weight = DEEP_QUERY_KEYWORD_WEIGHT if is_deep_query else DEFAULT_KEYWORD_WEIGHT
        
        for point_id, result in combined.items():
            semantic = result["semantic_score"]
            keyword = result["keyword_score"]
            
            # Weighted combination with dynamic weights
            combined_score = (semantic_weight * semantic) + (keyword_weight * keyword)
            
            # Boost if found in both searches
            if len(result["search_types"]) > 1:
                combined_score *= 1.2
            
            result["combined_score"] = combined_score
            result["is_deep_query"] = is_deep_query
        
        return combined
    
    def get_context_chunks(
        self,
        matched_chunk: Dict[str, Any],
        context_before: int = 2,
        context_after: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get context chunks (before and after) for a matched chunk.
        
        Args:
            matched_chunk: The matched chunk result with payload
            context_before: Number of chunks before to fetch
            context_after: Number of chunks after to fetch
        
        Returns:
            List of context chunks including the matched chunk, ordered by chunk_index
        """
        payload = matched_chunk.get("payload", {})
        group_id = payload.get("group_id")
        chunk_index = payload.get("chunk_index")
        global_chunk_index = payload.get("global_chunk_index")
        
        if not group_id:
            # If no group_id, return just the matched chunk
            return [matched_chunk]
        
        # Determine which index to use
        use_index = chunk_index if chunk_index is not None else global_chunk_index
        if use_index is None:
            return [matched_chunk]
        
        # Calculate index range
        start_index = max(0, use_index - context_before)
        end_index = use_index + context_after + 1
        
        # Build filter for same group_id and index range
        conditions = [
            FieldCondition(
                key="group_id",
                match=MatchValue(value=group_id)
            )
        ]
        
        # Avoid using group_id filter (requires keyword index)
        # Instead, scroll and filter in memory by group_id and index
        try:
            # Scroll to get chunks (without filter to avoid index requirement)
            scroll_result = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=5000,  # Get enough chunks to find context
                with_payload=True,
                with_vectors=False
            )
            
            # Collect and filter chunks by group_id and index range in memory
            context_chunks = []
            matched_chunk_id = str(matched_chunk.get("id", ""))
            
            for point in scroll_result[0]:
                # Filter by group_id in memory
                point_group_id = point.payload.get("group_id")
                if point_group_id != group_id:
                    continue
                
                chunk_idx = point.payload.get("chunk_index")
                if chunk_idx is None:
                    chunk_idx = point.payload.get("global_chunk_index")
                
                if chunk_idx is None:
                    continue
                
                # Check if chunk is in the desired range
                if start_index <= chunk_idx < end_index:
                    context_chunks.append({
                        "id": point.id,
                        "payload": point.payload,
                        "chunk_index": chunk_idx
                    })
                # Also include the matched chunk if we find it
                elif str(point.id) == matched_chunk_id:
                    context_chunks.append({
                        "id": point.id,
                        "payload": point.payload,
                        "chunk_index": chunk_idx
                    })
            
            # Sort by chunk_index
            context_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Ensure the matched chunk is included
            matched_found = any(
                str(chunk["id"]) == matched_chunk_id
                for chunk in context_chunks
            )
            
            if not matched_found:
                # Add matched chunk if not found
                matched_chunk_copy = matched_chunk.copy()
                matched_chunk_copy["chunk_index"] = use_index
                context_chunks.append(matched_chunk_copy)
                context_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Limit to the desired context range around the matched chunk
            # Find matched chunk position
            matched_pos = None
            for idx, chunk in enumerate(context_chunks):
                if str(chunk["id"]) == matched_chunk_id:
                    matched_pos = idx
                    break
            
            if matched_pos is not None:
                # Get context_before chunks before and context_after chunks after
                start_pos = max(0, matched_pos - context_before)
                end_pos = min(len(context_chunks), matched_pos + context_after + 1)
                context_chunks = context_chunks[start_pos:end_pos]
            
            return context_chunks
            
        except Exception as e:
            logger.warning(f"Error fetching context chunks: {e}, returning matched chunk only")
            return [matched_chunk]
    
    def get_page_chunks(
        self,
        group_id: str,
        page_no: int
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific page number within a document group.
        
        Args:
            group_id: The document group ID
            page_no: The page number to retrieve
        
        Returns:
            List of chunks for that page, ordered by chunk_index
        """
        try:
            # Scroll to get chunks and filter in memory
            scroll_result = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,  # Get enough chunks to find all pages
                with_payload=True,
                with_vectors=False
            )
            
            page_chunks = []
            
            for point in scroll_result[0]:
                # Filter by group_id and page_no in memory
                point_group_id = point.payload.get("group_id")
                point_page_no = point.payload.get("page_no")
                
                if point_group_id == group_id and point_page_no == page_no:
                    chunk_idx = point.payload.get("chunk_index")
                    if chunk_idx is None:
                        chunk_idx = point.payload.get("global_chunk_index", 0)
                    
                    page_chunks.append({
                        "id": point.id,
                        "payload": point.payload,
                        "chunk_index": chunk_idx
                    })
            
            # Sort by chunk_index
            page_chunks.sort(key=lambda x: x["chunk_index"])
            
            return page_chunks
            
        except Exception as e:
            logger.warning(f"Error fetching page chunks for page {page_no} in {group_id}: {e}")
            return []
    
    def get_document_chunks(
        self,
        group_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for an entire document (all pages).
        
        Args:
            group_id: The document group ID
        
        Returns:
            List of all chunks for the document, ordered by page_no and chunk_index
        """
        try:
            # Scroll to get chunks and filter in memory
            scroll_result = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,  # Get enough chunks to find all pages
                with_payload=True,
                with_vectors=False
            )
            
            document_chunks = []
            
            for point in scroll_result[0]:
                # Filter by group_id in memory
                point_group_id = point.payload.get("group_id")
                
                if point_group_id == group_id:
                    page_no = point.payload.get("page_no", 0)
                    chunk_idx = point.payload.get("chunk_index")
                    if chunk_idx is None:
                        chunk_idx = point.payload.get("global_chunk_index", 0)
                    
                    document_chunks.append({
                        "id": point.id,
                        "payload": point.payload,
                        "page_no": page_no,
                        "chunk_index": chunk_idx
                    })
            
            # Sort by page_no first, then chunk_index
            document_chunks.sort(key=lambda x: (x["page_no"], x["chunk_index"]))
            
            return document_chunks
            
        except Exception as e:
            logger.warning(f"Error fetching document chunks for {group_id}: {e}")
            return []
    
    def legal_accuracy_rerank(
        self,
        results: List[Dict[str, Any]],
        query_info: Dict[str, Any],
        is_deep_query: bool = False
    ) -> List[Dict[str, Any]]:
        """Rerank results for legal accuracy."""
        entities = query_info["entities"]
        
        for result in results:
            payload = result["payload"]
            legal_score = 0.0
            
            # Boost for exact notification number match
            if entities.get("notification_numbers"):
                if payload.get("notification_no") in entities["notification_numbers"]:
                    legal_score += 0.3
            elif is_deep_query:
                # For deep queries without notification numbers, boost semantic relevance
                # This is already handled by semantic score, but we can add small boost
                pass
            
            # Boost for tax type match
            if entities.get("tax_types"):
                if payload.get("tax_type") in entities["tax_types"]:
                    legal_score += 0.2
            
            # Boost for year match (from issued_on)
            if entities.get("years"):
                issued_on = payload.get("issued_on", "")
                if any(year in str(issued_on) for year in entities["years"]):
                    legal_score += 0.1
            
            # Boost for original documents over amendments/corrigendums
            doc_nature = payload.get("document_nature", "").lower()
            if doc_nature == "original":
                legal_score += 0.1
            elif doc_nature in ["amendment", "corrigendum"]:
                legal_score += 0.05  # Still relevant but less authoritative
            
            # Boost for page starts (often contain key information)
            if payload.get("is_page_start"):
                legal_score += 0.05
            
            # For deep queries, boost results with higher semantic scores
            if is_deep_query:
                semantic_score = result.get("semantic_score", 0)
                # Additional boost for high semantic relevance
                if semantic_score > 0.7:
                    legal_score += 0.15
                elif semantic_score > 0.5:
                    legal_score += 0.1
            
            # Apply legal score boost
            result["legal_score"] = legal_score
            result["final_score"] = result["combined_score"] * (1 + legal_score)
        
        # Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    
    def search(
        self,
        query: str,
        limit: int = FINAL_RESULT_LIMIT,
        tax_type: Optional[str] = None,
        notification_no: Optional[str] = None,
        year: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid search with legal accuracy.
        
        Args:
            query: User query string
            limit: Number of results to return
            tax_type: Optional filter by tax type
            notification_no: Optional filter by notification number
            year: Optional filter by year
        
        Returns:
            {
                "query": str,
                "query_info": Dict with processed query info,
                "results": List of search results with scores and metadata,
                "stats": Dict with search statistics
            }
        """
        # Process query
        query_info = self.query_processor.process_query(query)
        
        # Override entities if filters provided (do this FIRST)
        if tax_type:
            query_info["entities"]["tax_types"] = [tax_type]
        if notification_no:
            query_info["entities"]["notification_numbers"] = [notification_no]
        if year:
            query_info["entities"]["years"] = [year]
        
        # Check if query has notification number (from query or filter)
        has_notification = bool(
            query_info["entities"].get("notification_numbers") or 
            notification_no
        )
        
        # Detect query type for deep/logical queries (only if no notification)
        is_deep_query = False
        enhanced_query = query  # Default to original query
        expanded_queries = None
        
        if not has_notification:
            # Only enhance/expand if no notification number
            query_type_info = self.query_expander.detect_query_type(query)
            is_deep_query = query_type_info["is_deep_query"] and not query_type_info["is_specific_query"]
            
            if is_deep_query:
                # Get expanded queries for better semantic coverage
                expanded_queries = self.query_expander.expand_query(query, query_info)
                enhanced_query = self.query_expander.enhance_for_semantic_search(query)
                logger.info(f"🔍 Deep query detected. Using {len(expanded_queries)} query variations")
            else:
                # For non-deep queries without notification, still enhance slightly
                enhanced_query = self.query_expander.enhance_for_semantic_search(query)
                if enhanced_query != query:
                    expanded_queries = [query, enhanced_query]
        else:
            # Has notification number - use original query, no enhancement
            logger.info(f"🔍 Specific query detected (notification number present). Using original query.")
        
        # Build metadata filter
        metadata_filter = self._build_metadata_filter(query_info["entities"])
        
        # Perform semantic search
        logger.info(f"🔍 Performing semantic search for: {query[:50]}...")
        semantic_results = self.semantic_search(
            query=enhanced_query,
            limit=DEFAULT_SEMANTIC_LIMIT,
            filter_condition=metadata_filter,
            enhanced_queries=expanded_queries
        )
        
        # Perform keyword search
        logger.info(f"🔍 Performing keyword search with {len(query_info['keywords'])} keywords...")
        keyword_results = self.keyword_search(
            keywords=query_info["keywords"],
            limit=DEFAULT_KEYWORD_LIMIT,
            filter_condition=metadata_filter
        )
        
        # Combine results (pass is_deep_query flag)
        combined = self.combine_results(semantic_results, keyword_results, is_deep_query=is_deep_query)
        
        # Convert to list and rerank
        results_list = list(combined.values())
        results_list = self.legal_accuracy_rerank(results_list, query_info, is_deep_query=is_deep_query)
        
        # Limit results
        final_results = results_list[:limit]
        
        # Format results for output with context chunks
        formatted_results = []
        for result in final_results:
            # Get context chunks (2 before + matched + 2 after)
            context_chunks = self.get_context_chunks(result, context_before=2, context_after=2)
            
            # Find the matched chunk index in context
            matched_chunk_id = str(result.get("id", ""))
            matched_index = None
            for idx, chunk in enumerate(context_chunks):
                if str(chunk.get("id", "")) == matched_chunk_id:
                    matched_index = idx
                    break
            
            # Combine context chunks into a single result
            context_texts = []
            context_metadata = []
            
            for chunk in context_chunks:
                chunk_payload = chunk.get("payload", {})
                context_texts.append(chunk_payload.get("chunk_text", ""))
                context_metadata.append({
                    "chunk_index": chunk_payload.get("chunk_index") or chunk_payload.get("global_chunk_index"),
                    "page_no": chunk_payload.get("page_no"),
                    "is_matched": str(chunk.get("id", "")) == matched_chunk_id
                })
            
            # Combine all context text
            combined_text = "\n\n".join(context_texts)
            
            # Use metadata from the matched chunk
            matched_payload = result["payload"]
            
            formatted_results.append({
                "chunk_text": combined_text,  # Combined text from all context chunks
                "matched_chunk_text": matched_payload.get("chunk_text", ""),  # Just the matched chunk
                "matched_chunk_index": matched_index,  # Position of matched chunk in context
                "context_chunks_count": len(context_chunks),  # Total chunks in context
                "notification_no": matched_payload.get("notification_no"),
                "tax_type": matched_payload.get("tax_type"),
                "issued_on": matched_payload.get("issued_on"),
                "page_no": matched_payload.get("page_no"),
                "file_path": matched_payload.get("file_path") or matched_payload.get("source_file_path"),
                "group_id": matched_payload.get("group_id"),
                "context_metadata": context_metadata,  # Metadata for each chunk in context
                "scores": {
                    "semantic": result.get("semantic_score", 0),
                    "keyword": result.get("keyword_score", 0),
                    "combined": result.get("combined_score", 0),
                    "legal": result.get("legal_score", 0),
                    "final": result.get("final_score", 0)
                },
                "search_types": result.get("search_types", [])
            })
        
        return {
            "query": query,
            "query_info": query_info,
            "results": formatted_results,
            "stats": {
                "semantic_results": len(semantic_results),
                "keyword_results": len(keyword_results),
                "combined_results": len(combined),
                "final_results": len(formatted_results)
            }
        }

