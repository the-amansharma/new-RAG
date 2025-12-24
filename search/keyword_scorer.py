"""
BM25-like keyword scoring for improved keyword search accuracy.
"""
import re
from typing import List, Dict
from collections import Counter
import math


class BM25Scorer:
    """Simple BM25-like scorer for keyword matching."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def calculate_idf(self, term: str, document_freq: int, total_docs: int) -> float:
        """Calculate inverse document frequency."""
        if document_freq == 0:
            return 0.0
        return math.log((total_docs - document_freq + 0.5) / (document_freq + 0.5) + 1.0)
    
    def score_document(
        self,
        document_text: str,
        query_terms: List[str],
        avg_doc_length: float = 1000.0,
        doc_frequencies: Dict[str, int] = None,
        total_docs: int = 1000
    ) -> float:
        """
        Calculate BM25 score for a document against query terms.
        
        Args:
            document_text: Document text to score
            query_terms: List of query terms
            avg_doc_length: Average document length (for normalization)
            doc_frequencies: Term document frequencies (optional, for IDF)
            total_docs: Total number of documents (for IDF)
        
        Returns:
            BM25 score
        """
        doc_tokens = self.tokenize(document_text)
        doc_length = len(doc_tokens)
        doc_term_freq = Counter(doc_tokens)
        
        score = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            term_freq = doc_term_freq.get(term_lower, 0)
            
            if term_freq == 0:
                continue
            
            # Calculate IDF (simplified - can be improved with actual doc frequencies)
            if doc_frequencies:
                doc_freq = doc_frequencies.get(term_lower, 0)
                idf = self.calculate_idf(term_lower, doc_freq, total_docs)
            else:
                # Simple IDF approximation
                idf = math.log((total_docs + 1) / (term_freq + 1) + 1.0)
            
            # BM25 term score
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
            
            score += numerator / denominator
        
        return score
    
    def score_batch(
        self,
        documents: List[Dict[str, str]],
        query_terms: List[str],
        text_key: str = "chunk_text"
    ) -> List[Dict]:
        """
        Score a batch of documents.
        
        Args:
            documents: List of document dicts with text
            query_terms: Query terms
            text_key: Key in document dict containing text
        
        Returns:
            List of documents with 'bm25_score' added
        """
        # Calculate average document length
        doc_lengths = [len(self.tokenize(doc.get(text_key, ""))) for doc in documents]
        avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1000.0
        
        # Score each document
        scored_docs = []
        for doc in documents:
            text = doc.get(text_key, "")
            score = self.score_document(text, query_terms, avg_length)
            
            doc_copy = doc.copy()
            doc_copy["bm25_score"] = score
            scored_docs.append(doc_copy)
        
        # Sort by score
        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        
        return scored_docs

