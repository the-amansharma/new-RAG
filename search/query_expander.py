"""
Query expansion and enhancement for deep/logical queries.
Improves semantic search for conceptual questions without specific notification numbers.
"""
import re
from typing import List, Dict, Set
from search.query_processor import QueryProcessor


class QueryExpander:
    """Expands and enhances queries for better semantic matching."""
    
    # Legal concept patterns that indicate deep queries
    DEEP_QUERY_INDICATORS = [
        r'\b(what|how|when|why|where|explain|describe|tell me about|what are|what is)\b',
        r'\b(conditions|requirements|procedures|process|steps|method|way)\b',
        r'\b(applicable|applies|applied|application|scope)\b',
        r'\b(eligible|eligibility|qualify|qualification)\b',
        r'\b(calculate|computation|formula|rate|percentage)\b',
        r'\b(difference|compare|versus|vs|between)\b',
        r'\b(impact|effect|consequence|result|outcome)\b',
        r'\b(rule|regulation|provision|section|clause)\b',
    ]
    
    # Legal domain terms that need expansion
    LEGAL_CONCEPTS = {
        "input tax credit": [
            "ITC", "input tax credit", "credit", "tax credit", 
            "input credit", "GST credit", "credit mechanism"
        ],
        "reverse charge": [
            "RCM", "reverse charge mechanism", "reverse charge", 
            "recipient liable", "service recipient"
        ],
        "composition scheme": [
            "composition", "composition levy", "composition dealer",
            "simplified scheme", "composition tax"
        ],
        "exemption": [
            "exempt", "exempted", "exempt supply", "exempted supply",
            "not taxable", "zero rated", "nil rated"
        ],
        "refund": [
            "refund claim", "refund", "refund process", "refund procedure",
            "claim refund", "refund application"
        ],
        "registration": [
            "GST registration", "registration", "register", "GSTIN",
            "registration process", "registration requirements"
        ],
        "return filing": [
            "return", "GSTR", "filing", "file return", "return filing",
            "GST return", "return submission"
        ],
        "invoice": [
            "tax invoice", "invoice", "bill", "bill of supply",
            "commercial invoice", "invoice format"
        ],
        "rate": [
            "tax rate", "GST rate", "rate", "percentage", "tax percentage",
            "applicable rate", "rate of tax"
        ],
        "LUT": [
            "letter of undertaking", "LUT", "bond", "bank guarantee",
            "BG", "undertaking"
        ],
    }
    
    def __init__(self):
        self.query_processor = QueryProcessor()
    
    def is_deep_query(self, query: str) -> bool:
        """
        Detect if query is a deep/logical query (conceptual, not specific notification lookup).
        
        Deep queries are:
        - Questions (what, how, when, why)
        - Conceptual queries (conditions, requirements, procedures)
        - Queries without specific notification numbers
        """
        query_lower = query.lower()
        
        # Check for notification number - if present, likely not a deep query
        if re.search(r'\b\d{1,3}\s*[/-]\s*\d{4}\b', query):
            return False
        
        # Check for deep query indicators
        for pattern in self.DEEP_QUERY_INDICATORS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        # Check if query is asking about concepts/procedures
        concept_indicators = [
            "how to", "how do", "what is", "what are", "explain",
            "procedure", "process", "steps", "requirements", "conditions"
        ]
        
        if any(indicator in query_lower for indicator in concept_indicators):
            return True
        
        return False
    
    def expand_legal_concepts(self, query: str) -> Set[str]:
        """Expand query with related legal concepts and synonyms."""
        query_lower = query.lower()
        expanded_terms = set()
        
        # Add original query terms
        words = re.findall(r'\b\w+\b', query_lower)
        expanded_terms.update(words)
        
        # Find and expand legal concepts
        for concept, synonyms in self.LEGAL_CONCEPTS.items():
            if concept in query_lower:
                expanded_terms.update(synonyms)
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    expanded_terms.update([concept] + synonyms)
        
        return expanded_terms
    
    def create_query_variations(self, query: str) -> List[str]:
        """
        Create multiple query variations for better semantic coverage.
        Useful for deep queries where exact wording matters less.
        """
        variations = [query]  # Original query
        
        query_lower = query.lower()
        
        # Variation 1: Add "GST" context if not present
        if "gst" not in query_lower and "tax" not in query_lower:
            variations.append(f"GST {query}")
        
        # Variation 2: Rephrase questions
        if query_lower.startswith("what"):
            variations.append(query.replace("what", "how", 1))
            variations.append(query.replace("what", "explain", 1))
        elif query_lower.startswith("how"):
            variations.append(query.replace("how", "what", 1))
            variations.append(query.replace("how", "explain", 1))
        
        # Variation 3: Add "notification" or "provision" context
        if "notification" not in query_lower and "provision" not in query_lower:
            variations.append(f"{query} notification")
            variations.append(f"{query} provision")
        
        # Variation 4: Expand key terms
        expanded_terms = self.expand_legal_concepts(query)
        if expanded_terms:
            # Create variation with expanded terms
            key_terms = list(expanded_terms)[:3]  # Top 3 expanded terms
            if key_terms:
                variations.append(f"{query} {' '.join(key_terms)}")
        
        return variations[:5]  # Limit to 5 variations
    
    def enhance_for_semantic_search(self, query: str) -> str:
        """
        Enhance query specifically for semantic search.
        Adds context and legal terminology.
        Only enhances if query doesn't have notification number.
        """
        query_lower = query.lower()
        
        # Don't enhance if query has notification number
        if re.search(r'\b\d{1,3}\s*[/-]\s*\d{4}\b', query):
            return query  # Return original for specific queries
        
        enhanced = query
        
        # Add GST context if missing
        if "gst" not in query_lower and "tax" not in query_lower:
            enhanced = f"GST {enhanced}"
        
        # Add "notification" or "provision" context for deep queries only
        if self.is_deep_query(query):
            if "notification" not in query_lower:
                enhanced = f"{enhanced} in GST notification"
            if "provision" not in query_lower and "condition" not in query_lower:
                enhanced = f"{enhanced} provision"
        
        return enhanced
    
    def detect_query_type(self, query: str) -> Dict[str, bool]:
        """
        Detect the type of query to determine search strategy.
        
        Returns:
            {
                "is_deep_query": bool,
                "is_specific_query": bool,  # Has notification number
                "is_conceptual": bool,  # Asks about concepts
                "is_procedural": bool  # Asks about procedures
            }
        """
        query_lower = query.lower()
        
        has_notification = bool(re.search(r'\b\d{1,3}\s*[/-]\s*\d{4}\b', query))
        is_deep = self.is_deep_query(query)
        is_conceptual = any(
            term in query_lower 
            for term in ["what is", "what are", "explain", "define", "meaning"]
        )
        is_procedural = any(
            term in query_lower 
            for term in ["how to", "how do", "procedure", "process", "steps", "method"]
        )
        
        return {
            "is_deep_query": is_deep,
            "is_specific_query": has_notification,
            "is_conceptual": is_conceptual,
            "is_procedural": is_procedural
        }
    
    def expand_query(self, query: str, query_info: Dict = None) -> List[str]:
        """
        Main method to expand query for better search results.
        Returns list of query variations to search.
        """
        if query_info is None:
            query_info = self.query_processor.process_query(query)
        
        # If query has notification number, use as-is (specific lookup)
        if query_info["entities"].get("notification_numbers"):
            return [query]
        
        # For deep queries, create variations
        if self.is_deep_query(query):
            variations = self.create_query_variations(query)
            # Also add enhanced version
            enhanced = self.enhance_for_semantic_search(query)
            if enhanced not in variations:
                variations.append(enhanced)
            return variations
        
        # For other queries, return original + enhanced
        enhanced = self.enhance_for_semantic_search(query)
        if enhanced != query:
            return [query, enhanced]
        
        return [query]
