"""
Query processing and legal term extraction for tax notification search.
Enhances queries with legal terminology and extracts key entities.
"""
import re
from typing import List, Dict, Set, Tuple
from collections import Counter

# Indian Tax Legal Terminology Dictionary
LEGAL_TERMS = {
    # Tax Types
    "gst": ["GST", "Goods and Services Tax", "CGST", "SGST", "IGST", "UTGST"],
    "cgst": ["Central GST", "Central Tax", "CGST"],
    "sgst": ["State GST", "State Tax", "SGST"],
    "igst": ["Integrated GST", "Integrated Tax", "IGST"],
    "utgst": ["Union Territory GST", "Union Territory Tax", "UTGST"],
    "cess": ["Compensation Cess", "Cess"],
    
    # Common Tax Terms
    "input tax credit": ["ITC", "Input Tax Credit", "credit"],
    "reverse charge": ["RCM", "Reverse Charge Mechanism"],
    "composition": ["Composition Scheme", "Composition Levy"],
    "exempt": ["Exemption", "Exempted", "Exempt Supply"],
    "nil rated": ["Nil Rated", "Nil Rate", "0%"],
    "zero rated": ["Zero Rated", "Zero Rate"],
    "export": ["Export of Goods", "Export of Services", "Export"],
    "import": ["Import of Goods", "Import of Services", "Import"],
    "lut": ["Letter of Undertaking", "LUT"],
    "bond": ["Bond", "Bank Guarantee", "BG"],
    "refund": ["Refund", "Refund Claim"],
    "assessment": ["Assessment", "Assessment Order"],
    "appeal": ["Appeal", "Appellate Authority"],
    "penalty": ["Penalty", "Penal Interest"],
    "interest": ["Interest", "Late Fee"],
    
    # Notification Types
    "notification": ["Notification", "Notfn", "Ntfn"],
    "circular": ["Circular", "Clarification"],
    "order": ["Order", "F. No."],
    "corrigendum": ["Corrigendum", "Correction"],
    "amendment": ["Amendment", "Substitution", "Omission", "Insertion"],
    
    # Rate Related
    "rate": ["Rate", "Tax Rate", "GST Rate"],
    "hsn": ["HSN Code", "Harmonized System of Nomenclature"],
    "sac": ["SAC Code", "Service Accounting Code"],
    
    # Compliance Terms
    "return": ["GSTR", "Return", "Filing"],
    "invoice": ["Invoice", "Tax Invoice", "Bill of Supply"],
    "registration": ["Registration", "GSTIN", "GST Registration"],
    "cancellation": ["Cancellation", "Revocation"],
}

# Notification Number Patterns
NOTIFICATION_PATTERN = re.compile(r'\b(\d{1,3})\s*[/-]\s*(\d{4})\b', re.IGNORECASE)

# Date Patterns
DATE_PATTERNS = [
    re.compile(r'\b(\d{4})\b'),  # Year
    re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b', re.IGNORECASE),
    re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),  # DD/MM/YYYY
]

class QueryProcessor:
    """Processes and enriches queries for legal document search."""
    
    def __init__(self):
        self.legal_terms_map = self._build_legal_terms_map()
    
    def _build_legal_terms_map(self) -> Dict[str, List[str]]:
        """Build reverse mapping for quick lookup."""
        term_map = {}
        for key, synonyms in LEGAL_TERMS.items():
            for synonym in synonyms:
                term_map[synonym.lower()] = key
                term_map[key.lower()] = key
        return term_map
    
    def extract_notification_number(self, query: str) -> List[Tuple[str, str]]:
        """Extract notification numbers from query (e.g., '1/2018', '01/2022')."""
        matches = NOTIFICATION_PATTERN.findall(query)
        return [f"{num}/{year}" for num, year in matches]
    
    def extract_year(self, query: str) -> List[str]:
        """Extract years from query."""
        years = []
        for pattern in DATE_PATTERNS:
            matches = pattern.findall(query)
            for match in matches:
                if isinstance(match, tuple):
                    year = match[-1] if len(match) > 1 else match[0]
                else:
                    year = match
                if year.isdigit() and 2017 <= int(year) <= 2025:
                    years.append(year)
        return list(set(years))
    
    def extract_tax_type(self, query: str) -> List[str]:
        """Extract tax type from query."""
        query_lower = query.lower()
        tax_types = []
        
        if any(term in query_lower for term in ["central tax", "cgst", "central gst"]):
            tax_types.append("Central Tax")
        if any(term in query_lower for term in ["integrated tax", "igst", "integrated gst"]):
            tax_types.append("Integrated Tax")
        if any(term in query_lower for term in ["union territory tax", "utgst", "union territory gst"]):
            tax_types.append("Union Territory Tax")
        if any(term in query_lower for term in ["cess", "compensation cess"]):
            tax_types.append("Compensation Cess")
        
        return tax_types
    
    def expand_legal_terms(self, query: str) -> Set[str]:
        """Expand query with legal terminology synonyms."""
        query_lower = query.lower()
        expanded_terms = set()
        
        # Add original query terms
        words = re.findall(r'\b\w+\b', query_lower)
        expanded_terms.update(words)
        
        # Find and expand legal terms
        for term, synonyms in LEGAL_TERMS.items():
            if term in query_lower:
                expanded_terms.update([s.lower() for s in synonyms])
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    expanded_terms.update([term] + [s.lower() for s in synonyms])
        
        return expanded_terms
    
    def extract_key_entities(self, query: str) -> Dict[str, any]:
        """Extract key entities from query."""
        return {
            "notification_numbers": self.extract_notification_number(query),
            "years": self.extract_year(query),
            "tax_types": self.extract_tax_type(query),
            "legal_terms": list(self.expand_legal_terms(query)),
        }
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process query and return enriched information.
        
        Returns:
            {
                "original_query": str,
                "expanded_terms": Set[str],
                "entities": Dict with notification_numbers, years, tax_types, legal_terms,
                "keywords": List[str] - important keywords for keyword search
            }
        """
        query = query.strip()
        
        # Extract entities
        entities = self.extract_key_entities(query)
        
        # Extract important keywords (non-stop words, legal terms)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "may", "might", "can", "what", "when", "where", "who", "which", "how"}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add legal terms
        keywords.extend(entities["legal_terms"])
        
        return {
            "original_query": query,
            "expanded_terms": entities["legal_terms"],
            "entities": entities,
            "keywords": list(set(keywords)),
        }

