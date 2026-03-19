import re
from typing import List, Set


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "what", "which",
    "who", "whom", "this", "that", "these", "those", "it", "its",
}


def extract_keywords(text: str) -> Set[str]:
    words = re.findall(r'\b[a-zA-Z0-9$.,/-]+\b', text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_factual_claims(text: str) -> List[str]:
    claims = []
    # Dollar amounts
    claims.extend(re.findall(r'\$[\d,]+\.?\d*', text))
    # Dates (various formats)
    claims.extend(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
    claims.extend(re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b', text, re.I))
    # Numbers with units
    claims.extend(re.findall(r'\b[\d,]+\s*(?:lbs?|kg|tons?|miles?|ft|gallons?)\b', text, re.I))
    # Percentages
    claims.extend(re.findall(r'\b\d+\.?\d*\s*%', text))
    # Time patterns
    claims.extend(re.findall(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b', text))
    return claims


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
