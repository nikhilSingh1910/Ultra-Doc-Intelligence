from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    section_type: str = "general"  # "header", "key_value_block", "table", "body", "general"
    heading: str = ""
    page_number: int = 1
    char_offset_start: int = 0
    char_offset_end: int = 0
    word_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())


@dataclass
class ParsedDocument:
    text: str
    filename: str
    file_type: str  # "pdf", "docx", "txt"
    num_pages: int = 1
    metadata: Dict = field(default_factory=dict)
    tables: List[List[List[str]]] = field(default_factory=list)
    pages: List[str] = field(default_factory=list)  # per-page text for page-level chunking


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    file_type: str
    text: str
    num_chunks: int = 0
    num_pages: int = 1
    status: str = "processed"
