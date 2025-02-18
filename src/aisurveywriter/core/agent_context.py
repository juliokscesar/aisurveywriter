from dataclasses import dataclass
from typing import List

from langchain_community.docstore.document import Document
from langchain_core.prompts.chat import SystemMessage

from .llm_handler import LLMHandler
from .text_embedding import EmbeddingsHandler
from .pdf_processor import PDFProcessor

@dataclass
class WorkReferenceData:
    title:    str = ""
    abstract: str = ""
    keywords: str = ""
    
@dataclass
class GeneralTextData:
    docs: List[List[Document]] = None


class AgentRAG:
    def __init__(self):
        pass

@dataclass
class AgentContext:
    sys_instructions: List[SystemMessage] = None
    llm_handler: LLMHandler = None

    embed_handler: EmbeddingsHandler = None
    
    reference_pdfs: PDFProcessor = None
