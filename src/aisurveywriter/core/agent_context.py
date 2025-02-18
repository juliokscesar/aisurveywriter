from dataclasses import dataclass
from typing import List

from langchain_community.docstore.document import Document

from .llm_handler import LLMHandler
from .text_embedding import EmbeddingsHandler

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
    llm_handler: LLMHandler = None
    embed_handler: EmbeddingsHandler = None
    
