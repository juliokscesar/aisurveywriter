from dataclasses import dataclass
from typing import List

from langchain_community.docstore.document import Document
from langchain_core.prompts.chat import SystemMessage

from .llm_handler import LLMHandler
from .text_embedding import EmbeddingsHandler
from .reference_store import ReferenceStore
from .agent_rags import AgentRAG

@dataclass
class AgentContext:
    # List of system instructions for the agent
    sys_instructions: List[SystemMessage] = None
    
    llm_handler: LLMHandler = None

    embed_handler: EmbeddingsHandler = None

    # Reference store (content, their bibliography, paths, pdf processor, etc)
    references: ReferenceStore = None
    
    # bib, content and figures RAGs
    rags: AgentRAG = None