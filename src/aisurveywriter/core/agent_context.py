from dataclasses import dataclass
from typing import List

from langchain_community.docstore.document import Document
from langchain_core.prompts.chat import SystemMessage

from .llm_handler import LLMHandler
from .text_embedding import EmbeddingsHandler
from .reference_store import ReferenceStore
from .agent_rags import AgentRAG
from .paper import PaperData
from .prompt_store import PromptStore

@dataclass
class AgentContext:
    prompts: PromptStore
    
    # List of system instructions for the agent
    sys_instructions: List[SystemMessage] = None
    
    llm_handler: LLMHandler = None
    embed_handler: EmbeddingsHandler = None
    llm_cooldown: int = 0
    embed_cooldown: int = 0

    # Reference store (content, their bibliography, paths, pdf processor, etc)
    references: ReferenceStore = None
    
    # bib, content and figures RAGs
    rags: AgentRAG = None

    _working_paper: PaperData = None

    def copy(self):
        return AgentContext(
            sys_instructions=self.sys_instructions.copy() if self.sys_instructions else None,
            llm_handler=self.llm_handler,
            embed_handler=self.embed_handler,
            references=self.references,
            rags=self.rags,
            _working_paper=self._working_paper,
        )