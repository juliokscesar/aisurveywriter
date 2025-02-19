from typing import List, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass
import os

from aisurveywriter.core.paper import PaperData
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import AgentRAG, RAGType
from aisurveywriter.core.reference_store import ReferenceStore
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.text_embedding import EmbeddingsHandler
from aisurveywriter.core.prompt_store import PromptStore, PromptInfo
import aisurveywriter.tasks as tks

class SurveyAgentType(Enum):
    StructureGenerator = auto()
    Writer = auto()
    Reviewer = auto()

class SurveyContext:
    def __init__(
        self, 
        subject: str, 
        ref_paths: List[str], 
        llms: Union[LLMHandler, dict[SurveyAgentType, LLMHandler]],
        embed: EmbeddingsHandler,
        prompts: PromptStore,
        save_path: str = "survey_out",
        
        no_ref_faiss = False,
        no_review = False,
        no_figures = False,
        
        structure_json_path: Optional[str] = None,
        prewritten_tex_path: Optional[str] = None,
        
        bibdb_path: Optional[str] = None,
        faissbib_path: Optional[str] = None,
        faissfig_path: Optional[str] = None,
        faisscontent_path: Optional[str] = None,
        faiss_confidence: float = 0.6,
        
        llm_request_cooldown_sec: int = 30,
        embed_request_cooldown_sec: int = 0,
    ):
        save_path = os.path.abspath(save_path)
        if not save_path.endswith(".tex"):
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "survey.tex")
        
        self.save_path = save_path
        self.output_dir = os.path.dirname(save_path)
        
        # Initialize paper and load structure and pre-written .tex if provided
        self.paper = PaperData(subject=subject)
        if structure_json_path:
            self.paper.load_structure(structure_json_path)
        if prewritten_tex_path:
            self.paper.load_tex(prewritten_tex_path)
        
        self.references = ReferenceStore(ref_paths)
        self.references.bibtex_db_path = bibdb_path
        
        if isinstance(llms, LLMHandler):
            self.llms = {agent: llms for agent in SurveyAgentType}
        else:
            for agent in SurveyAgentType:
                if agent not in llms:
                    raise ValueError(f"Missing LLMHandler for agent {agent}")
            self.llms = llms
        
        self.embed = embed

        self.prompts = prompts
        
        self._llm_cooldown = llm_request_cooldown_sec
        self._embed_cooldown = embed_request_cooldown_sec
        
        self.rags = AgentRAG(self.embed, faissbib_path, faissfig_path, faisscontent_path, request_cooldown_sec=llm_request_cooldown_sec, output_dir=self.output_dir, confidence=faiss_confidence)
        self.pipeline = None
    
    def _create_pipeline(
        self,
        no_content_rag=False,
        skip_struct=False, 
        skip_fill=False, 
        skip_figures=False, 
        skip_references=False,
        skip_review=False, 
        skip_abstract=False,
        skip_tex_review=False
    ):
        # first ensure all rags are loaded/created
        using_rags = RAGType.All
        if no_content_rag:
            using_rags &= ~RAGType.GeneralText
        if skip_figures:
            using_rags &= ~RAGType.ImageData
        if skip_references:
            using_rags &= ~RAGType.BibTex
        
        self.rags.create_rags(using_rags, self.references)
        self.common_agent_ctx = AgentContext(
            prompts=self.prompts,
            sys_instructions=None,
            llm_handler=self.llms[SurveyAgentType.Writer],
            embed_handler=self.embed,
            llm_cooldown=self._llm_cooldown,
            embed_cooldown=self._embed_cooldown,
            references=self.references,
            rags=self.rags,
        )
        self.common_agent_ctx._working_paper = self.paper

        self.pipeline: List[tks.PipelineTask] = []
        if not skip_struct:
            self._check_input_variables(self.prompts.generate_struct, SurveyAgentType.StructureGenerator, tks.PaperStructureGenerator.required_input_variables)
                        
            struct_agent_ctx = self.common_agent_ctx.copy()
            struct_agent_ctx.llm_handler = self.llms[SurveyAgentType.StructureGenerator]
            struct_json_path = os.path.join(self.output_dir, "structure.yaml")
            self.pipeline.append(
                tks.PaperStructureGenerator(struct_agent_ctx, self.paper, struct_json_path)
            )

        if not skip_fill:
            self._check_input_variables(self.prompts.write_section, SurveyAgentType.Writer, tks.PaperWriter.required_input_variables)
            
            write_agent_ctx = self.common_agent_ctx.copy()
            write_agent_ctx.llm_handler = self.llms[SurveyAgentType.Writer]
            self.pipeline.append(
                tks.PaperWriter(write_agent_ctx, self.paper),
            )
            

    def _check_input_variables(self, prompt: PromptInfo, agent_type: SurveyAgentType, required_input_variables: List[str]):
        missing = set(prompt.input_variables - required_input_variables)
        assert(len(missing) == 0, f"Missing or additional input variables in prompt for {agent_type}: {missing}")
        
        
    def _is_initialized(self):
        return (self.pipeline is not None)
        
    def generate(self) -> PaperData:
       pass
    