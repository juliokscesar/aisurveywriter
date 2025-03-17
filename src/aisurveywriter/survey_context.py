from typing import List, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass
import os

from aisurveywriter.core.paper import PaperData
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import AgentRAG, RAGType
from aisurveywriter.store.reference_store import ReferenceStore, build_reference_store
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.text_embedding import EmbeddingsHandler
from aisurveywriter.store.prompt_store import PromptStore, PromptInfo
from aisurveywriter.core.pipeline import PaperPipeline
import aisurveywriter.tasks as tks
from aisurveywriter.utils.logger import named_log
from aisurveywriter.utils.helpers import time_func

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
        tex_template_path: str,
        save_path: str = "survey_out",

        reference_store_path: str = None,
        tesseract_executable: str = "tesseract",
        
        no_ref_faiss = False,
        no_review = False,
        no_figures = False,
        no_reference = False,
        no_abstract = False,
        no_tex_review = False,
        
        structure_json_path: Optional[str] = None,
        prewritten_tex_path: Optional[str] = None,
        
        bibdb_path: Optional[str] = None,
        faissbib_path: Optional[str] = None,
        faissfig_path: Optional[str] = None,
        faisscontent_path: Optional[str] = None,
        faiss_confidence: float = 0.75,
        
        images_dir: Optional[str] = None,
        
        llm_request_cooldown_sec: int = 30,
        embed_request_cooldown_sec: int = 0,

        ref_max_per_section: int = 90,
        ref_max_per_sentence: int = 4,
        ref_max_same_ref: int = 10,
        
        fig_max_figures: int = 30,
        
        **pipeline_kwargs,
    ):
        save_path = os.path.abspath(save_path)
        if not save_path.endswith(".tex"):
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "survey.tex")
        
        self.save_path = save_path
        self.output_dir = os.path.dirname(save_path)
        self.tex_template_path = tex_template_path
        
        # Initialize paper and load structure and pre-written .tex if provided
        self.paper = PaperData(subject=subject, sections=None)
        if structure_json_path:
            self.paper.load_structure(structure_json_path)
        if prewritten_tex_path:
            self.paper.load_tex(prewritten_tex_path)
        
        self.images_dir = images_dir
        if not self.images_dir:
            self.images_dir = os.path.join(self.output_dir, "images") # rag creates this directory if none was provided
        self.paper.fig_path = self.images_dir

        # self.references = ReferenceStore(ref_paths)
        if reference_store_path:
            self.references = ReferenceStore.from_local(reference_store_path)
            named_log(self, "loaded reference store from", reference_store_path)
        else:
            reference_store_path = os.path.join(self.output_dir, "refstore.pkl")
            self.references = build_reference_store(ref_paths, self.images_dir, 
                                                    save_local=reference_store_path, 
                                                    title_extractor_llm=None,
                                                    lp_tesseract_executable=tesseract_executable)
            named_log(self, "saved reference store to", reference_store_path)
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
        
        self.rags = AgentRAG(self.embed, self.llms[SurveyAgentType.StructureGenerator], faissbib_path, faissfig_path, faisscontent_path, request_cooldown_sec=6, output_dir=self.output_dir, confidence=faiss_confidence)
                
        self.confidence = faiss_confidence
        
        self.pipe_steps: List[tks.PipelineTask] = None
        self.pipeline: PaperPipeline = None
        self._create_pipeline(
            no_content_rag=no_ref_faiss, 
            skip_struct=(structure_json_path and structure_json_path.strip()),
            skip_fill=(prewritten_tex_path and prewritten_tex_path.strip()),
            skip_figures=no_figures,
            skip_references=no_reference,
            skip_review=no_review,
            skip_abstract=no_abstract,
            skip_tex_review=no_tex_review,
            
            ref_max_per_section=ref_max_per_section,
            ref_max_per_sentence=ref_max_per_sentence,
            ref_max_same_ref=ref_max_same_ref,
            
            fig_max_figures=fig_max_figures,
            
            **pipeline_kwargs,
        )
    
    def _create_pipeline(
        self,
        no_content_rag=False,
        skip_struct=False, 
        skip_fill=False, 
        skip_figures=False, 
        skip_references=False,
        skip_review=False, 
        skip_abstract=False,
        skip_tex_review=False,
        
        ref_max_per_section: int = 90,
        ref_max_per_sentence: int = 4,
        ref_max_same_ref: int = 10,
        
        fig_max_figures: int = 30,
        
        **pipeline_kwargs,
    ):
        self.pipe_steps = []

        # first ensure all rags are loaded/created
        using_rags = RAGType.All
        if no_content_rag or (skip_fill and skip_review):
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
            output_dir=self.output_dir,
        )
        self.common_agent_ctx._working_paper = self.paper

        self.pipe_steps: List[tks.PipelineTask] = []
        if not skip_struct:
            self._check_input_variables(self.prompts.generate_struct, SurveyAgentType.StructureGenerator, tks.PaperStructureGenerator.required_input_variables)
                        
            struct_agent_ctx = self.common_agent_ctx.copy()
            struct_agent_ctx.llm_handler = self.llms[SurveyAgentType.StructureGenerator]
            struct_json_path = os.path.join(self.output_dir, "structure.json")
            self.pipe_steps.append(
                ("Generate Structure", tks.PaperStructureGenerator(struct_agent_ctx, self.paper, struct_json_path))
            )

        if not skip_fill:
            self._check_input_variables(self.prompts.write_section, SurveyAgentType.Writer, tks.PaperWriter.required_input_variables)
            
            write_agent_ctx = self.common_agent_ctx.copy()
            write_agent_ctx.llm_handler = self.llms[SurveyAgentType.Writer]
            self.pipe_steps.extend([
                ("Write paper", tks.PaperWriter(write_agent_ctx, self.paper)),
                ("Save scratch", tks.PaperSaver(self.save_path.replace(".tex", "-scratch.tex"), self.tex_template_path)),
            ])
            
        if not skip_figures:
            self._check_input_variables(self.prompts.add_figures, SurveyAgentType.StructureGenerator, tks.PaperFigureAdd.required_input_variables)

            figures_agent_ctx = self.common_agent_ctx.copy()
            figures_agent_ctx.llm_handler = self.llms[SurveyAgentType.StructureGenerator] # use llm from structure generation because we use the entire reference content
            self.pipe_steps.extend([
                ("Add figures", tks.PaperFigureAdd(figures_agent_ctx, self.paper, self.images_dir, self.confidence, max_figures=fig_max_figures)),
                ("Save with figures", tks.PaperSaver(self.save_path.replace(".tex", "-figures.tex"), self.tex_template_path))
            ])
            
        if not skip_review:
            self._check_input_variables(self.prompts.review_section, SurveyAgentType.Reviewer, tks.PaperReviewer.required_input_variables)
            self._check_input_variables(self.prompts.apply_review_section, SurveyAgentType.Reviewer, tks.PaperReviewer.required_input_variables)

            review_agent_ctx = self.common_agent_ctx.copy()
            review_agent_ctx.llm_handler = self.llms[SurveyAgentType.Reviewer]
            self.pipe_steps.extend([
                ("Review paper", tks.PaperReviewer(review_agent_ctx, self.paper)),
                ("Save reviewed", tks.PaperSaver(self.save_path.replace(".tex", "-review.tex"), self.tex_template_path)),
            ])
            
        if not skip_references:
            # paper referencer doesn't need a prompt
            reference_agent_ctx = self.common_agent_ctx.copy()
            self.pipe_steps.extend([
                ("Reference paper", tks.PaperReferencer(reference_agent_ctx, self.paper, self.save_path.replace(".tex", ".bib"), max_per_section=ref_max_per_section, max_per_sentence=ref_max_per_sentence, max_same_ref=ref_max_same_ref)),
                ("Save referenced", tks.PaperSaver(self.save_path.replace(".tex", "-ref.tex"), self.tex_template_path))
            ])

        if not skip_abstract:
            self._check_input_variables(self.prompts.abstract_and_title, SurveyAgentType.Writer, tks.PaperRefiner.required_input_variables)
            
            refine_agent_ctx = self.common_agent_ctx.copy()
            refine_agent_ctx.llm_handler = self.llms[SurveyAgentType.Writer] # use struct gen LLM becase of tokens
            self.pipe_steps.extend([
                ("Refine paper (abstract and title)", tks.PaperRefiner(refine_agent_ctx, self.paper)),
                ("Save refined", tks.PaperSaver(self.save_path.replace(".tex", "-refined.tex"), self.tex_template_path)),
            ])

        if not skip_tex_review:
            # tex review doesn't need a prompt
            tex_review_agent_ctx = self.common_agent_ctx.copy()
            self.pipe_steps.append(("Tex Review", tks.TexReviewer(tex_review_agent_ctx, self.paper)))
        
        self.pipe_steps.append(("Save final paper", tks.PaperSaver(self.save_path, self.tex_template_path)))
        self.pipeline = PaperPipeline(self.pipe_steps, **pipeline_kwargs)

    def _check_input_variables(self, prompt: PromptInfo, agent_type: SurveyAgentType, required_input_variables: List[str]):
        missing = set(prompt.input_variables) - set(required_input_variables)
        assert len(missing) == 0, f"Missing or additional input variables in prompt for {agent_type}: {missing}"
        
    def _is_initialized(self):
        return (self.pipeline is not None)
        
    def generate(self) -> PaperData:
        assert(self._is_initialized())
        
        named_log(self, "==> BEGIN SURVEY GENERATION PIPELINE")
        named_log(self, "==> pipeline description:")
        print(self.pipeline.describe_steps())
        
        elapsed, final_paper = time_func(self.pipeline.run, initial_data=self.paper)
        
        named_log(self, "==> FINISH SURVEY GENERATION PIPELINE")
        named_log(self, f"==> time taken: {elapsed} s")
        
        return final_paper
        