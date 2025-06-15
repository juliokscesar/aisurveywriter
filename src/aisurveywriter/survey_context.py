from typing import List, Optional, Union, Tuple
from enum import ReprEnum, auto
from pydantic import BaseModel, Field
import os
from pathlib import Path
import yaml

from .core.paper import PaperData
from .core.agent_context import AgentContext
from .core.agent_rags import AgentRAG, RAGType
from .store.reference_store import ReferenceStore
from .core.llm_handler import LLMHandler, LLMConfig
from .core.lp_handler import LayoutParserSettings
from .core.text_embedding import EmbeddingsHandler
from .store.prompt_store import PromptStore, PromptInfo, default_prompt_store
from .core.pipeline import PaperPipeline
import aisurveywriter.tasks as tks
from .utils.logger import named_log
from .utils.helpers import time_func, load_pydantic_yaml, save_pydantic_yaml

class SurveyAgentType(str, ReprEnum):
    StructureGenerator: str = "structure_generator"
    Writer: str = "writer"
    Reviewer: str = "reviewer"
    
    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self.value)


class SurveyContextConfig(BaseModel):
    subject: str
    ref_paths: List[str]
    
    # Keep llms as SurveyAgentType -> LLMConfig
    llms: dict[Union[str, SurveyAgentType], LLMConfig] = Field(default_factory=dict)
    
    embed_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    embed_model_type: str = "huggingface"
    
    prompt_store_path: Optional[str] = None
    tex_template_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../templates/paper_template.tex"))
    save_path: str = "survey_out"
    
    reference_store_path: Optional[str] = None
    tesseract_executable: str = "tesseract"
    
    no_ref_faiss: bool = False
    no_review: bool = False
    no_figures: bool = False
    no_reference: bool = False
    no_abstract: bool = False
    no_tex_review: bool = False
    
    ref_max_per_section: int = 90
    ref_max_per_sentence: int = 4
    ref_max_same_ref: int = 10
    fig_max_figures: int = 30
    
    structure_json_path: Optional[str] = None
    prewritten_tex_path: Optional[str] = None
    
    bibdb_path: Optional[str] = None
    faissbib_path: Optional[str] = None

    images_dir: Optional[str] = None
    faissfig_path: Optional[str] = None

    faisscontent_path: Optional[str] = None
    faiss_confidence: float = 0.90

    llm_request_cooldown_sec: int = 30
    embed_request_cooldown_sec: int = 0

    def save_yaml(self, path: str):
        self.llms = {str(k): v for k, v in self.llms.items()}  # Convert SurveyAgentType to str for YAML serialization
        save_pydantic_yaml(self, path)
    
    @staticmethod
    def load_yaml(path: str):
        config = load_pydantic_yaml(path, SurveyContextConfig)
        config.llms = {SurveyAgentType(k): v for k, v in config.llms.items()}  # Convert str back to SurveyAgentType
        return config
        


class SurveyContext:
    def __init__(
        self, 
        config: Union[SurveyContextConfig, str],
        **pipeline_kwargs,
    ):
        # Load configuration
        if isinstance(config, str):
            config = load_pydantic_yaml(config, SurveyContextConfig)
        self.config = config

        # add .tex to save_path if not provided
        if not config.save_path.endswith(".tex"):
            os.makedirs(config.save_path, exist_ok=True)
            config.save_path = os.path.join(config.save_path, "survey.tex")
        
        
        self.save_path = config.save_path
        self.output_dir = os.path.dirname(config.save_path)
        self.tex_template_path = config.tex_template_path
        
        # Initialize paper and load structure and pre-written .tex if provided
        self.paper = PaperData(subject=config.subject, sections=None)
        if config.structure_json_path:
            self.paper.load_structure(config.structure_json_path)
        if config.prewritten_tex_path:
            self.paper.load_tex(config.prewritten_tex_path)
        
        self.images_dir = config.images_dir
        if not self.images_dir:
            self.images_dir = os.path.join(self.output_dir, "images") # rag creates this directory if none was provided
        self.paper.fig_path = self.images_dir

        # Initialize Layout Parser settings
        lp_config = "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
        lp_settings = LayoutParserSettings(config_path=lp_config, tesseract_executable=config.tesseract_executable)
        
        # Load or create reference store
        if self.config.reference_store_path:
            self.references = ReferenceStore.from_local(self.config.reference_store_path)
            self.references.update_references(self.config.ref_paths)
            self.references.lp_settings = lp_settings
            named_log(self, "loaded reference store from", self.config.reference_store_path, f"total of {len(self.references.documents)} references")
        else:
            self.config.reference_store_path = os.path.join(self.output_dir, "refstore.pkl")
            self.references = ReferenceStore.create_store(self.config.ref_paths, lp_settings, self.images_dir,
                                                save_local=self.config.reference_store_path, 
                                                title_extractor_llm=None)
            named_log(self, "saved reference store to", self.config.reference_store_path, f"total of {len(self.references.documents)} references")
        self.references.bibtex_db_path = config.bibdb_path
        self.references.images_dir = self.images_dir
        
        # Create LLM handlers
        llms_config = config.llms
        self.llms = {agent_type: None for agent_type in SurveyAgentType}
        self._llm_cooldown = config.llm_request_cooldown_sec
        
        # load configuration for each LLM
        for agent_type, llm_config in llms_config.items():
            agent_type = SurveyAgentType(agent_type)
            self.llms[agent_type] = LLMHandler.from_config(llm_config)
        
        # load Text Embedding Model
        self.embed = EmbeddingsHandler(name=config.embed_model, model_type=config.embed_model_type)
        self._embed_cooldown = config.embed_request_cooldown_sec


        if config.prompt_store_path:
            with open(config.prompt_store_path, "r", encoding="utf-8") as f:
                self.prompts = PromptStore.model_validate_json(f.read())
        else:
            self.prompts = default_prompt_store()
        
        self.confidence = config.faiss_confidence
        self.rags = AgentRAG(self.embed, self.llms[SurveyAgentType.StructureGenerator], 
                             config.faissbib_path, config.faissfig_path, config.faisscontent_path, 
                             request_cooldown_sec=6, output_dir=self.output_dir, 
                             confidence=self.confidence)
                
        
        self.pipe_steps: List[tks.PipelineTask] = None
        self.pipeline: PaperPipeline = None
        self._create_pipeline(
            no_content_rag=config.no_ref_faiss, 
            skip_struct=(config.structure_json_path and config.structure_json_path.strip()),
            skip_fill=(config.prewritten_tex_path and config.prewritten_tex_path.strip()),
            skip_figures=config.no_figures,
            skip_references=config.no_reference,
            skip_review=config.no_review,
            skip_abstract=config.no_abstract,
            skip_tex_review=config.no_tex_review,
            
            ref_max_per_section=config.ref_max_per_section,
            ref_max_per_sentence=config.ref_max_per_sentence,
            ref_max_same_ref=config.ref_max_same_ref,
            
            fig_max_figures=config.fig_max_figures,
            
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
            self._check_input_variables(self.prompts.add_references, SurveyAgentType.StructureGenerator, tks.PaperReferencer.required_input_variables)
            
            reference_agent_ctx = self.common_agent_ctx.copy()
            reference_agent_ctx.llm_handler = self.llms[SurveyAgentType.StructureGenerator]
            reference_agent_ctx.llm_cooldown = 6
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
        assert self._is_initialized()
        
        named_log(self, "BEGIN SURVEY GENERATION PIPELINE")
        named_log(self, "pipeline description:")
        print(self.pipeline.describe_steps())
        
        elapsed, final_paper = time_func(self.pipeline.run, initial_data=self.paper)
        
        named_log(self, "FINISH SURVEY GENERATION PIPELINE")
        named_log(self, f"time taken: {elapsed} s")
        
        return final_paper
        