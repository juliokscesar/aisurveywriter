from typing import Union, List
import os
import yaml
import json
import pydantic

from .core.pipeline import PaperPipeline
from .core.llm_handler import LLMHandler
from .core.agent_rags import AgentRAG, RAGType
from .core.agent_context import AgentContext
from .core.text_embedding import EmbeddingsHandler
from .core.lp_handler import LayoutParserSettings
from .core.paper import PaperData
import aisurveywriter.tasks as tks
from .store import ReferenceStore, PromptStore, PromptInfo, default_prompt_store
from .utils.helpers import get_all_files_from_paths, time_func, save_pydantic_json
from .utils.logger import named_log

class SurveyContext:
    def __init__(self, config: Union[str, dict], **pipeline_kwargs):
        self.config = self.load_config(config)
    
        # Initialize subject and references paths
        self.subject = self.config.get("subject").strip()
        if not self.subject:
            raise RuntimeError("Please provide a subject in configuration")
        
        self.references_dir = self.config.get("references_dir").strip()
        if not self.references_dir or not os.path.isdir(self.references_dir):
            raise RuntimeError("Please provide a valid directory containing references")
        
        save_path = self.config.get("save_path", "out")
        
        # assume path given is a directory if no .tex on the end
        if not save_path.endswith(".tex"):
            save_path = os.path.join(save_path, "survey.tex")
        
        self.save_path = save_path
        self.output_dir = os.path.dirname(save_path)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.extracted_images_dir = os.path.join(self.output_dir, "ref_images")
        
        self.tex_template_path = self.config.get("tex_template_path", None)
        if not self.tex_template_path:
            raise RuntimeError("No tex template given in config")
        if not os.path.isfile(self.tex_template_path):
            raise FileNotFoundError(f"Couldn't find TEX template: {self.tex_template_path}")
        
        # Get prompts
        prompt_store_path = self.config.get("prompt_store_path", None)
        if prompt_store_path:
            if not os.path.isfile(prompt_store_path):
                raise FileExistsError("Prompt store file doesn't exist:", prompt_store_path)
            with open(prompt_store_path, "r", encoding="utf-8") as f:
                prompt_store = PromptStore.model_validate_json(f.read())
        else:
            prompt_store = default_prompt_store()
        self.prompts = prompt_store

        if self.config.get("save_prompt_store", False):
            prompt_store_save_path = os.path.join(self.output_dir, "prompt_store.json")
            save_pydantic_json(prompt_store, prompt_store_save_path)
        
        import pdb; pdb.set_trace()
        
        self.references_paths = get_all_files_from_paths(self.references_dir, skip_ext=["png", "jpg", "jpeg"],
                                                         stem_sort=True)
        if not self.references_paths:
            raise RuntimeError(f"No references found in {self.references_dir}")
        
        pdf_parser_method = self.config.get("pdf_parser_method", "layout_parser")
        if pdf_parser_method == "layout_parser":
            lp_config = self.config.get("layout_parser_config", "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config")
            if not lp_config:
                raise RuntimeError("Please provide a valid config for Layout Parser")
            tesseract = self.config.get("tesseract_executable", "tessseract")
            lp_settings = LayoutParserSettings(config_path=lp_config, tesseract_executable=tesseract)
        else:
            raise RuntimeError(f"Invalid PDF parser method:", pdf_parser_method)
    
        # Initialize (or load) reference store
        reference_store_path = self.config.get("reference_store_path", None)
        if reference_store_path:
            if not os.path.isfile(reference_store_path):
                raise FileNotFoundError(f"Invalid or missing reference store file:", reference_store_path)
            
            try:
                reference_store = ReferenceStore.from_local(reference_store_path)
                reference_store.lp_settings = lp_settings
                reference_store.update_references(self.references_paths)
                named_log(self, "loaded reference store from", reference_store_path)
            except Exception as e:
                import traceback;
                named_log(self, "Unable to load reference store and/or update references:", traceback.format_exc(e))
                raise e
        else:
            reference_store_path = os.path.join(self.output_dir, "refstore.pkl")
            try:
                reference_store = ReferenceStore.create_store(self.references_paths, lp_settings, self.extracted_images_dir, reference_store_path)
                named_log(self, "saved reference store to", reference_store_path)
            except Exception as e:
                import traceback;
                named_log(self, "Unable to create a Reference Store:", traceback.format_exc(e))
                raise e
        
        self.reference_store = reference_store
        self.reference_store_path = reference_store_path
        
        self.reference_store.images_dir = self.extracted_images_dir
        
        # Load specified bibliography DB if given
        bibtex_db_path = self.config.get("bibtex_db_path", None)
        if bibtex_db_path and os.path.isfile(bibtex_db_path):
            self.reference_store.bibtex_db_path = bibtex_db_path
        
        # Initialize Paper data
        self.paper = PaperData(self.subject, sections=None)
        
        structure_json_path = self.config.get("structure_json_path", None)
        if structure_json_path and os.path.isfile(structure_json_path):
            self.paper.load_structure(structure_json_path)
            
            # Skip structure generation
            self.config["task_structure_generation"]["skip"] = True
        
        prewritten_tex_path = self.config.get("prewritten_tex_path", None)
        if prewritten_tex_path and os.path.isfile(prewritten_tex_path):
            self.paper.load_tex(prewritten_tex_path)
            
            # Skip writing step
            self.config["sections_fill"]["skip"] = True
        
        # Initialize LLM agents
        self.llm_agents = {
            "writer": None, 
            "structure_generator": None,
            "reviewer": None,
            "bib_extractor": None,
        }
        
        for agent_name in self.llm_agents:
            agent_config = self.config.get(agent_name, None)
            if not agent_config:
                raise RuntimeError("Missing configuration for agent: " + agent_name)
        
            self.llm_agents[agent_name] = LLMHandler(agent_config["model"], agent_config["type"], agent_config["temperature"])
        
        # Initialize Text Embedding model
        self.embed = EmbeddingsHandler(self.config["embed_model"], self.config["embed_model_type"])
        
        # Initialize RAGs    
        self.rags = AgentRAG(
            self.embed, 
            self.llm_agents["bib_extractor"],
            self.config.get("faiss_bib_path", None),
            self.config.get("faiss_fig_path", None),
            self.config.get("faiss_content_path", None),
            request_cooldown_sec=self.config["bib_extractor"]["cooldown"],
        )
        
        # Create pipeline
        self.pipe_steps = None
        self.pipeline = None
        self._create_pipeline(**pipeline_kwargs)
        
    def _create_pipeline(self, **pipeline_kwargs):
        self.pipe_steps = []
        
        using_rags = RAGType.All
        if self.config["task_sections_fill"]["skip"] and self.config["task_review_sections"]["skip"]:
            using_rags &= ~RAGType.GeneralText
        if self.config["task_add_figures"]["skip"]:
            using_rags &= ~RAGType.ImageData
        if self.config["task_add_references"]["skip"]:
            using_rags &= ~RAGType.BibTex
        
        self.rags.create_rags(using_rags, self.reference_store)
        
        self.common_agent_ctx = AgentContext(
            self.prompts, None, self.llm_agents["writer"], self.embed,
            0, self.config.get("embed_cooldown", 0), self.reference_store, 
            self.rags, self.output_dir, self.paper
        )
        
        if not self.config["task_structure_generation"]["skip"]:
            task_agent = self.config["task_structure_generation"].get("llm_agent", "strucutre_generator")
            self._check_input_variables(self.prompts.generate_struct, task_agent, tks.PaperStructureGenerator.required_input_variables)
            
            struct_agent_ctx = self._get_task_agent_ctx("task_structure_generation")
            structure_json_path = os.path.join(self.output_dir, "structure.json")
            self.pipe_steps.append(
                ("Generate Structure", tks.PaperStructureGenerator(struct_agent_ctx, self.paper, structure_json_path))
            )
            
        if not self.config["task_sections_fill"]["skip"]:
            task_agent = self.config["task_sections_fill"].get("llm_agent", "writer")
            self._check_input_variables(self.prompts.write_section, task_agent, tks.PaperWriter.required_input_variables)
            
            filler_agent_ctx = self._get_task_agent_ctx("task_sections_fill")
            self.pipe_steps.extend([
                ("Write Paper", tks.PaperWriter(filler_agent_ctx, self.paper)),
                ("Save scratch", tks.PaperSaver(self.save_path.replace(".tex", "-scratch.tex"), self.tex_template_path))
            ])
            
        if not self.config["task_add_figures"]["skip"]:
            task_agent = self.config["task_add_figures"].get("llm_agent", "writer")
            self._check_input_variables(self.prompts.add_figures, task_agent, tks.PaperFigureAdd.required_input_variables)
            
            figure_add_agent_ctx = self._get_task_agent_ctx("task_add_figures")
            
            sim_thresh = self.config["task_add_figures"].get("similarity_threshold", 0.75)
            max_figures = self.config["task_add_figures"].get("max_figures", 30)
            max_figures_per_section = self.config["task_add_figures"].get("max_figures_per_section", 3)
            
            self.pipe_steps.extend([
                ("Add Figures", tks.PaperFigureAdd(
                    figure_add_agent_ctx, self.paper, self.extracted_images_dir,
                    sim_thresh, max_figures, max_figures_per_section
                )),
                ("Save with figures", tks.PaperSaver(self.save_path.replace(".tex", "-figures.tex"), self.tex_template_path))
            ])
    
        if not self.config["task_review_sections"]["skip"]:
            task_agent = self.config["task_review_sections"].get("llm_agent", "reviewer")
            self._check_input_variables(self.prompts.review_section, task_agent, tks.PaperReviewer.required_input_variables)
            self._check_input_variables(self.prompts.apply_review_section, task_agent, tks.PaperReviewer.required_input_variables)
            
            reviewer_agent_ctx = self._get_task_agent_ctx("task_review_sections")
            
            self.pipe_steps.extend([
                ("Review paper", tks.PaperReviewer(reviewer_agent_ctx, self.paper)),
                ("Save reviewed", tks.PaperSaver(self.save_path.replace(".tex", "-review.tex"), self.tex_template_path)),
            ])
            
        if not self.config["task_add_references"]["skip"]:
            task_agent = self.config["task_add_references"].get("llm_agent", "writer")
            self._check_input_variables(self.prompts.add_references, task_agent, tks.PaperReferencer.required_input_variables)
            
            reference_agent_ctx = self._get_task_agent_ctx("task_add_references")
            
            # TODO: edit PaperReferencer to take retrieval_k as param
            retrieval_k = self.config["task_add_references"].get("retrieval_k_per_paragraph", 50)
            sim_thresh = self.config["task_add_references"].get("similarity_threshold", 0.8)
            max_refs_per_section = self.config["task_add_references"].get("max_references_per_section", 50)
            max_same_ref = self.config["task_add_references"].get("max_same_reference", 10)
            
            self.pipe_steps.extend([
                ("Reference paper", tks.PaperReferencer(reference_agent_ctx, self.paper, self.save_path.replace(".tex", ".bib"), max_refs_per_section, max_same_ref=max_same_ref)),
                ("Save referenced", tks.PaperSaver(self.save_path.replace(".tex", "-ref.tex"), self.tex_template_path))
            ])
            
        if not self.config["task_abstract_and_title"]["skip"]:
            task_agent = self.config["task_abstract_and_title"].get("llm_agent", "writer")
            self._check_input_variables(self.prompts.abstract_and_title, task_agent, tks.PaperRefiner.required_input_variables)
            
            refiner_agent_ctx = self._get_task_agent_ctx("task_abstract_and_title")
            self.pipe_steps.extend([
                ("Refine paper (abstract and title)", tks.PaperRefiner(refiner_agent_ctx, self.paper)),
                ("Save refined", tks.PaperSaver(self.save_path.replace(".tex", "-refined.tex"), self.tex_template_path)),
            ])
            
        if not self.config["task_tex_review"]["skip"]:
            tex_review_ctx = self._get_task_agent_ctx("task_tex_review")
            self.pipe_steps.append(("Tex Review", tks.TexReviewer(tex_review_ctx, self.paper)))
     
        self.pipe_steps.append(("Save final paper", tks.PaperSaver(self.save_path, self.tex_template_path)))
        self.pipeline = PaperPipeline(self.pipe_steps, **pipeline_kwargs)
     
    
    def _get_task_agent_ctx(self, task_key: str) -> AgentContext:
        llm_agent = self.config[task_key].get("llm_agent", "writer")
        cooldown = self.config[llm_agent].get("cooldown", 0)
        
        task_agent_configs = self.config[task_key].get("agent_configs", {})
        cooldown = task_agent_configs.get("cooldown", cooldown)
        
        task_agent_ctx = self.common_agent_ctx.copy()
        task_agent_ctx.llm_handler = self.llm_agents[llm_agent]
        task_agent_ctx.cooldown = cooldown
        
        return task_agent_ctx
    
    
    def load_config(self, config: Union[str, dict]) -> dict:
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
        return config

    def _check_input_variables(self, prompt: PromptInfo, agent_name: str, required_input_variables: List[str]):
        missing = set(prompt.input_variables) - set(required_input_variables) if len(prompt.input_variables) > len(required_input_variables) else set(required_input_variables) - set(prompt.input_variables)
        assert len(missing) == 0, f"Missing or additional input variables in prompt for {agent_name}: {missing}"
        
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
        