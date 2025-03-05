from typing import List, Optional
import os
import queue
import re

import aisurveywriter.core.file_handler as fh
from aisurveywriter.survey_context import SurveyContext, SurveyAgentType
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.text_embedding import EmbeddingsHandler

from aisurveywriter.store.prompt_store import PromptStore, default_prompt_store

def tex_filter_survey(tex_content: str):
    tex_content = re.sub(r"[`]+[\w]*", "", tex_content)
    tex_content = tex_content.replace("\\begin{document}", "")
    tex_content = tex_content.replace("\\end{document}", "")
    tex_content = tex_content.replace("\\begin{filecontents}", "")
    tex_content = tex_content.replace("\\end{filecontents}", "")
    tex_content = re.sub(r"\\usepackage\{([^}]+)\}", "", tex_content)
    return tex_content

def setup_credentials(credentials_path: str):
    credentials = fh.read_credentials(credentials_path)
    if credentials["google_key"]:
        os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    if credentials["openai_key"]:
        os.environ["OPENAI_API_KEY"] = credentials["openai_key"]

def generate_paper_survey(
    subject: str,
    ref_paths: List[str],
    save_path: str,
    
    structure_model: str = "gemini-2.0-flash",
    structure_model_type: str = "google",
    
    writer_model: str = "gemini-2.0-pro-exp",
    writer_model_type: str = "google",
    
    reviewer_model: str = "gemini-2.0-pro-exp",
    reviewer_model_type: str = "google",
    
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_model_type: str = "huggingface",
    
    custom_prompt_store: Optional[PromptStore] = None,
    tex_template_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../templates/paper_template.tex")),
    
    no_ref_faiss = False,
    no_review = False,
    no_figures = False,
    no_reference = False,
    no_abstract = False,
    no_tex_review = False,

    pregen_struct_json_path: Optional[str] = None,
    prewritten_tex_path: Optional[str] = None,
    
    bibdb_path: Optional[str] = None,
    faissbib_path: Optional[str] = None,
    faissfig_path: Optional[str] = None,
    faisscontent_path: Optional[str] = None,
    faiss_confidence: float = 0.7,

    images_dir: Optional[str] = None,
    
    llm_request_cooldown_sec: int = 30,
    embed_request_cooldown_sec: int = 0,
    
    pipeline_status_queue: queue.Queue = None,
    credentials_yaml_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../credentials.yaml")),

    ref_max_per_section: int = 90,
    ref_max_per_sentence: int = 4,
    ref_max_same_ref: int = 10,

    fig_max_figures: int = 30,
):
    setup_credentials(credentials_yaml_path)
    
    agent_llms = {
        SurveyAgentType.StructureGenerator: LLMHandler(structure_model, structure_model_type),
        SurveyAgentType.Writer: LLMHandler(writer_model, writer_model_type),
        SurveyAgentType.Reviewer: LLMHandler(reviewer_model, reviewer_model_type),
    }

    embed = EmbeddingsHandler(embed_model, embed_model_type)

    prompts = custom_prompt_store
    if not prompts:
        prompts = default_prompt_store()

    survey_ctx = SurveyContext(
        subject, ref_paths, agent_llms, embed, prompts, tex_template_path,
        save_path, no_ref_faiss, no_review, no_figures, no_reference, no_abstract,
        no_tex_review, pregen_struct_json_path, prewritten_tex_path, bibdb_path,
        faissbib_path, faissfig_path, faisscontent_path, faiss_confidence,
        images_dir, llm_request_cooldown_sec, embed_request_cooldown_sec, 
        ref_max_per_section, ref_max_per_sentence, ref_max_same_ref,
        status_queue=pipeline_status_queue
    )
    return survey_ctx.generate()

