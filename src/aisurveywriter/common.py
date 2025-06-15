from typing import List, Optional
import os
import queue
import re

from .core.file_handler import read_credentials
from .survey_context import SurveyContext, SurveyAgentType, SurveyContextConfig
from .core.llm_handler import LLMHandler, LLMConfig
from .core.text_embedding import EmbeddingsHandler

from .store.prompt_store import PromptStore, default_prompt_store
from .utils.helpers import load_pydantic_yaml

def tex_filter_survey(tex_content: str):
    tex_content = re.sub(r"[`]+[\w]*", "", tex_content)
    tex_content = tex_content.replace("\\begin{document}", "")
    tex_content = tex_content.replace("\\end{document}", "")
    tex_content = tex_content.replace("\\begin{filecontents}", "")
    tex_content = tex_content.replace("\\end{filecontents}", "")
    tex_content = re.sub(r"\\usepackage\{([^}]+)\}", "", tex_content)
    return tex_content

def setup_credentials(credentials_path: str):
    credentials = read_credentials(credentials_path)
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
    
    temperature: float = 0.65,
    
    embed_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
    embed_model_type: str = "huggingface",
    
    custom_prompt_store: Optional[str] = None,
    tex_template_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../templates/paper_template.tex")),

    local_reference_store: Optional[str] = None,
    tesseract_executable: str = "tesseract",
    
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
        SurveyAgentType.StructureGenerator: LLMConfig(model=structure_model, model_type=structure_model_type, temperature=temperature),
        SurveyAgentType.Writer: LLMConfig(model=writer_model, model_type=writer_model_type, temperature=temperature),
        SurveyAgentType.Reviewer: LLMConfig(model=reviewer_model, model_type=reviewer_model_type, temperature=temperature),
    }

    config = SurveyContextConfig(
        subject=subject,
        ref_paths=ref_paths,
        llms=agent_llms,
        embed_model=embed_model,
        embed_model_type=embed_model_type,
        prompt_store_path=custom_prompt_store,
        tex_template_path=tex_template_path,
        
        save_path=save_path,
        reference_store_path=local_reference_store,
        
        tesseract_executable=tesseract_executable,
        
        no_ref_faiss=no_ref_faiss,
        no_review=no_review,
        no_figures=no_figures,
        no_reference=no_reference,
        no_abstract=no_abstract,
        no_tex_review=no_tex_review,
        
        ref_max_per_section=ref_max_per_section,
        ref_max_per_sentence=ref_max_per_sentence,
        ref_max_same_ref=ref_max_same_ref,
        fig_max_figures=fig_max_figures,
        
        structure_json_path=pregen_struct_json_path,
        prewritten_tex_path=prewritten_tex_path,
        bibdb_path=bibdb_path,
        
        faissbib_path=faissbib_path,
        images_dir=images_dir,
        faissfig_path=faissfig_path,
        faisscontent_path=faisscontent_path,
        faiss_confidence=faiss_confidence,
        llm_request_cooldown_sec=llm_request_cooldown_sec,
        embed_request_cooldown_sec=embed_request_cooldown_sec
    )

    survey_ctx = SurveyContext(config, status_queue=pipeline_status_queue)
    survey_ctx.config.save_yaml(os.path.join(survey_ctx.output_dir, "survey_config.yaml"))
    res = survey_ctx.generate()
    return res

def generate_survey_from_config(credentials_path: str, config_path: str, **pipeline_kwargs):
    setup_credentials(credentials_path)
    config = SurveyContextConfig.load_yaml(config_path)
    
    survey_ctx = SurveyContext(config, **pipeline_kwargs)
    survey_ctx.config.save_yaml(os.path.join(survey_ctx.output_dir, "survey_config.yaml"))

    res = survey_ctx.generate()
    return res
