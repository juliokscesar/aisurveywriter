from typing import List, Optional
import os
import psutil

import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pipeline import PaperPipeline
import aisurveywriter.tasks as tks
from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.utils import init_driver
from aisurveywriter.core.paper import PaperData

def get_credentials(config: ConfigManager):
    credentials = fh.read_credentials(config.credentials_path)
    os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    os.environ["OPENAI_API_KEY"] = credentials["openai_key"]
    return credentials

def generate_paper_survey(subject: str, ref_paths: List[str], save_path: str, model: str = "gemini-1.5-flash", model_type: str = "google", pregen_struct_yaml: Optional[str] = None, config_path: Optional[str] = None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(__file__, "../../../config.yaml"))
    config = ConfigManager.from_file(config_path)

    credentials = get_credentials(config)
    if save_path is None or save_path == "":
        save_path = config.out_tex_path
    
    # LLM used to write the paper
    writer_llm = LLMHandler(
        model=model,
        model_type=model_type,
    )
    
    # LLM used on LaTeX syntax review (deepseek is local)
    # use deepseek only if there's enough memory
    if psutil.virtual_memory().available < 9*1024:
        tex_review_llm = LLMHandler(
            model="deepseek-coder-v2:16b",
            model_type="ollama",
        )
    else:
        tex_review_llm = writer_llm
    
    save_path = os.path.abspath(save_path)
    if not os.path.basename(save_path).endswith(".tex"):
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "generated.tex")

    if pregen_struct_yaml:
        first = tks.DeliverTask(PaperData.from_structure_yaml(subject=subject, path=pregen_struct_yaml))
    else:
        driver = init_driver(config.browser_path, config.driver_path)
        nblm = NotebookLMBot(
            user=credentials["nblm_email"],
            password=credentials["nblm_password"],
            driver=driver,
            src_paths=ref_paths,
        )
        nblm.login()
        first = tks.PaperStructureGenerator(nblm, subject, config.prompt_structure, save_path=save_path.replace(".tex", "-struct.yaml"))
    
    pipe = PaperPipeline([
        first,
        
        tks.PaperWriter(writer_llm, config.prompt_write, ref_paths=ref_paths),
        tks.PaperSaver(save_path.replace(".tex", "-rawscratch.tex"), config.tex_template_path),
        
        tks.TexReviewer(tex_review_llm, config.prompt_tex_review, bib_review_prompt=None),
        tks.PaperSaver(save_path.replace(".tex", "-texrevscratch.tex"), config.tex_template_path),
        
        tks.PaperReviewer(writer_llm, config.prompt_review, config.prompt_apply_review, ref_paths=ref_paths),
        tks.PaperSaver(save_path.replace(".tex", "-rev.tex"), config.tex_template_path),
        
        tks.TexReviewer(tex_review_llm, config.prompt_tex_review, bib_review_prompt=None),
        tks.PaperSaver(save_path, config.tex_template_path)
    ])
    pipe.run()
