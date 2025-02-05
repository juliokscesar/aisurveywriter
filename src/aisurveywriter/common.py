from typing import List
import os

import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pipeline import PaperPipeline
import aisurveywriter.tasks as tks
from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.utils import init_driver

def get_credentials(config: ConfigManager):
    credentials = fh.read_credentials(config.credentials_path)
    os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    os.environ["OPENAI_API_KEY"] = credentials["openai_key"]
    return credentials

def generate_paper_survey(subject: str, ref_paths: List[str], save_path: str, model: str = "gemini-1.5-flash", model_type: str = "google"):
    cwd = os.getcwd()
    config_path = "D:/Dev/aisurveywriter/config.yaml"
    os.chdir(os.path.dirname(config_path))
    config = ConfigManager.from_file("D:/Dev/aisurveywriter/config.yaml")
    os.chdir(cwd)

    credentials = get_credentials(config)
    if save_path is None or save_path == "":
        save_path = config.out_tex_path
    
    driver = init_driver(config.browser_path, config.driver_path)
    nblm = NotebookLMBot(
        user=credentials["nblm_email"],
        password=credentials["nblm_password"],
        driver=driver,
        src_paths=ref_paths,
    )
    nblm.login()
    
    llm = LLMHandler(
        model=model,
        model_type=model_type,
    )
    
    save_path = os.path.abspath(save_path)
    if not os.path.basename(save_path).endswith(".tex"):
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "generated.tex")
    
    pipe = PaperPipeline([
        tks.PaperStructureGenerator(nblm, subject, config.prompt_structure),
        
        tks.PaperWriter(llm, config.prompt_write, ref_paths=ref_paths),
        tks.PaperSaver(save_path.replace(".tex", "-rawscratch.tex"), config.tex_template_path),
        
        tks.TexReviewer(llm, config.prompt_tex_review, config.prompt_bib_review),
        tks.PaperSaver(save_path.replace(".tex", "-texrevscratch.tex"), config.tex_template_path),
        
        tks.PaperReviewer(llm, config.prompt_review, config.prompt_apply_review, ref_paths=ref_paths),
        tks.PaperSaver(save_path.replace(".tex", "-rev.tex"), config.tex_template_path),
        
        tks.TexReviewer(llm, config.prompt_tex_review, config.prompt_bib_review),
        tks.PaperSaver(save_path, config.tex_template_path)
    ])
    pipe.run()
