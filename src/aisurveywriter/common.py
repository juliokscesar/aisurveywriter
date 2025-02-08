from typing import List, Optional
import os
import queue

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
    if credentials["google_key"]:
        os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    if credentials["openai_key"]:
        os.environ["OPENAI_API_KEY"] = credentials["openai_key"]
    return credentials

def generate_paper_survey(
    subject: str, 
    ref_paths: List[str], 
    save_path: str, 
    model: str = "gemini-1.5-flash", 
    model_type: str = "google", 
    pregen_struct_yaml: Optional[str] = None, 
    prewritten_paper_tex: Optional[str] = None,
    config_path: Optional[str] = None, 
    use_nblm_generation=False,
    refdb_path: str = None,
    pipeline_status_queue: queue.Queue = None,
    request_cooldown_sec: int = int(60 * 1.5),
):
    if not config_path:
        config_path = os.path.abspath(os.path.join(__file__, "../../../config.yaml"))
    try:
        config = ConfigManager.from_file(config_path)
    except Exception as e:
        print("==> Exception on reading config:", e)
        raise e

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
    # if psutil.virtual_memory().available >= 9*1024:
    #     tex_review_llm = LLMHandler(
    #         model="deepseek-coder-v2:16b",
    #         model_type="ollama",
    #     )
    # else:
    tex_review_llm = writer_llm
    
    save_path = os.path.abspath(save_path)
    if not os.path.basename(save_path).endswith(".tex"):
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "generated.tex")

    if pregen_struct_yaml:
        first = tks.DeliverTask(PaperData.from_structure_yaml(subject=subject, path=pregen_struct_yaml))
        first_name = "Load YAML structure"
    else:
        if use_nblm_generation:
            driver = init_driver(config.browser_path, config.driver_path)
            nblm = NotebookLMBot(
                user=credentials["nblm_email"],
                password=credentials["nblm_password"],
                driver=driver,
                src_paths=ref_paths,
            )
            nblm.login()
            gen_llm = nblm
        else:
            gen_llm = writer_llm
            
        first = tks.PaperStructureGenerator(gen_llm, ref_paths, subject, config.prompt_structure, save_path=save_path.replace(".tex", "-struct.yaml"))
        first_name = "Paper Structure Generator"

    refs_llm = LLMHandler(model="gemini-2.0-flash-exp", model_type="google", temperature=0.3) # use gemini-2.0-flash-exp for references because of quota
    if not refdb_path:
        refdb_path = save_path.replace(".tex", "-bibdb.bib")
        ref_extract = tks.ReferenceExtractor(refs_llm, ref_paths, config.prompt_ref_extract,
                                             raw_save_path=save_path.replace(".tex", "-raw.ref"),
                                             rawbib_save_path=save_path.replaec(".tex", "-raw.bib"),
                                             bib_save_path=refdb_path)
        ref_extract_name = "Reference Extractor"
    else:
        if not os.path.isfile(refdb_path):
            raise FileNotFoundError(f"Unable to find file {refdb_path!r}")
        ref_extract = tks.DeliverTask()
        ref_extract_name = "Load References Database"

    if prewritten_paper_tex:
        write_step = tks.DeliverTask(PaperData.from_tex(prewritten_paper_tex, subject))
        write_step_name = "Load Paper .tex"
        next_write = tks.DeliverTask()
        next_write_name = "Input paper loaded to review"
    else:
        write_step = tks.PaperWriter(writer_llm, config.prompt_write, ref_paths=ref_paths, discard_ref_section=True, request_cooldown_sec=request_cooldown_sec)
        write_step_name = "Write Paper"
        next_write = tks.PaperSaver(save_path.replace(".tex", "-rawscratch.tex"), config.tex_template_path, find_bib_pattern=None)
        next_write_name = "Save Paper"
        
    pipe = PaperPipeline([
        (first_name, first),
        
        (write_step_name, write_step),
        (next_write_name, next_write),
        
        ("Review Paper", tks.PaperReviewer(writer_llm, config.prompt_review, config.prompt_apply_review, ref_paths=ref_paths, request_cooldown_sec=request_cooldown_sec)),
        ("Save Reviewed Paper", tks.PaperSaver(save_path.replace(".tex", "-rev.tex"), config.tex_template_path, find_bib_pattern=None)),
        
        (ref_extract_name, ref_extract),
        
        ("Add References", tks.PaperReferencer(refs_llm, bibdb_path=refdb_path,
                            prompt=config.prompt_ref_add, cooldown_sec=75, save_usedbib_path=save_path.replace(".tex", "-bib.bib"))),
        ("Save Paper with References", tks.PaperSaver(save_path.replace(".tex", "-revref.tex"), config.tex_template_path, find_bib_pattern=None)),
        
        ("Refine (Abstract+Tile)", tks.PaperRefiner(writer_llm, prompt=config.prompt_refine, cooldown_sec=request_cooldown_sec)),
        
        ("Review Tex", tks.TexReviewer(tex_review_llm, config.prompt_tex_review, bib_review_prompt=None, cooldown_sec=request_cooldown_sec)),
        ("Save Final Paper", tks.PaperSaver(save_path, config.tex_template_path, find_bib_pattern=None)),
    ], status_queue=pipeline_status_queue)
    
    print("==> BEGINNING PAPER SURVEY GENERATON PIPELINE")
    pipe.run()
    print("==> PAPER SURVEY GENRATION PIPELINE FINISHED")
