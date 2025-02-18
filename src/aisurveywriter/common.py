from typing import List, Optional
import os
import queue
import re

import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.reference_store import ReferenceStore
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import AgentRAG
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.text_embedding import EmbeddingsHandler
from aisurveywriter.core.pipeline import PaperPipeline
import aisurveywriter.tasks as tks
from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.utils import init_driver, time_func
from aisurveywriter.core.paper import PaperData

def tex_filter_survey(tex_content: str):
    tex_content = re.sub(r"[`]+[\w]*", "", tex_content)
    tex_content = tex_content.replace("\\begin{document}", "")
    tex_content = tex_content.replace("\\end{document}", "")
    tex_content = tex_content.replace("\\begin{filecontents}", "")
    tex_content = tex_content.replace("\\end{filecontents}", "")
    tex_content = re.sub(r"\\usepackage\{([^}]+)\}", "", tex_content)
    return tex_content

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
    use_ref_faiss=False,
    pregen_struct_yaml: Optional[str] = None, 
    prewritten_paper_tex: Optional[str] = None,
    no_review: bool = False,
    config_path: Optional[str] = None, 
    use_nblm_generation=False,
    bibdb_path: str = None,
    faissbibdb_path: str = None,
    faissfig_path: str = None,
    imgs_path: str = None,
    embed_model: str = "Salesforce/SFR-Embedding-Mistral",
    embed_model_type: str = "huggingface",
    pipeline_status_queue: queue.Queue = None,
    request_cooldown_sec: int = int(60 * 1.5),
    embed_request_cooldown_sec: int = int(20),
    faiss_confidence: float = 0.6,
    no_figures = False,
):
    # Initialize configuration and credentials
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
        
    # ensure save_path refers to the final .tex file, even if the user provided a directory as save path
    save_path = os.path.abspath(save_path)
    if not os.path.basename(save_path).endswith(".tex"):
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, "generated.tex")
    ###########################################

    # Text embedding model. This is the same accross all tasks
    print(f"Loading embed model: {embed_model}, {embed_model_type}")
    embed = EmbeddingsHandler(model, model_type)
    print(f"Loaded embeddings: {embed_model}")

    # LLM used in generating sections outline
    # ==> using gemini-2.0-flash as it has a very large context window, and we can use the entire PDFs content
    outline_sections_llm = LLMHandler(
        model="gemini-2.0-flash",
        model_type="google",
        temperature=0.4,
    )
    print(f"Loaded LLM (sections outline): gemini-2.0-flash")

    # LLM used in writing
    writer_llm = LLMHandler(
        model=model,
        model_type=model_type,
        temperature=0.4,
    )
    print(f"Loaded LLM (writer): {model}")
    
    # LLM used in reviewing (TODO: some configuration to make this custom)
    review_llm = writer_llm
    print(f"Loaded LLM (reviewer): {model}")
    
    # Create reference store
    references = ReferenceStore(ref_paths)
    
    # Create/Load RAGs accordingly
    shared_rag = AgentRAG(embed, faissbibdb_path, faissfig_path, )
    
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
            
        first = tks.PaperStructureGenerator(gen_llm, ref_paths, subject, config.prompt_structure, save_path=save_path.replace(".tex", "-struct.yaml"))
        first_name = "Paper Structure Generator"

    if not bibdb_path:
        refs_llm = LLMHandler(model="gemini-2.0-flash", model_type="google", temperature=0.3) # use gemini-2.0-flash for references because of quota
        bibdb_path = save_path.replace(".tex", "-bibdb.bib")
        ref_extract = tks.ReferenceExtractor(refs_llm, ref_paths, config.prompt_ref_extract,
                                             raw_save_path=save_path.replace(".tex", "-raw.ref"),
                                             rawbib_save_path=save_path.replace(".tex", "-raw.bib"),
                                             bib_save_path=bibdb_path)
        ref_extract_name = "Reference Extractor"
    else:
        ref_extract = tks.DeliverTask()
        ref_extract_name = "Load References Database"

    if prewritten_paper_tex:
        write_step = tks.LoadTask(prewritten_paper_tex, subject)
        write_step_name = "Load Paper .tex"
        next_write = tks.DeliverTask()
        next_write_name = "Input paper loaded to review"
    else:
        write_step = tks.PaperWriter(writer_llm, config.prompt_write, ref_paths=ref_paths, discard_ref_section=True, request_cooldown_sec=request_cooldown_sec, use_faiss=use_ref_faiss, embedding=embed)
        write_step_name = "Write Paper"
        next_write = tks.PaperSaver(save_path.replace(".tex", "-rawscratch.tex"), config.tex_template_path)
        next_write_name = "Save Paper"

    if not no_review:
        review_step = tks.PaperReviewer(writer_llm, config.prompt_review, config.prompt_apply_review, ref_paths=ref_paths, request_cooldown_sec=request_cooldown_sec, use_faiss=use_ref_faiss, embeddings=embed)
        review_step_name = "Review Paper"
        
        next_review = tks.PaperSaver(save_path.replace(".tex", "-rev.tex"), config.tex_template_path)
        next_review_name = "Save Reviewed Paper"
    else:
        review_step = tks.DeliverTask()
        review_step_name = "Skip review"
        
        next_review = tks.DeliverTask()
        next_review_name = "Skip review save"
        
    if not faissfig_path:
        faissfig_path = save_path.replace(".tex", f"-{embed_model}-imgfaiss")
        fig_extract = tks.FigureExtractor(writer_llm, embed, subject, ref_paths, save_dir=save_path.replace(".tex", "-usedimgs"), 
                                        faiss_save_path=faissfig_path, local_faiss_path=faissfig_path,
                                        imgs_dir=imgs_path, request_cooldown_sec=embed_request_cooldown_sec)
        fig_extract_name = "Extract figures from references"
    else:
        fig_extract = tks.DeliverTask()
        fig_extract_name = "Skip figure extract"

    if no_figures:
        fig_add = tks.DeliverTask()
        fig_add_name = "Skip figure add"

        fig_add_next = tks.DeliverTask()
        fig_add_next_name = "Skip save paper with figures"
    else:
        fig_add = tks.PaperFigureAdd(gen_llm, embed, faissfig_path, imgs_path, ref_paths, config.prompt_fig_add, os.path.dirname(save_path), 
                                    llm_cooldown=request_cooldown_sec, embed_cooldown=embed_request_cooldown_sec, max_figures=15, confidence=0.75)
        fig_add_name = "Add Figures"
        
        fig_add_next = tks.PaperSaver(save_path.replace(".tex", "-figs.tex"), config.tex_template_path)
        fig_add_next_name = "Save paper with figures"
        
    pipe = PaperPipeline([
        (first_name, first),
        
        (write_step_name, write_step),
        (next_write_name, next_write),
        
        (fig_extract_name, fig_extract),

        (fig_add_name, fig_add),      
        (fig_add_next_name, fig_add_next),
        
        (review_step_name, review_step),
        (next_review_name, next_review),
        
        (ref_extract_name, ref_extract),
        
        ("Add References", tks.PaperFAISSReferencer(embed, bibdb_path, local_faissdb_path=faissbibdb_path, save_usedbib_path=save_path.replace(".tex", ".bib"), 
                                            save_faiss_path=save_path.replace(".tex", f"-{embed_model}-bibfaiss"), max_per_section=80, max_per_sentence=4, confidence=faiss_confidence, max_same_ref=10)),
        ("Save Paper with References", tks.PaperSaver(save_path.replace(".tex", "-revref.tex"), config.tex_template_path)),
        
        ("Refine (Abstract+Tile)", tks.PaperRefiner(writer_llm, prompt=config.prompt_refine, cooldown_sec=request_cooldown_sec)),
        
        ("Review Tex", tks.TexReviewer(os.path.dirname(save_path), cooldown_sec=request_cooldown_sec)),
        ("Save Final Paper", tks.PaperSaver(save_path, config.tex_template_path, bib_path=save_path.replace(".tex", ".bib"), tex_filter_fn=tex_filter_survey)),
    ], status_queue=pipeline_status_queue)
    
    print(f"==> BEGINNING PAPER SURVEY GENERATION PIPELINE WITH {len(pipe.steps)} STEPS")
    elapsed, _ = time_func(pipe.run)
    print(f"==> PAPER SURVEY GENRATION PIPELINE FINISHED | TIME TAKEN: {elapsed} s")
