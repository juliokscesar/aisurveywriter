import os
import argparse

import undetected_chromedriver as uc

from .core.config_manager import ConfigManager
from .core.file_handler import FileHandler
from .core.chatbots import NotebookLMBot
from .core.llm_handler import LLMHandler
from .tasks.paper_generator import PaperStructureGenerator
from .tasks.paper_writer import PaperWriter
from .tasks.paper_reviewer import PaperReviewer
from .utils.helpers import init_driver, get_all_files_from_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("references_dir", help="Path to directory containg all PDF references")
    parser.add_argument("--llm", choices=["openai", "google"], default="google", help="Specify LLM to use. Either 'google' (gemini-1.5-pro) (default) or 'openai' (o1)")
    parser.add_argument("--llm-model", dest="llm_model", default="gemini-1.5-pro", help="Specific LLM model to use.")
    parser.add_argument("--summarize", action="store_true", help="Use a summary of references instead of their whole content.")
    parser.add_argument("--faiss", action="store_true", help="Use FAISS vector store to retrieve information from references instead of their whole content. If this and 'summarize' are enabled, this will be ignored.")
    return parser.parse_args()

def main():
    # Setup common configuration
    config = ConfigManager(
        browser_path="C:/Users/jcmcs/AppData/Local/BraveSoftware/Brave-Browser/Application/brave.exe",
        driver_path="../../drivers/chromedriver.exe",
        tex_template_path="../../templates/paper_template.tex",
        prompt_cfg_path="../../templates/prompt_config.yaml",
        review_cfg_path="../../templates/review_config.yaml",
        out_dir="../../out",
        out_structure_filename="laststrucutre.yaml",
        out_tex_filename="lastpaper.tex",
        out_dump_filename="lastpaper.dump",
    )
    os.makedirs("../../out", exist_ok=True)
    args = parse_args()

    # Read credentials
    if not os.path.isfile("../../credentials.yaml"):
        raise FileExistsError("File 'credentials.yaml' must exist with your credentials")
    credentials = FileHandler.read_yaml("../../credentials.yaml")
    os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    os.environ["OPENAI_API_KEY"] = credentials["openai_key"]

    # Get PDF paths for the references
    ref_pdf_paths = get_all_files_from_paths(args["references_dir"])

    # Initialize NotebookLM
    driver = init_driver(config.browser_path, config.driver_path)

    nblm = NotebookLMBot(
        user=credentials["nblm_email"],
        password=credentials["nblm_password"],
        driver=driver,
        src_paths=ref_pdf_paths,
    )
    nblm.login()

    # Generate paper structure
    structgen = PaperStructureGenerator(nblm)
    structure = structgen.generate_structure(
        prompt=config.prompt_structure,
        sleep_wait_response=40,
        save_yaml=True,
        out_yaml_path=config.out_structure_path,
    )

    llm = LLMHandler(
        model=args["llm_model"],
        model_type=args["llm"],
    )

    # Write paper sections
    writer = PaperWriter(
        subject=config.paper_subject,
        sections_structure=structure,
        llm=llm,
        pdf_refrences=ref_pdf_paths,
        config=config,
    )
    ctx_msg = writer.create_context_sysmsg(
        header_prompt=config.prompt_response_format,
        summarize_references=args["summarize"],
        alternate_summarizer_llm=None,
        use_faiss=args["faiss"],
        faiss_embeddings=args["llm"],
        show_metadata=True,
    )
    writer.init_llm_context(ctx_msg, config.prompt_write)
    sections = writer.write_all_sections(
        show_metadata=True,
        sleep_between=int(60 * 1.3),
        save_latex=True
    )

    # Review paper sections
    reviewer = PaperReviewer(
        subject=config.paper_subject,
        sections_content=sections,
        pdf_references=ref_pdf_paths,
        nblm=nblm,
        llm=llm,
        config=config,
    )
    reviewed = reviewer.improve_all_sections(
        save_latex=True,
        summarize_ref=args["summarize"],
        use_faiss=args["faiss"],
        faiss_embeddings=args["llm"],
        show_metadata=True,
        sleep_between=int(60 * 1.3)
    )

    driver.quit()

if __name__ == "__main__":
    main()