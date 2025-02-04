import os
import argparse
import re

from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.file_handler import FileHandler
from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.tasks.paper_generator import PaperStructureGenerator
from aisurveywriter.tasks.paper_writer import PaperWriter
from aisurveywriter.tasks.paper_reviewer import PaperReviewer
from aisurveywriter.utils.helpers import init_driver, get_all_files_from_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("references_dir", help="Path to directory containg all PDF references")
    parser.add_argument("--llm", "-l", choices=["openai", "google"], default="google", help="Specify LLM to use. Either 'google' (gemini-1.5-pro by default) or 'openai' (o1 by default)")
    parser.add_argument("--llm-model", "-m", dest="llm_model", default="gemini-1.5-pro", help="Specific LLM model to use.")
    parser.add_argument("--summarize", action="store_true", help="Use a summary of references instead of their whole content.")
    parser.add_argument("--faiss", action="store_true", help="Use FAISS vector store to retrieve information from references instead of their whole content. If this and 'summarize' are enabled, this will be ignored.")
    parser.add_argument("-c", "--config", default=os.path.abspath("config.yaml"), help="YAML file containg your configuration parameters")
    # TODO: review step depends on NBLM, so right now it will only work if structure and no-review are provided
    parser.add_argument("--structure", "-s", default=None, type=str, help="YAML file containing the structure to use. If provided, this will skip the structure generation process.")
    parser.add_argument("--no-review", dest="no_review", action="store_true", help="Disable the review step.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Setup common configuration
    config = ConfigManager.from_file(args.config)
    os.makedirs(config.out_dir, exist_ok=True)

    # Read credentials
    credentials = FileHandler.read_credentials(args.llm, config.credentials_path)
    os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    os.environ["OPENAI_API_KEY"] = credentials["openai_key"]

    # Get PDF paths for the references
    ref_pdf_paths = get_all_files_from_paths(args.references_dir)

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
    structgen = PaperStructureGenerator(nblm, config)
    structure = structgen.generate_structure(
        prompt=config.prompt_structure,
        sleep_wait_response=40,
        save_yaml=True,
        out_yaml_path=config.out_structure_path,
    )

    llm = LLMHandler(
        model=args.llm_model,
        model_type=args.llm,
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
        summarize_references=args.summarize,
        alternate_summarizer_llm=None,
        use_faiss=args.faiss,
        faiss_embeddings=args.llm,
        show_metadata=True,
    )
    writer.init_llm_context(ctx_msg, config.prompt_write)
    sections = writer.write_all_sections(
        show_metadata=True,
        sleep_between=int(60 * 1.3),
    )
    FileHandler.write_latex(
        template_path=config.tex_template_path,
        sections=sections,
        file_path=config.out_tex_path,
    )

    # Add a step in between to correct the biblatex file
    tex_file = config.out_tex_path
    bib_file = tex_file.replace(".tex", ".bib")
    with open(tex_file, "r", encoding="utf-8") as tf, open(bib_file, "r", encoding="utf-8") as bf:
        tex_content = tf.read()
        bib_content = bf.read()
    latex_review = llm.send_prompt(f"""The following is the content of a .tex and a .bib file. 
                                   
                                   # FOR THE .tex FILE:
                                   - You must read it and check for any duplicates and invalid citations, cross-checking them with the entries from the .bib file
                                   - Check for any latex syntax errors, and if you find any BibLatex entries in the .tex file, remove it and put it in the .bib block
                                   
                                   - Change if find any:
                                    - \\subsection* or \\section* -> \\subsection or \\section
                                    - Remove any \\title, \\author, \\date

                                   # FOR THE .bib FILE:
                                   - Double check for duplicate citation entries and invalid ones
                                   - If the DOI or any ther method of accessing the work in the entry is provided, check the veracity of the Bibtex entry
                                   
                                   **You must keep a correct LaTeX syntax and formatting. Do not modify any part of the document preamble, only within the sections and subsections.**
                                   
                                   **OUTPUT FORMAT**: you must output your response exactly as follows, knowing that "<texfilecontent>" and "<bibfilecontent>" are placeholders. It is essential that you follow this format.
                                   >BEGINTEXFILE()
                                   <texfilecontent>
                                   >ENDTEXFILE()

                                   >BEGINBIBFILE()
                                   <bibfilecontent>
                                   >ENDBIBFILE()

                                   Here are the file contents:
                                   > .tex:
                                   {tex_content}
                                   
                                   > .bib:
                                   {bib_content}
                                   """)
    print("LATEX REVIEW:", latex_review)
    print('\n'*3)
    tex_content = re.search(r'>BEGINTEXFILE\(\)\n(.*?)\n>ENDTEXFILE\(\)', latex_review.content, re.DOTALL).group(1)
    bib_content = re.search(r'>BEGINBIBFILE\(\)\n(.*?)\n>ENDBIBFILE\(\)', latex_review.content, re.DOTALL).group(1)

    new_tex_file = "last-techrev.tex"
    new_bib_file = new_tex_file.replace(".tex", ".bib")
    tex_content = re.sub(r'\\usebibresource\{.*?\}', f'\\usebibresource{{{new_bib_file}}}', tex_content)
    with open(new_tex_file, "w", encoding="utf-8") as tf, open(new_bib_file, "w", encoding="utf-8") as bf:
        tf.write(tex_content)
        bf.write(bib_content)
    return

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
        summarize_ref=args.summarize,
        use_faiss=args.faiss,
        faiss_embeddings=args.llm,
        show_metadata=True,
        sleep_between=int(60 * 1.3)
    )
    FileHandler.write_latex(
        template_path=config.tex_template_path,
        sections=reviewed,
        file_path=config.out_reviewed_tex_path,
    )

    # Do the same techincal revision step here

    driver.quit()

if __name__ == "__main__":
    main()