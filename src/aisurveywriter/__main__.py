import os
import argparse
import re

from aisurveywriter import generate_paper_survey
from aisurveywriter.utils import get_all_files_from_paths
from aisurveywriter.store.prompt_store import PromptStore

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("references_dir", help="Path to directory containg all PDF references")
    parser.add_argument("subject", help="Main subject of the survey. Can be the Title too")
    parser.add_argument("--save-dir", type=str, default="./out", help="Path to output directory")
    parser.add_argument("--llm", "-l", choices=["openai", "google"], default="google", help="Specify LLM to use. Either 'google', 'openai' or 'ollama'. Default is google")
    parser.add_argument("--llm-model", "-m", dest="llm_model", default="gemini-2.0-flash", help="Specific LLM model to use. Default is gemini-2.0-flash")
    parser.add_argument("--credentials", default="credentials.yaml", help="YAML file containing your API keys")
    parser.add_argument("--structure", "-s", default=None, type=str, help="JSON file containing the structure to use. If provided, this will skip the structure generation process.")
    parser.add_argument("--paper", "-p", default=None, help="Path to .TEX paper to use. If provided, won't write one from the structure, and will skip directly to reviewing it (unless --no-review) is provided")
    parser.add_argument("--embed-model", "-e", default="Snowflake/snowflake-arctic-embed-l-v2.0", help="Text embedding model name. Default is Snowflake/snowflake-arctic-embed-l-v2.0")
    parser.add_argument("--embed-type", "-t", default="huggingface", help="Text embedding model type (google, openai, huggingface)")
    parser.add_argument("--bibdb", "-b", type=str, default=None, help="Path to .bib database to use. If none is provided, one will be generated by extracting every reference across all PDFs")
    parser.add_argument("--faissbib", "-fb", type=str, default=None, help="Path to FAISS vector store of the .bib databse. If none is provided, one will be generated")
    parser.add_argument("--images", "-i", type=str, default=None, help="Path to all images extracted from the PDFs. If none is provided, all images will be extracted and saved to a temporary folder")
    parser.add_argument("--faissfig", "-ff", type=str, default=None, help="Path to FAISS vector store containing the metadata (id, path and description) for every image. If none is provided, one will be created")
    parser.add_argument("--faissref", "-fr", type=str, help="Path to FAISS of references contents to retrieve only a piece of information, instead of sending the entire document.")
    parser.add_argument("--no-ref-rag", action="store_true", help="Don't create a RAG for reference contents. Use entire PDF instead")
    parser.add_argument("--no-figures", action="store_true", help="Skip step of adding figures to the written paper.")
    parser.add_argument("--no-reference", action="store_true", help="Skip step of adding references to the text")
    parser.add_argument("--no-abstract", action="store_true", help="Skip step of writing Abstract and Title")
    parser.add_argument("--no-tex-review", action="store_true", help="Skip TEX review")
    parser.add_argument("--no-review", action="store_true", help="Skip content/writing review step")
    parser.add_argument("--cooldown", "-w", type=int, default=30, help="Cooldown between two consecutive requests made to the LLM API")
    parser.add_argument("--embed-cooldown", type=int, default=0, help="Cooldown between two consecutive requests made to the text embedding model API")
    parser.add_argument("--tex-template", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../templates/paper_template.tex")), help="Path to custom .tex template")
    parser.add_argument("--prompt-store", type=str, default=None, help="Path to a JSON containing custom prompts used in the system. If none is provided, the default prompts are used")
    parser.add_argument("--ref-max-section", type=int, default=100, help="Maximum references in one section. Default is 90"),
    parser.add_argument("--ref-max-sentence", type=int, default=4, help="Maximum references in one sentences. Default is 4"),
    parser.add_argument("--ref-max-same", type=int, default=12, help="Maximum repetitions of the same reference. Default is 12"),
    parser.add_argument("--fig-max", type=int, default=30, help="Maximum number of Figures to add (not counting TikZ generated).")
    parser.add_argument("--tesseract", type=str, default="tesseract", help="Tessearact executable/command")
    parser.add_argument("--reference-store", type=str, default=None, help="Path to local .pkl reference store")
    return parser.parse_args()

def main():
    args = parse_args()

    # load prompt store content if provided
    custom_prompt_store = None
    if args.prompt_store:
        with open(args.prompt_store, "r", encoding="utf-8") as f:
            custom_prompt_store = PromptStore.model_validate_json(f.read())

    generate_paper_survey(
        subject=args.subject,
        ref_paths=get_all_files_from_paths(args.references_dir, stem_sort=True),
        save_path=os.path.abspath(args.save_dir),
        
        writer_model=args.llm_model,
        writer_model_type=args.llm,
        reviewer_model=args.llm_model,
        reviewer_model_type=args.llm,
        
        embed_model=args.embed_model,
        embed_model_type=args.embed_type,

        custom_prompt_store=custom_prompt_store,
        tex_template_path=args.tex_template,
        
        local_reference_store=args.reference_store,
        tesseract_executable=args.tesseract,
        
        no_ref_faiss=args.no_ref_rag,
        no_review=args.no_review,
        no_figures=args.no_figures,
        no_reference=args.no_reference,
        no_abstract=args.no_abstract,
        no_tex_review=args.no_tex_review,
        
        pregen_struct_json_path=args.structure,
        prewritten_tex_path=args.paper,
        
        bibdb_path=args.bibdb,
        faissbib_path=args.faissbib,
        faissfig_path=args.faissfig,
        faisscontent_path=args.faissref,
        
        images_dir=os.path.abspath(args.images) if args.images else None,

        llm_request_cooldown_sec=args.cooldown,
        embed_request_cooldown_sec=args.embed_cooldown,

        ref_max_per_section=args.ref_max_section,
        ref_max_per_sentence=args.ref_max_sentence,
        ref_max_same_ref=args.ref_max_same,
    )


if __name__ == "__main__":
    main()