import os
import argparse
import re

from aisurveywriter import generate_paper_survey
from aisurveywriter.utils import get_all_files_from_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("references_dir", help="Path to directory containg all PDF references")
    parser.add_argument("subject", help="Main subject of the survey. Can be the Title too")
    parser.add_argument("--llm", "-l", choices=["openai", "google"], default="google", help="Specify LLM to use. Either 'google' (gemini-1.5-pro by default) or 'openai' (o1 by default)")
    parser.add_argument("--llm-model", "-m", dest="llm_model", default="gemini-1.5-pro", help="Specific LLM model to use.")
    # parser.add_argument("--summarize", action="store_true", help="Use a summary of references instead of their whole content.")
    # parser.add_argument("--faiss", action="store_true", help="Use FAISS vector store to retrieve information from references instead of their whole content. If this and 'summarize' are enabled, this will be ignored.")
    parser.add_argument("-c", "--config", default=os.path.abspath("config.yaml"), help="YAML file containg your configuration parameters")
    parser.add_argument("--structure", "-s", default=None, type=str, help="YAML file containing the structure to use. If provided, this will skip the structure generation process.")
    # parser.add_argument("--no-review", dest="no_review", action="store_true", help="Disable the review step.")

    return parser.parse_args()

def main():
    args = parse_args()

    generate_paper_survey(
        subject=args.subject,
        ref_paths=get_all_files_from_paths(args.references_dir),
        model=args.llm_model,
        model_type=args.llm,
        pregen_struct_yaml=args.structure,
        config_path=args.config,
    )

if __name__ == "__main__":
    main()