import os

from aisurveywriter.tasks.figure_extractor import FigureExtractor, PaperData
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.file_handler import read_credentials, read_yaml
from aisurveywriter.core.text_embedding import load_embeddings
from aisurveywriter.utils import get_all_files_from_paths

path = "/home/juliocesar/aisurveywriter"
os.environ["GOOGLE_API_KEY"] = read_credentials(f"{path}/credentials.yaml")["google_key"]

img_data = read_yaml(f"{path}/bib/refextract-imgdata.yaml")["data"]
img_data = [img for img in img_data if "OliveiraO2022" not in img["path"]]

llm = LLMHandler(model="gemini-2.0-flash", model_type="google")
embed = load_embeddings("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "huggingface")
fig = FigureExtractor(
    llm, embed, 
    "Langmuir and Langmuir-Blodgett films", 
    pdf_paths=get_all_files_from_paths(f"{path}/refexamples", stem_sort=True), 
    save_dir=f"./save", faiss_save_path=f"{path}/bib/qwen2-1.5B-nochuimgfaiss", 
    local_faiss_path=None, paper=None, request_cooldown_sec=15,
)
faiss = fig.create_faiss(img_data)
