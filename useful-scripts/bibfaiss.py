from aisurveywriter.tasks.paper_faiss_ref import PaperFAISSReferencer, PaperData
from aisurveywriter.core.text_embedding import load_embeddings
from aisurveywriter.core.latex_handler import write_latex

model = "Salesforce/SFR-Embedding-Mistral"
embed = load_embeddings(model, "huggingface")
ref = PaperFAISSReferencer(embed, "../bib/refextract-21papers.bib", local_faissdb_path=None, save_usedbib_path="temp/test.bib", 
                           save_faiss_path=f"temp/{model[model.find('/')+1:]}", max_per_section=60, max_per_sentence=1,confidence=0.7)
