from aisurveywriter.tasks.paper_faiss_ref import PaperFAISSReferencer, PaperData
from aisurveywriter.core.text_embedding import load_embeddings

path = "/home/juliocesar/dev/aisurveywriter"
model = "nomic-ai/nomic-embed-text-v2-moe"
embed = load_embeddings(model, "huggingface", trust_remote_code=True)
ref = PaperFAISSReferencer(embed, "../bib/refextract-21papers.bib", local_faissdb_path=None, save_usedbib_path="temp/test.bib", 
                           save_faiss_path=f"{path}/bib/{model[model.find('/')+1:]}-bibfaiss", max_per_section=60, max_per_sentence=1,confidence=0.7)
