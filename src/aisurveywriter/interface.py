import os
import argparse
import gradio as gr
import threading
import queue
import time

from aisurveywriter import generate_paper_survey
from aisurveywriter.utils import named_log
from aisurveywriter.core.pipeline import TaskStatus

g_generation_success = False
def run_generation(output_queue, *args, **kwargs):
    global g_generation_success
    try:
        g_generation_success = True
        generate_paper_survey(*args, **kwargs, pipeline_status_queue=output_queue)
    except Exception as e:
        print(f"generate_paper_survey raised an exception: {e}")
        g_generation_success = False
        raise e
        

# TODO: ADJUST TO NEW REFACTORED CODE
class GradioInterface:
    def __init__(self):
        self._back_thread = None
        self.supported_models = {
            "Gemini 1.5 Flash": ("google", "gemini-1.5-flash"),
            "Gemini 1.5 Pro": ("google", "gemini-1.5-pro"),
            "Gemini 2.0 Flash": ("google", "gemini-2.0-flash"),
            "Gemini 2.0 Flash-Lite": ("google", "gemini-2.0-flash-lite-preview-02-05"),
            "Gemini 2.0 Pro Exp": ("google", "gemini-2.0-pro-exp"),
            "OpenAI (TODO)": (None,None),
            "Deepseek-R1 32b": ("ollama", "deepseek-r1:32b"),
        }
        self.supported_text_embedding = {
            "Google Text Embedding 004": ("google", "models/text-embedding-004"),
            "OpenAI (TODO)": (None, None),
            "SFR-Embedding-Mistral (HuggingFace)": ("huggingface", "Salesforce/SFR-Embedding-Mistral"),
            "all-MiniLM-L6-v2 (HuggingFace)": ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
            "gte-Qwen2-1.5b (HuggingFace)": ("huggingface", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"),
            "all-mpnet-base-v2": ("huggingface", "sentence-transformers/all-mpnet-base-v2"),
            "multilingual-e5-large-instruct": ("huggingface", "intfloat/multilingual-e5-large-instruct"),
            "snowflake-arctic-embed-l-v2.0": ("huggingface", "Snowflake/snowflake-arctic-embed-l-v2.0"),
        }
        self.gr_interface = gr.ChatInterface(
            fn=self.chat_fn,
            type="messages",
            textbox=gr.Textbox(placeholder="Enter the subject of the survey paper..."),
            additional_inputs=[
                gr.File(label="Upload reference PDFs", file_types=[".pdf"], file_count="multiple"),
                gr.Textbox(label="Save path and name", placeholder="Enter the full path to save the paper (including its filename)", value=os.path.join(os.getcwd(), "out")),
                gr.Dropdown(label="Writer LLM model", choices=list(self.supported_models.keys())),
                gr.Dropdown(label="Text embedding model", choices=list(self.supported_text_embedding.keys()), info="Text embedding model used in retrieving relevant references from the FAISS (RAG) in the step of adding them to the paper."),
                gr.Number(label="LLM Request cooldown (seconds)", info="Cooldown time between two consecutive requests to LLM API. It is important to adjust this according to the amount of tokens, since depending on the LLM it may take a while to reset", value=int(40)),
                gr.Number(label="Text Embedding request cooldown (seconds)", value=int(0)),
                gr.Textbox(label="Pre-generated JSON structure (one will be generated if none is provided)", placeholder="Full path to pre-generated structure"),
                gr.Textbox(label="Pre-written .TEX paper (one will be written from the structure if none is provided)", placeholder="Full path to pre-written paper .tex"),
                gr.Checkbox(label="Skip review step"),
                gr.Checkbox(label="Skip add figures step"),
                gr.Checkbox(label="Skip add references step"),
                gr.Checkbox(label="Skip add abstract and title step"),
                gr.Checkbox(label="Skip TEX review step"),
                gr.Textbox(label="Path to .bib database to use (one will be generated from the references, if none is provided)", placeholder="Full path to BibTex database", value=os.path.abspath(os.path.join(__file__, "../../../out/generated-bibdb.bib"))),
                gr.Textbox(label="Path to a local FAISS vector store from the .bib database. If not provided, one will be created"),
                gr.Textbox(label="Path to local FAISS vector store of every figure and its description, accross all references (if none is provided, one will be created)"),
                gr.Textbox(label="Path to directory containing images used in FAISS figure. If none is provided, one will be created"),
                gr.Textbox(label="Path to local FAISS vector store of references' content. If not provided, one will be created"),
                gr.Checkbox(label="Don't use FAISS for PDF references", info="Use the entire PDFs' content instead of creating a RAG (not recommended, as it explodes the number of tokens)", value=False),
                gr.Number(value=float(0.6), label="Confidence score threshold for FAISS similarity search"),
            ],
            title="Survey Paper Writer",
            description="Provide a subject, reference PDFs, and a save path to generate and save a survey paper.",
        )
        self._is_running = False

    def launch(self, *args, **kwargs):
        self.gr_interface.launch(*args, **kwargs)

    def chat_fn(self, message, history, refs, save_path, llm_model, embed_model, llm_request_cooldown, embed_request_cooldown, 
                pregen_json_struct, prewritten_tex_paper, no_review, no_figures, no_reference, no_abstract, no_tex_review, 
                bibdb_path, faissbib_path, faissfig_path, images_dir, faisscontent_path, no_ref_faiss, faiss_confidence):
        if self._is_running:
            return "A paper survey generation is already running. Please wait -- this really takes a long time."
        if len(refs) == 0:
            return "Please provide at least one PDF reference"
        subject = message

        status_queue = queue.Queue()
        worker_thread = threading.Thread(
            target=run_generation, 
            args=(status_queue,),
            kwargs={
                "subject": subject,
                "ref_paths": refs,
                "save_path": save_path,
                
                "writer_model": llm_model[1],
                "writer_model_type": llm_model[0],
                "reviewer_model": llm_model[1], # use same model for write and review for now
                "reviewer_model_type": llm_model[0],
                
                "embed_model": embed_model[1],
                "embed_model_type": embed_model[0],
                
                "no_ref_faiss": no_ref_faiss,
                "no_review": no_review,
                "no_figures": no_figures,
                "no_reference": no_reference,
                "no_abstract": no_abstract,
                "no_tex_review": no_tex_review,
                
                "pregen_struct_json_path": pregen_json_struct,
                "prewritten_tex_path": prewritten_tex_paper,
                
                "bibdb_path": bibdb_path,
                "faissbib_path": faissbib_path,
                "faissfig_path": faissfig_path,
                "faisscontent_path": faisscontent_path,
                "faiss_confidence": faiss_confidence,
                
                "images_dir": images_dir,
                
                "llm_request_cooldown_sec": llm_request_cooldown,
                "embed_request_cooldown_sec": embed_request_cooldown,

                "pipeline_status_queue": status_queue
            }
        )
        worker_thread.start()
        self._is_running = False
        try:
            self._is_running = True
            while worker_thread.is_alive() or not status_queue.empty():
                while not status_queue.empty():
                    task_id, task_name, task_status = status_queue.get()
                    notify_msg = f"Task {task_id + 1}, \"{task_name}\", {'is running' if task_status==TaskStatus.RUNNING else 'completed'}"
                    named_log(self, notify_msg)
                    yield notify_msg
                    time.sleep(2)
        except Exception as e:
            named_log(self, f"generate_paper_survey raised an exception. Stopped generating paper. Exception: {e}")
            self._is_running = False
            yield f"Unable to generate paper: {e}"
            
        self._is_running = False
        global g_generation_success
        if g_generation_success:
            yield "Paper generated successfully and saved to " + save_path
        else:
            yield "Please try again"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--share", action="store_true", help="Launch gradio with share=True (create public link)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    interface = GradioInterface()
    interface.launch(share=args.share)
