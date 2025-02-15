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
            "snowflake-arctic-embed-l-v2.0": ("hugginface", "Snowflake/snowflake-arctic-embed-l-v2.0"),
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
                gr.Number(label="Request cooldown (seconds)", info="Cooldown time between two consecutive requests. It is important to adjust this according to the amount of tokens, since depending on the LLM it may take a while to reset", value=int(90)),
                gr.Textbox(label="Pre-generated YAML structure (one will be generated if none is provided)", placeholder="Full path to pre-generated structure"),
                gr.Textbox(label="Pre-written .TEX paper (one will be written from the structure if none is provided)", placeholder="Full path to pre-written paper .tex"),
                gr.Checkbox(label="Skip content/writing review step"),
                gr.Textbox(label="Path to YAML configuration file", placeholder="Full path to configuration", value = os.path.abspath(os.path.join(__file__, "../../../config.yaml"))),
                gr.Checkbox(label="Use NotebookLM to generate the paper structure", info="Slow compared to bare LLM models, but supports up to 50 PDFs", value=False),
                gr.Textbox(label="Path to .bib database to use (one will be generated from the references, if none is provided)", placeholder="Full path to BibTex database", value=os.path.abspath(os.path.join(__file__, "../../../out/generated-bibdb.bib"))),
                gr.Textbox(label="Path to a local FAISS vector store from the .bib database. If not provided, one will be created"),
                gr.Textbox(label="Path to local FAISS vector store of every figure and its description, accross all references (if none is provided, one will be created)"),
                gr.Textbox(label="Path to directory containing images used in FAISS figure. If none is provided, one will be created"),
                gr.Number(value=float(0.6), label="Confidence score threshold for FAISS similarity search"),
                gr.Checkbox(label="Use FAISS for PDF references", info="Use FAISS to retrieve only a part of the references, instead of the entire PDF", value=False),
            ],
            title="Survey Paper Writer",
            description="Provide a subject, reference PDFs, and a save path to generate and save a survey paper.",
        )
        self._is_running = False

    def launch(self, *args, **kwargs):
        self.gr_interface.launch(*args, **kwargs)

    def chat_fn(self, message, history, refs, save_path, model, embed_model, req_cooldown_sec, pregen_struct, prewritten_paper, no_review, config_path, nblm_generate, bibdb_path, faissdb_path, faissfig_path, imgs_dir, faiss_confidence, use_ref_faiss):
        if self._is_running:
            return "A paper survey generation is already running. Please wait -- this really takes a long time."
        if len(refs) == 0:
            return "Please provide at least one PDF reference"
        subject = message

        named_log(self, "DEBUG:", message, history, refs, save_path, model, req_cooldown_sec, pregen_struct, prewritten_paper, no_review, config_path, nblm_generate, bibdb_path)
        
        status_queue = queue.Queue()
        worker_thread = threading.Thread(
            target=run_generation, 
            args=(status_queue,),
            kwargs={
                "subject": subject,
                "ref_paths": refs,
                "save_path": save_path.strip(),
                "model": self.supported_models[model][1],
                "model_type": self.supported_models[model][0],
                "pregen_struct_yaml": pregen_struct.strip(),
                "prewritten_paper_tex": prewritten_paper.strip(),
                "use_ref_faiss": use_ref_faiss,
                "no_review": no_review,
                "config_path": config_path.strip(),
                "use_nblm_generation": nblm_generate,
                "refdb_path": bibdb_path,
                "faissdb_path": faissdb_path,
                "faissfig_path": faissfig_path,
                "imgs_path": imgs_dir,
                "embed_model": self.supported_text_embedding[embed_model][1],
                "embed_model_type": self.supported_text_embedding[embed_model][0],
                "request_cooldown_sec": int(req_cooldown_sec),
                "faiss_confidence": float(faiss_confidence),
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
