import os
import argparse
import gradio as gr
import threading
import queue
import time

# TODO: ADJUST TO NEW REFACTORED CODE
class GradioInterface:
    def __init__(self):
        self._back_thread = None
        self.supported_models = {
            "Gemini 1.5 Flash": ("google", "gemini-1.5-flash"),
            "Gemini 1.5 Pro": ("google", "gemini-1.5-pro"),
            "Gemini 2.0 Flash": ("google", "gemini-2.0-flash"),
            "Gemini 2.0 Flash-Lite": ("google", "gemini-2.0-flash-lite-preview-02-05"),
            "Gemini 2.0 Pro": ("google", "gemini-2.0-pro"),
            "Gemini 2.5 Flash": ("google", "gemini-2.5-flash"),
            "Gemini 2.5 Pro": ("google", "gemini-2.5-pro"),
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

    def __call__(self, **kwargs):
        self.launch_interface(**kwargs)
        
    def launch_interface(self, **gr_kwargs):
        global demo
        with gr.Blocks(title="AI Survey Writer") as demo:
            gr.Markdown("# AI Survey Writer\nGenerate comprehensive survey papers using AI agents")

            with gr.Row():
                # Left Column - Input Section
                with gr.Column(scale=1):
                    gr.Markdown("**Research Subject**")
                    subject_input = gr.Textbox(placeholder="e.g. Quantum Computing", label="")

                    gr.Markdown("**Reference Papers/Files**")
                    files_input = gr.File(file_types=[".pdf", ".txt", ".docx"], file_count="multiple", label="Upload Files")

                    # Display uploaded files (simulating UI look)
                    uploaded_files = gr.Dataframe(headers=["Filename"], datatype=["str"], label="Uploaded Files")

                # Middle Column - Pipeline Section
                with gr.Column(scale=1):
                    gr.Markdown("**AI Pipeline**")

                    parse_status = gr.Label(value="Idle", label="Parse & Summarize")
                    cluster_status = gr.Label(value="Idle", label="Cluster Topics")
                    generate_status = gr.Label(value="Idle", label="Generate Survey")

                    run_button = gr.Button("▶ Run Pipeline")
                    reset_button = gr.Button("↺ Reset Inputs")

                # Right Column - Output Preview
                with gr.Column(scale=1):
                    with gr.Tab("Survey"):
                        survey_output = gr.Textbox(label="Survey Preview", lines=20)
                    with gr.Tab("Logs"):
                        logs_output = gr.Textbox(label="Logs", lines=20)
                    with gr.Tab("Citations"):
                        citations_output = gr.Textbox(label="Citations", lines=20)

                    download_button = gr.Button("⬇ Download Draft")

            # Logic handlers (basic placeholders)
            def run_pipeline(subject, files):
                return (
                    "Running...", "Running...", "Running...",
                    "This is a survey preview based on your input.",
                    "Pipeline started...", "Citations will appear here."
                )

            def reset_inputs():
                return "", None, "", "", ""

            run_button.click(run_pipeline, [subject_input, files_input],
                            [parse_status, cluster_status, generate_status,
                            survey_output, logs_output, citations_output])

            reset_button.click(reset_inputs, [], [
                subject_input, files_input,
                survey_output, logs_output
            ])
                        
        demo.launch(**gr_kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--share", action="store_true", help="Launch gradio with share=True (create public link)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    interface = GradioInterface()
    interface.launch_interface(share=args.share)
