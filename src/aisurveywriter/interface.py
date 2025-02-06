import os
import sys
import gradio as gr

from aisurveywriter import generate_paper_survey
from aisurveywriter.utils import named_log

class GradioInterface:
    def __init__(self):
        self.supported_models = {
            "Gemini 1.5 Flash": ("google", "gemini-1.5-flash"),
            "Gemini 1.5 Pro": ("google", "gemini-1.5-pro"),
            "Gemini 2.0 Flash Exp": ("google", "gemini-2.0-flash-exp"),
            "OpenAI (TODO)": (None,None),
            "Deepseek-R1 32b": ("ollama", "deepseek-r1:32b"),
        }
        self.gr_interface = gr.ChatInterface(
            fn=self.chat_fn,
            textbox=gr.Textbox(placeholder="Enter the subject of the survey paper..."),
            additional_inputs=[
                gr.File(label="Upload reference PDFs", file_types=[".pdf"], file_count="multiple"),
                gr.Textbox(label="Save path and name", placeholder="Enter the full path to save the paper (including its filename)", value=os.path.join(os.getcwd(), "out")),
                # gr.Textbox(label="Writer LLM model", placeholder="Enter the name of the model to use to write and review the paper", value="gemini-2.0-flash-exp"),
                gr.Dropdown(label="Writer LLM model", choices=list(self.supported_models.keys())),
                gr.Textbox(label="Pre-generated YAML structure", placeholder="Full path to pre-generated structure"),
                gr.Textbox(label="Path to YAML configuration file", placeholder="Full path to configuration", value = os.path.abspath(os.path.join(__file__, "../../../config.yaml"))),
                gr.Checkbox(label="Use NotebookLM to generate the paper structure", info="Slow compared to bare LLM models, but supports up to 50 PDFs", value=False),
            ],
            title="Survey Paper Writer",
            description="Provide a subject, reference PDFs, and a save path to generate and save a survey paper.",
        )

    def launch(self, *args, **kwargs):
        self.gr_interface.launch(*args, **kwargs)

    def chat_fn(self, message, history, refs, save_path, model, pregen_struct, config_path, nblm_generate):
        if len(refs) == 0:
            return "Please provide at least one PDF reference"
        subject = message
        named_log(self, f"Got: message={message}, history={history}, refs={refs}, save_path={save_path}, model={model}, pregen_struct={pregen_struct}, config_path={config_path}")
        try:
            generate_paper_survey(
                subject=subject,
                ref_paths=refs,
                save_path=save_path.strip(),
                model=self.supported_models[model][1],
                model_type=self.supported_models[model][0],
                pregen_struct_yaml=pregen_struct.strip(),
                # config_path=config_path.strip(),
                config_path=config_path.strip(),
                use_nblm_generation=nblm_generate,
            )
            return "Paper generated successfully and saved to " + save_path
        except Exception as e:
            named_log(self, f"generate_paper_survey raised an exception. Stopped generating paper. Exception: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            named_log(self, "Exception full description:", exc_type, fname, exc_tb.tb_lineno)
            return f"Unable to generate paper: {e}"

if __name__ == "__main__":
    interface = GradioInterface()
    interface.launch(share=True)
