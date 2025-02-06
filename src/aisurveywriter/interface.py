import os
import gradio as gr

from aisurveywriter import generate_paper_survey
from aisurveywriter.utils import named_log

class GradioInterface:
    def __init__(self):
        self.gr_interface = gr.ChatInterface(
            fn=self.chat_fn,
            textbox=gr.Textbox(placeholder="Enter the subject of the survey paper..."),
            additional_inputs=[
                gr.File(label="Upload reference PDFs", file_types=[".pdf"], file_count="multiple"),
                gr.Textbox(label="Save path and name", placeholder="Enter the full path to save the paper (including its filename)", value=os.path.join(os.getcwd(), "out")),
                gr.Textbox(label="Writer LLM model", placeholder="Enter the name of the model to use to write and review the paper", value="gemini-2.0-flash-exp"),
                gr.Textbox(label="Pre-generated YAML structure", placeholder="Full path to pre-generated structure"),
                gr.Textbox(label="Path to YAML configuration file", placeholder="Full path to configuration", value = os.path.abspath(os.path.join(__file__, "../../../config.yaml"))),
            ],
            title="Survey Paper Writer",
            description="Provide a subject, reference PDFs, and a save path to generate and save a survey paper.",
        )

    def launch(self):
        self.gr_interface.launch()

    def chat_fn(self, message, history, refs, save_path, model, pregen_struct, config_path):
        subject = message
        model = model.strip().lower()
        if "gemini" in model:
            model_type = "google"
        elif "o1" in model or "o3" in model or "gpt" in model:
            model_type = "openai"
        elif "deepseek" in model:
            model_type = "ollama"
        else:
            return f"Model vendor {model_type!r} for model {model} is either invalid or unsupported. Please provide models only from google, openai or deepseek"
        named_log(self, f"Got: message={message}, history={history}, refs={refs}, save_path={save_path}, model={model}, pregen_struct={pregen_struct}, config_path={config_path}")
        try:
            generate_paper_survey(
                subject=subject,
                ref_paths=refs,
                save_path=save_path.strip(),
                model=model,
                model_type=model_type,
                pregen_struct_yaml=pregen_struct.strip(),
                config_path=config_path.strip(),
            )
            return "Paper generated successfully and saved to " + save_path
        except Exception as e:
            named_log(self, f"generate_paper_survey raised an exception. Stopped generating paper. Exception: {e}")
            return f"Unable to generate paper: {e}"

if __name__ == "__main__":
    interface = GradioInterface()
    interface.launch()
