import gradio as gr

from aisurveywriter import generate_paper_survey

class GradioInterface:
    def __init__(self):
        self.gr_interface = gr.ChatInterface(
            fn=self.chat_fn,
            textbox=gr.Textbox(placeholder="Enter the subject of the survey paper..."),
            additional_inputs=[
                gr.File(label="Upload reference PDFs", file_types=[".pdf"], file_count="multiple"),
                gr.Textbox(label="Save path and name", placeholder="Enter the full path to save the paper (including its filename)"),
            ],
            title="Survey Paper Writer",
            description="Provide a subject, reference PDFs, and a save path to generate and save a survey paper.",
        )

    def launch(self):
        self.gr_interface.launch()

    def chat_fn(self, message, history, refs, save_path):
        subject = message
        try:
            generate_paper_survey(
                subject=subject,
                ref_paths=refs,
                save_path=save_path,
                model="gemini-1.5-flash",
                model_type="google",
            )
            return "Paper generated successfully and saved to " + save_path
        except Exception as e:
            return f"Unable to generate paper: {e}"

if __name__ == "__main__":
    interface = GradioInterface()
    interface.launch()
