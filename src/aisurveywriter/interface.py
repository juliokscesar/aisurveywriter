import gradio as gr

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

    def chat_fn(self, message, history, refs, save_path):
        return "not implemented"