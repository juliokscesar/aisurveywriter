from typing import List, Optional
import yaml

from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.chatbots import NotebookLMBot
import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils import named_log

from .pipeline_task import PipelineTask

class PaperStructureGenerator(PipelineTask):
    def __init__(self, nblm_bot: NotebookLMBot, subject: str, prompt: str):
        self.nblm = nblm_bot
        self.subject = subject
        self.prompt = prompt

        if not self.nblm.is_logged_in():
            self.nblm.login()

    def pipeline_entry(self, input_data):
        sections = self.generate_structure()
        paper = PaperData(
            subject=self.subject,
            sections=[SectionData(s["title"], s["description"]) for s in sections]
        )
        return paper

    def generate_structure(self, prompt: Optional[str] = None, sleep_wait_response: int = 30, save_yaml = True, out_yaml_path: str = "genstructure.yaml") -> List[dict[str,str]]:
        """
        Generate a structure for the paper using NotebookLM with the sources provided with (ref_pdf_paths).
        """
        if prompt is not None:
            self.prompt = prompt
        if self.prompt.find("{subject}") != -1:
            self.prompt = self.prompt.replace("{subject}", self.subject)
        
        self.nblm.send_prompt(self.prompt, sleep_for=sleep_wait_response)
        response = self.nblm.get_last_response()

        # format response
        result = "sections:\n"
        for line in response.split("\n"):
            result += f"  {line}\n"

        if save_yaml:
            with open(out_yaml_path, "w", encoding="utf-8") as f:
                f.write(result)
        
        result = yaml.safe_load(result)
        named_log(self,f"Finished generating paper structure. Got a structure with {len(result['sections'])} sections.\n")
        return result["sections"]
    
    @staticmethod
    def load_structure(sturcture_yaml: str) -> List[dict[str,str]]:
        sections = fh.read_yaml(sturcture_yaml)["sections"]
        return sections
    
    @staticmethod
    def yaml_to_paperdata(structure_yaml: str) -> PaperData:
        yaml_sections = fh.read_yaml(structure_yaml)["sections"]
        paper = PaperData(
            subject="None",
            sections=[SectionData(s["title"], s["description"]) for s in yaml_sections]
        )
        return paper
