from typing import List
import yaml

from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.core.file_handler import FileHandler

class PaperStructureGenerator:
    def __init__(self, nblm_bot: NotebookLMBot, config: ConfigManager):
        self.nblm = nblm_bot
        self.config = config

        if not self.nblm.is_logged_in():
            self.nblm.login()

    def _print(self, *msgs):
        print(f"({self.__class__.__name__})", *msgs)

    def generate_structure(self, prompt: str, sleep_wait_response: int = 40, save_yaml = True, out_yaml_path: str = "genstructure.yaml") -> List[dict[str,str]]:
        """
        Generate a structure for the paper using NotebookLM with the sources provided with (ref_pdf_paths).
        """
        if prompt.find("{subject}") != -1:
            prompt = prompt.replace("{subject}", self.config.paper_subject)
        
        self.nblm.send_prompt(prompt, sleep_for=sleep_wait_response)
        response = self.nblm.get_last_response()

        # format response
        result = "sections:\n"
        for line in response.split("\n"):
            result += f"  {line}\n"

        if save_yaml:
            with open(out_yaml_path, "w", encoding="utf-8") as f:
                f.write(result)
        
        result = yaml.safe_load(result)
        self._print(f"Finished generating paper structure. Got a structure with {len(result['sections'])} sections.\n")
        return result["sections"]
    
    @staticmethod
    def load_structure(sturcture_yaml: str) -> List[dict[str,str]]:
        sections = FileHandler.read_yaml(sturcture_yaml)["sections"]
        return sections
