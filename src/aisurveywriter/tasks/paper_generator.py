from typing import List, Optional, Union
import yaml
import re

from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.core.llm_handler import LLMHandler
import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils import named_log, countdown_log

from .pipeline_task import PipelineTask

class PaperStructureGenerator(PipelineTask):
    def __init__(self, llm: Union[NotebookLMBot, LLMHandler], ref_paths: List[str], subject: str, prompt: str, save_path: Optional[str] = None):
        super().__init__()
        self.no_divide = True
        self.llm = llm
        self.subject = subject
        self.prompt = prompt
        self.save_path = save_path
        self.ref_paths = ref_paths.copy()

        if isinstance(llm, NotebookLMBot) and not self.llm.is_logged_in():
            self.is_nblm = True
            self.llm.login()
            self.llm.add_sources(ref_paths)
        else:
            self.is_nblm = False

    def __call__(self, prompt: Optional[str] = None, sleep_wait_response: int = 30, save_path: Optional[str] = None):
        return self.generate_structure(prompt, sleep_wait_response, save_path)

    def pipeline_entry(self, input_data):
        sections = self.generate_structure()
        paper = PaperData(
            subject=self.subject,
            sections=[SectionData(s["title"], s["description"]) for s in sections]
        )
        return paper

    def generate_structure(self, prompt: Optional[str] = None, sleep_wait_response: int = 30, save_path: Optional[str] = None) -> List[dict[str,str]]:
        """
        Generate a structure for the paper using NotebookLM with the sources provided with (ref_pdf_paths).
        """
        if save_path:
            self.save_path = save_path
        
        if prompt is not None:
            self.prompt = prompt
        if self.prompt.find("{subject}") != -1:
            self.prompt = self.prompt.replace("{subject}", self.subject)
        
        named_log(self, f"==> started generating paper structure")
        if self.is_nblm:
            response = self._nblm_gen_structure(sleep_wait_respose=sleep_wait_response)
        else:
            response = self._llm_gen_structure(return_metadata=True)
            named_log(self, "LLM generation metadata:", response.usage_metadata)
            response = response.content
            named_log(self, "==> initiating cooldown of 40 s because of request limitations")
            countdown_log("", 40)
        response = re.sub(r"[`]+[\w]*", "", response)
        named_log(self, f"==> finished generating paper structure")

        # format response
        result = "sections:\n"
        for line in response.split("\n"):
            result += f"  {line}\n"

            if self.save_path:
                with open(self.save_path, "w", encoding="utf-8") as f:
                    f.write(result)
        
        result = yaml.safe_load(result)
        named_log(self,f"Finished generating paper structure. Got a structure with {len(result['sections'])} sections.\n")
        return result["sections"]
    
    def _nblm_gen_structure(self, sleep_wait_response: int = 30):
        if not self.is_nblm or not isinstance(self.llm, NotebookLMBot):
            raise TypeError(f"Private function _nblm_gen_structure must be used with NotebookLM")
        self.llm.send_prompt(self.prompt, sleep_for=sleep_wait_response)
        return self.llm.get_last_response()

    def _llm_gen_structure(self, return_metadata = False):
        if self.is_nblm or not isinstance(self.llm, LLMHandler):
            raise TypeError(f"Private function _llm_gen_structure must be used with LLMHandler")
        
        # First need to get PDF contents
        pdf_contents = PDFProcessor(self.ref_paths).extract_content()
        # remove references section from each pdf (decreases token count)
        for i in range(len(pdf_contents)):
            ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", pdf_contents[i], re.IGNORECASE)
            if ref_match:
                pdf_contents[i] = pdf_contents[i][:ref_match.start()].strip()
        
        input_prompt = "\n".join(pdf_contents) + "\n\n" + self.prompt
        response = self.llm.send_prompt(input_prompt)
        if return_metadata:
            return response
        return response.content
    
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
