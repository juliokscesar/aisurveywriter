from typing import List, Optional, Union
import yaml
import re
import json

from langchain_core.prompts.chat import HumanMessagePromptTemplate

from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.core.llm_handler import LLMHandler
import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log
from aisurveywriter.utils.helpers import time_func

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


    def divide_subtasks(self, n, input_data=None):
        raise NotImplemented()
    
    def merge_subtasks_data(self, data):
        raise NotImplemented()


class PaperStructureGenerator(PipelineTask):
    required_input_variables: List[str] = ["subject", "refcontent"]
    
    def __init__(self, agent_ctx: AgentContext, paper_with_subject: PaperData, save_json_path: Optional[str] = None):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        
        self.agent_ctx._working_paper = paper_with_subject
        self.save_json_path = save_json_path
        
    def generate(self, save_json_path: Optional[str] = None) -> dict[str,List[dict[str,str]]]:
        named_log(self, f"==> start generating structure for paper on subject {self.agent_ctx._working_paper.subject!r}")
        
        self.agent_ctx.llm_handler.init_chain_messages(
            HumanMessagePromptTemplate.from_template(self.prompt.text),
        )
        elapsed, response = self.agent_ctx.llm_handler.invoke({
            "subject": self.agent_ctx._working_paper.subject,
            "refcontent": self.agent_ctx.references.full_content(discard_bibliography=True),
        })

        try:
            resp_json = re.search(r"```json\s*([\s\S]+?)\s*```|({[\s\S]+})", response.content.strip()).group()
            resp_json = resp_json[resp_json.find("{"):resp_json.rfind("}")+1]
            structure = json.loads(structure)
        except Exception as e:
            named_log(self, f"==> failed to parse JSON from LLM response. Raising exception")
            raise e
            
        metadata_log(self, elapsed, response)
        if self.agent_ctx.llm_cooldown:
            cooldown_log(self, self.agent_ctx.llm_cooldown)

        if save_json_path:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(structure, f)

        named_log(self, f"==> finish generating structure")
        return structure


    def pipeline_entry(self, input_data):
        if input_data:
            self.agent_ctx._working_paper = input_data
        
        structure = self.generate(self.save_json_path)
        self.agent_ctx._working_paper.sections = [
            SectionData(title=s["title"], description=s["description"]) for s in structure["sections"]
        ]
        return self.agent_ctx._working_paper
