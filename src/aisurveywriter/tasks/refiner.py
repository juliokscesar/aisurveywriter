from typing import Optional
import re

from .pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils import named_log, time_func, countdown_log
import json

class PaperRefiner(PipelineTask):
    """
    Adds Title and Abstract to the paper
    """
    def __init__(self, llm: LLMHandler, paper: Optional[PaperData] = None, prompt: Optional[str] = None, cooldown_sec: int = 0):
        self.llm = llm
        self.paper = paper
        self.prompt = prompt

        self._cooldown_sec = cooldown_sec
        
    def pipeline_entry(self, input_data: PaperData):
        if not isinstance(input_data, PaperData):
            raise TypeError(f"Task {self.__class__.__name__} requires input data to be of type PaperData in pipeline entry, but got {type(input_data)}")
        return self.refine(input_data)
    
    def refine(self, paper: Optional[PaperData] = None, prompt: Optional[str] = None):
        if paper:
            self.paper = paper
        if prompt:
            self.prompt = prompt
            
        named_log(self, f"==> asking LLM to produce title and abstract")
        # this prompts takes the whole paper content + subject
        # and outputs title and abstract, maybe in a JSON format ({"title": ..., "abstract": ...})
        self.llm.init_chain(None, prompt)
        elapsed, response = time_func(self.llm.invoke, {
            "subject": self.paper.subject,
            "content": self.paper.full_content(),
        })

        named_log(self, f"==> got title and abstract from LLM | time elapsed: {elapsed} | metadata: {response.usage_metadata}")
        if self._cooldown_sec:
            named_log(self, f"==> initiating cooldown of {self._cooldown_sec} s")
            countdown_log("", self._cooldown_sec)

        try:
            resp_json = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response.content.strip())
            resp = json.loads(resp_json)
            # TODO: add title and abstract fields to PaperData, instead of this gambiarra
            self.paper.sections.insert(0, SectionData(
                title="Abstract",
                description="paper abstract",
                content=f"\\section{{Abstract}}\n{resp['abstract']}"
            ))
            self.paper.title = resp["title"]
        except:
            named_log(self, f"==> failed to parse JSON from LLM response. Adding abstract section as it is")
            self.paper.sections.insert(0, SectionData(
                title="Abstract",
                description="paper abstract",
                content=response.content.strip(),
            ))

        return self.paper