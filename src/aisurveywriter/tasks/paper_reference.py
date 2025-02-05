import os

from .pipeline_task import PipelineTask

from aisurveywriter.core.paper import PaperData
from aisurveywriter.core.llm_handler import LLMHandler

class PaperReferencer(PipelineTask):
    def __init__(self, llm: LLMHandler, paper: PaperData, bibdb_path: str, prompt: str):
        self.llm = llm
        self.paper = paper
        self.bibdb_path = bibdb_path
        self.prompt = prompt

        if not os.path.isfile(bibdb_path):
            raise FileNotFoundError(f"Unable to find file {bibdb_path}")
        
    def pipeline_entry(self, input_data):
        if not isinstance(input_data, PaperData):
            raise TypeError(f"Task {self.__class__.__name__} requires input of type PaperData in pipe entry")
        
        paper = self.reference(paper, self.bibdb_path, self.prompt)
        return paper
    
    def reference(self, paper: PaperData = None, bibdb_path: str = None, prompt: str = None):
        if paper is not None:
            self.paper = paper
        if bibdb_path is not None:
            self.bibdb_path = bibdb_path
        if prompt:
            self.prompt = prompt
            
        