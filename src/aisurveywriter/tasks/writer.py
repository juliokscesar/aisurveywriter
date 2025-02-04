from typing import List, Union

from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler

class PaperWriter(PipelineTask):
    """
    A class abstraction for the process of writing a survey paper
    """
    def __init__(self, llm: LLMHandler, paper: PaperData, prompt: str, ref_contents: str = None):
        """
        Intializes a PaperWriter
        
        Parameters:
            llm (LLMHandler): the LLMHandler object with the LLM model to use
            paper (PaperData): a PaperData object containing the paper information. Each section in paper.section must have the 'title' and 'description' filled, and 'content' will be filled by this task.
            ref_contents (str): a string containing all the content for every reference provided.
            prompt (str): the prompt to give specific instructions for the LLM to write this paper. The prompt must have the placeholders: {subject}, {title}, {description}
        """
        self.llm = llm
        self.paper = paper
        self.prompt = prompt
        self.ref_content = ref_contents
        
    def pipeline_entry(self, input_data):
        pass

    def 