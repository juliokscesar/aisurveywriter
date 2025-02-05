from typing import List, Union, Optional

from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.utils import named_log, countdown_log, diff_keys

class PaperWriter(PipelineTask):
    """
    A class abstraction for the process of writing a survey paper
    """
    def __init__(self, llm: LLMHandler, prompt: str, paper: Optional[PaperData] = None, ref_paths: Optional[List[str]] = None, request_cooldown_sec: int = 60 * 1.5):
        """
        Intializes a PaperWriter
        
        Parameters:
            llm (LLMHandler): the LLMHandler object with the LLM model to use
            
            prompt (str): the prompt to give specific instructions for the LLM to write this paper. The prompt must have the placeholders: {subject}, {title}, {description}
            
            paper (Optional[PaperData]): a PaperData object containing the paper information. Each section in paper.section must have the 'title' and 'description' filled, and 'content' will be filled by this task.
             if this is None, a PaperData must be provided when calling write()
            
            ref_paths (List[str]): a List of the path of every PDF reference for this paper.
        """
        self.llm = llm
        self.paper = paper
        self.prompt = prompt
        self.ref_paths = ref_paths.copy()
        
        self._cooldown_sec = int(request_cooldown_sec)
        
    def pipeline_entry(self, input_data: Union[PaperData, dict]) -> PaperData:
        """
        This is the entry point in the pipeline. 
        The input of this task should be either a PaperData with a "subject" set, and the "title" and "description" fields for every SectionData set;
        or it can be a dictionary with the format: {"subject": str, "sections": [{"title": str, "description": str}, {"title": str, "description": str}, ...]}

        Parameters:
            input_data (Union[PaperData, dict]): the input from the previous task (or from the first call).
        """
        if isinstance(input_data, dict):
            if diff := diff_keys(list(PaperData.__dict__.keys()), input_data):
                raise TypeError(f"Missing keys for input data (task {PaperWriter.__class__.__name__}): {", ".join(diff)}")
            
            paper = PaperData(
                subject=input_data["subject"],
                sections=[SectionData(title=s["title"], description=s["description"]) for s in input_data["sections"]],
            )
        else:
            paper = input_data
        
        paper = self.write(paper)
        return paper
    
    def write(self, paper: Optional[PaperData] = None, prompt: Optional[str] = None):
        """
        Write the provided paper.
        
        Parameters:
            paper (Optional[PaperData]): a PaperData object containing "subject" and each section "title" and "description". If this is None, try to use the one that was set in the constructor.
            
            prompt (Optional[str]): the prompt to invoke langchain's chain. If this is None, try to use the one set in the constructor. The prompt must have the placeholders "{subject}", "{title}" and "{description}"
        """
        if paper:
            self.paper = paper
        if self.paper is None:
            raise RuntimeError("The PaperData (PaperWriter.paper) must be set to write")
        if prompt:
            self.prompt = prompt
        
        # TODO: read reference content and initialize llm chain
        #self.llm.init_chain()
        
        sz = len(self.paper.sections)
        word_count = 0
        for i, section in enumerate(self.paper.sections):
            named_log(self, f"==> begin writing section ({i+1}/{sz}): {section.title}")
            response = self.llm.invoke({
                "subject": self.paper.subject,
                "title": section.title,
                "description": section.description,
            })
            section.content = response.content
            word_count += len(section.content.split())
            named_log(self, f"==> finished writing section ({i+1}/{sz}): {section.title} | total words count: {word_count}")
            named_log(self, f"==> response metadata:", response.usage_metadata)

            named_log(self, "==> initiating cooldown (request limitations)")
            countdown_log("", self._cooldown_sec)

        return self.paper