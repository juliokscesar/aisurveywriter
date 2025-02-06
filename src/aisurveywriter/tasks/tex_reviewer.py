from typing import Optional
import re

from .pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.paper import PaperData
from aisurveywriter.utils import named_log, countdown_log, time_func

class TexReviewer(PipelineTask):
    def __init__(self, llm: LLMHandler, tex_review_prompt: str, bib_review_prompt: Optional[str] = None, cooldown_sec: int = 40):
        """
        Initialize a TexReviewer
        
        Parameters:
            llm (LLMHandler): the llm to use
            tex_review_prompt (str): the prompt to send to LLM. This must be the full prompt (i.e. containing all instructions). This also must include the placeholders: {title}, {content}
            bib_review_prompt (str): the prompt to send to review the Bib file. This must be the full prompt (i.e. containing all instructions). This also must include the placeholders: {bibcontent}
        """
        self.llm = llm
        self.tex_prompt = tex_review_prompt
        self.bib_prompt = bib_review_prompt
        self._cooldown_sec = int(cooldown_sec)
        
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if not isinstance(input_data, PaperData):
            raise TypeError(f"{self.__class__.__name__} requires an input data of type PaperData in pipeline entry, got {type(input_data)} instead")
        return self.review(input_data)

    def review(self, paper: PaperData):
        paper = self.review_tex(paper)
        if self.bib_prompt is not None:
            paper = self.review_bib(paper)
        return paper
        
    def review_tex(self, paper: PaperData):
        """
        Review LaTeX syntax and commands section by section
        """
        self.llm.init_chain(None, self.tex_prompt)
        
        sz = len(paper.sections)
        for i, section in enumerate(paper.sections):
            if section.content is None:
                continue
            
            named_log(self, f"==> begin review LaTeX syntax for section ({i+1}/{sz}): {section.title}")
            elapsed, response = time_func(self.llm.invoke, {
                "content": section.content,
            })
            section.content = response.content
            named_log(self, f"==> finished review LaTeX syntax for section ({i+1}/{sz}): {section.title}")
            
            try:
                named_log(self, f"==> response metadata:", response.usage_metadata)
            except:
                named_log(self, f"==> (debug) reponse object:", response)

            section.content = re.sub(r"[`]+[\w]*", "", section.content)
            if self._cooldown_sec:
                cooldown = max(0, self._cooldown_sec - elapsed)
                named_log(self, f"==> initiating cooldown of {cooldown} s (request limitations)")
                countdown_log("", cooldown)

        return paper

    def review_bib(self, paper: PaperData):
        """
        Revew bib content
        """
        if self.bib_prompt is None:
            raise ValueError(f"The biblatex review prompt must be set (it's None)")
        if paper.bib is None:
            named_log(self, f"Requested BibTex review, but paper's bib is None. Doing nothing")
            return paper

        self.llm.init_chain(None, self.bib_prompt)
        
        named_log(self, "==> begin review of BibTex")
        elapsed, response = time_func(self.llm.invoke, {
            "bibcontent": paper.bib,
        })
        paper.bib = response.content
        named_log(self, "==> finished review of BibTex")
        named_log(self, f"==> response metadata:", response.usage_metadata)

        if self._cooldown_sec:
            countdown = max(0, self._cooldown_sec - elapsed)
            named_log(self, f"==> initiating cooldown of {countdown} s (request limitations)")
            countdown_log("", countdown)

        return paper