from typing import Optional, List
import re
import os
import time

from .pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils import named_log, countdown_log, time_func

class TexReviewer(PipelineTask):
    def __init__(self, out_dir: str, paper: Optional[PaperData] = None, cooldown_sec: int = 40):
        # self.llm = llm
        # self.tex_prompt = tex_review_prompt
        # self.bib_prompt = bib_review_prompt
        self.paper = paper
        self._cooldown_sec = int(cooldown_sec)
        self.out_dir = out_dir
        
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if not isinstance(input_data, PaperData):
            raise TypeError(f"{self.__class__.__name__} requires an input data of type PaperData in pipeline entry, got {type(input_data)} instead")
        return self.review(input_data)

    def review(self, paper: PaperData):
        self.paper = paper
        self.paper = self.review_tex(self.paper)
        return self.paper
        
    def review_tex(self, paper: PaperData):
        """
        Review LaTeX syntax and commands section by section
        """
        # self.llm.init_chain(None, self.tex_prompt)
        
        sz = len(paper.sections)
        
        with open(paper.bib_path, "r", encoding="utf-8") as f:
            bib_content = f.read()
        
        for i, section in enumerate(paper.sections):
            if section.content is None:
                continue
            
            named_log(self, f"==> begin review LaTeX syntax for section ({i+1}/{sz}): {section.title}")
            # elapsed, response = time_func(self.llm.invoke, {
            #     "content": section.content,
            # })
            # section.content = response.content
            
            # try:
            #     named_log(self, f"==> response metadata:", response.usage_metadata)
            # except:
            #     named_log(self, f"==> (debug) reponse object:", response)
            
            start = time.time()
            
            section.content = re.sub(r"[`]+[\w]*", "", section.content)
            section = self._remove_invalid_figures(section)
            section = self._remove_invalid_refs(section, bib_content)

            elapsed = int(time.time() - start)
            
            # if self._cooldown_sec:
            #     named_log(self, f"==> initiating cooldown of {self._cooldown_sec} s (request limitations)")
            #     countdown_log("", self._cooldown_sec)
            named_log(self, f"==> finished review LaTeX syntax for section ({i+1}/{sz}): {section.title} | time elapsed: {elapsed} s")

        return paper


    def _remove_invalid_figures(self, section: SectionData):
        img_dir = re.search(r"\\graphicspath{[\s\S]*\{([\s\S]*)\}", section.content, re.DOTALL)
        if not img_dir:
            named_log(self, "==> couldn't find image directory. Skipping _remove_invalid_figures")
            return section
        img_dir = os.path.abspath(os.path.join(self.out_dir, img_dir.group(1)))
        
        # Regex to find all \begin{figure} ... \end{figure} blocks
        figure_pattern = re.compile(r'\\begin{figure}.*?\\includegraphics(?:\[.*?\])?{(.*?)}.*?\\end{figure}', re.DOTALL)
        
        def replace_invalid_figure(match):
            nonlocal img_dir
            image_path = match.group(1).strip()
            full_image_path = os.path.join(img_dir, image_path)
            
            if not os.path.exists(full_image_path):
                return ''  # Remove the entire figure block if image doesn't exist
            return match.group(0)  # Keep the figure block if the image exists
        
        # Replace invalid figure blocks
        section.content = figure_pattern.sub(replace_invalid_figure, section.content)
    
        return section
    

    def _remove_invalid_refs(self, section: SectionData, bib_content: str):
        cite_pattern = re.compile(r"\\cite{([^}]+)}")
        def replace_invalid(match):
            keys = match.group(1).split(',')
            valid_keys = [key.strip for key in keys if key.strip() in bib_content]
            if valid_keys:
                return f"\\cite{{{', '.join(valid_keys)}}}"
            else:
                return ""
        
        section.content = cite_pattern.sub(replace_invalid, section.content)
        return section


    def review_bib(self, paper: PaperData):
        """
        Revew bib content
        """
        if self.bib_prompt is None:
            raise ValueError(f"The biblatex review prompt must be set (it's None)")
        if paper.bib_path is None:
            named_log(self, f"Requested BibTex review, but paper's bib is None. Doing nothing")
            return paper

        self.llm.init_chain(None, self.bib_prompt)
        
        named_log(self, "==> begin review of BibTex")
        elapsed, response = time_func(self.llm.invoke, {
            "bibcontent": paper.bib_path,
        })
        paper.bib_path = response.content
        named_log(self, "==> finished review of BibTex")
        named_log(self, f"==> response metadata:", response.usage_metadata)

        if self._cooldown_sec:
            countdown = max(0, self._cooldown_sec - elapsed)
            named_log(self, f"==> initiating cooldown of {countdown} s (request limitations)")
            countdown_log("", countdown)

        return paper

    def divide_subtasks(self, n, input_data: PaperData =None):
        paper = input_data
        sub = []
        n_sections = len(paper.sections)
        per_task = n_sections // n
        for i in range(0, n, per_task):
            subpaper = PaperData(paper.subject, paper.sections[i:i+per_task], paper.title, paper.bib_path)
            sub.append(TexReviewer(self.out_dir, subpaper, self._cooldown_sec))
        return sub

    def merge_subtasks_data(self, data: List[PaperData]):
        paper = PaperData(data[0].subject, data[0].sections, data[0].title, data[0].bib_path)
        for subpaper in data[1:]:
            paper.sections.extend(subpaper.sections)
        return paper
        