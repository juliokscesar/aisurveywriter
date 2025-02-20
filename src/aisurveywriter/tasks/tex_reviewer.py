from typing import Optional, List
import re
import os
import time

from .pipeline_task import PipelineTask
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils.logger import named_log
from aisurveywriter.utils.helpers import assert_type

class TexReviewer(PipelineTask):
    def __init__(self, agent_ctx: AgentContext, paper: PaperData):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = paper

        self._figure_pattern = re.compile(r'\\begin{figure}.*?\\includegraphics(?:\[.*?\])?{(.*?)}.*?\\end{figure}', re.DOTALL)
        self._cite_pattern = re.compile(r"\\cite{([^}]+)}")
        
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        return self.review_tex()
        
    def review_tex(self) -> PaperData:
        """
        Review LaTeX syntax and commands section by section
        """
        with open(self.agent_ctx._working_paper.bib_path, "r", encoding="utf-8") as f:
            bib_content = f.read()

        section_amount = len(self.agent_ctx._working_paper.sections)
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            if section.content is None:
                continue
            
            named_log(self, f"==> begin review LaTeX syntax for section ({i+1}/{section_amount}): {section.title}")

            start = time.time()

            section.content = re.sub(r"[`]+[\w]*", "", section.content)
            section = self._remove_invalid_figures(section, self.agent_ctx._working_paper.fig_path)
            section = self._remove_invalid_refs(section, bib_content)

            elapsed = int(time.time() - start)
            
            named_log(self, f"==> finished review LaTeX syntax for section ({i+1}/{section_amount}): {section.title} | time elapsed: {elapsed} s")

        return self.agent_ctx._working_paper


    def _remove_invalid_figures(self, section: SectionData, img_dir: str):
        if not os.path.isdir(img_dir):
            named_log(self, "==> invalid image directory:", img_dir)
            return
        
        def replace_invalid_figure(match):
            nonlocal img_dir
            image_path = match.group(1).strip()
            full_image_path = os.path.join(img_dir, image_path)
            
            if not os.path.exists(full_image_path):
                return ''  # Remove the entire figure block if image doesn't exist
            return match.group(0)  # Keep the figure block if the image exists
        
        # Replace invalid figure blocks
        section.content = self._figure_pattern.sub(replace_invalid_figure, section.content)
    
        return section
    

    def _remove_invalid_refs(self, section: SectionData, bib_content: str):
        def replace_invalid(match):
            keys = match.group(1).split(',')
            valid_keys = [key.strip() for key in keys if key.strip() in bib_content]
            if valid_keys:
                return f"\\cite{{{', '.join(valid_keys)}}}"
            else:
                return ""
        
        section.content = self._cite_pattern.sub(replace_invalid, section.content)
        return section


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
        