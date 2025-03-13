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

        self._re_patterns = {
            "figure": re.compile(r"\\begin{figure}.*?\\includegraphics(?:\[.*?\])?{(.*?)}.*?\\end{figure}", re.DOTALL),
            "cite": re.compile(r"\\cite{([^}]+)}"),
            "empty_cite": re.compile(r"\\cite{\s*}"),
            "percent": re.compile(r"(\d+)%"),
            "preamble": [
                re.compile(r"\\usepackage{[\w]+}"),
                re.compile(r"\\(begin|end){document}"),
                re.compile(r"\\documentclass(?:\[(?:.*?)\])*{(?:.*?)}"),
                re.compile(r"\\geometry{(?:.*?)}"),
            ],
            "mk_code_block": re.compile(r"[`]+[\w]*"),
            "mk_bold": re.compile(r"\*\*(.*?)\*\*"),
            "mk_italic": re.compile(r"\*(.*?)\*"),
            "mk_num_list": re.compile(r"^(\d+)[\.-]"),
        }
        
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
            bib_keys = set(re.findall(r"@\w+{([^,]+),", f.read())) # pre-extract citation keys

        section_amount = len(self.agent_ctx._working_paper.sections)
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            if section.content is None:
                continue
            
            named_log(self, f"==> begin review LaTeX syntax for section ({i+1}/{section_amount}): {section.title}")

            start = time.time()

            section = self._convert_markdown(section)
            section = self._remove_preamble(section)
            section = self._remove_invalid_figures(section, self.agent_ctx._working_paper.fig_path)
            section = self._remove_invalid_refs(section, bib_keys)
            
            # escape number percentage
            section.content = self._re_patterns["percent"].sub(r"\1\\%", section.content)

            elapsed = int(time.time() - start)
            
            named_log(self, f"==> finished review LaTeX syntax for section ({i+1}/{section_amount}): {section.title} | time elapsed: {elapsed} s")

        return self.agent_ctx._working_paper

    def _remove_preamble(self, section: SectionData):
        for pat in self._re_patterns["preamble"]:
            section.content = pat.sub("", section.content)

        return section

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
        section.content = self._re_patterns["figure"].sub(replace_invalid_figure, section.content)
    
        return section
    

    def _remove_invalid_refs(self, section: SectionData, bib_keys: set):
        def replace_invalid(match):
            keys = match.group(1).split(',')
            valid_keys = [key.strip() for key in keys if key.strip() in bib_keys]
            if valid_keys:
                return f"\\cite{{{', '.join(valid_keys)}}}"
            else:
                return ""

        # remove citations that do not exist in .bib        
        section.content = self._re_patterns["cite"].sub(replace_invalid, section.content)

        # remove empty citations
        section.content = self._re_patterns["empty_cite"].sub("", section.content)
        
        return section


    def _convert_markdown(self, section: SectionData):
        section.content = self._re_patterns["mk_code_block"].sub("", section.content) # remove markdown code block
        section.content = self._re_patterns["mk_bold"].sub(r"\\textbf{\1}", section.content) # replace bold text
        section.content = self._re_patterns["mk_italic"].sub(r"\\textit{\1}", section.content) # replace italic text

        in_itemize = False
        in_enumerate = False
        section_lines = section.content.split("\n")    
        converted_lines = []
        for line in section_lines:
            stripped = line.strip()
            if not stripped:
                converted_lines.append(line)
                continue
            
            if stripped.startswith("*") and not stripped.endswith("*"):
                if not in_itemize:
                    converted_lines.append("\\begin{itemize}")
                    in_itemize = True
                converted_lines.append(stripped.replace("*", "\\item{", 1) + "}")
                continue
            elif in_itemize:
                converted_lines.append("\\end{itemize}")
                in_itemize = False
            
            num_match = self._re_patterns["mk_num_list"].match(stripped)
            if num_match:
                if not in_enumerate:
                    converted_lines.append("\\begin{enumerate}")
                    in_enumerate = True
                converted_lines.append("\\item{" + stripped[num_match.end():] + "}")
                continue
            elif in_enumerate:
                converted_lines.append("\\end{enumerate}")
                in_enumerate = False
            
            converted_lines.append(line)
        
        if in_itemize:
            converted_lines.append("\\end{itemize}")
        if in_enumerate:
            converted_lines.append("\\end{enumerate}")

        section.content = "\n".join(converted_lines)
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
        