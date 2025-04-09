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

        self._re_patterns: dict[str, re.Pattern] = {
            "figure": re.compile(r"\\begin{figure}.*?\\includegraphics(?:\[.*?\])?{(.*?)}.*?\\end{figure}", re.DOTALL),
            "cite": re.compile(r"\\cite{([^}]+)}"),
            "empty_cite": re.compile(r"\\cite{\s*}"),
            "invalid_cite_command": re.compile(r"\\cite\w+{(?:.*?)}"),
            "false_reference": re.compile(r"\[(?:(?:\d+),*\s*)*\]"),
            "percent": re.compile(r"(?<!\\)([\w\.]+)%"),
            "preamble": [
                re.compile(r"\\usepackage{[\w]+}"),
                re.compile(r"\\(begin|end){document}"),
                re.compile(r"\\documentclass(?:\[(?:.*?)\])*{(?:.*?)}"),
                re.compile(r"\\geometry{(?:.*?)}"),
            ],
            "mk_code_block": re.compile(r"[`]+[\w]*"),
            "mk_bold": re.compile(r"\*{2}(.*?)\*{2}"),
            "mk_italic": re.compile(r"\*{1}(.*?)\*{1}"),
            "mk_num_list": re.compile(r"^(\d+)\s*[\.-]"),
            "begin_not_alter_block": re.compile(r"\\begin{(?:figure)}"),
            "end_not_alter_block": re.compile(r"\\end{(?:figure)}")
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
            
            start = time.time()

            section.content = self._convert_markdown(section.content)
            section.content = self._remove_preamble(section.content)
            section.content = self._remove_invalid_figures(section.content, self.agent_ctx._working_paper.fig_path)
            section.content = self._remove_invalid_refs(section.content, bib_keys)
           
            # escape number percentage
            section.content = self._re_patterns["percent"].sub(r"\1\\%", section.content)

            elapsed = time.time() - start
            
            named_log(self, f"review LaTeX syntax for section ({i+1}/{section_amount}): {section.title} | time elapsed: {elapsed} s")

        return self.agent_ctx._working_paper

    def _remove_preamble(self, section_content: str) -> str:
        for pat in self._re_patterns["preamble"]:
            section_content = pat.sub("", section_content)

        return section_content

    def _remove_invalid_figures(self, section_content: str, img_dir: str) -> str:
        if not os.path.isdir(img_dir):
            named_log(self, "==> invalid image directory:", img_dir)
            return section_content
        
        def replace_invalid_figure(match):
            nonlocal img_dir
            image_path = match.group(1).strip()
            full_image_path = os.path.join(img_dir, image_path)
            
            if not os.path.exists(full_image_path):
                return ''  # Remove the entire figure block if image doesn't exist
            return match.group(0)  # Keep the figure block if the image exists
        
        # Replace invalid figure blocks
        section_content = self._re_patterns["figure"].sub(replace_invalid_figure, section_content)
    
        return section_content
    

    def _remove_invalid_refs(self, section_content: str, bib_keys: set) -> str:
        def replace_invalid(match):
            keys = match.group(1).split(',')
            valid_keys = [key.strip() for key in keys if key.strip() in bib_keys]
            if valid_keys:
                return f"\\cite{{{', '.join(valid_keys)}}}"
            else:
                return ""

        # remove citations that do not exist in .bib        
        section_content = self._re_patterns["cite"].sub(replace_invalid, section_content)

        # remove empty citations
        section_content = self._re_patterns["empty_cite"].sub("", section_content)
        
        # remove [xxx] references sometimes added by the LLM
        section_content = self._re_patterns["false_reference"].sub("", section_content)

        # remove invalid cite commands (for some reasom some appear with \citep, \citea or whatever)
        section_content = self._re_patterns["invalid_cite_command"].sub("", section_content)
        
        return section_content


    def _convert_markdown(self, section_content: str) -> str:
        section_content = self._re_patterns["mk_code_block"].sub("", section_content) # remove markdown code block
        section_content = self._re_patterns["mk_bold"].sub(r"\\textbf{\1}", section_content) # replace bold text
        section_content = self._re_patterns["mk_italic"].sub(r"\\textit{\1}", section_content) # replace italic text

        in_itemize = False
        in_enumerate = False
        in_not_alter_block = False
        section_lines = section_content.split("\n")    
        converted_lines = []
        for line in section_lines:
            stripped = line.strip()
            if not stripped:
                converted_lines.append(line)
                continue
            
            if self._re_patterns["begin_not_alter_block"].match(stripped):
                in_not_alter_block = True
            elif self._re_patterns["end_not_alter_block"].match(stripped):
                in_not_alter_block = False
                
            if in_not_alter_block:
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

        section_content = "\n".join(converted_lines)
        return section_content


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
        