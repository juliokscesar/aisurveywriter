import os
import shutil
import re
from typing import List

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from .pipeline_task import PipelineTask
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import RAGType, ImageData
from aisurveywriter.core.paper import PaperData
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log
from aisurveywriter.utils.helpers import time_func, assert_type

class PaperFigureAdd(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, written_paper: PaperData, images_dir: str, confidence: float = 0.8, max_figures: int = 30):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = written_paper
        
        self._system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.add_figures.text)
        self._human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n"+
                                                               "{refcontent}\n\n"+
                                                               "[end: references_content]\n\n"+
                                                               "Add figures to this section:\n"+
                                                               "- Section title: {title}\n"+
                                                               "- Section content:\n{content}")
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        self.confidence = confidence
        self.max_figures = max_figures
    
        self.images_dir = os.path.abspath(images_dir)
        self.used_imgs_dest = os.path.join(self.agent_ctx.output_dir, "used-imgs")
        os.makedirs(self.used_imgs_dest, exist_ok=True)
        self.agent_ctx._working_paper.fig_path = self.used_imgs_dest
        
    def add_figures(self):
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        # this task uses full pdf content
        refcontent = "\n\n".join(self.agent_ctx.references.full_content())
        
        section_amount = len(self.agent_ctx._working_paper.sections)
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            named_log(self, f"==> start adding figures in section ({i+1}/{section_amount}): \"{section.title}\"")
            
            content = self._llm_add_figures(refcontent, section.title, section.content)
            content, used_imgs = self._replace_figures(content, retrieve_k=5)
            
            named_log(self, f"==> finish adding figures in section ({i+1}/{section_amount}), replaced {len(used_imgs)} figures")
            
            section.content = re.sub(r"[`]+[\w]*", "", content)
        
        self.agent_ctx._working_paper.fig_path = self.used_imgs_dest
        return self.agent_ctx._working_paper
    
    def _llm_add_figures(self, refcontent: str, title: str, content: str) -> str:
        elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
            "refcontent": refcontent,
            "subject": self.agent_ctx._working_paper.subject,
            "title": title,
            "content": content,
        })
        named_log(self, "==> got section with figures from LLM")
        metadata_log(self, elapsed, response)
        if self.agent_ctx.llm_cooldown:
            cooldown_log(self, self.agent_ctx.llm_cooldown)
        
        return response.content

    def _replace_figures(self, content: str, retrieve_k: int = 5):
        assert(not self.agent_ctx.rags.is_disabled(RAGType.ImageData))
        
        fig_pattern =  r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}.*?\\caption\{([^}]+)\}"
        fig_matches = [(m.start(), m.end(), m.group(1), m.group(2)) for m in re.finditer(fig_pattern, content, re.DOTALL)]
        used_imgs = set()
        for start, end, figname, caption in fig_matches:
            if len(used_imgs) >= self.max_figures:
                break

            # cut "Adapted from..." from caption (don't include it as query)
            idx = caption.rfind("Adapted from")
            if idx != -1:
                caption = caption[:idx].strip()

            # use caption to retrieve an image            
            query = f"{caption}"
            results: List[ImageData] = self.agent_ctx.rags.retrieve(RAGType.ImageData, query, k=retrieve_k, confidence=self.confidence)
            if not results:
                named_log(self, f"==> unable to find images for figure: {figname!r}, caption: {caption}")
                if self.agent_ctx.embed_cooldown:
                    cooldown_log(self, self.agent_ctx.embed_cooldown)
                continue
            
            image_used = None
            for result in results:
                if result.basename in used_imgs:
                    continue
                used_imgs.add(result.basename)
                
                try:
                    image_used = os.path.join(self.used_imgs_dest, result.basename)
                    shutil.copy(os.path.join(self.images_dir, result.basename), image_used)
                    break
                except Exception as e:
                    image_used = result.basename
                    named_log(self, f"==> couldn't copy {image_used} to save directory: {e}")
                    break
            
            if not image_used:
                named_log(self, f"==> couldn't find a match for {figname}: {caption!r} that had confidence >= {self.confidence} or wasn't used before")
                continue
            else:
                named_log(self, f"==> best match for {figname}: {os.path.basename(image_used)}")
            
            replacement = rf"\\includegraphics[width=0.97\\textwidth]{{{os.path.basename(image_used)}}}"
            re_pattern = rf"\\includegraphics(\[[^\]]*\])?{{{re.escape(figname)}}}"
            content = re.sub(re_pattern, replacement, content, count=1)
            
            if self.agent_ctx.embed_cooldown:
                cooldown_log(self, self.agent_ctx.embed_cooldown)
        
        return content, used_imgs
            
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data

        return self.add_figures()
    