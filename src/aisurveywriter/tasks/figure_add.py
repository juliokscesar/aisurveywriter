import os
import shutil
import re
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from .pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.paper import PaperData
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.utils import time_func, named_log, countdown_log

class PaperFigureAdd(PipelineTask):
    def __init__(self, llm: LLMHandler, embed, faiss_path: str, imgs_path: str, ref_paths: List[str], prompt: str, out_path: str, llm_cooldown: int = 40, embed_cooldown: int = 0):
        self.no_divide = True
        
        self.llm = llm
        self.embed = embed
        self.faiss = FAISS.load_local(faiss_path, embed, allow_dangerous_deserialization=True)
        self.imgs_path = imgs_path
        self.prompt = prompt
        self.out_path = out_path
        self.ref_paths = ref_paths
        self._llm_cooldown = llm_cooldown
        self._embed_cooldown = embed_cooldown
        
    def pipeline_entry(self, input_data: PaperData):
        if not isinstance(input_data, PaperData):
            raise TypeError(f"Task {self.__class__.__name__} requires input data type of PaperData in pipeline entry")
        
        paper = self.add_figures(input_data)
        return paper
    
    def add_figures(self, paper: PaperData):
        used_imgs_path = os.path.join(os.path.abspath(self.out_path), "used-imgs")
        os.makedirs(used_imgs_path, exist_ok=True)
        
        fig_pattern = r"\\begin\{figure\}[\s\S]+?\\includegraphics[\S]*[\s]*\]]*?\{([\s\S]+?)\}[\s\S]*?\\caption\{([\s\S]+?)\}"
        
        sysmsg = SystemMessagePromptTemplate.from_template(self.prompt)
        hummsg = HumanMessagePromptTemplate.from_template("Content for this section:\n- Title: {title}\n- Content:\n\n{content}")
        self.llm.init_chain_messages(sysmsg, hummsg)
        
        for i, section in paper.sections:
            named_log(self, f"==> begin adding figures for section ({i+1}/{len(self.paper.sections)}): {section.title}")
            elapsed, response = time_func(self.llm.invoke, {
                "subect": paper.subject,
                "title": section.title,
                "content": section.content,
            })
            named_log(self, "==> response metadata (ask llm to add figures):", response.usage_metadata, f" | time elapsed: {elapsed} s")
            
            if self._llm_cooldown:
                countdown_log("Cooldown (request limitations):", self._llm_cooldown)
            
            content, used_imgs = self.replace_figures(response.content, fig_pattern)
            named_log(self, f"==> replaced {len(used_imgs)} in section {section.title}")

            if self._embed_cooldown:
                countdown_log("Cooldown (request limitations):", self._embed_cooldown)

            section.content = content

            named_log(self, f"==> finish adding figures for section ({i+1}/{len(self.paper.sections)}): {section.title}")

        return section

    def replace_figures(self, content: str, fig_pattern: str):
        fig_matches = [(m.start(), m.end(), m.group(1), m.group(2)) for m in re.finditer(fig_pattern, content)]
        used_imgs = []
        for start, end, figname, caption in fig_matches:
            # use caption to retrieve an image
            results = self.faiss.similarity_search(f"{figname}: {caption}", k=5)
            named_log(self, f"Best match for caption {figname}: {caption!r} is: {', '.join([re.metadata["path"] for re in results])}")
            
            path = None
            for result in results:
                if result.metadata["path"] in used_imgs:
                    continue
                used_imgs.append(result.metadata["path"])
                
                try:
                    path = os.path.join(self.save_dir, result.metadata["path"])
                    shutil.copy(os.path.join(self.imgs_dir, result.metadata["path"]), path)
                except Exception as e:
                    path = result.metadata["path"]
                    named_log(self, f"Couldn't copy {path} to save directory: {e}")
            
            if path:
                content = content.replace(figname, os.path.basename(path))
            else:
                named_log(self, f"Couldn't find a match for {figname}: {caption!r} that wasn't used before")

            if self._embed_cooldown:
                named_log(self, f"Initiating cooldown of {self._embed_cooldown} for Text Embedding model request")
                countdown_log("", self._embed_cooldown)

        return content, used_imgs        
            
    def ref_contents(self):
        pdfs = PDFProcessor(self.ref_paths).extract_content()
        content = ""
        for pdf in pdfs:
            ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", pdf, re.IGNORECASE)
            if ref_match:
                content += pdf[:ref_match.start()].strip()
            else:
                content += pdf.strip()
            content += "\n" 