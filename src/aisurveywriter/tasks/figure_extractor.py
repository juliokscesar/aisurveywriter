from typing import List, Optional
import os
from pathlib import Path
import re
import shutil
import yaml

from langchain_core.messages import HumanMessage
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS

from .pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.core.paper import PaperData
from aisurveywriter.utils import named_log, countdown_log, image_to_base64

class FigureExtractor(PipelineTask):
    def __init__(self, llm: LLMHandler, embed, subject: str, pdf_paths: List[str], save_dir: str, faiss_save_path: Optional[str] = None, local_faiss_path: Optional[str] = None, paper: Optional[PaperData] = None, request_cooldown_sec: int = 30):
        self.no_divide = True
        self.llm = llm
        self.embed = embed
        self.subject = subject
        self.paper = paper
        self.pdf_paths = pdf_paths.copy()
        self.save_dir = os.path.abspath(save_dir)
        self.faiss_save_path = faiss_save_path
        self.faiss = None
        if local_faiss_path:
            self.faiss = FAISS.load_local(local_faiss_path, embed, allow_dangerous_deserialization=True)
        
        self._cooldown_sec = request_cooldown_sec
        
    def pipeline_entry(self, input_data: PaperData = None):
        if input_data:
            self.paper = input_data
        if not self.faiss:
            img_data = self.extract()
            with open("refextract_imgdata.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump({"data": img_data}, f)
            
            self.faiss = self._imgdata_faiss(img_data, self.faiss_save_path)
        self.paper = self.add_to_paper(self.faiss, self.paper, self.save_dir)
        return self.paper
    
    def __call__(self, input_data: PaperData = None):
        return self.pipeline_entry(input_data)
    
    def extract(self, pdf_paths: List[str] = None, save_dir: str = None, faiss_save_path: str = None):
        if pdf_paths:
            self.pdf_paths = pdf_paths
        if save_dir:
            self.save_dir = save_dir
        if faiss_save_path:
            self.faiss_save_path = faiss_save_path
        
        img_data = []
        self.save_dir = os.path.abspath(self.save_dir)
        print(self.pdf_paths)
        prompt_template = f"'''\n{{pdfcontent}}\n'''\n\nThe text above is from a reference relevant for the subject {self.subject}. Based on that, the image provided next is from this document.\n\nProvide a detailed but concise description of the image, be direct, objective and clear in your description -- a maximum of 100 words."
        img_input_template = "data:image/png;base64,{imgb64}"
        for pdf in self.pdf_paths:
            proc = PDFProcessor([pdf])
            
            content = proc.extract_content()[0]
            ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", content, re.IGNORECASE)
            if ref_match:
                content = content[:ref_match.start()].strip()
            
            imgs = proc.extract_images(save_dir=self.save_dir)
            named_log(self, f"Extracted {len(imgs)} images from PDF {pdf}")
            for i, img in enumerate(imgs):
                imgb64 = image_to_base64(img["path"])
                
                msg = HumanMessage(content=[
                    {"type": "text", "text": prompt_template.replace("{pdfcontent}", content)},
                    {"type": "image_url", "image_url": {"url": img_input_template.replace("{imgb64}", imgb64)}}
                ])
                response = self.llm.llm.invoke([msg])
                
                named_log(self, f"LLM response metadata for image {i+1} {os.path.basename(img["path"])}:", response.usage_metadata)
                named_log(self, f"LLM description: {response.content}")

                if self._cooldown_sec:
                    named_log(self, f"Initiating cooldown of {self._cooldown_sec} (request limitations)")
                    countdown_log("", self._cooldown_sec)
                
                img_data.append({
                    "id": len(img_data),
                    "path": img["path"],
                    "description": response.content,
                })

        return img_data

    def _imgdata_faiss(self, img_data: List[dict], faiss_save_path: str = None):
        documents = [
            Document(page_content=img["description"], metadata={"id": img["id"], "path": img["path"]})
            for img in img_data
        ]
        vector_store = FAISS.from_documents(documents, self.embed)
        if faiss_save_path:
            vector_store.save_local(faiss_save_path)
            
        return vector_store


    def add_to_paper(self, figure_faiss: Optional[FAISS] = None, paper: Optional[PaperData] = None, save_dir: str = None):
        if paper:
            self.paper = paper
        if figure_faiss:
            self.faiss = figure_faiss
        if save_dir:
            self.save_dir = os.path.abspath(save_dir)
        
        os.makedirs(os.path.join(self.save_dir, "used"), exist_ok=True)
        
        fig_pattern = r"\\begin\{figure\}[\s\S]+?\\includegraphics[\S]*[\s]*\]]*?\{([\s\S]+?)\}[\s\S]*?\\caption\{([\s\S]+?)\}"
        
        for section in self.paper.sections:
            fig_matches = [(m.start(), m.end(), m.group(1), m.group(2)) for m in re.finditer(fig_pattern, section.content)]
            
            for start, end, figname, caption in fig_matches:
                # use caption to retrieve an image
                result = self.faiss.similarity_search(f"{figname}: {caption}", k=1)[0]
                named_log(self, f"Best match for caption {figname}: {caption!r} in section {section.title} is: {result.metadata["path"]}")
                
                try:
                    path = os.path.join(os.path.join(self.save_dir, "used"), os.path.basename(result.metadata["path"]))
                    shutil.copy(result.metadata["path"], path)
                except Exception as e:
                    path = result.metadata["path"]
                    named_log(self, f"Couldn't copy {path} to save directory: {e}")
                
                section.content = section.content.replace(figname, os.path.basename(path))
            
        return self.paper
        
    def divide_subtasks(self, n, input_data=None):
        raise NotImplemented()
    
    def merge_subtasks_data(self, data):
        raise NotImplemented()
    