from typing import List
import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS

from .pipeline_task import PipelineTask
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.utils import named_log, countdown_log, image_to_base64

class FigureExtractor(PipelineTask):
    def __init__(self, llm: LLMHandler, embed, subject: str, pdf_paths: List[str], save_dir: str, faiss_save_path: str, request_cooldown_sec: int = 30):
        self.no_divide = True
        self.llm = llm
        self.embed = embed
        self.subject = subject
        self.pdf_paths = pdf_paths.copy()
        self.save_dir = os.path.abspath(save_dir)
        self.faiss_save_path = faiss_save_path
        self.faiss = None
        
        self._cooldown_sec = request_cooldown_sec
        
    def pipeline_entry(self, input_data=None):
        img_data = self.extract()
        self.faiss = self._imgdata_faiss(img_data, self.faiss_save_path)
        return input_data
    
    def extract(self, pdf_paths: List[str] = None, save_dir: str = None, faiss_save_path: str = None):
        if pdf_paths:
            self.pdf_paths = pdf_paths
        if save_dir:
            self.save_dir = save_dir
        if faiss_save_path:
            self.faiss_save_path = faiss_save_path
        
        img_data = []
        self.save_dir = os.path.abspath(self.save_dir)
        
        prompt_template = f"'''\n{{pdfcontent}}\n'''\n\nThe text above is from a reference relevant for the subject {self.subject}. Based on that, the image provided next is from this document.\n\nProvide a detailed but concise description of the image, be direct, objective and clear in your description -- a maximum of 100 words."
        img_input_template = "data:image/png;base64,{imgb64}"
        for pdf in self.pdf_paths:
            proc = PDFProcessor(pdf)
            content = proc.extract_content()[0]
            imgs = proc.extract_images(save_dir=self.save_dir)
            named_log(self, f"Extracted {len(imgs)} images from PDF {pdf}")
            for i, img in enumerate(imgs):
                imgb64 = image_to_base64(img["path"])
                
                msg = HumanMessage(content={
                    {"type": "text", "text": prompt_template.replace("{pdfcontent}", content)},
                    {"type": "image_url", "image_url": {"url": img_input_template.replace("{imgb64}", imgb64)}}
                })
                response = self.llm.invoke([msg])
                
                named_log(self, f"LLM response metadata for image {i+1} {os.path.basename(img["path"])}:", response.usage_metadata)
                
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


