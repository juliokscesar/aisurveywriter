from typing import List
import os

from langchain_core.messages import SystemMessage

from .pipeline_task import PipelineTask

from aisurveywriter.core.llm_handler import LLMHandler
import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.utils import named_log, get_bibtex_entry, bib_entries_to_str

class ReferenceExtractor(PipelineTask):
    def __init__(self, llm: LLMHandler, ref_paths: List[str], prompt: str, save_path: str = None):
        self.llm = llm
        self.ref_paths = ref_paths
        self.prompt = prompt
        self.llm.init_chain(None, prompt)
        self.save_path = save_path
    
    def pipeline_entry(self, input_data):
        self.extract_references(self.save_path)
        return self.save_path
    
    def extract_references(self, save_path: str):
        references = ""
        pdfs = PDFProcessor(self.ref_paths).extract_content()
        for path,pdf in zip(self.ref_paths,pdfs):
            named_log(self, f"==> started extracting references from pdf {os.path.basename(path)}")
            response = self.llm.invoke({
                "pdfcontent": pdf,
            })
            references += response.content + '\n'
            
            named_log(self, f"==> finished extracting references from pdf {os.path.basename(path)}")
            named_log(self, f"==> started extracting references from pdf {os.path.basename(path)}")
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(references)
        
    def refs_to_bib(self, ref_path: str):
        """
        Get BibTex entries from title,author references
        "ref_path" must be a yaml file like:
        - references:
          - title: ...
            author: ...
          - title: ...
            author: ...
        """
        refs = fh.read_yaml(ref_path)
        
        bibs = ""
        for ref in refs["references"]:
            try:
                entry = get_bibtex_entry(ref["title"], ref["author"])
                bibs += bib_entries_to_str([entry]) + '\n'
            except Exception as e:
                named_log(self, f"Bibtex entry failed for: {ref}")
        
        return bibs