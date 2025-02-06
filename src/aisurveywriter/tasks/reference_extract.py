from typing import List, Optional
import os
import yaml
import re

from .pipeline_task import PipelineTask

from aisurveywriter.core.llm_handler import LLMHandler
import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.utils import named_log, countdown_log, get_bibtex_entry, bib_entries_to_str, time_func

class ReferenceExtractor(PipelineTask):
    def __init__(self, llm: LLMHandler, ref_paths: List[str], prompt: str, raw_save_path: Optional[str] = None, rawbib_save_path: Optional[str] = None, bib_save_path: Optional[str] = None, cooldown_sec: int = 30):
        self.llm = llm
        self.ref_paths = ref_paths
        self.prompt = prompt
        self.llm.init_chain(None, prompt)
        
        self.raw_save_path = raw_save_path
        self.rawbib_save_path = rawbib_save_path
        self.bib_save_path = bib_save_path

        self._cooldown_sec = cooldown_sec
    
    def pipeline_entry(self, input_data=None):
        refs = self.extract_references(save_path=self.raw_save_path)
        refs = self.refs_to_bib(refs, self.rawbib_save_path)
        refs = self.remove_duplicates(refs, self.bib_save_path)
        return self.bib_save_path
    
    def __call__(self, raw_save_path: Optional[str] = None, rawbib_save_path: Optional[str] = None, bib_save_path: Optional[str] = None):
        if raw_save_path:
            self.raw_save_path = raw_save_path
        if rawbib_save_path:
            self.rawbib_save_path = rawbib_save_path
        if bib_save_path:
            self.bib_save_path = bib_save_path
        
        return self.pipeline_entry()
    
    def extract_references(self, ref_paths: Optional[List[str]] = None, save_path: Optional[str] = None):
        """
        Extract every reference Title and Author from all papers in "ref_paths" (or self.ref_paths, if ref_paths is None). 
        The output is a dictionary where the first and only key is "references", which is a list that contains
        multiple entries in the format: {"title": ..., "author": ...}
        """
        
        if ref_paths is not None:
            self.ref_paths = ref_paths
        references = ""
        pdfs = PDFProcessor(self.ref_paths).extract_content()
        for path,pdf in zip(self.ref_paths,pdfs):
            named_log(self, f"==> started extracting references from pdf {os.path.basename(path)}")
            # response = self.llm.invoke({
            #     "pdfcontent": pdf,
            # })
            # elapsed = int(time() - start)
            elapsed, response = time_func(self.llm.invoke, {"pdfcontent": pdf})
            elapsed = int(elapsed)
            
            references += response.content + '\n'
            
            named_log(self, f"==> finished extracting references from pdf {os.path.basename(path)} | time elapsed: {elapsed}")
 
            try:
                named_log(self, f"==> response metadata:", response.usage_metadata)
            except:
                named_log(self, f"==> (debug) reponse object:", response)

            if self._cooldown_sec:
                countdown_log("Cooldown:", max(0, self._cooldown_sec - elapsed))
        
        references = re.sub(r"[`]+[\w]*", "", references)
        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(references)
        return yaml.safe_load(references)
    
    def remove_duplicates(self, bib_entries: list, save_path: Optional[str] = None):
        """
        Remove any duplicate entries on references by comparing their DOIs
        """ 
        doi_history = []
        to_remove = []
        for i, entry in enumerate(bib_entries):
            if "doi" not in entry:
                continue
            doi = entry["doi"]
            if doi not in doi_history:
                doi_history.append(doi)
            else:
                to_remove.append(i)
        
        named_log(self, f"Found {len(to_remove)} duplicate references")
        
        to_remove.sort(reverse=True)
        no_duplicates = bib_entries.copy()
        for rem in to_remove:
            no_duplicates.remove(rem)
        
        if save_path is not None:
            bibs = "\n".join([bib_entries_to_str([entry]) for entry in no_duplicates])
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(bibs)
        
        return no_duplicates
     
    def refs_to_bib(self, refs: dict, save_path: Optional[str] = None):
        """
        Get BibTex entries from title,author references
        """
        
        bibs = ""
        for ref in refs["references"]:
            try:
                entry = get_bibtex_entry(ref["title"], ref["author"])
                bibs += bib_entries_to_str([entry]) + '\n'
            except Exception as e:
                named_log(self, f"Bibtex entry failed for: {ref}")
        
        return bibs