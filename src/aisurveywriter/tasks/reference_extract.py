from typing import List, Optional
import os
import yaml
import re
from time import sleep

from .pipeline_task import PipelineTask

from aisurveywriter.core.llm_handler import LLMHandler
import aisurveywriter.core.file_handler as fh
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.utils import named_log, countdown_log, get_bibtex_entry, bib_entries_to_str, time_func

class ReferenceExtractor(PipelineTask):
    def __init__(self, llm: LLMHandler, ref_paths: List[str], prompt: str, raw_save_path: Optional[str] = None, rawbib_save_path: Optional[str] = None, bib_save_path: Optional[str] = None, cooldown_sec: int = 30, batches: int = 3):
        self.no_divide = True
        self.llm = llm
        self.ref_paths = ref_paths
        self.prompt = prompt
        
        self.raw_save_path = raw_save_path
        self.rawbib_save_path = rawbib_save_path
        self.bib_save_path = bib_save_path

        self._cooldown_sec = int(cooldown_sec)
        self._batches = int(batches) if batches >= 1 else 1
    
    def pipeline_entry(self, input_data=None):
        refs = self.extract_references(save_path=self.raw_save_path)
        named_log(self, "==> gathering bib entries from extracted title and author")
        refs = self.refs_to_bib(refs, self.rawbib_save_path)
        named_log(self, "==> filtering duplicate refernces")
        refs = self.remove_duplicates(refs, self.bib_save_path)
        return input_data
    
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
        self.llm.init_chain(None, self.prompt)
        pdfs = PDFProcessor(self.ref_paths).extract_content()
        for path,pdf in zip(self.ref_paths,pdfs):
            # get only the "references" section if possible
            ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", pdf, re.IGNORECASE)
            if ref_match:
                start_idx = ref_match.start()
                pdf = pdf[start_idx:].strip()
            else:
                named_log(self, f"couldn't match references regex for pdf {os.path.basename(path)}, using entire content")
            
            named_log(self, f"==> started extracting references from pdf {os.path.basename(path)}")
            total_elapsed = 0
            
            # Send the PDF in batches because if it has too many references, some llms might truncate their output
            batch_contents = self._split_content(pdf, self._batches)
            for batch in batch_contents:
                elapsed, response = time_func(self.llm.invoke, {"pdfcontent": batch})
                references += response.content + '\n'
                total_elapsed += int(elapsed)
            
            named_log(self, f"==> finished extracting references from pdf {os.path.basename(path)} | time elapsed: {total_elapsed}")
 
            try:
                named_log(self, f"==> response metadata:", response.usage_metadata)
            except:
                named_log(self, f"==> (debug) reponse object:", response)

            if self._cooldown_sec:
                cooldown = max(0, self._cooldown_sec - total_elapsed)
                named_log(self, f"==> initiating  cooldown of {cooldown} s (request limitations)")
                countdown_log("", cooldown)
        
        references = re.sub(r"[`]+[\w]*", "", references)
        references = "\n".join([line for line in references.splitlines() if line.strip()]) # remove any blank lines
        
        # format to save yaml
        # res = "references:\n"
        # for line in references.split("\n"):
        #     res += f"  {line}\n"
        # references = res

        references = self._fmt_to_reflist(references)
        ref_dict = {"references": references}
        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(ref_dict, f)
                
        return ref_dict
    
    def _split_content(self, content: str, n_batches: int):
        """
        Split the content while making sure that every line remains intact
        """
        
        lines = content.strip().split("\n")
        nlines = len(lines)
        
        if n_batches <= 0:
            return content

        batch_sz = nlines // n_batches
        remainder = nlines % n_batches
        
        batches = []
        start = 0
        for i in range(n_batches):
            extra = 1 if i < remainder else 0
            end = start + batch_sz + extra
            batches.append("\n".join(lines[start:end]))
            start = end
        return batches
    
    def _fmt_to_reflist(self, references: str):
        pattern = re.compile(r'- title: "(.*?)"\s+author: "(.*?)"', re.DOTALL)
        matches = pattern.findall(references)
        result = [{"title": title, "author": author} for title, author in matches]
        return result
    
    def remove_duplicates(self, bib_entries: list, save_path: Optional[str] = None):
        """
        Remove any duplicate entries on references by comparing their DOIs
        """ 
        doi_history = []
        to_remove = []
        for i, entry in enumerate(bib_entries):
            if "doi" not in entry:
                continue
            doi = entry["doi"].strip()
            if doi not in doi_history:
                doi_history.append(doi)
            else:
                to_remove.append(i)
        
        named_log(self, f"Found {len(to_remove)} duplicate references")
        
        to_remove.sort(reverse=True)
        no_duplicates = bib_entries.copy()
        for rem in to_remove:
            no_duplicates.pop(rem)
        
        if save_path is not None:
            bibs = "\n".join([bib_entries_to_str([entry]) for entry in no_duplicates])
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(bibs)
        
        return no_duplicates
     
    def refs_to_bib(self, refs: dict, save_path: Optional[str] = None):
        """
        Get BibTex entries from title,author references
        """
        
        bibs = []
        for ref in refs["references"]:
            # sleep(2) # avoid api timeout
            try:
                entry = get_bibtex_entry(ref["title"], ref["author"])
                if not entry:
                    continue
                bibs.append(entry)
            except Exception as e:
                named_log(self, f"Bibtex entry failed for: {ref}")
        
        if save_path:
            bibstr = "\n".join([bib_entries_to_str([entry]) for entry in bibs])
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(bibstr)
        
        return bibs
