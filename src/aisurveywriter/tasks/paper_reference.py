import re
from typing import Optional

from .pipeline_task import PipelineTask

from aisurveywriter.core.paper import PaperData
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.utils import named_log, time_func, countdown_log

class PaperReferencer(PipelineTask):
    def __init__(self, llm: LLMHandler, bibdb_path: str, prompt: str, paper: PaperData = None, cooldown_sec: int = 30, save_usedbib_path: Optional[str] = None):
        self.llm = llm
        self.paper = paper
        self.bibdb_path = bibdb_path
        self.prompt = prompt
        self.save_usedbib_path = save_usedbib_path
                
        self._cooldown_sec = int(cooldown_sec)
        
    def pipeline_entry(self, input_data):
        if not isinstance(input_data, PaperData):
            raise TypeError(f"Task {self.__class__.__name__} requires input of type PaperData in pipe entry")
        paper = self.reference(input_data, self.bibdb_path, self.prompt)
        return paper
    
    def reference(self, paper: PaperData = None, bibdb_path: str = None, prompt: str = None):
        if paper is not None:
            self.paper = paper
        if bibdb_path is not None:
            self.bibdb_path = bibdb_path
        if prompt:
            self.prompt = prompt
        
        with open(bibdb_path, "r", encoding="utf-8") as f:
            bibdb = f.read()
        
        # TODO: figure out how to save only the used citations (at the end of this step) in a different .bib path
        
        self.llm.init_chain(None, self.prompt)
        sz = len(self.paper.sections)
        for i, section in enumerate(self.paper.sections):
            named_log(self, f"==> started adding references in section ({i+1}/{sz}):", section.title)
            # TODO: have a way to keep track of the number of references added (maybe I can just regex \cite or something)
            elapsed, response = time_func(self.llm.invoke, {
                "bibdatabase": bibdb,
                "subject": self.paper.subject,
                "title": section.title,
                "content": section.content,
            })
            elapsed = int(elapsed)
            section.content = re.sub(r"[`]+[\w]*", "", response.content)
            
            named_log(self, f"==> finished adding references in section ({i+1}/{sz}) | time elapsed: {elapsed} s")
            named_log(self, f"==> response metadata:", response.usage_metadata)
            
            if self._cooldown_sec:
                cooldown = max(0, self._cooldown_sec - elapsed)
                named_log(self, f"==> initiating  cooldown of {cooldown} s (request limitations)")
                countdown_log("", cooldown)

        if self.save_usedbib_path:
            self._dump_used_bib(paper, bibdb, self.save_usedbib_path)

        return self.paper
        
    def _dump_used_bib(self, paper: PaperData, bibdb_content: str, save_path: str):
        latex_content = paper.full_content()
        # Find all citation keys inside \cite{...}
        cited_keys = set(re.findall(r'\\cite\{(.*?)\}', latex_content))
        
        # Flatten nested citations (e.g., \cite{ref1, ref2})
        cited_keys = {key.strip() for group in cited_keys for key in group.split(',')}
        
        # Dictionary to store matched references
        used_references = {}
        
        # Find and extract matching BibTeX entries
        bib_entries = re.split(r'(@\w+\{)', bibdb_content)[1:]
        
        for i in range(0, len(bib_entries), 2):
            entry_type = bib_entries[i].strip()
            entry_body = bib_entries[i + 1]
            match = re.match(r'([^,\s]+),', entry_body)
            if match:
                entry_key = match.group(1)
                if entry_key in cited_keys:
                    used_references[entry_key] = entry_type + entry_body
        
    
        used_bib_content = '\n\n'.join(used_references.values())
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(used_bib_content)
    