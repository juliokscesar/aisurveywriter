from typing import List
import re
import random
import bibtexparser

from .pipeline_task import PipelineTask
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import RAGType, BibTexData
from aisurveywriter.core.paper import PaperData
from aisurveywriter.utils.logger import named_log, cooldown_log
from aisurveywriter.utils.helpers import assert_type

class PaperReferencer(PipelineTask):
    def __init__(self, agent_ctx: AgentContext, reviewed_paper: PaperData, save_bib_path: str, max_per_section: int = 90, max_per_sentence: int = 4, confidence: float = 0.9, max_same_ref: int = 10):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = reviewed_paper
        
        self.save_bib_path = save_bib_path
        self.max_per_section = max_per_section
        self.max_per_sentence = max_per_sentence
        self.max_same_ref = max_same_ref
        self.confidence = confidence
        
    def reference(self) -> PaperData:
        assert(not self.agent_ctx.rags.is_disabled(RAGType.BibTex))
        
        used_keys = {}
        probabilities = {1: 0.6, 2: 0.2, 3: 0.12, 4: 0.08}
        
        section_amount = len(self.agent_ctx._working_paper.sections)
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            ref_count = 0
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)((?<=\.|\?)\s)", section.content)
            cited_sentences = []

            for j, sentence in enumerate(sentences):
                if ref_count >= self.max_per_section:
                    cited_sentences.extend(sentences[j:])
                    break
                    
                if not sentence.strip() or '\\' in sentence or '{' in sentence or '}' in sentence or sentence[0].isdigit() or sentence[0] == '-' or sentence[0] == '*':
                    cited_sentences.append(sentence)
                    continue
                
                results: List[BibTexData] = self.agent_ctx.rags.retrieve(RAGType.BibTex, sentence, k=self.max_per_section)
                if not results:
                    continue
                
                valid = set()    
                num_references = random.choices(
                    population=list(probabilities.keys()),
                    weights=list(probabilities.values()),
                    k=1
                )[0]
                num_references = min(num_references, self.max_per_sentence)
                
                for result in results:
                    if len(valid) >= num_references:
                        break
                    
                    key = result.bibtex_key
                    if key not in used_keys:
                        used_keys[key] = 0
                    elif used_keys[key] >= self.max_same_ref:
                        continue
                    
                    valid.add(key)
                    ref_count += 1
                    used_keys[key] += 1
                
                if valid:
                    cite_command = f"\\cite{{{', '.join(valid)}}}"
                    if sentence.endswith(('.', ',', ';', '!', '?')):
                        sentence = sentence[:-1] + ' ' + cite_command + sentence[-1]
                    else:
                        sentence += f" {cite_command}"
                
                cited_sentences.append(sentence)
                
                if self.agent_ctx.embed_cooldown:
                    cooldown_log(self, self.agent_ctx.embed_cooldown)

            section.content = " ".join(cited_sentences)
            named_log(self, f"==> added {ref_count} references to section ({i+1}/{section_amount}): \"{section.title}\"")

        self._dump_used_bib(used_keys)
        self.agent_ctx._working_paper.bib_path = self.save_bib_path
        return self.agent_ctx._working_paper
        
    def _dump_used_bib(self, used_keys: list):
        with open(self.agent_ctx.references.bibtex_db_path, "r", encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)
        
        used_entries = [entry for entry in bib_db.entries if entry["ID"] in used_keys]
        used_db = bibtexparser.bibdatabase.BibDatabase()
        used_db.entries = used_entries
        
        with open(self.save_bib_path, "w", encoding="utf-8") as f:
            bibtexparser.dump(used_db, f)

    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        return self.reference()
            