from typing import List
import re
import random
import bibtexparser

from .pipeline_task import PipelineTask
from ..core.agent_context import AgentContext
from ..core.agent_rags import RAGType, BibTexData, FAISS
from ..core.paper import PaperData
from ..utils.logger import named_log, cooldown_log
from ..utils.helpers import assert_type

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
        probabilities = {1: 0.55, 2: 0.2, 3: 0.14, 4: 0.11}
        
        section_amount = len(self.agent_ctx._working_paper.sections)


        env_pattern = re.compile(r"\\begin{(figure|table|lstlisting|algorithm)}.*?\\end{\1}", re.DOTALL)
        env_placeholder = "@@LATEX_ENV_{}@@"
        section_env_blocks = []
        def store_env_block(match):
            nonlocal section_env_blocks
            section_env_blocks.append(match.group(0))
            return env_placeholder.format(len(section_env_blocks)-1)

        paragraph_pattern = re.compile(r"\n{2,}")
        sentence_pattern = re.compile(r"\. ")
        skip_section_pattern = re.compile(r"(Abstract|Resumo|References|Referências|Bibliography|Bibliografia)", re.IGNORECASE)
        
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            # skip some sections (like abstract)
            if skip_section_pattern.match(section.title):
                continue            

            # remove any environment blocks to avoid misreferencing
            preprocessed = env_pattern.sub(store_env_block, section.content)

            # split section into paragraphs
            paragraphs = paragraph_pattern.split(preprocessed)

            ref_count = 0
            # use less references in last section (Conclusion)
            max_refs_section = self.max_per_section if i != section_amount-1 else self.max_per_section//5          
            for paragraph_idx, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    continue
                
                # get most relevant references for this paragraph and make it a "sub-database"
                paragraph_refs = self.agent_ctx.rags.retrieve(RAGType.BibTex, paragraph.strip(), k=30)
                paragraph_refs_subdatabase = FAISS.from_documents([r.to_document() for r in paragraph_refs], self.agent_ctx.embed_handler.model)

                lines = paragraph.split("\n")
                cited_lines = []

                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    
                    # skip latex environments
                    if "@@LATEX_ENV" in stripped:
                        cited_lines.append(line)
                        continue
                    
                    # skip empty lines or lines that may not be adequate to add references
                    if stripped.startswith("\\") or stripped[0].isdigit() or stripped[0] == '-' or stripped[0] == '*' or stripped.startswith('%') or stripped.endswith(':'):
                        cited_lines.append(line)
                        continue
                
                    sentences = sentence_pattern.split(line)
                    cited_sentences = []

                    for j, sentence in enumerate(sentences):
                        if ref_count >= max_refs_section:
                            cited_sentences.extend(sentences[j:])
                            break
                            
                        # skip small sentences
                        if len(sentence.strip().split()) < 5:
                            cited_sentences.append(sentence)
                            continue
                        
                        valid = set()    
                        num_references = random.choices(
                            population=list(probabilities.keys()),
                            weights=list(probabilities.values()),
                            k=1
                        )[0]
                        num_references = min(num_references, self.max_per_sentence)
            
                        # retrieve references from sub-database
                        results = paragraph_refs_subdatabase.similarity_search(sentence, k=num_references+5)
                        if not results:
                            cited_sentences.append(sentence)
                            continue

                        results = [BibTexData.from_document(result) for result in results]
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
                    
                    # join sentences back together
                    cited_lines.append(". ".join(cited_sentences))

                # join lines back together into paragraph
                paragraphs[paragraph_idx] = "\n".join(cited_lines)

            # join paragraphs back together
            referenced_paragraphs = "\n\n".join(paragraphs)

            # restore environment blocks
            for i, block in enumerate(section_env_blocks):
                referenced_paragraphs = referenced_paragraphs.replace(env_placeholder.format(i), block)

            section.content = referenced_paragraphs
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
            