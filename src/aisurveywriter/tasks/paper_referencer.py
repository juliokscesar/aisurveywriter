from typing import List
import re
import random
import bibtexparser

from langchain_core.prompts.chat import ChatMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from .pipeline_task import PipelineTask
from ..core.agent_context import AgentContext
from ..core.agent_rags import RAGType, BibTexData, FAISS
from ..core.paper import PaperData
from ..utils.logger import named_log, cooldown_log, metadata_log
from ..utils.helpers import assert_type, time_func


class PaperReferencer(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, reviewed_paper: PaperData, save_bib_path: str, max_per_section: int = 90, max_per_sentence: int = 4, confidence: float = 0.9, max_same_ref: int = 10):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = reviewed_paper
        
        self.save_bib_path = save_bib_path
        self.max_per_section = max_per_section
        self.max_per_sentence = max_per_sentence
        self.max_same_ref = max_same_ref
        self.confidence = confidence
        
        self._system_prompt = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.add_references.text)
        self._human_prompt = HumanMessagePromptTemplate.from_template("[begin: references]\n\n"+
                                                                      "{references}\n\n"+
                                                                      "[end: references]\n\n"+
                                                                      "[begin: paragraph_info]\n\n"+
                                                                      "{paragraph_info}\n\n"+
                                                                      "[end: paragraph_info]")
        
    def reference(self) -> PaperData:
        assert self.agent_ctx.rags.is_enabled(RAGType.BibTex)
        self.agent_ctx.llm_handler.init_chain_messages(self._system_prompt, self._human_prompt)
        
        used_keys = {}
        probabilities = {1: 0.55, 2: 0.2, 3: 0.14, 4: 0.11}
        
        section_amount = len(self.agent_ctx._working_paper.sections)

        # patterns for removing latex environment from text
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
        cite_keys_pattern = re.compile(r"\\cite{(.+?)}")
        
        for section_idx, section in enumerate(self.agent_ctx._working_paper.sections):
            # skip some sections (like abstract)
            if skip_section_pattern.match(section.title):
                continue            

            # remove any environment blocks to avoid misreferencing
            preprocessed = env_pattern.sub(store_env_block, section.content)

            # split section into paragraphs
            paragraphs = paragraph_pattern.split(preprocessed)
            
            # use less references in last section (Conclusion)
            max_refs_section = self.max_per_section if section_idx != section_amount-1 else self.max_per_section//5          
            ref_count = 0
            
            for paragraph_idx, paragraph in enumerate(paragraphs):
                if ref_count >= max_refs_section:
                    break
                if not paragraph.strip() or "\\section" in paragraph or "\\subsection" in paragraph or "@@LATEX_ENV" in paragraph:
                    continue
                
                # get most relevant references for this paragraph and make it a "sub-database"
                paragraph_refs = self.agent_ctx.rags.retrieve(RAGType.BibTex, paragraph.strip(), k=50)
                # filter keys overly used
                paragraph_refs = [ref for ref in paragraph_refs if ref.bibtex_key not in used_keys or used_keys[ref.bibtex_key] < self.max_same_ref]
                if not paragraph_refs:
                    continue

                named_log(self, f"request LLM to add references in paragraph {paragraph_idx+1}/{len(paragraphs)}, section {section_idx+1}/{section_amount}")
                refs_str = "\n\n".join([f"Title: {ref.title}\nAbstract: {ref.abstract}\nKeywords: {ref.keywords}\n**BIBTEX_KEY**: {ref.bibtex_key}" for ref in paragraph_refs])
                elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
                    "subject": self.agent_ctx._working_paper.subject,
                    "references": refs_str,
                    "paragraph_info": f"*SECTION TITLE*: {section.title}\n*PARAGRAPH*:\n{paragraph}"
                })
                response.content = re.sub(r"`+\w*", "", response.content)
                
                # get every key used by the LLM
                cites = cite_keys_pattern.findall(response.content)
                paragraph_keys = []
                for cite in cites:
                    keys = [key.strip() for key in cite.split(",")]
                    for key in keys:
                        if key not in used_keys:
                            used_keys[key] = 0
                        used_keys[key] += 1
                        ref_count += 1
                    paragraph_keys.extend(keys)
                paragraph_keys = set(paragraph_keys)

                # log metadata usage and how many references were used in this paragraph
                metadata_log(self, elapsed, response)
                named_log(self, f"{len(paragraph_keys)} references added to paragraph {paragraph_idx+1}/{len(paragraphs)}")

                if self.agent_ctx.llm_cooldown:
                    cooldown_log(self, self.agent_ctx.llm_cooldown)
                    
                # update paragraph in section
                paragraphs[paragraph_idx] = response.content
                
            # join paragraphs back together
            referenced_paragraphs = "\n\n".join(paragraphs)

            # restore environment blocks
            for block_idx, block in enumerate(section_env_blocks):
                referenced_paragraphs = referenced_paragraphs.replace(env_placeholder.format(block_idx), block)

            section.content = referenced_paragraphs
            named_log(self, f"added {ref_count} references to section ({section_idx+1}/{section_amount}): \"{section.title}\"")

        self._dump_used_bib(used_keys)
        self.agent_ctx._working_paper.bib_path = self.save_bib_path
        return self.agent_ctx._working_paper
        
    def _dump_used_bib(self, used_keys: list):
        with open(self.agent_ctx.references.bibtex_db_path, "r", encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)
        
        used_entries = [entry for entry in bib_db.entries if entry["ID"] in used_keys]
        # also add all document bibtex entries for the used keys
        used_entries.extend([doc.bibtex_entry for doc in self.agent_ctx.references.documents if doc.bibtex_entry])
        
        used_db = bibtexparser.bibdatabase.BibDatabase()
        used_db.entries = used_entries
        
        with open(self.save_bib_path, "w", encoding="utf-8") as f:
            bibtexparser.dump(used_db, f)

    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        return self.reference()
            