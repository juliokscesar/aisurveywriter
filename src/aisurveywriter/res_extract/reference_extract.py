from typing import List, Optional
from pydantic import BaseModel, Field
import bibtexparser
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.llm_handler import LLMHandler
from ..store.reference_store import ReferenceStore
from ..utils.logger import named_log, metadata_log, cooldown_log
from ..utils.helpers import time_func
from ..utils.helpers import get_bibtex_entry, random_str

class BibliographyInfo(BaseModel):
    title: str | None = Field(description="Title of the referenced work")
    authors: str | None = Field(description="Authors of the referenced work")

class BibExtractorOutput(BaseModel):
    bibliography: List[BibliographyInfo]

REFERENCE_EXTRACTOR_SYSTEM_PROMPT = """- You work in extracting the Title and Authors from \"references\" sections in scientific papers.  
- You will receive the references section from a paper.
- Extract the **Title** and **Authors** for every reference entry.  
- DO NOT SKIP ANY REFERENCE.  
- **Output in this EXACT JSON format:**  
{format_instructions}  

- Ensure that every variable value (title and authors) is enclosed in double quotes.  
- If you can't clearly identify a Title and/or Author, return: ""; (an empty string)
- Examples of reference formats (X: number, LN: last name, F: first name abbreviation):  
    - **"(X) Author1LN, Author1F.; Author2LN, Author2F; ... AuthorNLN, AuthorNF. Some title finished by a dot."**  
      → Output as: `{{"bibliography": [{{"title": "Some title finished by a dot", "authors": "Author1LN, Author1F.; Author2LN, Author2F.; ... AuthorNLN, AuthorNF."}}]}}`  
- Do not confuse the "Journal" for the Author or Title. Journals usually contain "Science", "Review", "Letters", etc."""


REFERENCE_EXTRACTOR_HUMAN_PROMPT = """Extract from the given references:
\"\"\"
{references}
\"\"\""""

class ReferencesBibExtractor:
    def __init__(self, llm: LLMHandler, references: ReferenceStore, system_prompt: str = REFERENCE_EXTRACTOR_SYSTEM_PROMPT, human_prompt: str = REFERENCE_EXTRACTOR_HUMAN_PROMPT, n_batches: int = 10, request_cooldown_sec: int = 30):
        self.llm = llm
        self._cooldown = request_cooldown_sec
        self.parser = PydanticOutputParser(pydantic_object=BibExtractorOutput)
        self.n_batches = max(1, n_batches)
        
        self._system = SystemMessagePromptTemplate.from_template(system_prompt)
        self._human = HumanMessagePromptTemplate.from_template(human_prompt)
        self._prompt = ChatPromptTemplate([self._system, self._human], partial_variables={"format_instructions": self.parser.get_format_instructions()})

        self.llm.init_chain(self._prompt)

        self.references = references
        
        
    def extract(self):
        bib_sections = self.references.bibliography_sections()
        
        chunk_sz = sum(len(bib) for bib in bib_sections) // self.n_batches
        chunk_overlap = int(chunk_sz * 0.05)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_sz,
            chunk_overlap=chunk_overlap,
        )

        bib_docs = text_splitter.create_documents(bib_sections)
        n_chunks = len(bib_docs)

        bib_info: List[BibliographyInfo] = []
        
        for i, bib_doc in enumerate(bib_docs):
            named_log(self, f"==> start extracting title and author from batch {i+1}/{n_chunks}")
            
            elapsed, response = time_func(self.llm.invoke, {
                "references": bib_doc.page_content,
            })
            try:
                bib_output: BibExtractorOutput = self.parser.invoke(response)
                bib_info.extend(bib_output.bibliography)
                named_log(self, f"==> finish extracting title and author from batch {i+1}/{n_chunks}")
                named_log(self, f"==> got {len(bib_output.bibliography)} references | total: {len(bib_info)}")
            except Exception as e:
                named_log(self, f"unable to get references from batch {i+1}. Exception raised: {e}")
            
            metadata_log(self, elapsed, response)
            if self._cooldown:
                cooldown_log(self, self._cooldown)

        return bib_info

    def to_bibtex_db(self, bib_info: List[BibliographyInfo], filter_duplicates = True, save_path: Optional[str] = None):
        bibtex_db = bibtexparser.bibdatabase.BibDatabase()
        for ref in bib_info:
            try:
                entry = get_bibtex_entry(ref.title, ref.authors)
                if not entry:
                    named_log(self, f"==> no bibtex entry for: {ref}")
                    continue
                
                entry["ID"] = f"key{len(bibtex_db.entries)}"
                bibtex_db.entries.append(entry)
            except Exception as e:
                named_log(self, f"==> bibtex entry raised exception for {ref!r}: {e}")
        
        if filter_duplicates:
            bibtex_db, amount = self._filter_duplicates_bibtexdb(bibtex_db)
            named_log(self, f"==> removed {amount} duplicated bibtex entries")
        if save_path:
            with open(save_path, "w") as bibfile:
                bibtexparser.dump(bibtex_db, bibfile)
                
        return bibtex_db

    def _filter_duplicates_bibtexdb(self, bibtex_db: bibtexparser.bibdatabase.BibDatabase):
        seen_dois = set()
        seen_keys = set()
        seen_titles = set()
        unique_entries = []
        for entry in bibtex_db.entries:
            title = entry.get("title", entry.get("booktitle", None))
            if title:
                title = title.strip().lower()
            if title and title in seen_titles:
                continue
            seen_titles.add(title)
            
            doi = entry.get("doi", None)
            if doi:
                doi = doi.replace(" ", "").strip().lower()
            if doi and doi in seen_dois:
                continue
            
            key = entry.get("ID", None)
            if not key or key in seen_keys:
                entry["ID"] = random_str(8)
            seen_keys.add(entry["ID"])
            seen_dois.add(doi)
            unique_entries.append(entry)
            
        new_db = bibtexparser.bibdatabase.BibDatabase()
        new_db.entries = unique_entries
        return new_db, max(0, len(bibtex_db.entries) - len(unique_entries))
