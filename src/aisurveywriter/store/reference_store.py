from dataclasses import dataclass
from pydantic import BaseModel
from typing import List, Optional
import re
import os
import pickle

# from ..core.pdf_processor import PDFProcessor
from ..core.pdf_processor import PDFProcessor
from ..core.document import Document, DocFigure, DocPage
from ..core.llm_handler import LLMHandler
from ..utils.logger import named_log, global_log
from ..utils.helpers import get_bibtex_entry

class ReferenceStore(BaseModel):
    documents: List[Document]
    paths: List[str]
    
    bibtex_db_path: Optional[str] = None

    def __init__(self, reference_documents: List[Document]):
        super().__init__(documents=reference_documents, paths=[doc.path for doc in reference_documents], bibtex_db_path=None)

        self._cache: dict[str, List[str]] = {
            "doc_full_contents": None,
            "doc_nobib_contents": None,
            "doc_bib_sections": None,
        }
        self._load_cache()
        
    def _load_cache(self, full_contents=True, bib_sections=True):
        # load full contents
        if full_contents:
            self._cache["doc_full_contents"] = [doc.full_content() for doc in self.documents]
        
        # extract bibliography from contents
        if bib_sections:
            self._cache["doc_nobib_contents"], self._cache["doc_bib_sections"] = self._extract_bibliography()

    def _extract_bibliography(self) -> tuple[List[str], List[str]]:
        """
        Extract the bibliography/references/works cited sections from every reference document
        
        Returns:
            nobib_contents (List[str]): content without bibliography section for every document
            bib_sections (List[str]): bibliography section content for every document
        """
        nobib_contents = []
        bib_sections = []
        bib_section_pat = re.compile(r"(References|Bibliography|Works Cited|References and Notes|Referências|Referencias)\s*[\n\r]+", re.IGNORECASE)
        for i, document in enumerate(self.documents):
            content = document.full_content()
            # extract bibliography section with regex
            ref_match = bib_section_pat.search(content)
            if ref_match:
                bib_sections.append(content[ref_match.start():].strip())
                nobib_contents.append(content[:ref_match.start()])
            else:
                named_log(self, f"couldn't match references regex for pdf {os.path.basename(self.paths[i])}, using entire content")
                bib_sections.append(content)
                nobib_contents.append(content)

        return nobib_contents, bib_sections
            
    
    def docs_contents(self, discard_bibliography=True) -> List[str]:
        if not discard_bibliography:
            if not self._cache["doc_full_contents"]:
                self._load_cache(full_contents=True, bib_sections=False)
            return self._cache["doc_full_contents"]
        
        if not self._cache["doc_nobib_contents"]:
            self._load_cache()
        return self._cache["doc_nobib_contents"]
    
    def full_content(self, discard_bibliography=True) -> str:
        contents = self.docs_contents(discard_bibliography)
        return "\n\n".join(contents)

    def bibliography_sections(self) -> List[str]:
        if not self._cache["doc_bib_sections"]:
            self._load_cache()
        return self._cache["doc_bib_sections"]

    def all_figures(self) -> List[DocFigure]:
        all_figures = []
        for doc in self.documents:
            if doc.figures:
                all_figures.extend(doc.figures)
        return all_figures
    
    def bibtex_entries(self) -> List[dict]:
        entries = []
        for doc in self.documents:
            if doc.bibtex_entry:
                entries.append(doc.bibtex_entry)
        return entries

    def doc_from_path(self, path: str) -> Document | None:
        for doc in self.documents:
            if path in doc.path:
                return doc
        return None

    @staticmethod
    def from_local(path: str):
        with open(path, "rb") as f:
            store: ReferenceStore = pickle.load(f)
        return store

def load_nonpdf_references(paths: List[str], title_extractor_llm: Optional[LLMHandler] = None):
    non_pdf_documents: List[Document] = []
    title_pattern = re.compile(r"^(?:title)\s*[:\.-]*\s*(.+?)[\n]", re.IGNORECASE)
    for path in paths:
        if not os.path.isfile(path):
            global_log("(Non-PDF reference load) skipping invalid file:", path)
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # try some approaches to extract title from unknown type of file
        portion = content[int(len(content) * 0.2)] # use only first 20% of the content
        title = None
        if title_match := title_pattern.search: # try with regex
            title = title_match.group(1)
        elif title_extractor_llm: # try with llm
            try:
                title = title_extractor_llm.send_prompt(f"Extract the title from this document based on its content:\n"+
                                                        f"\n[begin: document_content]\n"+
                                                        f"{portion}\n"+
                                                        f"[end: document_content]\n\n"+
                                                        f"Return ONLY the title, nothing else.").content
            except Exception as e:
                global_log(f"(Non-PDF reference load) unable to send prompt to extract title:", e)
        
        # try to get bibtex entry if title was found
        bibtex_entry: dict = None
        if title:
            try:
                bibtex_entry = get_bibtex_entry(title, None)
                if not bibtex_entry:
                    global_log(f"(Non-PDF reference load) unable to get bibtex entry for file {path}, title: {title}")
            except Exception as e:
                bibtex_entry = None
                global_log(f"(Non-PDF reference load) bibtex entry raised exception for file {path}: {e}\n")
        author = None
        if bibtex_entry:
            bibtex_entry["ID"] = f"key_ref{os.path.basename(path).replace(".","_")}"
            author = bibtex_entry.get("author", None)
        
        doc_page = DocPage(0, content, source_path=path)
        document = Document(path=path, title=title, author=author, bibtex_entry=bibtex_entry,
                            pages=[doc_page], figures=None)
        non_pdf_documents.append(document)
    
    return non_pdf_documents

def build_reference_store(reference_paths: List[str], images_output_dir: str = "output", 
                          save_local: Optional[str] = None, 
                          title_extractor_llm: Optional[LLMHandler] = None, **lp_kwards) -> ReferenceStore:
    # separate between pdf and non-pdf files
    pdf_paths: List[str] = []
    non_pdf_paths: List[str] = []
    for path in reference_paths:
        if path.endswith(".pdf"):
            pdf_paths.append(path)
        else:
            non_pdf_paths.append(path)
            
    # process pdfs first
    pdf_processor = PDFProcessor(pdf_paths, images_output_dir, **lp_kwards)
    pdf_documents = pdf_processor.documents
    if not non_pdf_paths:
        reference_store = ReferenceStore(pdf_documents)
    else:
        # process non-pdfs
        non_pdf_documents: List[Document] = load_nonpdf_references(non_pdf_paths, title_extractor_llm)
        reference_store = ReferenceStore(pdf_documents + non_pdf_documents)

    if save_local:
        with open(save_local, "wb") as f:
            pickle.dump(reference_store, f)
        
    return reference_store
