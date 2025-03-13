from dataclasses import dataclass
from typing import List
import re
import os

from ..core.pdf_processor import PDFProcessor
from ..utils.logger import named_log

@dataclass
class ReferenceStore:
    all_paths: List[str] = None

    pdf_proc: PDFProcessor = None
    pdf_paths: List[str] = None
    
    non_pdf_paths: List[str] = None
    
    _full_contents: List[str] = None
    
    _bibliographies: List[str] = None
    bibtex_db_path: str = None
    
    _no_bib_contents: List[str] = None
    
    def __init__(self, reference_paths: List[str]):
        self.all_paths = reference_paths.copy()
        named_log(self, f"Loading {len(self.all_paths)} references")
        
        # separate paths between PDF and non-pdf
        self.pdf_paths = []
        self.non_pdf_paths = []
        for path in self.all_paths:
            if path.endswith(".pdf"):
                self.pdf_paths.append(path)
            else:
                self.non_pdf_paths.append(path)
        
        named_log(self, f"# PDF references: {len(self.pdf_paths)} | Non-PDF: {len(self.non_pdf_paths)}")

        self.pdf_proc = PDFProcessor(self.pdf_paths)
        # call functions to initialize values
        self.full_content(discard_bibliography=False)
        self.extract_bib_sections()

    def copy(self):
        cp = ReferenceStore(self.pdf_paths)
        
        try:
            cp._full_contents = self._full_contents.copy()
            cp._bibliographies = self._bibliographies.copy()
            cp.bibtex_db_path = self.bibtex_db_path
            cp._no_bib_contents = self._no_bib_contents.copy()
        except:
            pass
        
        return cp

    def full_content(self, discard_bibliography = True) -> List[str]:
        """
        Get full content for every reference, with the option to discard bibliography section.
        The contents are cached so they are initialized only once (in the first call).
        
        Parameters:
            - discard_bibliography (bool): discard the bibliography/references section inside each content. Recommended to reduce the number of tokens
        """
        assert(self.pdf_proc is not None)
        
        if not self._full_contents:
            # read content from PDF references first
            self._full_contents = self.pdf_proc.extract_content()
            
            # read from rest 
            if self.non_pdf_paths:
                for path in self.non_pdf_paths:
                    with open(path, "r", encoding="utf-8") as f:
                        self._full_contents.append(f.read())
        
        if discard_bibliography:
            if not self._no_bib_contents:
                self.extract_bib_sections()
            return self._no_bib_contents

        return self._full_contents
    
    def extract_bib_sections(self) -> List[str]:
        """
        Get "Bibliography/References" section of each reference PDF 
        """
        # return cached biblioghraphies
        if self._bibliographies:
            return self._bibliographies

        if not self._full_contents:
            self.full_content(discard_bibliography=False)

        # Initialize bibliographies content
        self._bibliographies = []
        self._no_bib_contents = []
        for i, content in enumerate(self._full_contents):
            # extract bibliography section with regex
            ref_match = re.search(r"(References|Bibliography|Works Cited|References and Notes|Referências|Referencias)\s*[\n\r]+", content, re.IGNORECASE)
            if ref_match:
                self._bibliographies.append(content[ref_match.start():].strip())
                self._no_bib_contents.append(content[:ref_match.start()])
            else:
                named_log(self, f"couldn't match references regex for pdf {os.path.basename(self.all_paths[i])}, using entire content")
                self._bibliographies.append(content)
                self._no_bib_contents.append(content)
    
        return self._bibliographies        

    def extract_images(self, save_dir: str, verbose=False):
        return self.pdf_proc.extract_images(save_dir, verbose=verbose)
