from dataclasses import dataclass
from typing import List
import re
import os

from .pdf_processor import PDFProcessor
from aisurveywriter.utils.logger import named_log

@dataclass
class ReferenceStore:
    pdf_proc: PDFProcessor = None
    pdf_paths: List[str] = None
    
    _full_contents: List[str] = None
    _bibliographies: List[str] = None
    _no_bib_contents: List[str] = None
    
    def __init__(self, reference_paths: List[str]):
        self.pdf_paths = reference_paths.copy()
        self.pdf_proc = PDFProcessor(self.pdf_paths)

    def copy(self):
        cp = ReferenceStore(self.pdf_paths)
        
        try:
            cp._full_contents = self._full_contents.copy()
            cp._bibliographies = self._bibliographies.copy()
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
            self._full_contents = self.pdf_proc.extract_content()
        
        if discard_bibliography:
            return self.extract_bib_sections()

        return self._full_contents
    
    def extract_bib_sections(self) -> List[str]:
        """
        Get "Bibliography/References" section of each reference PDF 
        """
        # return cached biblioghraphies
        if self._bibliographies:
            return self._bibliographies

        assert(self._full_contents is not None)
        # Initialize bibliographies content
        self._bibliographies = []
        for i, content in enumerate(self._full_contents):
            # extract bibliography section with regex
            ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", content, re.IGNORECASE)
            if ref_match:
                self._bibliographies.append(content[ref_match.start():].strip())
            else:
                named_log(self, f"couldn't match references regex for pdf {os.path.basename(self.pdf_paths[i])}, using entire content")
                self._bibliographies.append(content)
    
        return self._bibliographies        
