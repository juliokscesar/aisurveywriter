from dataclasses import dataclass
from typing import Union, List

from aisurveywriter.core.file_handler import read_yaml

@dataclass
class SectionData:
    title: str
    description: str
    content: Union[None, str] = None
    
    
@dataclass
class PaperData:
    subject: str
    sections: List[SectionData]
    bib: Union[None, str] = None
    
    @staticmethod
    def from_structure_yaml(subject: str, path: str):
        sections = read_yaml(path)["sections"]
        paper = PaperData(
            subject=subject,
            sections=[SectionData(s["title"], s["description"]) for s in sections]              
        )
        return paper
            
        