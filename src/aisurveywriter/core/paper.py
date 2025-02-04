from dataclasses import dataclass
from typing import Union, List

@dataclass
class SectionData:
    title: str
    description: str
    content: str
    bib: Union[None, str] = None
    
@dataclass
class PaperData:
    subject: str
    sections: List[SectionData]
    