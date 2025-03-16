from typing import Optional, List
from pydantic import BaseModel

class DocPage(BaseModel):
    id: int
    content: str
    metadata: Optional[dict]

class DocFigure(BaseModel):
    id: int
    image_path: str
    caption: Optional[str]

class Document(BaseModel):
    path: str
    title: str
    author: str
    bib_key: str
    
    pages: List[DocPage]
    figures: Optional[List[DocFigure]]

    