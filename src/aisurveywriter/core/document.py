from typing import Optional, List
from pydantic import BaseModel

class DocPage(BaseModel):
    id: int
    content: str
    metadata: Optional[dict] = None
    source_path: Optional[str] = None

class DocFigure(BaseModel):
    id: int
    image_path: str
    caption: Optional[str] = None
    source_path: Optional[str] = None

class Document(BaseModel):
    path: str
    title: str
    author: Optional[str] = None
    bibtex_entry: Optional[dict] = None
    
    pages: List[DocPage]
    figures: Optional[List[DocFigure]] = None

    def full_content(self) -> str:
        page_contents = [page.content for page in self.pages]
        return "\n".join(page_contents)
