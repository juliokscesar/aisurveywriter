from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_community.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .text_embedding import EmbeddingsHandler

@dataclass
class BaseRAGData(ABC):
    @abstractmethod
    def to_document(self, *args, **kwargs) -> Document:
        pass

@dataclass
class WorkReferenceData(BaseRAGData):
    title:    str = ""
    abstract: str = ""
    keywords: str = ""
    bibtex_key: str = ""

    def to_document(self, *args, **kwargs):
        return Document(
            page_content=f"Title: {self.title}\nAbstract: {self.abstract}\nKeywords: {self.keywords}",
            metadata={"bibtex_key": self.bibtex_key}
        )
    
@dataclass
class GeneralTextData(BaseRAGData):
    docs: List[List[Document]] = None

    def to_document(self, *args, **kwargs):
        flat_docs = []
        for doc_list in self.docs:
            flat_docs.extend(doc_list)
    
        return flat_docs

@dataclass
class ImageData(BaseRAGData):
    id: int = -1
    basename: str = ""
    description: str = ""

    def to_document(self, *args, **kwargs):
        return Document(
            page_content=self.description,
            metadata={"id": self.id, "basename": self.basename}
        )

class AgentRAG:
    def __init__(self, embeddings: EmbeddingsHandler, bib_faiss_path: Optional[str] = None, figures_faiss_path: Optional[str] = None, content_faiss_path: Optional[str] = None):
        self._embed = embeddings

        self.bib_faiss:     FAISS = FAISS.load_local(bib_faiss_path, self._embed.model) if bib_faiss_path else None
        self.figures_faiss: FAISS = FAISS.load_local(figures_faiss_path, self._embed.model) if figures_faiss_path else None
        self.content_faiss: FAISS = FAISS.load_local(content_faiss_path, self._embed.model) if content_faiss_path else None
        

    def create_faiss(self, data_list: List[BaseRAGData], save_path: Optional[str] = None, *splitter_args, **splitter_kwargs):
        splitter = RecursiveCharacterTextSplitter(*splitter_args, **splitter_kwargs)

        docs = [data.to_document() for data in data_list]
        split_docs = splitter.split_documents(docs)
    
        faiss = FAISS.from_documents(split_docs, self._embed.model)
        if save_path:
            faiss.save_local(save_path)
        
        if isinstance(data_list[0], WorkReferenceData):
            self.bib_faiss = faiss
        elif isinstance(data_list[0], GeneralTextData):
            self.content_faiss = faiss
        elif isinstance(data_list[0], ImageData):
            self.figures_faiss = faiss
        else:
            raise TypeError(f"Unknown RAG data type: {type(data_list[0])}")
    
        return faiss
