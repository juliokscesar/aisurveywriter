from abc import ABC, abstractmethod
from typing import List, Optional
from enum import IntFlag, auto
from functools import reduce
import operator
from pydantic import BaseModel
import os

from langchain_community.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .text_embedding import EmbeddingsHandler
from .reference_store import ReferenceStore

from aisurveywriter.tasks.new_reference_extract import ReferencesBibExtractor
from aisurveywriter.utils.helpers import random_str

class RAGType(IntFlag):
    Null        = 0
    BibTex      = auto()
    GeneralText = auto()
    ImageData   = auto()

    @property
    def All(self):
        return reduce(operator.or_, self.__class__)


class BaseRAGData(ABC, BaseModel):
    data_type: RAGType = RAGType.Null
    
    @abstractmethod
    def to_document(self, *args, **kwargs) -> Document:
        pass


class WorkReferenceData(BaseRAGData):
    data_type: RAGType = RAGType.BibTex
    
    title:    str = ""
    abstract: str = ""
    keywords: str = ""
    bibtex_key: str = ""

    def to_document(self, *args, **kwargs):
        return Document(
            page_content=f"Title: {self.title}\nAbstract: {self.abstract}\nKeywords: {self.keywords}",
            metadata={"bibtex_key": self.bibtex_key}
        )
    

class GeneralTextData(BaseRAGData):
    data_type: RAGType = RAGType.GeneralText
    text: str

    def to_document(self, *args, **kwargs):
        return Document(page_content=self.text)


class ImageData(BaseRAGData):
    data_type: RAGType = RAGType.ImageData
    id: int = -1
    basename: str = ""
    description: str = ""

    def to_document(self, *args, **kwargs):
        return Document(
            page_content=self.description,
            metadata={"id": self.id, "basename": self.basename}
        )

class AgentRAG:
    def __init__(self, embeddings: EmbeddingsHandler, bib_faiss_path: Optional[str] = None, figures_faiss_path: Optional[str] = None, content_faiss_path: Optional[str] = None, ref_bib_extractor: Optional[ReferencesBibExtractor] = None, request_cooldown_sec: int = 30, output_dir: str = "out"):
        self._embed = embeddings

        self.bib_faiss:     FAISS = FAISS.load_local(bib_faiss_path, self._embed.model) if bib_faiss_path else None
        self.figures_faiss: FAISS = FAISS.load_local(figures_faiss_path, self._embed.model) if figures_faiss_path else None
        self.content_faiss: FAISS = FAISS.load_local(content_faiss_path, self._embed.model) if content_faiss_path else None
        
        self.faiss_rags = {
            RAGType.BibTex: self.bib_faiss,
            RAGType.GeneralText: self.content_faiss,
            RAGType.ImageData: self.figures_faiss,
        }
        self.create_rags_funcmap = {
            RAGType.BibTex: self.create_bib_rag,
            RAGType.GeneralText: self._create_content_rag,
            RAGType.ImageData: self._create_figures_rag,
        }

        self.ref_bib_extractor = ref_bib_extractor

        self._cooldown = request_cooldown_sec
        self.output_dir = os.path.abspath(output_dir)

    def create_rags(self, rag_types: RAGType, references: ReferenceStore):
        for rag_type in rag_types:
            func_create_rag = self.create_rags_funcmap[rag_type]
            self.faiss_rags[rag_type] = func_create_rag(references)
   

    @staticmethod
    def create_faiss(embed: EmbeddingsHandler, data_list: List[BaseRAGData], save_path: Optional[str] = None, *splitter_args, **splitter_kwargs):
        splitter = RecursiveCharacterTextSplitter(*splitter_args, **splitter_kwargs)

        docs = [data.to_document() for data in data_list]
        split_docs = splitter.split_documents(docs)
    
        faiss = FAISS.from_documents(split_docs, embed.model)
        if save_path:
            faiss.save_local(save_path)
        
        return faiss

    
    def create_bib_rag(self, references: ReferenceStore): 
        if not self.ref_bib_extractor:
            self.ref_bib_extractor = ReferencesBibExtractor(self.llm, references, request_cooldown_sec=self._cooldown)
        
        bib_info = self.ref_bib_extractor.extract()
        save_path = os.path.join(self.output_dir, "refextract-bibdb.bib")
        bibtex_db = self.ref_bib_extractor.to_bibtex_db(bib_info, save_path=save_path)

        bib_data: List[WorkReferenceData] = []
        for entry in bibtex_db.entries:
            bib_data.append(WorkReferenceData(
                title=entry.get("title", "Unknown"),
                abstract=entry.get("abstract", ""),
                keywords=entry.get("keyword", ""),
                bibtex_key=entry.get("ID", random_str()),
            ))
        
        return AgentRAG.create_faiss(self._embed, bib_data, save_path=save_path.replace(".bib", ".faiss"))

    def create_content_rag(self, references: ReferenceStore):
        content = references.full_content(discard_bibliography=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
        docs = splitter.create_documents(content)
        docs = splitter.split_documents(docs)
        
        content_data = [GeneralTextData(text=doc.page_content) for doc in docs]
        save_path = os.path.join(self.output_dir, "content-rag.faiss")
        return AgentRAG.create_faiss(self._embed, content_data, save_path=save_path)
        