from abc import ABC, abstractmethod
from typing import List, Optional
from enum import IntFlag, auto
from functools import reduce
import operator
from pydantic import BaseModel
import os
import bibtexparser
import re

from langchain_community.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .text_embedding import EmbeddingsHandler
from .llm_handler import LLMHandler
from ..store.reference_store import ReferenceStore, DocFigure
from ..res_extract import ReferencesBibExtractor
from ..utils.helpers import random_str
from ..utils.logger import named_log

class RAGType(IntFlag):
    Null        = 0
    BibTex      = auto()
    GeneralText = auto()
    ImageData   = auto()

RAGType.All = reduce(operator.or_, RAGType)

class BaseRAGData(ABC, BaseModel):
    data_type: RAGType = RAGType.Null
    
    @abstractmethod
    def to_document(self, *args, **kwargs) -> Document:
        pass


class BibTexData(BaseRAGData):
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

    @staticmethod
    def from_document(doc: Document):
        # Regex pattern
        pattern = r"Title:\s*(?P<title>.*?)\nAbstract:\s*(?P<abstract>.*?)\nKeywords:\s*(?P<keywords>.*)"

        # Extract information
        re_match = re.search(pattern, doc.page_content, re.DOTALL)
        if re_match:
            return BibTexData(
                title=re_match.group("title"),
                abstract=re_match.group("abstract"),
                keywords=re_match.group("keywords"),
                bibtex_key=doc.metadata["bibtex_key"]
            )
        else:
            return BibTexData(
                title="Unknown",
                abstract=doc.page_content,
                keywords="-",
                bibtex_key=doc.metadata["bibtex_key"]
            )
    

class GeneralTextData(BaseRAGData):
    data_type: RAGType = RAGType.GeneralText
    
    text: str
    source_pdf: str = ""

    def to_document(self, *args, **kwargs):
        return Document(page_content=self.text, metadata={"source_pdf": self.source_pdf})

    @staticmethod
    def from_document(doc: Document):
        return GeneralTextData(text=doc.page_content, **doc.metadata)


class ImageData(BaseRAGData):
    data_type: RAGType = RAGType.ImageData
    id: int = -1
    basename: str = ""
    source_pdf: str = ""
    caption: str = ""

    def to_document(self, *args, **kwargs):
        return Document(
            page_content=self.caption,
            metadata={"id": self.id, "basename": self.basename, "source_pdf": self.source_pdf}
        )
        
    @staticmethod
    def from_document(doc: Document):
        return ImageData(caption=doc.page_content, **doc.metadata)

    def to_doc_figure(self, reference_store: ReferenceStore) -> DocFigure:
        doc = reference_store.doc_from_path(self.source_pdf)
        if not doc:
            return DocFigure(id=self.id, image_path=self.basename, caption=self.caption, source_path=self.source_pdf)
        else:
            return doc.figures[self.id]
        

class AgentRAG:
    def __init__(self, embeddings: EmbeddingsHandler, llm: LLMHandler = None, 
                 bib_faiss_path: Optional[str] = None, figures_faiss_path: Optional[str] = None, 
                 content_faiss_path: Optional[str] = None, ref_bib_extractor: Optional[ReferencesBibExtractor] = None, 
                 request_cooldown_sec: int = 30, output_dir: str = "out", confidence: float = 0.6):
        self._embed = embeddings
        self._llm = llm

        self.bib_faiss:     FAISS = FAISS.load_local(bib_faiss_path, self._embed.model, allow_dangerous_deserialization=True) if bib_faiss_path else None
        self.figures_faiss: FAISS = FAISS.load_local(figures_faiss_path, self._embed.model, allow_dangerous_deserialization=True) if figures_faiss_path else None
        self.content_faiss: FAISS = FAISS.load_local(content_faiss_path, self._embed.model, allow_dangerous_deserialization=True) if content_faiss_path else None
        
        self.rag_type_data = {
            RAGType.BibTex: BibTexData,
            RAGType.GeneralText: GeneralTextData,
            RAGType.ImageData: ImageData,
        }
        self.faiss_rags = {
            RAGType.BibTex: self.bib_faiss,
            RAGType.GeneralText: self.content_faiss,
            RAGType.ImageData: self.figures_faiss,
        }
        self.create_rags_funcmap = {
            RAGType.BibTex: self.create_bib_rag,
            RAGType.GeneralText: self.create_content_rag,
            RAGType.ImageData: self.create_figures_rag,
        }

        self.confidence = confidence

        self.ref_bib_extractor = ref_bib_extractor

        self._cooldown = request_cooldown_sec
        self.output_dir = os.path.abspath(output_dir)

    def create_rags(self, rag_types: RAGType, references: ReferenceStore):
        for rag_type in rag_types:
            if self.faiss_rags[rag_type]:
                named_log(self, f"FAISS type: {rag_type.name} already loaded, skipping creation...")
                continue
            named_log(self, f"Creating FAISS: {rag_type.name}")
            create_rag_func = self.create_rags_funcmap[rag_type]
            self.faiss_rags[rag_type] = create_rag_func(references)
    
    def is_disabled(self, rag_type: RAGType):
        return (self.faiss_rags[rag_type] is None)

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
            self.ref_bib_extractor = ReferencesBibExtractor(self._llm, references, request_cooldown_sec=self._cooldown, n_batches=150)
        
        if not references.bibtex_db_path:
            bib_info = self.ref_bib_extractor.extract()

            references.bibtex_db_path = os.path.join(self.output_dir, "refextract-bibdb.bib")
            bibtex_db = self.ref_bib_extractor.to_bibtex_db(bib_info)
            
            # insert keys from loaded references at the beginning
            bibtex_db.entries = references.bibtex_entries() + bibtex_db.entries

            with open(references.bibtex_db_path, "w", encoding="utf-8") as f:
                bibtexparser.dump(bibtex_db, f)
            
        else:
            with open(references.bibtex_db_path, "r", encoding="utf-8") as f:
                bibtex_db = bibtexparser.load(f)

        bib_data: List[BibTexData] = []
        for entry in bibtex_db.entries:
            bib_data.append(BibTexData(
                title=entry.get("title", "Unknown"),
                abstract=entry.get("abstract", ""),
                keywords=entry.get("keyword", ""),
                bibtex_key=entry.get("ID", random_str()),
            ))
        
        return AgentRAG.create_faiss(self._embed, bib_data, save_path=references.bibtex_db_path.replace(".bib", ".faiss"))


    def create_content_rag(self, references: ReferenceStore):
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
        
        content_data = []
        # use content without refrences
        nobib_contents = references.docs_contents()
        for ref_path, ref_content in zip(references.paths, nobib_contents):
            ref_basename = os.path.basename(ref_path)
            
            # split into chunks
            split_docs = splitter.create_documents(ref_content)
            splitted = splitter.split_documents(split_docs)
            content_data.extend([
                GeneralTextData(text=split.page_content, source_pdf=ref_basename) for split in splitted
            ])
        
        save_path = os.path.join(self.output_dir, "content-rag.faiss")
        return AgentRAG.create_faiss(self._embed, content_data, save_path=save_path)
        

    def create_figures_rag(self, references: ReferenceStore):
        doc_figures = references.all_figures()
        figures_rag_data: List[ImageData] = []
        for doc_figure in doc_figures:
            # skip figures without captions/description
            if not doc_figure.caption:
                continue
            rag_data = ImageData(
                id=doc_figure.id,
                basename=os.path.basename(doc_figure.image_path),
                source_pdf=os.path.basename(doc_figure.source_path),
                caption=doc_figure.caption
            )
            figures_rag_data.append(rag_data)
        
        save_path = os.path.join(self.output_dir, "figures-rag.faiss")
        return AgentRAG.create_faiss(self._embed, figures_rag_data, save_path)

    def retrieve(self, rag: RAGType, query: str, k: int = 10, confidence: Optional[float] = None):
        assert(not self.is_disabled(rag))
        
        if not confidence:
            results = self.faiss_rags[rag].similarity_search(query, k)
            return [self.rag_type_data[rag].from_document(doc) for doc in results]
    
        assert(0.0 < confidence < 1.0)
        results = self.faiss_rags[rag].similarity_search_with_score(query, k)
        valid = []
        for result, score in results:
            if score < confidence:
                continue
            valid.append(self.rag_type_data[rag].from_document(result))
        
        return valid
    
