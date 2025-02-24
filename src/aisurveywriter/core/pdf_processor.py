from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import fitz
from pathlib import Path
import re
from pydantic import BaseModel

from aisurveywriter.utils import named_log

class PDFImageData(BaseModel):
    pdf_source: str
    data: bytes
    ext: str
    path: str

class PDFProcessor:
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths: List[str] = pdf_paths
        self.pdf_documents: List[List[Document]] = [None] * len(pdf_paths)
        self.img_readers: List[fitz.Document] = [None] * len(pdf_paths)
        self._load()
        

    def _load(self):
        for i, pdf in enumerate(self.pdf_paths):
            self.pdf_documents[i] = PyPDFLoader(pdf).load()
            self.img_readers[i] = fitz.open(pdf)

    def _print(self, *msgs: str):
        print(f"({self.__class__.__name__})", *msgs)

    def extract_content(self) -> List[str]:
        contents = [""] * len(self.pdf_documents)
        for i, doc in enumerate(self.pdf_documents):
            contents[i] = "\n".join([d.page_content for d in doc])
        return contents

    def extract_images(self, save_dir: str = None, verbose=True) -> List[PDFImageData]:
        imgs = []
        if save_dir:
            save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            
        for pdf_idx, reader in enumerate(self.img_readers):
            for page_idx in range(len(reader)):
                page = reader.load_page(page_idx)
                img_list = page.get_images(full=True)
                
                for img_idx, img in enumerate(img_list):
                    xref = img[0]
                    base_img = reader.extract_image(xref)
                    img_bytes = base_img["image"]
                    img_ext = base_img["ext"]
                    img_name = f"{Path(self.pdf_paths[pdf_idx]).stem}_page{page_idx}_image{img_idx}.{img_ext}"
                    
                    if save_dir:
                        path = os.path.join(save_dir, img_name)
                        with open(path, "wb") as f:
                            f.write(img_bytes)
                        
                        if verbose:
                            named_log(self, f"Image saved: {path}")

                    imgs.append(PDFImageData(
                        pdf_source=self.pdf_paths[pdf_idx],
                        data=img_bytes,
                        ext=img_ext,
                        path=os.path.join(save_dir, img_name) if save_dir else img_name
                    ))
                    
        return imgs
            

    def summarize_content(self, summarizer, chunk_size: int = 2000, chunk_overlap: int = 200, show_metadata = False) -> List[str]:
        # Split all pdf content in smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(self.pdf_documents)

        # Summarize each chunk
        size = len(chunks)
        summaries = [None] * size
        for i, chunk in enumerate(chunks):
            summary = summarizer.invoke(f"Summarize the content of the following text:\n\n{chunk}")
            if show_metadata:
                self._print(f"Summary response metadata (chunk {i+1}/{size}):", summary.usage_metadata)
            summaries[i] = summary.content
        
        return "\n".join(summaries)
        
    def faiss(self, embeddings, chunk_size: int = 4000) -> FAISS:
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        docs = []
        # first remove references
        for doc in self.pdf_documents:
            for pg in doc:
                ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", pg.page_content, re.IGNORECASE)
                if ref_match:
                    pg.page_content = pg.page_content[:ref_match.start()].strip()
            docs.extend(doc)
        docs = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
