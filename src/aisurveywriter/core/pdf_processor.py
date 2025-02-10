from typing import List
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

class PDFProcessor:
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths = pdf_paths
        self.pdf_documents = [None] * len(pdf_paths)
        self.img_readers = [None] * len(pdf_paths)
        self._load()
        

    def _load(self):
        for i, pdf in enumerate(self.pdf_paths):
            self.pdf_documents[i] = PyPDFLoader(pdf).load()
            self.img_readers[i] = PdfReader(pdf)

    def _print(self, *msgs: str):
        print(f"({self.__class__.__name__})", *msgs)

    def extract_content(self) -> List[str]:
        contents = [""] * len(self.pdf_documents)
        for i, doc in enumerate(self.pdf_documents):
            contents[i] = "\n".join([d.page_content for d in doc])
        return contents

    def extract_images(self, save_dir: str = None):
        imgs = []
        if save_dir:
            save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            
        for reader in self.img_readers:
            img_count = 0
            for page in reader.pages:
                for img_file_obj in page.images:
                    if save_dir:
                        with open(os.path.join(save_dir, str(img_count) + img_file_obj.name), "wb") as fp:
                            fp.write(img_file_obj.data)
                    imgs.append(img_file_obj)
                    
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
        
    def vector_store(self, embeddings) -> FAISS:
        vector_store = FAISS.from_documents(self.pdf_documents, embeddings)
        return vector_store
