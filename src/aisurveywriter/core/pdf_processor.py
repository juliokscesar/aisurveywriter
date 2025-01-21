from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import Embeddings
from langchain.vectorstores import FAISS

class PDFProcessor:
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths = pdf_paths
        self.pdf_documents = [None] * len(pdf_paths)
        self._load()

    def _load(self):
        for i, pdf in enumerate(self.pdf_paths):
            self.pdf_documents[i] = PyPDFLoader(pdf).load()

    def _print(self, *msgs: str):
        print(f"({self.__class__.__name__})", *msgs)

    def extract_content(self) -> List[str]:
        contents = [""] * len(self.pdf_documents)
        for i, doc in enumerate(self.pdf_documents):
            contents[i] = doc.page_content
        return contents

    def summarize_content(self, summarizer, chunk_size: int = 2000, chunk_overlap: int = 200, show_metadata = False) -> List[str]:
        # Split all pdf content in smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(self.pdf_documents)

        # Summarize each chunk
        size = len(chunks)
        summaries = [None] * size
        for i, chunk in enumerate(chunks):
            summary = summarizer.invoke(f"Summarize this text: {chunk}")
            if show_metadata:
                self._print(f"Summary response metadata (chunk {i+1}/{size}):", summary.usage_metadata)
            summaries[i] = summary.content
        
        return "\n".join(summaries)
        
    def vector_store(self, embeddings: Embeddings) -> FAISS:
        vector_store = FAISS.from_documents(self.pdf_documents, embeddings)
        return vector_store
