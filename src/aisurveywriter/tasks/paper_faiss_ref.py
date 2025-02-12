from typing import List
import re
from typing import Optional
import bibtexparser
from time import sleep

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from .pipeline_task import PipelineTask
from aisurveywriter.core.paper import PaperData
from aisurveywriter.utils import named_log, time_func, countdown_log

class PaperFAISSReferencer(PipelineTask):
    def __init__(self, embed_model, bibdb_path: str, paper: PaperData = None, local_faissdb_path=None, save_usedbib_path: str = None, save_faiss_path: str = None, max_per_section: int = 60, max_per_sentence: int = 1, confidence=0.9):
        self.llm = None
        self.paper = paper
        self.embed_model = embed_model
        self.bibdb_path = bibdb_path
        self.save_usedbib_path = save_usedbib_path
        self.local_faissdb_path = local_faissdb_path
        self.max_per_section = max_per_section
        self.max_per_sentence = max_per_sentence
        self.confidence = confidence
        self.save_faiss_path = save_faiss_path
        
        if local_faissdb_path:
            self.vector_store = FAISS.load_local(local_faissdb_path, embed_model, allow_dangerous_deserialization=True)
            self.vector_store
            named_log(self, f"Loaded local FAISS")
        else:
            self.vector_store = self._faiss_from_bibdb(embed_model, bibdb_path, chunks=1000)
            named_log(self, "Created FAISS from .bib database")
            if save_faiss_path:
                self.vector_store.save_local(save_faiss_path)
        
        
    def pipeline_entry(self, input_data):
        if not isinstance(input_data, PaperData) or not input_data:
            raise TypeError(f"Task {self.__class__.__name__} requires input of type PaperData in pipe entry")
        paper = self.reference(input_data)

        return paper
    
    def reference(self, paper: PaperData = None):
        if paper:
            self.paper = paper
        used_keys = []
        # go section by section
        for section in self.paper.sections:
            ref_count = 0
            # best_for_section = self.vector_store.similarity_search(section.content, k=self.ref_per_section)
            
            # add according to the sentences
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", section.content)
            cited_sentences = []
            for i, sentence in enumerate(sentences):
                if ref_count >= self.max_per_section:
                    cited_sentences.extend(sentences[i:])
                    break

                if '\\' in sentence or '{' in sentence or '}' in sentence: # skip sentences with commands
                    cited_sentences.append(sentences[i])
                    continue
                
                #embedding = self.embed_model.embed_documents([sentence])[0]
                results, scores = zip(*self.vector_store.similarity_search_with_score(sentence, k=self.max_per_section))
                valid = {}
                for res, score in zip(results, scores):
                    if len(valid) >= self.max_per_sentence:
                        break
                    if score < self.confidence:
                        continue    

                    key = res.metadata["citation_key"]
                    valid[key] = res.page_content
                    if key not in used_keys:
                        used_keys.append(key)
                    ref_count += 1
                        
                if valid:
                    cite_command = f"\\cite{{{', '.join(list(valid.keys()))}}}"

                    if sentence.endswith(('.', ',', ';', '!', '?')):
                        sentence = sentence[:-1] + ' ' + cite_command + sentence[-1]
                    else:
                        sentence += f" {cite_command}"
        
                cited_sentences.append(sentence)
        
            section.content = " ".join(cited_sentences)
            named_log(self, f"Added {ref_count} references to section {section.title}")
        
        if self.save_usedbib_path:
            self._dump_used_bib(used_keys, self.bibdb_path, self.save_usedbib_path)
        
        return paper
    
    def _dump_used_bib(self, used_keys: list, bibdb_path: str, save_path: str):
        with open(bibdb_path, "r", encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)
        
        used_entries = [entry for entry in bib_db.entries if entry["ID"] in used_keys]
        
        used_db = bibtexparser.bibdatabase.BibDatabase()
        used_db.entries = used_entries
        
        with open(save_path, "w") as f:
            bibtexparser.dump(used_db, f)
        
    
    def _faiss_from_bibdb(self, embed_model, bibdb_path: str, chunks: int = 1200):
        bib_chunks = self._load_bibtex_to_docs(bibdb_path)
        splitter = CharacterTextSplitter(chunk_size=chunks, chunk_overlap=0)
        bib_docs = splitter.split_documents(bib_chunks)
        return FAISS.from_documents(bib_docs, embed_model)
    
    def _load_bibtex_to_docs(self, bib_path: str) -> List[Document]:
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_db = bibtexparser.load(f)
        
        entries = []
        for entry in bib_db.entries:
            title = entry.get("title", "No title")
            abstract = entry.get("abstract", "")
            keywords = entry.get("keywords", "")
            authors = entry.get("author", "")
            
            content = content = f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}\nAuthors: {authors}"
            entries.append(Document(page_content=content, metadata={"citation_key": entry.get("ID", "Unknown")}))
            
        return entries
    
    def divide_subtasks(self, n, input_data: PaperData = None):
        if input_data:
            self.paper = input_data
        sub = [None] * n
        n_sections = len(self.paper.sections)
        per_task = n_sections // n
        for i in range(0, n, per_task):
            paper = PaperData(subject=self.paper.subject, sections=self.paper.sections[i:i+per_task].copy(), title=self.paper.title, bib=self.paper.bib)
            sub[i] = PaperFAISSReferencer(self.embed_model, self.bibdb_path, paper, self.local_faissdb_path, self.save_usedbib_path, self.save_faiss_path, self.max_per_section, self.max_per_sentence, self.confidence)
        return sub

    def merge_subtasks_data(self, data: List[PaperData]):
        merged_paper = PaperData(data[0].subject, sections=data[0].sections, title=data[0].title, bib=data[0].bib)
        for paper in data[1:]:
            merged_paper.sections.extend(paper.sections)
        return merged_paper
