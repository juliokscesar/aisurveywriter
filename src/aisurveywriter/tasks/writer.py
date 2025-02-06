from typing import List, Union, Optional
from time import time
import re

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage


from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.utils import named_log, countdown_log, diff_keys, time_func

class PaperWriter(PipelineTask):
    """
    A class abstraction for the process of writing a survey paper
    """
    def __init__(self, llm: LLMHandler, prompt: str, paper: Optional[PaperData] = None, ref_paths: Optional[List[str]] = None, request_cooldown_sec: int = 60 * 1.4, summarize_refs=False, use_faiss=False, faiss_embeddings: str = "google"):
        """
        Intializes a PaperWriter
        
        Parameters:
            llm (LLMHandler): the LLMHandler object with the LLM model to use
            
            prompt (str): the prompt to give specific instructions for the LLM to write this paper. The prompt must have the placeholders: {subject}, {title}, {description}
            
            paper (Optional[PaperData]): a PaperData object containing the paper information. Each section in paper.section must have the 'title' and 'description' filled, and 'content' will be filled by this task.
             if this is None, a PaperData must be provided when calling write()
            
            ref_paths (List[str]): a List of the path of every PDF reference for this paper.
            
            request_cooldown_sec (int): cooldown time in seconds between two requests to the LLM API.
            summarize_refs (bool): summarize references before sending to the LLM (using the LLM itself to first summarize it)
            use_faiss (bool): use FAISS vector store to retrieve "similar" information from the references
            faiss_embeddings (str): vendor of embedding (google, openai). Will only have an effect if 'use_faiss' is True.
        """
        self.llm = llm
        self.paper = paper
        self.prompt = prompt
        self.ref_paths = ref_paths.copy()
        
        self._cooldown_sec = int(request_cooldown_sec)
        self._summarize = summarize_refs
        self._use_faiss = use_faiss
        self._faiss_embeddings = faiss_embeddings
        
    def pipeline_entry(self, input_data: Union[PaperData, dict]) -> PaperData:
        """
        This is the entry point in the pipeline. 
        The input of this task should be either a PaperData with a "subject" set, and the "title" and "description" fields for every SectionData set;
        or it can be a dictionary with the format: {"subject": str, "sections": [{"title": str, "description": str}, {"title": str, "description": str}, ...]}

        Parameters:
            input_data (Union[PaperData, dict]): the input from the previous task (or from the first call).
        """
        if isinstance(input_data, dict):
            if diff := diff_keys(list(PaperData.__dict__.keys()), input_data):
                raise TypeError(f"Missing keys for input data (task {PaperWriter.__class__.__name__}): {", ".join(diff)}")
            
            paper = PaperData(
                subject=input_data["subject"],
                sections=[SectionData(title=s["title"], description=s["description"]) for s in input_data["sections"]],
            )
        else:
            paper = input_data
        
        paper = self.write(paper)
        return paper
    
    def write(self, paper: Optional[PaperData] = None, prompt: Optional[str] = None):
        """
        Write the provided paper.
        
        Parameters:
            paper (Optional[PaperData]): a PaperData object containing "subject" and each section "title" and "description". If this is None, try to use the one that was set in the constructor.
            
            prompt (Optional[str]): the prompt to invoke langchain's chain. If this is None, try to use the one set in the constructor. The prompt must have the placeholders "{subject}", "{title}" and "{description}"
        """
        # use parameters if provided
        if paper:
            self.paper = paper
        if self.paper is None:
            raise RuntimeError("The PaperData (PaperWriter.paper) must be set to write")
        if prompt:
            self.prompt = prompt
        
        # read reference content and initialize llm chain
        sysmsg = SystemMessage(content=self._get_ref_content(self._summarize, self._use_faiss, self._faiss_embeddings))
        self.llm.init_chain(sysmsg, self.prompt)
        
        sz = len(self.paper.sections)
        word_count = 0
        
        # write section by section
        for i, section in enumerate(self.paper.sections):
            named_log(self, f"==> begin writing section ({i+1}/{sz}): {section.title}")
            elapsed, response = time_func(self.llm.invoke, {
                "subject": self.paper.subject,
                "title": section.title,
                "description": section.description,
            })
            section.content = response.content
            section.content = re.sub(r"[`]+[\w]*", "", section.content) # remove markdown code blocks if any
            word_count += len(section.content.split())
            named_log(self, f"==> finished writing section ({i+1}/{sz}): {section.title} | total words count: {word_count} | time elapsed: {int(elapsed)} s")

            try:
                named_log(self, f"==> response metadata:", response.usage_metadata)
            except:
                named_log(self, f"==> (debug) reponse object:", response)

            if self._cooldown_sec:
                cooldown = max(0, self._cooldown_sec - elapsed)
                named_log(self, f"==> initiating cooldown of {cooldown} s (request limitations)")
                countdown_log("", cooldown)

        return self.paper


    def _get_ref_content(self, summarize=False, use_faiss=False, faiss_embeddings: str ="google") -> Union[str,None]:
        """
        Returns the content of all PDF references in a single string
        If the references weren't set in the constructor, return None

        Parameters:
            summarize (bool): summarize the content of every reference, instead of using the whole PDF.
            
            use_faiss (bool): use FAISS to create a vector store and retrieve only a part of the information using text embedding.
            faiss_embeddings (str): vendor of embedding (google, openai). Will only have an effect if 'use_faiss' is True.
        """
        pdfs = PDFProcessor(self.ref_paths)
        
        if summarize:
            content = pdfs.summarize_content(self.llm.llm, show_metadata=True)
        elif use_faiss:
            match faiss_embeddings.strip().lower():
                case "openai":
                    embed = OpenAIEmbeddings()
                case "google":
                    embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                case _:
                    raise ValueError(f"Invalid embeddings vendor {faiss_embeddings!r}")
            vec = pdfs.vector_store(embed)
            relevant = vec.similarity_search(f"Get useful, techinal, and analytical information on the subject {self.paper.subject}")
            content = "\n".join([doc.page_content for doc in relevant])
        else:
            content = "\n".join(pdfs.extract_content())
        
        return content