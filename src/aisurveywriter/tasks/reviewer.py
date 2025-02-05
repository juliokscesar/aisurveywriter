from typing import List, Union, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage


from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.utils import named_log, countdown_log, diff_keys

class PaperReviewer(PipelineTask):
    """
    A class abstraction for the process of reviewing a generated survey paper
    """
    def __init__(self, llm: LLMHandler, review_prompt: str, apply_prompt: str, paper: Optional[PaperData] = None, ref_paths: Optional[List[str]] = None, request_cooldown_sec: int = 60 * 1.5, summarize_refs=False, use_faiss=False, faiss_embeddings: str = "google"):
        """
        Intializes a PaperWriter
        
        Parameters:
            llm (LLMHandler): the LLMHandler object with the LLM model to use
            
            prompt (str): the prompt to give specific instructions for the LLM to write this paper. The prompt must have the placeholders: {subject}, {title}, {description}
            
            paper (PaperData): a PaperData object containing the paper information. Each section in paper.section must have all fields filled.
            
            ref_paths (List[str]): a List of the path of every PDF reference for this paper.
            
            request_cooldown_sec (int): cooldown time in seconds between two requests to the LLM API.
            summarize_refs (bool): summarize references before sending to the LLM (using the LLM itself to first summarize it)
            use_faiss (bool): use FAISS vector store to retrieve "similar" information from the references
            faiss_embeddings (str): vendor of embedding (google, openai). Will only have an effect if 'use_faiss' is True.
        """
        self.llm = llm
        self.paper = paper
        self.review_prompt = review_prompt
        self.apply_prompt = apply_prompt
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
                raise TypeError(f"Missing keys for input data in pipe entry (task {self.__class__.__name__}): {", ".join(diff)}")
            
            paper = PaperData(
                subject=input_data["subject"],
                sections=[SectionData(title=s["title"], description=s["description"], content=s["content"]) for s in input_data["sections"]],
            )
        else:
            paper = input_data
        
        paper = self.review(paper)
        return paper
    
    def review(self, paper: Optional[PaperData] = None, prompt: Optional[str] = None):
        """
        Review the provided paper.
        
        Parameters:
            paper (Optional[PaperData]): a PaperData object containing "subject" and each section "title" and "description". If this is None, try to use the one that was set in the constructor.
            
            prompt (Optional[str]): the prompt to invoke langchain's chain. If this is None, try to use the one set in the constructor. The prompt must have the placeholders "{subject}", "{title}" and "{description}"
        """
        # use parameters if provided
        if paper:
            self.paper = paper
        if self.paper is None:
            raise RuntimeError("The PaperData (PaperReviewer.paper) must be set to review")
        if prompt:
            self.prompt = prompt
        
        # read reference content and initialize llm chain
        refmsg = SystemMessage(content=self._get_ref_content(self._summarize, self._use_faiss, self._faiss_embeddings))
        
        sz = len(self.paper.sections)
        
        # review section by section
        for i, section in enumerate(self.paper.sections):
            named_log(self, f"==> begin reviewing section ({i+1}/{sz}): {section.title}")

            # first get review from llm
            self.llm.init_chain(refmsg, self.review_prompt)
            response = self.llm.invoke({
                "subject": self.paper.subject,
                "title": section.title,
                "content": section.content,
            })
            
            # now apply the review points
            response = self.llm.invoke({
                "subject": self.paper.subject,
                "title": section.title,
                "content": section.content,
                "review_points": response.content,
            })

            section.content = response.content
            word_count += len(section.content.split())
            named_log(self, f"==> finished reviewing section ({i+1}/{sz}): {section.title} | total words count: {word_count}")
            named_log(self, f"==> response metadata:", response.usage_metadata)

            named_log(self, "==> initiating cooldown (request limitations)")
            countdown_log("", self._cooldown_sec)

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
        pdfs = PDFProcessor(self.pdf_refs)
        
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
            content = "\n".joing(pdfs.extract_content())
        
        return content