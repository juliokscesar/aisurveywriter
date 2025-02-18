from typing import List, Union, Optional
from time import time
import re

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.utils import named_log, countdown_log, diff_keys, time_func

class PaperReviewer(PipelineTask):
    """
    A class abstraction for the process of reviewing a generated survey paper
    """
    def __init__(self, llm: LLMHandler, review_prompt: str, apply_prompt: str, paper: Optional[PaperData] = None, ref_paths: Optional[List[str]] = None, request_cooldown_sec: int = 60 * 1.5, discard_ref_sections=True, summarize_refs=False, use_faiss=False, embeddings=None):
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
        self._discard_ref_sections = discard_ref_sections
        self._summarize = summarize_refs
        self._use_faiss = use_faiss
        self._embed = embeddings
        
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
        if not self._use_faiss:
            refcontent = self._get_ref_content(self._discard_ref_sections, self._summarize, self._use_faiss, faiss_k=8)
        review_sys = SystemMessagePromptTemplate.from_template(self.review_prompt)
        review_hum = HumanMessagePromptTemplate.from_template("Section to review:\n- Title: {title}\n- Content:\n{content}")
        apply_sys = SystemMessagePromptTemplate.from_template(self.apply_prompt)
        apply_hum = HumanMessagePromptTemplate.from_template("- Directives for this section:\n{directives}\n\n- Section latex content:\n\n{content}")

        sz = len(self.paper.sections)
        word_count = 0
        # review section by section
        for i, section in enumerate(self.paper.sections):
            named_log(self, f"==> begin reviewing section ({i+1}/{sz}): {section.title}")

            named_log(self, f"==> asking LLM for review points")
            # first get review from llm
            self.llm.init_chain_messages(review_sys, review_hum)
            if self._use_faiss:
                refcontent = self._get_ref_content(self._discard_ref_sections, use_faiss=self._use_faiss, section=section, faiss_k=15)

            elapsed, response = time_func(self.llm.invoke, {
                "refcontents": refcontent,
                "subject": self.paper.subject,
                "title": section.title,
                "content": section.content,
            })
            named_log(self, f"==> got llm review points | time elapsed: {elapsed} s | metadata:", response.usage_metadata)
            
            named_log(self, f"==> cooldown sending reveiw points to llm ({self._cooldown_sec} s)")
            countdown_log("", self._cooldown_sec)
            
            # now apply the review points
            self.llm.init_chain_messages(apply_sys, apply_hum)
            elapsed, response = time_func(self.llm.invoke, {
                "refcontents": refcontent,
                "subject": self.paper.subject,
                "directives": response.content,
                "content": section.content,
                # "review_points": improv_points, testing with improve points as system message
            })

            section.content = re.sub(r"[`]+[\w]*", "", response.content) # remove markdown code blocks if any
            word_count += len(section.content.split())

            named_log(self, f"==> finished reviewing section ({i+1}/{sz}): {section.title} | total words count: {word_count} | time elapsed: {int(elapsed)} s")
            
            try:
                named_log(self, f"==> response metadata:", response.usage_metadata)
            except:
                named_log(self, f"==> (debug) reponse object:", response)

            if self._cooldown_sec:
                named_log(self, f"==> initiating cooldown of {self._cooldown_sec} s (request limitations)")
                countdown_log("", self._cooldown_sec)

        return self.paper


    def _get_ref_content(self, discard_ref_section=True, summarize=False, use_faiss=False, section: SectionData = None, faiss_k: int = 5) -> Union[str,None]:
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
            content = pdfs.summarize_content(self.llm.model, show_metadata=True)
        elif use_faiss:
            vec = pdfs.faiss(self._embed)
            relevant = vec.similarity_search(f"Retrieve contextual, techinal, and analytical information on the subject {self.paper.subject} for a section titled \"{section.title}\", description:\n{section.description}", k=faiss_k)
            content = "\n".join([doc.page_content for doc in relevant])
        else:
            if discard_ref_section:
                pdf_contents = pdfs.extract_content()
                content = ""
                for pdf_content in pdf_contents:
                    ref_match = re.search(r"(References|Bibliography|Works Cited)\s*[\n\r]+", pdf_content, re.IGNORECASE)
                    if ref_match:
                        content += pdf_content[:ref_match.start()].strip()
                    else:
                        content += pdf_content.strip()
                    content += "\n"
            else:
                content = "\n".join(pdfs.extract_content())
        
        return content

    def divide_subtasks(self, n: int, input_data: PaperData = None):
        self.paper = input_data
        sub = []
        n_sections = len(self.paper.sections)
        per_task = n_sections // n
        for i in range(0, n, per_task):
            subpaper = PaperData(self.paper.subject, self.paper.sections[i:i+per_task], self.paper.title, self.paper.bib_path)
            sub.append(PaperReviewer(self.llm, self.review_prompt, self.apply_prompt, subpaper, self.ref_paths, self._cooldown_sec, self._discard_ref_sections, self._summarize, self._use_faiss, self._faiss_embeddings))
        return sub    
    
    def merge_subtasks_data(self, data: List[PaperData]):
        merged_paper = PaperData(data[0].subject, data[0].sections, data[0].title, data[0].bib_path)
        for paper in data[1:]:
            merged_paper.sections.extend(paper.sections)
        return merged_paper