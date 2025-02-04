from typing import List, Union

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage

from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.core.file_handler import FileHandler
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils.helpers import countdown_print

class PaperWriter:
    def __init__(self, subject: str, sections_structure: List[dict[str,str]], llm: LLMHandler, pdf_refrences: List[str]):
        self.subject = subject
        self.sections_structure = sections_structure
        self.llm = llm
        self.is_llm_initialized = False
        self.pdf_references = pdf_refrences.copy()

    def init_llm_context(
        self,
        context_msg: SystemMessage,
        write_prompt: str,
    ):
        self.llm.init_chain(context_msg, write_prompt)
        self.is_llm_initialized = True
        
    def create_context_sysmsg(
        self,
        header_prompt: str,
        summarize_references = False,
        alternate_summarizer_llm: Union[None, LLMHandler] = None,
        use_faiss = False,
        faiss_embeddings: str = "google",
        show_metadata = False,
    ) -> SystemMessage:
        pdfs = PDFProcessor(self.pdf_references)
        
        context = header_prompt
        if summarize_references:
            if alternate_summarizer_llm is None:
                alternate_summarizer_llm = self.llm
            context += "\n\n" + pdfs.summarize_content(alternate_summarizer_llm.llm, show_metadata=show_metadata)
        elif use_faiss:
            embed = None
            match faiss_embeddings.strip().lower():
                case "openai":
                    embed = OpenAIEmbeddings()
                case "google":
                    embed = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                case _:
                    raise ValueError(f"Invalid embeddings {faiss_embeddings!r}. Possible options are: openai, google")
            vec = pdfs.vector_store(embed)
            relevant_docs = vec.similarity_search(f"Retrieve all relevant information from these sources on the subject {self.subject}")
            context += "\n"
            context += "\n".join([doc.page_content for doc in relevant_docs])
        else:
            contents = pdfs.extract_content()
            context += "\n\n The content of the given references are:\n"
            context += "\n".join(contents)
        
        return SystemMessage(content=context)

    
    def _print(self, *msgs):
        print(f"({self.__class__.__name__})", *msgs)

    # TODO!!!
    def filter_section(self, content: str) -> str:
        pass

    def write_section(self, subject: str, title: str, description: str):
        """
        Write a specific section given the paper subject, and the section's title and description.
        """
        airesponse = self.llm.invoke({
            "subject": subject,
            "title": title,
            "description": description,
        })
        return airesponse

    def write_all_sections(self, show_metadata = False, sleep_between: int = 60) -> List[dict[str,str]]:
        if not self.is_llm_initialized:
            raise RuntimeError("LLM context was not initialized")
    
        sections_content = []
        for i, section in enumerate(self.sections_structure):
            self._print(f"===> STARTED WRITING SECTION ({i+1}/{len(self.sections_structure)}): {section['title']!r}")
            response = self.write_section(
                subject=self.subject,
                title=section["title"],
                description=section["description"],
            )
            self._print(f"===> FINISHED WRITING SECTION ({i+1}/{len(self.sections_structure)}): {section['title']!r}")
            if show_metadata:
                self._print("===> RESPONSE METADATA:", response.usage_metadata)

            sections_content.append({
                "title": section["title"],
                "content": response.content,
            })

            self._print("Initiating cooldown because of request limitations...")
            countdown_print("Countdown:", int(sleep_between))
        
        return sections_content