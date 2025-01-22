from typing import List, Union
import os

from langchain_core.messages import SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from core.chatbots import NotebookLMBot
from core.llm_handler import LLMHandler
from core.config_manager import ConfigManager
from core.pdf_processor import PDFProcessor
from core.file_handler import FileHandler
from utils.helpers import countdown_print

class PaperReviewer:
    def __init__(self, subject: str, sections_content: List[dict[str,str]], pdf_references: List[str], nblm: NotebookLMBot, llm: LLMHandler, config: ConfigManager):
        self.subject = subject
        self.sections_content = sections_content
        self.pdf_references = pdf_references.copy()
        self.llm = llm
        self.nblm = nblm
        if not nblm.is_logged_in():
            raise AssertionError("NotebookLM bot passed to PaperReviewer must have already been initialized with the respective sources")

        self.config = config

    def _print(self, *msgs):
        print(f"({self.__class__.__name__})", *msgs)

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

    def nblm_review_section(self, title: str, number: int) -> str:
        """
        Ask NotebookLM to point what must be improved for the section 'title'
        """
        prompt = self.config.review_nblm_prompt.replace("{generatedpaperfile}", self.config.out_tex_path).replace("{title}", title).replace("{number}", number)
        self.nblm.send_prompt(prompt, sleep_for=40)
        improv_points = self.nblm.get_last_response()
        improv_points = improv_points[improv_points.find("Clarity and Coherence"):]
        return improv_points

    def apply_improvement_section(self, nblm_review: str, title: str, content: str, bib_content: str):
        """
        Use LLM to apply the improvements to the section 'title' based on the output of NotebookLM's review.
        """
        response = self.llm.write({
            "title": title,
            "sectionlatex": content,
            "sectionimprovement": nblm_review,
            "biblatex": bib_content,
        })
        return response

    def improve_all_sections(
        self, 
        save_latex = True,
        summarize_ref=False,
        use_faiss=False,
        faiss_embeddings: str = "google",
        show_metadata=True,
        sleep_between: int = 60,
    ) -> List[dict[str,str]]:
        # Check if TEX and BIB files actually exist
        tex_file = self.config.out_tex_path
        bib_file = tex_file.replace(".tex", ".bib")
        if not os.path.isfile(tex_file) or not os.path.isfile(bib_file):
            raise AssertionError("To run PaperReview, latex (.tex) and bibtex (.bib) files must exist in the path provided by the ConfigManager")
        with open(bib_file, "r", encoding="utf-8") as f:
            bib_content = f.read()

        # Add as source to NotebookLM
        self.nblm.append_sources([tex_file, bib_file], sleep_for=20)

        # First setup LLM context message
        ctx = self.create_context_sysmsg(
            header_prompt="The PDF content of the references is:",
            summarize_references=summarize_ref,
            use_faiss=use_faiss,
            faiss_embeddings=faiss_embeddings,
            show_metadata=show_metadata,
        )
        self.llm.init_chain(ctx, self.config.review_improve_prompt)

        reviewed = []
        for i, section in enumerate(self.sections_content):
            self._print(f"===> STARTED REVIEWING SECTION ({i+1}/{len(self.sections_content)}) {section['title']}")
            self._print("Asking NotebookLM for review points...")
            nblm_review = self.nblm_review_section(section["title"], i+1)
            
            self._print("Sending NotebookLM review to apply improvement with LLM...")
            response = self.apply_improvement_section(nblm_review, section["title"], section["content"], bib_content)

            reviewed.append({
                "title": section["title"],
                "content": response.content,
            })
            self._print(f"===> FINISHED REVIEWING SECTION {section['title']}")

            if show_metadata:
                self._print("===> RESPONSE METADATA:", response.usage_metadata, "\n")
            
            self._print("Initiating cooldown because of request limitations...")
            countdown_print("Countdown:", int(sleep_between))
        
        if save_latex:
            FileHandler.write_latex(
                template_path=self.config.tex_template_path,
                sections=reviewed,
                file_path=self.config.out_reviewed_tex_path,
            )
        return reviewed