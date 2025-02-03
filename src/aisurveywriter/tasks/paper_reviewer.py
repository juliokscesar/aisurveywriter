from typing import List, Union
import os
import shutil

from langchain_core.messages import SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from aisurveywriter.core.chatbots import NotebookLMBot
from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.core.config_manager import ConfigManager
from aisurveywriter.core.pdf_processor import PDFProcessor
from aisurveywriter.core.file_handler import FileHandler
from aisurveywriter.utils.helpers import countdown_print

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
        prompt = self.config.review_nblm_prompt.replace("{paperfilename}", os.path.basename(self.config.out_tex_path)).replace("{title}", title).replace("{number}", str(number)).replace("{subject}", self.config.paper_subject)
        self.nblm.send_prompt(prompt, sleep_for=40)
        improv_points = self.nblm.get_last_response()
        improv_points = improv_points[improv_points.find("Clarity and Coherence"):]
        return improv_points

    # TODO!!!
    def filter_section(self, content: str) -> str:
        pass

    def apply_improvement_section(self, nblm_review: str, title: str, content: str, bib_content: str):
        """
        Use LLM to apply the improvements to the section 'title' based on the output of NotebookLM's review.
        """
        response = self.llm.invoke({
            "subject": self.config.paper_subject,
            "title": title,
            "sectionlatex": content,
            "sectionimprovement": nblm_review,
            "biblatex": bib_content,
        })
        return response

    def improve_all_sections(
        self, 
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

        # Add as source to NotebookLM. First have to generate .txt file
        try:
            tmp_tex = tex_file + ".txt"
            tmp_bib = bib_file + ".txt"
            shutil.copyfile(tex_file, tmp_tex)
            shutil.copyfile(bib_file, tmp_bib)
            self.nblm.append_sources([tmp_tex, tmp_bib], sleep_for=25)
            os.remove(tmp_tex)
            os.remove(tmp_bib)
        except:
            raise RuntimeError("Unable to add tex and bib file as sources to NotebookLM")


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
        
        return reviewed