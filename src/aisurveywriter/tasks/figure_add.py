import os
import shutil
import re
from typing import List
import copy
from pydantic import BaseModel

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from .pipeline_task import PipelineTask
from ..core.agent_context import AgentContext
from ..core.agent_rags import RAGType, ImageData
from ..core.paper import PaperData
from ..core.document import DocFigure
from ..utils.logger import named_log, cooldown_log, metadata_log
from ..utils.helpers import time_func, assert_type

class FigureAddInfo(BaseModel):
    add_after: str | None = None
    caption: str | None = None
    label: str | None = None

class FigureAddResponse(BaseModel):
    figures: List[FigureAddInfo]

class PaperFigureAdd(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, written_paper: PaperData, images_dir: str, confidence: float = 0.75, max_figures: int = 30, max_figures_per_section: int = 3):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = written_paper
        
        self.response_parser = PydanticOutputParser(pydantic_object=FigureAddResponse)

        self._system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.add_figures.text)
        self._human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n"+
                                                               "{refcontent}\n\n"+
                                                               "[end: references_content]\n\n"+
                                                               "**MAXIMUM AMOUNT OF FIGURES**:"+str(max_figures_per_section)+"\n\n"+
                                                               "[begin: section_content]\n"+
                                                               "**TITLE**: {title}\n"+
                                                               "**CONTENT**:\n{content}\n\n"+
                                                               "[end: section_content]")
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        self.confidence = confidence
        self.max_figures = max_figures
    
    
        self.caption_credits_pattern = re.compile(r"\.\s+(?:Adapted|Reprinted)\s+(with|from)", re.IGNORECASE)
        self.figure_caption_pattern_format = r"((?:fig\.|figure|figura)\.*\s*{}(.+?)\n+)"
        self.figure_prefix_pattern = re.compile(r"(?:fig\.|figure|figura)\.*\s+(?:\d+)\.", re.IGNORECASE)
    
        self.images_dir = os.path.abspath(images_dir)
        self.used_imgs_dest = os.path.join(self.agent_ctx.output_dir, "used-imgs")
        os.makedirs(self.used_imgs_dest, exist_ok=True)
        self.agent_ctx._working_paper.fig_path = self.used_imgs_dest
        
    def add_figures(self):
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        used_figures: tuple[str, List[DocFigure]] = [] # label in text, docfigure object
        
        section_amount = len(self.agent_ctx._working_paper.sections)
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            named_log(self, f"start adding figures in section ({i+1}/{section_amount}): \"{section.title}\"")

            # dont add figures in Conclusion
            if i == section_amount-1:
                named_log(self, f"skipping conclusions section")
                break
            
            # first, get reference content with used figures removed
            refcontent = self._refremove_used_figures(used_figures)
            
            add_figures = self._llm_add_figures(refcontent, section.title, section.content, used_figures)
            content = self._include_figures(section.content, add_figures, used_figures, retrieve_k=5)
            
            named_log(self, f"finish adding figures in section ({i+1}/{section_amount}), total {len(used_figures)} figures")
            
            section.content = re.sub(r"[`]+[\w]*", "", content)
        
        self.agent_ctx._working_paper.fig_path = self.used_imgs_dest
        return self.agent_ctx._working_paper

    def _refremove_used_figures(self, used_figures: tuple[str, List[DocFigure]]) -> str:
        """
        Remove used figures from references to ensure LLM won't use them again
        """
        if not used_figures:
            return self.agent_ctx.references.full_content()
        
        docs_contents = copy.deepcopy(self.agent_ctx.references.docs_contents())
        for label, figure in used_figures:
            # identify in the text by using the figure document id (e.g. Fig. 1, Figure 2, ...)
            doc_figureid = figure.id
            fig_pattern = re.compile(self.figure_caption_pattern_format.format(doc_figureid), re.IGNORECASE)
            
            # remove all occurences of Figure X. if figure X is used
            doc_idx = self.agent_ctx.references.doc_index(figure.source_path)
            docs_contents[doc_idx] = fig_pattern.sub("", docs_contents[doc_idx])

        return "\n\n".join(docs_contents)
    
    def _llm_add_figures(self, refcontent: str, section_title: str, section_content: str, used_figures: List[DocFigure]) -> FigureAddResponse:
        elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
            "refcontent": refcontent,
            "subject": self.agent_ctx._working_paper.subject,
            "title": section_title,
            "content": section_content,
        })

        # remove markdown code block tag
        response.content = re.sub(r"`+\w*", "", response.content)
        
        try:
            figures: FigureAddResponse = self.response_parser.invoke(response)
            named_log(self, f"llm provided {len(figures.figures)} figures to add in section {section_title!r}")
        except Exception as e:
            named_log(self, f"unable to parse figures response: {e}")
            figures = FigureAddResponse(figures=[])
        
        metadata_log(self, elapsed, response)
        if self.agent_ctx.llm_cooldown:
            cooldown_log(self, self.agent_ctx.llm_cooldown)
        
        return figures

    def _include_figures(self, section_content: str, response_figures: FigureAddResponse, used_figures: List[DocFigure], retrieve_k: int = 5) -> str:
        assert self.agent_ctx.rags.is_enabled(RAGType.ImageData)
        
        figure_block_fmt = r"""
        \begin{{figure}}[h!]
        \includegraphics[width=0.9\textwidth]{{{}}}
        \caption{{{}}}
        \label{{{}}}
        \end{{figure}}
        """
        
        used_imgs = set([os.path.basename(fig.image_path) for _, fig in used_figures])
        content_altered = section_content
        for add_figure in response_figures.figures:
            # find figure by matching caption
            results = self.agent_ctx.rags.retrieve(RAGType.ImageData, add_figure.caption, k=retrieve_k)
            if not results:
                named_log(self, "unable to match a figure with caption:", add_figure.caption)
                continue
            results = [result for result in results if result.basename not in used_imgs]
            if not results:
                named_log(self, "unable to match an unused figure for caption:", add_figure.caption)
                continue
            used_imgs.add(results[0].basename)
            fig_result: DocFigure = results[0].to_doc_figure(self.agent_ctx.references)
            
            # copy image to destination path
            try:
                used_image_path = os.path.join(self.used_imgs_dest, results[0].basename)
                shutil.copy(fig_result.image_path, used_image_path)
            except Exception as e:
                used_image_path = fig_result.image_path
                named_log(self, f"couldn't copy {used_image_path} to save directory: {e}")

            # preprocess caption
            caption = add_figure.caption
            # first cut existing credits if any
            credits_match = self.caption_credits_pattern.search(caption)
            if credits_match:
                caption = caption[:credits_match.start()].strip()
            # cut "Fig..." and number prefix if the llm included it
            prefix_match = self.figure_prefix_pattern.match(caption)
            if prefix_match:
                caption = caption[prefix_match.end():].strip()
                
            # try add credits to the end of caption
            doc_source = self.agent_ctx.references.doc_from_path(fig_result.source_path)
            if doc_source:
                year = doc_source.bibtex_entry.get("year", None) if doc_source.bibtex_entry else None
                credits_fmt = f"Adapted from {{}}, {year}" if year else "Adapted from {}"
                if doc_source.author:
                    authors = [author.strip() for author in doc_source.author.split("and")]
                    if len(authors) > 1:
                        authors_caption = f"{authors[0].strip()}, et al."
                    else:
                        authors_caption = authors[0].strip()
                    caption = caption.strip() + " " + credits_fmt.format(authors_caption)
                    caption = caption.replace("\n", " ").strip()
                    if doc_source.bibtex_entry:
                        caption += f" \\cite{{{doc_source.bibtex_entry["ID"]}}}."
                    else:
                        caption += "."
            
            # add an index to label to make sure it is unique
            add_figure.label += str(len(used_figures))
            
            # format latex figure block
            figure_block = figure_block_fmt.format(os.path.basename(fig_result.image_path),
                                                   caption, add_figure.label)
        
            # find place to add figure or add at the end if no match
            figure_place = content_altered.lower().find(add_figure.add_after.lower())
            if figure_place == -1:
                figure_place = len(content_altered)
        
            # add figure block to section
            content_altered = content_altered[:figure_place].strip() + "\n" + figure_block + "\n" + \
                            (content_altered[figure_place:].strip() if figure_place < len(content_altered) else "")

            # register used figure (label, DocFigure)
            used_figures.append((add_figure.label, fig_result))
        
        return content_altered
 
            
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data

        return self.add_figures()
    