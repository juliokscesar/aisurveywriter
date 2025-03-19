import os
import shutil
import re
from typing import List
import copy

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from .pipeline_task import PipelineTask
from ..core.agent_context import AgentContext
from ..core.agent_rags import RAGType, ImageData
from ..core.paper import PaperData
from ..core.document import DocFigure
from ..utils.logger import named_log, cooldown_log, metadata_log
from ..utils.helpers import time_func, assert_type

class PaperFigureAdd(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, written_paper: PaperData, images_dir: str, confidence: float = 0.75, max_figures: int = 30):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = written_paper
        
        self._system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.add_figures.text)
        self._human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n"+
                                                               "{refcontent}\n\n"+
                                                               "[end: references_content]\n\n"+
                                                            #    "[begin: used_figures]\n\n"+
                                                            #    "{used_figures}\n\n"+
                                                            #    "[end: used_figures]\n\n"+
                                                               "Add figures to this section:\n"+
                                                               "- Section title: {title}\n"+
                                                               "- Section content:\n{content}")
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        self.confidence = confidence
        self.max_figures = max_figures
    
        self.fig_pattern = re.compile(
            r"(\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\})"
            r"(.*?)"
            r"(\\caption\{([^}]+)\})"
            r"(.*?)"
            r"(\\label\{([^}]+)\})",
            re.DOTALL
        )
        self.caption_credits_pattern = re.compile(r"\.\s+(?:Adapted\s+from|Reprinted\s+with)", re.IGNORECASE)
        self.figure_caption_pattern_format = r"((?:fig\.|figure|figura)\.*\s*{}(.+?)\n+)"
    
        self.images_dir = os.path.abspath(images_dir)
        self.used_imgs_dest = os.path.join(self.agent_ctx.output_dir, "used-imgs")
        os.makedirs(self.used_imgs_dest, exist_ok=True)
        self.agent_ctx._working_paper.fig_path = self.used_imgs_dest
        
    def add_figures(self):
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        used_figures: tuple[str, List[DocFigure]] = [] # label in text, docfigure object
        
        section_amount = len(self.agent_ctx._working_paper.sections)
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            named_log(self, f"==> start adding figures in section ({i+1}/{section_amount}): \"{section.title}\"")
            
            # first, get reference content with used figures removed
            refcontent = self._refremove_used_figures(used_figures)
            
            # ask llm to add figures and replace them using RAG
            content = self._llm_add_figures(refcontent, section.title, section.content, used_figures)
            content = self._replace_figures(content, used_figures, retrieve_k=5)
            
            named_log(self, f"==> finish adding figures in section ({i+1}/{section_amount}), total {len(used_figures)} figures")
            
            section.content = re.sub(r"[`]+[\w]*", "", content)
        
        self.agent_ctx._working_paper.fig_path = self.used_imgs_dest
        return self.agent_ctx._working_paper

    def _refremove_used_figures(self, used_figures: tuple[str, List[DocFigure]]) -> str:
        if not used_figures:
            return self.agent_ctx.references.full_content()
        
        docs_contents = copy.deepcopy(self.agent_ctx.references.docs_contents())
        for label, figure in used_figures:
            doc_figureid = figure.id+1
            doc_idx = self.agent_ctx.references.paths.index(figure.source_path)
            
            fig_pattern = re.compile(self.figure_caption_pattern_format.format(doc_figureid), re.IGNORECASE)
            
            # remove all occurences of Figure X. if figure X is used
            docs_contents[doc_idx] = fig_pattern.sub("", docs_contents[doc_idx])

        return "\n\n".join(docs_contents)
    
    def _llm_add_figures(self, refcontent: str, title: str, content: str, used_figures: list) -> str:
        # used_figures_str = "\n".join([f"- FIG_LABEL: {fig[0]!r} | FIG_CAPTION: {fig[1].caption.replace("\n","")!r}" for fig in used_figures]) if used_figures else ""
        elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
            # "used_figures": used_figures_str,
            "refcontent": refcontent,
            "subject": self.agent_ctx._working_paper.subject,
            "title": title,
            "content": content,
        })
        named_log(self, "==> got section with figures from LLM")
        metadata_log(self, elapsed, response)
        if self.agent_ctx.llm_cooldown:
            cooldown_log(self, self.agent_ctx.llm_cooldown)
        
        return response.content

    def _replace_figures(self, content: str, used_figures: list, retrieve_k: int = 5):
        assert(not self.agent_ctx.rags.is_disabled(RAGType.ImageData))
            
        used_imgs = set([os.path.basename(fig.image_path) for _,fig in used_figures])
        # find figures added by llm
        content_replaced = content
        for figmatch in self.fig_pattern.finditer(content):
            if len(used_imgs) >= self.max_figures:
                break

            figname = figmatch.group(2)
            figcaption = figmatch.group(5)
            figlabel = figmatch.group(7)

            # cut "Adapted from..." from caption (don't include it as query)
            caption = figcaption
            credits_match = self.caption_credits_pattern.search(caption)
            if credits_match:
                caption = caption[:credits_match.start()].strip()

            # use caption to retrieve an image      
            query = caption
            results: List[ImageData] = self.agent_ctx.rags.retrieve(RAGType.ImageData, query, k=retrieve_k)
            if not results:
                named_log(self, f"unable to find images for figure: {figname!r}, caption: {caption}")
                if self.agent_ctx.embed_cooldown:
                    cooldown_log(self, self.agent_ctx.embed_cooldown)
                # remove this unused block
                content_replaced = content.replace(figmatch.group(0), 1)
                continue
            
            used_image_path = None
            used_doc_figure: DocFigure = None
            for result in results:
                if result.basename in used_imgs:
                    continue
                used_imgs.add(result.basename)
                
                # keep track of used figures and their label in the text
                used_doc_figure = result.to_doc_figure(self.agent_ctx.references)
                used_figures.append((figlabel, used_doc_figure))

                try:
                    used_image_path = os.path.join(self.used_imgs_dest, result.basename)
                    shutil.copy(os.path.join(self.images_dir, result.basename), used_image_path)
                except Exception as e:
                    used_image_path = result.basename
                    named_log(self, f"couldn't copy {used_image_path} to save directory: {e}")

                break # we only loop over the results to find one that wasn't used
            
            if not used_image_path:
                named_log(self, f"couldn't find a match for {figname}: {caption!r} that had confidence >= {self.confidence} or wasn't used before")
                # remove this unused block
                content_replaced = content.replace(figmatch.group(0), 1)
                continue
            else:
                named_log(self, f"best match for {figname}: {os.path.basename(used_image_path)}")
            
            # add "adapted from..." manually to caption
            new_caption = figcaption
            if doc := self.agent_ctx.references.doc_from_path(used_doc_figure.source_path):
                year = doc.bibtex_entry.get("year", None) if doc.bibtex_entry else None
                credits_fmt = f"Adapted from {{}}, {year}" if year else "Adapted from {}"
                if doc.author:
                    authors = [author.strip() for author in doc.author.split("and")]
                    if len(authors) > 1:
                        authors_caption = f"{authors[0].strip()}, et al."
                    else:
                        authors_caption = authors[0].strip() + "."
                    new_caption = caption.strip() + ". " + credits_fmt.format(authors_caption)
                    new_caption = new_caption.replace("\n", " ")
                        
            modified_figure = (
                re.sub(rf"\\includegraphics(\[[^\]]*\])?{{{re.escape(figname)}}}", 
                       rf"\\includegraphics[width=0.95\\textwidth]{{{os.path.basename(used_image_path)}}}", figmatch.group(1)) +  # replace image path
                figmatch.group(3) +  # preserve text between \includegraphics and \caption
                figmatch.group(4).replace(figcaption, new_caption) +  # replace caption
                figmatch.group(5) +  # preserve text between \caption and \label
                figmatch.group(6).replace(figlabel, figlabel)  # replace label
            )
            
            content_replaced = content.replace(figmatch.group(0), modified_figure, 1)
             
            if self.agent_ctx.embed_cooldown:
                cooldown_log(self, self.agent_ctx.embed_cooldown)
        
        return content_replaced
            
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data

        return self.add_figures()
    