from typing import List, Union, Optional
import re

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import RAGType, GeneralTextData
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log
from aisurveywriter.utils.helpers import time_func, assert_type
        
class PaperWriter(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, structured_paper: PaperData):
        super().__init__(no_divide=False, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = structured_paper
    
        self._system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.write_section.text)
        self._human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n"+
                                                               "{refcontent}\n\n"+
                                                               "[end: references_content]\n\n"+
                                                               "[begin: survey_info]\n\n"+
                                                               "ALL_SECTIONS: {paper_sections}\n"+
                                                               "WRITTEN: {paper_written_sections}\n\n"+
                                                               "[end: survey_info]\n\n"+
                                                               "[begin: section]\n\n"+
                                                               "- Section title: {title}\n"+
                                                               "- Section description:\n{description}\n\n"+
                                                               "[end: section]")

        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)
    
    def write(self) -> PaperData:
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)
        
        section_amount = len(self.agent_ctx._working_paper.sections)
        all_sections = [f"{i+1}. {s.title}" for i, s in enumerate(self.agent_ctx._working_paper.sections)]
        written_sections = []
        
        total_words = 0
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            assert(section.description is not None)

            named_log(self, f"==> start writing content for section ({i+1}/{section_amount}): \"{section.title}\"")
            
            elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
                "refcontent": self._get_reference_content(section),
                "subject": self.agent_ctx._working_paper.subject,
                "paper_sections": "; ".join(all_sections),
                "paper_written_sections": "; ".join(written_sections),
                "title": section.title,
                "description": section.description,
            })
            
            section.content = re.sub(r"[`]+[\w]*", "", response.content)
            total_words += len(section.content.split())

            named_log(self, f"==> finished writing section ({i+1}/{section_amount}) | total word count: {total_words}")
            metadata_log(self, elapsed, response)
            if self.agent_ctx.llm_cooldown:
                cooldown_log(self, self.agent_ctx.llm_cooldown)

            written_sections.append(f"{i+1}. {section.title}")

        return self.agent_ctx._working_paper

    def _get_reference_content(self, section: SectionData):
        # use full content if content RAG is disabled
        if self.agent_ctx.rags.is_disabled(RAGType.GeneralText):
            return "\n\n".join(self.agent_ctx.references.full_content())
        
        # retrieve relevant blocks for this section from content RAG
        k = 30
        query = f"Retrieve contextual, technical, and analytical information on the subject {self.agent_ctx._working_paper.subject} for a section titled \"{section.title}\", description:\n{section.description}"
        relevant: List[GeneralTextData] = self.agent_ctx.rags.retrieve(RAGType.GeneralText, query, k)
        return "\n\n".join([data.text for data in relevant])

    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data

        return self.write()