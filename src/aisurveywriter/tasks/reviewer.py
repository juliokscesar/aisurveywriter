from typing import List, Union, Optional
from time import time
import re

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import RAGType, GeneralTextData
from aisurveywriter.tasks.pipeline_task import PipelineTask
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log
from aisurveywriter.utils.helpers import time_func, assert_type

class PaperReviewer(PipelineTask):
    required_input_variables: List[str] = ["refcontent", "subject"]
    
    def __init__(self, agent_ctx: AgentContext, written_paper: PaperData):
        super().__init__(no_divide=False, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = written_paper
        
        self._review_system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.review_section.text)
        self._review_human = HumanMessagePromptTemplate.from_template("Review the following section:\nSection title: {title}\nSection content:\n{content}")

        self._apply_system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.apply_review_section.text)
        self._apply_human = HumanMessagePromptTemplate.from_template("[begin: review_directives]\n\n{review_directives}\n\n[end: review_directives]\n\n- Apply the review directives to the following section:\nSection title: {title}\nSection content:\n{content}")
    
    def review(self) -> PaperData:
        section_amount = len(self.agent_ctx._working_paper.sections)
        total_words = 0
        for i, section in enumerate(self.agent_ctx._working_paper.sections):
            assert(section.content is not None)
            section_reference = self._get_reference_content(section)
            
            named_log(self, f"==> start reviewing section ({i+1}/{section_amount}): \"{section.title}\"")
            named_log(self, f"==> getting review points from LLM")
    
            self.agent_ctx.llm_handler.init_chain_messages(self._review_system, self._review_human)
            elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
                "refcontent": section_reference,
                "subject": self.agent_ctx._working_paper.subject,
                "title": section.title,
                "content": section.content,
            })
            
            named_log(self, f"==> review points gathered (word count: {len(response.content.split())})")
            metadata_log(self, elapsed, response)
            if self.agent_ctx.llm_cooldown:
                cooldown_log(self, self.agent_ctx.llm_cooldown)
                
            named_log(self, f"==> sending directives for LLM to apply")
            
            self.agent_ctx.llm_handler.init_chain_messages(self._apply_system, self._apply_human)
            elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
                "refcontent": section_reference,
                "subject": self.agent_ctx._working_paper.subject,
                "review_directives": response.content,
                "title": section.title,
                "content": section.content,
            })
            section.content = re.sub(r"[`]+[\w]*", "", response.content)
            total_words += len(section.content.split())

            named_log(self, f"==> finish reviewing section ({i+1}/{section_amount}) | total word count: {total_words}")
            
            metadata_log(self, elapsed, response)
            if self.agent_ctx.llm_cooldown:
                cooldown_log(self, self.agent_ctx.llm_cooldown)
        
        return self.agent_ctx._working_paper
    
    def _get_reference_content(self, section: SectionData):
        # use full content if content RAG is disabled
        if self.agent_ctx.rags.is_disabled(RAGType.GeneralText):
            return "\n\n".join(self.agent_ctx.references.full_content())
        
        # retrieve relevant blocks for this section from content RAG
        k = 25
        query = f"Retrieve contextual, technical, and analytical information on the subject {self.agent_ctx._working_paper.subject} for a section titled \"{section.title}\", description:\n{section.description}"
        relevant: List[GeneralTextData] = self.agent_ctx.rags.retrieve(RAGType.GeneralText, query, k)
        return "\n\n".join([data.text for data in relevant])

    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        return self.review()
        