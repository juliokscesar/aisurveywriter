from typing import List
import re

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.agent_rags import RAGType, GeneralTextData
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.tasks import PipelineTask
from aisurveywriter.utils.logger import named_log, metadata_log, cooldown_log
from aisurveywriter.utils.helpers import time_func

class HumanReview(PipelineTask):
    def __init__(self, agent_ctx: AgentContext, paper: PaperData):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = paper
        
        self._apply_system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.apply_review_section.text)
        self._apply_human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n{refcontent}\n\n[end: references_content]\n\n"+
                                                                     "[begin: review_directives]\n\n{review_directives}\n\n[end: review_directives]\n\n"+
                                                                     "- Apply the review directives to the following section:\nSection title: {title}\nSection content:\n{content}")
        
    def run(self) -> PaperData:
        self.agent_ctx.llm_handler.init_chain_messages(self._apply_system, self._apply_human)
        self._cmd_listener()
        
        return self.agent_ctx._working_paper
        
    def _cmd_listener(self):
        cmd = ""
        while "/exit" not in cmd:
            sec_num = int(input(f"Enter section number (1-{len(self.agent_ctx._working_paper.sections)}):")) - 1
            assert sec_num <= len(self.agent_ctx._working_paper.sections)
            section = self.agent_ctx._working_paper.sections[sec_num]
            
            directives = []
            user_review = input(f"What would you like to change in section {sec_num}, {section.title}?\n> ")
            while "/exit" not in user_review:
                directives.append(user_review.strip())
                user_review = input("> ")
                
            print("Sending your directives to LLM...")

            elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
                "refcontent": self._get_reference_content(section),
                "subject": self.agent_ctx._working_paper.subject,
                "review_directives": "\n".join(directives),
                "title": section.title,
                "content": section.content,
            })
            named_log(self, "got response from LLM")
            metadata_log(self, elapsed, response)
            
            if self.agent_ctx.llm_cooldown:
                cooldown_log(self, self.agent_ctx.llm_cooldown)
            
            print("Changes applied to the paper")
            self.agent_ctx._working_paper.sections[sec_num].content = re.sub(r"[`]+[\w]*", "", response.content)
            
            cmd = input("Write '/exit' to exit or anything else to review another section...\n> ").strip().lower()

    
    def _get_reference_content(self, section: SectionData):
        # use full content if content RAG is disabled
        if self.agent_ctx.rags.is_disabled(RAGType.GeneralText):
            return "\n\n".join(self.agent_ctx.references.full_content())
        
        # retrieve relevant blocks for this section from content RAG
        k = 23
        query = f"Retrieve contextual, technical, and analytical information on the subject {self.agent_ctx._working_paper.subject} for a section titled \"{section.title}\", description:\n{section.description}"
        relevant: List[GeneralTextData] = self.agent_ctx.rags.retrieve(RAGType.GeneralText, query, k)
        return "\n\n".join([data.text for data in relevant])
    