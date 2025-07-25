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


EXTRA_REVIEW_DIRECTIVES: List[str] = [
    #r"""The paper title includes Langmuir and LB films, but there is some confusion in the use of the terms. In many cases, the term “LB” is used when it should be used “Langmuir”. For example, in the sentence “The length of the hydrocarbon chain and the presence of any unsaturation (double or triple bonds) significantly influence the packing behavior and stability of the resulting LB film.” The term which should have been used is Langmuir (not LB). This also happens in other parts of the paper, as in Section 2.5. Indeed, Section 2.5 is entitled “Characterization techniques for LB films”, but it is mostly associated with Langmuir monolayers. Moreover, the main application of Langmuir monolayers – which today is in mimicking cell membranes – is almost entirely ignored. When some mentions are made, the applications are attributed (erroneously in most cases) to LB films. Section 5.6 entitled “Biomimetic Films” is mostly associated with Langmuir monolayers, but the films are normally referred to as LB films, which is not correct"""
    #r"""There is some confusion in the use of the terms “Langmuir films” and “LB films”, as in some cases use is made of LB where it should be Langmuir. For instance, in Section 5.3 (Biomimetic Films), examples are given of applications of Langmuir monolayers, but referred to as LB films. Also, in Section 6.5 the use of molecular dynamics is mentioned with regard to LB films, but it should be Langmuir films."""
    # r"""The authors presented a long list of characterization methods for Langmuir monolayers, but none for LB films.b)The importance of Langmuir monolayers for mimicking cell membranes was mentioned, but no discussion or example was presented, except in the section of challenges. c)The emphasis on nanoarchitectonics seems excessive, which also brought some repetition of similar contents."""
    # r"""The survey must include discussions about Langmuir and Langmuir-Blodgett (LB) films. Make sure it isn't focused only on one and there must be enough information about both."""
]


class PaperReviewer(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, written_paper: PaperData):
        super().__init__(no_divide=False, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = written_paper
        
        self._review_system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.review_section.text)
        self._review_human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n{refcontent}\n\n[end: references_content]\n\n"+
                                                                      "[begin: survey_info]\n\nALL_SECTIONS: {paper_sections}\n\nREVIEWED: {paper_reviewed_sections}\n\n[end: survey_info]\n\n"+
                                                                      "Review the following section:\nSection title: {title}\nSection content:\n{content}")

        self._apply_system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.apply_review_section.text)
        self._apply_human = HumanMessagePromptTemplate.from_template("[begin: references_content]\n\n{refcontent}\n\n[end: references_content]\n\n"+
                                                                     "[begin: review_directives]\n\n{review_directives}\n\n[end: review_directives]\n\n"+
                                                                      "[begin: survey_info]\n\nALL_SECTIONS: {paper_sections}\n\nREVIEWED: {paper_reviewed_sections}\n\n[end: survey_info]\n\n"+
                                                                     "- Apply the review directives to the following section:\nSection title: {title}\nSection content:\n{content}")
        
        global EXTRA_REVIEW_DIRECTIVES
        self._extra_review_directives: List[str] = EXTRA_REVIEW_DIRECTIVES if EXTRA_REVIEW_DIRECTIVES else None
    
    def review(self) -> PaperData:
        section_amount = len(self.agent_ctx._working_paper.sections)
        all_sections = [f"{i+1}. {s.title}" for i, s in enumerate(self.agent_ctx._working_paper.sections)]
        reviewed_sections = []
        
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
                "paper_sections": "; ".join(all_sections),
                "paper_reviewed_sections": "; ".join(reviewed_sections),
                "title": section.title,
                "content": section.content,
            })
            
            named_log(self, f"==> review points gathered (word count: {len(response.content.split())})")
            metadata_log(self, elapsed, response)
            if self.agent_ctx.llm_cooldown:
                cooldown_log(self, self.agent_ctx.llm_cooldown)
                
            named_log(self, f"==> sending directives for LLM to apply")
            
            review_directives = response.content
            if self._extra_review_directives:
                review_directives += "\n\n" + "\n".join(self._extra_review_directives)
            
            self.agent_ctx.llm_handler.init_chain_messages(self._apply_system, self._apply_human)
            elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
                "refcontent": section_reference,
                "subject": self.agent_ctx._working_paper.subject,
                "paper_sections": "; ".join(all_sections),
                "paper_reviewed_sections": "; ".join(reviewed_sections),
                "review_directives": review_directives,
                "title": section.title,
                "content": section.content,
            })
            section.content = re.sub(r"[`]+[\w]*", "", response.content)
            total_words += len(section.content.split())

            named_log(self, f"==> finish reviewing section ({i+1}/{section_amount}) | total word count: {total_words}")
            
            metadata_log(self, elapsed, response)
            if self.agent_ctx.llm_cooldown:
                cooldown_log(self, self.agent_ctx.llm_cooldown)

            reviewed_sections.append(f"{i+1}. {section.title}")
        
        return self.agent_ctx._working_paper
    
    def _get_reference_content(self, section: SectionData):
        # use full content if content RAG is disabled
        if self.agent_ctx.rags.is_disabled(RAGType.GeneralText):
            return "\n\n".join(self.agent_ctx.references.full_content())
        
        # retrieve relevant blocks for this section from content RAG
        k = 28
        query = f"Retrieve contextual, technical, and analytical information on the subject {self.agent_ctx._working_paper.subject} for a section titled \"{section.title}\", description:\n{section.description}"
        relevant: List[GeneralTextData] = self.agent_ctx.rags.retrieve(RAGType.GeneralText, query, k)
        return "\n\n".join([data.text for data in relevant])

    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        return self.review()
        