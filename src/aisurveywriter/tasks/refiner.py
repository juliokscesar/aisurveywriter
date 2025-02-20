from typing import Optional, List
import re
import json

from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from .pipeline_task import PipelineTask
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log
from aisurveywriter.utils.helpers import assert_type, time_func

class PaperRefiner(PipelineTask):
    required_input_variables: List[str] = ["subject"]
    
    def __init__(self, agent_ctx: AgentContext, full_paper: PaperData):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        self.agent_ctx._working_paper = full_paper
        
        self._system = SystemMessagePromptTemplate.from_template(self.agent_ctx.prompts.abstract_and_title.text)
        self._human = HumanMessagePromptTemplate.from_template("Produce Title and Abstract for this LaTeX paper:\n\n{content}")
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)
        
    def refine(self) -> PaperData:
        self.agent_ctx.llm_handler.init_chain_messages(self._system, self._human)

        named_log(self, f"==> asking LLM to produce title and abstract")
        
        elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
            "subject": self.agent_ctx._working_paper.subject,
            "content": self.agent_ctx._working_paper.full_content(),
        })
        
        named_log(self, f"==> got title and abstract from LLM")
        metadata_log(self, elapsed, response)
        if self.agent_ctx.llm_cooldown:
            cooldown_log(self, self.agent_ctx.llm_cooldown)
        
        try:
            resp_json = re.search(r"```json\s*([\s\S]+?)\s*```|({[\s\S]+})", response.content.strip()).group()
            resp = resp_json[resp_json.find("{"):resp_json.rfind("}")+1]
            resp = json.loads(resp)
            self.agent_ctx._working_paper.sections.insert(0, SectionData(
                title="Abstract",
                description="paper abstract",
                content=f"\\begin{{abstract}}\n\n{resp['abstract']}\n\n\\end{{abstract}}"
            ))
            self.agent_ctx._working_paper.title = resp["title"]
        except:
            named_log(self, f"==> failed to parse JSON from LLM response. Adding abstract section as it is")
            self.agent_ctx._working_paper.sections.insert(0, SectionData(
                title="Abstract",
                description="paper abstract",
                content=response.content.strip(),
            ))
            self.agent_ctx._working_paper.title = "CHECK JSON OUTPUT"

        return self.agent_ctx._working_paper
    
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        return self.refine()
