from typing import List, Optional, Union
import yaml
import re
import json

from langchain_core.prompts.chat import HumanMessagePromptTemplate

from aisurveywriter.core.paper import PaperData, SectionData
from aisurveywriter.core.agent_context import AgentContext
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log
from aisurveywriter.utils.helpers import time_func, assert_type

from .pipeline_task import PipelineTask

class PaperStructureGenerator(PipelineTask):
    required_input_variables: List[str] = ["subject", "refcontent"]
    
    def __init__(self, agent_ctx: AgentContext, paper_with_subject: PaperData, save_json_path: Optional[str] = None):
        super().__init__(no_divide=True, agent_ctx=agent_ctx)
        
        self.agent_ctx._working_paper = paper_with_subject
        self.save_json_path = save_json_path
        
    def generate(self, save_json_path: Optional[str] = None) -> dict[str,List[dict[str,str]]]:
        named_log(self, f"==> start generating structure for paper on subject {self.agent_ctx._working_paper.subject!r}")
        
        self.agent_ctx.llm_handler.init_chain_messages(
            HumanMessagePromptTemplate.from_template(self.agent_ctx.prompts.generate_struct.text),
        )
        elapsed, response = time_func(self.agent_ctx.llm_handler.invoke, {
            "subject": self.agent_ctx._working_paper.subject,
            "refcontent": self.agent_ctx.references.full_content(discard_bibliography=True),
        })

        try:
            resp_json = re.search(r"```json\s*([\s\S]+?)\s*```|({[\s\S]+})", response.content.strip()).group()
            resp_json = resp_json[resp_json.find("{"):resp_json.rfind("}")+1]
            structure = json.loads(structure)
        except Exception as e:
            named_log(self, f"==> failed to parse JSON from LLM response. Raising exception")
            raise e
            
        metadata_log(self, elapsed, response)
        if self.agent_ctx.llm_cooldown:
            cooldown_log(self, self.agent_ctx.llm_cooldown)

        if save_json_path:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(structure, f)

        named_log(self, f"==> finish generating structure")
        return structure


    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        if input_data:
            assert_type(self, input_data, PaperData, "input_data")
            self.agent_ctx._working_paper = input_data
        
        structure = self.generate(self.save_json_path)
        self.agent_ctx._working_paper.sections = [
            SectionData(title=s["title"], description=s["description"]) for s in structure["sections"]
        ]
        return self.agent_ctx._working_paper
