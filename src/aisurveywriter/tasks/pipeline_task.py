from abc import ABC, abstractmethod

from aisurveywriter.core.paper import PaperData
from aisurveywriter.core.agent_context import AgentContext

class PipelineTask(ABC):
    def __init__(self, no_divide = False, agent_ctx: AgentContext = None):
        self.no_divide = no_divide
        self.agent_ctx: AgentContext = agent_ctx
    
    def pipeline_entry(self, input_data):
        return input_data

    def divide_subtasks(self, n: int, input_data=None):
        return [self] * n

    def merge_subtasks_data(self, data):
        return data

class DeliverTask(PipelineTask):
    """
    This is just a task that takes and returns the input as it is
    """
    def __init__(self, data_to_deliver=None):
        super().__init__()
        self.no_divide = True
        self.deliver = data_to_deliver
    
    def pipeline_entry(self, input_data):
        if input_data:
            self.deliver = input_data
        return self.deliver

    def divide_subtasks(self, n, input_data=None):
        raise NotImplemented()
    
    def merge_subtasks_data(self, data):
        raise NotImplemented()

class LoadTask(PipelineTask):
    def __init__(self, tex_path: str, subject: str):
        self.no_divide = True
        self.tex_path = tex_path
        self.subject = subject

    def pipeline_entry(self, input_data):
        paper = PaperData.from_tex(self.tex_path, self.subject)
        if isinstance(input_data, PaperData):
            for ps, ins in zip(paper.sections, input_data.sections):
                ps.description = ins.description
        return paper

    def divide_subtasks(self, n, input_data=None):
        raise NotImplemented()
    
    def merge_subtasks_data(self, data):
        raise NotImplemented()