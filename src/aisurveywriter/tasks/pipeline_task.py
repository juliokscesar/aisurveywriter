from abc import ABC, abstractmethod

class PipelineTask(ABC):
    @abstractmethod
    def pipeline_entry(self, input_data):
        pass

class DeliverTask(PipelineTask):
    """
    This is just a task that takes and returns the input as it is
    """
    def pipeline_entry(self, input_data):
        return input_data
