from abc import ABC, abstractmethod

class PipelineTask(ABC):
    @abstractmethod
    def pipeline_entry(self, input_data):
        pass
