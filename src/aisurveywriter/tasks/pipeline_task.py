from abc import ABC, abstractmethod

class PipelineTask(ABC):
    @abstractmethod
    def pipeline_entry(self, input_data):
        pass

class DeliverTask(PipelineTask):
    """
    This is just a task that takes and returns the input as it is
    """
    def __init__(self, data_to_deliver):
        self.deliver = data_to_deliver
    
    def pipeline_entry(self, input_data):
        if input_data is not None:
            self.deliver = input_data
        return self.deliver
