from abc import ABC, abstractmethod

class PipelineTask(ABC):
    def __init__(self):
        self.no_divide = False
    
    @abstractmethod
    def pipeline_entry(self, input_data):
        pass

    @abstractmethod
    def divide_subtasks(self, n: int, input_data=None):
        pass

    @abstractmethod
    def merge_subtasks_data(self, data):
        pass

class DeliverTask(PipelineTask):
    """
    This is just a task that takes and returns the input as it is
    """
    def __init__(self, data_to_deliver=None):
        super().__init__()
        self.no_divide = True
        self.deliver = data_to_deliver
    
    def pipeline_entry(self, input_data):
        if input_data is not None:
            self.deliver = input_data
        return self.deliver
