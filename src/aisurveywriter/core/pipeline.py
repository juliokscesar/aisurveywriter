from typing import Optional, List, Tuple
from enum import Enum, auto
import queue

from aisurveywriter.tasks import PipelineTask

class TaskStatus(Enum):
    WAITING     = auto()
    RUNNING     = auto()
    COMPLETED   = auto()

class PaperPipeline:
    def __init__(self, steps: List[Tuple[str,PipelineTask]], status_queue: Optional[queue.Queue] = None):
        # check for "pipeline_entry" function definition
        for name,task in steps:
            if not callable(getattr(task, "pipeline_entry", None)):
                raise TypeError(f"Step {name} ({task}) must have a defined implementation for 'pipeline_entry(data)'")
        
        self.steps = steps
        self.status_queue = status_queue
        self._step_output = [None] * len(steps)
    
    def __call__(self):
        self.run()
    
    def run(self, initial_data=None):
        data = initial_data
        for i, (name,task) in enumerate(self.steps):
            self._notify_queue(i, name, TaskStatus.RUNNING)
            data = task.pipeline_entry(data)
            self._step_output[i] = data
            self._notify_queue(i, name, TaskStatus.COMPLETED)
            
        return data
    
    def _notify_queue(self, idx, name, status):
        if self.status_queue:
            self.status_queue.put((idx, name, status))
    