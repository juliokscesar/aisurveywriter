from typing import Optional, Union, List
import concurrent.futures

from .core.pipeline import PaperPipeline
from .core.llm_handler import LLMHandler

class Agent:
    def __init__(self, id: int, task = None, llm: LLMHandler = None):
        self.id = id
        self.task = task
        self.output_data = None
        self.llm = llm
    
    def set_task(self, task, llm: LLMHandler = None):
        self.task = task
        self.llm = llm
        
    def run_task(self):
        if self.llm:
            self.task.llm = self.llm
        self.output_data = self.task.pipeline_entry()

                

class MultiAgentPipeline:
    def __init__(self, n: int, pipeline: PaperPipeline, agent_llms: Optional[List[LLMHandler]] = None):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=n)
        self.agents = [Agent(i) for i in range(n)]
        self.n = n
        self.pipeline = pipeline
        
        # case where there is an LLM list for every agent for every task
        if agent_llms:
            assert(len(agent_llms) == n)
        self.agent_llms = agent_llms
        

    def run(self, initial_data = None):
        data = initial_data
        for step in self.pipeline.steps:
            # check if can divide task into subtasks
            if step.no_divide or not getattr(step, "divide_subtasks") or not getattr(step, "merge_subtasks_data"):
                data = step.pipeline_entry(data)
                continue
            
            # Divide the task 
            subtasks = step.divide_subtasks(n=self.n, input_data=data)
            futures = [None] * self.n
            for i in range(self.n):
                diff_llm = None
                if self.agent_llms:
                    diff_llm = self.agent_llms[i]
                self.agents[i].set_task(subtasks[i], data, diff_llm)
                futures[i] = self.executor.submit(self.agents[i].run_task)
            
            # get each step result
            results = [future.result() for future in futures]
            data = step.merge_subtasks_data(results)

        return data
            
    def shutdown(self):
        self.executor.shutdown(wait=True)
