class PaperPipeline:
    def __init__(self, steps: list):
        # check for "pipeline_entry" function definition
        for step in steps:
            if not callable(getattr(step, "pipeline_entry", None)):
                raise TypeError(f"Step {step} must have a defined implementation for 'pipeline_entry(data)'")
        
        self.steps = steps
        self._step_output = [None] * len(steps)
    
    def __call__(self):
        self.run()
    
    def run(self, initial_data=None):
        data = initial_data
        for i, step in enumerate(self.steps):
            data = step.pipeline_entry(data)
            self._step_output[i] = data
            
        return data
    