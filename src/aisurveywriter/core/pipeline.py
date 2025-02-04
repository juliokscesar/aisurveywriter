class PaperPipeline:
    def __init__(self, steps: list):
        # check for "pipeline_entry" function definition
        for step in steps:
            if not callable(getattr(step, "pipeline_entry", None)):
                raise TypeError(f"Step {step} must have a defined implementation for 'pipeline_entry(data)'")
        self.steps = steps
    
    def __call__(self):
        self.run()
    
    def run(self):
        pass
    