from typing import Optional, Callable

from .pipeline_task import PipelineTask

from aisurveywriter.core.paper import PaperData
import aisurveywriter.core.file_handler as fh
import aisurveywriter.core.latex_handler as lh
from aisurveywriter.utils.helpers import assert_type

class PaperSaver(PipelineTask):
    def __init__(self, save_path: str, template_path: str, bib_path: Optional[str] = None, tex_filter_fn: Callable[[str], str] = None):
        self.no_divide = True
        self.save_path = save_path
        self.template_path = template_path
        self.bib_path = bib_path
        self.tex_filter_fn = tex_filter_fn
    
    def pipeline_entry(self, input_data: PaperData) -> PaperData:
        assert_type(self, input_data, PaperData, "input_data")
        self.save(input_data)
        return input_data
    
    def save(self, paper: PaperData, save_path: Optional[str] = None):
        if save_path is not None:
            self.save_path = save_path
        
        paper.to_tex(self.template_path, self.save_path)

    def divide_subtasks(self, n, input_data=None):
        raise NotImplemented()
    
    def merge_subtasks_data(self, data):
        raise NotImplemented()
