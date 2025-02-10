from typing import Optional, Callable

from .pipeline_task import PipelineTask

from aisurveywriter.core.paper import PaperData
import aisurveywriter.core.file_handler as fh
import aisurveywriter.core.latex_handler as lh
from aisurveywriter.utils import named_log

class PaperSaver(PipelineTask):
    def __init__(self, save_path: str, template_path: str, bib_path: Optional[str] = None, tex_filter_fn: Callable[[str], str] = None):
        self.save_path = save_path
        self.template_path = template_path
        self.bib_path = bib_path
        self.tex_filter_fn = tex_filter_fn
    
    def pipeline_entry(self, input_data: PaperData):
        if not isinstance(input_data, PaperData):
            raise TypeError(f"Task {self.__class__.__name__} expects PaperData in pipe entry, but got {type(input_data)}")
        self.save(input_data)
        return input_data
    
    def save(self, paper: PaperData, save_path: Optional[str] = None):
        if save_path is not None:
            self.save_path = save_path
        
        lh.write_latex(
            self.template_path,
            paper,
            self.save_path,
            bib_path=self.bib_path,
            tex_filter_fn=self.tex_filter_fn,
        )
