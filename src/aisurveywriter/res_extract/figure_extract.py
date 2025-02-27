from typing import List
from pydantic import BaseModel
from pydantic.json import pydantic_encoder
import json
import os

from langchain_core.messages import SystemMessage, HumanMessage

from aisurveywriter.core.llm_handler import LLMHandler
from aisurveywriter.store.reference_store import ReferenceStore
from aisurveywriter.utils.helpers import image_to_base64, time_func
from aisurveywriter.utils.logger import named_log, cooldown_log, metadata_log

class FigureInfo(BaseModel):
    id: int
    basename: str
    description: str
    
FIGURE_EXTRACTOR_SYSTEM_PROMPT = """- You are an academic writer and peer reviewer, specialist in understanding figures and their context
- You will be given the content of a PDF, an image in Base64, and its respective Fig. number in the reference
- You must give a description for the provided image:
    - If you are able to identify the figure in the PDF and its original caption, copy it as-it-is
    - If not, provide a detailed description:
        - Be direct, objective, and clear in your description
        - The image belongs to the PDF content provided. Base your description in context with the this content.
        - Prioritize the use of keywords that would link to this image (specially by similarity)
    
    - Attention: some images may be not related at all to the content (such as copyright images, license symbols, journal cover, pictures of people, blank/single-color images).
        - In this case, just describe this image as \"NOT RELATED\" and don't need to describe it
        - If the image does not appear to have anything to do with the paper's context, also just respond \"NOT RELATED\" and don't need to describe it

- The image description must be between 100-300 words."""

FIGURE_EXTRACTOR_HUMAN_PROMPT = """- PDF paper content:
[begin: pdf_content]

{pdf_content}

[end: pdf_content]

[begin: fig_info]

Figure number: {figure_number}

[end: fig_info]"""


class FigureExtractor:
    def __init__(self, llm: LLMHandler, references: ReferenceStore, output_dir: str, system_prompt: str = FIGURE_EXTRACTOR_SYSTEM_PROMPT, human_prompt: str = FIGURE_EXTRACTOR_HUMAN_PROMPT, request_cooldown_sec: int = 30):
        self.llm = llm
        self._cooldown = request_cooldown_sec
        
        self.references = references
        self.output_dir = os.path.abspath(output_dir)
        
        self._system = SystemMessage(content=system_prompt)
        self._human_template = human_prompt
        self._image_template = "data:image/png;base64,{imgb64}"
        
    def extract(self, dump_data=True):
        images = self.references.extract_images(self.output_dir)
        pdf_contents = self.references.full_content()
        
        figures_info: List[FigureInfo] = []

        images_amount = len(images)    
        named_log(self, f"==> describing {images_amount} images")    

        for i, image in enumerate(images):
            image_base64 = image_to_base64(image.path)
            pdf_idx = self.references.pdf_paths.index(image.pdf_source)
            pdf_content = pdf_contents[pdf_idx]
    
            image_prompt = HumanMessage(content=[
                {"type": "text", "text": self._human_template.replace("{pdf_content}", pdf_content).replace("{figure_number}", str(i+1))},
                {"type": "image_url", "image_url": {"url": self._image_template.replace("{imgb64}", image_base64)}}
            ])
    
            elapsed, response = time_func(self.llm.model.invoke, [self._system, image_prompt])

            figures_info.append(FigureInfo(
                id=i,
                basename=os.path.basename(image.path),
                description=response.content,
            ))

            named_log(self, f"==> finish describing image {i+1}/{images_amount} ({figures_info[-1].basename})")
            metadata_log(self, elapsed, response)

            if self._cooldown:
                cooldown_log(self, self._cooldown)

        if dump_data:
            with open(os.path.join(self.output_dir, "figures_info.json"), "w", encoding="utf-8") as f:
                json.dump(figures_info, f, default=pydantic_encoder)

        return figures_info
