from typing import List
from pydantic import BaseModel
from pydantic.json import pydantic_encoder
import json
import os
import re

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
- You will be given the content of a PDF, an image in Base64
- You must give a description for the provided image:
    - Identify the figure around its context and what it's illustratng
    - Provide a description for the image:
        - Be direct, objective, and clear in your description, while being concise
        - The image belongs to the PDF content provided. Base your description taking its context into account.
        - Prioritize the use of keywords that would link to this image (by similarity)
    
    - **ATTENTION** some images may be not related at all to the content (such as copyright images, license symbols, single-letter images, journal cover, pictures of people, blank/single-color images).
        - In this case, just describe this image as \"UNRELATED FIGURE\" and don't need to describe it
        - If the image does not appear to have anything to do with the paper's context, also just respond \"UNRELATED FIGURE\" and don't need to describe it

- The image description must be between 100-300 words.
- **OUTPUT FORMAT**:
    - Give ONLY the image description/caption
    - DO NOT include the figure number (e.g. "Figure 1...")
    - DO NOT write any message directed to the human (e.g. "here's the description", "okay, here it is", "The figure shows...")
    - USE KEYWORDS (for similarity search)"""

FIGURE_EXTRACTOR_HUMAN_PROMPT = """- PDF paper content:
[begin: pdf_content]

{pdf_content}

[end: pdf_content]"""

DEBLOAT_CAPTION_SYSTEM_PROMPT = """- You are an academic writer and peer reviewer, specialist in understanding figures and their context
- You will receive from the human Figure captions extracted automatically from an article PDF
- Your job is to read the extracted caption and keep only the important and actual body of the caption:
    - Some captions come extracted with the name of the journal or some PDF metadata
    - You must debloat (remove) all this extra junk from the caption, and maintain only the actual caption content
    - To judge what's junk, understand the figure caption from most of the content, and the junk will be next to the end, usually something that has nothing to do with the figure itself
    
- DO NOT alter the actual caption content. Copy it as-it-is, remove ONLY the extra junk stuff
- Examples of what you should remove:
    - ""Fig. 1. Schematic diagram for the preparation of O-g-C 3 N 4 @(Pd-TAPP) 3 nanocomposites and their LB films. C.-N. Ye et al.         International Journal of Hydrogen Energy 98 (2025) 1119–1130 1121"" -> Remove from "C.-N." beyond -> Output: "Fig. 1. Schematic diagram for the preparation of O-g-C 3 N 4 @(Pd-TAPP) 3 nanocomposites and their LB films." ""
    - 
"""

class FigureExtractor:
    def __init__(self, llm: LLMHandler, references: ReferenceStore, output_dir: str, system_prompt: str = FIGURE_EXTRACTOR_SYSTEM_PROMPT, human_prompt: str = FIGURE_EXTRACTOR_HUMAN_PROMPT, request_cooldown_sec: int = 30):
        self.llm = llm
        self._cooldown = request_cooldown_sec
        
        self.references = references
        self.output_dir = os.path.abspath(output_dir)
        
        self._system = SystemMessage(content=system_prompt)
        self._human_template = human_prompt
        self._image_template = "data:image/png;base64,{imgb64}"
    
    def extract_captions(self, dump_data=True) -> List[FigureInfo]:
        images = self.references.pdf_proc.extract_images(self.output_dir, verbose=False, filter_min_wh=[40,40], extract_captions=True)
        figures_info: List[FigureInfo] = []
        figure_num_pattern = re.compile(r"^(Figure|Fig\.|Figura)\s*(?:\n\s*)*(\d+)\.?\s*", re.MULTILINE)
        
        for img_idx, image in enumerate(images):
            # skip image with no caption found
            if image.caption is None:
                named_log(self, "removing image with unidentified caption:", os.path.basename(image.path))
                os.remove(image.path)
                continue
            
            # remove "Figure X./Fig. X/Figura X." from caption
            if fig_num_match := figure_num_pattern.match(image.caption):
                image.caption = image.caption[fig_num_match.end():].strip()
            
            figures_info.append(FigureInfo(
                id=img_idx,
                basename=os.path.basename(image.path),
                description=image.caption,
            ))

        if dump_data:
            with open(os.path.join(self.output_dir, "figures_info.json"), "w", encoding="utf-8") as f:
                json.dump(figures_info, f, default=pydantic_encoder, indent=2)

        return figures_info
        
    def extract_describe(self, dump_data=True):
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
                {"type": "text", "text": self._human_template.replace("{pdf_content}", pdf_content)},
                {"type": "image_url", "image_url": {"url": self._image_template.replace("{imgb64}", image_base64)}}
            ])
    
            elapsed, response = time_func(self.llm.model.invoke, [self._system, image_prompt])
            
            # check if image is unrelated
            if "UNRELATED FIGURE" in response.content.upper().strip():
                os.remove(image.path)
                named_log(self, f"identified unrelated image: {os.path.basename(image.path)}")
            else: 
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
                json.dump(figures_info, f, default=pydantic_encoder, indent=2)

        return figures_info
