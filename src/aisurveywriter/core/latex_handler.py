from typing import List, Callable, Optional
import os
import re

from aisurveywriter.core.paper import PaperData, SectionData

def write_latex(template_path: str, paper: PaperData, file_path: str, bib_path: Optional[str] = None, bib_template_variable: Optional[str] = "bibresourcefile", tex_filter_fn: Optional[Callable[[str],str]] = None):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    paper_content = paper.full_content()
    if paper.title:
        paper_content = f"\\title{{{paper.title}}}\n" + paper_content

    # Replace variables in template
    if tex_filter_fn is not None:
        paper_content = tex_filter_fn(paper_content)
    if bib_path:
        template = template.replace(f"{{{bib_template_variable}}}", bib_path)

    tex_content = template.replace("{content}", paper_content)

    # Save files
    with open(file_path, "w", encoding="utf-8") as tex_f:
        tex_f.write(tex_content)

def read_latex(file_path: str, bib_path: Optional[str] = None) -> PaperData:
    with open(file_path, "r", encoding="utf-8") as f:
        latex_content = f.read()
    
    # Extract title (assuming \title{} is present)
    title_match = re.search(r"\\title\{(.+?)\}", latex_content)
    subject = title_match.group(1) if title_match else "Unknown Title"
    
    # Extract sections
    sections = []
    section_matches = re.finditer(r"\\section\{(.+?)\}([\s\S]*?)(?=\\section|\Z)", latex_content)

    for match in section_matches:
        sec_title = match.group(1).strip()
        sec_content = match.group(2).strip()
        sections.append(SectionData(title=sec_title, description=sec_title, content=sec_content))

    # Read bibliography if provided
    bib_content = None
    if bib_path:
        with open(bib_path, "r", encoding="utf-8") as bib_file:
            bib_content = bib_file.read()

    return PaperData(subject=subject, sections=sections, bib=bib_content)