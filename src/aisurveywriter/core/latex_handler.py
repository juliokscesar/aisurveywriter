from typing import List, Callable, Optional
import os
import re

from aisurveywriter.core.paper import PaperData, SectionData

def write_latex(template_path: str, paper: PaperData, file_path: str, find_bib_pattern: Optional[str] = None, tex_filter_fn: Optional[Callable[[str],str]] = None):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    paper_content = ""
    bib_content = ""

    #bib_pattern = r"\\begin{filecontents\*}(.*?)\\end{filecontents\*}"

    if find_bib_pattern is not None:
        for section in paper.sections:
            # Extract biblatex file content (from the pattern provided in the prompt)
            match = re.search(find_bib_pattern, section.content, re.DOTALL)
            sec_bib_content = match.group(1).strip() if match else None
            if sec_bib_content is not None:
                bib_content += sec_bib_content
                section_text = re.sub(find_bib_pattern, "", section.content, flags=re.DOTALL)
            else:
                section_text = section.content
                print("FAILED TO MATCH BIBLATEX CONTENT IN SECTION:", section.title)
            paper_content += section_text

        bib_content = bib_content.replace("{mybib.bib}", "")
        bib_file = file_path.replace(".tex", ".bib")
        with open(bib_file, "w", encoding="utf-8") as bib_f:
            bib_f.write(bib_content)

    # Replace variables in template
    tex_content = template.replace("{content}", paper_content)
    tex_content = tex_content.replace("{bibresourcefile}", os.path.basename(bib_file))
    if tex_filter_fn is not None:
        tex_content = tex_filter_fn(tex_content)

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