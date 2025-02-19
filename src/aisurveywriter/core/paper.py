from dataclasses import dataclass
from typing import Union, List, Optional
import re

from aisurveywriter.core.file_handler import read_yaml

@dataclass
class SectionData:
    title: str
    description: str
    content: Union[None, str] = None
    
    
@dataclass
class PaperData:
    subject: str
    sections: List[SectionData]
    title: Union[None, str] = None
    bib_path: Union[None, str] = None
    fig_path: Union[None, str] = None
    
    @staticmethod
    def from_structure_yaml(subject: str, path: str):
        sections = read_yaml(path)["sections"]
        paper = PaperData(
            subject=subject,
            sections=[SectionData(s["title"], s["description"]) for s in sections]              
        )
        return paper

    @staticmethod
    def from_tex(path: str, subject: Optional[str] = None, bib_path: Optional[str] = None, fig_path: Optional[str] = None):
        with open(path, "r", encoding="utf-8") as f:
            latex_content = f.read()
    
        doc_match = re.search(r"\\begin\{document\}(.+?)\\end\{document\}", latex_content)
        if doc_match:
            latex_content = doc_match.group()
    
        # Extract title (assuming \title{} is present)
        title_match = re.search(r"\\title\{(.+?)\}", latex_content)
        title = title_match.group(1) if title_match else None
        
        # Extract image directory (assuming \graphicspath{} is present)
        if not fig_path:
            dir_match = re.search(r"\\graphicspath\s*\{\s*\{([^}]*)\}\s*\}", latex_content)
            fig_path = dir_match.group(1) if dir_match else None

        # Extract sections
        sections = []
        section_matches = re.finditer(r"\\section\{(.+?)\}([\s\S]*?)(?=\\section|\Z)", latex_content)

        for match in section_matches:
            sec_title = match.group(1).strip()
            sec_content = match.group(2).strip()
            sec_content = f"\n\\section{{{sec_title}}}\n\n" + sec_content
            sections.append(SectionData(title=sec_title, description=sec_title, content=sec_content))

        # Read bibliography if provided
        return PaperData(subject=subject, sections=sections, title=title, bib_path=bib_path, fig_path=fig_path)
            
    def load_tex(self, tex_path: str):
        tex = PaperData.from_tex(tex_path)
        
        if self.sections and len(self.sections) == len(tex.sections):
            for section, loaded_section in zip(self.sectons, tex.sections):
                section.content = loaded_section.content
                section.title = loaded_section.title
                
        elif self.sections:
            self.sections.extend(tex.sections)
            
        else:
            self.sections = tex.sections.copy()
    
    def load_structure(self, structure_json_path: str):
        struct = PaperData.from_structure_yaml(self.subject, structure_json_path)
        
        if self.sections and len(self.sections) == len(struct.sections):
            for section, struct_section in zip(self.sections, struct.sections):
                section.description = struct_section.description
                section.title = struct_section.title
    
        elif self.sections:
            self.sections.extend(struct.sections)
        
        else:
            self.sections = struct.sections.copy()
    
    def full_content(self) -> str:
        content = ""
        for section in self.sections:
            content += section.content + "\n"
        return content  
