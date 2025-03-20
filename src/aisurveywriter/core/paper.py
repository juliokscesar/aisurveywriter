from dataclasses import dataclass
from typing import Union, List, Optional
import re
import json
import os

from .file_handler import read_yaml
from ..utils.logger import named_log

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
    def from_structure_json(subject: str, path: str):
        with open(path, "r", encoding="utf-8") as f:
            sections = json.load(f)["sections"]
        paper = PaperData(
            subject=subject,
            sections=[SectionData(s["title"], s["description"]) for s in sections]
        )
        return paper

    @staticmethod
    def from_tex(path: str, subject: Optional[str] = None, bib_path: Optional[str] = None, fig_path: Optional[str] = None):
        with open(path, "r", encoding="utf-8") as f:
            latex_content = f.read()
    
        # Extract bib path if found
        bib_match = re.search(r"\\addbibresource{([\w]+\.bib)}", latex_content)
        if bib_match:
            bib_path = os.path.join(os.path.dirname(path), bib_match.group(1))
            
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
            if "\\printbibliography" in sec_content:
                sec_content = sec_content[:sec_content.rfind("\\printbibliography")].strip()
            sections.append(SectionData(title=sec_title, description=sec_title, content=sec_content))

        # prepend abstract if found
        abstract_match = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", latex_content)
        if abstract_match:
            sections.insert(0, SectionData(
                title="Abstract", 
                description="abstract", 
                content=f"\\begin{{abstract}}{abstract_match.group(1)}\\end{{abstract}}"
            ))
        

        return PaperData(subject=subject, sections=sections, title=title, bib_path=bib_path, fig_path=fig_path)
            
    def load_tex(self, tex_path: str):
        tex = PaperData.from_tex(tex_path)
        
        if self.sections and len(self.sections) == len(tex.sections):
            for section, loaded_section in zip(self.sections, tex.sections):
                section.content = loaded_section.content
                section.title = loaded_section.title
                
        elif self.sections:
            new_sections = []
            for loaded_section in tex.sections:
                added = False
                for section in self.sections:
                    if section.title and section.title.strip().lower() == loaded_section.title.strip().lower():
                        section.title = loaded_section.title
                        section.content = loaded_section.content
                        new_sections.append(section)
                        added = True
                        break
                if not added:
                    new_sections.append(loaded_section)
            self.sections = new_sections
            
        else:
            self.sections = tex.sections.copy()
        
        self.bib_path = tex.bib_path if tex.bib_path else self.bib_path
        self.fig_path = tex.fig_path if tex.fig_path else self.fig_path
        self.title = tex.title if tex.title else self.title
        self.subject = tex.subject if tex.subject else self.subject
    
    def load_structure(self, structure_path: str):
        if structure_path.endswith("yaml"):
            struct = PaperData.from_structure_yaml(self.subject, structure_path)
        else:
            struct = PaperData.from_structure_json(self.subject, structure_path)
        
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
            if not section.content:
                named_log(self, f"Warning: section content empty: {section}")
            content += section.content + "\n"
        return content  

    def to_tex(self, template_path: str, save_path: str, bib_template_variable: Optional[str] = "bibresourcefile", figpath_template_variable: Optional[str] = "figspath"):
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
            
        paper_content = self.full_content()
        if self.title:
            paper_content = f"\\title{{{self.title}}}\n\\maketitle\n\\tableofcontents\n\n" + paper_content

        if self.bib_path:
            bib_replace = f"{{{bib_template_variable}}}"
            assert(bib_replace in template)
            template = template.replace(bib_replace, os.path.basename(self.bib_path))
        if self.fig_path:
            fig_replace = f"{{{figpath_template_variable}}}"
            assert(fig_replace in template)
            template = template.replace(fig_replace, os.path.basename(self.fig_path))

        tex_content = template.replace("{content}", paper_content)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        
