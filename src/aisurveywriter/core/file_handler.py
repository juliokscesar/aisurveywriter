from typing import List, Union
import yaml
import re
import os

class FileHandler:
    @staticmethod
    def read_yaml(file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    
    @staticmethod
    def write_yaml(data: dict, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data)
        
    @staticmethod
    def write_latex(template_path: str, sections: List[dict], file_path: str):
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        paper_content = ""
        bib_content = ""
        bib_pattern = r"\\begin{filecontents\*}(.*?)\\end{filecontents\*}"

        for section in sections:
            # Extract biblatex file content (from the pattern provided in the prompt)
            match = re.search(bib_pattern, section["content"], re.DOTALL)
            sec_bib_content = match.group(1).strip() if match else None
            if sec_bib_content is not None:
                bib_content += sec_bib_content
                section_text = re.sub(bib_pattern, "", section["content"], flags=re.DOTALL)
            else:
                section_text = section["content"]
                print("FAILED TO MATCH BIBLATEX CONTENT IN SECTION:", section["title"])
            paper_content += section_text

        bib_content = bib_content.replace("{mybib.bib}", "")
        bib_file = file_path.replace(".tex", "")+"bib.bib"

        # Replace variables in template
        tex_content = template.replace("{content}", paper_content)
        tex_content = tex_content.replace("{bibresourcefile}", bib_file)

        # Save files
        with open(file_path, "w", encoding="utf-8") as tex_f:
            tex_f.write(tex_content)
        with open(bib_file, "w", encoding="utf-8") as bib_f:
            bib_f.write(bib_content)


