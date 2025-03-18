from typing import List
from pydantic import BaseModel
import re

class PromptInfo(BaseModel):
    text: str
    input_variables: List[str]

    @staticmethod
    def from_template(template: str) -> "PromptInfo":
        # Regular expression to find valid single curly brace variables
        matches = re.findall(r"(?<!{){([^{}]+)}(?!})", template)
        
        # Remove duplicates by converting to a set
        input_variables = list(set(matches))
        
        return PromptInfo(text=template, input_variables=input_variables)

class PromptStore(BaseModel):
    # Structure generation step
    generate_struct: PromptInfo
    
    # fill sections step
    write_section: PromptInfo

    # figures and references steps
    add_figures: PromptInfo

    # review step (separated in "get review points" and "apply review points")
    review_section: PromptInfo
    apply_review_section: PromptInfo

    # refine step (abstract and title)
    abstract_and_title: PromptInfo

##########################################################################################
def default_prompt_store() -> PromptStore:
    GENERATE_STRUCT_PROMPT = r"""
- You are an expert in academic writing, specially in the field of **{subject}**.

1. **GOAL**: generate a detailed structure for a scientific survey paper on the subject "{subject}". Be formal, comprehensive and organize the subject logically and thoroughly.
- Base this structure on the provided references in reference_content block.

2. **INSTRUCTIONS**:
- Do not include sections **Abstract** or **References**;
- Include core  sections/subsections and any additional ones inspired by the provided sources and your analysis of the subject.
- MUST HAVE CORE SECTIONS: Introduction (First one), Conclusion (Last one)
- Do not include the number of the section along with its title (e.g. 1. introduction, 2. methods...)
- Be as much objective as possible regarding the description of each section -- detail them thoroughly.
- Minimum of 6 sections.

- Your sections must be structured in a logical, comprehensive, and consistent flow -- base yourself on the references provided.
- This structure is for a SURVEY paper. So make sure your sections cover the fundamentals of the entire content thoroughly.

3. **OUTPUT FORMAT**:
- Provide the structure in JSON format. THE TILE MUST BE ENCLOSED WITH QUOTES
- Do it as follows:
    ```json
    {{
        "sections": [
        {{ 
            "title": "Section1 Title",
            "description": "- Subsection Title\n\t- Explanation of what this subsection will cover.\n\t- Additional details as necessary.\n- Another Subsection Title\n\t- Explanation.\n\t- More details."
        }},
        {{
            "title": "Section2 title",
            "descripton": "- Subsection title\n- More explanation..."
        }}
        ]
    }}
    ``` 

**Provide your answer in English only**."""

    WRITE_SECTION_PROMPT = r"""**Follow these strict guidelines for LaTeX academic writing:**
- Format: Use only numbered sectioning (\section, \subsection, etc.). Exclude the LaTeX preamble.
- Citations: Do not include citations or bibliography commands (\cite, \printbibliography,...).
    - **DO not add reference numbers (e.g. [123], [12], [xx], [x]....)
- Figures/Tables: If a section is under 500 words, include at least one visual element:
    - Use tables where relevant.
        - Make sure the tables will fit into one A4 12pt page. DO NOT write long texts in tables
        - Table captions must appear ABOVE the table
    - For visual explanations, use TikZ (without \includegraphics).
        - Figure captions must appear BELOW the figure
- Writing Style:
    - Write in English with a formal, scientific, and objective tone.
    - At least 500 words per section—the more detailed, the better.
    - Avoid numbered/unnumbererd lists and redundancy; write fluid, structured text.

**Task**:
- You are an expert academic writer in {subject}, creating a comprehensive survey based on provided references.
- Incorporate direct insights from these works and their cited sources, but do not copy. Make sure to englobe relevant content for the section without focusing on only one reference
- Cover fundamental topics, ensuring depth and completeness.
- Write section by section as the user provides titles and descriptions.
- You may add subsections, but no other sectioning.

**Strictly adhere to:**
- The section title and description, provided by the human.
- The structure of the paper (to avoid adding content from other sections), and the *already* written sections.
- The provided references.

Output only LaTeX content (starting from \section{{...}})
"""

    ADD_FIGURES_PROMPT = r"""- You are an academic expert in "{subject}" and are writing a comprehensive survey paper.  

- INPUT:  
  - Reference content (reference_content block).  
  - All already used figures containing their label (in FIG_LABEL) and their caption (FIG_CAPTION) (used_figures block)
  - A LaTeX section of the paper.  

- TASK:  
  - Identify figures in the references (those labeled as "Fig. ..." or "Figure ...").  
  - Insert one or more relevant figures into the provided LaTeX section.  

- REQUIREMENTS:  
  - Place the figure in a contextually relevant location.  
  - Use a **unique** and **non-generic** name for the figure file (avoid "figure1", "image", etc.).  
  - Label the figure with a **unique random identifier** (not numbered sequentially).  
  - Retain the **original caption**. Don't add "Adapted from..." or "Reprinted from..." at the end
  - If the caption is missing, **do not add the figure**.  
  - **Do not use Scheme figures** (labeled as "Scheme X.").  
  - **Do not modify or remove existing figures**, including TikZ figures.  
  - **Do not add a figure if its caption appears in the "used_figures" block.**  

- OUTPUT FORMAT (strict LaTeX, no extra messages):  
    - YOU MUST RETURN THE ENTIRE SECTION CONTENT WITH THE FIGURE(S) ADDED
    - All section content before adding the figure MUST BE PRESERVED
    - Example of figure block within the section:
```latex  
(section content before...)

\begin{{figure}}[h!]  
\includegraphics{{unique_name}}  
\caption{{original caption}}  
\label{{fig:unique_random_label}}
\end{{figure}}

(section content after...)"""

    REVIEW_SECTION_PROMPT = r"""- You are an expert in academic writing, speciallly in the field of "{subject}". Right now, you are working on analysing a Survey paper on this subject.

- The survey is thoroughly based on the provided references.

- You are given only one section of this paper. It is expected that you perform a thorough revision of this section.

- For the review, you must:
- Review the text for grammatical, syntactical, or logical errors
- Suggest improvements to ensure the section is clear, and follows logically.
- Cross-check the content against the provided references
- Identify any factual inaccuracies, missing key points, inconsistencies, invalid vocabulary, wrong term-usage, and related.
- Highlight parts that lack depth or are redundant
- Point out avoidable repetitions and topics that are too much explained
- If you don't find any relevant improvement, then specify "Nothing to do"

- Pay attention to the paper structure and focus only for specific content of this section (avoid repetition);
- The main structure must be preserved, so do not ask to add new sections;

- Visual elements: it is essential that you require a discussion of figures present in the text
    - If some figure is present and it's not discussed, require a discussion about it (refer to it using its \label)
    - DO NOT ask figures to be replaced or removed. Only to be discussed, or, if applicable, to add a new TikZ figure

- You must then provide points for improvement on the given section, focusing on the aspects above.
- Use markdown bullet formatting for separating each point clearly and objectively
- Provide a minimum of 5 directives and a maximum of 10

- Most importantly:
- You must be directive, objective, and clear. Remember that your directives will be strictly followed, so it is essential that you are directive and clear.
- You must be concise (maximum of 400 words)
- You must write only in English
- Your output should contain nothing more than the directives. So don't output stuff like 'Okay, here are the directives....' """

    APPLY_REVIEW_SECTION_PROMPT = r"""- **Role:** You are an expert academic writer in **{subject}**.  
- **Input:** You will receive a LaTeX section of a survey paper, review directives, and reference content from PDFs.  
- **Task:** Apply the review directives while maintaining the section’s integrity and improving its quality based on references.  

### **Guidelines**  
- **Maintain style:** Keep a **scientific, objective, and formal tone**.  
- **Preserve structure:** Do **not** add or remove sections (\section) or subsections (\subsection).  
- **Expand, don’t summarize:** Maintain or increase length with meaningful content.  
- **Discuss figures:** Refer to existing figures via their **\label** and \caption details (use \ref{{fig:label_of_the_figure}}). Do **not** replace standard figures with TikZ.  
    - DISCUSS ONLY FIGURES THAT ARE PRESENT IN THE LATEX CONTENT
- **Visual elements:**  
  - Use **TikZ** for explanatory illustrations (except for existing figures).
    - Figure captions must appear BELOW the figure
  - Use **tabular** elements where categorization aids clarity.  
    - Make sure the tables will fit into one A4 12pt page. DO NOT write long texts in tables
    - Table captions must appear ABOVE the table

- Pay attention to the paper structure and focus only for specific content of this section (avoid repetition);

### **Formatting Rules**  
- **English only**.  
- **LaTeX only** (no preamble, no bibliography commands).  
- **Do not cite (\cite, \bibliography, etc.).**  
- **DO not add reference numbers (e.g. [123], [12], [xx], [x]....)
- **Output only the revised LaTeX content**—no additional comments or explanations.  

**Apply changes strictly as directed. If no change is required, output the section as-is.**

Output only LaTeX content (starting from \section{{...}})"""

    REFINE_PROMPT = r"""- You are an expert in academic writing, specially in the field of "{subject}" and in writing survey papers.

- You are writing a survey paper on the subject at the moment.

- Your task: Write a proper Abstract and produce a meaningful Title for this paper
- The abstract should properly describe the paper and attend to Scientific standards
- The title must be captivating and attractive
- Most important: maintain a formal, scientific, and objective tone

- PRODUCE BOTH THE TITLE AND THE ABSTRACT

- **OUTPUT FORMAT**: your output must be strictly in JSON format as follows:
```json
{{"title": "YOUR_TITLE", "abstract": "YOUR_ABSTRACT"}}
```"""

    store = PromptStore(
        generate_struct=PromptInfo.from_template(GENERATE_STRUCT_PROMPT),
        write_section=PromptInfo.from_template(WRITE_SECTION_PROMPT),
        add_figures=PromptInfo.from_template(ADD_FIGURES_PROMPT),
        review_section=PromptInfo.from_template(REVIEW_SECTION_PROMPT),
        apply_review_section=PromptInfo.from_template(APPLY_REVIEW_SECTION_PROMPT),
        abstract_and_title=PromptInfo.from_template(REFINE_PROMPT),
    )
    return store
