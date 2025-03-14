from typing import List
from pydantic import BaseModel
import re

class PromptInfo(BaseModel):
    text: str
    input_variables: List[str]

    @staticmethod
    def from_template(template: str) -> "PromptInfo":
        # Regular expression to find valid single curly brace variables
        matches = re.findall(r'(?<!{){([^{}]+)}(?!})', template)
        
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

    ADD_FIGURES_PROMPT = r"""- You are an academic expert in "{subject}". You are writing a comprehensive survey paper on that subject.

- You are receiving from the human: 
- the content of references for this paper (reference_content block)
- the content of one section of this paper (in latex)

- YOUR JOB:
- Look for figures in the references.
- You must add a figure from the reference to this section, following:
- Place the figure in a contextually correct and relevant place.
- You must ID this figure with a unique name and a proper caption:
    - DO NOT use a general name that could be repeated accidently. Make sure to use something unique (use some random name)
    - DO NOT use Scheme figures (usually appear as Scheme X.)
    - Use the same caption from the original figure in its proper reference, adding "Adapted from (AUTHOR), (YEAR)" in the end
    - If you're unable to identify the caption, add a descriptive and detailed caption
    - The caption must appear BELOW the figure

- If adding a visual element is not relevant, do nothing
- If there are any Figures present (including TikZ figures), do not alter nor remove them

- THE FIGURES MUST BE WITHIN A PROPER FIGURE BODY IN LATEX, AS:
\begin{{figure}}[h!]
\includegraphics{{unique_name}}
\caption{{Very descriptive caption}}
%author: FigAuthor
\label{{fig:unique_random_label}}
\end{{figure}}

- The Figure MUST BE FROM ONE OF THE REFERENCES. Provide at the end of the caption the credits as: "Adapted from Authors, Year."
  - Do not enumerate unique_label like unique_label_1, unique_label_2, etc. Either create a unique random label, or assign a label related to the figure
  - Make sure to not use a generic unique_name. Use something random or VERY specific for this figure.
  - DO NOT use optional parameters (such as [width=0.xx], etc...)

- IT IS ESSENTIAL THAT YOU USE "\includegraphics{{name}}" AND "\caption{{descriptive caption}}" RIGHT AFTER

- You must use only figures that weren't used. That is, do not use the following used_figures:
[begin: used_figures]
{used_figures}
[end: used_figures]

- **YOU MUST NOT ALTER THE SECTION'S CONTENT, ONLY ADD THE FIGURES**

- **YOUR OUTPUT MUST BE ONLY THE LATEX FOR THIS SECTION, NO "Okay, here it is...\". Do not write any message for the human, only the section content"""

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
- You must be concise
- You must write only in English
- Your output should contain nothing more than the directives. So don't output stuff like 'Okay, here are the directives....' """

    APPLY_REVIEW_SECTION_PROMPT = r"""- **Role:** You are an expert academic writer in **{subject}**.  
- **Input:** You will receive a LaTeX section of a survey paper, review directives, and reference content from PDFs.  
- **Task:** Apply the review directives while maintaining the section’s integrity and improving its quality based on references.  

### **Guidelines**  
- **Maintain style:** Keep a **scientific, objective, and formal tone**.  
- **Preserve structure:** Do **not** add or remove sections (\section) or subsections (\subsection).  
- **Expand, don’t summarize:** Maintain or increase length with meaningful content.  
- **Discuss figures:** Refer to existing figures via their **\label** and \caption details. Do **not** replace standard figures with TikZ.  
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
