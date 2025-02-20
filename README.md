# AI Survey Writer

An approach to automating the process of writing Survey Papers using state-of-the-art LLMs.

## Usage:

This tool can be used directly from the terminal. 


### Installation 

```
git clone https://github.com/juliokscesar/aisurveywriter.git
cd aisurveywriter
pip install -e .
```

And then you can run it using 
```
python -m aisurveywriter
```

### Example usage (CLI)
The only two positional arguments are "references_dir" and "subject", respectively, and the two most useful optional arguments are "--llm-model (-m)" and "--llm (-l)". 

For example, to generate a paper on the subject "Langmuir and Langmuir-Blodgett films" with PDF references in the directory "./refexamples", using Google's gemini-2.0-pro-exp, it's simply done by:

```
python -m aisurveywriter -l google -m gemini-2.0-pro-exp ./refexamples "Langmuir and Langmuir-Blodgetts films"
```

CLI help:
```
usage: python -m aisurveywriter [-h] [--save-dir SAVE_DIR] [--llm {openai,google}] [--llm-model LLM_MODEL] [--credentials CREDENTIALS] [--structure STRUCTURE] [--paper PAPER] [--embed-model EMBED_MODEL] [--embed-type EMBED_TYPE] [--bibdb BIBDB] [--faissbib FAISSBIB] [--images IMAGES]
                   [--faissfig FAISSFIG] [--faissref FAISSREF] [--no-ref-rag] [--no-figures] [--no-reference] [--no-abstract] [--no-tex-review] [--no-review] [--cooldown COOLDOWN] [--embed-cooldown EMBED_COOLDOWN] [--tex-template TEX_TEMPLATE]
                   references_dir subject

positional arguments:
  references_dir        Path to directory containg all PDF references
  subject               Main subject of the survey. Can be the Title too

options:
  -h, --help            show this help message and exit
  --save-dir SAVE_DIR   Path to output directory
  --llm {openai,google}, -l {openai,google}
                        Specify LLM to use. Either 'google' or 'openai'. Default is google
  --llm-model LLM_MODEL, -m LLM_MODEL
                        Specific LLM model to use. Default is gemini-2.0-flash
  --credentials CREDENTIALS
                        YAML file containing your API keys
  --structure STRUCTURE, -s STRUCTURE
                        JSON file containing the structure to use. If provided, this will skip the structure generation process.
  --paper PAPER, -p PAPER
                        Path to .TEX paper to use. If provided, won't write one from the structure, and will skip directly to reviewing it (unless --no-review) is provided
  --embed-model EMBED_MODEL, -e EMBED_MODEL
                        Text embedding model name
  --embed-type EMBED_TYPE, -t EMBED_TYPE
                        Text embedding model type (google, openai, huggingface)
  --bibdb BIBDB, -b BIBDB
                        Path to .bib database to use. If none is provided, one will be generated by extracting every reference across all PDFs
  --faissbib FAISSBIB, -fb FAISSBIB
                        Path to FAISS vector store of the .bib databse. If none is provided, one will be generated
  --images IMAGES, -i IMAGES
                        Path to all images extracted from the PDFs. If none is provided, all images will be extracted and saved to a temporary folder
  --faissfig FAISSFIG, -ff FAISSFIG
                        Path to FAISS vector store containing the metadata (id, path and description) for every image. If none is provided, one will be created
  --faissref FAISSREF, -fr FAISSREF
                        Path to FAISS of references contents to retrieve only a piece of information, instead of sending the entire document.
  --no-ref-rag          Don't create a RAG for reference contents. Use entire PDF instead
  --no-figures          Skip step of adding figures to the written paper.
  --no-reference        Skip step of adding references to the text
  --no-abstract         Skip step of writing Abstract and Title
  --no-tex-review       Skip TEX review
  --no-review           Skip content/writing review step
  --cooldown COOLDOWN, -w COOLDOWN
                        Cooldown between two consecutive requests made to the LLM API
  --embed-cooldown EMBED_COOLDOWN
                        Cooldown between two consecutive requests made to the text embedding model API
  --tex-template TEX_TEMPLATE
                        Path to custom .tex template
```

### Example usage (Gradio Interface)
There's a [gradio](https://www.gradio.app/docs) interface built around this tool, just a quick wrapper around the CLI usage. To run it, execute:
```
python -m aisurveywriter.interface
```

and then you can access it at "localhost:7860" in your browser.

## Flowchart
![](flowchart.webp)
