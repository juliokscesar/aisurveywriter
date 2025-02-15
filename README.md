# AI Survey Writer

An approach to automating the process of writing Survey Papers using state-of-the-art LLMs.

## Usage:

The tool can be used directly from the terminal. First, the installation commands are (a separate virtual environment is strongly recommended):

```
git clone https://github.com/juliokscesar/aisurveywriter.git
cd aisurveywriter
pip instal -e .
```

And then you can run it using 
```
python -m aisurveywriter
```

CLI help:
```
usage: python -m aisurveywriter [-h] [--save-dir SAVE_DIR] [--llm {openai,google}]
                   [--llm-model LLM_MODEL] [-c CONFIG] [--structure STRUCTURE]
                   [--paper PAPER] [--no-review] [--nblm]
                   [--embed-model EMBED_MODEL] [--embed-type EMBED_TYPE]
                   [--bibdb BIBDB] [--faissbib FAISSBIB] [--images IMAGES]
                   [--faissfig FAISSFIG] [--cooldown COOLDOWN]
                   [--embed-cooldown EMBED_COOLDOWN]
                   references_dir subject

positional arguments:
  references_dir        Path to directory containg all PDF references
  subject               Main subject of the survey. Can be the Title too

options:
  -h, --help            show this help message and exit
  --save-dir SAVE_DIR   Path to output directory
  --llm {openai,google}, -l {openai,google}
                        Specify LLM to use. Either 'google' (gemini-1.5-pro by
                        default) or 'openai' (o1 by default)
  --llm-model LLM_MODEL, -m LLM_MODEL
                        Specific LLM model to use.
  -c CONFIG, --config CONFIG
                        YAML file containg your configuration parameters
  --structure STRUCTURE, -s STRUCTURE
                        YAML file containing the structure to use. If provided,
                        this will skip the structure generation process.
  --paper PAPER, -p PAPER
                        Path to .TEX paper to use. If provided, won't write one
                        from the structure, and will skip directly to reviewing it
                        (unless --no-review) is provided
  --no-review           Skip content/writing review step
  --nblm                Use NotebookLM for generating the structure
  --embed-model EMBED_MODEL, -e EMBED_MODEL
                        Text embedding model name
  --embed-type EMBED_TYPE, -t EMBED_TYPE
                        Text embedding model type (google, openai, huggingface)
  --bibdb BIBDB, -b BIBDB
                        Path to .bib database to use. If none is provided, one will
                        be generated by extracting every reference across all PDFs
  --faissbib FAISSBIB, -fb FAISSBIB
                        Path to FAISS vector store of the .bib databse. If none is
                        provided, one will be generated
  --images IMAGES, -i IMAGES
                        Path to all images extracted from the PDFs. If none is
                        provided, all images will be extracted and saved to a
                        temporary folder
  --faissfig FAISSFIG, -ff FAISSFIG
                        Path to FAISS vector store containing the metadata (id,
                        path and description) for every image. If none is provided,
                        one will be created
  --cooldown COOLDOWN, -w COOLDOWN
                        Cooldown between two consecutive request made to the LLM
                        API
  --embed-cooldown EMBED_COOLDOWN
                        Cooldown between two consecutive requests made to the text
                        embedding model API
```

## Flowchart
![](survey-flowchart-crop.webp)
