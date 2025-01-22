# AI Survey Writer

An approach to automating the process of writing Survey Papers using state-of-the-art LLMs.

CLI help:
```
usage: python -m aisurveywriter [-h] [--llm {openai,google}] [--llm-model LLM_MODEL] [--summarize] [--faiss] [-c CONFIG] references_dir

positional arguments:
  references_dir        Path to directory containg all PDF references

options:
  -h, --help            show this help message and exit
  --llm {openai,google}, -l {openai,google}
                        Specify LLM to use. Either 'google' (gemini-1.5-pro by default) or 'openai' (o1 by default)    
  --llm-model LLM_MODEL, -m LLM_MODEL
                        Specific LLM model to use.
  --summarize           Use a summary of references instead of their whole content.
  --faiss               Use FAISS vector store to retrieve information from references instead of their whole content. 
If this and 'summarize' are enabled, this will be ignored.
  -c CONFIG, --config CONFIG
                        YAML file containg your configuration parameters
```
