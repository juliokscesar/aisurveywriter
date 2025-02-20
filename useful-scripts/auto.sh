#!/bin/bash

# Ensure the last directory in `pwd` is "aisurveywriter"
if [[ "$(basename "$PWD")" != "aisurveywriter" ]]; then
    echo "Error: This script must be run from the 'aisurveywriter' directory."
    exit 1
fi

# Run the first command
python -m aisurveywriter \
    -b bib/refextract-21papers.bib \
    -fb bib/snowflake-arctic-embed-l-v2.0-bibfaiss/ \
    -ff bib/snowflake-arctic-embed-l-v2.0-allimgfaiss/ \
    -fr -i bib/filteredimgs/ \
    -m gemini-2.0-pro-exp \
    -e Snowflake/snowflake-arctic-embed-l-v2.0 \
    -t huggingface -w 40 refexamples \
    "Langmuir and Langmuir-Blodgett Films" \
    # -s out/generated-struct.yaml \
    # -p out/generated-rev.tex \
    # --no-review --no-figures


# Move files after first run
mv refexamples/OliveiraO2022_PastAndFuture.pdf .
mv out comchu

# Edit YAML file
sed -i 's/google_key/google_key_jcmcs/' credentials.yaml
sed -i 's/google_key_jc12/google_key/' credentials.yaml

# Run the second command
python -m aisurveywriter \
    -b bib/refextract-21papers.bib \
    -fb bib/snowflake-arctic-embed-l-v2.0-bibfaiss/ \
    -ff bib/snowflake-arctic-embed-l-v2.0-nochuimgfaiss/ \
    -fr -i bib/filteredimgs/ \
    -m gemini-2.0-pro-exp \
    -e Snowflake/snowflake-arctic-embed-l-v2.0 \
    -t huggingface -w 40 refexamples \
    "Langmuir and Langmuir-Blodgett Films"

# Move files after second run
mv OliveiraO2022_PastAndFuture.pdf refexamples
mv out semchu

echo "Script execution completed successfully."
