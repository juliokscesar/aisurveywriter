from typing import List, Optional
import os
import queue
import re
import yaml

from .survey_context import SurveyContext

def setup_credentials(credentials_path: str):
    with open(credentials_path, "r", encoding="utf-8") as f:
        credentials = yaml.safe_load(f)
    if credentials["google_key"]:
        os.environ["GOOGLE_API_KEY"] = credentials["google_key"]
    if credentials["openai_key"]:
        os.environ["OPENAI_API_KEY"] = credentials["openai_key"]

def start_survey_pipeline(config_path: str, credentials_path: str):
    setup_credentials(credentials_path)
    
    survey = SurveyContext(config_path)
    return survey.generate()
