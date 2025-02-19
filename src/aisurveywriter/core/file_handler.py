from typing import List, Union
import yaml
import re
import os

def read_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def write_yaml(data: dict, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data)


def read_credentials(file_path: str):
    if not os.path.isfile(file_path):
        raise FileExistsError(f"The file {file_path!r} does not exist and is required")
    with open(file_path, "r", encoding="utf-8") as f:
        credentials = yaml.safe_load(f)
    
    return credentials
