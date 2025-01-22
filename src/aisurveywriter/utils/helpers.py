from typing import Union, List
from time import sleep
import os
from pathlib import Path
import yaml
import undetected_chromedriver as uc
from fake_useragent import UserAgent

def countdown_print(msg: str, sec: int):
    for t in range(1, sec+1):
        sleep(1)
        print(f"{msg} {t} s", end='\r')

def init_driver(browser_path: Union[str,None] = None, driver_path: Union[str,None] = None) -> uc.Chrome:
    op = uc.ChromeOptions()
    op.add_argument(f"user-agent={UserAgent.random}")
    op.add_argument("user-data-dir=./")
    op.add_experimental_option("detach", True)
    op.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = uc.Chrome(
        chrome_options=op,
        browser_executable_path=browser_path,
        driver_executable_path=driver_path
    )
    return driver

def file_ext(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return ext

def sort_stem(item):
    s = Path(item).stem
    return int(s) if s.isnumeric() else s

def get_all_files_from_paths(*args, skip_ext: List[str] = None, stem_sort=False):
    files = []
    for path in args:
        if os.path.isfile(path):
            if skip_ext is not None:
                if file_ext(path) in skip_ext:
                    continue
            files.append(path)

        elif os.path.isdir(path):
            for (root, _, filenames) in os.walk(path):
                if skip_ext is not None:
                    files.extend([os.path.join(root, file) for file in filenames if file_ext(file) not in skip_ext])
                else:
                    files.extend([os.path.join(root, file) for file in filenames])
        
        else:
            raise RuntimeError(f"{path} is an invalid file source")
    if stem_sort:
        files = sorted(files, key=sort_stem)
    return files


def validate_credentials(llmtype: str, credentials: dict):
    REQUIRED_KEYS = [
        "nblm_email",
        "nblm_password",
    ]
    LLM_API_KEYS = {
        "google": "google_key",
        "openai": "openai_key",
    }

    # First check if required keys are present
    diff = [i for i in REQUIRED_KEYS if i not in list(credentials.keys())]
    if len(diff) > 0:
        raise ValueError(f"Credentials are missing keys: {", ".join(diff)}")

    # Now check if the llm type is present
    llm = llmtype.strip().lower()
    if llm not in LLM_API_KEYS:
        raise ValueError(f"No API key provided for llm of type {llm!r}")

