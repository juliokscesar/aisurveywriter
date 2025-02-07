from typing import Union, List
from time import sleep, time
import os
from pathlib import Path
import yaml
import undetected_chromedriver as uc
from fake_useragent import UserAgent
import requests
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bibdatabase import BibDatabase
import re
import html

def time_func(func, *args, **kwargs):
    start = time()
    ret = func(*args, **kwargs)
    elapsed = time() - start
    return int(elapsed), ret

def diff_keys(keys: list, d: dict):
    diff = set(keys - list(d.keys()))
    return None if len(diff) == 0 else diff

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


def validate_credentials(credentials: dict):
    REQUIRED_KEYS = [
        "nblm_email",
        "nblm_password",
    ]

    # First check if required keys are present
    diff = [i for i in REQUIRED_KEYS if i not in list(credentials.keys())]
    if len(diff) > 0:
        raise ValueError(f"Credentials are missing keys: {", ".join(diff)}")

    # TODO: check for api keys

def search_crossref(title, author):
    """
    Search CrossRef API using title and author to retrieve the DOI.
    """
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "query.author": author, "rows": 1}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        items = data.get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    return None

def get_bibtext(doi, cache={}):
    """
    Use DOI Content Negotiation to retrieve a string with the bibtex entry.
    """
    if doi in cache:
        return cache[doi]
    url = f'https://doi.org/{doi}'
    headers = {'Accept': 'application/x-bibtex'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        bibtext = response.text
        cache[doi] = bibtext
        return bibtext
    return None

def get_abstract(doi):
    """
    Retrieve the abstract of a paper using the CrossRef API.
    """
    url = f"http://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Erro ao acessar API: {response.status_code}")
    
    data = response.text
    
    match = re.search(r'(?<="abstract":").*?(?=","DOI")', data)
    
    if not match:
        return None

    abstract = match.group(0)
    abstract = re.sub(r'<jats:p>|<\/jats:p>', '\n', abstract)
    abstract = re.sub(r'<[^>]*>', ' ', abstract)
    abstract = html.unescape(abstract)
    
    return abstract

def get_bibtex_entry(title, author, bibtext_cache={}):
    """
    Retrieve BibTeX entry using title and author.
    """
    doi = search_crossref(title, author)
    if not doi:
        print(f"DOI not found for given title and author: {title}. {author}")
        return None
    
    bibtext = get_bibtext(doi, cache=bibtext_cache)
    if not bibtext:
        print("BibTeX entry not found.")
        return None
    
    parser = BibTexParser()
    parser.ignore_nonstandard_types = False
    bibdb = bibtexparser.loads(bibtext, parser)
    entry, = bibdb.entries
    entry['link'] = f'https://doi.org/{doi}'
    
    if 'author' in entry:
        entry['author'] = ' and '.join(entry['author'].rstrip(';').split('; '))
    
    entry['ID'] = doi.split('/')[-1]
    
    # Retrieve and add abstract
    abstract = get_abstract(doi)
    if abstract:
        entry['abstract'] = abstract
    
    return entry

def bib_entries_to_str(entries):
    """
    Pass a list of bibtexparser entries and return a bibtex formatted string.
    """
    db = BibDatabase()
    db.entries = entries
    return bibtexparser.dumps(db)