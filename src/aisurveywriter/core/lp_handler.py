from typing import Optional
from pydantic import BaseModel
import layoutparser as lp
import os
import copy
import requests

def load_lp_model(config_path: str = 'lp://<dataset_name>/<model_name>/config',
                  extra_config=None):

    config_path_split = config_path.split('/')
    dataset_name = config_path_split[-3]
    model_name = config_path_split[-2]

    # get the URLs from the MODEL_CATALOG and the CONFIG_CATALOG 
    # (global variables .../layoutparser/models/detectron2/catalog.py)
    model_url = lp.models.detectron2.catalog.MODEL_CATALOG[dataset_name][model_name]
    config_url = lp.models.detectron2.catalog.CONFIG_CATALOG[dataset_name][model_name]

    # override folder destination:
    if 'model' not in os.listdir():
        os.mkdir('model')

    config_file_path, model_file_path = None, None

    for url in [model_url, config_url]:
        filename = url.split('/')[-1].split('?')[0]
        save_to_path = f"model/" + filename
        if 'config' in filename:
            config_file_path = copy.deepcopy(save_to_path)
        if 'model_final' in filename:
            model_file_path = copy.deepcopy(save_to_path)

        # skip if file exist in path
        if filename in os.listdir("model"):
            continue
        # Download file from URL
        r = requests.get(url, stream=True, headers={'user-agent': 'Wget/1.16 (linux-gnu)'})

        with open(save_to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)

    # load the label map
    label_map = lp.models.detectron2.catalog.LABEL_MAP_CATALOG[dataset_name]

    return lp.models.Detectron2LayoutModel(
        config_path=config_file_path,
        model_path=model_file_path,
        label_map=label_map,
        extra_config=extra_config,
    )


def init_lp_agents(config: str,
                   score_threshold: float = 0.8,
                   tesseract_exectuable: str ="tesseract") -> tuple[lp.models.Detectron2LayoutModel, lp.TesseractAgent]:
    """
    Initialize LayoutParser Detectron2 and OCR agents
    
    Parameters:
        config (str): path to layoutparser detectron2 model config.
        score_threshold (float): score threshold configuration for detectron2 model
        tesseract_executable (str): OCR Tesseract execution command/path

    Returns:
        det2_model: LayoutParser Detectron2 model
        ocr_agent: LayoutParser OCR Tesseract Agent
    """
    det2_model = load_lp_model(config_path=config, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_threshold])
    ocr_agent = lp.TesseractAgent.with_tesseract_executable(tesseract_exectuable)
    
    return (det2_model, ocr_agent)

class LayoutParserSettings(BaseModel):
    config_path: Optional[str] = ""
    score_threshold: float = 0.8
    tesseract_executable: str = "tesseract"
    

class LayoutParserAgents:
    settings: LayoutParserSettings
    
    model: lp.models.Detectron2LayoutModel
    ocr: lp.TesseractAgent
    
    def __init__(self, config: str, score_threshold: float = 0.8, tesseract_executable: str = "tesseract"):
        self.settings = LayoutParserSettings(config_path=config, score_threshold=score_threshold, 
                                             tesseract_executable=tesseract_executable)
        
        self.model, self.ocr = init_lp_agents(config, score_threshold, tesseract_executable)
        self.tesseract_executable = tesseract_executable
        