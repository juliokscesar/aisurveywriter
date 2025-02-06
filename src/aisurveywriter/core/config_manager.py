import os

import aisurveywriter.core.file_handler as fh

def abs_join(par: str, path: str) -> str:
    return os.path.abspath(os.path.join(par, path))

class ConfigManager:
    def __init__(
            self,
            credentials_path: str,
            browser_path: str,
            driver_path: str,
            tex_template_path: str,
            prompt_cfg_path: str,
            review_cfg_path: str,
            out_dir: str,
            out_structure_filename: str,
            out_tex_filename: str,
            out_dump_filename: str,
        ):
        # YAML file containing NBLM login info, API keys, etc
        self.credentials_path = credentials_path

        # Selenium configurations
        self.browser_path = browser_path
        self.driver_path = driver_path

        # Templates and configuration files
        self.tex_template_path = tex_template_path
        self.prompt_config_path = prompt_cfg_path
        self.review_config_path = review_cfg_path

        # Output paths
        self.cwd = os.getcwd()
        self.out_dir = abs_join(self.cwd, out_dir)
        self.out_structure_path = abs_join(self.out_dir, out_structure_filename)
        self.out_tex_path = abs_join(self.out_dir, out_tex_filename)
        self.out_dump_path = abs_join(self.out_dir, out_dump_filename)
        self.out_reviewed_tex_path = abs_join(self.out_dir, "reviwed-"+os.path.basename(out_tex_filename))
        self.out_reviewed_dump_path = abs_join(self.out_dir, "reviewed-"+os.path.basename(out_dump_filename))

        # Prompt configurations
        cfg = fh.read_yaml(self.prompt_config_path)
        if self._validate_prompt_cfg(cfg):
            self.paper_subject = cfg["subject"]
            self.prompt_structure = cfg["gen_struct_prompt"]
            self.prompt_write = cfg["write_prompt"]
            self.prompt_ref_extract = cfg["reference_extract_prompt"]
            self.prompt_ref_add = cfg["add_reference_prompt"]
        else:
            raise ValueError(f"Missing configuration items in {self.prompt_config_path}")
        
        cfg = fh.read_yaml(self.review_config_path)
        if self._validate_review_cfg(cfg):
            self.prompt_review = cfg["review_prompt"]
            self.prompt_apply_review = cfg["apply_prompt"]
            self.prompt_tex_review = cfg["tex_review_prompt"]
            self.prompt_bib_review = cfg["bib_review_prompt"]
        else:
            raise ValueError(f"Missing configuration items in {self.review_config_path}")

    @staticmethod
    def from_file(file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Configuration file is inexistent of invalid: {file_path}")
        
        cfg = fh.read_yaml(file_path)
        # make all paths absolute
        for key in cfg:
            if ("path" not in key) or (cfg[key] is None):
                continue
            cfg[key] = os.path.abspath(cfg[key])

        print(cfg)
        return ConfigManager(**cfg)

    def _print(self, *msgs):
        print(f"({self.__class__.__name__})", *msgs)

    def _validate_prompt_cfg(self, cfg: dict) -> bool:
        PROMPT_KEYS = [
            "subject",
            "gen_struct_prompt",
            "write_prompt",
            "reference_extract_prompt",
            "add_reference_prompt",
        ]
        diff = list(set(cfg.keys()) - set(PROMPT_KEYS))
        if len(diff) != 0:
            self._print("Missing keys in prompt config:", ", ".join(diff))
            return False
        return True

    def _validate_review_cfg(self, cfg: dict) -> bool:
        REVIEW_KEYS = [
            "review_prompt",
            "apply_prompt",
            "tex_review_prompt",
            "bib_review_prompt"
        ]
        diff = list(set(cfg.keys()) - set(REVIEW_KEYS))
        if len(diff) != 0:
            self._print("Missing keys in review config:", ", ".join(diff))
            return False
        return True
