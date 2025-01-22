import os
from .file_handler import FileHandler

def abs_join(par: str, path: str) -> str:
    return os.path.abspath(os.path.join(par, path))

class ConfigManager:
    def __init__(
            self,
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
        cfg = FileHandler.read_yaml(self.prompt_config_path)
        if self._validate_prompt_cfg(cfg):
            self.paper_subject = cfg["subject"]
            self.prompt_structure = cfg["gen_struct_prompt"]
            self.prompt_response_format = cfg["response_format"]
            self.prompt_write = cfg["write_prompt"]
            self.prompt_input_variables = cfg["prompt_input_variables"]
        else:
            raise ValueError(f"Missing configuration items in {self.prompt_config_path}")
        
        cfg = FileHandler.read_yaml(self.review_config_path)
        if self._validate_review_cfg(cfg):
            self.review_nblm_prompt = cfg["nblm_point_prompt"]
            self.review_improve_prompt = cfg["improve_prompt"]
        else:
            raise ValueError(f"Missing configuration items in {self.review_config_path}")


    def _print(self, *msgs):
        print(f"({self.__class__.__name__})", *msgs)

    def _validate_prompt_cfg(self, cfg: dict) -> bool:
        PROMPT_KEYS = [
            "subject",
            "gen_struct_prompt",
            "response_format",
            "write_prompt",
            "prompt_input_variables",
        ]
        diff = list(set(cfg.keys()) - set(PROMPT_KEYS))
        if len(diff) != 0:
            self._print("Missing keys in prompt config:", ", ".join(diff))
            return False
        return True

    def _validate_review_cfg(self, cfg: dict) -> bool:
        REVIEW_KEYS = [
            "nblm_point_prompt",
            "improve_prompt",
        ]
        diff = list(set(cfg.keys()) - set(REVIEW_KEYS))
        if len(diff) != 0:
            self._print("Missing keys in review config:", ", ".join(diff))
            return False
        return True
