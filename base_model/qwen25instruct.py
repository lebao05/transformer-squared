import os

from .base import BaseModel


class Qwen25Instruct15B(BaseModel):
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        self.dec_param_file_n = "qwen25_1b5_decomposed_params.pt"

    def get_model_id(self):
        return self.model_id

    def get_model_name(self):
        return self.model_id.split("/")[1]

    def get_param_file(self, param_folder_path=""):
        return os.path.join(param_folder_path, self.dec_param_file_n)
