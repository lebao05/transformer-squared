import os

from .base import BaseModel


class Llama32Instruct1B(BaseModel):
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        self.dec_param_file_n = "llama32_1b_decomposed_params.pt"

    def get_model_id(self):
        return self.model_id

    def get_model_name(self):
        return self.model_id.split("/")[1]

    def get_param_file(self, param_folder_path=""):
        return os.path.join(param_folder_path, self.dec_param_file_n)
