from pathlib import Path
from torch import cuda

class GlobalConfig():
    def __init__(self):
        self.__dataset_path = "/media/lihanting/Elements/visual-group/ArmorId"
        self.__device = "cuda:0" if cuda.is_available() else "cpu"
    @property
    def DATASET_PATH(self):
        assert Path(self.__dataset_path).exists(), NotADirectoryError(f"DATASET_PATH {self.__dataset_path} no found!")
        return self.__dataset_path
    @property
    def DEVICE(self):
        return self.__device

global_config = GlobalConfig()