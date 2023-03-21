from pathlib import Path
from torch import cuda

class GlobalConfig():
    def __init__(self):
        self.__dataset_path = "/media/lihanting/Elements/visual-group/ArmorId"
        # self.__dataset_path = "/home/lihanting/ArmorId"
        self.__device = "cuda:0" if cuda.is_available() else "cpu"
        self.__model_name = "MobileNetv2"
        self.__classes_num = 4
        self.__classes_name = {1: 'hero', 2: 'engineer', 3: 'standard3', 4: 'standard4'}  # remove 'base' cls
        self.__input_size = (128, 128)

    @property
    def DATASET_PATH(self):
        assert Path(self.__dataset_path).exists(), NotADirectoryError(f"DATASET_PATH {self.__dataset_path} no found!")
        return self.__dataset_path

    @property
    def DEVICE(self):
        return self.__device

    @property
    def MODEL_NAME(self):
        return self.__model_name

    @property
    def CLASSES_NUM(self):
        return self.__classes_num

    @property
    def INPUT_SIZE(self):
        return self.__input_size

    @property
    def CLASSES_NAME(self):
        return self.__classes_name

global_config = GlobalConfig()