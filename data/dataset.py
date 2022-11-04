import cv2
import torch
from torch import nn
from pathlib import Path
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F

from config.global_config import global_config

def compute_dataset_size(data_path) -> (int, dict, dict):
    data_num = 0
    data_dict = {}
    file_len = []
    subdir_list = []
    choose_data = []
    choose_dict = {}
    for _, subdir_name, file_name in os.walk(data_path):
        if subdir_name is not None:
            subdir_list.extend(subdir_name)
        if len(file_name) != 0:
            file_len.append(file_name)
            data_num += len(file_name)
            choose_data.append(data_num)
    data_dict.update({a: b for a, b in zip(subdir_list, file_len)})
    choose_dict.update({a: b for a, b in zip(choose_data, subdir_list)})
    return data_num, data_dict, choose_dict

class NumberDataset(Dataset):
    def __init__(self, root_path, input_size: tuple, classes_num, phase="train", transform=None):
        self.root_path = root_path
        self.input_size = input_size
        self.phase = phase
        self.transform = transform
        self.dataset_size, self.data_dict, self.choose_dict = compute_dataset_size(str(Path(self.root_path) / Path(self.phase)))
        self.classes_num = classes_num
        self.device = global_config.DEVICE

    def __len__(self):
        return self.dataset_size

    def preprocess(self, img):
        scale_w = img.shape[1] / self.input_size[0]
        scale_h = img.shape[2] / self.input_size[1]
        if scale_w != 1 or scale_h != 1:
            img = nn.functional.interpolate(img, size=self.input_size, mode="bilinear", align_corners=False)  # 一个用于上采样和下采样的插值操作
        return img

    def __getitem__(self, item):
        choose_list = list(self.choose_dict.keys())
        choose_mask = item <= np.array(choose_list)
        label = ''
        index = 0
        for id, lb in enumerate(choose_mask):
            if lb:
                label = self.choose_dict[choose_list[id]]
                index = (item - choose_list[id - 1] - 1) if id != 0 else item - 1
                break
        img_name = self.data_dict[label][index]
        img_path = Path(self.root_path) / Path(self.phase) / Path(label) / Path(img_name)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        width, height = int(img.shape[1] * r), int(img.shape[0] * r)
        resize_img = cv2.resize(
            img,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        padding_img = np.ones((*self.input_size, 1), dtype=resize_img.dtype) * 114
        padding_img[:resize_img.shape[0], :resize_img.shape[1], 0] = resize_img

        # padding_img = cv2.cvtColor(padding_img, cv2.COLOR_BGR2RGB, padding_img)
        padding_img = torch.from_numpy(padding_img).permute(2, 0, 1)
        label = F.one_hot(torch.tensor(int(label)-1), num_classes=self.classes_num).to(self.device)
        return padding_img.to(self.device).to(torch.float32), label



