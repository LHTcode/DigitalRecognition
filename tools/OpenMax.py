import torch
from torch import utils
from torch import nn
from torch.utils.data import DataLoader
import libmr
from pathlib import Path
from tqdm import tqdm

from model.MobileNetv2 import MobileNetv2
from data.dataset import NumberDataset
from data.classes import NUMBER_CLASSES, CLASSES_NAME
from config.global_config import global_config

@torch.no_grad()
def compute_weibull(model) -> dict:
    model_params_path = Path.cwd().parent / "model_params" / "best_model.pt"
    assert model_params_path.is_file(), FileNotFoundError(f"Path {model_params_path} is wrong!")
    param = torch.load(model_params_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(param['state'])
    model.to(torch.device("cuda:0"))
    model.eval()    # TODO: 先尝试将model置为eval()，不清楚模型去掉了训练时所用到的一些方法会不会有影响

    dataset = NumberDataset(global_config.DATASET_PATH, input_size=(28, 28), classes_num=NUMBER_CLASSES)
    dataloader = DataLoader(dataset, 8, True)

    mu = torch.zeros((NUMBER_CLASSES), dtype=torch.float64, device=global_config.DEVICE)
    # S = torch.empty((NUMBER_CLASSES, 0, NUMBER_CLASSES), dtype=torch.float64, device=global_config.DEVICE)
    # S = {a: torch.empty(0, NUMBER_CLASSES, dtype=torch.float64, device=global_config.DEVICE) for a in CLASSES_NAME.keys()}
    S = {a: torch.empty(0, dtype=torch.float64, device=global_config.DEVICE) for a in CLASSES_NAME.keys()}
    print("Start to compute weibull...")
    data_loop = tqdm(dataloader)
    for imgs, labels in data_loop:
        output = model(imgs).squeeze()
        label = torch.max(labels, dim=1)[-1].squeeze()
        mask = torch.max(output, dim=1)[-1].squeeze() == label
        for idx in label[mask]:
            # S[int(idx.item())+1] = torch.cat((S[int(idx.item())+1], output[mask]))
            S[int(idx.item())+1] = torch.cat((S[int(idx.item())+1], torch.max(output[mask], dim=1)[0]))
            # TODO: 这里存疑：MAV到底是整个激活值向量还是说只是正例的激活值
            # mu[idx] += torch.sum(S[int(idx.item())+1]).item()
            mu[idx] += torch.sum(S[int(idx.item())+1])
    for cls in S.keys():
        mu[cls-1] /= S[cls].shape[0]
    print(mu)
    #=============== EVT Fit ===============#
    eta = 5
    rou_mr = {a: libmr.MR() for a in CLASSES_NAME.keys()}
    rou = {}
    for cls in CLASSES_NAME.keys():
        rou_mr[cls].fit_high(torch.pow(torch.pow(S[cls] - mu[cls-1], 2), 0.5).tolist(), eta)
        rou[cls] = [*rou_mr[cls].get_params()]
    print(rou)
    return rou


if __name__ == '__main__':
    model = MobileNetv2(wid_mul=0.5, output_channels=NUMBER_CLASSES)
    rou = compute_weibull(model)
    #========= save weibull params =========#
    import json
    params_save_path = Path.cwd().parent / 'config' / 'config.json'
    with open(str(params_save_path), 'w') as f:
        json.dump(rou, f)
        print(f"Weibull parameters has been written at {params_save_path}!")