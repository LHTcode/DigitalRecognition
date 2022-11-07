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
def compute_weibull(model, input_size) -> (dict, list):
    model_params_path = Path.cwd().parent / "model_params" / (global_config.MODEL_NAME + ".pt")
    assert model_params_path.is_file(), FileNotFoundError(f"Path {model_params_path} is wrong!")
    param = torch.load(model_params_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(param['state'])
    model.to(torch.device("cuda:0"))
    model.eval()    # TODO: 先尝试将model置为eval()，不清楚模型去掉了训练时所用到的一些方法会不会有影响

    dataset = NumberDataset(global_config.DATASET_PATH, input_size=input_size, classes_num=NUMBER_CLASSES)
    dataloader = DataLoader(dataset, 8, True)

    # mu = torch.zeros((NUMBER_CLASSES), dtype=torch.float64, device=global_config.DEVICE)
    mu = torch.zeros((NUMBER_CLASSES, NUMBER_CLASSES), dtype=torch.float64, device=global_config.DEVICE)
    # S = torch.empty((NUMBER_CLASSES, 0, NUMBER_CLASSES), dtype=torch.float64, device=global_config.DEVICE)
    S = {a: torch.empty(0, NUMBER_CLASSES, dtype=torch.float64, device=global_config.DEVICE) for a in CLASSES_NAME.keys()}
    # S = {a: torch.empty(0, dtype=torch.float64, device=global_config.DEVICE) for a in CLASSES_NAME.keys()}

    print(f"Start to compute {global_config.MODEL_NAME} weibull...")
    data_loop = tqdm(dataloader)
    for imgs, labels in data_loop:
        output = model(imgs).squeeze()
        label = torch.max(labels, dim=1)[-1].squeeze()
        max_output = torch.max(output, dim=0) if len(output.shape) == 1 else torch.max(output, dim=1)[-1]
        mask = max_output == label
        for cls, cls_out in zip(label[mask], output[mask]):
            S[int(cls.item()) + 1] = torch.cat((S[int(cls.item())+1], cls_out.unsqueeze(0)))
    for cls in S.keys():
        mu[cls-1] = torch.sum(S[cls], dim=0) / S[cls].shape[0]
    #=============== EVT Fit ===============#
    eta = 10
    rou_mr = {a: libmr.MR() for a in CLASSES_NAME.keys()}
    rou = {}
    for cls in CLASSES_NAME.keys():
        # print(torch.pow(torch.sum(torch.pow(S[cls] - mu[cls-1], 2), dim=0), 0.5))
        rou_mr[cls].fit_high(torch.pow(torch.sum(torch.pow(S[cls] - mu[cls-1], 2), dim=1), 0.5).tolist(), eta)
        rou[cls] = [*rou_mr[cls].get_params()]
    mu = mu.tolist()
    print(f"Class weibull params:\n{rou}\n"
          f"MAV: \n{mu}\n")
    return (rou, mu)

def save_weibull_params(rou, mu) -> None:
    import json
    params_save_path = Path.cwd().parent / 'config' / 'config.json'
    with open(str(params_save_path), 'w') as f:
        json.dump({"weibull": rou, "mu": mu}, f)
        print(f"Weibull parameters has been written at {params_save_path}!")
    return

if __name__ == '__main__':
    model = MobileNetv2(wid_mul=1, output_channels=NUMBER_CLASSES).to("cuda:0")
    rou, mu = compute_weibull(model, input_size=global_config.INPUT_SIZE)
    #========= save weibull params =========#
    save_weibull_params(rou, mu)