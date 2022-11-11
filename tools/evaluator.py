import torch
from tqdm import tqdm
from torch import nn

from config.global_config import global_config

@torch.no_grad()
def evaluate(model, test_dataloader, model_save_path):
    import numpy as np
    print(f"Testing {model.name}...\n")
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device(global_config.DEVICE))["state"])
    model.eval()
    activate_func = nn.Softmax(dim=1)

    precs = {a: np.empty(0) for a in global_config.CLASSES_NAME.keys()}
    recs = {a: np.empty(0) for a in global_config.CLASSES_NAME.keys()}
    threshold_loop = tqdm(np.arange(0.4, 1.0, 0.05))
    for threshold in threshold_loop:
        tp = {a: 0 for a in global_config.CLASSES_NAME.keys()}
        fp = {a: 0 for a in global_config.CLASSES_NAME.keys()}
        test_loop = tqdm(test_dataloader)
        for imgs, labels in test_loop:
            outputs = model(imgs)
            outputs = activate_func(outputs)
            pred = torch.max(outputs, dim=1)
            for idx, label in enumerate(labels):
                label = torch.max(label, dim=0)[-1].item()
                if pred[-1][idx] == label and pred[0][idx] >= threshold:
                    tp[label+1] += 1
                else:
                    fp[label+1] += 1
        prec, rec = test_dataloader.dataset.eval(tp, fp)
        print(f"Threshold {threshold}, prec is:{prec}, rec is: {rec}")
        for cls in global_config.CLASSES_NAME.keys():
            precs[cls] = np.append(precs[cls], prec[cls])
            recs[cls] = np.append(recs[cls], rec[cls])

    import numpy as np
    ap = {a: 0.0 for a in global_config.CLASSES_NAME.keys()}
    map = 0.0
    for cls in global_config.CLASSES_NAME.keys():
        for t in np.arange(0, 1.1, 0.1):  # 11点插值法：选取当recall>=0.5~1.0间隔为0.05共11个点时的precision最大值的平均作为ap
            if np.sum(recs[cls] >= t) == 0:
                p = 0
            else:
                p = np.max(precs[cls][recs[cls] >= t])
            ap[cls] += p / 11.0
        map += ap[cls]
    map /= global_config.CLASSES_NUM
    for cls in global_config.CLASSES_NAME:
        print(f"Class {cls} ap: {ap[cls]}")
    print(f"MAP: {map}")
