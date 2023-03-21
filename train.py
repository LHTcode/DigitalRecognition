import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import copy

from model.MobileNetv2 import MobileNetv2, ont_hot_cross_entropy
from model.LeNet5 import LeNet5
from model.Linear import Linear
from config.global_config import global_config
from data.dataset import NumberDataset
from tools import evaluator

@torch.no_grad()
def test(model, test_dataloader, model_save_path, writer, ep):
    print(f"Testing {model.name}...\n")
    test_loop = tqdm(test_dataloader)
    correct_num = 0
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device(global_config.DEVICE))["state"])
    model.eval()
    activate_func = nn.Softmax()
    for img, label in test_loop:
        output = model(img)
        output = activate_func(output)
        correct_num += torch.sum(torch.max(output, dim=1)[-1] == torch.max(label, dim=1)[-1])
    precision = correct_num / len(test_dataloader.dataset)
    print(f"Test precision is {precision}\n")
    writer.add_scalar("Test precision", precision, ep)

@torch.enable_grad()
def train(model, dataloader, test_dataloader, epoch, writer):
    model.train()
    test_model = copy.deepcopy(model)
    test_model.to(torch.device(global_config.DEVICE))
    model_save_path = Path.cwd() / "model_params"
    if not model_save_path.is_dir():
        Path.mkdir(model_save_path)
    model_save_path = model_save_path / (global_config.MODEL_NAME + '.pt')

    lr = 0.05
    optim = SGD(model.parameters(), lr, momentum=0.5, nesterov=True)
    activate_func = nn.Softmax()
    epoch_loop = tqdm(range(epoch), total=epoch)
    train_count = 1  # 用于计算runtime_loss
    for ep in epoch_loop:
        if (ep+1) % 5 == 0:
            optim = SGD(model.parameters(), 0.1)
        if (ep+1) % 10 == 0:
            optim = SGD(model.parameters(), 0.2)
        if (ep + 1) % 15 == 0:
            optim = SGD(model.parameters(), 0.01)
        if (ep + 1) % 20 == 0:
            optim = SGD(model.parameters(), 0.005)
        if (ep + 1) % 25 == 0:
            optim = SGD(model.parameters(), 0.0025)

            # dataloader.dataset.use_argumentation = True
            # optim = SGD(model.parameters(), lr * 1.125)
        # if ep == 20:
        #     optim = SGD(model.parameters(), lr * 1.125)
        # if ep == 25:
        #     optim = SGD(model.parameters(), lr * 0.25)

        # if ep == 20:
        #     optim = SGD(model.parameters(), lr * 1.25)
        runtime_loss = 0
        data_loop = tqdm(dataloader)
        correct_num = 0
        for img, label in data_loop:
            optim.zero_grad()
            output = model(img)
            output = activate_func(output)
            output = output.squeeze()
            loss = ont_hot_cross_entropy(output, label)
            loss.backward()
            optim.step()
            runtime_loss += loss
            if train_count % 3 == 0:
                print("runtime_loss= {:s}\n".format(str(runtime_loss / 3)))
                writer.add_scalar('Loss', runtime_loss, train_count)
                runtime_loss = 0
            correct_num += torch.sum(torch.max(output, dim=1)[-1] == torch.max(label, dim=1)[-1])
            train_count += 1
        precision = correct_num / len(dataloader.dataset)

        print(f"precision is: {precision}\n")
        writer.add_scalar('Precision', precision, ep)
        save_state_dict = {"state": model.state_dict()}
        torch.save(save_state_dict, str(model_save_path))
        print(f"Successfully save {model.name} model_params state.\n")
        if (ep+1) % 100 == 0:
            test(test_model, test_dataloader, model_save_path, writer, ep)

if __name__ == '__main__':
    root_path = global_config.DATASET_PATH
    # test_root_path = "/home/lihanting/ArmorId"
    test_root_path = "/media/lihanting/Elements/visual-group/ArmorId"

    model = MobileNetv2(wid_mul=0.6, output_channels=global_config.CLASSES_NUM).to(global_config.DEVICE)
    # model = LeNet5(8, True).to(global_config.DEVICE)
    # model = Linear().to(global_config.DEVICE)

    dataset = NumberDataset(root_path, input_size=global_config.INPUT_SIZE, classes_num=global_config.CLASSES_NUM, use_argumentation=True)
    test_dataset = NumberDataset(test_root_path, classes_num=global_config.CLASSES_NUM, input_size=global_config.INPUT_SIZE, phase='test', use_argumentation=False)

    training_config = {
        "batch_size": 128,
        "epoch": 30
    }
    dataloader = DataLoader(dataset, training_config['batch_size'], True)
    test_dataloader = DataLoader(test_dataset, 1, True)
    writer = SummaryWriter()
    train(model, dataloader, test_dataloader, training_config['epoch'], writer)
    evaluator.evaluate(model, test_dataloader, "/home/lihanting/PycharmProjects/DigitalRecognition/model_params/" + global_config.MODEL_NAME + '.pt')
    # test(model, dataloader, "/home/lihanting/PycharmProjects/DigitalRecognition/model_params/" + global_config.MODEL_NAME + '.pt', writer, 0)