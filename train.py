import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import copy

from model.MobileNetv2 import MobileNetv2, ont_hot_cross_entropy
from model.LeNet import LeNet
from config.global_config import global_config
from data.dataset import NumberDataset
from data.classes import NUMBER_CLASSES

@torch.no_grad()
def test(model, test_dataloader, model_save_path, writer, ep):
    print(f"Testing {model.name}...\n")
    test_loop = tqdm(test_dataloader)
    correct_num = 0
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device(global_config.DEVICE))["state"])
    model.eval()

    for img, label in test_loop:
        output = model(img)
        correct_num += torch.sum(torch.max(output, dim=1)[-1] == torch.max(label, dim=1)[-1])
    precision = correct_num / len(test_dataloader.dataset)
    print(f"Test precision is {precision}\n")
    writer.add_scalar("Test precision", precision, ep)

@torch.enable_grad()
def train(model, dataloader, test_dataloader, epoch, writer):
    test_model = copy.deepcopy(model)
    test_model.to(torch.device(global_config.DEVICE))

    model_save_path = Path.cwd() / "model_params"
    if not model_save_path.is_dir():
        Path.mkdir(model_save_path)
    model_save_path = model_save_path / (global_config.MODEL_NAME + '.pt')
    optim = SGD(model.parameters(), 0.1)

    epoch_loop = tqdm(range(epoch), total=epoch)
    train_count = 1  # 用于计算runtime_loss
    for ep in epoch_loop:
        runtime_loss = 0
        data_loop = tqdm(dataloader)
        correct_num = 0
        for img, label in data_loop:
            optim.zero_grad()
            output = model(img)
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
        if (ep+1) % 1 == 0:
            test(test_model, test_dataloader, model_save_path, writer, ep)

if __name__ == '__main__':
    root_path = global_config.DATASET_PATH
    model = MobileNetv2(wid_mul=1, output_channels=NUMBER_CLASSES).to(global_config.DEVICE)
    # model = LeNet(classes_num=NUMBER_CLASSES).to(global_config.DEVICE)
    dataset = NumberDataset(root_path, input_size=global_config.INPUT_SIZE, classes_num=NUMBER_CLASSES)
    test_dataset = NumberDataset(root_path, classes_num=global_config.CLASSES_NUM, input_size=global_config.INPUT_SIZE, phase="test")

    training_config = {
        "batch_size": 80,
        "epoch": 10
    }
    dataloader = DataLoader(dataset, training_config['batch_size'], True)
    test_dataloader = DataLoader(test_dataset, 1, True)
    writer = SummaryWriter()
    # train(model, dataloader, test_dataloader, training_config['epoch'], writer)
    test(model, test_dataloader, "/home/lihanting/PycharmProjects/DigitalRecognition/model_params/" + global_config.MODEL_NAME + '.pt', writer, 0)