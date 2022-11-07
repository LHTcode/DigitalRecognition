import torch
import torchvision
from torch import nn

class LeNet(nn.Module):
    def __init__(self, classes_num):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 3),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 26, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(26 * 30 * 30, 120),
            # nn.Linear(1 * 28 * 28, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes_num)
        )
        self.name = "LeNet"
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        # output = self.fc(img.view(img.shape[0], -1))
        return output