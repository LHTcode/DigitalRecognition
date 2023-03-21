import torch
import torchvision
from torch import nn
from torch import functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(LeNet5, self).__init__()
        self.name = "LeNet5"
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale: # 可以适用单通道和三通道的图像
            in_channels = 1
        else:
            in_channels = 3

        # 卷积神经网络
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=2)   # 原始的模型使用的是 平均池化

        )
        self.relu = nn.Sequential(
            nn.ReLU6()
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(784, 120),  # 这里把第三个卷积当作是全连接层了
            # nn.Linear(784, 120),
            nn.ReLU6(),
            nn.Linear(120, 84),
            nn.ReLU6(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        # x = self.features(x) # 输出 16*5*5 特征图
        x = self.relu(x)
        x = torch.flatten(x, 1) # 展平 （1， 16*5*5/）
        logits = self.classifier(x) # 输出 10
        # probas = F.softmax(logits, dim=1)
        return logits