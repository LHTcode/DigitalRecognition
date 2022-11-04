import torch
from torch import nn
from torch.nn import functional as F

from data.classes import NUMBER_CLASSES

class ConvBNActivation(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=(1, 1),
                 activation_layer=nn.ReLU6(inplace=True),
                 groups=1
                 ):
        super(ConvBNActivation, self).__init__()
        if kernel_size == (1, 1):   # padding = 98
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, groups=groups, bias=False)
        if kernel_size == (3, 3):   # padding = 1
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = activation_layer
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        """
                Args:
                :param expand_ratio: 扩展因子，用于控制bottleneck结构升维比例
                :param inverted_residual_setting: 残差连接模块的配置，[t, c, n, s]，分别为：扩展系数，输出通道数，重复个数，步长。
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.conv = nn.Sequential(
            ConvBNActivation(inp, inp*expand_ratio, kernel_size=(1, 1), stride=1),
            ConvBNActivation(inp*expand_ratio, inp*expand_ratio, kernel_size=(3, 3), stride=stride, groups=inp*expand_ratio),
            nn.Conv2d(inp*expand_ratio, oup, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(oup, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        if self.stride == 1 and self.inp == self.oup:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileNetv2(nn.Module):
    def __init__(self, wid_mul=1.0,
                 resolu_mul=1.0,
                 round_nearest=8,
                 inverted_residual_setting=None,
                 output_channels=NUMBER_CLASSES,
                 ):
        """
        Args:
        :param wid_mul: 宽度乘子，用于控制mobilenet每一层的channel
        :param resolu_mul: 分辨率乘子，用于控制每一层的输入分辨率的大小
        :param round_nearest: Round the number of channels in each layer to be a multiple of this number. Set to 1 to turn off rounding
        :param inverted_residual_setting: 逆残差连接的配置, [t, c, n, s]
        :param output_channels: 输出类别数量
        """
        super(MobileNetv2, self).__init__()
        self.output_channels = output_channels
        if inverted_residual_setting == None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("Please give a right format inverted residual setting.")
        input_channel = 32

        input_channel = self._make_divisible(input_channel*min(1.0, wid_mul), round_nearest)
        self.phase = nn.Sequential()     # 按照 feature_map size 是否改变来划分阶段
        phase_sequential = nn.Sequential()
        phase_count = 0
        inverted_residual_count = 0
        phase_sequential.add_module(f'InvertedResidual_{inverted_residual_count}', ConvBNActivation(1, input_channel, kernel_size=(3, 3), stride=2))
        inverted_residual_count += 1
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(c*min(1.0, wid_mul), round_nearest)
            if s == 2:
                self.phase.add_module(f"phase_{phase_count}", phase_sequential)
                phase_count += 1
                inverted_residual_count = 0
                phase_sequential = nn.Sequential()
            for i in range(n):
                stride = s if i == 0 else 1  # 除了第一层之外的所有层的步长都为1
                phase_sequential.add_module(f'InvertedResidual_{inverted_residual_count}', InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
                inverted_residual_count += 1
        output_channel = self._make_divisible(1280*min(1.0, wid_mul), round_nearest)
        phase_sequential.add_module('last_conv', ConvBNActivation(input_channel, output_channel, kernel_size=(1, 1)))
        phase_sequential.add_module('AdaptiveAvgPool2d', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        phase_sequential.add_module('output_layer', nn.Conv2d(in_channels=output_channel, out_channels=self.output_channels, kernel_size=1, stride=1))
        self.phase.add_module(f"phase_{phase_count}", phase_sequential)

    def forward(self, x):
        output = self.phase(x)
        return output


    def _make_divisible(self, v: float, divisor: int, min_value=None) -> int:
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)   # 保证输出数据能够被divisor整除, 并且不小于min_value
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

def ont_hot_cross_entropy(output, targets):
    N = targets.shape[0]
    # log_softmax
    log_prob = F.log_softmax(output, dim=1)
    # nnloss
    loss = -torch.sum(log_prob * targets) / N
    return loss
