import torch.nn as nn
from tools.dconv import DyConv2D


class DSPM(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, num_experts, stride, padding=None):
        super(DSPM, self).__init__()
        if not padding:
            padding = (0, 0)
        self.conv1 = DyConv2D(C_in, C_out, kernel_size=kernel_size, num_experts=num_experts, stride=stride,
                              padding=padding)
        self.conv2 = DyConv2D(C_in, C_out, kernel_size=kernel_size, num_experts=num_experts, stride=stride,
                              padding=padding)
        self.sw = SwitchAtt(C_in, C_out, kernel_size, stride, padding)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        mask = self.sw(x)
        return x1 * mask + x2 * (1-mask)


class SwitchAtt(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SwitchAtt, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(C_out),
            nn.ELU()
        )
        self.mask = nn.Sequential(
            nn.Conv2d(C_out, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.trans(x)
        mask = self.mask(x1)
        return mask