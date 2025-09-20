import torch
import torch.nn as nn


class MLKCM(nn.Module):
    """
    Multi-Level Kernel Convolution Module (MLKCM)
    动态适配输入/输出通道，支持 YOLOv8 自定义结构
    """
    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1  # 默认保持通道数不变

        # 多尺度卷积分支
        self.branch1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.branch3 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.branch5 = nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2)

        # 融合层
        self.fuse = nn.Conv2d(c2 * 3, c2, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)

        y = torch.cat((b1, b3, b5), dim=1)  # [B, 3*c2, H, W]
        y = self.fuse(y)                    # [B, c2, H, W]
        y = self.bn(y)
        return self.act(y)
