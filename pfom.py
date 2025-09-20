import torch
import torch.nn as nn


class PFOM(nn.Module):
    """
    Parallel Feature Optimization Module (PFOM)
    动态适配输入/输出通道，保证 split/cat 时维度对齐
    """
    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1  # 默认保持输入输出一致

        # 将输入通道划分成两半，如果是奇数自动补齐
        self.split1 = c1 // 2
        self.split2 = c1 - self.split1

        # 两个分支：分别处理
        self.branch1 = nn.Conv2d(self.split1, c2 // 2, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Conv2d(self.split2, c2 // 2, kernel_size=3, stride=1, padding=1)

        # 融合层
        self.fuse = nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 动态分割
        x1, x2 = torch.split(x, [self.split1, self.split2], dim=1)

        y1 = self.branch1(x1)
        y2 = self.branch2(x2)

        # 拼接两个分支，确保通道数= c2
        y = torch.cat((y1, y2), dim=1)
        y = self.fuse(y)
        y = self.bn(y)
        return self.act(y)
