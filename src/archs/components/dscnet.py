"""DSCNet: Dynamic Snake Convolution Network for vessel segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSnakeConv(nn.Module):
    """Dynamic Snake Convolution (approximated)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.alpha = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.5))
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        snake = y + self.alpha * torch.sin(y) + self.beta * torch.cos(y)
        return self.bn(snake)


class DSCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = DynamicSnakeConv(in_channels, out_channels)
        self.conv2 = DynamicSnakeConv(out_channels, out_channels, dilation=2)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = y + self.skip(x)
        return F.relu(y)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = DSCBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)
        return feat, self.pool(feat)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = DSCBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DSCNet(nn.Module):
    """Dynamic Snake Convolution Network (simplified)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = DSCBlock(in_channels, c1)
        self.down1 = DownSample(c1, c2)
        self.down2 = DownSample(c2, c3)

        self.bottleneck = DSCBlock(c3, c4)

        self.up2 = UpSample(c4, c3)
        self.up1 = UpSample(c3, c2)
        self.head = nn.Sequential(
            UpSample(c2, c1),
            nn.Conv2d(c1, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1, p1 = self.down1(s0)
        s2, p2 = self.down2(s1)

        bottleneck = self.bottleneck(p2)

        x = self.up2(bottleneck, s2)
        x = self.up1(x, s1)
        x = self.head[0](x, s0)
        logits = self.head[1](x)
        return logits

