"""CSNet: Channel and Spatial attention Network for OCT segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ChannelAttention, SpatialAttention


class CSBlock(nn.Module):
    """Channel and Spatial attention block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)
        
        # Residual connection for different channel sizes
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        # Handle residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        x = x + residual
        return F.relu(x)


class CSNet(nn.Module):
    """Channel and Spatial attention Network."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc2 = CSBlock(base_channels, base_channels * 2)
        self.enc3 = CSBlock(base_channels * 2, base_channels * 4)
        self.enc4 = CSBlock(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = CSBlock(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = CSBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = CSBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = CSBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = CSBlock(base_channels * 2, base_channels)
        
        # Final classification
        self.final = nn.Conv2d(base_channels, num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)
