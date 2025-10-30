"""UTNet: U-shaped Transformer for OCT segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import OCTTransformer


class UTNet(nn.Module):
    """UTNet: U-shaped Transformer for medical image segmentation."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 64,
                 embed_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, num_transformer_layers: int = 4):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # Transformer bottleneck
        self.transformer_bottleneck = nn.ModuleList([
            OCTTransformer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Projection layers for transformer
        self.proj_to_transformer = nn.Conv2d(base_channels * 8, embed_dim, 1)
        self.proj_from_transformer = nn.Conv2d(embed_dim, base_channels * 8, 1)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(embed_dim + base_channels * 8, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final classification
        self.final = nn.Conv2d(base_channels // 2, num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Transformer bottleneck
        # Project to transformer dimension
        x_trans = self.proj_to_transformer(e4)
        B, C, H, W = x_trans.shape
        
        # Reshape for transformer
        x_trans = x_trans.flatten(2).transpose(1, 2)  # B, H*W, C
        
        # Apply transformer blocks
        for transformer_block in self.transformer_bottleneck:
            x_trans = transformer_block(x_trans)
        
        # Reshape back to spatial
        x_trans = x_trans.transpose(1, 2).view(B, C, H, W)
        
        # Project back to original dimension
        x_trans = self.proj_from_transformer(x_trans)
        
        # Decoder
        d4 = self.dec4(torch.cat([F.interpolate(x_trans, size=e4.shape[2:], mode='bilinear', align_corners=False), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False), e1], dim=1))
        
        return self.final(d1)


