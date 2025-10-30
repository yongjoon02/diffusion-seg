"""AACA-UNet: Attention Augmented Convolution Attention UNet for OCT segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AugmentedConv(nn.Module):
    """Augmented convolution with attention."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dk: int = 40, dv: int = 4, Nh: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        
        self.conv_out = nn.Conv2d(in_channels, out_channels - dv, kernel_size, padding=kernel_size//2)
        self.qkv_conv = nn.Conv2d(in_channels, 2 * dk + dv, kernel_size, padding=kernel_size//2)
        self.attn_out = nn.Conv2d(dv, dv, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regular convolution
        conv_out = self.conv_out(x)
        
        # Attention
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.shape
        
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        
        # Reshape for multi-head attention
        q = q.view(N, self.Nh, self.dk // self.Nh, H * W)
        k = k.view(N, self.Nh, self.dk // self.Nh, H * W)
        v = v.view(N, self.Nh, self.dv // self.Nh, H * W)
        
        # Attention weights
        attn = torch.softmax(torch.sum(q * k, dim=2, keepdim=True) / (self.dk // self.Nh) ** 0.5, dim=3)
        
        # Apply attention
        v = v * attn
        v = v.view(N, self.dv, H, W)
        attn_out = self.attn_out(v)
        
        # Concatenate
        return torch.cat([conv_out, attn_out], dim=1)


class AACAUNet(nn.Module):
    """Attention Augmented Convolution Attention UNet."""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 64):
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
            AugmentedConv(base_channels, base_channels * 2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            AugmentedConv(base_channels * 2, base_channels * 2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            AugmentedConv(base_channels * 2, base_channels * 4),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            AugmentedConv(base_channels * 4, base_channels * 4),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            AugmentedConv(base_channels * 4, base_channels * 8),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            AugmentedConv(base_channels * 8, base_channels * 8),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            AugmentedConv(base_channels * 8, base_channels * 16),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            AugmentedConv(base_channels * 16, base_channels * 16),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )
        
        # Attention gates
        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels // 2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
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
        
        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)
