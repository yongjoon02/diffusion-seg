"""TransUNet: Transformer UNet for OCT segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import OCTTransformer, PatchEmbedding


class TransUNet(nn.Module):
    """TransUNet: Transformer UNet for medical image segmentation."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 num_classes: int = 2, embed_dim: int = 768, depth: int = 12, 
                 num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            OCTTransformer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # CNN encoder for skip connections
        current_channels = in_channels
        cnn_blocks = []
        channels = [64, 128, 256, 512]
        in_ch = in_channels
        for out_ch in channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            cnn_blocks.append(block)
            in_ch = out_ch
        self.cnn_encoder = nn.ModuleList(cnn_blocks)
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 512, 2, stride=2),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 2, stride=2),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final classification
        self.final = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # CNN encoder for skip connections
        skip_connections = []
        for i, encoder in enumerate(self.cnn_encoder):
            x_cnn = encoder(x_cnn if i > 0 else x)
            skip_connections.append(x_cnn)
            if i < len(self.cnn_encoder) - 1:
                x_cnn = self.pool(x_cnn)
        
        # Transformer encoder
        x_trans = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_trans = torch.cat((cls_tokens, x_trans), dim=1)
        x_trans = x_trans + self.pos_embed
        x_trans = self.dropout(x_trans)
        
        for transformer_block in self.transformer_blocks:
            x_trans = transformer_block(x_trans)
        
        x_trans = self.norm(x_trans)
        
        # Remove cls token and reshape
        x_trans = x_trans[:, 1:, :]  # Remove cls token
        H = W = int(x_trans.shape[1] ** 0.5)
        x_trans = x_trans.transpose(1, 2).view(B, self.embed_dim, H, W)
        
        # Decoder with skip connections
        for i, decoder in enumerate(self.decoder):
            x_trans = decoder(x_trans)
            # Skip connection
            if i < len(skip_connections):
                skip_idx = len(skip_connections) - 1 - i
                x_trans = x_trans + skip_connections[skip_idx]
        
        return self.final(x_trans)
