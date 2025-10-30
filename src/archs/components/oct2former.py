"""OCT2Former: Transformer-based OCT segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import OCTEncoder, ConvEncoder
from .decoder import OCTDecoder, TransformerDecoder


class OCT2Former(nn.Module):
    """OCT2Former model for OCT image segmentation."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, num_classes: int = 2,
                 embed_dim: int = 768, depth: int = 12, 
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, use_conv_encoder: bool = False):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        if use_conv_encoder:
            self.encoder = ConvEncoder(in_channels, 64)
            self.decoder = OCTDecoder(256, num_classes)
        else:
            self.encoder = OCTEncoder(img_size, patch_size, in_channels, 
                                    embed_dim, depth, num_heads, mlp_ratio, dropout)
            self.decoder = TransformerDecoder(embed_dim, num_classes, patch_size, img_size)
            
        self.use_conv_encoder = use_conv_encoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv_encoder:
            skip_connections, encoded = self.encoder(x)
            output = self.decoder(encoded, skip_connections)
        else:
            encoded = self.encoder(x)
            output = self.decoder(encoded)
            
        return output


class OCT2FormerSmall(OCT2Former):
    """Smaller version of OCT2Former."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, num_classes: int = 2):
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            num_classes=num_classes, embed_dim=384, depth=6, num_heads=6
        )


class OCT2FormerLarge(OCT2Former):
    """Larger version of OCT2Former."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, num_classes: int = 2):
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            num_classes=num_classes, embed_dim=1024, depth=24, num_heads=16
        )


class OCT2FormerHybrid(nn.Module):
    """Hybrid model combining CNN and Transformer."""
    
    def __init__(self, img_size: int = 224, in_channels: int = 3, 
                 num_classes: int = 2, embed_dim: int = 768):
        super().__init__()
        self.conv_encoder = ConvEncoder(in_channels, 64)
        self.transformer_encoder = OCTEncoder(img_size, 16, 256, embed_dim, 6, 8)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = OCTDecoder(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN encoding
        skip_connections, conv_features = self.conv_encoder(x)
        
        # Transformer encoding
        transformer_features = self.transformer_encoder(conv_features)
        transformer_features = transformer_features[:, 1:, :]  # Remove cls token
        transformer_features = transformer_features.transpose(1, 2).view(
            transformer_features.shape[0], -1, 
            int(conv_features.shape[2]), int(conv_features.shape[3])
        )
        
        # Feature fusion
        fused_features = torch.cat([conv_features, transformer_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Decoding
        output = self.decoder(fused_features, skip_connections)
        return output
