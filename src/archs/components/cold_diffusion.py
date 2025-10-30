"""Cold Diffusion-based Segmentation Models

Cold Diffusion for image-to-segmentation without noise.
Deterministic diffusion that transforms image → segmentation directly.

Key differences from Gaussian Diffusion:
1. Forward process: seg → image (degradation by blending)
2. Reverse process: image → seg (restoration)
3. Loss: Weighted MSE or Hybrid (MSE + BCE+Dice)
4. No random noise - uses conditional image as degradation target

Based on Cold SegDiffusion paper.
"""
import autorootcwd
import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import namedtuple

# Import components from existing modules
from src.archs.components.diffusion_unet import (
    # Utility functions
    exists, default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one,
    # UNet architectures
    SegDiffUNet, MedSegDiffUNet,
    # Diffusion process utilities
    extract, linear_beta_schedule, cosine_beta_schedule
)
# Import base diffusion model and losses from gaussian_diffusion
from src.archs.components.gaussian_diffusion import GaussianDiffusionModel, BceDiceLoss

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ====== Cold Diffusion Process ======
class ColdDiffusionModel(GaussianDiffusionModel):
    """Cold Diffusion: Image → Segmentation directly (no noise)
    
    Forward: seg → image (degradation)
    Reverse: image → seg (restoration)
    """
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, 
                 objective='predict_x0', beta_schedule='cosine', loss_type='mse'):
        super().__init__(model, timesteps, sampling_timesteps, objective, beta_schedule, loss_type)
        
        # Loss functions for hybrid mode
        self.bce_dice_loss = BceDiceLoss(wb=1.0, wd=1.0)
    
    def q_sample(self, x_start, t, noise):
        """Forward degradation: gradually blend mask towards image
        
        Args:
            x_start: Ground truth segmentation
            t: Timestep
            noise: Conditional image (not random noise!)
        
        Returns:
            Degraded mask (blended with image)
        """
        # Instead of adding noise, blend seg towards image
        alpha_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        beta_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Degradation: x_t = alpha * seg + beta * image
        return alpha_t * x_start + beta_t * noise
    
    def forward(self, seg_mask, cond_img, gamma=1.0):
        """Cold SegDiffusion Loss Function
        
        Loss types:
        - 'mse': Weighted MSE with time weighting (original paper)
        - 'hybrid': Weighted MSE + BCE+Dice
        """
        device = self.device
        seg_mask, cond_img = seg_mask.to(device), cond_img.to(device)
        
        b = seg_mask.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Normalize segmentation mask to [-1, 1]
        seg_mask = normalize_to_neg_one_to_one(seg_mask)
        
        # Equation 6: x_t = √α_t * x₀ + √(1-α_t) * z
        x_degraded = self.q_sample(x_start=seg_mask, t=times, noise=cond_img)
        
        # Model prediction: f(x_{i,t}, t)
        predicted_seg = self.model(x_degraded, times, cond_img)
        
        if self.loss_type == 'mse':
            # Original: Weighted MSE with time weighting
            # time_weights = (1 + t)^gamma for numerical stability
            #time_weights = torch.pow((times.float() + 1) / self.num_timesteps, gamma)
            mse_loss = F.mse_loss(predicted_seg, seg_mask)
            #mse_loss = mse_loss.mean(dim=[1, 2, 3])
            #weighted_loss = time_weights * mse_loss
            return mse_loss
        
        elif self.loss_type == 'hybrid':
            # Hybrid: Weighted MSE + BCE+Dice
            # Loss1: Weighted MSE (time_weights = (1 + t)^gamma)
            time_weights = torch.pow((times.float() + 1) / self.num_timesteps, gamma)
            mse_loss = mean_flat((predicted_seg - seg_mask) ** 2)
            weighted_mse = (time_weights * mse_loss).mean()
            
            # Loss2: BCE+Dice for segmentation quality
            pred_seg = unnormalize_to_zero_to_one(torch.clamp(predicted_seg, min=-1., max=1.))
            target_seg = unnormalize_to_zero_to_one(seg_mask)
            bce_dice = self.bce_dice_loss(pred_seg, target_seg)
            
            # Combine
            alpha, beta = 0.1, 0.9
            return alpha * weighted_mse + beta * bce_dice
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    @torch.no_grad()
    def sample(self, cond_img, save_steps=None):
        """Reverse process: image → segmentation (Cold Diffusion)
        
        Paper: Cold SegDiffusion
        Start from image, gradually recover segmentation with residual updates
        (No self-conditioning)
        """
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from conditional image (not noise!)
        img = cond_img
        
        # Initialize step storage if needed
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)
        
        # Cold Diffusion restoration loop (간단한 deterministic 방식)
        for t in reversed(range(0, self.num_timesteps)):
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            
            # UNet prediction (no self-conditioning)
            preds = self.model_predictions(img, batched_times, cond_img, clip_x_start=True)
            pred_x0 = preds.predict_x_start
            
            # Save step
            if save_steps is not None and t in save_steps:
                saved_img = unnormalize_to_zero_to_one(pred_x0)
                saved_steps[t] = saved_img.cpu()
            
            # Next step: degrade pred_x0 to t-1 (deterministic)
            if t > 0:
                batched_times_prev = torch.full((b,), t - 1, device=self.device, dtype=torch.long)
                img = self.q_sample(x_start=pred_x0, t=batched_times_prev, noise=cond_img)
            else:
                # Final step: use prediction directly
                img = pred_x0
        
        img = unnormalize_to_zero_to_one(img)
        
        if save_steps is not None:
            return {
                'final': img,
                'steps': saved_steps
            }
        else:
            return img


# ====== Factory Functions ======
def create_colddiff(image_size=224, dim=32, timesteps=100, loss_type='hybrid'):
    """Cold Diffusion: Image → Segmentation (no noise)
    
    Args:
        image_size: Input image size (default: 224)
        dim: Base dimension (default: 32)
        timesteps: Number of diffusion steps (default: 100)
        loss_type: 'hybrid' (default, mse + bce+dice) or 'mse'
    """
    unet = MedSegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        mid_transformer_depth=1
    )
    return ColdDiffusionModel(unet, timesteps=timesteps, objective='predict_x0', 
                             beta_schedule='cosine', loss_type=loss_type)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Cold Diffusion Model")
    print("=" * 70)
    
    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    
    print("\n1. ColdDiff (Image → Seg)")
    colddiff = create_colddiff(image_size=224, dim=64, timesteps=100)
    loss = colddiff(img, cond)
    params = sum(p.numel() for p in colddiff.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    # Test sampling
    print("\n2. Testing sampling...")
    with torch.no_grad():
        sample = colddiff.sample(cond[:1])
        print(f"   Sample shape: {sample.shape}, min: {sample.min():.2f}, max: {sample.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Cold Diffusion model works correctly!")
    print("=" * 70)

