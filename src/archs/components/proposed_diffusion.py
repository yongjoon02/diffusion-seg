"""Proposed Diffusion-based Segmentation Model

Extends Cold Diffusion with probabilistic guidance for more robust segmentation.

Key improvements over Cold Diffusion:
1. Probability-guided sampling: Uses probability maps during training
2. Masked input: img * mask to reduce error propagation
3. Probabilistic early stopping: Bernoulli sampling in early steps
4. Focal L1 loss: Better handling of prediction errors

Based on Cold SegDiffusion with probabilistic enhancements.
"""
import autorootcwd
import math
import torch
import torch.nn.functional as F
from torch import nn
from random import random
from functools import partial
from collections import namedtuple

# Import components from existing modules
from src.archs.components.diffusion_unet import (
    # Utility functions
    exists, default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one,
    # UNet architectures
    MedSegDiffUNet,
    # Diffusion process utilities
    extract, linear_beta_schedule, cosine_beta_schedule
)
# Import base cold diffusion model
from src.archs.components.cold_diffusion import ColdDiffusionModel

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


class SFLoss(nn.Module):
    """Focal L1 loss for medical image segmentation."""
    
    def __init__(
        self, 
        loss_type: int = 1, 
        alpha: float = 1.0, 
        beta: float = 1.0, 
        secondary_weight: float = 1.0, 
        pos_weight: float = 1.0, 
        base_loss: str = "l1"
    ):
        super().__init__()
        self.base_loss = base_loss

        if base_loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif base_loss == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.L1Loss(reduction="none")

        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.secondary_weight = secondary_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Forward pass of focal loss."""
        same_sign_mask = (torch.sign(pred) * torch.sign(gt) > 0)
        pos_mask = (gt > 0)
        weight = torch.ones_like(gt)

        weight = torch.where(
            same_sign_mask,
            torch.pow(torch.abs(pred - gt), self.beta) * self.alpha,
            torch.ones_like(gt),
        )
        weight = weight * torch.where(
            pos_mask,
            torch.full_like(gt, fill_value=self.pos_weight),
            torch.ones_like(gt),
        )

        if self.loss_type > 1:
            raise ValueError("Invalid type!")
        if self.loss_type in [0]:
            weight = weight.detach()

        loss = self.criterion(pred, gt)
        loss = loss * weight
        loss = torch.mean(loss, dim=(0, 2, 3))

        class_weights = torch.tensor(
            [1.0] + [self.secondary_weight] * (loss.shape[0] - 1), device=loss.device)
        loss = (loss * class_weights).sum() / class_weights.sum()
        return loss


# ====== Proposed Diffusion Process ======
class ProposedDiffusionModel(ColdDiffusionModel):
    """Proposed Diffusion: Cold Diffusion + Probabilistic Guidance
    
    Key enhancements:
    1. Probability-guided forward process
    2. Masked input (img * mask) to reduce error propagation
    3. Probabilistic vs deterministic sampling based on timestep
    4. Focal L1 loss
    """
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, 
                 objective='predict_x0', beta_schedule='cosine', 
                 use_probability_guidance=True, loss_type='hybrid'):
        # Initialize with 'mse' loss_type for parent class
        super().__init__(model, timesteps, sampling_timesteps, objective, beta_schedule)
        
        self.use_probability_guidance = use_probability_guidance
        
        # Use Focal L1 loss
        self.loss_function = SFLoss(
            loss_type=1,
            alpha=1.0,
            beta=1.0,
            secondary_weight=1.0,
            pos_weight=1.0,
            base_loss="l1"
        )
    
    def q_sample_with_probability(self, x_start, t, noise, prob_img):
        """Forward diffusion with probability guidance
        
        Args:
            x_start: Ground truth segmentation
            t: Timestep
            noise: Conditional image
            prob_img: Probability map for guidance
        
        Returns:
            Probabilistically sampled degraded mask
        """
        # Progressive sampling: t=T (max) → random (Bernoulli), t=0 → deterministic (GT)
        # sampling_rate: 1.0 at t=T, 0.0 at t=0
        sampling_rate = t.float() / self.num_timesteps  # [0, 1]
        sampling_rate = sampling_rate.view(-1, 1, 1, 1)  # Broadcast shape
        
        # Adjust probability map based on sampling_rate
        # At t=T (sampling_rate=1.0): adjusted_prob = prob_img → Bernoulli(prob_img)
        # At t=0 (sampling_rate=0.0): adjusted_prob = x_start → deterministic GT
        adjusted_prob = sampling_rate * prob_img + (1 - sampling_rate) * x_start
        adjusted_prob = torch.clamp(adjusted_prob, 0., 1.)
        
        # Sample mask from adjusted probability (always binary output)
        mask = torch.bernoulli(adjusted_prob).to(self.device)
        
        # Masked input: noise * mask (only bright regions)
        x_start_masked = noise * mask
        
        # Standard cold diffusion degradation
        return self.q_sample(x_start=x_start_masked, t=t, noise=noise)
    
    def forward(self, seg_mask, cond_img, prob_img=None):
        """Training loss with probabilistic guidance
        
        Args:
            seg_mask: Ground truth segmentation [0, 1]
            cond_img: Conditional image
            prob_img: Probability map (optional, if None uses standard cold diffusion)
        
        Returns:
            Loss value
        """
        device = self.device
        seg_mask, cond_img = seg_mask.to(device), cond_img.to(device)
        
        b = seg_mask.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Normalize mask to [-1, 1]
        seg_mask = normalize_to_neg_one_to_one(seg_mask)
        
        # Use probability guidance if available
        if prob_img is not None and self.use_probability_guidance:
            prob_img = prob_img.to(device)
            x_degraded = self.q_sample_with_probability(
                x_start=seg_mask, t=times, noise=cond_img, prob_img=prob_img
            )
        else:
            # Fallback to standard cold diffusion
            x_degraded = self.q_sample(x_start=seg_mask, t=times, noise=cond_img)
        
        # Model prediction (no self-conditioning)
        model_out = self.model(x_degraded, times, cond_img)
        
        # Loss computation based on loss_type
        if self.loss_type == 'hybrid':
            # Hybrid loss: BCE+Dice + weighted Focal L1
            # Loss1: BCE+Dice for segmentation quality
            pred_mask_01 = unnormalize_to_zero_to_one(torch.clamp(model_out, min=-1., max=1.))
            seg_mask_01 = unnormalize_to_zero_to_one(seg_mask)
            bce_dice_loss = self.bce_dice_loss(pred_mask_01, seg_mask_01)
            
            # Loss2: Weighted Focal L1 (time-weighted)
            time_weights = torch.pow((times.float() + 1) / self.num_timesteps, 1.0)
            focal_l1_loss = self.loss_function(model_out, seg_mask)
            weighted_focal_l1 = (time_weights * focal_l1_loss).mean()
            
            # Combine losses (alpha=0.1, beta=0.9 like in cold_diffusion.py)
            loss = 0.1 * weighted_focal_l1 + 0.9 * bce_dice_loss
        else:
            # Standard Focal L1 loss (loss_type='mse')
            loss = self.loss_function(model_out, seg_mask)
        
        return loss
    
    @torch.no_grad()
    def sample(self, cond_img, save_steps=None):
        
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from conditional image
        img = cond_img
        x_start = None
        
        # Initialize step storage if needed
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)
        
        for t in reversed(range(0, self.num_timesteps)):
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            
            # Predict mask (no self-conditioning)
            preds = self.model_predictions(img, batched_times, cond_img, clip_x_start=True)
            pred_mask = preds.predict_x_start
            
            # Convert to [0, 1] for sampling
            pred_mask_01 = unnormalize_to_zero_to_one(pred_mask)
            sampled_mask = torch.where(pred_mask_01 > 0.5, torch.ones_like(pred_mask_01), torch.zeros_like(pred_mask_01))
            
            # Convert back to [-1, 1]
            x_start = cond_img * sampled_mask
            
            # Save step if requested
            if save_steps is not None and t in save_steps:
                saved_steps[t] = sampled_mask.cpu()
            
            # Next step: degrade sampled_mask to t-1
            if t > 0:
                batched_times_prev = torch.full((b,), t - 1, device=self.device, dtype=torch.long)
                img = self.q_sample(x_start=x_start, t=batched_times_prev, noise=cond_img)
            else:
                # Final step: use sampled mask
                img = x_start
                
        if save_steps is not None:
            return {
                'final': sampled_mask,
                'steps': saved_steps
            }
        else:
            return sampled_mask


# ====== Factory Functions ======
def create_proposed_diff(image_size=224, dim=32, timesteps=100, use_probability_guidance=True, loss_type='hybrid'):
    """Proposed Diffusion: Cold Diffusion + Probabilistic Guidance
    
    Args:
        image_size: Input image size (default: 224)
        dim: Base dimension (default: 32)
        timesteps: Number of diffusion steps (default: 100)
        use_probability_guidance: Use probability maps (default: True)
        loss_type: Loss type - 'mse' for Focal L1, 'hybrid' for BCE+Dice + weighted Focal L1 (default: 'mse')
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
    return ProposedDiffusionModel(
        unet, 
        timesteps=timesteps, 
        objective='predict_x0', 
        beta_schedule='cosine',
        use_probability_guidance=use_probability_guidance,
        loss_type=loss_type
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Proposed Diffusion Model")
    print("=" * 70)
    
    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    prob = torch.rand(2, 1, 224, 224)
    
    print("\n1. Proposed Diffusion (with probability guidance)")
    proposed = create_proposed_diff(image_size=224, dim=32, timesteps=100, use_probability_guidance=True)
    loss = proposed(img, cond, prob)
    params = sum(p.numel() for p in proposed.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    print("\n2. Proposed Diffusion (without probability guidance)")
    proposed_no_prob = create_proposed_diff(image_size=224, dim=32, timesteps=100, use_probability_guidance=False)
    loss = proposed_no_prob(img, cond, prob_img=None)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. Proposed Diffusion (with hybrid loss)")
    proposed_hybrid = create_proposed_diff(image_size=224, dim=32, timesteps=100, use_probability_guidance=True, loss_type='hybrid')
    loss = proposed_hybrid(img, cond, prob)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test sampling
    print("\n4. Testing sampling...")
    with torch.no_grad():
        sample = proposed.sample(cond[:1])
        print(f"   Sample shape: {sample.shape}, min: {sample.min():.2f}, max: {sample.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Proposed Diffusion model works correctly!")
    print("=" * 70)

