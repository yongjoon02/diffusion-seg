"""Proposed Diffusion-based Segmentation Model v2

Extends Cold Diffusion with probabilistic guidance and Signed Distance Transform.

Key improvements over v1:
1. Probability-guided sampling: Uses probability maps during training
2. Masked input: img * mask to reduce error propagation
3. Probabilistic early stopping: Bernoulli sampling in early steps
4. Focal L1 loss: Better handling of prediction errors
5. Signed Distance Transform: Joint learning of binary mask and distance field

Based on Cold SegDiffusion with probabilistic enhancements and SDF.
"""
import autorootcwd
import math
import torch
import torch.nn.functional as F
from torch import nn
from random import random
from functools import partial
from collections import namedtuple
from scipy.ndimage import distance_transform_edt
import numpy as np

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


def compute_sdf(mask, threshold=10.0):
    """Compute Signed Distance Function from binary mask with threshold
    
    Args:
        mask: Binary mask tensor [B, 1, H, W] in range [0, 1]
        threshold: Distance threshold in pixels (default: 10.0)
    
    Returns:
        SDF tensor [B, 1, H, W] in range [-1, 1] (thresholded and normalized)
    """
    batch_size = mask.shape[0]
    sdfs = []
    
    for i in range(batch_size):
        binary_mask = (mask[i, 0] > 0.5).cpu().numpy()
        
        # Distance transform inside (positive)
        dist_inside = distance_transform_edt(binary_mask)
        
        # Distance transform outside (negative)
        dist_outside = distance_transform_edt(1 - binary_mask)
        
        # Signed distance function
        sdf = dist_inside - dist_outside
        
        # Threshold to ±threshold pixels
        sdf = np.clip(sdf, -threshold, threshold)
        
        # Normalize to [-1, 1]
        sdf = sdf / threshold
        
        sdfs.append(torch.from_numpy(sdf).unsqueeze(0))
    
    sdf_tensor = torch.stack(sdfs, dim=0).to(mask.device).float()
    return sdf_tensor


def sdf_to_mask(sdf, threshold=0.0):
    """Convert SDF back to binary mask
    
    Args:
        sdf: SDF tensor [B, 1, H, W]
        threshold: Threshold for binarization (default: 0.0)
    
    Returns:
        Binary mask [B, 1, H, W] in [0, 1]
    """
    return (sdf > threshold).float()


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


class BceDiceLoss(nn.Module):
    """Combined BCE + Dice loss for mask prediction"""
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.wb * bce_loss + self.wd * dice_loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        smooth = 1.0
        size = pred.size(0)
        
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size
        
        return dice_loss


class CombinedLoss(nn.Module):
    """Combined loss for binary mask and SDF prediction with consistency loss."""
    
    def __init__(self, mask_weight=1.0, sdf_weight=1.0, consistency_weight=1.0, 
                 mask_bce_weight=1.0, mask_dice_weight=1.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.sdf_weight = sdf_weight
        self.consistency_weight = consistency_weight
        
        # Loss functions
        self.mask_loss = BceDiceLoss(wb=mask_bce_weight, wd=mask_dice_weight)  # BCE + Dice
        self.sdf_loss = SFLoss(loss_type=1, alpha=1.0, beta=1.0, base_loss="l1")  # Focal L1
        self.consistency_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_mask, pred_sdf, gt_mask, gt_sdf):
        """Forward pass of combined loss with consistency.
        
        Args:
            pred_mask: Predicted binary mask [B, 1, H, W] in [0, 1]
            pred_sdf: Predicted SDF [B, 1, H, W] in [-1, 1]
            gt_mask: Ground truth binary mask [B, 1, H, W] in [0, 1]
            gt_sdf: Ground truth SDF [B, 1, H, W] in [-1, 1]
        """
        # Binary mask loss (BCE + Dice)
        mask_loss = self.mask_loss(pred_mask, gt_mask)
        
        # SDF loss (Focal L1)
        sdf_loss = self.sdf_loss(pred_sdf, gt_sdf)
        
        # Consistency loss: SDF -> mask 변환과 원본 mask 간 일관성
        # SDF에서 mask 복원 (threshold = 0)
        pred_sdf_mask = (pred_sdf > 0.0).float()
        consistency_loss = self.consistency_loss(pred_mask, pred_sdf_mask)
        
        # Combined loss
        total_loss = (self.mask_weight * mask_loss + 
                     self.sdf_weight * sdf_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss, mask_loss, sdf_loss, consistency_loss


# ====== Proposed Diffusion Process ======
class ProposedDiffusionModelV2(ColdDiffusionModel):
    """Proposed Diffusion v2: Cold Diffusion + Probabilistic Guidance + SDF
    
    Key enhancements:
    1. Probability-guided forward process
    2. Masked input (img * mask) to reduce error propagation
    3. Probabilistic vs deterministic sampling based on timestep
    4. Combined loss: Binary mask + Signed Distance Transform
    5. Joint learning of segmentation and distance field
    """
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, 
                 objective='predict_x0', beta_schedule='cosine', 
                 use_probability_guidance=True, use_sdf=True,
                 mask_weight=1.0, sdf_weight=1.0, consistency_weight=1.0):
        # Initialize with 'mse' loss_type for parent class
        super().__init__(model, timesteps, sampling_timesteps, objective, beta_schedule, loss_type='mse')
        
        self.use_probability_guidance = use_probability_guidance
        self.use_sdf = use_sdf
        
        # Use combined loss for mask + SDF + consistency
        if use_sdf:
            self.loss_function = CombinedLoss(
                mask_weight=mask_weight,
                sdf_weight=sdf_weight,
                consistency_weight=consistency_weight,
                mask_bce_weight=1.0,
                mask_dice_weight=1.0
            )
        else:
            # Fallback to single mask loss
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
            x_start: Ground truth segmentation (can be 1 or 2 channels)
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
        
        if self.use_sdf and x_start.shape[1] == 2:
            # Handle 2-channel input (mask + SDF)
            mask_channel = x_start[:, :1, :, :]  # First channel is mask
            sdf_channel = x_start[:, 1:2, :, :]  # Second channel is SDF
            
            # Adjust probability map based on sampling_rate for mask channel only
            adjusted_prob = sampling_rate * prob_img + (1 - sampling_rate) * mask_channel
            adjusted_prob = torch.clamp(adjusted_prob, 0., 1.)
            
            # Sample mask from adjusted probability
            sampled_mask = torch.bernoulli(adjusted_prob).to(self.device)
            
            # Keep SDF channel unchanged, only mask channel is sampled
            x_start_masked = torch.cat([sampled_mask, sdf_channel], dim=1)
        else:
            # Handle 1-channel input (mask only)
            adjusted_prob = sampling_rate * prob_img + (1 - sampling_rate) * x_start
            adjusted_prob = torch.clamp(adjusted_prob, 0., 1.)
            
            # Sample mask from adjusted probability
            mask = torch.bernoulli(adjusted_prob).to(self.device)
            
            # Masked input: noise * mask (only bright regions)
            x_start_masked = noise * mask
        
        # Standard cold diffusion degradation
        return self.q_sample(x_start=x_start_masked, t=t, noise=noise)
    
    def forward(self, seg_mask, cond_img, prob_img=None):
        """Training loss with probabilistic guidance and SDF
        
        Args:
            seg_mask: Ground truth segmentation [0, 1]
            cond_img: Conditional image
            prob_img: Probability map (optional, if None uses standard cold diffusion)
        
        Returns:
            Loss value (and individual losses if SDF is used)
        """
        device = self.device
        seg_mask, cond_img = seg_mask.to(device), cond_img.to(device)
        
        b = seg_mask.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Normalize mask to [-1, 1]
        seg_mask = normalize_to_neg_one_to_one(seg_mask)
        
        # Compute SDF if needed and prepare input
        if self.use_sdf:
            # Convert to [0, 1] for SDF computation
            seg_mask_01 = unnormalize_to_zero_to_one(seg_mask)
            gt_sdf = compute_sdf(seg_mask_01, threshold=10.0)  # Already in [-1, 1] range
            
            # Concatenate mask and SDF for 2-channel input
            x_start = torch.cat([seg_mask, gt_sdf], dim=1)  # [B, 2, H, W]
        else:
            x_start = seg_mask  # [B, 1, H, W]
        
        # Use probability guidance if available
        if prob_img is not None and self.use_probability_guidance:
            prob_img = prob_img.to(device)
            x_degraded = self.q_sample_with_probability(
                x_start=x_start, t=times, noise=cond_img, prob_img=prob_img
            )
        else:
            # Fallback to standard cold diffusion
            x_degraded = self.q_sample(x_start=x_start, t=times, noise=cond_img)
        
        # Model prediction (no self-conditioning)
        model_out = self.model(x_degraded, times, cond_img)
        
        # Compute loss
        if self.use_sdf:
            # Split model output: first channel = mask, second channel = SDF
            pred_mask = model_out[:, :1, :, :]  # [B, 1, H, W]
            pred_sdf = model_out[:, 1:2, :, :]  # [B, 1, H, W]
            
            # Convert predictions to correct ranges
            pred_mask_01 = unnormalize_to_zero_to_one(pred_mask)  # [0, 1] for BCE+Dice
            seg_mask_01 = unnormalize_to_zero_to_one(seg_mask)    # [0, 1] for BCE+Dice
            
            # Ensure values are in [0, 1] range
            pred_mask_01 = torch.clamp(pred_mask_01, 0., 1.)
            seg_mask_01 = torch.clamp(seg_mask_01, 0., 1.)
            
            # Combined loss with consistency
            total_loss, mask_loss, sdf_loss, consistency_loss = self.loss_function(
                pred_mask_01, pred_sdf, seg_mask_01, gt_sdf
            )
            return total_loss
        else:
            # Single mask loss
            loss = self.loss_function(model_out, seg_mask)
            return loss
    
    @torch.no_grad()
    def sample(self, cond_img, save_steps=None):
        
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from conditional image
        if self.use_sdf:
            # Initialize with zeros for both mask and SDF channels
            img = torch.zeros(b, 2, h, w, device=self.device)
        else:
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
            
            # Handle SDF output
            if self.use_sdf:
                # Split output: first channel = mask, second channel = SDF
                pred_mask_only = pred_mask[:, :1, :, :]  # [B, 1, H, W]
                pred_sdf = pred_mask[:, 1:2, :, :]  # [B, 1, H, W]
            else:
                pred_mask_only = pred_mask
            
            # Convert to [0, 1] for sampling
            pred_mask_01 = unnormalize_to_zero_to_one(pred_mask_only)
            
            # Progressive probability guidance: t=T → random, t=0 → deterministic
            # sampling_rate: 1.0 at t=T (max), 0.0 at t=0
            sampling_rate = float(t) / self.num_timesteps  # [0, 1]
            sampling_rate = torch.tensor(sampling_rate, device=self.device).view(-1, 1, 1, 1)  # Broadcast shape
            
            # Adjust probability based on sampling_rate
            # At t=T (sampling_rate=1.0): use predicted probability → Bernoulli
            # At t=0 (sampling_rate=0.0): use threshold → deterministic
            adjusted_prob = sampling_rate * pred_mask_01 + (1 - sampling_rate) * torch.where(
                pred_mask_01 > 0.5, torch.ones_like(pred_mask_01), torch.zeros_like(pred_mask_01)
            )
            adjusted_prob = torch.clamp(adjusted_prob, 0., 1.)
            
            # Sample mask from adjusted probability
            sampled_mask = torch.bernoulli(adjusted_prob).to(self.device)
            
            # Prepare x_start for next step
            if self.use_sdf:
                # Concatenate sampled mask and predicted SDF
                x_start = torch.cat([sampled_mask, pred_sdf], dim=1)
            else:
                # Convert back to [-1, 1] and multiply with conditional image
                sampled_mask_norm = normalize_to_neg_one_to_one(sampled_mask)
                x_start = cond_img * sampled_mask_norm
            
            # Save step if requested
            if save_steps is not None and t in save_steps:
                if self.use_sdf:
                    saved_steps[t] = {
                        'mask': sampled_mask.cpu(),
                        'sdf': pred_sdf.cpu()
                    }
                else:
                    saved_steps[t] = sampled_mask.cpu()
            
            # Next step: degrade sampled_mask to t-1
            if t > 0:
                batched_times_prev = torch.full((b,), t - 1, device=self.device, dtype=torch.long)
                img = self.q_sample(x_start=x_start, t=batched_times_prev, noise=cond_img)
            else:
                # Final step: use sampled mask
                img = x_start
                
        if save_steps is not None:
            if self.use_sdf:
                return {
                    'final_mask': sampled_mask,
                    'final_sdf': pred_sdf,
                    'steps': saved_steps
                }
            else:
                return {
                    'final': sampled_mask,
                    'steps': saved_steps
                }
        else:
            if self.use_sdf:
                return {
                    'mask': sampled_mask,
                    'sdf': pred_sdf
                }
            else:
                return sampled_mask


# ====== Factory Functions ======
def create_proposed_diff_v2(image_size=224, dim=32, timesteps=100, 
                           use_probability_guidance=True, use_sdf=True,
                           mask_weight=1.0, sdf_weight=1.0, consistency_weight=1.0):
    """Proposed Diffusion v2: Cold Diffusion + Probabilistic Guidance + SDF + Consistency
    
    Args:
        image_size: Input image size (default: 224)
        dim: Base dimension (default: 32)
        timesteps: Number of diffusion steps (default: 100)
        use_probability_guidance: Use probability maps (default: True)
        use_sdf: Use Signed Distance Transform (default: True)
        mask_weight: Weight for binary mask loss (default: 1.0)
        sdf_weight: Weight for SDF loss (default: 1.0)
        consistency_weight: Weight for consistency loss (default: 1.0)
    """
    # Determine output channels based on SDF usage
    output_channels = 2 if use_sdf else 1
    
    unet = MedSegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=output_channels,  # 2 channels for mask + SDF
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        mid_transformer_depth=1
    )
    return ProposedDiffusionModelV2(
        unet, 
        timesteps=timesteps, 
        objective='predict_x0', 
        beta_schedule='cosine',
        use_probability_guidance=use_probability_guidance,
        use_sdf=use_sdf,
        mask_weight=mask_weight,
        sdf_weight=sdf_weight,
        consistency_weight=consistency_weight
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Proposed Diffusion Model v2")
    print("=" * 70)
    
    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    prob = torch.rand(2, 1, 224, 224)
    
    print("\n1. Proposed Diffusion v2 (with SDF + Consistency)")
    proposed_v2 = create_proposed_diff_v2(image_size=224, dim=32, timesteps=100, 
                                        use_probability_guidance=True, use_sdf=True,
                                        mask_weight=1.0, sdf_weight=1.0, consistency_weight=1.0)
    loss, mask_loss, sdf_loss, consistency_loss = proposed_v2(img, cond, prob)
    params = sum(p.numel() for p in proposed_v2.parameters())
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Mask Loss: {mask_loss.item():.4f}, SDF Loss: {sdf_loss.item():.4f}, Consistency Loss: {consistency_loss.item():.4f}")
    print(f"   Params: {params:,}")
    
    print("\n2. Proposed Diffusion v2 (without SDF)")
    proposed_v2_no_sdf = create_proposed_diff_v2(image_size=224, dim=32, timesteps=100, 
                                               use_probability_guidance=True, use_sdf=False)
    loss = proposed_v2_no_sdf(img, cond, prob)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test sampling
    print("\n3. Testing sampling with SDF...")
    with torch.no_grad():
        result = proposed_v2.sample(cond[:1])
        print(f"   Mask shape: {result['mask'].shape}, min: {result['mask'].min():.2f}, max: {result['mask'].max():.2f}")
        print(f"   SDF shape: {result['sdf'].shape}, min: {result['sdf'].min():.2f}, max: {result['sdf'].max():.2f}")
    
    print("\n4. Testing sampling without SDF...")
    with torch.no_grad():
        sample = proposed_v2_no_sdf.sample(cond[:1])
        print(f"   Sample shape: {sample.shape}, min: {sample.min():.2f}, max: {sample.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Proposed Diffusion v2 model works correctly!")
    print("=" * 70)

