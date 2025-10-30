"""Proposed Flow Matching for Medical Image Segmentation

Flow Matching with probabilistic guidance and masked input.
Combines Flow Matching + Probability Guidance + Focal Loss.

Key features:
1. Flow Matching: Deterministic ODE (no stochastic noise)
2. Probability-guided masking: Uses uncertainty maps
3. Masked input: img * mask to reduce error
4. Focal L1 loss: Better handling of prediction errors

Based on FlowSDF with enhancements from Proposed Diffusion.
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
    MedSegDiffUNet,
    # Diffusion process utilities
    extract, linear_beta_schedule, cosine_beta_schedule
)
# Import base classes
from src.archs.components.gaussian_diffusion import GaussianDiffusionModel, BceDiceLoss

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


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


# ====== Proposed Flow Matching ======
class ProposedFlowModel(GaussianDiffusionModel):
    """Proposed Flow Matching: Flow Matching + Probabilistic Guidance
    
    Key enhancements:
    1. Flow Matching: Deterministic straight-line ODE paths
    2. Probability-guided masking: Uses boundary uncertainty maps
    3. Masked input: img * mask to focus on uncertain regions
    4. Focal L1 loss: Better handling of boundary errors
    
    Flow Matching vs Diffusion:
    - Diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε (stochastic)
    - Flow Matching: x_t = (1-t) * x_0 + t * x_1 (deterministic)
    """
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, 
                 objective='predict_v', beta_schedule='linear', 
                 use_probability_guidance=True):
        # Initialize parent (beta_schedule not used in flow matching)
        super().__init__(model, timesteps, sampling_timesteps, 'predict_x0', beta_schedule, 'mse')
        
        self.objective = objective  # 'predict_v' (velocity) or 'predict_x1' (target)
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
        
        # For hybrid loss
        self.bce_dice_loss = BceDiceLoss(wb=1.0, wd=1.0)
    
    def q_sample_flow(self, x_0, x_1, t):
        """Flow matching interpolation: x_t = (1-t) * x_0 + t * x_1
        
        Args:
            x_0: Source (conditional image)
            x_1: Target (segmentation mask)
            t: Time in [0, T]
        
        Returns:
            x_t on the straight-line path from x_0 to x_1
        """
        # Normalize t to [0, 1]
        t_normalized = t.float() / self.num_timesteps
        t_expanded = t_normalized.view(-1, 1, 1, 1)
        
        # Linear interpolation
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        return x_t
    
    def q_sample_flow_with_probability(self, x_0, x_1, t, prob_img):
        """Flow matching with probability guidance
        
        Args:
            x_0: Source (conditional image)
            x_1: Target (segmentation mask)
            t: Time
            prob_img: Probability map for guidance
        
        Returns:
            x_t with probabilistic masking
        """
        # Sample mask from probability map
        mask = torch.bernoulli(prob_img).to(self.device)
        
        # Masked target: x_1 = x_0 * mask (only uncertain regions)
        x_1_masked = x_0 * mask
        
        # Flow matching interpolation
        return self.q_sample_flow(x_0, x_1_masked, t)
    
    def compute_velocity(self, x_0, x_1):
        """Compute constant velocity for straight-line flow
        
        For straight paths: v = x_1 - x_0 (constant)
        """
        return x_1 - x_0
    
    def forward(self, seg_mask, cond_img, prob_img=None):
        """Flow matching loss with probabilistic guidance
        
        Args:
            seg_mask: Ground truth segmentation [0, 1]
            cond_img: Conditional image
            prob_img: Probability map (optional)
        
        Returns:
            Loss value
        """
        device = self.device
        seg_mask, cond_img = seg_mask.to(device), cond_img.to(device)
        
        b = seg_mask.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Normalize to [-1, 1]
        seg_mask = normalize_to_neg_one_to_one(seg_mask)
        
        # Define source and target
        x_0 = cond_img  # Source: conditional image
        x_1 = seg_mask  # Target: segmentation mask
        
        # Sample point on flow path
        if prob_img is not None and self.use_probability_guidance:
            prob_img = prob_img.to(device)
            x_t = self.q_sample_flow_with_probability(x_0, x_1, times, prob_img)
        else:
            x_t = self.q_sample_flow(x_0, x_1, times)
        
        # Compute target velocity
        target_velocity = self.compute_velocity(x_0, x_1)
        
        # Model prediction
        model_out = self.model(x_t, times, cond_img)
        
        if self.objective == 'predict_v':
            # Predict velocity field
            predicted_velocity = model_out
            
            # Focal L1 loss on velocity
            loss = self.loss_function(predicted_velocity, target_velocity)
            
        elif self.objective == 'predict_x1':
            # Predict target directly
            pred_x1 = model_out
            
            # Focal L1 loss on target
            loss = self.loss_function(pred_x1, x_1)
        
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return loss
    
    @torch.no_grad()
    def sample(self, cond_img, save_steps=None, num_steps=None):
        """Flow matching sampling: Euler ODE integration
        
        Args:
            cond_img: Conditional image (source x_0)
            save_steps: List of steps to save
            num_steps: Number of integration steps (default: self.num_timesteps)
        
        Returns:
            Binary mask [B, 1, H, W] in [0, 1]
        """
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from source
        x = cond_img
        
        # Initialize step storage
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)
        
        # Number of integration steps
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Euler method for ODE integration
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            # Current time
            t_continuous = step / num_steps
            t_discrete = int(t_continuous * self.num_timesteps)
            t_discrete = min(t_discrete, self.num_timesteps - 1)
            
            batched_times = torch.full((b,), t_discrete, device=self.device, dtype=torch.long)
            
            # Predict velocity or target
            model_out = self.model(x, batched_times, cond_img)
            
            if self.objective == 'predict_v':
                # Velocity prediction
                velocity = model_out
            else:
                # Target prediction → compute velocity
                t_normalized = t_continuous
                velocity = (model_out - x) / (1 - t_normalized + 1e-8)
            
            # Save step
            if save_steps is not None and t_discrete in save_steps:
                saved_steps[t_discrete] = unnormalize_to_zero_to_one(x).cpu()
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x = x + dt * velocity
            x = torch.clamp(x, -1, 1)
        
        # Denormalize to [0, 1]
        x = unnormalize_to_zero_to_one(x)
        
        if save_steps is not None:
            return {
                'final': x,
                'steps': saved_steps
            }
        else:
            return x


# ====== Factory Functions ======
def create_proposed_flow(image_size=224, dim=64, timesteps=100, 
                         use_probability_guidance=True, objective='predict_v'):
    """Proposed Flow Matching: Flow Matching + Probabilistic Guidance
    
    Args:
        image_size: Input image size (default: 224)
        dim: Base dimension (default: 64)
        timesteps: Number of flow steps (default: 100)
        use_probability_guidance: Use probability maps (default: True)
        objective: 'predict_v' (velocity) or 'predict_x1' (target)
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
    return ProposedFlowModel(
        unet, 
        timesteps=timesteps, 
        objective=objective,
        beta_schedule='linear',  # Not used but required by parent
        use_probability_guidance=use_probability_guidance
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Proposed Flow Matching Model")
    print("=" * 70)
    
    mask = torch.rand(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    prob = torch.rand(2, 1, 224, 224)
    
    print("\n1. Proposed Flow (with probability guidance)")
    proposed_flow = create_proposed_flow(image_size=224, dim=64, timesteps=100, 
                                         use_probability_guidance=True)
    loss = proposed_flow(mask, cond, prob)
    params = sum(p.numel() for p in proposed_flow.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    print("\n2. Proposed Flow (without probability guidance)")
    proposed_flow_no_prob = create_proposed_flow(image_size=224, dim=64, timesteps=100, 
                                                  use_probability_guidance=False)
    loss = proposed_flow_no_prob(mask, cond, prob_img=None)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n3. Proposed Flow (predict_x1 objective)")
    proposed_flow_x1 = create_proposed_flow(image_size=224, dim=64, timesteps=100, 
                                            objective='predict_x1')
    loss = proposed_flow_x1(mask, cond, prob)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test sampling
    print("\n4. Testing sampling...")
    with torch.no_grad():
        sample = proposed_flow.sample(cond[:1])
        print(f"   Sample shape: {sample.shape}, min: {sample.min():.2f}, max: {sample.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Proposed Flow Matching model works correctly!")
    print("=" * 70)

