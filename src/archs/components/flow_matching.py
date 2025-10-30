"""Flow Matching for Medical Image Segmentation using Distance Transforms (FlowSDF)

Flow Matching-based segmentation using Signed Distance Functions (SDF).
Uses Optimal Transport Flow instead of diffusion process.

Key differences from Diffusion:
1. Flow Matching: Deterministic ODE path (no noise)
2. SDF representation: Distance transform instead of binary mask
3. Conditional Flow: Image-conditioned straight paths
4. Loss: Flow matching loss (velocity field prediction)

Based on FlowSDF (IJCV 2025).
Reference: https://github.com/leabogensperger/FlowSDF
"""
import autorootcwd
import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import namedtuple
from scipy.ndimage import distance_transform_edt

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


def compute_sdf(mask):
    """Compute Signed Distance Function from binary mask
    
    Args:
        mask: Binary mask tensor [B, 1, H, W] in range [0, 1]
    
    Returns:
        SDF tensor [B, 1, H, W] (positive inside, negative outside)
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
        
        sdfs.append(torch.from_numpy(sdf).unsqueeze(0))
    
    sdf_tensor = torch.stack(sdfs, dim=0).to(mask.device).float()
    return sdf_tensor


def sdf_to_mask(sdf, threshold=0.0):
    """Convert SDF to binary mask
    
    Args:
        sdf: SDF tensor [B, 1, H, W]
        threshold: Threshold for binarization (default: 0.0)
    
    Returns:
        Binary mask [B, 1, H, W] in range [0, 1]
    """
    return (sdf > threshold).float()


# ====== Flow Matching Process ======
class FlowMatchingModel(GaussianDiffusionModel):
    """FlowSDF: Flow Matching for Segmentation using SDF
    
    Flow Matching with Signed Distance Functions for medical image segmentation.
    Uses Optimal Transport (OT) flow instead of diffusion.
    
    Key concepts:
    1. x_0 = cond_img (source: input image)
    2. x_1 = SDF(mask) (target: signed distance function)
    3. x_t = (1-t) * x_0 + t * x_1 (linear interpolation)
    4. v_t = x_1 - x_0 (constant velocity field for straight paths)
    """
    
    def __init__(self, model, timesteps=1000, sampling_timesteps=None, 
                 objective='predict_v', beta_schedule='linear', loss_type='mse',
                 sdf_normalize=True):
        # Initialize parent with dummy objective (we'll override)
        super().__init__(model, timesteps, sampling_timesteps, 'predict_x0', beta_schedule, loss_type)
        
        self.objective = objective  # 'predict_v' or 'predict_x1'
        self.sdf_normalize = sdf_normalize
        
        # Loss functions
        self.bce_dice_loss = BceDiceLoss(wb=1.0, wd=1.0)
    
    def get_sdf_from_mask(self, mask):
        """Convert binary mask to normalized SDF
        
        Args:
            mask: Binary mask [B, 1, H, W] in [0, 1]
        
        Returns:
            Normalized SDF [B, 1, H, W] in [-1, 1] if sdf_normalize else raw SDF
        """
        sdf = compute_sdf(mask)
        
        if self.sdf_normalize:
            # Normalize to [-1, 1] for stability
            # Use tanh-based normalization
            sdf = torch.tanh(sdf / 10.0)  # Scale factor: 10 pixels
        
        return sdf
    
    def get_mask_from_sdf(self, sdf):
        """Convert SDF back to binary mask
        
        Args:
            sdf: SDF tensor [B, 1, H, W]
        
        Returns:
            Binary mask [B, 1, H, W] in [0, 1]
        """
        return sdf_to_mask(sdf, threshold=0.0)
    
    def q_sample(self, x_start, t, noise):
        """Flow matching forward: linear interpolation from x_0 to x_1
        
        Args:
            x_start: Target SDF (x_1)
            t: Time in [0, 1]
            noise: Source image (x_0)
        
        Returns:
            x_t = (1-t) * x_0 + t * x_1
        """
        # Convert t to [0, 1] range
        t_normalized = t.float() / self.num_timesteps
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        # x_0 = noise (source: cond_img)
        # x_1 = x_start (target: SDF)
        t_expanded = t_normalized.view(-1, 1, 1, 1)
        
        x_t = (1 - t_expanded) * noise + t_expanded * x_start
        
        return x_t
    
    def compute_velocity(self, x_0, x_1):
        """Compute constant velocity field for straight path
        
        For straight paths: v_t = x_1 - x_0 (constant)
        
        Args:
            x_0: Source (cond_img)
            x_1: Target (SDF)
        
        Returns:
            Velocity field v_t
        """
        return x_1 - x_0
    
    def forward(self, seg_mask, cond_img, gamma=1.0):
        """FlowSDF Loss Function
        
        Loss types:
        - 'mse': MSE on velocity field
        - 'hybrid': MSE + BCE+Dice on final mask
        """
        device = self.device
        seg_mask, cond_img = seg_mask.to(device), cond_img.to(device)
        
        b = seg_mask.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Convert mask to SDF (target x_1)
        sdf_target = self.get_sdf_from_mask(seg_mask)
        
        # x_0 = cond_img (source)
        # x_1 = sdf_target (target)
        x_0 = cond_img
        x_1 = sdf_target
        
        # Sample point on flow path: x_t = (1-t) * x_0 + t * x_1
        x_t = self.q_sample(x_start=x_1, t=times, noise=x_0)
        
        # Compute target velocity: v_t = x_1 - x_0 (constant for straight paths)
        target_velocity = self.compute_velocity(x_0, x_1)
        
        # Model prediction: predict velocity field
        predicted_velocity = self.model(x_t, times, cond_img)
        
        if self.loss_type == 'mse':
            # Flow matching loss: MSE on velocity field
            return F.mse_loss(predicted_velocity, target_velocity)
        
        elif self.loss_type == 'hybrid':
            # Hybrid: Flow matching loss + Segmentation loss
            # Loss1: MSE on velocity
            velocity_loss = F.mse_loss(predicted_velocity, target_velocity)
            
            # Loss2: Predict x_1 from velocity and compute mask loss
            # x_1 = x_t + (1-t) * v_t (for straight paths)
            t_normalized = times.float() / self.num_timesteps
            t_expanded = t_normalized.view(-1, 1, 1, 1)
            
            pred_x1 = x_t + (1 - t_expanded) * predicted_velocity
            
            # Convert SDF to mask
            pred_mask = self.get_mask_from_sdf(pred_x1)
            target_mask = seg_mask
            
            # BCE + Dice on mask
            seg_loss = self.bce_dice_loss(pred_mask, target_mask)
            
            # Combine
            alpha, beta = 0.5, 0.5
            return alpha * velocity_loss + beta * seg_loss
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    @torch.no_grad()
    def sample(self, cond_img, save_steps=None, num_steps=None):
        """Flow matching sampling: ODE integration from x_0 to x_1
        
        Args:
            cond_img: Conditional image (source x_0)
            save_steps: List of steps to save
            num_steps: Number of integration steps (default: self.num_timesteps)
        
        Returns:
            Binary mask [B, 1, H, W] in [0, 1]
        """
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape
        
        # Start from source (conditional image)
        x = cond_img
        
        # Initialize step storage if needed
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)
        
        # Number of integration steps
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Euler method for ODE integration
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_continuous = step / num_steps
            t_discrete = int(t_continuous * self.num_timesteps)
            t_discrete = min(t_discrete, self.num_timesteps - 1)
            
            batched_times = torch.full((b,), t_discrete, device=self.device, dtype=torch.long)
            
            # Predict velocity field
            preds = self.model_predictions(x, batched_times, cond_img, clip_x_start=False)
            predicted_velocity = self.model(x, batched_times, cond_img)
            
            # Save step
            if save_steps is not None and t_discrete in save_steps:
                # Convert current x to mask for visualization
                current_mask = self.get_mask_from_sdf(x)
                saved_steps[t_discrete] = current_mask.cpu()
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x = x + dt * predicted_velocity
        
        # Final x should be approximately x_1 (SDF)
        # Convert SDF to binary mask
        final_mask = self.get_mask_from_sdf(x)
        
        if save_steps is not None:
            return {
                'final': final_mask,
                'steps': saved_steps
            }
        else:
            return final_mask
    
    def model_predictions(self, x, t, c, clip_x_start=False):
        """Model predictions for flow matching
        
        Predicts velocity field v_t
        """
        model_output = self.model(x, t, c)
        
        if self.objective == 'predict_v':
            # Predict velocity directly
            predicted_velocity = model_output
            
            # Compute x_1 from velocity (for compatibility)
            t_normalized = t.float() / self.num_timesteps
            t_expanded = t_normalized.view(-1, 1, 1, 1)
            
            # x_1 = x + (1-t) * v
            pred_x1 = x + (1 - t_expanded) * predicted_velocity
            
            return ModelPrediction(predicted_velocity, pred_x1)
        
        elif self.objective == 'predict_x1':
            # Predict target directly
            pred_x1 = model_output
            
            # Compute velocity from prediction
            t_normalized = t.float() / self.num_timesteps
            t_expanded = t_normalized.view(-1, 1, 1, 1)
            
            # v = (x_1 - x) / (1-t)
            predicted_velocity = (pred_x1 - x) / (1 - t_expanded + 1e-8)
            
            return ModelPrediction(predicted_velocity, pred_x1)
        
        else:
            raise ValueError(f'unknown objective {self.objective}')


# ====== Factory Functions ======
def create_flowsdf(image_size=224, dim=64, timesteps=100, loss_type='hybrid', sdf_normalize=True):
    """FlowSDF: Flow Matching + SDF for Segmentation
    
    Args:
        image_size: Input image size (default: 224)
        dim: Base dimension (default: 64)
        timesteps: Number of flow steps (default: 100)
        loss_type: 'mse' or 'hybrid' (default, flow + seg loss)
        sdf_normalize: Normalize SDF to [-1, 1] (default: True)
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
    return FlowMatchingModel(
        unet, 
        timesteps=timesteps, 
        objective='predict_v',  # Predict velocity field
        beta_schedule='linear',  # Not used in flow matching, but required by parent
        loss_type=loss_type,
        sdf_normalize=sdf_normalize
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FlowSDF Model")
    print("=" * 70)
    
    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)
    mask = (torch.randn(2, 1, 224, 224) > 0).float()
    
    print("\n1. FlowSDF (Flow Matching + SDF)")
    flowsdf = create_flowsdf(image_size=224, dim=64, timesteps=100)
    loss = flowsdf(mask, cond)
    params = sum(p.numel() for p in flowsdf.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")
    
    # Test SDF computation
    print("\n2. Testing SDF computation...")
    sdf = flowsdf.get_sdf_from_mask(mask)
    print(f"   SDF shape: {sdf.shape}, min: {sdf.min():.2f}, max: {sdf.max():.2f}")
    
    # Test mask recovery
    recovered_mask = flowsdf.get_mask_from_sdf(sdf)
    accuracy = (recovered_mask == mask).float().mean()
    print(f"   Mask recovery accuracy: {accuracy:.4f}")
    
    # Test sampling
    print("\n3. Testing sampling...")
    with torch.no_grad():
        sample = flowsdf.sample(cond[:1])
        print(f"   Sample shape: {sample.shape}, min: {sample.min():.2f}, max: {sample.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✓ FlowSDF model works correctly!")
    print("=" * 70)

