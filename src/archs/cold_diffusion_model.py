"""Diffusion models for vessel segmentation.
Based on supervised_model.py structure with SegDiff and MedSegDiff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.archs.components.diffusion_unet import (
    SegDiffUNet, MedSegDiffUNet, create_segdiff, create_medsegdiff
)
from src.metrics import (
    Dice, Precision, Recall, Specificity, JaccardIndex,
    clDice, Betti0Error, Betti1Error
)


MODEL_REGISTRY = {
    'segdiff': create_segdiff,
    'medsegdiff': create_medsegdiff,
}


class ColdDiffusionModel(L.LightningModule):
    """Cold diffusion segmentation model with sliding window inference."""
    
    def __init__(
        self,
        arch_name: str = 'segdiff',
        image_size: int = 224,
        dim: int = 64,
        timesteps: int = 1000,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        
        # Create diffusion model
        if arch_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(MODEL_REGISTRY.keys())}")
        
        create_fn = MODEL_REGISTRY[arch_name]
        self.diffusion_model = create_fn(image_size=image_size, dim=dim, timesteps=timesteps)
        
        # Sliding window inferer for validation
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )
        
        # Metrics
        self.val_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
    
    def forward(self, img: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns loss during training."""
        return self.diffusion_model(img, cond_img)
    
    def sample(self, cond_img: torch.Tensor) -> torch.Tensor:
        """Sample from diffusion model (inference).
        
        This function is called by sliding window inferer for each patch.
        Each patch goes through the full diffusion sampling process.
        """
        return self.diffusion_model.sample(cond_img)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Convert to float [0, 1] if needed
        if labels.dtype != torch.float32:
            labels = labels.float()
        if labels.max() > 1:
            labels = labels / 255.0
        
        # Compute diffusion loss
        loss = self(labels, images)
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Sliding window inference - sample 함수가 각 패치별로 diffusion sampling 수행
        pred_masks = self.inferer(images, self.sample)
        
        # Convert predictions to class indices
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        preds = (pred_masks > 0.5).long()
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        return general_metrics['dice']
    
    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Sliding window inference - sample 함수가 각 패치별로 diffusion sampling 수행
        pred_masks = self.inferer(images, self.sample)
        
        # Convert predictions to class indices
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        preds = (pred_masks > 0.5).long()
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})
        
        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks_binary = (preds > 0).float()
            label_masks = (labels > 0).float()
            
            # Prepare metrics for each sample
            sample_metrics = []
            for i in range(images.shape[0]):
                sample_metric = {}
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in general_metrics.items()})
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in vessel_metrics.items()})
                sample_metrics.append(sample_metric)
            
            # Save predictions
            self.trainer.logger.save_predictions(
                sample_names, images, pred_masks_binary, label_masks, sample_metrics
            )
        
        return general_metrics['dice']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=20,
            factor=0.5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train/loss',
                'interval': 'epoch',
            }
        }
