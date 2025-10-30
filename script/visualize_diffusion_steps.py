"""
Visualize diffusion sampling steps for a single sample.

이 스크립트는 diffusion model의 sampling 과정을 단계별로 시각화합니다.
좌우로 나열된 이미지: Input Image → Ground Truth → Step 100 → Step 90 → ... → Final

Usage Examples:
    # SegDiff sampling 단계 시각화
    uv run python script/visualize_diffusion_steps.py --model_name segdiff --sample_idx 0
    
    # 커스텀 스텝으로 MedSegDiff 시각화
    uv run python script/visualize_diffusion_steps.py --model_name medsegdiff --sample_idx 5 --steps "0,20,40,60,80,100"
    
    # 6M 데이터셋으로 ColdDiffusion 시각화
    uv run python script/visualize_diffusion_steps.py --model_name colddiff --data_name octa500_6m --sample_idx 0
    
    # 모든 모델의 여러 샘플 일괄 시각화
    ./script/visualize_all_diffusion_steps.sh
    
    # 도움말 보기
    uv run python script/visualize_diffusion_steps.py --help

Parameters:
    --model_name: Diffusion model name (segdiff, medsegdiff, colddiff)
    --data_name: Dataset name (default: octa500_3m)
    --sample_idx: 시각화할 샘플 인덱스 (default: 0)
    --steps: 저장할 timestep들 (default: "0,10,20,30,40,50,60,70,80,90,100")
    --output_dir: 시각화 결과 저장 디렉토리 (default: results/diffusion_model/visualization)

Output:
    - 시각화 이미지: {output_dir}/{model_name}_sample_{sample_idx}_steps.png
    - 좌우 나열: Input → Ground Truth → Step 100 → Step 90 → ... → Final
    - 제목: {model_name} ({data_name}) - Sample {sample_idx}
"""

import autorootcwd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lightning as L
import click
from src.archs.diffusion_model import DiffusionModel
from src.data.octa500 import OCTA500_3M_DataModule, OCTA500_6M_DataModule
import torchvision.transforms as T


def find_checkpoint(data_name, model_name):
    """Find checkpoint for diffusion model."""
    checkpoint = None
    ckpt_files = list(Path("lightning_logs").glob(f"**/{data_name}/{model_name}/checkpoints/*.ckpt"))
    
    if ckpt_files:
        checkpoint = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        return str(checkpoint)
    
    return None


def get_data_module(data_name):
    """Get appropriate data module based on data_name."""
    if data_name == "octa500_3m":
        return OCTA500_3M_DataModule(
            train_dir="data/OCTA500_3M/train",
            val_dir="data/OCTA500_3M/val", 
            test_dir="data/OCTA500_3M/test",
            crop_size=224,
            train_bs=1,
        )
    elif data_name == "octa500_6m":
        return OCTA500_6M_DataModule(
            train_dir="data/OCTA500_6M/train",
            val_dir="data/OCTA500_6M/val",
            test_dir="data/OCTA500_6M/test", 
            crop_size=224,
            train_bs=1,
        )
    else:
        raise ValueError(f"Unknown data_name: {data_name}")


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # Remove channel dimension
    return tensor.cpu().numpy()


def create_step_visualization(image, label, steps_dict, model_name, data_name, sample_idx, output_dir):
    """Create horizontal visualization of diffusion steps."""
    # Convert tensors to numpy
    image_np = tensor_to_numpy(image)
    label_np = tensor_to_numpy(label)
    
    # Create figure with subplots
    n_steps = len(steps_dict) + 2  # +2 for input image and ground truth
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3))
    
    # Set main title
    fig.suptitle(f'{model_name.upper()} ({data_name}) - Sample {sample_idx}', fontsize=14, fontweight='bold')
    
    if n_steps == 1:
        axes = [axes]
    
    # Plot input image
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('Input Image', fontsize=10)
    axes[0].axis('off')
    
    # Plot ground truth
    axes[1].imshow(label_np, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=10)
    axes[1].axis('off')
    
    # Plot diffusion steps
    sorted_steps = sorted(steps_dict.keys(), reverse=True)  # From high to low timestep
    for i, (timestep, step_tensor) in enumerate(zip(sorted_steps, [steps_dict[t] for t in sorted_steps])):
        step_np = tensor_to_numpy(step_tensor)
        axes[i + 2].imshow(step_np, cmap='gray')
        
        # 마지막 step은 "Final"로 표시, 나머지는 "Step X"로 표시
        if i == len(sorted_steps) - 1:
            axes[i + 2].set_title(f'Final (Step {timestep})', fontsize=10)
        else:
            axes[i + 2].set_title(f'Step {timestep}', fontsize=10)
        axes[i + 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dir / f"{model_name}_sample_{sample_idx}_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved step visualization to: {output_path}")


@click.command()
@click.option('--model_name', required=True, help='Diffusion model name (segdiff, medsegdiff, colddiff)')
@click.option('--data_name', default='octa500_3m', help='Dataset name (e.g., octa500_3m, octa500_6m)')
@click.option('--sample_idx', default=0, help='Sample index to visualize')
@click.option('--steps', default='0,10,20,30,40,50,60,70,80,90,100', 
              help='Comma-separated timesteps to save')
@click.option('--output_dir', default='results/diffusion_model/visualization', help='Output directory for visualizations')
def main(model_name, data_name, sample_idx, steps, output_dir):
    """Visualize diffusion sampling steps for a single sample."""
    # Parse steps
    step_list = [int(s.strip()) for s in steps.split(',')]
    
    print(f"Visualizing {model_name} sampling steps on {data_name} dataset...")
    print(f"Sample index: {sample_idx}")
    print(f"Steps to save: {step_list}")
    
    # Find checkpoint
    checkpoint = find_checkpoint(data_name, model_name)
    if not checkpoint:
        print(f"No checkpoint found for {data_name}/{model_name}")
        return
    
    print(f"Loading checkpoint: {checkpoint}")
    
    try:
        # Load model
        model = DiffusionModel.load_from_checkpoint(checkpoint)
        model.eval()
        
        # Setup data
        data_module = get_data_module(data_name)
        data_module.setup("test")
        test_loader = data_module.test_dataloader()
        
        # Get specific sample
        if sample_idx >= len(test_loader.dataset):
            print(f"Sample index {sample_idx} out of range (max: {len(test_loader.dataset)-1})")
            return
        
        sample = test_loader.dataset[sample_idx]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        label = sample['label'].unsqueeze(0)  # Add batch dimension
        
        print(f"Processing sample {sample_idx}...")
        
        # Sample with step saving
        with torch.no_grad():
            result = model.sample(image, save_steps=step_list)
        
        # Handle different return types
        if isinstance(result, dict):
            final_pred = result['final']
            steps_dict = result['steps']
        else:
            final_pred = result
            steps_dict = {}
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization
        create_step_visualization(image, label, steps_dict, model_name, data_name, sample_idx, output_dir)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
