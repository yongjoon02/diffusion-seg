"""
Evaluate all trained diffusion models on test data with ensemble sampling.

Uses multiple sampling runs and averages predictions for more stable results.

Usage Examples:
    # Basic usage (evaluate all diffusion models on octa500_3m with 10 ensemble samples)
    uv run python script/evaluate_diffusion_ensemble_models.py
    
    # Different dataset
    uv run python script/evaluate_diffusion_ensemble_models.py --data_name octa500_6m
    
    # Custom ensemble count
    uv run python script/evaluate_diffusion_ensemble_models.py --num_ensemble 5
    
    # Specific models only
    uv run python script/evaluate_diffusion_ensemble_models.py --models "segdiff,medsegdiff,berdiff"
    
    # Custom output directory
    uv run python script/evaluate_diffusion_ensemble_models.py --data_name octa500_3m --output_dir results/ensemble_experiment
    
    # Help
    uv run python script/evaluate_diffusion_ensemble_models.py --help

Parameters:
    --data_name: Dataset name (default: octa500_3m)
    --models: Comma-separated model names (default: all diffusion models)
    --num_ensemble: Number of ensemble samples (default: 10)
    --output_dir: Output directory for results (default: results/diffusion_ensemble)

Output:
    - CSV file: {output_dir}/evaluation_results_{data_name}_ensemble{N}.csv
    - Individual predictions: lightning_logs/{model_name}/predictions_ensemble{N}/
"""

import autorootcwd
import pandas as pd
import torch
from pathlib import Path
import lightning as L
import click
from src.archs.diffusion_model import DiffusionModel
from src.data.octa500 import OCTA500_3M_DataModule, OCTA500_6M_DataModule
from src.loggers import PredictionLogger


def find_checkpoint(data_name, model_name):
    """Find checkpoint for diffusion model."""
    # Find the latest checkpoint for this model
    checkpoint = None
    ckpt_files = list(Path("lightning_logs").glob(f"**/{data_name}/{model_name}/checkpoints/*.ckpt"))
    
    if ckpt_files:
        # Sort by modification time and get the latest
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
            crop_size=224,  # Diffusion models use 224
            train_bs=1,
        )
    elif data_name == "octa500_6m":
        return OCTA500_6M_DataModule(
            train_dir="data/OCTA500_6M/train",
            val_dir="data/OCTA500_6M/val",
            test_dir="data/OCTA500_6M/test", 
            crop_size=224,  # Diffusion models use 224
            train_bs=1,
        )
    else:
        raise ValueError(f"Unknown data_name: {data_name}")


def evaluate_model(data_name, model_name, num_ensemble):
    """Evaluate single diffusion model with ensemble sampling."""
    checkpoint = find_checkpoint(data_name, model_name)
    if not checkpoint:
        print(f"No checkpoint found for {data_name}/{model_name}")
        return None
    
    print(f"Evaluating {model_name} with {num_ensemble} ensemble samples...")
    
    try:
        # Load model from checkpoint and set num_ensemble
        model = DiffusionModel.load_from_checkpoint(
            checkpoint,
            num_ensemble=num_ensemble
        )
        
        # Setup data
        data_module = get_data_module(data_name)
        data_module.setup("test")
        
        # Create prediction logger
        logger = PredictionLogger(
            save_dir=f"lightning_logs/{data_name}/{model_name}",
            name=f"predictions_ensemble{num_ensemble}",
            version=None
        )
        
        # Create trainer with prediction logger
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        
        # Run test
        results = trainer.test(model, data_module)
        
        if results and len(results) > 0:
            test_metrics = results[0]
            test_metrics["Model"] = model_name
            test_metrics["Ensemble"] = num_ensemble
            return test_metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return None


@click.command()
@click.option('--data_name', default='octa500_3m', help='Dataset name (e.g., octa500_3m, octa500_6m)')
@click.option('--models', default='segdiff,medsegdiff,berdiff', 
              help='Comma-separated list of diffusion model names')
@click.option('--num_ensemble', default=10, type=int,
              help='Number of ensemble samples (default: 20)')
@click.option('--output_dir', default='results/diffusion_ensemble', 
              help='Output directory for results')
def main(data_name, models, num_ensemble, output_dir):
    """Evaluate all diffusion models with ensemble sampling and create results table."""
    model_list = [m.strip() for m in models.split(',')]
    results = []
    
    print("=" * 80)
    print(f"DIFFUSION MODEL ENSEMBLE EVALUATION")
    print("=" * 80)
    print(f"Dataset: {data_name}")
    print(f"Models: {', '.join(model_list)}")
    print(f"Ensemble samples: {num_ensemble}")
    print("=" * 80)
    print()
    
    for model in model_list:
        result = evaluate_model(data_name, model, num_ensemble)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        df = df[["Model", "Ensemble"] + [col for col in df.columns if col not in ["Model", "Ensemble"]]]
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save and print results
        results_path = f"{output_dir}/evaluation_results_{data_name}_ensemble{num_ensemble}.csv"
        df.to_csv(results_path, index=False)
        print("\n" + "="*80)
        print("DIFFUSION MODEL ENSEMBLE EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        print(f"Results saved to: {results_path}")
    else:
        print("No evaluation results found!")


if __name__ == "__main__":
    main()

