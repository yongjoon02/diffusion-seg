"""
Evaluate all trained models on test data.

Usage Examples:
    # Basic usage (evaluate all models on octa500_3m)
    uv run python script/evaluate_supervised_models.py
    
    # Different dataset
    uv run python script/evaluate_supervised_models.py --data_name octa500_6m
    
    # Specific models only
    uv run python script/evaluate_supervised_models.py --models "cenet,csnet,unet3plus"
    
    # Custom output directory
    uv run python script/evaluate_supervised_models.py --data_name octa500_3m --output_dir results/my_experiment
    
    # Help
    uv run python script/evaluate_supervised_models.py --help

Parameters:
    --data_name: Dataset name (default: octa500_3m)
    --models: Comma-separated model names (default: all models)
    --output_dir: Output directory for results (default: results/supervised_model)

Output:
    - CSV file: {output_dir}/evaluation_results_{data_name}.csv
    - Individual predictions: lightning_logs/{model_name}/predictions/
"""

import autorootcwd
import pandas as pd
import torch
from pathlib import Path
import lightning as L
import click
from src.archs.supervised_model import SupervisedModel
from src.data.octa500 import OCTA500_3M_DataModule
from src.loggers import PredictionLogger


def find_checkpoint(data_name, model_name):
    """Find checkpoint for model."""
    # Find the latest checkpoint for this model
    checkpoint = None
    ckpt_files = list(Path("lightning_logs").glob(f"**/{data_name}/{model_name}/checkpoints/*.ckpt"))
    
    if ckpt_files:
        # Sort by modification time and get the latest
        checkpoint = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        return str(checkpoint)
    
    return None


def evaluate_model(data_name, model_name):
    """Evaluate single model."""
    checkpoint = find_checkpoint(data_name, model_name)
    if not checkpoint:
        print(f"No checkpoint found for {data_name}/{model_name}")
        return None
    
    print(f"Evaluating {model_name}...")
    
    try:
        # Load model from checkpoint
        model = SupervisedModel.load_from_checkpoint(checkpoint)
        
        # Setup data
        data_module = OCTA500_3M_DataModule(
            train_dir="data/OCTA500_3M/train",
            val_dir="data/OCTA500_3M/val", 
            test_dir="data/OCTA500_3M/test",
            crop_size=128,
            train_bs=1,
        )
        data_module.setup("test")
        
        # Create prediction logger
        logger = PredictionLogger(
            save_dir=f"lightning_logs/{data_name}/{model_name}",
            name="predictions",
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
            return test_metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
    
    return None


@click.command()
@click.option('--data_name', default='octa500_3m', help='Dataset name (e.g., octa500_3m, octa500_6m)')
@click.option('--models', default='cenet,csnet,aacaunet,unet3plus,vesselnet,transunet,dscnet', 
              help='Comma-separated list of model names')
@click.option('--output_dir', default='results/supervised_model', help='Output directory for results')
def main(data_name, models, output_dir):
    """Evaluate all models and create results table."""
    model_list = [m.strip() for m in models.split(',')]
    results = []
    
    print(f"Evaluating models on {data_name} dataset...")
    print(f"Models: {', '.join(model_list)}")
    
    for model in model_list:
        result = evaluate_model(data_name, model)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        df = df[["Model"] + [col for col in df.columns if col != "Model"]]
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save and print results
        results_path = f"{output_dir}/evaluation_results_{data_name}.csv"
        df.to_csv(results_path, index=False)
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        print(f"Results saved to: {results_path}")
    else:
        print("No evaluation results found!")


if __name__ == "__main__":
    main()
