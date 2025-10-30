"""Supervised training script."""

import os
os.environ['NCCL_P2P_DISABLE'] = '1'
import torch
torch.set_float32_matmul_precision('medium')

import sys
import yaml
import autorootcwd
from lightning.pytorch.cli import LightningCLI
from src.archs.supervised_model import SupervisedModel
from src.utils.registry import DATASET_REGISTRY


if __name__ == "__main__":
    # Add default config if not provided
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', 'configs/octa500_3m_supervised_models.yaml'])
    
    # Extract data_name from config file
    config_path = None
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            config_path = sys.argv[config_idx + 1]
    
    data_name = None
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                data_name = config.get('data', {}).get('name')
                
                # Remove 'name' from data config before passing to LightningCLI
                if 'data' in config and 'name' in config['data']:
                    del config['data']['name']
                    # Write modified config to a temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                        yaml.dump(config, tmp)
                        temp_config_path = tmp.name
                    # Replace config path in sys.argv
                    sys.argv[sys.argv.index(config_path)] = temp_config_path
        except Exception as e:
            print(f"Warning: Could not parse config file {config_path}: {e}")
    
    if data_name is None:
        print("Error: data.name not found in config file")
        sys.exit(1)
    
    # Select appropriate DataModule
    DataModuleClass = DATASET_REGISTRY.get(data_name)
    
    # Convert --arch_name to LightningCLI overrides
    if '--arch_name' in sys.argv:
        arch_idx = sys.argv.index('--arch_name')
        if arch_idx + 1 < len(sys.argv):
            arch_name = sys.argv[arch_idx + 1]
            # Remove --arch_name and its value
            sys.argv.pop(arch_idx)
            sys.argv.pop(arch_idx)
            # Add LightningCLI overrides
            sys.argv.extend(['--model.arch_name', arch_name])
            # Set TensorBoard logger name and version
            sys.argv.extend(['--trainer.logger.init_args.name', f"{data_name}"])
            sys.argv.extend(['--trainer.logger.init_args.version', f"{arch_name}"])
    
    cli = LightningCLI(
        SupervisedModel,
        DataModuleClass,
        save_config_kwargs={'overwrite': True},
    )
