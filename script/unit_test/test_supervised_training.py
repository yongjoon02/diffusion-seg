"""Test supervised training with LightningCLI."""

import os
import autorootcwd
from lightning.pytorch.cli import LightningCLI
from src.archs.supervised_model import SupervisedModel
from src.data.octa500 import OCTADataModule
# Disable NCCL P2P to avoid multi-GPU stuck issues
os.environ['NCCL_P2P_DISABLE'] = '1'


if __name__ == "__main__":
    cli = LightningCLI(
        SupervisedModel,
        OCTADataModule,
        save_config_kwargs={'overwrite': True},
    )

