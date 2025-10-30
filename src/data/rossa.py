"""ROSSA Retinal Vessel Segmentation Dataset

ROSSA dataset with manual and SAM annotations.
No label_prob - simpler than OCTA dataset.
"""
import autorootcwd
import os
import lightning as L
from monai.data import PILReader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
)
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from src.utils.visualize_dataloader import visualize_dataset
from src.utils.registry import DATASET_REGISTRY


class ROSSADataset(Dataset):
    """ROSSA Dataset (without label_prob)"""
    
    def __init__(self, path: str, augmentation: bool = False, crop_size: int = 128, 
                 num_samples_per_image: int = 1) -> None:
        """
        Args:
            path: Dataset split path (e.g., data/ROSSA/train_manual)
            augmentation: Whether to apply data augmentation
            crop_size: Random crop size
            num_samples_per_image: Number of samples per image (default: 1)
        """
        super().__init__()
        self.path = path
        self.augmentation = augmentation
        self.crop_size = crop_size
        self.num_samples_per_image = num_samples_per_image

        # Paths
        self.image_dir = os.path.join(path, "image")
        self.label_dir = os.path.join(path, "label")
        self.label_sauna_dir = os.path.join(path, "label_sauna")

        # List all image files
        self.image_files = sorted(os.listdir(self.image_dir))

        # Validate and collect valid data pairs
        self.data = []
        for file in self.image_files:
            image_path = os.path.join(self.image_dir, file)
            label_path = os.path.join(self.label_dir, file)
            label_sauna_path = os.path.join(self.label_sauna_dir, file)
            
            # Check if all required files exist
            if all(os.path.exists(p) for p in [image_path, label_path, label_sauna_path]):
                self.data.append({
                    "image": image_path,
                    "label": label_path,
                    "label_sauna": label_sauna_path,
                    "name": f"{os.path.basename(path)}/image/{file}",
                })
            else:
                missing = [p for p in [image_path, label_path, label_sauna_path] if not os.path.exists(p)]
                print(f"Warning: Missing files for {file}: {missing}")
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {path}. Check if image, label, and label_sauna directories exist.")

        self.image_loader = LoadImage(reader=PILReader(), image_only=True)
        
        # Default transforms (normalization)
        self.default_transforms = Compose([
            EnsureChannelFirstd(keys=["image", "label", "label_sauna"]),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            ScaleIntensityd(keys=["label"], minv=0.0, maxv=1.0),
            ScaleIntensityd(keys=["label_sauna"], minv=-1.0, maxv=1.0),  # SAUNA: 0~1 -> -1~1
        ])

        # Augmentation transforms
        if self.num_samples_per_image > 1:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=["image", "label", "label_sauna"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label", "label_sauna"], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=["image", "label", "label_sauna"], prob=0.5, max_k=3),
                RandCropByPosNegLabeld(
                    keys=["image", "label", "label_sauna"],
                    label_key="label",
                    spatial_size=(self.crop_size, self.crop_size),
                    pos=1,
                    neg=1,
                    num_samples=self.num_samples_per_image,
                ),
            ])
        else:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=["image", "label", "label_sauna"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label", "label_sauna"], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=["image", "label", "label_sauna"], prob=0.5, max_k=3),
                RandSpatialCropd(keys=["image", "label", "label_sauna"], 
                               roi_size=(self.crop_size, self.crop_size), 
                               random_size=False),
            ])

    def __len__(self):
        if self.augmentation and self.num_samples_per_image > 1:
            return len(self.data) * self.num_samples_per_image
        return len(self.data)

    def __getitem__(self, index):
        # Map index to actual data index
        if self.augmentation and self.num_samples_per_image > 1:
            actual_index = index // self.num_samples_per_image
        else:
            actual_index = index
        
        item = self.data[actual_index]
        
        try:
            # Load image, label, and label_sauna
            image = self.image_loader(item["image"])
            label = self.image_loader(item["label"])
            label_sauna = self.image_loader(item["label_sauna"])
        except Exception as e:
            raise RuntimeError(f"Failed to load data for {item['name']}: {e}")

        # Create data dict
        data = {"image": image, "label": label, "label_sauna": label_sauna, "name": item["name"]}

        # Apply default transforms
        data = self.default_transforms(data)

        # Apply augmentation if training
        if self.augmentation:
            data = self.augmentation_transforms(data)
            
            # Handle multiple samples per image
            if self.num_samples_per_image > 1 and isinstance(data, list):
                sample_idx = index % self.num_samples_per_image
                data = data[sample_idx]

        return data


class ROSSADataModule(L.LightningDataModule):
    """ROSSA DataModule combining train_manual and train_sam"""
    
    def __init__(self, 
                 train_manual_dir="data/ROSSA/train_manual",
                 train_sam_dir="data/ROSSA/train_sam",
                 val_dir="data/ROSSA/val", 
                 test_dir="data/ROSSA/test", 
                 crop_size=128, 
                 train_bs=8, 
                 num_samples_per_image=1,
                 name="rossa"):
        """
        Args:
            train_manual_dir: Manual annotation training data
            train_sam_dir: SAM annotation training data
            val_dir: Validation data
            test_dir: Test data
            crop_size: Crop size for training
            train_bs: Training batch size
            num_samples_per_image: Samples per image for augmentation
            name: Dataset name
        """
        super().__init__()
        self.train_manual_dir = train_manual_dir
        self.train_sam_dir = train_sam_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.crop_size = crop_size
        self.train_bs = train_bs
        self.name = name
        self.num_samples_per_image = num_samples_per_image

        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        """Setup datasets"""
        # Train: Combine manual and SAM annotations
        train_manual_dataset = ROSSADataset(
            self.train_manual_dir, 
            augmentation=True, 
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )
        train_sam_dataset = ROSSADataset(
            self.train_sam_dir, 
            augmentation=True, 
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )
        
        # Concatenate manual and SAM datasets
        self.train_dataset = ConcatDataset([train_manual_dataset, train_sam_dataset])
        
        # Val and Test
        self.val_dataset = ROSSADataset(
            self.val_dir, 
            augmentation=False, 
            crop_size=self.crop_size,
            num_samples_per_image=1
        )
        self.test_dataset = ROSSADataset(
            self.test_dir, 
            augmentation=False, 
            crop_size=self.crop_size,
            num_samples_per_image=1
        )
        
        print(f"ROSSA Dataset loaded:")
        print(f"  Train (manual): {len(train_manual_dataset)} samples")
        print(f"  Train (SAM): {len(train_sam_dataset)} samples")
        print(f"  Train (total): {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool = False):
        """Create DataLoader with common settings"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=2
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, self.train_bs, shuffle=True)
    
    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, 1, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, 1, shuffle=False)


@DATASET_REGISTRY.register(name='rossa')
class ROSSA_DataModule(ROSSADataModule):
    def __init__(self, 
                 train_manual_dir="data/ROSSA/train_manual",
                 train_sam_dir="data/ROSSA/train_sam",
                 val_dir="data/ROSSA/val", 
                 test_dir="data/ROSSA/test", 
                 crop_size=128, 
                 train_bs=8, 
                 num_samples_per_image=1):
        super().__init__(
            train_manual_dir=train_manual_dir,
            train_sam_dir=train_sam_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            name='rossa'
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ROSSA Dataset")
    print("=" * 70)
    
    datamodule = DATASET_REGISTRY.get('rossa')()
    datamodule.setup()
    
    # Visualize
    visualize_dataset(datamodule.train_dataloader(), "rossa_train")
    visualize_dataset(datamodule.val_dataloader(), "rossa_val")
    visualize_dataset(datamodule.test_dataloader(), "rossa_test")
    
    print("\n✓ ROSSA dataset works correctly!")
