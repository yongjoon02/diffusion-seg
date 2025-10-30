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
from torch.utils.data import DataLoader, Dataset
from src.utils.visualize_dataloader import visualize_dataset
from src.utils.registry import DATASET_REGISTRY

class OCTADataset(Dataset):    
    def __init__(self, path: str, augmentation: bool = False, crop_size: int = 128, 
                 num_samples_per_image: int = 1) -> None:
        """
        Args:
            path (str): 데이터셋 분할 경로 (예: data/OCTA500_3M/train)
            augmentation (bool): 데이터 증강 적용 여부 (훈련시에만 True)
            crop_size (int): 랜덤 크롭할 이미지 크기 (roi_size)
            num_samples_per_image (int): 한 이미지당 생성할 샘플 개수 (기본: 1)
        """
        super().__init__()
        self.path = path
        self.augmentation = augmentation
        self.crop_size = crop_size
        self.num_samples_per_image = num_samples_per_image

        # Paths to image and label directories
        self.image_dir = os.path.join(path, "image")
        self.label_dir = os.path.join(path, "label")
        self.label_prob_dir = os.path.join(path, "label_prob")
        self.label_sauna_dir = os.path.join(path, "label_sauna")

        # List all image files
        self.image_files = sorted(os.listdir(self.image_dir))

        # Validate and collect valid data triplets
        self.data = []
        for file in self.image_files:
            image_path = os.path.join(self.image_dir, file)
            label_path = os.path.join(self.label_dir, file)
            label_prob_path = os.path.join(self.label_prob_dir, file)
            label_sauna_path = os.path.join(self.label_sauna_dir, file)
            
            # Check if all required files exist
            if all(os.path.exists(p) for p in [image_path, label_path, label_prob_path, label_sauna_path]):
                self.data.append({
                    "image": image_path,
                    "label": label_path,
                    "label_prob": label_prob_path,
                    "label_sauna": label_sauna_path,
                    "name": f"{os.path.basename(path)}/image/{file}",
                })
            else:
                missing = [p for p in [image_path, label_path, label_prob_path, label_sauna_path] if not os.path.exists(p)]
                print(f"Warning: Missing files for {file}: {missing}")
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {path}. Check if image, label, label_prob, and label_sauna directories exist.")

        self.image_loader = LoadImage(reader=PILReader(), image_only=True)
        
        self.default_transforms = Compose([
            EnsureChannelFirstd(keys=["image", "label", "label_prob", "label_sauna"]),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            ScaleIntensityd(keys=["label", "label_prob"], minv=0.0, maxv=1.0),
            ScaleIntensityd(keys=["label_sauna"], minv=-1.0, maxv=1.0),  # SAUNA: 0~1 -> -1~1
        ])

        # Use RandCropByPosNegLabeld for multiple samples per image
        if self.num_samples_per_image > 1:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=["image", "label", "label_prob", "label_sauna"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label", "label_prob", "label_sauna"], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=["image", "label", "label_prob", "label_sauna"], prob=0.5, max_k=3),
                RandCropByPosNegLabeld(
                    keys=["image", "label", "label_prob", "label_sauna"],
                    label_key="label",
                    spatial_size=(self.crop_size, self.crop_size),
                    pos=1,  # Positive sample ratio
                    neg=1,  # Negative sample ratio
                    num_samples=self.num_samples_per_image,  # Multiple crops per image
                ),
            ])
        else:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=["image", "label", "label_prob", "label_sauna"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label", "label_prob", "label_sauna"], spatial_axis=1, prob=0.5),
                RandRotate90d(keys=["image", "label", "label_prob", "label_sauna"], prob=0.5, max_k=3),
                RandSpatialCropd(keys=["image", "label", "label_prob", "label_sauna"], 
                               roi_size=(self.crop_size, self.crop_size), 
                               random_size=False),
            ])

    def __len__(self):
        """
        데이터셋의 총 샘플 개수를 반환합니다.
        PyTorch Dataset의 필수 구현 메서드입니다.
        
        num_samples_per_image > 1이면 실제 데이터 개수 * num_samples_per_image 반환
        """
        if self.augmentation and self.num_samples_per_image > 1:
            return len(self.data) * self.num_samples_per_image
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index: 가져올 샘플의 인덱스
            
        Returns:
            dict: 이미지, 라벨, 확률맵, 파일명을 포함한 딕셔너리
        """
        # Map index to actual data index when using multiple samples per image
        if self.augmentation and self.num_samples_per_image > 1:
            actual_index = index // self.num_samples_per_image
        else:
            actual_index = index
        
        item = self.data[actual_index]
        
        try:
            # 각 파일을 MONAI의 LoadImage로 로드
            image = self.image_loader(item["image"])
            label = self.image_loader(item["label"])
            label_prob = self.image_loader(item["label_prob"])
            label_sauna = self.image_loader(item["label_sauna"])
        except Exception as e:
            raise RuntimeError(f"Failed to load data for {item['name']}: {e}")

        # 딕셔너리 형태로 데이터 구성 (MONAI 변환에서 키를 사용)
        data = {"image": image, "label": label, "label_prob": label_prob, "label_sauna": label_sauna, "name": item["name"]}

        # 기본 전처리 적용 (정규화, 채널 순서 변경 등)
        data = self.default_transforms(data)

        # 훈련시에만 데이터 증강 적용
        if self.augmentation:
            data = self.augmentation_transforms(data)
            
            # RandCropByPosNegLabeld returns list of dicts when num_samples > 1
            # Extract the appropriate sample
            if self.num_samples_per_image > 1 and isinstance(data, list):
                sample_idx = index % self.num_samples_per_image
                data = data[sample_idx]

        return data

class OCTADataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule 클래스입니다.
    
    Lightning DataModule을 사용하는 이유:
    1. 데이터 로딩 로직을 모델 코드와 분리하여 재사용성 향상
    2. 분산 학습시 데이터 분할을 자동으로 처리
    3. train/val/test 데이터로더를 체계적으로 관리
    4. Lightning Trainer와 완벽하게 통합되어 사용하기 쉬움
    5. 코드의 일관성과 가독성 향상
    
    기존 방식 (번거로움):
        train_dataset = OCTADataset(train_path)
        val_dataset = OCTADataset(val_path)
        train_loader = DataLoader(train_dataset, ...)
        val_loader = DataLoader(val_dataset, ...)
        # 모델 코드에서 각각 관리해야 함
    
    DataModule 사용 (깔끔함):
        datamodule = OCTADataModule(config)
        trainer.fit(model, datamodule)  # 알아서 train/val 로더 사용
        trainer.test(model, datamodule)  # 알아서 test 로더 사용
    """
    
    def __init__(self, train_dir, val_dir, test_dir, crop_size, train_bs=8, 
                 name="octa500_3m", num_samples_per_image=1):
        """
        Args:
            train_dir: 훈련 데이터 경로
            val_dir: 검증 데이터 경로
            test_dir: 테스트 데이터 경로
            crop_size: 크롭 크기
            train_bs: 훈련 배치 크기
            name: 데이터셋 이름
            num_samples_per_image: 한 이미지당 생성할 샘플 개수 (기본: 1)
        """
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.crop_size = crop_size
        self.train_bs = train_bs
        self.name = name
        self.num_samples_per_image = num_samples_per_image

        self.save_hyperparameters()

        # 데이터셋들을 None으로 초기화 (setup에서 실제 생성)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        """
        데이터셋을 설정하는 메서드입니다.
        Lightning Trainer가 자동으로 호출합니다.
        
        Args:
            stage: 'fit', 'test', 'predict' 등의 단계 (보통 None)
        """
        # 각 단계별로 데이터셋 생성
        # 훈련 데이터는 augmentation=True, 나머지는 False
        self.train_dataset = OCTADataset(
            self.train_dir, 
            augmentation=True, 
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )
        self.val_dataset = OCTADataset(
            self.val_dir, 
            augmentation=False, 
            crop_size=self.crop_size,
            num_samples_per_image=1  # val/test는 항상 1개
        )
        self.test_dataset = OCTADataset(
            self.test_dir, 
            augmentation=False, 
            crop_size=self.crop_size,
            num_samples_per_image=1  # val/test는 항상 1개
        )

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool = False):
        """
        공통 DataLoader 설정으로 DataLoader를 생성합니다.
        중복 코드를 줄이기 위한 헬퍼 메서드입니다.
        
        DataLoader 주요 파라미터 설명:
        - batch_size: 한 번에 처리할 샘플 수
        - shuffle: 데이터 순서를 섞을지 여부 (훈련시에만 True)
        - num_workers: 데이터 로딩용 프로세스 수 (병렬 처리)
        - pin_memory: GPU 전송 속도 향상 (CUDA 사용시 True 권장)
        - prefetch_factor: 미리 로드할 배치 수 (메모리vs속도 트레이드오프)
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=False,
            prefetch_factor=2
        )

    def train_dataloader(self):
        """훈련용 DataLoader 반환 (데이터 순서 섞임)"""
        return self._create_dataloader(self.train_dataset, self.train_bs, shuffle=True)
    
    def val_dataloader(self):
        """검증용 DataLoader 반환 (데이터 순서 고정)"""
        return self._create_dataloader(self.val_dataset, 1, shuffle=False)

    def test_dataloader(self):
        """테스트용 DataLoader 반환 (데이터 순서 고정)"""
        return self._create_dataloader(self.test_dataset, 1, shuffle=False)

@DATASET_REGISTRY.register(name='octa500_3m')
class OCTA500_3M_DataModule(OCTADataModule):
    def __init__(self, train_dir="data/OCTA500_3M/train", val_dir="data/OCTA500_3M/val", 
                 test_dir="data/OCTA500_3M/test", crop_size=128, train_bs=8, num_samples_per_image=1):
        super().__init__(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, 
                        crop_size=crop_size, train_bs=train_bs, num_samples_per_image=num_samples_per_image)

@DATASET_REGISTRY.register(name='octa500_6m')
class OCTA500_6M_DataModule(OCTADataModule):
    def __init__(self, train_dir="data/OCTA500_6M/train", val_dir="data/OCTA500_6M/val", 
                 test_dir="data/OCTA500_6M/test", crop_size=128, train_bs=8, num_samples_per_image=1):
        super().__init__(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, 
                        crop_size=crop_size, train_bs=train_bs, num_samples_per_image=num_samples_per_image)
    

if __name__ == "__main__":
    datamodule = DATASET_REGISTRY.get('octa500_3m')()
    datamodule.setup()
    visualize_dataset(datamodule.train_dataloader(), "octa500_3m_train")
    visualize_dataset(datamodule.val_dataloader(), "octa500_3m_val")
    visualize_dataset(datamodule.test_dataloader(), "octa500_3m_test")

    datamodule = DATASET_REGISTRY.get('octa500_6m')()
    datamodule.setup()
    visualize_dataset(datamodule.train_dataloader(), "octa500_6m_train")
    visualize_dataset(datamodule.val_dataloader(), "octa500_6m_val")
    visualize_dataset(datamodule.test_dataloader(), "octa500_6m_test")    