import autorootcwd
import src.data
from src.utils.visualize_dataloader import visualize_dataset
from src.utils.registry import DATASET_REGISTRY

def main():
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
    print('ALL TESTS PASSED')

if __name__ == "__main__":
    main()