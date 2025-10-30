import autorootcwd
from src.archs.components import (
    CENet, CSNet, AACAUNet, UNet3Plus, 
    VesselNet, TransUNet, DSCNet
)
import torch
from thop import profile

# 모델 생성
models = {
    'cenet': CENet(in_channels=1, num_classes=2),
    'csnet': CSNet(in_channels=1, num_classes=2),
    'aacaunet': AACAUNet(in_channels=1, num_classes=2),
    'unet3plus': UNet3Plus(in_channels=1, num_classes=2),
    'vesselnet': VesselNet(in_channels=1, num_classes=2),
    'transunet': TransUNet(in_channels=1, img_size=224, num_classes=2),
    'dscnet': DSCNet(in_channels=1, num_classes=2)
}

def main():
    for model_name, model in models.items():
        print(f"{model_name} model created")
        x = torch.randn(1, 1, 224, 224)
        y = model(x)
        if isinstance(y, dict):
            shapes = {}
            for k, v in y.items():
                if isinstance(v, (list, tuple)):
                    shapes[k] = [tuple(t.shape) for t in v]
                else:
                    shapes[k] = tuple(v.shape)
            print(f"{model_name} model output shapes: {shapes}")
        else:
            print(f"{model_name} model output shape: {tuple(y.shape)}")
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f"{model_name} model FLOPS: {flops}")
        print(f"{model_name} model params: {params}")
        
    print('ALL TESTS PASSED')

if __name__ == "__main__":
    main()