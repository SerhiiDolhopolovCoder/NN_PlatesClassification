import torch
from torchvision.transforms import v2


class AugmentationV1:
    size = (64, 64)
    mean_normalize = [0.485, 0.456, 0.406]
    std_normalize = [0.229, 0.224, 0.225]
        
    def get_train(self):
        compose_list = [
            v2.Resize(self.size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.RandomResizedCrop(self.size, scale=(0.8, 1.0)),
            v2.GaussianBlur(kernel_size=(3, 3)),
            v2.RandomAdjustSharpness(sharpness_factor=2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]        
        augment = v2.Compose(compose_list)
        return augment
        
    def get_test(self):
        compose_list = [
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        augment = v2.Compose(compose_list)
        return augment