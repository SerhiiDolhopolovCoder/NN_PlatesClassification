import torch
from torchvision.transforms import v2


class AugmentationV1:
    size = (64, 64)
    mean_normalize = [0.485, 0.456, 0.406]
    std_normalize = [0.229, 0.224, 0.225]
        
    def get_train(self):
        compose_list = [
            v2.Resize(self.size),
            v2.TrivialAugmentWide(num_magnitude_bins=31),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]        
        augment = v2.Compose(compose_list)
        return augment
        
    def get_test(self):
        compose_list = [
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        augment = v2.Compose(compose_list)
        return augment
    
    
class AugmentationV2:
    size = (224, 224)
    mean_normalize = [0.485, 0.456, 0.406]
    std_normalize = [0.229, 0.224, 0.225]
        
    def get_train(self):
        compose_list = [
            v2.Resize(self.size),
            v2.TrivialAugmentWide(num_magnitude_bins=31),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]        
        augment = v2.Compose(compose_list)
        return augment
        
    def get_test(self):
        compose_list = [
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        augment = v2.Compose(compose_list)
        return augment