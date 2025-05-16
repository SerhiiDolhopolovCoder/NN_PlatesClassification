import torch
from torchvision.transforms import v2


class AugmentationV1:
    size = (224, 224)
    mean_normalize = [0.485, 0.456, 0.406]
    std_normalize = [0.229, 0.224, 0.225]
        
    def get_train(self):
        augment = v2.Compose([
            v2.Resize(self.size),
            v2.TrivialAugmentWide(num_magnitude_bins=31),
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomRotation(degrees=15),
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # яркість
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=self.mean_normalize,
                         std=self.std_normalize)
        ])
        return augment
        
    def get_test(self):
        augment = v2.Compose([
            v2.Resize(self.size),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=self.mean_normalize,
                         std=self.std_normalize)
        ])
        return augment