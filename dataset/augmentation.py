from torchvision import transforms
from torchvision.transforms import RandAugment



class SimpleDataAugmentation(object):
    def __init__(
        self,
        image_size=224,
    ):
        # Color and intensity transformations
        self.rand_augmentations = RandAugment(
            num_ops=2, 
            magnitude=5,  
        )
        
        # Resizing and Normalization
        self.normalize = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, image):
        # Apply transformations sequentially
        image = self.rand_augmentations(image)
        image = self.normalize(image)
        
        return image


def get_train_transforms(image_size=224):
    """Returns simple transforms for training"""
    return SimpleDataAugmentation(image_size=image_size)


def get_val_transforms(image_size=224):
    """Returns transforms for validation (no augmentation)"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])