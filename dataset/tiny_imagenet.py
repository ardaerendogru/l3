import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List


class TinyImageNetDataset(Dataset):
    """
    Dataset class for Tiny ImageNet.
    
    This class provides a template that students can extend and customize.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the Tiny ImageNet dataset.
        
        Args:
            root_dir (str): Path to the Tiny ImageNet dataset root directory
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Transform to be applied to the images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        
        # Load class mapping
        self.class_to_idx = self._load_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.samples = self._load_split()
    
    def _load_class_mapping(self) -> Dict[str, int]:
        """
        Load class names and their corresponding indices.
        
        Returns:
            Dict mapping class names to indices
        """
        class_to_idx = {}
        classes_file = self.root_dir / "wnids.txt"
        
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f):
                class_id = line.strip()
                class_to_idx[class_id] = idx
        
        return class_to_idx
    
    def _load_split(self) -> List[Tuple[str, int]]:
        """
        Load image paths and labels for the specified split.
        
        Returns:
            List of tuples (image_path, class_idx)
        """
        samples = []
        
        if self.split == 'train':
            # Each class has its own directory in train
            for class_id, class_idx in self.class_to_idx.items():
                class_dir = self.root_dir / 'train' / class_id / 'images'
                if not class_dir.exists():
                    continue
                
                for img_file in class_dir.glob('*.JPEG'):
                    samples.append((str(img_file), class_idx))
        
        elif self.split == 'val':
            # Validation set is organized by class after preprocessing
            val_dir = self.root_dir / 'val'
            for class_id, class_idx in self.class_to_idx.items():
                class_dir = val_dir / class_id
                if not class_dir.exists():
                    continue
                
                for img_file in class_dir.glob('*.JPEG'):
                    samples.append((str(img_file), class_idx))
        
        
        return samples
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where label is the class index
        """
        img_path, label = self.samples[idx]
        

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
