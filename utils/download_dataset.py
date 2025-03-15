#!/usr/bin/env python3
import os
import shutil
import urllib.request
import zipfile
import argparse
from pathlib import Path

def download_tiny_imagenet(data_dir='../data', extract_dir=None):
    """
    Download Tiny ImageNet dataset and organize validation set by class.
    
    Args:
        data_dir (str): Directory to save the dataset
        extract_dir (str, optional): Directory to extract the dataset. If None, uses data_dir.
    """
    # Create data directory if it doesn't exist
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    if extract_dir is None:
        extract_dir = data_dir
    else:
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(exist_ok=True, parents=True)
    
    # URL for the Tiny ImageNet dataset
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = data_dir / "tiny-imagenet-200.zip"
    
    # Download the dataset if it doesn't exist
    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet dataset to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    else:
        print(f"Dataset already exists at {zip_path}")
    
    # Extract the dataset
    extract_path = extract_dir / "tiny-imagenet-200"
    if not extract_path.exists():
        print(f"Extracting dataset to {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")
    else:
        print(f"Dataset already extracted at {extract_path}")
    
    # Organize validation images by class
    val_dir = extract_path / "val"
    val_images_dir = val_dir / "images"
    val_annotations_file = val_dir / "val_annotations.txt"
    
    # Check if validation images are already organized
    if val_images_dir.exists():
        print("Organizing validation images by class...")
        
        # Read validation annotations
        with open(val_annotations_file, 'r') as f:
            for line in f:
                fn, cls, *_ = line.strip().split('\t')
                # Create class directory if it doesn't exist
                class_dir = val_dir / cls
                class_dir.mkdir(exist_ok=True)
                
                # Copy image to class directory
                src_path = val_images_dir / fn
                dst_path = class_dir / fn
                shutil.copyfile(src_path, dst_path)
        
        # Remove the original images directory
        shutil.rmtree(val_images_dir)
        print("Validation images organized!")
    else:
        print("Validation images already organized")
    
    print(f"Tiny ImageNet dataset is ready at {extract_path}")
    return extract_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Tiny ImageNet dataset')
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='Directory to save the dataset')
    parser.add_argument('--extract-dir', type=str, default='./data',
                        help='Directory to extract the dataset')
    args = parser.parse_args()
    
    download_tiny_imagenet(args.data_dir, args.extract_dir)
