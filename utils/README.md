# Dataset Utilities

This directory contains utility scripts for dataset management.

## Downloading Tiny ImageNet Dataset

The `download_dataset.py` script downloads the Tiny ImageNet dataset, extracts it, and organizes the validation set by class.

### Usage

```bash
# Run from the utils directory
python download_dataset.py

# Specify a custom data directory
python download_dataset.py --data-dir /path/to/data

# Specify a separate extraction directory
python download_dataset.py --data-dir /path/to/data --extract-dir /path/to/extract
```

By default, the script will:
1. Download the dataset to `./data/tiny-imagenet-200.zip`
2. Extract it to `./data/tiny-imagenet-200/`
3. Organize the validation images by class

### Using in Python Code

You can also import and use the function in your Python code:

```python
from utils.download_dataset import download_tiny_imagenet

# Use default paths
dataset_path = download_tiny_imagenet()

# Or specify custom paths
dataset_path = download_tiny_imagenet(data_dir='path/to/data', extract_dir='path/to/extract')
``` 