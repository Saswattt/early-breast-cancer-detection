import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms

class MammogramDataset(Dataset):
    """Mammogram dataset for PyTorch."""
    
    def __init__(self, csv_file, transform=None, num_classes=3):
        """
        Args:
            csv_file: Path to the CSV file with image paths and labels
            transform: Optional transform to be applied on a sample
            num_classes: Number of classes (3 for normal/benign/malignant)
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.num_classes = num_classes
        
        # Validate image paths
        invalid_paths = [
            path for path in self.data_frame.iloc[:, 0] 
            if not os.path.exists(path)
        ]
        if invalid_paths:
            raise ValueError(
                f"Found {len(invalid_paths)} invalid image paths. First few: {invalid_paths[:3]}"
            )
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_frame.iloc[idx, 0]
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Convert to RGB (3 channels) by duplicating the grayscale channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            label = self.data_frame.iloc[idx, 1]
            
            if self.transform:
                image = self.transform(image)
            
            # One-hot encode the label if needed
            if self.num_classes > 2:
                label_onehot = torch.zeros(self.num_classes)
                label_onehot[label] = 1
                return image, label_onehot
            
            return image, label
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")

def get_data_loaders(data_dir, batch_size=32, num_workers=4, img_size=224):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing processed data and CSV files
        batch_size: Batch size for data loaders
        num_workers: Number of worker threads for data loading
        img_size: Size to resize images to (default: 224)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # ImageNet normalization statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Define transformations with consistent image size
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Create datasets with error handling
    try:
        train_dataset = MammogramDataset(
            csv_file=os.path.join(data_dir, 'train.csv'),
            transform=train_transform
        )
        
        val_dataset = MammogramDataset(
            csv_file=os.path.join(data_dir, 'val.csv'),
            transform=val_test_transform
        )
        
        test_dataset = MammogramDataset(
            csv_file=os.path.join(data_dir, 'test.csv'),
            transform=val_test_transform
        )
    except Exception as e:
        raise RuntimeError(f"Error creating datasets: {str(e)}")
    
    # Create data loaders with error handling
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch for consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        raise RuntimeError(f"Error creating data loaders: {str(e)}")