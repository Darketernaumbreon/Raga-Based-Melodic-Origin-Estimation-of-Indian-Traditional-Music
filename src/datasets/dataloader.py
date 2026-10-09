import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from src.datasets.mel_dataset import MelDataset

def get_dataloaders(batch_size=32, num_workers=4, pin_memory=True, feature_type="mel"):
    """
    Creates and returns DataLoaders for train, validation, and test splits.
    """
    train_dataset = MelDataset("train", feature_type=feature_type)
    val_dataset = MelDataset("validation", feature_type=feature_type)
    test_dataset = MelDataset("test", feature_type=feature_type)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader


def get_hybrid_dataloaders(batch_size=32, num_workers=4, pin_memory=True):
    """
    Creates DataLoaders for the Dual-Branch Hybrid model.
    Each batch returns ((mel_tensor, swara_tensor), label).
    """
    from src.datasets.hybrid_dataset import HybridDataset
    
    train_dataset = HybridDataset("train")
    val_dataset   = HybridDataset("validation")
    test_dataset  = HybridDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader, test_loader

# Quick test if run directly
if __name__ == "__main__":
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)
    
    print("\nFetching first training batch...")
    images, labels = next(iter(train_loader))
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")