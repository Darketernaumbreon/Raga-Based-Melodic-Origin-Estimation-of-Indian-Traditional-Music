import torch
from pathlib import Path
import logging

logger = logging.getLogger("RaagAnalysis")

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    metrics: dict, 
    filepath: Path,
    label_mapping: dict = None
):
    """
    Saves a model checkpoint including optimizer state and metadata.
    """
    # Extract state dict (handle DataParallel if used)
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if label_mapping is not None:
        checkpoint['label_mapping'] = label_mapping
        
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logger.debug(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: Path, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None
):
    """
    Loads a model checkpoint.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    logger.info(f"Loading checkpoint from {filepath} onto {device}...")
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    logger.info(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', 'Unknown')}")
    
    return checkpoint
