"""Checkpoint management for saving and loading models."""

import torch
from pathlib import Path
from config import Config


def save_checkpoint(model, optimizer, step, elo, loss, filepath):
    """
    Save model checkpoint with training state.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        step: Current training step
        elo: Current ELO rating
        loss: Current loss value
        filepath: Path to save the checkpoint
    """
    checkpoint = {
        'step': step,
        'elo': elo,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'D_MODEL': Config.D_MODEL,
            'N_HEAD': Config.N_HEAD,
            'N_LAYERS': Config.N_LAYERS,
            'NUM_ACTIONS': Config.NUM_ACTIONS,
        }
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: The neural network model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (step, elo, loss)
    """
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['elo'], checkpoint.get('loss', 0)


def setup_checkpoint_dir():
    """
    Create checkpoint directory if it doesn't exist.
    
    Returns:
        Tuple of (checkpoint_dir, best_model_path, latest_model_path)
    """
    checkpoint_dir = Path(Config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
    latest_model_path = checkpoint_dir / "latest_model.pt"
    return checkpoint_dir, best_model_path, latest_model_path

