"""Checkpoint management for training."""

import os
import torch
from src.backgammon.config import Config


def setup_checkpoint_dir():
    """
    Create checkpoint directory and return paths.
    
    Returns:
        Tuple of (checkpoint_dir, best_model_path, latest_model_path)
    """
    checkpoint_dir = Config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
    
    return checkpoint_dir, best_path, latest_path


def save_checkpoint(model, optimizer, step, elo, loss, path):
    """
    Save a training checkpoint.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        step: Current training step
        elo: Current ELO rating
        loss: Recent average loss
        path: File path to save to
    """
    # Handle compiled models
    model_to_save = model
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod
    
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'elo': elo,
        'loss': loss,
        'config': {
            'model_type': Config.MODEL_TYPE,
            'd_model': Config.D_MODEL,
            'n_layers': Config.N_LAYERS,
        }
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """
    Load a training checkpoint.
    
    Args:
        path: File path to load from
        model: The neural network model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
    
    Returns:
        Dictionary with 'step', 'elo', 'loss' keys, or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None
    
    checkpoint = torch.load(path, map_location=device)
    
    # Handle compiled models
    model_to_load = model
    if hasattr(model, '_orig_mod'):
        model_to_load = model._orig_mod
    
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'step': checkpoint.get('step', 0),
        'elo': checkpoint.get('elo', Config.INITIAL_ELO),
        'loss': checkpoint.get('loss', 0.0)
    }


def get_model_state_dict(model):
    """Get state dict, handling compiled models."""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod.state_dict()
    return model.state_dict()


def load_model_state_dict(model, state_dict):
    """Load state dict, handling compiled models."""
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

