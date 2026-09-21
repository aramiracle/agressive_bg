"""Checkpoint management for training."""

import os
import torch
from src.config import Config
import importlib.util
from src.model import BackgammonTransformer, BackgammonCNN, LegacyValueTransformer

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


def baseline_artifact_paths():
    """Return (model_path, config_path, equity_path) for the frozen baseline."""
    directory = Config.BASELINE_DIR
    return (
        os.path.join(directory, Config.BASELINE_MODEL_NAME),
        os.path.join(directory, "config.py"),
        os.path.join(directory, "match_equity.pt"),
    )


def default_model_paths(repo_root="."):
    """Preferred play/eval order: newest training stage, then frozen baseline."""
    root = os.path.join(repo_root, "checkpoints")
    paths = []
    for folder in ("stage2", "stage1", "baseline"):
        for name in ("best_model.pt", "latest_model.pt"):
            paths.append(os.path.join(root, folder, name))
    return paths


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


def warm_start(model, path, device='cpu'):
    """
    Load only the model weights from `path` (e.g. the stage-1 best model when
    starting stage 2). Returns True if weights were loaded.
    """
    if not path or not os.path.exists(path):
        return False
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    load_model_state_dict(model, state)
    return True


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

def _config_class_from_path(config_path):
    spec = importlib.util.spec_from_file_location("baseline_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


def _instantiate_saved_model(saved_config):
    head_kind = getattr(saved_config, "HEAD_KIND", "outcome")
    if saved_config.MODEL_TYPE == "transformer":
        if head_kind == "value_policy":
            return LegacyValueTransformer(config=saved_config)
        return BackgammonTransformer(config=saved_config)
    if saved_config.MODEL_TYPE == "cnn":
        return BackgammonCNN(config=saved_config)
    raise ValueError(f"Unknown MODEL_TYPE in baseline: {saved_config.MODEL_TYPE}")


def build_model_from_config_path(config_path, device):
    """Instantiate a model using the architecture defined in a saved config file."""
    saved_config = _config_class_from_path(config_path)
    return _instantiate_saved_model(saved_config).to(device)


def load_model_with_config(config_path, model_path, device):
    """
    Load a frozen baseline using its own config file so an older architecture
    still works after the training Config has changed.
    """
    model = build_model_from_config_path(config_path, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        elo = checkpoint.get('elo', Config.INITIAL_ELO)
    else:
        state_dict = checkpoint
        elo = Config.INITIAL_ELO

    load_model_state_dict(model, state_dict)
    model.eval()
    return model, elo