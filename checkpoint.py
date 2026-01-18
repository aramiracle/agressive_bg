"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.checkpoint import (
    setup_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
    get_model_state_dict,
    load_model_state_dict
)

__all__ = [
    'setup_checkpoint_dir',
    'save_checkpoint',
    'load_checkpoint',
    'get_model_state_dict',
    'load_model_state_dict'
]
