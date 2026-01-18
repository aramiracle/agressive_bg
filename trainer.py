"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.trainer import (
    get_cube_decision,
    play_one_game,
    finalize_history,
    train_batch,
    train
)

__all__ = [
    'get_cube_decision',
    'play_one_game',
    'finalize_history',
    'train_batch',
    'train'
]

if __name__ == "__main__":
    train()
