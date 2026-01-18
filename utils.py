"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.utils import (
    move_to_indices,
    indices_to_move,
    format_move,
    format_board
)

__all__ = [
    'move_to_indices',
    'indices_to_move',
    'format_move',
    'format_board'
]
