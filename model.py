"""Compatibility shim - imports from new location."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from backgammon.model import (
    LearnedPositionalEncoding,
    BackgammonTransformer,
    ResidualBlock1D,
    BackgammonCNN,
    get_model
)

__all__ = [
    'LearnedPositionalEncoding',
    'BackgammonTransformer', 
    'ResidualBlock1D',
    'BackgammonCNN',
    'get_model'
]
